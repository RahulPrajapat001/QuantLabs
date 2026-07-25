"""The model registry: the five families PRD Module 11 permits, and no others.

Section 2.2 of the PRD and §4 of ``CLAUDE.md`` cut deep sequence models — LSTM,
GRU, Transformer — deliberately, not for lack of time. Daily equity panels give a
few thousand cross-sections of a few hundred names; a model with millions of
parameters fitted to that does not learn a market, it memorises a sample. The
restriction is enforced structurally rather than editorially: there is no path to
an estimator except :func:`build_model`, and it resolves names against
:data:`MODEL_FAMILIES`. A name that is not in the registry cannot be built.

Two design points worth stating, because both affect whether the reported numbers
mean anything:

* **Scaling lives inside the estimator.** Ridge and lasso are wrapped in a
  :class:`~sklearn.pipeline.Pipeline` with a :class:`StandardScaler`, so the
  scaler is fitted on the *training fold only* and applied to the test fold.
  Scaling the whole panel once before splitting is a leak — the test fold's mean
  and variance would inform the training data — and it is the most common way a
  purged CV is quietly undone after all the trouble of building one.
* **"Ridge" and "lasso" for classification** mean L2- and L1-regularised logistic
  regression. That is the direct analogue (same penalty, same geometry, different
  link function), and it is named here rather than silently substituted.

Every estimator is constructed with a fixed ``random_state`` so a configuration
that gets logged as a trial can be reproduced from its logged parameters alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

__all__ = [
    "DEFAULT_SEARCH_SPACE",
    "MODEL_FAMILIES",
    "Estimator",
    "MissingDependencyError",
    "ModelSpec",
    "Task",
    "UnknownModelError",
    "build_model",
    "supports_probability",
]


class Task(StrEnum):
    """The two prediction targets of PRD Module 11."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class UnknownModelError(ValueError):
    """Raised when a model family outside the permitted set is requested."""


class MissingDependencyError(ImportError):
    """Raised when a gradient-boosting library is requested but not installed."""


@runtime_checkable
class Estimator(Protocol):
    """The scikit-learn estimator surface this module relies on."""

    def fit(self, features: Any, target: Any) -> Any: ...

    def predict(self, features: Any) -> np.ndarray: ...


def supports_probability(estimator: Estimator) -> bool:
    """Whether ``estimator`` can produce class probabilities (for ROC-AUC)."""

    return callable(getattr(estimator, "predict_proba", None))


# --- Builders ----------------------------------------------------------------
#
# Every import is deferred into the builder. Importing this module must stay
# cheap: the CLI imports it to list model names, and pulling in scikit-learn,
# XGBoost and LightGBM to print five strings would make every command slow.


def _linear_pipeline(estimator: Any) -> Estimator:
    """Wrap a scale-sensitive estimator so the scaler is fitted per fold."""

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return cast(
        "Estimator",
        Pipeline([("scale", StandardScaler()), ("model", estimator)]),
    )


def _logistic(*, l1_ratio: float, c: float, max_iter: int, seed: int) -> Any:
    """A logistic regression with the requested penalty, across scikit-learn APIs.

    scikit-learn 1.8 deprecated ``penalty=`` in favour of ``l1_ratio=`` and the
    two disagree loudly when both are supplied — ``penalty="l1"`` alongside the
    new default ``l1_ratio=0.0`` warns about "inconsistent values" and leaves it
    genuinely ambiguous which penalty was applied. A "lasso" classifier that
    silently fitted an L2 penalty would invalidate every comparison in the search,
    so the API in use is detected rather than assumed: the deprecated parameter
    announces itself with the sentinel default ``"deprecated"``.

    ``saga`` is the solver for anything with an L1 component; ``lbfgs`` is faster
    and is used for pure L2.
    """

    import inspect

    from sklearn.linear_model import LogisticRegression

    solver = "saga" if l1_ratio > 0.0 else "lbfgs"
    common: dict[str, Any] = {
        "C": c,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": seed,
    }
    default = (
        inspect.signature(LogisticRegression.__init__).parameters["penalty"].default
    )
    if default == "deprecated":
        return LogisticRegression(l1_ratio=l1_ratio, **common)
    return LogisticRegression(penalty="l1" if l1_ratio > 0.0 else "l2", **common)


def _build_ridge(task: Task, params: Mapping[str, Any], seed: int) -> Estimator:
    alpha = float(params.get("alpha", 1.0))
    if task is Task.REGRESSION:
        from sklearn.linear_model import Ridge

        return _linear_pipeline(Ridge(alpha=alpha, random_state=seed))

    # C is the inverse of the regularisation strength, so alpha maps to 1/alpha.
    return _linear_pipeline(
        _logistic(
            l1_ratio=0.0,
            c=1.0 / alpha,
            max_iter=int(params.get("max_iter", 2000)),
            seed=seed,
        )
    )


def _build_lasso(task: Task, params: Mapping[str, Any], seed: int) -> Estimator:
    alpha = float(params.get("alpha", 1e-3))
    if task is Task.REGRESSION:
        from sklearn.linear_model import Lasso

        return _linear_pipeline(
            Lasso(
                alpha=alpha,
                random_state=seed,
                max_iter=int(params.get("max_iter", 10_000)),
            )
        )

    return _linear_pipeline(
        _logistic(
            l1_ratio=1.0,
            c=1.0 / alpha,
            max_iter=int(params.get("max_iter", 5000)),
            seed=seed,
        )
    )


def _build_random_forest(task: Task, params: Mapping[str, Any], seed: int) -> Estimator:
    kwargs: dict[str, Any] = {
        "n_estimators": int(params.get("n_estimators", 300)),
        "max_depth": params.get("max_depth", 5),
        "min_samples_leaf": int(params.get("min_samples_leaf", 50)),
        "max_features": params.get("max_features", "sqrt"),
        "n_jobs": int(params.get("n_jobs", -1)),
        "random_state": seed,
    }
    if task is Task.REGRESSION:
        from sklearn.ensemble import RandomForestRegressor

        return cast("Estimator", RandomForestRegressor(**kwargs))

    from sklearn.ensemble import RandomForestClassifier

    return cast("Estimator", RandomForestClassifier(**kwargs))


def _require(module: str, extra: str = "research") -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise MissingDependencyError(
            f"the {module!r} model family needs the {module} package, which is not "
            f"installed. Install the optional group: pip install -e '.[{extra}]'"
        ) from exc


def _build_xgboost(task: Task, params: Mapping[str, Any], seed: int) -> Estimator:
    xgb = _require("xgboost")
    kwargs: dict[str, Any] = {
        "n_estimators": int(params.get("n_estimators", 300)),
        "max_depth": int(params.get("max_depth", 3)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "min_child_weight": float(params.get("min_child_weight", 10.0)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "tree_method": params.get("tree_method", "hist"),
        "n_jobs": int(params.get("n_jobs", -1)),
        "random_state": seed,
        "verbosity": 0,
    }
    if task is Task.REGRESSION:
        return cast("Estimator", xgb.XGBRegressor(**kwargs))
    return cast("Estimator", xgb.XGBClassifier(**kwargs, eval_metric="logloss"))


def _build_lightgbm(task: Task, params: Mapping[str, Any], seed: int) -> Estimator:
    lgb = _require("lightgbm")
    kwargs: dict[str, Any] = {
        "n_estimators": int(params.get("n_estimators", 300)),
        "num_leaves": int(params.get("num_leaves", 15)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "subsample": float(params.get("subsample", 0.8)),
        "subsample_freq": int(params.get("subsample_freq", 1)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "min_child_samples": int(params.get("min_child_samples", 50)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "n_jobs": int(params.get("n_jobs", -1)),
        "random_state": seed,
        "verbose": -1,
    }
    if task is Task.REGRESSION:
        return cast("Estimator", lgb.LGBMRegressor(**kwargs))
    return cast("Estimator", lgb.LGBMClassifier(**kwargs))


#: The permitted model families. Extending this dict is a scope decision, not an
#: implementation detail — see ``CLAUDE.md`` §4.
MODEL_FAMILIES: Mapping[str, Any] = {
    "ridge": _build_ridge,
    "lasso": _build_lasso,
    "random_forest": _build_random_forest,
    "xgboost": _build_xgboost,
    "lightgbm": _build_lightgbm,
}

#: Families whose feature attribution goes through SHAP's tree explainer.
TREE_FAMILIES = frozenset({"random_forest", "xgboost", "lightgbm"})

#: The declared hyper-parameter grid per family. PRD Module 11 requires the
#: search space to be *reported alongside the results*, so it is data here rather
#: than a literal buried in a script: :mod:`quantlab.ml_models.search` reads it,
#: counts the configurations, and prints both with the numbers they produced.
DEFAULT_SEARCH_SPACE: Mapping[str, Mapping[str, Sequence[Any]]] = {
    "ridge": {"alpha": (0.1, 1.0, 10.0, 100.0)},
    "lasso": {"alpha": (1e-5, 1e-4, 1e-3, 1e-2)},
    "random_forest": {
        "max_depth": (3, 5, 8),
        "min_samples_leaf": (20, 50, 200),
    },
    "xgboost": {
        "max_depth": (2, 3, 4),
        "learning_rate": (0.03, 0.1),
        "n_estimators": (200, 500),
    },
    "lightgbm": {
        "num_leaves": (7, 15, 31),
        "learning_rate": (0.03, 0.1),
        "n_estimators": (200, 500),
    },
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model configuration: a family, a task, and its hyper-parameters.

    This is the unit that becomes a row in ``research_log``. It is frozen and
    fully describes the estimator, so :meth:`parameters` — which is what gets
    logged — is sufficient to rebuild it.
    """

    family: str
    task: Task
    params: Mapping[str, Any] = field(default_factory=dict)
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.family not in MODEL_FAMILIES:
            raise UnknownModelError(
                f"unknown model family {self.family!r}. PRD Module 11 permits "
                f"exactly {sorted(MODEL_FAMILIES)} — sequence models (LSTM, GRU, "
                "Transformer) are a deliberate scope cut (CLAUDE.md §4), not an "
                "omission."
            )

    @property
    def name(self) -> str:
        """The ``research_log`` strategy name for this configuration."""

        return f"ml:{self.family}:{self.task.value}"

    @property
    def is_tree(self) -> bool:
        return self.family in TREE_FAMILIES

    def build(self) -> Estimator:
        """Construct a fresh, unfitted estimator.

        Fresh every time by design: reusing one estimator across CV folds would
        carry the previous fold's fitted state into the next one on any model
        that supports warm starts.
        """

        builder = MODEL_FAMILIES[self.family]
        return cast("Estimator", builder(self.task, self.params, self.random_state))

    def with_params(self, **overrides: Any) -> ModelSpec:
        """A copy with ``overrides`` merged over the current parameters."""

        return replace(self, params={**dict(self.params), **overrides})

    def parameters(self) -> dict[str, Any]:
        """The logged parameter mapping — everything needed to rebuild this."""

        return {
            "family": self.family,
            "task": self.task.value,
            "random_state": self.random_state,
            **{str(k): v for k, v in self.params.items()},
        }

    def describe(self) -> str:
        if not self.params:
            return f"{self.family} ({self.task.value}, defaults)"
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.family} ({self.task.value}, {rendered})"


def build_model(
    family: str,
    task: Task | str,
    *,
    random_state: int = 0,
    **params: Any,
) -> Estimator:
    """Build an estimator from the permitted registry.

    The only construction path in the codebase. Anything not in
    :data:`MODEL_FAMILIES` raises :class:`UnknownModelError`.
    """

    spec = ModelSpec(
        family=family, task=Task(task), params=params, random_state=random_state
    )
    return spec.build()
