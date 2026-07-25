"""SHAP attribution: which features the model actually used.

PRD Module 11 lists SHAP among the required outputs, and the reason is diagnostic
rather than decorative. A tree ensemble on a 70-feature panel will report *some*
ranking of feature importances whether or not it has learned anything; what is
informative is the shape of that ranking. Attribution concentrated in one or two
features that are themselves near-duplicates of the naive momentum baseline says
the model has rediscovered the baseline at greater cost. Attribution spread
evenly across seventy features says it has found nothing and is fitting noise.

Two explainers, chosen by model family rather than by guesswork:

* **Tree models** (random forest, XGBoost, LightGBM) use ``shap.TreeExplainer``:
  exact, fast, and the reason SHAP is worth using at all on ensembles.
* **Linear models** (ridge, lasso) use ``shap.LinearExplainer`` on the *scaled*
  features, because the estimator is a pipeline whose first step is a
  ``StandardScaler``. Explaining the unscaled inputs against a model fitted on
  scaled ones would attribute the scaler's variance to the features.

Attribution is computed on a **subsample** by default. SHAP on half a million
panel rows is minutes of compute for a ranking that stabilises after a few
thousand; the sample size is reported so the reader knows what it was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import structlog

from quantlab.ml_models.models import Estimator, MissingDependencyError, ModelSpec

log = structlog.get_logger(__name__)

__all__ = ["ShapAttribution", "shap_attribution"]


def _require_shap() -> Any:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise MissingDependencyError(
            "SHAP attribution needs the 'shap' package, which is not installed. "
            "Install the optional group: pip install -e '.[research]'"
        ) from exc
    return shap


@dataclass(frozen=True, slots=True)
class ShapAttribution:
    """Mean absolute SHAP value per feature, and how it was computed."""

    feature_names: tuple[str, ...]
    mean_abs_shap: np.ndarray
    explainer: str
    n_samples: int

    @property
    def ranked(self) -> list[tuple[str, float]]:
        """Features by attribution, strongest first."""

        order = np.argsort(-self.mean_abs_shap)
        return [(self.feature_names[i], float(self.mean_abs_shap[i])) for i in order]

    @property
    def concentration(self) -> float:
        """Share of total attribution carried by the top five features.

        Near 1.0 means the model is effectively a five-feature model — worth
        knowing before claiming that seventy engineered factors were needed.
        """

        total = float(self.mean_abs_shap.sum())
        if total <= 0.0:
            return float("nan")
        top = sorted(self.mean_abs_shap, reverse=True)[:5]
        return float(sum(top) / total)

    def as_dict(self) -> dict[str, Any]:
        return {
            "explainer": self.explainer,
            "n_samples": self.n_samples,
            "top_5_concentration": self.concentration,
            "mean_abs_shap": dict(self.ranked),
        }

    def render(self, top: int = 15) -> str:
        lines = [
            f"SHAP attribution ({self.explainer}, {self.n_samples:,} samples)",
            f"  top-5 concentration: {self.concentration:.1%}",
        ]
        total = float(self.mean_abs_shap.sum()) or 1.0
        for name, value in self.ranked[:top]:
            bar = "#" * round(40 * value / (self.ranked[0][1] or 1.0))
            lines.append(f"  {name:28s} {value:10.6f}  {value / total:6.1%}  {bar}")
        remaining = len(self.feature_names) - top
        if remaining > 0:
            lines.append(f"  … and {remaining} further features")
        return "\n".join(lines)


def _unwrap_pipeline(estimator: Estimator) -> tuple[Any, Any]:
    """``(final_estimator, transform)`` for a pipeline, else ``(estimator, None)``."""

    steps = getattr(estimator, "named_steps", None)
    if steps is None:
        return estimator, None
    scaler = steps.get("scale")
    model = steps.get("model")
    if model is None:  # pragma: no cover - defensive
        return estimator, None
    return model, scaler


def _class_one(values: Any) -> np.ndarray:
    """Reduce SHAP output to a 2-D ``(n_samples, n_features)`` array.

    Classifiers return per-class attributions — a list of two arrays on older
    SHAP releases, a 3-D array on newer ones. Both describe the same thing, and
    the class of interest is "up".
    """

    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    return array


def shap_attribution(
    estimator: Estimator,
    frame: pd.DataFrame,
    *,
    spec: ModelSpec,
    max_samples: int = 2000,
    seed: int = 0,
) -> ShapAttribution:
    """Mean absolute SHAP value per feature for a **fitted** estimator.

    ``frame`` should be the data the attribution is meant to describe — normally the
    training features, since attribution answers "what did this model learn",
    not "how did it perform".
    """

    shap = _require_shap()
    if len(frame) == 0:
        raise ValueError("cannot attribute a model over an empty feature frame")

    rng = np.random.default_rng(seed)
    if len(frame) > max_samples:
        rows = np.sort(rng.choice(len(frame), size=max_samples, replace=False))
        sample = frame.iloc[rows]
    else:
        sample = frame

    model, transform = _unwrap_pipeline(estimator)

    if spec.is_tree:
        explainer_name = "TreeExplainer"
        explainer = shap.TreeExplainer(model)
        values = _class_one(explainer.shap_values(sample))
    else:
        explainer_name = "LinearExplainer"
        scaled = (
            pd.DataFrame(
                transform.transform(sample),
                columns=sample.columns,
                index=sample.index,
            )
            if transform is not None
            else sample
        )
        explainer = shap.LinearExplainer(model, scaled)
        values = _class_one(explainer.shap_values(scaled))

    mean_abs = np.abs(np.asarray(values, dtype=float)).mean(axis=0)
    attribution = ShapAttribution(
        feature_names=tuple(str(c) for c in frame.columns),
        mean_abs_shap=np.asarray(mean_abs, dtype=float).ravel(),
        explainer=explainer_name,
        n_samples=len(sample),
    )
    log.info(
        "ml.shap.computed",
        family=spec.family,
        explainer=explainer_name,
        samples=attribution.n_samples,
        top_feature=attribution.ranked[0][0] if attribution.ranked else None,
        concentration=attribution.concentration,
    )
    return attribution
