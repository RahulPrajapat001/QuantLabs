"""Baseline honesty: every model reported against rules that require no fitting.

PRD Module 11 makes this a **required** deliverable, not a courtesy: *"Every model
reported against buy-and-hold and a naive momentum rule. Where a model fails to
beat the baselines, the report states so in plain language."*

The reason is that a gradient-boosted model with a rank IC of 0.004 and a net
Sharpe of 0.3 sounds like a finding until you notice that holding the index
equally weighted over the same period returned more, at lower turnover, with no
model at all. Most published retail ML-for-equities results never make that
comparison, which is precisely why they do not replicate.

Two baselines, both evaluated on **the same out-of-sample folds, through the same
metric functions and the same trading rule** as the model:

* **Buy-and-hold** — hold the point-in-time universe equally weighted. As a
  predictor it makes the constant prediction that carries no cross-sectional
  information (the training-set mean return, or the training base rate for the
  direction target), so its directional accuracy is the honest floor: the rate at
  which the market went up anyway.
* **Naive momentum** — rank on trailing momentum, hold the top slice. The signal
  is read straight out of the feature panel; nothing is fitted. For the
  regression target the raw feature is rescaled by the ratio of training standard
  deviations so that MAE and RMSE are in the same units as the target — a pure
  positive scaling, computed on training data only, which cannot change the
  rule's ranking, its sign, or its directional accuracy.

:meth:`BaselineComparison.verdict` renders the conclusion in plain language,
including the case where the model loses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from quantlab.ml_models.dataset import MLDataset, TargetKind
from quantlab.ml_models.metrics import (
    PredictionMetrics,
    RegressionMetrics,
    Sample,
    compute_classification_metrics,
    compute_regression_metrics,
)
from quantlab.ml_models.models import Task
from quantlab.ml_models.trading import TradedResult, TradingRule, trade_predictions

__all__ = [
    "Baseline",
    "BaselineComparison",
    "BaselineOutcome",
    "BuyAndHoldBaseline",
    "NaiveMomentumBaseline",
    "UncalibratedBaselineError",
    "baseline_outcome",
    "compare_to_baselines",
    "default_baselines",
    "evaluate_baseline",
    "prediction_frame",
    "score_predictions",
]

#: Feature columns searched, in order, for the naive momentum rule.
MOMENTUM_FEATURE_CANDIDATES: tuple[str, ...] = (
    "momentum_12_1",
    "momentum_20d",
    "rolling_return_60d",
    "return_20d",
)


@runtime_checkable
class Baseline(Protocol):
    """A reference rule the model has to beat. Nothing here is *fitted*.

    Calibration and scoring are separate calls because the two contexts that need
    baselines index different frames. Within cross-validation, both come from one
    dataset (calibrate on the training fold, score the test fold). For the
    validation-split evaluation the calibration comes from the train dataset and
    the scores from a different one. A single ``predict(dataset, train, test)``
    cannot express the second, and quietly calibrating on the validation data is
    exactly the leak this module exists to prevent.
    """

    @property
    def name(self) -> str:
        """Stable identifier, used as the label in the comparison table."""
        ...

    def calibrate(self, dataset: MLDataset, idx: np.ndarray | None = None) -> Baseline:
        """Return a copy calibrated on ``dataset``'s rows at ``idx`` (all if None)."""
        ...

    def score(self, dataset: MLDataset, idx: np.ndarray) -> np.ndarray:
        """Scores for ``dataset``'s rows at ``idx``, using the calibration."""
        ...

    @property
    def decision_threshold(self) -> float:
        """Score above which the rule calls "up", for the classification target."""
        ...

    @property
    def holds_whole_universe(self) -> bool:
        """Whether the traded form ignores the scores and holds everything."""
        ...


class UncalibratedBaselineError(RuntimeError):
    """Raised when a baseline is scored before it has been calibrated."""


def _rows(idx: np.ndarray | None, n: int) -> np.ndarray:
    return np.arange(n) if idx is None else np.asarray(idx)


@dataclass(frozen=True, slots=True)
class BuyAndHoldBaseline:
    """Hold the point-in-time universe, equally weighted. No cross-sectional view.

    As a *predictor* this is the constant that a model must beat before any of
    its cross-sectional discrimination can be called real: the training-set mean
    return for the regression target, the training base rate for direction. Its
    directional accuracy is therefore the honest floor — the rate at which the
    market went up anyway.
    """

    name: str = "buy_and_hold"
    constant: float | None = None

    def calibrate(
        self, dataset: MLDataset, idx: np.ndarray | None = None
    ) -> BuyAndHoldBaseline:
        rows = _rows(idx, len(dataset))
        train_y = dataset.y.to_numpy()[rows]
        return BuyAndHoldBaseline(
            name=self.name,
            constant=float(np.mean(train_y)) if train_y.size else 0.0,
        )

    def score(self, dataset: MLDataset, idx: np.ndarray) -> np.ndarray:
        if self.constant is None:
            raise UncalibratedBaselineError(
                "buy-and-hold must be calibrated on training data before scoring; "
                "its constant is the training mean, and taking it from the test "
                "set would be a leak"
            )
        return np.full(np.asarray(idx).size, self.constant, dtype=float)

    @property
    def decision_threshold(self) -> float:
        return 0.5

    @property
    def holds_whole_universe(self) -> bool:
        return True

    def describe(self) -> str:
        return "buy-and-hold (equal-weight the point-in-time universe)"


@dataclass(frozen=True, slots=True)
class NaiveMomentumBaseline:
    """Rank on a trailing momentum feature. Not fitted, not tuned.

    The rule a practitioner would try before reaching for a model, and the one a
    model has to justify its complexity against.
    """

    feature: str
    name: str = "naive_momentum"
    #: Units conversion only — see :meth:`calibrate`.
    scale: float = 1.0

    def _column(self, dataset: MLDataset) -> np.ndarray:
        if self.feature not in dataset.X.columns:
            raise KeyError(
                f"the naive momentum baseline needs feature {self.feature!r}, which "
                f"is not in the dataset. Available: {sorted(dataset.feature_names)}. "
                "PRD Module 11 requires this baseline — build the dataset with a "
                "momentum feature rather than dropping the comparison."
            )
        return dataset.X[self.feature].to_numpy(dtype=float)

    def calibrate(
        self, dataset: MLDataset, idx: np.ndarray | None = None
    ) -> NaiveMomentumBaseline:
        """Set the units conversion from training data. Nothing else is learned.

        A trailing 12-month return is on the order of 0.15; a next-day return is
        on the order of 0.005. Comparing their MAEs directly would penalise the
        baseline for its units rather than its accuracy, so the raw figure is
        multiplied by the ratio of training standard deviations. It is a positive
        scaling with no intercept: the rule's ranking, its sign, its rank IC and
        its directional accuracy are all unchanged by it.
        """

        if dataset.target.kind is TargetKind.DIRECTION:
            return self

        rows = _rows(idx, len(dataset))
        train_y = dataset.y.to_numpy()[rows]
        train_feature = self._column(dataset)[rows]
        feature_sd = float(np.std(train_feature))
        target_sd = float(np.std(train_y))
        scale = (
            target_sd / feature_sd
            if feature_sd > 0.0 and np.isfinite(feature_sd)
            else 0.0
        )
        return NaiveMomentumBaseline(feature=self.feature, name=self.name, scale=scale)

    def score(self, dataset: MLDataset, idx: np.ndarray) -> np.ndarray:
        scores: np.ndarray = self._column(dataset)[np.asarray(idx)] * self.scale
        return scores

    @property
    def decision_threshold(self) -> float:
        # "Up if trailing momentum is positive" — the rule as anyone would state
        # it. Thresholding a raw momentum figure at 0.5 would be nonsense.
        return 0.0

    @property
    def holds_whole_universe(self) -> bool:
        return False

    def describe(self) -> str:
        return f"naive momentum (rank on {self.feature}, no fitting)"


def default_baselines(dataset: MLDataset) -> list[Baseline]:
    """The two baselines PRD Module 11 requires, resolved against the panel.

    Raises if no momentum feature is present. Silently returning only
    buy-and-hold would turn a required comparison into an optional one.
    """

    momentum = next(
        (f for f in MOMENTUM_FEATURE_CANDIDATES if f in dataset.X.columns), None
    )
    if momentum is None:
        raise KeyError(
            "no momentum feature found for the naive momentum baseline (looked for "
            f"{list(MOMENTUM_FEATURE_CANDIDATES)}; the dataset has "
            f"{sorted(dataset.feature_names)}). Module 11 requires every model to "
            "be reported against a naive momentum rule, so this is a build error, "
            "not a reason to skip the comparison."
        )
    return [BuyAndHoldBaseline(), NaiveMomentumBaseline(feature=momentum)]


@dataclass(frozen=True, slots=True)
class BaselineOutcome:
    """One baseline's out-of-sample record, on the model's own folds."""

    name: str
    description: str
    metrics: PredictionMetrics
    traded: TradedResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "prediction": dict(self.metrics.as_dict()),
            "traded": dict(self.traded.as_dict()),
        }


def score_predictions(
    dataset: MLDataset,
    idx: np.ndarray,
    scores: np.ndarray,
    *,
    sample: Sample,
    threshold: float = 0.5,
) -> PredictionMetrics:
    """Metrics for ``scores`` against the dataset's labels at ``idx``."""

    truth = dataset.y.to_numpy()[idx]
    if dataset.task is Task.REGRESSION:
        return compute_regression_metrics(truth, scores, sample=sample)
    return compute_classification_metrics(
        truth, scores, sample=sample, threshold=threshold
    )


def prediction_frame(
    dataset: MLDataset, idx: np.ndarray, scores: np.ndarray
) -> pd.DataFrame:
    """Long ``[date, security_id, prediction, forward_return]`` for the trader."""

    return pd.DataFrame(
        {
            "date": dataset.times.t0[idx],
            "security_id": dataset.security_ids[idx],
            "prediction": np.asarray(scores, dtype=float),
            "forward_return": dataset.forward_return[idx],
        }
    )


def baseline_outcome(
    baseline: Baseline,
    dataset: MLDataset,
    idx: np.ndarray,
    scores: np.ndarray,
    *,
    rule: TradingRule,
    horizon: int,
) -> BaselineOutcome:
    """Build an outcome from scores already computed over ``idx``.

    Separate from :func:`evaluate_baseline` because a cross-validated evaluation
    scores each baseline fold by fold and then pools, exactly as it pools the
    model's predictions — comparing a pooled model series against a
    single-fold baseline would not be a comparison.
    """

    metrics = score_predictions(
        dataset,
        idx,
        scores,
        sample=Sample.OUT_OF_SAMPLE,
        threshold=baseline.decision_threshold,
    )
    traded = trade_predictions(
        prediction_frame(dataset, idx, scores),
        label=baseline.name,
        rule=rule,
        horizon=horizon,
        equal_weight_all=baseline.holds_whole_universe,
    )
    describe = getattr(baseline, "describe", None)
    return BaselineOutcome(
        name=baseline.name,
        description=describe() if callable(describe) else baseline.name,
        metrics=metrics,
        traded=traded,
    )


def evaluate_baseline(
    baseline: Baseline,
    dataset: MLDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    rule: TradingRule,
    horizon: int,
) -> BaselineOutcome:
    """Score a baseline on exactly the fold and rule the model was scored on."""

    scores = baseline.calibrate(dataset, train_idx).score(dataset, test_idx)
    return baseline_outcome(
        baseline, dataset, test_idx, scores, rule=rule, horizon=horizon
    )


def _headline_name(metrics: PredictionMetrics) -> str:
    return "rank IC" if isinstance(metrics, RegressionMetrics) else "ROC-AUC"


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """The model against its baselines, and the plain-language verdict.

    Comparison is on two axes, both required to claim the model won:

    * **Predictive** — rank IC (regression) or ROC-AUC (classification), plus
      directional accuracy. A model can rank well and still lose money.
    * **Traded** — net annualised Sharpe of the same long-short rule applied to
      each set of scores. A model can make money and still be ranking no better
      than trailing momentum, in which case the momentum rule is the cheaper
      implementation of the same idea.
    """

    model_label: str
    model_metrics: PredictionMetrics
    model_traded: TradedResult
    baselines: tuple[BaselineOutcome, ...]

    @property
    def losses_on_prediction(self) -> tuple[str, ...]:
        """Baselines whose headline predictive metric the model failed to beat."""

        return tuple(
            b.name
            for b in self.baselines
            if not self.model_metrics.headline > b.metrics.headline
        )

    @property
    def losses_on_trading(self) -> tuple[str, ...]:
        """Baselines whose net Sharpe the model failed to beat."""

        mine = self.model_traded.annualised_sharpe
        return tuple(
            b.name for b in self.baselines if not mine > b.traded.annualised_sharpe
        )

    @property
    def beats_all_baselines(self) -> bool:
        """Strictly better than every baseline on both axes. Deliberately strict."""

        return not self.losses_on_prediction and not self.losses_on_trading

    def verdict(self) -> str:
        """The finding, in plain language — including when the model loses."""

        headline = _headline_name(self.model_metrics)
        mine_pred = self.model_metrics.headline
        mine_sharpe = self.model_traded.annualised_sharpe

        if self.beats_all_baselines:
            return (
                f"VERDICT: {self.model_label} beat both required baselines "
                f"out-of-sample ({headline} {mine_pred:.4f}, net Sharpe "
                f"{mine_sharpe:.3f}). That is a necessary condition, not a "
                "sufficient one — this configuration is one of the trials "
                "recorded in research_log, and the deflated Sharpe is what "
                "decides whether it survives the search that produced it."
            )

        lines = [f"VERDICT: {self.model_label} did NOT beat the required baselines."]
        for baseline in self.baselines:
            notes = []
            if baseline.name in self.losses_on_prediction:
                notes.append(
                    f"{headline} {mine_pred:.4f} vs {baseline.metrics.headline:.4f}"
                )
            if baseline.name in self.losses_on_trading:
                notes.append(
                    f"net Sharpe {mine_sharpe:.3f} vs "
                    f"{baseline.traded.annualised_sharpe:.3f}"
                )
            if notes:
                lines.append(f"  - lost to {baseline.name} on: {'; '.join(notes)}")
        lines.append(
            "  On this evidence the model does not earn its complexity over a rule "
            "that requires no fitting. This is the result; it is reported as the "
            "result rather than buried under a chart of the one fold where it won."
        )
        return "\n".join(lines)

    def render(self) -> str:
        headline = _headline_name(self.model_metrics)
        rows = [
            f"{'':22s} {headline:>10s} {'dir.acc':>10s} {'net Sharpe':>12s} "
            f"{'ann.return':>12s} {'turnover':>10s}",
            f"{self.model_label[:22]:22s} "
            f"{self.model_metrics.headline:10.4f} "
            f"{self.model_metrics.directional_accuracy:10.4f} "
            f"{self.model_traded.annualised_sharpe:12.3f} "
            f"{self.model_traded.annualised_return:11.2%} "
            f"{self.model_traded.mean_turnover:9.2%}",
        ]
        for baseline in self.baselines:
            rows.append(
                f"{baseline.name[:22]:22s} "
                f"{baseline.metrics.headline:10.4f} "
                f"{baseline.metrics.directional_accuracy:10.4f} "
                f"{baseline.traded.annualised_sharpe:12.3f} "
                f"{baseline.traded.annualised_return:11.2%} "
                f"{baseline.traded.mean_turnover:9.2%}"
            )
        rows.append("")
        rows.append(self.verdict())
        return "\n".join(rows)

    def as_dict(self) -> Mapping[str, Any]:
        """Nested under ``metrics["baselines"]`` in the log — never flattened.

        The top level of a run's metrics mapping is scanned by
        :func:`quantlab.research.reporting.assert_no_bare_sharpe`, and a
        baseline's Sharpe is not this run's Sharpe. Keeping it nested is the
        accurate statement of that, and prevents a baseline number from ever
        being paired with the model's deflation.
        """

        return {
            "beats_all_baselines": self.beats_all_baselines,
            "lost_on_prediction_to": list(self.losses_on_prediction),
            "lost_on_trading_to": list(self.losses_on_trading),
            "verdict": self.verdict(),
            **{b.name: b.as_dict() for b in self.baselines},
        }


def compare_to_baselines(
    *,
    model_label: str,
    model_metrics: PredictionMetrics,
    model_traded: TradedResult,
    outcomes: Sequence[BaselineOutcome],
) -> BaselineComparison:
    """Assemble the comparison. Requires at least the two mandated baselines."""

    names = {o.name for o in outcomes}
    required = {"buy_and_hold", "naive_momentum"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            f"missing required baseline(s) {missing}. PRD Module 11 requires every "
            "model to be reported against buy-and-hold and a naive momentum rule; "
            "an evaluation without them is not reportable."
        )
    return BaselineComparison(
        model_label=model_label,
        model_metrics=model_metrics,
        model_traded=model_traded,
        baselines=tuple(outcomes),
    )
