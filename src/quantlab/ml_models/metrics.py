"""Prediction metrics that cannot be printed without saying which sample they came from.

PRD Module 11: *"Every reported metric out-of-sample; in-sample metrics logged
separately and explicitly labelled."* An in-sample R² of 0.4 and an out-of-sample
R² of 0.4 are different claims about the world, and the difference between them is
the entire content of the module. So the sample label is not an argument to
:meth:`render` that a caller can forget — it is a required field of the metrics
object itself, set at the moment the metrics are computed, and it prefixes every
key :meth:`as_dict` emits (``oos_mae`` / ``is_mae``). A metrics dict from a
training fold cannot be mistaken for a result later, because its keys say so.

In-sample metrics are kept rather than discarded because the *gap* between them is
the overfitting diagnostic: a tree ensemble at in-sample R² 0.85 and out-of-sample
R² 0.001 has memorised the panel, and only the pair reveals that.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

__all__ = [
    "ClassificationMetrics",
    "PredictionMetrics",
    "RegressionMetrics",
    "Sample",
    "compute_classification_metrics",
    "compute_regression_metrics",
]


class Sample(StrEnum):
    """Which sample a metric was computed on. Never optional."""

    IN_SAMPLE = "in_sample"
    OUT_OF_SAMPLE = "out_of_sample"

    @property
    def prefix(self) -> str:
        return "is_" if self is Sample.IN_SAMPLE else "oos_"

    @property
    def banner(self) -> str:
        if self is Sample.IN_SAMPLE:
            return "IN-SAMPLE (diagnostic only — not a result)"
        return "OUT-OF-SAMPLE"


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Next-day-return regression metrics (PRD Module 11)."""

    sample: Sample
    n: int
    mae: float
    rmse: float
    r2: float
    #: Share of predictions whose *sign* matches the realised return's. For daily
    #: equity returns this is usually the only part of a regression that a
    #: strategy can trade, so it is reported beside the error metrics.
    directional_accuracy: float
    #: Spearman correlation of prediction against realised return — the pooled
    #: information coefficient. Rank-based, so a few outliers cannot manufacture
    #: it the way they can manufacture an R².
    rank_ic: float

    @property
    def task(self) -> str:
        return "regression"

    def as_dict(self) -> dict[str, Any]:
        p = self.sample.prefix
        return {
            f"{p}n": self.n,
            f"{p}mae": self.mae,
            f"{p}rmse": self.rmse,
            f"{p}r2": self.r2,
            f"{p}directional_accuracy": self.directional_accuracy,
            f"{p}rank_ic": self.rank_ic,
        }

    def render(self) -> str:
        return "\n".join(
            [
                f"{self.sample.banner} — regression, n={self.n}",
                f"  MAE:                   {self.mae:12.6f}",
                f"  RMSE:                  {self.rmse:12.6f}",
                f"  R²:                    {self.r2:12.6f}",
                f"  Directional accuracy:  {self.directional_accuracy:12.4f}",
                f"  Rank IC (Spearman):    {self.rank_ic:12.4f}",
            ]
        )

    @property
    def headline(self) -> float:
        """The single number model selection ranks on. Higher is better."""

        return self.rank_ic


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Direction-classification metrics (PRD Module 11)."""

    sample: Sample
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    #: ``nan`` when the test fold contains a single class — an AUC is undefined
    #: there, and returning 0.5 would be inventing a number.
    roc_auc: float
    #: ``((tn, fp), (fn, tp))``.
    confusion: tuple[tuple[int, int], tuple[int, int]]
    #: Share of the sample that is class 1. The honest floor: a classifier that
    #: always predicts "up" scores this, and equity direction is not 50/50.
    base_rate: float

    @property
    def task(self) -> str:
        return "classification"

    @property
    def directional_accuracy(self) -> float:
        """Alias — for direction classification, accuracy *is* directional accuracy."""

        return self.accuracy

    @property
    def lift_over_base_rate(self) -> float:
        """Accuracy minus the always-predict-the-majority-class accuracy.

        Negative means the model is worse than a constant. That happens more
        often than published results suggest, which is why it is a property here
        rather than something the reader is left to compute.
        """

        majority = max(self.base_rate, 1.0 - self.base_rate)
        return self.accuracy - majority

    def as_dict(self) -> dict[str, Any]:
        p = self.sample.prefix
        (tn, fp), (fn, tp) = self.confusion
        return {
            f"{p}n": self.n,
            f"{p}accuracy": self.accuracy,
            f"{p}precision": self.precision,
            f"{p}recall": self.recall,
            f"{p}f1": self.f1,
            f"{p}roc_auc": self.roc_auc,
            f"{p}base_rate": self.base_rate,
            f"{p}lift_over_base_rate": self.lift_over_base_rate,
            f"{p}confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    def render(self) -> str:
        (tn, fp), (fn, tp) = self.confusion
        auc = "     n/a" if np.isnan(self.roc_auc) else f"{self.roc_auc:12.4f}"
        return "\n".join(
            [
                f"{self.sample.banner} — classification, n={self.n}",
                f"  Accuracy:              {self.accuracy:12.4f}",
                f"  Precision:             {self.precision:12.4f}",
                f"  Recall:                {self.recall:12.4f}",
                f"  F1:                    {self.f1:12.4f}",
                f"  ROC-AUC:               {auc}",
                f"  Base rate (class 1):   {self.base_rate:12.4f}",
                f"  Lift over base rate:   {self.lift_over_base_rate:+12.4f}",
                "  Confusion  [pred 0, pred 1]",
                f"    actual 0  [{tn:8d}, {fp:8d}]",
                f"    actual 1  [{fn:8d}, {tp:8d}]",
            ]
        )

    @property
    def headline(self) -> float:
        """The single number model selection ranks on. Higher is better."""

        return 0.5 if np.isnan(self.roc_auc) else self.roc_auc


#: Either metric family. Used wherever the task is not known statically.
PredictionMetrics = RegressionMetrics | ClassificationMetrics


def _finite_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true has {true.size} observations and y_pred has {pred.size}; "
            "metrics on misaligned arrays are meaningless"
        )
    keep = np.isfinite(true) & np.isfinite(pred)
    return true[keep], pred[keep]


def compute_regression_metrics(
    y_true: Any, y_pred: Any, *, sample: Sample
) -> RegressionMetrics:
    """MAE, RMSE, R², directional accuracy and rank IC for one sample."""

    true, pred = _finite_pair(y_true, y_pred)
    n = int(true.size)
    if n == 0:
        raise ValueError("no finite (y_true, y_pred) pairs to score")

    errors = pred - true
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors**2).mean()))
    total_ss = float(((true - true.mean()) ** 2).sum())
    r2 = float(1.0 - (errors**2).sum() / total_ss) if total_ss > 0.0 else float("nan")

    # Zero predictions and zero returns are not directional calls; scoring them
    # as correct would let a model that predicts 0.0 everywhere post 100%.
    directional = np.sign(pred) == np.sign(true)
    scoreable = (np.sign(pred) != 0) & (np.sign(true) != 0)
    hit = float(directional[scoreable].mean()) if scoreable.any() else float("nan")

    from scipy import stats

    rank_ic = float("nan")
    if n > 2 and np.ptp(true) > 0.0 and np.ptp(pred) > 0.0:
        rank_ic = float(stats.spearmanr(pred, true).statistic)

    return RegressionMetrics(
        sample=sample,
        n=n,
        mae=mae,
        rmse=rmse,
        r2=r2,
        directional_accuracy=hit,
        rank_ic=0.0 if np.isnan(rank_ic) else rank_ic,
    )


def compute_classification_metrics(
    y_true: Any,
    y_score: Any,
    *,
    sample: Sample,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix.

    ``y_score`` is the predicted probability of class 1 where the estimator can
    produce one; hard 0/1 labels also work, and then ROC-AUC degenerates to a
    rank statistic over two values (still meaningful, just coarse).
    """

    true_raw, score = _finite_pair(y_true, y_score)
    n = int(true_raw.size)
    if n == 0:
        raise ValueError("no finite (y_true, y_score) pairs to score")
    true = (true_raw > 0.5).astype(int)
    pred = (score >= threshold).astype(int)

    tp = int(np.count_nonzero((pred == 1) & (true == 1)))
    tn = int(np.count_nonzero((pred == 0) & (true == 0)))
    fp = int(np.count_nonzero((pred == 1) & (true == 0)))
    fn = int(np.count_nonzero((pred == 0) & (true == 1)))

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    roc_auc = float("nan")
    if len(np.unique(true)) == 2:
        from sklearn.metrics import roc_auc_score

        roc_auc = float(roc_auc_score(true, score))

    return ClassificationMetrics(
        sample=sample,
        n=n,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=roc_auc,
        confusion=((tn, fp), (fn, tp)),
        base_rate=float(true.mean()),
    )


def metrics_as_dict(metrics: Mapping[str, PredictionMetrics]) -> dict[str, Any]:
    """Flatten several labelled metric objects into one logged mapping."""

    out: dict[str, Any] = {}
    for prefix, value in metrics.items():
        for key, item in value.as_dict().items():
            out[f"{prefix}_{key}" if prefix else key] = item
    return out
