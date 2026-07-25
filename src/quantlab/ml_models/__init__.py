"""Machine learning under validation discipline (PRD Module 11).

Five model families — ridge, lasso, random forest, XGBoost, LightGBM — against two
targets: the next-day return (regression) and its direction (classification). Deep
sequence models are a deliberate scope cut (``CLAUDE.md`` §4), and
:data:`~quantlab.ml_models.models.MODEL_FAMILIES` is the only construction path,
so the cut is enforced rather than documented.

The models are the least important part of this package. What it is actually for:

* :mod:`~quantlab.ml_models.validation` — purged k-fold with an embargo, and
  walk-forward in expanding and rolling form. Standard k-fold leaks in a time
  series; purging removes training samples whose label windows overlap the test
  block, and the embargo removes what serial correlation carries across the
  boundary afterwards. Every fold reports how much it dropped, and every splitter
  re-checks its own output for leakage before returning it.
* :mod:`~quantlab.ml_models.dataset` — samples carry both a feature date and a
  label date, rows are restricted to the point-in-time universe, and the holdout
  lock is checked against the *label* date, so a horizon cannot reach past the
  boundary unnoticed.
* :mod:`~quantlab.ml_models.metrics` — every metric object states which sample it
  came from, and prefixes its own keys accordingly. An in-sample number cannot be
  mistaken for a result downstream.
* :mod:`~quantlab.ml_models.baselines` — buy-and-hold and a naive momentum rule,
  scored on the same folds through the same functions, with a plain-language
  verdict when the model loses.
* :mod:`~quantlab.ml_models.evaluate` and :mod:`~quantlab.ml_models.search` —
  every configuration writes a ``research_log`` row from inside the engine, and
  the winner's Sharpe is deflated by the whole log's trials count.
"""

from quantlab.ml_models.attribution import ShapAttribution, shap_attribution
from quantlab.ml_models.baselines import (
    Baseline,
    BaselineComparison,
    BaselineOutcome,
    BuyAndHoldBaseline,
    NaiveMomentumBaseline,
    default_baselines,
)
from quantlab.ml_models.dataset import (
    MLDataset,
    SampleTimes,
    TargetKind,
    TargetSpec,
    build_dataset,
    restrict_to_universe,
)
from quantlab.ml_models.evaluate import (
    FoldOutcome,
    ModelEvaluation,
    ModelEvaluationRun,
    evaluate_model,
)
from quantlab.ml_models.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    Sample,
    compute_classification_metrics,
    compute_regression_metrics,
)
from quantlab.ml_models.models import (
    DEFAULT_SEARCH_SPACE,
    MODEL_FAMILIES,
    ModelSpec,
    Task,
    build_model,
)
from quantlab.ml_models.search import (
    SearchResult,
    SearchSpace,
    default_space,
    run_search,
)
from quantlab.ml_models.trading import TradedResult, TradingRule, trade_predictions
from quantlab.ml_models.validation import (
    Fold,
    LeakageError,
    PurgedKFold,
    Splitter,
    WalkForward,
    assert_no_label_leakage,
)

__all__ = [
    "DEFAULT_SEARCH_SPACE",
    "MODEL_FAMILIES",
    "Baseline",
    "BaselineComparison",
    "BaselineOutcome",
    "BuyAndHoldBaseline",
    "ClassificationMetrics",
    "Fold",
    "FoldOutcome",
    "LeakageError",
    "MLDataset",
    "ModelEvaluation",
    "ModelEvaluationRun",
    "ModelSpec",
    "NaiveMomentumBaseline",
    "PurgedKFold",
    "RegressionMetrics",
    "Sample",
    "SampleTimes",
    "SearchResult",
    "SearchSpace",
    "ShapAttribution",
    "Splitter",
    "TargetKind",
    "TargetSpec",
    "Task",
    "TradedResult",
    "TradingRule",
    "WalkForward",
    "assert_no_label_leakage",
    "build_dataset",
    "build_model",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "default_baselines",
    "default_space",
    "evaluate_model",
    "restrict_to_universe",
    "run_search",
    "shap_attribution",
    "trade_predictions",
]
