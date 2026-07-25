"""Evaluating one model configuration — and logging it, whether or not it worked.

This is where PRD Module 11's trials-accounting rule bites: *"Every model
configuration writes to ``research_log`` and counts toward the trials total. A
model's traded Sharpe is deflated by the full trials count including model
search."* A hyper-parameter sweep is a search over strategies by another name, and
a deflated Sharpe that counts the twelve backtests but not the two hundred model
fits is not a correction, it is a rounding of the problem down to a comfortable
size.

So the write is not optional and not a decorator. :class:`ModelEvaluationRun`
subclasses :class:`~quantlab.backtesting.logged.LoggedBacktestEngine`, whose
``run`` is ``@final`` and performs the log write itself; this class implements
``execute``, which is called from *inside* the log's context manager. A crash
leaves a ``failed`` row; returning without recording metrics raises. The same
machinery that makes a backtest unloggable-only makes a model fit unloggable-only,
which is the point — they are the same kind of event.

**What gets reported.** Every headline number is out-of-sample:

* ``out_of_sample`` — predictions pooled across the purged CV folds. Each fold's
  test block is disjoint in time from the others, so pooling produces one
  full-period series rather than an average of averages.
* ``validation`` — the model fitted on the whole training dataset and scored on
  the separate validation dataset. This is the number model *selection* uses
  (PRD: "model selection on validation split"), and when present it is the run's
  logged Sharpe.
* ``in_sample`` — the same final fit scored on its own training data. Kept, and
  labelled ``IN-SAMPLE (diagnostic only — not a result)`` by the metrics type
  itself, because the *gap* between it and the out-of-sample figure is the
  overfitting measurement. An ensemble at in-sample R² 0.8 and out-of-sample R²
  0.001 has memorised the panel, and only the pair shows it.

The holdout is not reachable from here. :func:`build_dataset` refuses to assemble
one that reads a locked price, and the research log independently re-checks the
window before it writes a row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import structlog

from quantlab.backtesting.logged import LoggedBacktestEngine
from quantlab.ml_models.attribution import ShapAttribution, shap_attribution
from quantlab.ml_models.baselines import (
    Baseline,
    BaselineComparison,
    BaselineOutcome,
    baseline_outcome,
    compare_to_baselines,
    default_baselines,
    prediction_frame,
    score_predictions,
)
from quantlab.ml_models.dataset import MLDataset
from quantlab.ml_models.metrics import PredictionMetrics, Sample
from quantlab.ml_models.models import Estimator, ModelSpec, Task, supports_probability
from quantlab.ml_models.trading import TradedResult, TradingRule, trade_predictions
from quantlab.ml_models.validation import Fold, PurgedKFold, Splitter
from quantlab.research.log import LoggedRun, ResearchLog, RunResult, RunSpec
from quantlab.research.reporting import SharpeReport

log = structlog.get_logger(__name__)

__all__ = [
    "FoldOutcome",
    "ModelEvaluation",
    "ModelEvaluationRun",
    "evaluate_model",
]


def score_estimator(
    estimator: Estimator, frame: pd.DataFrame, task: Task
) -> np.ndarray:
    """The estimator's score for each row: a return, or P(up).

    Classification uses the probability rather than the hard label wherever it is
    available. ROC-AUC needs a ranking, and so does the trading rule — a
    long-short book built from 0/1 labels cannot tell its most confident call
    from its least.
    """

    if task is Task.CLASSIFICATION and supports_probability(estimator):
        proba = np.asarray(estimator.predict_proba(frame))  # type: ignore[attr-defined]
        return np.asarray(proba[:, -1], dtype=float)
    return np.asarray(estimator.predict(frame), dtype=float)


@dataclass(frozen=True, slots=True)
class FoldOutcome:
    """One fold's record: the split, and what the model scored on either side."""

    fold: Fold
    out_of_sample: PredictionMetrics
    in_sample: PredictionMetrics

    def render(self) -> str:
        return (
            f"  {self.fold.render()}\n"
            f"      OOS {self.out_of_sample.headline:+.4f} | "
            f"IS {self.in_sample.headline:+.4f} | "
            f"gap {self.in_sample.headline - self.out_of_sample.headline:+.4f}"
        )


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Everything one model configuration produced, and its logged identity."""

    model: ModelSpec
    dataset_description: str
    splitter_description: str
    trading_rule: TradingRule
    folds: tuple[FoldOutcome, ...]
    #: Pooled across the purged CV folds. Out-of-sample.
    out_of_sample: PredictionMetrics
    #: Final fit scored on its own training data. Diagnostic only.
    in_sample: PredictionMetrics
    #: The CV out-of-sample predictions, traded.
    traded: TradedResult
    comparison: BaselineComparison
    run_id: str
    n_samples: int
    feature_names: tuple[str, ...]
    #: Validation-split results, when a validation dataset was supplied. This is
    #: what model selection ranks on.
    validation: PredictionMetrics | None = None
    validation_traded: TradedResult | None = None
    validation_comparison: BaselineComparison | None = None
    shap: ShapAttribution | None = None
    search_iteration: int | None = None

    @property
    def selection_metric(self) -> float:
        """The number model selection ranks on. Higher is better.

        The validation-split score when there is one, because that is the split
        PRD Module 11 reserves for selection; the pooled CV score otherwise.
        """

        return (
            self.validation.headline
            if self.validation is not None
            else self.out_of_sample.headline
        )

    @property
    def selection_basis(self) -> str:
        return "validation" if self.validation is not None else "cv_out_of_sample"

    @property
    def headline_traded(self) -> TradedResult:
        """The traded series the logged Sharpe was computed from.

        Explicit ``is not None`` rather than ``or``: :class:`TradedResult` defines
        ``__len__``, so an empty series would be falsy and silently fall back to
        the CV one — reporting a cross-validated Sharpe as though it were a
        held-back one.
        """

        return self.traded if self.validation_traded is None else self.validation_traded

    @property
    def headline_comparison(self) -> BaselineComparison:
        if self.validation_comparison is None:
            return self.comparison
        return self.validation_comparison

    @property
    def overfitting_gap(self) -> float:
        """In-sample headline minus out-of-sample headline. Large is bad."""

        return self.in_sample.headline - self.out_of_sample.headline

    @property
    def beats_baselines(self) -> bool:
        return self.headline_comparison.beats_all_baselines

    def sharpe_report(
        self,
        research_log: ResearchLog,
        *,
        pbo: float | None = None,
        scope_strategy: str | None = None,
    ) -> SharpeReport:
        """The traded Sharpe with its trials count and deflation attached.

        The only path from this evaluation to a reportable Sharpe. The trials
        count is read from ``research_log`` and spans **the whole log** by
        default — every strategy backtest and every model configuration — because
        they are one selection process.
        """

        return SharpeReport.from_log(
            research_log, self.run_id, pbo=pbo, scope_strategy=scope_strategy
        )

    def render(self, research_log: ResearchLog | None = None) -> str:
        lines = [
            f"{self.model.describe()}",
            f"  dataset:  {self.dataset_description}",
            f"  folds:    {self.splitter_description}",
            f"  trading:  {self.trading_rule.describe()}",
            f"  features: {len(self.feature_names)}",
            "",
            self.out_of_sample.render(),
            "",
            self.in_sample.render(),
            f"  overfitting gap (IS - OOS): {self.overfitting_gap:+.4f}",
            "",
        ]
        if self.validation is not None:
            lines.append("Validation split (model selection runs on this):")
            lines.append(self.validation.render())
            lines.append("")
        lines.append("Per fold:")
        lines.extend(f.render() for f in self.folds)
        lines.append("")
        lines.append(self.headline_traded.render())
        lines.append("")
        lines.append(self.headline_comparison.render())
        if self.shap is not None:
            lines.append("")
            lines.append(self.shap.render())
        lines.append("")
        if research_log is not None:
            lines.append(self.sharpe_report(research_log).render())
        else:
            lines.append(
                f"run_id {self.run_id} — deflate with "
                f"`quantlab research deflate --run-id {self.run_id}`"
            )
        return "\n".join(lines)


class ModelEvaluationRun(LoggedBacktestEngine):
    """A single model configuration, evaluated inside the research log.

    Instantiating this does nothing; :meth:`run` (inherited and ``final``) opens
    the log row, calls :meth:`execute`, and closes the row. Use
    :func:`evaluate_model` unless you need the engine handle.
    """

    def __init__(
        self,
        research_log: ResearchLog,
        *,
        dataset: MLDataset,
        model: ModelSpec,
        splitter: Splitter | None = None,
        rule: TradingRule | None = None,
        validation: MLDataset | None = None,
        baselines: Sequence[Baseline] | None = None,
        hypothesis_id: str | None = None,
        allow_holdout: bool = False,
        notes: str | None = None,
        search_iteration: int | None = None,
        compute_shap: bool = False,
        shap_max_samples: int = 2000,
    ) -> None:
        super().__init__(research_log)
        if model.task is not dataset.task:
            raise ValueError(
                f"model {model.family!r} is configured for {model.task.value} but "
                f"the dataset's target is {dataset.task.value}. A regressor scored "
                "against direction labels produces metrics that look valid and "
                "mean nothing."
            )
        if validation is not None and validation.task is not dataset.task:
            raise ValueError(
                "the validation dataset has a different target from the training "
                "dataset; they must predict the same thing to be comparable"
            )
        self._dataset = dataset
        self._model = model
        self._splitter: Splitter = splitter or PurgedKFold()
        self._rule = rule or TradingRule()
        self._validation = validation
        self._baselines = list(baselines) if baselines else default_baselines(dataset)
        self._hypothesis_id = hypothesis_id
        self._allow_holdout = allow_holdout
        self._notes = notes
        self._search_iteration = search_iteration
        self._compute_shap = compute_shap
        self._shap_max_samples = shap_max_samples
        self._evaluation: ModelEvaluation | None = None

    # --- Logged-run contract ---

    def build_spec(self) -> RunSpec:
        dataset = self._dataset
        horizon = dataset.target.horizon
        # The window ends at the last date any label reads a price from — not the
        # last feature date. The split label and the holdout check both depend on
        # it, and a five-day horizon reaches five days past where it looks like it
        # stops.
        end = (
            max(dataset.label_end, self._validation.label_end)
            if self._validation is not None
            else dataset.label_end
        )
        start = (
            min(dataset.start, self._validation.start)
            if self._validation is not None
            else dataset.start
        )
        parameters: dict[str, Any] = {
            **self._model.parameters(),
            **dataset.target.parameters(),
            **self._splitter.parameters(),
            **self._rule.parameters(),
            "n_samples": len(dataset),
            "n_features": len(dataset.feature_names),
            "features": list(dataset.feature_names),
            "baselines": [b.name for b in self._baselines],
            "validation_split_used": self._validation is not None,
        }
        if self._search_iteration is not None:
            parameters["search_iteration"] = self._search_iteration
        return RunSpec(
            strategy_name=self._model.name,
            universe="point-in-time",
            index_name=dataset.index_name,
            start=start,
            end=end,
            cost_model=self._rule.effective_cost_model,
            parameters=parameters,
            hypothesis_id=self._hypothesis_id,
            holdout=self._allow_holdout,
            periods_per_year=max(1, round(252 / horizon)),
            notes=self._notes,
        )

    def execute(self, run: LoggedRun) -> RunResult:
        dataset = self._dataset
        horizon = dataset.target.horizon
        folds = self._splitter.split(dataset.times)
        if not folds:
            raise ValueError("the splitter produced no folds; nothing to evaluate")

        fold_outcomes: list[FoldOutcome] = []
        oos_idx: list[np.ndarray] = []
        oos_scores: list[np.ndarray] = []
        baseline_scores: dict[str, list[np.ndarray]] = {
            b.name: [] for b in self._baselines
        }

        for fold in folds:
            estimator = self._model.build()
            train_features = dataset.X.iloc[fold.train_idx]
            train_target = dataset.y.iloc[fold.train_idx]
            estimator.fit(train_features, train_target)

            test_scores = score_estimator(
                estimator, dataset.X.iloc[fold.test_idx], dataset.task
            )
            train_scores = score_estimator(estimator, train_features, dataset.task)
            fold_outcomes.append(
                FoldOutcome(
                    fold=fold,
                    out_of_sample=score_predictions(
                        dataset,
                        fold.test_idx,
                        test_scores,
                        sample=Sample.OUT_OF_SAMPLE,
                    ),
                    in_sample=score_predictions(
                        dataset,
                        fold.train_idx,
                        train_scores,
                        sample=Sample.IN_SAMPLE,
                    ),
                )
            )
            oos_idx.append(fold.test_idx)
            oos_scores.append(test_scores)
            for baseline in self._baselines:
                baseline_scores[baseline.name].append(
                    baseline.calibrate(dataset, fold.train_idx).score(
                        dataset, fold.test_idx
                    )
                )

        pooled_idx = np.concatenate(oos_idx)
        pooled_scores = np.concatenate(oos_scores)
        out_of_sample = score_predictions(
            dataset, pooled_idx, pooled_scores, sample=Sample.OUT_OF_SAMPLE
        )
        traded = trade_predictions(
            prediction_frame(dataset, pooled_idx, pooled_scores),
            label=self._model.name,
            rule=self._rule,
            horizon=horizon,
        )
        outcomes = [
            baseline_outcome(
                baseline,
                dataset,
                pooled_idx,
                np.concatenate(baseline_scores[baseline.name]),
                rule=self._rule,
                horizon=horizon,
            )
            for baseline in self._baselines
        ]
        comparison = compare_to_baselines(
            model_label=self._model.name,
            model_metrics=out_of_sample,
            model_traded=traded,
            outcomes=outcomes,
        )

        # The final fit: everything the model is allowed to see. Supplies the
        # in-sample diagnostic, the SHAP attribution, and the validation scores.
        final = self._model.build()
        final.fit(dataset.X, dataset.y)
        in_sample = score_predictions(
            dataset,
            np.arange(len(dataset)),
            score_estimator(final, dataset.X, dataset.task),
            sample=Sample.IN_SAMPLE,
        )

        validation_metrics: PredictionMetrics | None = None
        validation_traded: TradedResult | None = None
        validation_comparison: BaselineComparison | None = None
        if self._validation is not None:
            validation_metrics, validation_traded, validation_comparison = (
                self._score_validation(final)
            )

        attribution: ShapAttribution | None = None
        if self._compute_shap:
            attribution = shap_attribution(
                final, dataset.X, spec=self._model, max_samples=self._shap_max_samples
            )

        headline_traded = traded if validation_traded is None else validation_traded
        headline_comparison = (
            comparison if validation_comparison is None else validation_comparison
        )

        metrics: dict[str, Any] = {
            "splitter": self._splitter.describe(),
            "n_folds": len(folds),
            "n_samples": len(dataset),
            "n_features": len(dataset.feature_names),
            "total_purged": sum(f.n_purged for f in folds),
            "total_embargoed": sum(f.n_embargoed for f in folds),
            "sharpe_source": (
                "validation" if validation_traded is not None else "cv_out_of_sample"
            ),
            "beats_all_baselines": headline_comparison.beats_all_baselines,
            "overfitting_gap": in_sample.headline - out_of_sample.headline,
            **out_of_sample.as_dict(),
            **in_sample.as_dict(),
            # Nested deliberately: the top level of a metrics mapping is scanned
            # by assert_no_bare_sharpe, and neither the CV series' Sharpe nor a
            # baseline's is *this run's* Sharpe. Flattening them would pair a
            # different series' number with this run's deflation.
            "cv_traded": dict(traded.as_dict()),
            "baselines": dict(headline_comparison.as_dict()),
            "verdict": headline_comparison.verdict(),
        }
        if validation_metrics is not None and validation_traded is not None:
            metrics["validation"] = {
                **validation_metrics.as_dict(),
                "traded": dict(validation_traded.as_dict()),
            }
        if attribution is not None:
            metrics["shap"] = dict(attribution.as_dict())

        result = RunResult.from_returns(
            headline_traded.net_returns,
            metrics=metrics,
            periods_per_year=headline_traded.periods_per_year,
        )

        self._evaluation = ModelEvaluation(
            model=self._model,
            dataset_description=dataset.describe(),
            splitter_description=self._splitter.describe(),
            trading_rule=self._rule,
            folds=tuple(fold_outcomes),
            out_of_sample=out_of_sample,
            in_sample=in_sample,
            traded=traded,
            comparison=comparison,
            run_id=run.run_id,
            n_samples=len(dataset),
            feature_names=dataset.feature_names,
            validation=validation_metrics,
            validation_traded=validation_traded,
            validation_comparison=validation_comparison,
            shap=attribution,
            search_iteration=self._search_iteration,
        )
        log.info(
            "ml.evaluated",
            run_id=run.run_id,
            model=self._model.name,
            folds=len(folds),
            oos_headline=out_of_sample.headline,
            is_headline=in_sample.headline,
            beats_baselines=headline_comparison.beats_all_baselines,
        )
        return result

    def _score_validation(
        self, final: Estimator
    ) -> tuple[PredictionMetrics, TradedResult, BaselineComparison]:
        """Score the train-fitted model on the held-back validation dataset.

        Baselines are calibrated on the **training** dataset and scored on the
        validation one — the same asymmetry the model gets, and the reason
        :class:`Baseline` separates calibration from scoring.
        """

        validation = self._validation
        assert validation is not None  # narrowed by the caller
        idx = np.arange(len(validation))
        scores = score_estimator(final, validation.X, validation.task)
        metrics = score_predictions(
            validation, idx, scores, sample=Sample.OUT_OF_SAMPLE
        )
        traded = trade_predictions(
            prediction_frame(validation, idx, scores),
            label=f"{self._model.name} (validation)",
            rule=self._rule,
            horizon=validation.target.horizon,
        )
        outcomes: list[BaselineOutcome] = []
        for baseline in self._baselines:
            calibrated = baseline.calibrate(self._dataset)
            outcomes.append(
                baseline_outcome(
                    calibrated,
                    validation,
                    idx,
                    calibrated.score(validation, idx),
                    rule=self._rule,
                    horizon=validation.target.horizon,
                )
            )
        comparison = compare_to_baselines(
            model_label=f"{self._model.name} (validation)",
            model_metrics=metrics,
            model_traded=traded,
            outcomes=outcomes,
        )
        return metrics, traded, comparison

    def evaluate(self) -> ModelEvaluation:
        """Run inside the log and return the evaluation."""

        self.run()
        if self._evaluation is None:  # pragma: no cover - execute always sets it
            raise RuntimeError("execute() did not produce an evaluation")
        return self._evaluation


def evaluate_model(
    research_log: ResearchLog,
    dataset: MLDataset,
    model: ModelSpec,
    *,
    splitter: Splitter | None = None,
    rule: TradingRule | None = None,
    validation: MLDataset | None = None,
    baselines: Sequence[Baseline] | None = None,
    hypothesis_id: str | None = None,
    allow_holdout: bool = False,
    notes: str | None = None,
    search_iteration: int | None = None,
    compute_shap: bool = False,
    shap_max_samples: int = 2000,
) -> ModelEvaluation:
    """Evaluate one model configuration and log it as a trial.

    There is no unlogged variant, and adding one would silently deflate every
    Sharpe computed afterwards by too little.
    """

    return ModelEvaluationRun(
        research_log,
        dataset=dataset,
        model=model,
        splitter=splitter,
        rule=rule,
        validation=validation,
        baselines=baselines,
        hypothesis_id=hypothesis_id,
        allow_holdout=allow_holdout,
        notes=notes,
        search_iteration=search_iteration,
        compute_shap=compute_shap,
        shap_max_samples=shap_max_samples,
    ).evaluate()
