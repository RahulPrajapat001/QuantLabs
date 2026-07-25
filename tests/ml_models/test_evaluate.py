"""Evaluation: what gets logged, what gets reported, and what refuses to run."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantlab.config import Split
from quantlab.ml_models.dataset import TargetKind, TargetSpec, build_dataset
from quantlab.ml_models.evaluate import ModelEvaluationRun, evaluate_model
from quantlab.ml_models.metrics import Sample
from quantlab.ml_models.models import ModelSpec, Task
from quantlab.ml_models.trading import TradingRule
from quantlab.ml_models.validation import PurgedKFold, WalkForward
from quantlab.research.log import STATUS_COMPLETE, STATUS_FAILED, ResearchLogError
from quantlab.research.reporting import ReportingRuleError, assert_no_bare_sharpe

from .conftest import FEATURES, full_universe, make_panel

RIDGE = ModelSpec(family="ridge", task=Task.REGRESSION)
SPLITTER = PurgedKFold(n_splits=3, embargo_days=5)
GROSS = TradingRule(cost_per_unit_traded=0.0)


def rows(session):
    from quantlab.db.models import ResearchLogEntry

    return session.query(ResearchLogEntry).all()


# --- Every configuration is a logged trial -----------------------------------


def test_evaluating_a_model_writes_exactly_one_completed_trial(
    research_log, session, dataset
):
    assert research_log.trials() == 0
    evaluation = evaluate_model(
        research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS
    )

    entries = rows(session)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.run_id == evaluation.run_id
    assert entry.status == STATUS_COMPLETE
    assert entry.strategy_name == "ml:ridge:regression"
    assert research_log.trials() == 1


def test_the_logged_row_records_the_configuration_that_produced_it(
    research_log, session, dataset
):
    evaluate_model(
        research_log,
        dataset,
        ModelSpec(family="ridge", task=Task.REGRESSION, params={"alpha": 42.0}),
        splitter=SPLITTER,
        rule=GROSS,
    )
    parameters = rows(session)[0].parameters
    assert parameters["family"] == "ridge"
    assert parameters["alpha"] == 42.0
    assert parameters["splitter"] == "PurgedKFold"
    assert parameters["n_splits"] == 3
    assert parameters["target"] == "return"
    assert parameters["features"] == list(FEATURES)
    assert parameters["baselines"] == ["buy_and_hold", "naive_momentum"]


def test_the_logged_window_ends_at_the_label_date_not_the_feature_date(
    research_log, session, panel, universe
):
    """A five-day horizon reaches five days past where the features stop."""

    dataset = build_dataset(
        panel,
        features=FEATURES,
        universe=universe,
        index_name="T",
        target=TargetSpec(horizon=5),
    )
    evaluate_model(research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS)
    entry = rows(session)[0]
    assert entry.end_date == dataset.label_end
    assert entry.end_date > dataset.end


def test_a_model_that_raises_leaves_a_failed_row_rather_than_nothing(
    research_log, session, dataset, monkeypatch
):
    def explode(self):
        raise RuntimeError("solver did not converge")

    monkeypatch.setattr(ModelSpec, "build", explode)
    with pytest.raises(RuntimeError, match="did not converge"):
        evaluate_model(research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS)

    entries = rows(session)
    assert len(entries) == 1
    assert entries[0].status == STATUS_FAILED
    assert "did not converge" in entries[0].error
    # A failed run is not a trial, but it is evidence that the attempt happened.
    assert research_log.trials() == 0


def test_the_run_method_cannot_be_bypassed_by_a_subclass():
    """The log write lives in the final run(); a subclass must not reach around it."""

    with pytest.raises(TypeError, match="may not override run"):

        class Sneaky(ModelEvaluationRun):
            def run(self):  # type: ignore[override]
                return None


# --- Out-of-sample discipline ------------------------------------------------


def test_every_headline_metric_is_labelled_by_its_sample(
    research_log, session, dataset
):
    evaluation = evaluate_model(
        research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS
    )
    assert evaluation.out_of_sample.sample is Sample.OUT_OF_SAMPLE
    assert evaluation.in_sample.sample is Sample.IN_SAMPLE

    metrics = rows(session)[0].metrics
    assert "oos_rank_ic" in metrics
    assert "is_rank_ic" in metrics
    assert metrics["oos_rank_ic"] != metrics["is_rank_ic"]


def test_the_model_recovers_the_planted_signal_out_of_sample(research_log, dataset):
    """A sanity check on the harness itself, not on ridge regression.

    The fixture generates ``forward_return = 0.35 * signal + noise``. If a purged
    CV over that panel cannot find a positive out-of-sample rank IC, the harness
    is broken somewhere upstream of any model.
    """

    evaluation = evaluate_model(
        research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS
    )
    assert evaluation.out_of_sample.rank_ic > 0.1
    assert evaluation.out_of_sample.n == len(dataset)


def test_the_overfitting_gap_is_reported_as_in_sample_minus_out_of_sample(
    research_log, session, dataset
):
    evaluation = evaluate_model(
        research_log,
        dataset,
        ModelSpec(
            family="random_forest",
            task=Task.REGRESSION,
            params={"n_estimators": 40, "max_depth": None, "min_samples_leaf": 1},
        ),
        splitter=SPLITTER,
        rule=GROSS,
    )
    assert evaluation.overfitting_gap == pytest.approx(
        evaluation.in_sample.headline - evaluation.out_of_sample.headline
    )
    # A depth-unlimited forest with single-sample leaves memorises the panel.
    assert evaluation.overfitting_gap > 0.2
    assert rows(session)[0].metrics["overfitting_gap"] == pytest.approx(
        evaluation.overfitting_gap
    )


def test_each_fold_reports_its_own_purge_counts(research_log, dataset):
    evaluation = evaluate_model(
        research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS
    )
    assert len(evaluation.folds) == 3
    assert all(f.fold.n_purged >= 0 for f in evaluation.folds)
    assert sum(f.fold.n_embargoed for f in evaluation.folds) > 0
    assert "purged" in evaluation.folds[0].render()


def test_walk_forward_is_accepted_as_a_splitter(research_log, session, dataset):
    evaluate_model(
        research_log,
        dataset,
        RIDGE,
        splitter=WalkForward(mode="rolling", n_splits=3, train_days=120),
        rule=GROSS,
    )
    metrics = rows(session)[0].metrics
    assert "WalkForward" in metrics["splitter"]
    assert rows(session)[0].parameters["mode"] == "rolling"


# --- Classification ----------------------------------------------------------


def test_the_direction_target_logs_classification_metrics(
    research_log, session, panel, universe
):
    dataset = build_dataset(
        panel,
        features=FEATURES,
        universe=universe,
        index_name="T",
        target=TargetSpec(kind=TargetKind.DIRECTION),
    )
    evaluation = evaluate_model(
        research_log,
        dataset,
        ModelSpec(family="ridge", task=Task.CLASSIFICATION),
        splitter=SPLITTER,
        rule=GROSS,
    )
    metrics = rows(session)[0].metrics
    for key in ("oos_accuracy", "oos_precision", "oos_recall", "oos_roc_auc"):
        assert key in metrics
    assert metrics["oos_confusion"].keys() == {"tn", "fp", "fn", "tp"}
    assert evaluation.out_of_sample.roc_auc > 0.5


def test_a_regressor_cannot_be_pointed_at_direction_labels(
    research_log, panel, universe
):
    dataset = build_dataset(
        panel,
        features=FEATURES,
        universe=universe,
        index_name="T",
        target=TargetSpec(kind=TargetKind.DIRECTION),
    )
    with pytest.raises(ValueError, match="configured for regression"):
        evaluate_model(research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS)


# --- Baselines and reporting -------------------------------------------------


def test_the_baseline_verdict_is_written_into_the_log(research_log, session, dataset):
    evaluate_model(research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS)
    metrics = rows(session)[0].metrics
    assert "verdict" in metrics
    assert "VERDICT" in metrics["verdict"]
    assert metrics["baselines"].keys() >= {"buy_and_hold", "naive_momentum"}
    assert isinstance(metrics["beats_all_baselines"], bool)


def test_the_logged_metrics_satisfy_the_no_bare_sharpe_rule(
    research_log, session, dataset
):
    """A metrics dict from an ML run must be safe to hand to the report layer."""

    evaluation = evaluate_model(
        research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS
    )
    metrics = rows(session)[0].metrics
    report = evaluation.sharpe_report(research_log)
    assert_no_bare_sharpe(metrics, report)  # must not raise

    with pytest.raises(ReportingRuleError, match="without a SharpeReport"):
        assert_no_bare_sharpe(metrics, None)


def test_the_sharpe_is_reachable_only_with_its_trials_count(research_log, dataset):
    evaluation = evaluate_model(
        research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS
    )
    report = evaluation.sharpe_report(research_log)
    assert report.trials == research_log.trials() == 1
    assert 0.0 <= report.deflated_sharpe <= 1.0
    assert report.cost_model == "ZeroCostModel"
    assert report.split == "train"
    assert "trials" in report.render()


def test_a_costed_run_names_the_turnover_approximation_in_the_log(
    research_log, session, dataset
):
    evaluate_model(research_log, dataset, RIDGE, splitter=SPLITTER)
    assert rows(session)[0].cost_model == "IndianEquityCostModel(turnover-approx)"


def test_trials_accumulate_across_configurations(research_log, dataset):
    for alpha in (0.1, 1.0, 10.0):
        evaluate_model(
            research_log,
            dataset,
            ModelSpec(family="ridge", task=Task.REGRESSION, params={"alpha": alpha}),
            splitter=SPLITTER,
            rule=GROSS,
        )
    assert research_log.trials() == 3
    # A deflation reads the count itself; it is never handed in.
    last = research_log.entries()[0]
    from quantlab.research.deflated import deflate_run

    assert deflate_run(research_log, last.run_id).trials == 3


# --- Validation-split selection ----------------------------------------------


def test_supplying_a_validation_dataset_moves_selection_onto_it(research_log, session):
    panel = make_panel(n_days=900, start=date(2019, 1, 1))
    universe = full_universe(panel)
    full = build_dataset(panel, features=FEATURES, universe=universe, index_name="T")
    train = full.split(Split.TRAIN)
    validation = full.split(Split.VALIDATION)

    evaluation = evaluate_model(
        research_log,
        train,
        RIDGE,
        splitter=SPLITTER,
        rule=GROSS,
        validation=validation,
    )
    assert evaluation.validation is not None
    assert evaluation.selection_basis == "validation"
    assert evaluation.selection_metric == evaluation.validation.headline

    entry = rows(session)[0]
    assert entry.metrics["sharpe_source"] == "validation"
    assert "validation" in entry.metrics
    assert entry.split == "train+validation"
    assert entry.parameters["validation_split_used"] is True


def test_without_a_validation_dataset_selection_falls_back_to_the_cv_score(
    research_log, dataset
):
    evaluation = evaluate_model(
        research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS
    )
    assert evaluation.validation is None
    assert evaluation.selection_basis == "cv_out_of_sample"
    assert evaluation.selection_metric == evaluation.out_of_sample.headline


def test_render_states_the_sample_of_every_block(research_log, dataset):
    evaluation = evaluate_model(
        research_log, dataset, RIDGE, splitter=SPLITTER, rule=GROSS
    )
    text = evaluation.render(research_log)
    assert "OUT-OF-SAMPLE" in text
    assert "IN-SAMPLE (diagnostic only — not a result)" in text
    assert "overfitting gap" in text
    assert "VERDICT" in text
    assert "deflated" in text


# --- SHAP --------------------------------------------------------------------


def test_shap_attribution_is_attached_when_requested(research_log, session, dataset):
    evaluation = evaluate_model(
        research_log,
        dataset,
        ModelSpec(family="lightgbm", task=Task.REGRESSION, params={"n_estimators": 30}),
        splitter=SPLITTER,
        rule=GROSS,
        compute_shap=True,
        shap_max_samples=300,
    )
    assert evaluation.shap is not None
    ranked = evaluation.shap.ranked
    assert ranked[0][0] == "signal", "the planted signal should dominate"
    assert set(rows(session)[0].metrics["shap"]) >= {"explainer", "mean_abs_shap"}


def test_an_unlogged_evaluation_path_does_not_exist(dataset):
    """There is no way to get a ModelEvaluation without a ResearchLog."""

    import inspect

    signature = inspect.signature(evaluate_model)
    first = next(iter(signature.parameters.values()))
    assert first.name == "research_log"
    assert first.default is inspect.Parameter.empty


def test_an_engine_that_never_records_metrics_raises(research_log, dataset):
    class Silent(ModelEvaluationRun):
        def execute(self, run):
            from quantlab.research.log import RunResult

            RunResult.from_returns(np.array([0.01, -0.01, 0.02]))
            raise ResearchLogError("deliberate")

    engine = Silent(
        research_log, dataset=dataset, model=RIDGE, splitter=SPLITTER, rule=GROSS
    )
    with pytest.raises(ResearchLogError):
        engine.run()
