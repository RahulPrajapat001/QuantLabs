"""Model search: every configuration counts, and the count is reported."""

from __future__ import annotations

import pytest

from quantlab.ml_models.models import DEFAULT_SEARCH_SPACE, MODEL_FAMILIES, ModelSpec
from quantlab.ml_models.search import SearchSpace, default_space, run_search
from quantlab.ml_models.trading import TradingRule
from quantlab.ml_models.validation import PurgedKFold

SPLITTER = PurgedKFold(n_splits=3, embargo_days=5)
GROSS = TradingRule(cost_per_unit_traded=0.0)
SMALL = SearchSpace(family="ridge", grid={"alpha": (0.1, 10.0, 1000.0)})


# --- The space is data -------------------------------------------------------


def test_the_grid_expands_to_the_cartesian_product():
    space = SearchSpace(
        family="xgboost", grid={"max_depth": (2, 3), "learning_rate": (0.03, 0.1)}
    )
    assert space.size == 4
    configurations = space.configurations()
    assert len(configurations) == 4
    assert {tuple(sorted(c.items())) for c in configurations} == {
        (("learning_rate", 0.03), ("max_depth", 2)),
        (("learning_rate", 0.03), ("max_depth", 3)),
        (("learning_rate", 0.1), ("max_depth", 2)),
        (("learning_rate", 0.1), ("max_depth", 3)),
    }


def test_the_configuration_order_is_stable():
    """A shifting order would make ``search_iteration`` in the log meaningless."""

    space = SearchSpace(family="lightgbm", grid={"num_leaves": (7, 15, 31)})
    assert space.configurations() == space.configurations()


def test_the_declared_default_space_is_used_when_none_is_given():
    for family in MODEL_FAMILIES:
        assert default_space(family).grid == DEFAULT_SEARCH_SPACE[family]


def test_an_unknown_family_cannot_be_searched():
    with pytest.raises(ValueError, match="unknown model family"):
        SearchSpace(family="transformer", grid={"depth": (1, 2)})


def test_an_empty_axis_is_rejected():
    with pytest.raises(ValueError, match="no values"):
        SearchSpace(family="ridge", grid={"alpha": ()})


# --- Every configuration is a trial ------------------------------------------


def test_the_log_gains_one_completed_trial_per_configuration(research_log, dataset):
    before = research_log.trials()
    result = run_search(
        research_log, dataset, spaces=[SMALL], splitter=SPLITTER, rule=GROSS
    )
    assert result.n_configurations == 3
    assert result.n_completed == 3
    assert result.trials_before == before
    assert result.trials_after == before + 3
    assert research_log.trials() == before + 3


def test_each_row_records_its_iteration_number(research_log, session, dataset):
    from quantlab.db.models import ResearchLogEntry

    run_search(research_log, dataset, spaces=[SMALL], splitter=SPLITTER, rule=GROSS)
    iterations = sorted(
        entry.parameters["search_iteration"]
        for entry in session.query(ResearchLogEntry).all()
    )
    assert iterations == [1, 2, 3]


def test_a_search_across_families_counts_all_of_them(research_log, dataset):
    spaces = [
        SearchSpace(family="ridge", grid={"alpha": (1.0, 100.0)}),
        SearchSpace(family="lasso", grid={"alpha": (1e-4, 1e-3)}),
    ]
    result = run_search(
        research_log, dataset, spaces=spaces, splitter=SPLITTER, rule=GROSS
    )
    assert result.n_configurations == 4
    assert research_log.trials() == 4
    assert {e.model.family for e in result.evaluations} == {"ridge", "lasso"}


def test_a_failing_configuration_is_recorded_and_the_search_continues(
    research_log, session, dataset, monkeypatch
):
    from quantlab.db.models import ResearchLogEntry
    from quantlab.research.log import STATUS_COMPLETE, STATUS_FAILED

    original = ModelSpec.build
    seen: list[float] = []

    def flaky(self):
        alpha = self.params.get("alpha")
        seen.append(alpha)
        if alpha == 10.0:
            raise RuntimeError("deliberate failure")
        return original(self)

    monkeypatch.setattr(ModelSpec, "build", flaky)
    result = run_search(
        research_log, dataset, spaces=[SMALL], splitter=SPLITTER, rule=GROSS
    )

    assert result.n_completed == 2
    assert len(result.failures) == 1
    assert "deliberate failure" in result.failures[0][1]

    entries = session.query(ResearchLogEntry).all()
    assert len(entries) == 3, "the failed attempt still leaves evidence"
    assert sum(e.status == STATUS_COMPLETE for e in entries) == 2
    assert sum(e.status == STATUS_FAILED for e in entries) == 1
    # A failed attempt is not a completed trial.
    assert result.trials_after == 2


def test_stop_on_error_aborts_instead(research_log, dataset, monkeypatch):
    def explode(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(ModelSpec, "build", explode)
    with pytest.raises(RuntimeError, match="boom"):
        run_search(
            research_log,
            dataset,
            spaces=[SMALL],
            splitter=SPLITTER,
            rule=GROSS,
            stop_on_error=True,
        )


# --- Selection and reporting -------------------------------------------------


def test_the_winner_is_the_highest_selection_metric(research_log, dataset):
    result = run_search(
        research_log, dataset, spaces=[SMALL], splitter=SPLITTER, rule=GROSS
    )
    assert result.best.selection_metric == max(
        e.selection_metric for e in result.evaluations
    )
    assert result.ranked()[0] is result.best


def test_the_report_leads_with_the_space_and_the_iteration_count(research_log, dataset):
    """PRD Module 11: the search space and iteration count go beside the results."""

    result = run_search(
        research_log, dataset, spaces=[SMALL], splitter=SPLITTER, rule=GROSS
    )
    text = result.render(research_log)

    space_position = text.index("Search space")
    iterations_position = text.index("Iterations run")
    winner_position = text.index("Winner")
    assert space_position < winner_position
    assert iterations_position < winner_position

    assert "3 configuration(s) total" in text
    assert "alpha: [0.1, 10.0, 1000.0]" in text
    assert "3 completed, 0 failed" in text
    assert "Trials in research_log: 0 before -> 3 after (+3)" in text
    assert "deflated by" in text


def test_the_report_carries_the_baseline_verdict_for_the_winner(research_log, dataset):
    result = run_search(
        research_log, dataset, spaces=[SMALL], splitter=SPLITTER, rule=GROSS
    )
    assert "VERDICT" in result.render(research_log)


def test_the_winner_is_deflated_by_the_whole_log_not_the_grid_size(
    research_log, dataset
):
    """Model search is one part of the project's selection process, not all of it."""

    from quantlab.research.log import RunResult, RunSpec

    # A strategy backtest logged before the search — a trial like any other.
    with research_log.run(
        RunSpec(
            strategy_name="momentum_12_1",
            universe="point-in-time",
            index_name="T",
            start=dataset.start,
            end=dataset.end,
            cost_model="ZeroCostModel",
        )
    ) as handle:
        handle.record(RunResult.from_returns([0.01, -0.005, 0.02, 0.0, 0.01]))

    result = run_search(
        research_log, dataset, spaces=[SMALL], splitter=SPLITTER, rule=GROSS
    )
    report = result.sharpe_report(research_log)
    assert result.n_configurations == 3
    assert report.trials == 4, "the pre-existing backtest counts too"


def test_a_search_where_nothing_completed_says_so(research_log, dataset, monkeypatch):
    def explode(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(ModelSpec, "build", explode)
    result = run_search(
        research_log, dataset, spaces=[SMALL], splitter=SPLITTER, rule=GROSS
    )
    assert result.n_completed == 0
    assert "No configuration completed" in result.render()
    with pytest.raises(ValueError, match="no best model"):
        _ = result.best
