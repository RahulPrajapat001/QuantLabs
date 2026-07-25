"""The reporting rule: a Sharpe without its trials count and deflation raises."""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.reports.generator import render_metrics_table, render_run_report
from quantlab.research.deflated import probability_of_backtest_overfitting
from quantlab.research.log import ResearchLog, RunResult
from quantlab.research.reporting import (
    ReportingRuleError,
    SharpeReport,
    assert_no_bare_sharpe,
    find_bare_sharpe_keys,
)

from .conftest import make_spec, zero_edge_returns


@pytest.fixture
def logged_run(research_log: ResearchLog) -> str:
    rng = np.random.default_rng(5)
    for i in range(30):
        with research_log.run(make_spec(f"variant_{i}", variant=i)) as run:
            run.record(
                RunResult.from_returns(
                    zero_edge_returns(rng, n_periods=756),
                    metrics={
                        "max_drawdown": -0.18,
                        "turnover": 2.4,
                        "volatility": 0.012,
                    },
                )
            )
    best = research_log.best_run()
    assert best is not None
    return best.run_id


def test_report_generator_refuses_a_bare_sharpe() -> None:
    metrics = {"sharpe": 1.42, "max_drawdown": -0.2}
    with pytest.raises(ReportingRuleError, match="without a SharpeReport"):
        render_metrics_table(metrics)


def test_the_rule_catches_sharpe_hidden_among_many_metrics() -> None:
    metrics = {
        "total_return": 0.42,
        "annual_return": 0.13,
        "max_drawdown": -0.2,
        "net_sharpe_ratio": 1.1,
        "turnover": 3.0,
    }
    assert find_bare_sharpe_keys(metrics) == ["net_sharpe_ratio"]
    with pytest.raises(ReportingRuleError, match="net_sharpe_ratio"):
        assert_no_bare_sharpe(metrics, None)


def test_non_sharpe_metrics_render_freely() -> None:
    table = render_metrics_table({"total_return": 0.42, "max_drawdown": -0.2})
    assert "total_return" in table
    assert "sharpe" not in table


def test_sharpe_report_cannot_be_built_without_its_context() -> None:
    # trials and deflated_sharpe are required positionally — there is no way to
    # construct this type with a Sharpe alone.
    with pytest.raises(TypeError):
        SharpeReport(value=1.4)  # type: ignore[call-arg]
    with pytest.raises(ReportingRuleError, match="trials must be at least 1"):
        SharpeReport(
            value=1.4,
            trials=0,
            deflated_sharpe=0.5,
            expected_max_sharpe=1.0,
            split="train",
            cost_model="ZeroCostModel",
            run_id="abc",
            n_periods=100,
        )
    with pytest.raises(ReportingRuleError, match="probability"):
        SharpeReport(
            value=1.4,
            trials=10,
            deflated_sharpe=1.4,
            expected_max_sharpe=1.0,
            split="train",
            cost_model="ZeroCostModel",
            run_id="abc",
            n_periods=100,
        )
    with pytest.raises(ReportingRuleError, match="cost_model is required"):
        SharpeReport(
            value=1.4,
            trials=10,
            deflated_sharpe=0.6,
            expected_max_sharpe=1.0,
            split="train",
            cost_model="",
            run_id="abc",
            n_periods=100,
        )


def test_report_from_log_carries_the_trials_count(
    research_log: ResearchLog, logged_run: str
) -> None:
    report = SharpeReport.from_log(research_log, logged_run)
    assert report.trials == 30
    assert report.run_id == logged_run
    assert report.split == "train"
    assert report.cost_model == "IndianEquityCostModel"
    assert report.provenance == "exploratory"

    rendered = report.render()
    assert "30 trials" in rendered
    assert "deflated" in rendered
    # A best-of-30 noise Sharpe must be labelled, not merely reported.
    assert not report.significant
    assert "not significant after deflation" in rendered

    payload = report.as_dict()
    assert set(payload) >= {"value", "trials", "deflated_sharpe", "split", "pbo"}


def test_mismatched_report_and_metric_raises(
    research_log: ResearchLog, logged_run: str
) -> None:
    report = SharpeReport.from_log(research_log, logged_run)
    with pytest.raises(ReportingRuleError, match="does not match the accompanying"):
        assert_no_bare_sharpe({"sharpe": report.value + 0.5}, report)


def test_metrics_table_renders_the_sharpe_from_the_report(
    research_log: ResearchLog, logged_run: str
) -> None:
    report = SharpeReport.from_log(research_log, logged_run)
    entry = research_log.entry(logged_run)
    table = render_metrics_table(entry.metrics, report)
    assert "trials" in table
    assert "deflated" in table
    # The raw number never appears on its own row.
    assert f"| sharpe | {report.value:.4f} |" not in table


def test_run_report_is_traceable_and_self_labelling(
    research_log: ResearchLog, logged_run: str
) -> None:
    rng = np.random.default_rng(6)
    pbo = probability_of_backtest_overfitting(
        rng.normal(0.0, 0.01, size=(756, 10)), n_splits=8
    )
    report = render_run_report(research_log, logged_run, pbo=pbo)

    assert logged_run in report
    assert "trials" in report.lower()
    assert "Deflated Sharpe" in report
    assert "Exploratory" in report
    assert "probability of backtest overfitting" in report.lower()
    assert "not significant after deflation" in report
    # Provenance: the commit is on the page, so the run is reproducible.
    assert "**commit**" in report
