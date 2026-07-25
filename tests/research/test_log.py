"""The research log: automatic, unskippable, and fatal when it fails."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from sqlalchemy.orm import Session

from quantlab.config import HOLDOUT_START, HoldoutViolationError, get_settings
from quantlab.db.models import ResearchLogEntry
from quantlab.research.log import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    ResearchLog,
    ResearchLogError,
    RunResult,
    RunSpec,
)

from .conftest import make_spec, zero_edge_returns


def test_run_writes_a_complete_row(research_log: ResearchLog) -> None:
    spec = make_spec("momentum_12_1", lookback=252, skip=21)
    with research_log.run(spec) as run:
        run.record(RunResult.from_returns([0.001, -0.002, 0.003, 0.0005]))

    entry = research_log.entry(run.run_id)
    assert entry.status == STATUS_COMPLETE
    assert entry.strategy_name == "momentum_12_1"
    assert entry.parameters == {"lookback": 252, "skip": 21}
    assert entry.universe == "point-in-time"
    assert entry.index_name == "NIFTY500"
    assert entry.cost_model == "IndianEquityCostModel"
    assert entry.sharpe is not None
    assert entry.n_periods == 4
    assert entry.returns_skew is not None
    assert entry.returns_kurtosis is not None
    assert entry.holdout_touched is False
    assert entry.exploratory is True


def test_split_is_derived_not_asserted(research_log: ResearchLog) -> None:
    # A window straddling the train/validation boundary must say so, so that a run
    # cannot be filed under the cleaner of the two splits.
    spec = make_spec(start=date(2021, 1, 1), end=date(2022, 6, 30))
    with research_log.run(spec) as run:
        run.record(RunResult.from_returns([0.01, -0.01, 0.02]))
    assert research_log.entry(run.run_id).split == "train+validation"


def test_row_is_committed_before_the_run_executes(research_log: ResearchLog) -> None:
    # A long or crashing run must be visible in the log while it is still running.
    with research_log.run(make_spec("slow")) as run:
        mid_flight = research_log.entry(run.run_id)
        assert mid_flight.status == "running"
        assert mid_flight.sharpe is None
        run.record(RunResult.from_returns([0.01, 0.02]))


def test_run_without_recording_raises(research_log: ResearchLog) -> None:
    # The central guarantee: producing results without logging them is an error,
    # not a silent omission that quietly shrinks the trials count.
    with (
        pytest.raises(ResearchLogError, match="without recording any metrics"),
        research_log.run(make_spec("forgot_to_log")),
    ):
        pass

    rows = research_log.entries(status=None)
    assert len(rows) == 1
    assert rows[0].status == STATUS_FAILED
    assert rows[0].error is not None and "without recording" in rows[0].error
    # A failed run contributes no trial and no Sharpe.
    assert research_log.trials() == 0


def test_engine_exception_marks_the_row_failed_and_propagates(
    research_log: ResearchLog,
) -> None:
    with (
        pytest.raises(ZeroDivisionError),
        research_log.run(make_spec("explodes")) as run,
    ):
        run.record(RunResult.from_returns([0.01, 0.02]))
        raise ZeroDivisionError("bad cost model")

    rows = research_log.entries(status=None)
    assert rows[0].status == STATUS_FAILED
    assert "ZeroDivisionError: bad cost model" in (rows[0].error or "")


def test_recording_twice_raises(research_log: ResearchLog) -> None:
    with (
        pytest.raises(ResearchLogError, match="already recorded"),
        research_log.run(make_spec()) as run,
    ):
        run.record(RunResult.from_returns([0.01, 0.02]))
        run.record(RunResult.from_returns([0.03, 0.04]))


def test_git_commit_is_recorded(research_log: ResearchLog) -> None:
    # This repository is a git work tree, so provenance must be captured.
    with research_log.run(make_spec()) as run:
        run.record(RunResult.from_returns([0.01, -0.01, 0.02]))
    entry = research_log.entry(run.run_id)
    assert entry.git_commit_hash is not None
    assert len(entry.git_commit_hash) == 40
    assert entry.git_dirty is not None


def test_holdout_run_is_refused_without_the_env_var(
    research_log: ResearchLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Independent of the DataHandler: a run whose window reaches the holdout is
    # refused at the log too, so an analytic backtest cannot slip past.
    get_settings.cache_clear()
    monkeypatch.delenv("QUANTLAB_UNLOCK_HOLDOUT", raising=False)
    spec = make_spec(start=date(2024, 1, 1), end=date(2024, 6, 30), holdout=True)
    with (
        pytest.raises(HoldoutViolationError, match="QUANTLAB_UNLOCK_HOLDOUT"),
        research_log.run(spec),
    ):
        pass
    assert research_log.entries(status=None) == []
    get_settings.cache_clear()


def test_holdout_run_records_the_flag_when_unlocked(
    research_log: ResearchLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("QUANTLAB_UNLOCK_HOLDOUT", "1")
    spec = make_spec(start=HOLDOUT_START, end=date(2024, 6, 30), holdout=True)
    with research_log.run(spec) as run:
        run.record(RunResult.from_returns([0.01, -0.01, 0.02]))
    entry = research_log.entry(run.run_id)
    assert entry.holdout_touched is True
    assert entry.split == "holdout"
    # The audit trail requirement: a holdout run carries its commit.
    assert entry.git_commit_hash is not None
    get_settings.cache_clear()


def test_unnamed_or_reversed_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="strategy_name is required"):
        RunSpec(
            strategy_name="",
            universe="point-in-time",
            index_name="NIFTY500",
            start=date(2015, 1, 1),
            end=date(2016, 1, 1),
            cost_model="ZeroCostModel",
        )
    with pytest.raises(ValueError, match="precedes start"):
        RunSpec(
            strategy_name="backwards",
            universe="point-in-time",
            index_name="NIFTY500",
            start=date(2016, 1, 1),
            end=date(2015, 1, 1),
            cost_model="ZeroCostModel",
        )


def test_from_returns_captures_non_excess_kurtosis() -> None:
    rng = np.random.default_rng(7)
    result = RunResult.from_returns(rng.normal(0.0, 0.01, 20_000))
    assert result.returns_kurtosis is not None
    # Non-excess: a normal sample sits near 3.0, not near 0.0.
    assert result.returns_kurtosis == pytest.approx(3.0, abs=0.15)
    assert result.returns_skew == pytest.approx(0.0, abs=0.1)


def test_from_returns_refuses_a_single_observation() -> None:
    with pytest.raises(ValueError, match="at least 2 finite return observations"):
        RunResult.from_returns([0.01])


def test_metrics_survive_numpy_and_non_finite_values(research_log: ResearchLog) -> None:
    with research_log.run(make_spec()) as run:
        run.record(
            RunResult.from_returns(
                [0.01, -0.02, 0.015],
                metrics={
                    "max_drawdown": np.float64(-0.21),
                    "turnover": np.int64(42),
                    "calmar": float("nan"),
                    "started": date(2015, 1, 1),
                },
            )
        )
    metrics = research_log.entry(run.run_id).metrics
    assert metrics["max_drawdown"] == pytest.approx(-0.21)
    assert metrics["turnover"] == 42
    assert metrics["calmar"] is None
    assert metrics["started"] == "2015-01-01"


def test_trials_and_variance_come_from_the_table(research_log: ResearchLog) -> None:
    rng = np.random.default_rng(3)
    for i in range(5):
        with research_log.run(make_spec(f"strategy_{i % 2}")) as run:
            run.record(RunResult.from_returns(zero_edge_returns(rng, n_periods=300)))

    assert research_log.trials() == 5
    assert research_log.trials(strategy_name="strategy_0") == 3
    assert research_log.trials(strategy_name="strategy_1") == 2
    assert research_log.sharpe_variance() > 0.0
    assert len(research_log.trial_sharpes()) == 5

    best = research_log.best_run()
    assert best is not None
    assert best.sharpe == max(research_log.trial_sharpes())


def test_summary_counts_every_category(research_log: ResearchLog) -> None:
    rng = np.random.default_rng(11)
    for i in range(3):
        with research_log.run(make_spec(f"s{i}")) as run:
            run.record(RunResult.from_returns(zero_edge_returns(rng, n_periods=300)))
    with pytest.raises(ResearchLogError), research_log.run(make_spec("unlogged")):
        pass

    summary = research_log.summary()
    assert summary.total_runs == 4
    assert summary.complete_runs == 3
    assert summary.failed_runs == 1
    assert summary.strategies == 3
    assert summary.exploratory_runs == 3
    assert summary.preregistered_runs == 0
    assert summary.holdout_runs == 0
    rendered = summary.render()
    assert "Trials (complete runs):   3" in rendered
    assert "deflate" in rendered  # never shows a best Sharpe without pointing at it


def test_two_hundred_zero_edge_strategies_are_all_logged(
    research_log: ResearchLog, session: Session
) -> None:
    """200 edgeless strategies through the log: every one counts as a trial."""

    rng = np.random.default_rng(20240101)
    for i in range(200):
        with research_log.run(make_spec(f"noise_{i:03d}", seed=i)) as run:
            run.record(RunResult.from_returns(zero_edge_returns(rng, n_periods=1260)))

    assert session.query(ResearchLogEntry).count() == 200
    assert research_log.trials() == 200
    sharpes = research_log.trial_sharpes()
    assert len(sharpes) == 200
    # Sampling noise alone produces a spread of Sharpes, and a best one that looks
    # like a real strategy. This is the phenomenon the deflation exists to correct.
    assert max(sharpes) > 0.8
    assert abs(float(np.mean(sharpes))) < 0.2
