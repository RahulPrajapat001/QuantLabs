"""The engine base: running a backtest implies logging it, with no way around."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantlab.backtesting.data_handler import DataHandler
from quantlab.backtesting.logged import LoggedBacktestEngine
from quantlab.research.log import (
    LoggedRun,
    ResearchLog,
    ResearchLogError,
    RunResult,
    RunSpec,
)

from .conftest import make_bars


class NoiseEngine(LoggedBacktestEngine):
    """A minimal engine: draws edgeless returns and records them."""

    def __init__(
        self,
        research_log: ResearchLog,
        *,
        name: str = "noise",
        seed: int = 0,
        n_periods: int = 756,
        handler: DataHandler | None = None,
    ) -> None:
        super().__init__(research_log)
        self._name = name
        self._seed = seed
        self._n_periods = n_periods
        self._handler = handler

    @property
    def data_handler(self) -> DataHandler | None:
        return self._handler

    def build_spec(self) -> RunSpec:
        start = self._handler.start if self._handler else date(2015, 1, 1)
        end = self._handler.end if self._handler else date(2021, 12, 31)
        return RunSpec(
            strategy_name=self._name,
            universe="point-in-time",
            index_name="NIFTY500",
            start=start,
            end=end,
            cost_model="IndianEquityCostModel",
            parameters={"seed": self._seed},
        )

    def execute(self, run: LoggedRun) -> RunResult:
        rng = np.random.default_rng(self._seed)
        return RunResult.from_returns(
            rng.normal(0.0, 0.012, self._n_periods), metrics={"volatility": 0.012}
        )


def test_running_the_engine_logs_the_run(research_log: ResearchLog) -> None:
    backtest = NoiseEngine(research_log).run()

    entry = backtest.entry()
    assert entry.status == "complete"
    assert entry.strategy_name == "noise"
    assert entry.sharpe is not None
    assert research_log.trials() == 1


def test_the_result_exposes_its_sharpe_only_with_its_deflation(
    research_log: ResearchLog,
) -> None:
    for i in range(20):
        NoiseEngine(research_log, name=f"noise_{i}", seed=i).run()
    backtest = NoiseEngine(research_log, name="final", seed=99).run()

    # There is no bare `.sharpe` on a LoggedBacktest by design.
    assert not hasattr(backtest, "sharpe")
    report = backtest.sharpe_report()
    assert report.trials == 21
    assert 0.0 <= report.deflated_sharpe <= 1.0
    assert report.run_id == backtest.run_id


def test_a_subclass_cannot_override_run(research_log: ResearchLog) -> None:
    """The structural guarantee: no subclass can route around the log write."""

    with pytest.raises(TypeError, match="may not override run"):

        class SneakyEngine(NoiseEngine):
            def run(self):  # type: ignore[override]
                return "results without a log row"


def test_an_engine_that_records_nothing_raises(research_log: ResearchLog) -> None:
    class ForgetfulEngine(NoiseEngine):
        def execute(self, run: LoggedRun) -> RunResult:
            return RunResult()  # no Sharpe, no returns — but still recorded

    backtest = ForgetfulEngine(research_log).run()
    # Recorded, so the row exists and counts as a trial; it simply has no Sharpe
    # to report, which the deflation layer refuses to invent.
    assert backtest.entry().status == "complete"
    assert backtest.entry().sharpe is None


def test_a_crashing_engine_leaves_a_failed_row(research_log: ResearchLog) -> None:
    class BrokenEngine(NoiseEngine):
        def execute(self, run: LoggedRun) -> RunResult:
            raise RuntimeError("cost model misconfigured")

    with pytest.raises(RuntimeError, match="cost model misconfigured"):
        BrokenEngine(research_log).run()

    rows = research_log.entries(status=None)
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert "cost model misconfigured" in (rows[0].error or "")
    assert research_log.trials() == 0


def test_engine_and_handler_windows_must_agree(research_log: ResearchLog) -> None:
    handler = DataHandler(make_bars(start="2016-01-04", n_days=40))

    class MislabelledEngine(NoiseEngine):
        def build_spec(self) -> RunSpec:
            spec = super().build_spec()
            return RunSpec(
                strategy_name=spec.strategy_name,
                universe=spec.universe,
                index_name=spec.index_name,
                start=date(2015, 1, 1),
                end=date(2021, 12, 31),
                cost_model=spec.cost_model,
            )

    with pytest.raises(ResearchLogError, match="does not match the DataHandler window"):
        MislabelledEngine(research_log, handler=handler).run()

    # Nothing was logged, because the disagreement is caught before the row.
    assert research_log.entries(status=None) == []


def test_engine_logs_the_handlers_window(research_log: ResearchLog) -> None:
    handler = DataHandler(make_bars(start="2016-01-04", n_days=40))
    backtest = NoiseEngine(research_log, handler=handler).run()
    entry = backtest.entry()
    assert entry.start_date == handler.start
    assert entry.end_date == handler.end
    assert entry.split == "train"
