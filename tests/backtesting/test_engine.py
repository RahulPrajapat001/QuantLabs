"""The engine — priority tests 1 (benchmark identity) and 2 (look-ahead raises)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantlab.backtesting.costs import (
    IndianEquityCostModel,
    ZeroCostModel,
    ZeroSlippage,
)
from quantlab.backtesting.data_handler import BarBatch, DataHandler, LookAheadError
from quantlab.backtesting.engine import BacktestConfig, BacktestEngine, DelistingEvent
from quantlab.backtesting.events import Signal
from quantlab.backtesting.portfolio import Portfolio, RiskLimits, SizingMethod
from quantlab.research.log import ResearchLog
from quantlab.strategies.base import Strategy
from quantlab.strategies.systematic import BuyAndHoldStrategy
from quantlab.strategies.technical import MovingAverageCrossStrategy

from .conftest import make_bars, make_flat_bars, make_trending_bars


def _fully_invested_portfolio(*, fractional: bool = True) -> Portfolio:
    """One position, 100% of capital, no reserve — the validation configuration."""

    return Portfolio(
        initial_capital=1_000_000.0,
        fractional_shares=fractional,
        sizing=SizingMethod.EQUAL_WEIGHT,
        limits=RiskLimits(
            max_position_weight=1.0,
            max_positions=1,
            cash_reserve=0.0,
            min_order_value=0.0,
        ),
        target_positions=1,
    )


def _frictionless_config() -> BacktestConfig:
    return BacktestConfig(
        cost_model=ZeroCostModel(),
        slippage=ZeroSlippage(),
        max_participation=None,
        initial_capital=1_000_000.0,
    )


# --- PRIORITY TEST 1 ---------------------------------------------------------


def test_buy_and_hold_under_zero_costs_reproduces_the_benchmark(
    research_log: ResearchLog,
) -> None:
    """Buy-and-hold must equal the underlying's return to floating-point tolerance.

    This is the engine's own validation instrument. If the accounting is sound, a
    strategy that buys once and never trades again earns exactly what the security
    earned from the price it was bought at. Any discrepancy is the engine inventing
    or destroying money — leaked cash, double-counted costs, a marking error — and
    every other number it reports would be equally wrong.

    The fill is at the *second* bar's open, because the signal is raised on the first
    bar's close and execution is T+1. So the benchmark is the return from that open
    to the final close.
    """

    bars = make_bars(n_days=250, securities=1, seed=11)
    engine = BacktestEngine(
        research_log,
        strategy=BuyAndHoldStrategy(),
        data_handler=DataHandler(bars),
        portfolio=_fully_invested_portfolio(),
        config=_frictionless_config(),
    )
    engine.run()

    series = bars.reset_index(drop=True)
    fill_price = float(series.loc[1, "open"])
    final_close = float(series.loc[len(series) - 1, "close"])
    expected = final_close / fill_price - 1.0

    assert engine.metrics is not None
    assert engine.metrics.total_return == pytest.approx(expected, abs=1e-12)
    # Nothing left on the table: the notional order deployed every rupee.
    assert engine.portfolio.cash == pytest.approx(0.0, abs=1e-9)
    assert engine.metrics.total_costs == 0.0
    assert engine.metrics.trades == 0  # never sold, so no closed round trip


def test_buy_and_hold_matches_a_supplied_benchmark_series(
    research_log: ResearchLog,
) -> None:
    """The same identity, checked through the benchmark-comparison path."""

    bars = make_bars(n_days=200, securities=1, seed=5)
    prices = bars.set_index("date")["close"]
    # Benchmark from the fill date onward, which is what the strategy actually held.
    benchmark = prices.iloc[1:]

    engine = BacktestEngine(
        research_log,
        strategy=BuyAndHoldStrategy(),
        data_handler=DataHandler(bars),
        portfolio=_fully_invested_portfolio(),
        config=_frictionless_config(),
        benchmark=benchmark,
    )
    engine.run()
    assert engine.metrics is not None
    # Equity is marked from bar 1 while the position is only held from bar 2, so the
    # two series differ by one flat day; the annualised figures agree closely.
    assert engine.metrics.excess_annual_return == pytest.approx(0.0, abs=0.005)


def test_whole_share_rounding_leaves_a_documented_residue(
    research_log: ResearchLog,
) -> None:
    """With whole shares the identity is approximate, and that is expected.

    A residue exists because you cannot buy 1,847.3 shares. The test pins the
    behaviour so the exactness of the fractional case is not mistaken for a general
    guarantee.
    """

    bars = make_bars(n_days=120, securities=1, seed=7)
    engine = BacktestEngine(
        research_log,
        strategy=BuyAndHoldStrategy(),
        data_handler=DataHandler(bars),
        portfolio=_fully_invested_portfolio(fractional=False),
        config=_frictionless_config(),
    )
    engine.run()
    series = bars.reset_index(drop=True)
    expected = (
        float(series.loc[len(series) - 1, "close"]) / float(series.loc[1, "open"]) - 1.0
    )
    assert engine.metrics is not None
    # Close, but not exact — and the residue is small and positive cash.
    assert engine.metrics.total_return == pytest.approx(expected, abs=1e-3)
    assert 0.0 <= engine.portfolio.cash < float(series.loc[1, "open"])


# --- PRIORITY TEST 2 ---------------------------------------------------------


class _PeekingStrategy(Strategy):
    """A strategy that tries to read a bar it has not been shown."""

    name = "peeking"

    def __init__(self, target: date) -> None:
        self.target = target

    def on_bar(self, batch: BarBatch, handler: DataHandler) -> list[Signal]:
        # Asking for a future date must raise, not return the future quietly.
        handler.history_as_of(self.target)
        return []


class _FrameGrabbingStrategy(Strategy):
    """A strategy that tries to reach the full panel through a private attribute."""

    name = "frame_grabber"

    def on_bar(self, batch: BarBatch, handler: DataHandler) -> list[Signal]:
        full = handler.history()  # cursor-bounded by construction
        assert full["date"].max().date() <= batch.date
        # There is no public accessor that returns anything later.
        assert not hasattr(handler, "all_bars")
        assert not hasattr(handler, "future_bars")
        return []


def test_a_strategy_reaching_past_the_current_bar_raises(
    research_log: ResearchLog,
) -> None:
    """Look-ahead is an error, not a silent success.

    The run fails loudly and the research log records a failed row — an attempt to
    see the future does not produce a result at all, let alone a good one.
    """

    bars = make_bars(n_days=30, securities=1)
    final_date = pd.Timestamp(bars["date"].max()).date()
    engine = BacktestEngine(
        research_log,
        strategy=_PeekingStrategy(final_date),
        data_handler=DataHandler(bars),
        portfolio=_fully_invested_portfolio(),
        config=_frictionless_config(),
    )

    with pytest.raises(LookAheadError, match="look-ahead bug"):
        engine.run()

    rows = research_log.entries(status=None)
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert "LookAheadError" in (rows[0].error or "")
    # A failed run is not a trial: it produced no result to count.
    assert research_log.trials() == 0


def test_history_handed_to_a_strategy_never_includes_the_future(
    research_log: ResearchLog,
) -> None:
    """The positive form: every bar a strategy can see is at or before today."""

    engine = BacktestEngine(
        research_log,
        strategy=_FrameGrabbingStrategy(),
        data_handler=DataHandler(make_bars(n_days=40, securities=3)),
        portfolio=_fully_invested_portfolio(),
        config=_frictionless_config(),
    )
    engine.run()  # the assertions live inside the strategy
    assert engine.metrics is not None


def test_orders_fill_at_the_next_open_not_todays_close(
    research_log: ResearchLog,
) -> None:
    """T+1 execution, verified against the actual bar prices."""

    bars = make_flat_bars(n_days=5, price=100.0)
    bars.loc[bars.index[1], "open"] = (
        105.0  # tomorrow's open differs from today's close
    )
    engine = BacktestEngine(
        research_log,
        strategy=BuyAndHoldStrategy(),
        data_handler=DataHandler(bars),
        portfolio=_fully_invested_portfolio(),
        config=_frictionless_config(),
    )
    engine.run()

    fills = engine.ledger.fills
    assert len(fills) == 1
    assert fills[0].price == 105.0, "filled at tomorrow's open, not today's ₹100 close"
    assert fills[0].order_date == pd.Timestamp(bars["date"].iloc[0]).date()
    assert fills[0].date == pd.Timestamp(bars["date"].iloc[1]).date()


# --- Logging is automatic ----------------------------------------------------


def test_every_run_writes_to_the_research_log(research_log: ResearchLog) -> None:
    """Not a decorator, not optional — the engine cannot return without a row."""

    engine = BacktestEngine(
        research_log,
        strategy=MovingAverageCrossStrategy(fast=5, slow=15),
        data_handler=DataHandler(make_bars(n_days=120, securities=4)),
        portfolio=Portfolio(initial_capital=1_000_000.0),
        config=BacktestConfig(cost_model=IndianEquityCostModel()),
    )
    backtest = engine.run()

    entry = backtest.entry()
    assert entry.status == "complete"
    assert entry.strategy_name == "ma_cross"
    assert entry.cost_model == "IndianEquityCostModel"
    assert entry.split == "train"
    assert entry.parameters["fast"] == 5
    assert entry.parameters["slow"] == 15
    assert entry.parameters["cost_model"] == "IndianEquityCostModel"
    assert entry.sharpe is not None
    assert entry.n_periods == len(engine.ledger.returns)
    # And the Sharpe is only reachable with its deflation attached.
    report = backtest.sharpe_report()
    assert report.trials == 1
    assert report.cost_model == "IndianEquityCostModel"


def test_the_engine_cannot_override_run() -> None:
    """Structural: a subclass that tries to bypass the log fails at import time."""

    with pytest.raises(TypeError, match="may not override run"):

        class SneakyEngine(BacktestEngine):
            def run(self):  # type: ignore[override]
                return "unlogged results"


def test_metrics_are_recorded_in_the_log(research_log: ResearchLog) -> None:
    engine = BacktestEngine(
        research_log,
        strategy=MovingAverageCrossStrategy(fast=5, slow=15),
        data_handler=DataHandler(make_bars(n_days=150, securities=5)),
        portfolio=Portfolio(initial_capital=1_000_000.0),
        config=BacktestConfig(),
    )
    entry = engine.run().entry()
    for key in (
        "total_return",
        "annual_return",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "turnover",
        "total_costs",
        "cost_stt",
        "rejected_circuit_limit",
    ):
        assert key in entry.metrics, f"{key} missing from the logged metrics"


# --- Delisting ---------------------------------------------------------------


def test_a_delisted_position_is_liquidated_at_its_last_price(
    research_log: ResearchLog,
) -> None:
    """Per PRD Module 2: liquidate at the last real price, never forward-filled."""

    bars = make_trending_bars(n_days=40, daily_return=0.002)
    dates = sorted({pd.Timestamp(d).date() for d in bars["date"]})
    delisting_date = dates[20]
    last_price = float(
        bars.loc[bars["date"] == pd.Timestamp(delisting_date), "close"].iloc[0]
    )

    engine = BacktestEngine(
        research_log,
        strategy=BuyAndHoldStrategy(),
        data_handler=DataHandler(bars),
        portfolio=_fully_invested_portfolio(),
        config=_frictionless_config(),
        delistings=[
            DelistingEvent(
                security_id="SEC-0000",
                effective_date=delisting_date,
                price=last_price,
                reason="merger",
            )
        ],
    )
    engine.run()

    assert len(engine.ledger.liquidations) == 1
    when, security_id, price, _realised = engine.ledger.liquidations[0]
    assert when == delisting_date
    assert security_id == "SEC-0000"
    assert price == last_price
    # The position is gone and the proceeds are in cash by the end.
    assert not engine.portfolio.open_positions()
    assert engine.portfolio.cash > 0
    # A forced corporate action is not counted as a strategy trade.
    assert engine.metrics is not None
    assert engine.metrics.trades == 0


def test_a_bankruptcy_liquidation_can_wipe_the_position(
    research_log: ResearchLog,
) -> None:
    """With a zero recovery rate the proceeds are zero — a total loss, as specified."""

    bars = make_flat_bars(n_days=20, price=100.0)
    delisting_date = pd.Timestamp(bars["date"].iloc[10]).date()
    engine = BacktestEngine(
        research_log,
        strategy=BuyAndHoldStrategy(),
        data_handler=DataHandler(bars),
        portfolio=_fully_invested_portfolio(),
        config=_frictionless_config(),
        delistings=[
            DelistingEvent(
                security_id="SEC-0000",
                effective_date=delisting_date,
                price=0.0,  # last_price × recovery_rate, resolved upstream
                reason="insolvency (NCLT)",
            )
        ],
    )
    engine.run()
    assert engine.metrics is not None
    assert engine.metrics.total_return == pytest.approx(-1.0, abs=1e-9)


# --- Housekeeping ------------------------------------------------------------


def test_a_run_produces_one_mark_per_trading_day(research_log: ResearchLog) -> None:
    bars = make_bars(n_days=90, securities=2)
    handler = DataHandler(bars)
    engine = BacktestEngine(
        research_log,
        strategy=MovingAverageCrossStrategy(fast=5, slow=10),
        data_handler=handler,
        portfolio=Portfolio(initial_capital=1_000_000.0),
        config=BacktestConfig(),
    )
    engine.run()
    assert len(engine.ledger.marks) == len(handler.dates)
    assert engine.ledger.equity_curve.iloc[0] == pytest.approx(1_000_000.0)


def test_running_twice_gives_identical_results(research_log: ResearchLog) -> None:
    """State is reset between runs, so a second run is not contaminated."""

    def build() -> BacktestEngine:
        return BacktestEngine(
            research_log,
            strategy=MovingAverageCrossStrategy(fast=5, slow=15),
            data_handler=DataHandler(make_bars(n_days=120, securities=3, seed=2)),
            portfolio=Portfolio(initial_capital=1_000_000.0),
            config=BacktestConfig(),
        )

    engine = build()
    first = engine.run().result.metrics
    second = engine.run().result.metrics  # same instance, run again
    assert first["total_return"] == pytest.approx(second["total_return"])
    assert first["total_costs"] == pytest.approx(second["total_costs"])
    assert research_log.trials() == 2  # both runs counted


def test_the_engine_refuses_a_window_that_disagrees_with_its_data(
    research_log: ResearchLog,
) -> None:
    """The logged window must be the window that was actually streamed."""

    handler = DataHandler(make_bars(n_days=40, securities=1))

    class MislabelledEngine(BacktestEngine):
        def build_spec(self):  # type: ignore[override]
            spec = super().build_spec()
            from quantlab.research.log import RunSpec

            return RunSpec(
                strategy_name=spec.strategy_name,
                universe=spec.universe,
                index_name=spec.index_name,
                start=date(2015, 1, 1),
                end=date(2015, 6, 30),
                cost_model=spec.cost_model,
            )

    engine = MislabelledEngine(
        research_log,
        strategy=BuyAndHoldStrategy(),
        data_handler=handler,
        portfolio=_fully_invested_portfolio(),
        config=_frictionless_config(),
    )
    from quantlab.research.log import ResearchLogError

    with pytest.raises(ResearchLogError, match="does not match the DataHandler"):
        engine.run()
