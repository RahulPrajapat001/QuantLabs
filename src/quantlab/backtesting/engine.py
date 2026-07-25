"""``BacktestEngine`` — the event loop, and the only place the pieces meet.

The loop, per bar, in this exact order:

1. **Settle.** Sale proceeds whose T+1 date has arrived become spendable.
2. **Execute yesterday's orders** at *today's* open. This is what makes execution
   T+1: an order raised on the close of *t-1* transacts at the open of *t*, the
   earliest price the decision could have reached.
3. **Liquidate delistings** effective today, at the last real price (never a
   forward-filled one), per PRD Module 2.
4. **Strategy sees today's bar** and history up to and including it, and emits
   signals. The handler makes anything later unreachable.
5. **Portfolio sizes** those signals into orders against today's close.
6. **Queue the orders** for tomorrow's open. They are not filled today.
7. **Mark to market** at today's close and write the daily mark.

Steps 4 and 2 being on opposite sides of the bar is the whole design. Every
vectorised backtest that computes a signal and a return from the same row has
already lost this distinction, and no amount of care elsewhere recovers it.

The engine subclasses :class:`~quantlab.backtesting.logged.LoggedBacktestEngine`,
whose ``run()`` is final: **the research-log write is not something this class can
skip, forget, or refactor away.** ``execute()`` below is what the log calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

from quantlab.backtesting.costs import (
    CostModel,
    IndianEquityCostModel,
    Segment,
    SlippageModel,
    ZeroSlippage,
)
from quantlab.backtesting.data_handler import DataHandler
from quantlab.backtesting.events import Order
from quantlab.backtesting.execution import (
    CircuitLimitPolicy,
    ExcessPolicy,
    ExecutionHandler,
)
from quantlab.backtesting.ledger import DailyMark, Ledger
from quantlab.backtesting.logged import LoggedBacktestEngine
from quantlab.backtesting.metrics import BacktestMetrics, compute_metrics
from quantlab.backtesting.portfolio import Portfolio
from quantlab.research.log import LoggedRun, ResearchLog, RunResult, RunSpec

if TYPE_CHECKING:
    # Type-only: ``quantlab.strategies.base`` imports the DataHandler from this
    # package, so importing Strategy at runtime here would make the two packages
    # circular — and which one broke would depend on which the caller imported
    # first, which is exactly the kind of failure a test suite misses.
    from quantlab.strategies.base import Strategy

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DelistingEvent:
    """A security leaving the exchange mid-backtest.

    ``price`` is the last *actual* traded price, resolved by
    :func:`quantlab.universe.delisting.liquidate_position` — never forward-filled,
    and already haircut by the recovery rate if the exit was a bankruptcy.
    """

    security_id: str
    effective_date: date
    price: float
    reason: str | None = None


@dataclass
class BacktestConfig:
    """Everything about a run that is not the data or the strategy."""

    initial_capital: float = 1_000_000.0
    cost_model: CostModel = field(default_factory=IndianEquityCostModel)
    slippage: SlippageModel = field(default_factory=ZeroSlippage)
    max_participation: float | None = 0.01
    excess_policy: ExcessPolicy = ExcessPolicy.REJECT
    circuit_policy: CircuitLimitPolicy = field(default_factory=CircuitLimitPolicy)
    segment: Segment = Segment.DELIVERY
    settlement_days: int = 1
    risk_free_rate: float = 0.0
    #: Universe label recorded in the research log, e.g. ``"point-in-time"``.
    universe: str = "point-in-time"
    index_name: str = "NIFTY500"
    #: Pre-registered hypothesis this run tests. ``None`` logs it as exploratory.
    hypothesis_id: str | None = None
    notes: str | None = None

    def parameters(self) -> dict[str, Any]:
        """Config as loggable parameters."""

        return {
            "initial_capital": self.initial_capital,
            "cost_model": self.cost_model.name,
            "slippage": self.slippage.name,
            "max_participation": self.max_participation,
            "excess_policy": self.excess_policy.value,
            "circuit_band": self.circuit_policy.band,
            "segment": self.segment.value,
            "settlement_days": self.settlement_days,
        }


class BacktestEngine(LoggedBacktestEngine):
    """Event-driven backtester. One instance per run.

    Every run writes to ``research_log`` automatically, because ``run()`` is
    inherited and final.
    """

    def __init__(
        self,
        research_log: ResearchLog,
        *,
        strategy: Strategy,
        data_handler: DataHandler,
        portfolio: Portfolio,
        config: BacktestConfig | None = None,
        benchmark: pd.Series | None = None,
        delistings: list[DelistingEvent] | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        super().__init__(research_log)
        self.strategy = strategy
        self._handler = data_handler
        self.portfolio = portfolio
        self.config = config or BacktestConfig()
        self.benchmark = benchmark
        self.ledger = ledger or Ledger()
        self.execution = ExecutionHandler(
            cost_model=self.config.cost_model,
            slippage=self.config.slippage,
            max_participation=self.config.max_participation,
            excess_policy=self.config.excess_policy,
            circuit_policy=self.config.circuit_policy,
            segment=self.config.segment,
        )
        self._delistings: dict[date, list[DelistingEvent]] = {}
        for event in delistings or []:
            self._delistings.setdefault(event.effective_date, []).append(event)
        self.metrics: BacktestMetrics | None = None

        # Keep the portfolio's settlement consistent with the config; two
        # different T+ conventions in one run would be silently wrong.
        self.portfolio.settlement_days = self.config.settlement_days

    @property
    def data_handler(self) -> DataHandler:
        """Exposed so the base class can cross-check the logged window."""

        return self._handler

    def build_spec(self) -> RunSpec:
        """Describe the run for the research log, before it executes."""

        parameters = {**self.config.parameters(), **self.strategy.parameters()}
        return RunSpec(
            strategy_name=self.strategy.name,
            universe=self.config.universe,
            index_name=self.config.index_name,
            start=self._handler.start,
            end=self._handler.end,
            cost_model=self.config.cost_model.name,
            parameters=parameters,
            hypothesis_id=self.config.hypothesis_id,
            holdout=self._handler.holdout,
            notes=self.config.notes,
        )

    def execute(self, run: LoggedRun) -> RunResult:
        """Run the event loop and return the metrics. Called inside the log."""

        self._reset()
        pending: list[Order] = []

        for batch in self._handler.stream():
            today = batch.date

            # 1. Settle: yesterday's sale proceeds become spendable.
            self.portfolio.settle(today)

            # 2. Execute yesterday's orders at today's open. This is the T+1 rule.
            if pending:
                fills, rejections = self.execution.execute(
                    pending,
                    self._handler,
                    available_cash=self.portfolio.settled_cash,
                    positions=self.portfolio.open_positions(),
                )
                for fill in fills:
                    self.portfolio.apply_fill(fill)
                    self.ledger.record_fill(fill)
                for rejection in rejections:
                    self.ledger.record_rejection(rejection)
                pending = []

            # 3. Delistings: liquidate at the last real price.
            for event in self._delistings.get(today, []):
                realised = self.portfolio.liquidate(
                    event.security_id, event.price, today
                )
                if realised or event.security_id in self.portfolio.positions:
                    self.ledger.record_liquidation(
                        today, event.security_id, event.price, realised
                    )

            # 4. Strategy: today's bar plus history through today, nothing later.
            signals = []
            if len(self.ledger.marks) >= self.strategy.warmup:
                signals = self.strategy.on_bar(batch, self._handler)
                self.ledger.record_signals(signals)

            # 5 & 6. Size into orders against today's close; queue for tomorrow.
            closes = self._closes(batch)
            if signals:
                pending = self.portfolio.generate_orders(signals, closes, on=today)

            # 7. Mark to market at today's close.
            self.ledger.record_mark(self._mark(today, closes))

        if pending:
            log.debug(
                "engine.trailing_orders_expired",
                count=len(pending),
                detail="orders raised on the final bar have no next open to fill at",
            )

        self.metrics = compute_metrics(
            self.ledger,
            initial_capital=self.config.initial_capital,
            benchmark=self.benchmark,
            risk_free_rate=self.config.risk_free_rate,
        )
        returns = self.ledger.returns
        if len(returns) < 2:
            # A run with no return series cannot produce a Sharpe, and inventing
            # one would poison the trials statistics. Log the metrics without it.
            return RunResult(metrics=self.metrics.as_dict())

        return RunResult(
            metrics=self.metrics.as_dict(),
            sharpe=self.metrics.sharpe,
            n_periods=len(returns),
            returns_skew=self.metrics.returns_skew,
            returns_kurtosis=self.metrics.returns_kurtosis,
        )

    # --- Internals ---

    def _reset(self) -> None:
        self._handler.reset()
        self.portfolio.reset()
        self.ledger.reset()
        self.execution.reset()
        self.strategy.reset()

    def _closes(self, batch: Any) -> dict[str, float]:
        """Today's closes, for sizing and marking. Today's — never tomorrow's."""

        bars = batch.bars
        return {
            str(row.security_id): float(row.close)
            for row in bars.itertuples(index=False)
        }

    def _mark(self, today: date, closes: dict[str, float]) -> DailyMark:
        """Mark the book at today's close.

        Positions in securities that did not trade today are held at their last
        visible price rather than dropped — a suspended security is still owned.
        """

        prices = dict(closes)
        for security_id, position in self.portfolio.positions.items():
            if position.is_open and security_id not in prices:
                last = self._handler.latest_price(security_id)
                if last is not None:
                    prices[security_id] = last

        equity = self.portfolio.equity(prices)
        gross = self.portfolio.gross_exposure(prices)
        return DailyMark(
            date=today,
            cash=self.portfolio.cash,
            unsettled_cash=self.portfolio.unsettled_cash,
            holdings_value=gross,
            equity=equity,
            positions=sum(1 for p in self.portfolio.positions.values() if p.is_open),
            gross_exposure=gross,
        )
