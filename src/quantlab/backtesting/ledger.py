"""``Ledger`` — the complete record of what the backtest actually did.

Every fill, every rejection, every daily mark. The metrics module computes nothing
the ledger cannot substantiate, and the research log's numbers all trace back here.

Two things it does that a simple equity-curve recorder does not:

**Closed-trade accounting.** Win rate, profit factor, average trade and expectancy
are properties of *round trips*, not of daily returns, so buys and sells are matched
FIFO into :class:`ClosedTrade` records with net P&L. Computing a "win rate" from
daily equity changes measures something else entirely and flatters a strategy that
grinds small gains and takes occasional large losses.

**Cost attribution.** Charges are accumulated by statutory component, so the report
can say which head consumed the return rather than reporting one aggregate. At
Indian rates, STT alone is usually the dominant term, and that is actionable —
it argues for lower turnover, not for better execution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from quantlab.backtesting.costs import CostBreakdown
from quantlab.backtesting.events import (
    Fill,
    RejectedOrder,
    RejectionReason,
    Side,
    Signal,
)


@dataclass(frozen=True, slots=True)
class DailyMark:
    """End-of-day state. One row per trading day, always — including flat days."""

    date: date
    cash: float
    unsettled_cash: float
    holdings_value: float
    equity: float
    positions: int
    gross_exposure: float

    @property
    def invested_fraction(self) -> float:
        return self.gross_exposure / self.equity if self.equity else 0.0


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """One completed round trip, matched FIFO.

    ``pnl`` is **net**: entry costs are in the cost basis and exit costs are
    deducted from proceeds, so a "winning" trade here is one that won after the
    broker and the taxman were paid.
    """

    security_id: str
    entry_date: date
    exit_date: date
    quantity: float
    entry_price: float
    exit_price: float
    #: Costs attributable to this round trip, both legs.
    costs: float
    pnl: float
    strategy_name: str = ""

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days

    @property
    def return_pct(self) -> float:
        invested = self.entry_price * self.quantity
        return self.pnl / invested if invested else 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class _OpenLot:
    """An unmatched buy, waiting for a sale to close it (FIFO)."""

    date: date
    quantity: float
    price: float
    #: Acquisition costs allocated to this lot.
    costs: float
    strategy_name: str


@dataclass
class Ledger:
    """Accumulates the full record of a backtest run."""

    fills: list[Fill] = field(default_factory=list)
    rejections: list[RejectedOrder] = field(default_factory=list)
    marks: list[DailyMark] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    #: Running itemised cost total.
    costs: CostBreakdown = field(default_factory=CostBreakdown)
    #: Realised P&L from delisting liquidations, which are not strategy trades.
    liquidations: list[tuple[date, str, float, float]] = field(default_factory=list)

    _open_lots: dict[str, deque[_OpenLot]] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self.fills.clear()
        self.rejections.clear()
        self.marks.clear()
        self.signals.clear()
        self.closed_trades.clear()
        self.liquidations.clear()
        self.costs = CostBreakdown()
        self._open_lots.clear()

    # --- Recording ---

    def record_signals(self, signals: list[Signal]) -> None:
        self.signals.extend(signals)

    def record_fill(self, fill: Fill) -> None:
        """Record a fill and update FIFO lot matching."""

        self.fills.append(fill)
        self.costs = self.costs + fill.costs
        lots = self._open_lots.setdefault(fill.security_id, deque())

        if fill.side is Side.BUY:
            lots.append(
                _OpenLot(
                    date=fill.date,
                    quantity=fill.quantity,
                    price=fill.price,
                    costs=float(fill.costs.total),
                    strategy_name=fill.strategy_name,
                )
            )
            return

        # A sale closes lots oldest-first. Exit costs are apportioned across the
        # quantity sold, so a partial exit carries its share and no more.
        remaining = fill.quantity
        exit_costs_per_share = (
            float(fill.costs.total) / fill.quantity if fill.quantity else 0.0
        )
        while remaining > 1e-9 and lots:
            lot = lots[0]
            matched = min(remaining, lot.quantity)
            entry_cost_share = (
                lot.costs * (matched / lot.quantity) if lot.quantity else 0.0
            )
            exit_cost_share = exit_costs_per_share * matched
            gross = (fill.price - lot.price) * matched
            trade_costs = entry_cost_share + exit_cost_share
            self.closed_trades.append(
                ClosedTrade(
                    security_id=fill.security_id,
                    entry_date=lot.date,
                    exit_date=fill.date,
                    quantity=matched,
                    entry_price=lot.price,
                    exit_price=fill.price,
                    costs=trade_costs,
                    pnl=gross - trade_costs,
                    strategy_name=lot.strategy_name or fill.strategy_name,
                )
            )
            lot.quantity -= matched
            lot.costs -= entry_cost_share
            remaining -= matched
            if lot.quantity <= 1e-9:
                lots.popleft()

    def record_rejection(self, rejection: RejectedOrder) -> None:
        self.rejections.append(rejection)

    def record_mark(self, mark: DailyMark) -> None:
        self.marks.append(mark)

    def record_liquidation(
        self, on: date, security_id: str, price: float, realised: float
    ) -> None:
        """Record a delisting liquidation, which is not a strategy trade.

        Kept separate from ``closed_trades`` deliberately: a forced exit at a
        delisting price is not evidence about the strategy's decision quality, and
        folding it into the win rate would misattribute a corporate action.
        """

        self.liquidations.append((on, security_id, price, realised))
        lots = self._open_lots.get(security_id)
        if lots:
            lots.clear()

    # --- Views ---

    @property
    def equity_curve(self) -> pd.Series:
        """Equity by date. The primary input to every return-based metric."""

        if not self.marks:
            return pd.Series(dtype=float, name="equity")
        return pd.Series(
            [m.equity for m in self.marks],
            index=pd.DatetimeIndex([m.date for m in self.marks], name="date"),
            name="equity",
        )

    @property
    def returns(self) -> pd.Series:
        """Daily returns from the equity curve."""

        curve = self.equity_curve
        if curve.empty:
            return pd.Series(dtype=float, name="return")
        return curve.pct_change().dropna().rename("return")

    def marks_frame(self) -> pd.DataFrame:
        """Every daily mark as a frame."""

        return pd.DataFrame([vars(m) for m in self.marks])

    def fills_frame(self) -> pd.DataFrame:
        """Every fill, with costs itemised into columns."""

        rows = []
        for fill in self.fills:
            row = {
                "date": fill.date,
                "security_id": fill.security_id,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.price,
                "reference_price": fill.reference_price,
                "slippage_per_share": fill.slippage_per_share,
                "gross_value": fill.gross_value,
                "cash_flow": fill.cash_flow,
                "reason": fill.reason,
                "order_date": fill.order_date,
            }
            row.update({f"cost_{k}": v for k, v in fill.costs.as_dict().items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def trades_frame(self) -> pd.DataFrame:
        """Every closed round trip."""

        return pd.DataFrame(
            [
                {
                    "security_id": t.security_id,
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "holding_days": t.holding_days,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "costs": t.costs,
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                    "is_win": t.is_win,
                }
                for t in self.closed_trades
            ]
        )

    def rejections_frame(self) -> pd.DataFrame:
        """Every rejected order and why — a finding, not a diagnostic."""

        return pd.DataFrame(
            [
                {
                    "date": r.date,
                    "security_id": r.security_id,
                    "side": r.order.side.value,
                    "quantity": r.order.quantity,
                    "filled_quantity": r.filled_quantity,
                    "rejected_quantity": r.rejected_quantity,
                    "reason": r.reason.value,
                    "detail": r.detail,
                }
                for r in self.rejections
            ]
        )

    def cost_attribution(self) -> dict[str, float]:
        """Total paid per statutory component, plus each as a share of the total."""

        totals = self.costs.as_dict()
        total = totals["total"]
        attribution = dict(totals)
        if total > 0:
            for key, value in totals.items():
                if key != "total":
                    attribution[f"{key}_share"] = value / total
        return attribution

    def rejection_counts(self) -> dict[str, int]:
        """Rejections by reason. Circuit-limit counts matter most."""

        counts = {reason.value: 0 for reason in RejectionReason}
        for rejection in self.rejections:
            counts[rejection.reason.value] += 1
        return counts

    def total_turnover(self) -> float:
        """Sum of the absolute value transacted, both sides."""

        return sum(fill.gross_value for fill in self.fills)
