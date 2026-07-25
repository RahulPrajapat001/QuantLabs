"""``Portfolio`` — signals become sized orders, subject to capital and risk limits.

**This module knows nothing about strategies.** It receives
:class:`~quantlab.backtesting.events.Signal` objects as data — a security, a
direction, a conviction — and decides how much to transact given available capital
and its risk limits. It cannot ask which strategy produced a signal, or reach into
one to see what it is about to do next. That boundary is what allows the same
sizing and risk logic to be trusted across ten different strategies.

**T+1 settlement is modelled as capital availability, not just timing.** When a
position is sold, the proceeds are not spendable until the following trading day.
A backtest that recycles sale proceeds into a same-day purchase is quietly running
on credit it did not have, and the effect compounds: a high-turnover strategy can
show returns that require a margin facility nobody granted it. Sale proceeds land
in :attr:`unsettled_cash` and are released by :meth:`settle`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum

import structlog

from quantlab.backtesting.events import Direction, Fill, Order, Side, Signal

log = structlog.get_logger(__name__)


class SizingMethod(StrEnum):
    """How a signal is converted into a quantity."""

    #: Split available capital evenly across the target number of positions.
    EQUAL_WEIGHT = "equal_weight"
    #: A fixed fraction of current equity per position.
    FIXED_FRACTION = "fixed_fraction"
    #: Equal weight scaled by the signal's conviction.
    SIGNAL_WEIGHTED = "signal_weighted"
    #: Inverse of recent volatility, so each position contributes similar risk.
    #: Requires the caller to supply volatility; falls back to equal weight.
    INVERSE_VOLATILITY = "inverse_volatility"


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Hard constraints applied to every order, before it reaches execution.

    These are limits, not targets: an order that would breach one is trimmed to
    the boundary or dropped. They exist because a sizing rule alone will happily
    concentrate the whole book into one name after a run of signals.
    """

    #: Maximum fraction of equity in any single security.
    max_position_weight: float = 0.10
    #: Maximum number of simultaneous positions.
    max_positions: int = 50
    #: Maximum gross exposure as a fraction of equity. 1.0 means no leverage.
    max_gross_exposure: float = 1.0
    #: Fraction of equity never deployed, so a rebalance is never blocked by
    #: having spent the last rupee.
    cash_reserve: float = 0.02
    #: Minimum order value; below this, costs dominate and the trade is noise.
    min_order_value: float = 1_000.0

    def __post_init__(self) -> None:
        if not 0.0 < self.max_position_weight <= 1.0:
            raise ValueError("max_position_weight must lie in (0, 1]")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        if not 0.0 <= self.cash_reserve < 1.0:
            raise ValueError("cash_reserve must lie in [0, 1)")


@dataclass
class Position:
    """A holding, with enough history to compute realised P&L on exit."""

    security_id: str
    quantity: float = 0.0
    #: Average price paid per share, **including** the costs of acquisition, so
    #: realised P&L is net rather than flattering.
    cost_basis: float = 0.0
    opened_on: date | None = None

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealised_pnl(self, price: float) -> float:
        return (price - self.cost_basis) * self.quantity


@dataclass
class Portfolio:
    """Capital, positions, sizing, and risk limits.

    Parameters
    ----------
    initial_capital:
        Starting cash.
    sizing:
        How signals become quantities.
    limits:
        Hard risk constraints.
    target_positions:
        Denominator for equal-weight sizing. Defaults to ``limits.max_positions``.
    fractional_shares:
        Whether fractional quantities are allowed. ``False`` (whole shares) is
        realistic for Indian equities; ``True`` exists so a buy-and-hold validation
        run can deploy exactly 100% of capital and reproduce the benchmark to
        floating-point tolerance, which whole-share rounding would otherwise spoil.
    settlement_days:
        Trading days before sale proceeds become spendable. T+1 since 2023.
    """

    initial_capital: float = 1_000_000.0
    sizing: SizingMethod = SizingMethod.EQUAL_WEIGHT
    limits: RiskLimits = field(default_factory=RiskLimits)
    target_positions: int | None = None
    fractional_shares: bool = False
    settlement_days: int = 1

    cash: float = field(init=False)
    positions: dict[str, Position] = field(init=False, default_factory=dict)
    #: Sale proceeds not yet spendable, as ``(available_from, amount)``.
    _pending_credits: list[tuple[date, float]] = field(init=False, default_factory=list)
    realised_pnl: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.cash = self.initial_capital

    def reset(self) -> None:
        """Return to the starting state, so one portfolio can drive several runs."""

        self.cash = self.initial_capital
        self.positions = {}
        self._pending_credits = []
        self.realised_pnl = 0.0

    # --- Capital ---

    @property
    def unsettled_cash(self) -> float:
        """Sale proceeds received but not yet spendable."""

        return sum(amount for _, amount in self._pending_credits)

    @property
    def settled_cash(self) -> float:
        """Cash that may actually fund a purchase today."""

        return self.cash

    def deployable_cash(self, equity: float) -> float:
        """Settled cash minus the reserve — what sizing is allowed to spend."""

        reserve = equity * self.limits.cash_reserve
        return max(0.0, self.cash - reserve)

    def settle(self, on: date) -> float:
        """Release credits whose settlement date has arrived. Returns the amount."""

        if not self._pending_credits:
            return 0.0
        released = 0.0
        still_pending: list[tuple[date, float]] = []
        for available_from, amount in self._pending_credits:
            if available_from <= on:
                released += amount
            else:
                still_pending.append((available_from, amount))
        self._pending_credits = still_pending
        if released:
            self.cash += released
            log.debug("portfolio.settled", on=on.isoformat(), amount=released)
        return released

    def equity(self, prices: dict[str, float]) -> float:
        """Total equity: settled cash, unsettled proceeds, and positions at mark.

        Unsettled proceeds are included — the money exists, it simply cannot be
        spent yet. Excluding it would understate equity and distort every metric.
        """

        holdings = sum(
            position.market_value(prices[security_id])
            for security_id, position in self.positions.items()
            if position.is_open and security_id in prices
        )
        return self.cash + self.unsettled_cash + holdings

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(
            position.market_value(prices[security_id])
            for security_id, position in self.positions.items()
            if position.is_open and security_id in prices
        )

    def open_positions(self) -> dict[str, float]:
        """Quantity held per security, for the execution handler's sell checks."""

        return {
            security_id: position.quantity
            for security_id, position in self.positions.items()
            if position.is_open
        }

    # --- Signals to orders ---

    def generate_orders(
        self,
        signals: list[Signal],
        prices: dict[str, float],
        *,
        on: date,
        volatilities: dict[str, float] | None = None,
    ) -> list[Order]:
        """Convert signals into sized, risk-checked orders.

        Exits are processed first, so capital freed by a closing position is
        accounted for before entries are sized — though it is not *spendable* until
        it settles, which is exactly the constraint being modelled.
        """

        equity = self.equity(prices)
        exits: list[Order] = []
        entries: list[Signal] = []

        for signal in signals:
            if signal.direction is Direction.EXIT:
                order = self._exit_order(signal, on)
                if order is not None:
                    exits.append(order)
            elif signal.direction is Direction.SHORT:
                # Long-only for now: a short signal on a held name is an exit, and
                # otherwise ignored. Shorting Indian equities in the cash segment
                # is not available to a delivery-based strategy, so pretending
                # otherwise would overstate what the strategy could do.
                order = self._exit_order(signal, on)
                if order is not None:
                    exits.append(order)
            else:
                entries.append(signal)

        orders = list(exits)
        orders.extend(
            self._entry_orders(
                entries, prices, equity=equity, on=on, volatilities=volatilities
            )
        )
        return orders

    def _exit_order(self, signal: Signal, on: date) -> Order | None:
        position = self.positions.get(signal.security_id)
        if position is None or not position.is_open:
            return None
        return Order(
            date=on,
            security_id=signal.security_id,
            side=Side.SELL,
            quantity=position.quantity,
            reason="exit",
            strategy_name=signal.strategy_name,
        )

    def _entry_orders(
        self,
        signals: list[Signal],
        prices: dict[str, float],
        *,
        equity: float,
        on: date,
        volatilities: dict[str, float] | None,
    ) -> list[Order]:
        if not signals:
            return []

        # A repeated entry signal is a hold, not a pyramid — unless the signal
        # explicitly says it is a contribution (``accumulate``), which is what a
        # dollar-cost-averaging schedule emits.
        priced = [
            s for s in signals if s.security_id in prices and prices[s.security_id] > 0
        ]
        fresh = [
            s
            for s in priced
            if s.accumulate
            or not self.positions.get(s.security_id, Position(s.security_id)).is_open
        ]
        if not fresh:
            return []

        open_ids = {
            security_id
            for security_id, position in self.positions.items()
            if position.is_open
        }
        # Adding to a name already held consumes no new position slot.
        new_names = {s.security_id for s in fresh} - open_ids
        room = max(0, self.limits.max_positions - len(open_ids))
        if room == 0 and new_names:
            log.debug(
                "portfolio.position_limit_reached", limit=self.limits.max_positions
            )
            fresh = [s for s in fresh if s.security_id in open_ids]
            if not fresh:
                return []
        elif len(new_names) > room:
            # Highest conviction first when the limit binds; adds are never crowded
            # out by it, since they need no slot.
            ranked_new = sorted(
                (s for s in fresh if s.security_id not in open_ids),
                key=lambda s: -s.strength,
            )[:room]
            fresh = [s for s in fresh if s.security_id in open_ids] + ranked_new

        budget = self.deployable_cash(equity)
        gross_room = max(
            0.0, equity * self.limits.max_gross_exposure - self.gross_exposure(prices)
        )
        budget = min(budget, gross_room)
        if budget <= 0:
            return []

        weights = self._weights(fresh, volatilities)
        orders: list[Order] = []
        for signal in fresh:
            price = prices[signal.security_id]
            # The weight cap applies to the *resulting* position, so an add is
            # limited by the headroom left after what is already held. Without
            # this, a monthly contribution would top the position back up to the
            # cap every month and the limit would not bind at all.
            held = self.positions.get(signal.security_id)
            existing_value = held.market_value(price) if held and held.is_open else 0.0
            headroom = max(
                0.0, equity * self.limits.max_position_weight - existing_value
            )
            allocation = min(budget * weights[signal.security_id], headroom)
            if allocation < self.limits.min_order_value:
                continue
            quantity = allocation / price
            if not self.fractional_shares:
                quantity = float(int(quantity))
            if quantity <= 0:
                continue
            orders.append(
                Order(
                    date=on,
                    security_id=signal.security_id,
                    side=Side.BUY,
                    quantity=quantity,
                    reason="entry",
                    strategy_name=signal.strategy_name,
                    reference_price=price,
                    # Sized by value: the share count is resolved at the fill price
                    # so the overnight gap between this close and tomorrow's open
                    # does not leave an arbitrary cash residue.
                    notional=allocation,
                    allow_fractional=self.fractional_shares,
                )
            )
        return orders

    def _weights(
        self, signals: list[Signal], volatilities: dict[str, float] | None
    ) -> dict[str, float]:
        """Fraction of the budget each signal receives. Sums to at most 1."""

        target = self.target_positions or self.limits.max_positions

        if self.sizing is SizingMethod.FIXED_FRACTION:
            share = min(1.0 / max(1, target), self.limits.max_position_weight)
            return {s.security_id: share for s in signals}

        if self.sizing is SizingMethod.SIGNAL_WEIGHTED:
            total = sum(s.strength for s in signals)
            if total <= 0:
                return {s.security_id: 1.0 / len(signals) for s in signals}
            return {s.security_id: s.strength / total for s in signals}

        if self.sizing is SizingMethod.INVERSE_VOLATILITY and volatilities:
            inverses = {
                s.security_id: 1.0 / volatilities[s.security_id]
                for s in signals
                if volatilities.get(s.security_id, 0.0) > 0
            }
            total = sum(inverses.values())
            if inverses and total > 0:
                # Signals with no volatility estimate get an equal-weight share of
                # nothing rather than a silently wrong one.
                return {
                    s.security_id: inverses.get(s.security_id, 0.0) / total
                    for s in signals
                }

        # EQUAL_WEIGHT, and the fallback for every degenerate case above. Divide by
        # the number of signals when they fit inside the target, so a strategy
        # holding fewer names than its cap still deploys its capital.
        divisor = max(len(signals), 1) if len(signals) >= target else target
        share = 1.0 / divisor
        return {s.security_id: share for s in signals}

    # --- Applying fills ---

    def apply_fill(self, fill: Fill) -> float:
        """Update cash and positions for an executed trade.

        Returns realised P&L, which is zero for a purchase. Sale proceeds go to
        :attr:`unsettled_cash`, not to spendable cash.
        """

        position = self.positions.setdefault(
            fill.security_id, Position(fill.security_id)
        )

        if fill.side is Side.BUY:
            total_cost = fill.gross_value + float(fill.costs.total)
            self.cash -= total_cost
            new_quantity = position.quantity + fill.quantity
            # Cost basis includes charges, so realised P&L is net of what it took
            # to acquire the position.
            position.cost_basis = (
                position.cost_basis * position.quantity + total_cost
            ) / new_quantity
            position.quantity = new_quantity
            if position.opened_on is None:
                position.opened_on = fill.date
            return 0.0

        quantity = min(fill.quantity, position.quantity)
        proceeds = quantity * fill.price - float(fill.costs.total)
        realised = proceeds - position.cost_basis * quantity
        self.realised_pnl += realised

        available_from = fill.date + timedelta(days=self.settlement_days)
        self._pending_credits.append((available_from, proceeds))

        position.quantity -= quantity
        if position.quantity <= 1e-9:
            position.quantity = 0.0
            position.cost_basis = 0.0
            position.opened_on = None
        return realised

    def liquidate(self, security_id: str, price: float, on: date) -> float:
        """Force-close a position at ``price`` — used for delisting events.

        No costs are applied: a delisting liquidation is a corporate action, not a
        trade the strategy chose to make. Proceeds settle like any other sale.
        """

        position = self.positions.get(security_id)
        if position is None or not position.is_open:
            return 0.0
        quantity = position.quantity
        proceeds = quantity * price
        realised = proceeds - position.cost_basis * quantity
        self.realised_pnl += realised
        self._pending_credits.append(
            (on + timedelta(days=self.settlement_days), proceeds)
        )
        position.quantity = 0.0
        position.cost_basis = 0.0
        position.opened_on = None
        log.info(
            "portfolio.liquidated",
            security_id=security_id,
            on=on.isoformat(),
            quantity=quantity,
            price=price,
            realised=realised,
        )
        return realised
