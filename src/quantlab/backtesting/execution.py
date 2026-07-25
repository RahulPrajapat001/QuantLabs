"""``ExecutionHandler`` — orders become fills, or documented rejections.

Default execution is the **next bar's open**. An order raised on the close of day
*t* is filled at the open of day *t+1*, because that is the earliest price a
decision made at *t* could actually transact at. Filling at day *t*'s close is the
single most common way a backtest invents returns it could never have earned.

Three realism rules, each of which removes returns that a naive engine keeps:

**Circuit limits.** NSE applies price bands (2%, 5%, 10%, 20%). A security locked
at its band cannot be traded at that price — there is no counterparty on the other
side. The order receives no fill and is recorded as a
:class:`~quantlab.backtesting.events.RejectedOrder`. This matters more than it
sounds: momentum and breakout signals fire on exactly the large-move days that
trigger bands, so the returns a naive engine books from these days are precisely
the ones that were never available. Vendor ``circuit_hit`` data is authoritative
when present; otherwise the band is inferred from the move off the previous close.

**Volume participation cap.** An order may not take more than a configured share
of the bar's volume (default 1%). The excess is either rejected or carried into
following days, configurable — a strategy that only works when it can buy 30% of a
day's volume does not work.

**T+1 settlement.** Handled in the portfolio, not here: sale proceeds are not
spendable until the following day, so settlement constrains *capital availability*
rather than merely execution timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import pandas as pd
import structlog

from quantlab.backtesting.costs import (
    CostModel,
    Segment,
    SlippageModel,
    Trade,
    ZeroSlippage,
)
from quantlab.backtesting.data_handler import DataHandler
from quantlab.backtesting.events import (
    Fill,
    Order,
    RejectedOrder,
    RejectionReason,
    Side,
)

log = structlog.get_logger(__name__)

#: NSE price bands, as fractions. 20% is the widest ordinary band; securities in
#: the F&O segment have none, which ``band=None`` models.
NSE_PRICE_BANDS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)


class ExcessPolicy(StrEnum):
    """What happens to the part of an order a volume cap will not admit."""

    #: Drop the excess and record it as a rejection. Conservative and honest:
    #: the strategy simply does not get the position it asked for.
    REJECT = "reject"
    #: Carry the excess into subsequent bars until it fills or expires.
    SPLIT = "split"


@dataclass(frozen=True, slots=True)
class CircuitLimitPolicy:
    """Decides whether a security is untradeable on a given bar.

    ``band`` is the fraction the price may move from the previous close before it
    is locked. ``None`` disables inference entirely (for F&O-segment names with no
    band), in which case only explicit vendor ``circuit_hit`` data can reject.

    Inference is deliberately conservative: it fires when the move *reaches* the
    band, and additionally when the bar has no range at all (``high == low``),
    which is what a bar locked at its band looks like in daily data.
    """

    band: float | None = 0.20
    #: Treat a zero-range bar as locked. A bar whose high equals its low had no
    #: two-sided market; that is the signature of a band lock in daily data.
    treat_zero_range_as_locked: bool = True

    def is_locked(
        self, bar: pd.Series, previous_close: float | None
    ) -> tuple[bool, str]:
        """Return ``(locked, human-readable reason)`` for this bar."""

        explicit = bar.get("circuit_hit") if hasattr(bar, "get") else None
        if explicit is not None and not pd.isna(explicit):
            if bool(explicit):
                return True, "vendor circuit_hit flag set for this bar"
            # Explicit data saying "not locked" is authoritative — trust it over
            # inference, which cannot distinguish a band lock from a large gap.
            return False, ""

        if self.treat_zero_range_as_locked:
            high, low = float(bar["high"]), float(bar["low"])
            if high == low:
                return True, f"zero-range bar at {high:.2f} — no two-sided market"

        if self.band is None or previous_close is None or previous_close <= 0:
            return False, ""

        move = abs(float(bar["open"]) / previous_close - 1.0)
        if move >= self.band:
            return (
                True,
                f"open moved {move * 100:.2f}% from the previous close, at or "
                f"beyond the {self.band * 100:.0f}% price band",
            )
        return False, ""


@dataclass
class ExecutionHandler:
    """Turns orders into fills at the next bar's open, applying every friction.

    Parameters
    ----------
    cost_model:
        Prices the itemised charges on each fill.
    slippage:
        Adverse price movement between the reference open and the executed price.
    max_participation:
        Cap on the share of a bar's volume one order may take. ``None`` disables
        the cap (validation only — a real order always competes for liquidity).
    excess_policy:
        Whether a capped order's remainder is rejected or carried forward.
    circuit_policy:
        Band-lock detection.
    segment:
        Delivery or intraday, which changes the tax treatment.
    max_carry_days:
        How long a split remainder may keep trying before it expires. Without a
        limit, an illiquid name accumulates an unfillable backlog that silently
        distorts every later bar.
    """

    cost_model: CostModel
    slippage: SlippageModel = field(default_factory=ZeroSlippage)
    max_participation: float | None = 0.01
    excess_policy: ExcessPolicy = ExcessPolicy.REJECT
    circuit_policy: CircuitLimitPolicy = field(default_factory=CircuitLimitPolicy)
    segment: Segment = Segment.DELIVERY
    max_carry_days: int = 5

    #: Remainders carried forward under ``ExcessPolicy.SPLIT``, with the date each
    #: was first raised so it can expire.
    _carried: list[tuple[Order, date]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if (
            self.max_participation is not None
            and not 0.0 < self.max_participation <= 1.0
        ):
            raise ValueError(
                f"max_participation must lie in (0, 1], got {self.max_participation}"
            )

    def reset(self) -> None:
        """Clear carried remainders, so one handler can drive several runs."""

        self._carried.clear()

    @property
    def carried_orders(self) -> list[Order]:
        """Order remainders still waiting to fill."""

        return [order for order, _ in self._carried]

    def execute(
        self,
        orders: list[Order],
        handler: DataHandler,
        *,
        available_cash: float,
        positions: dict[str, float] | None = None,
    ) -> tuple[list[Fill], list[RejectedOrder]]:
        """Attempt ``orders`` against the current bar of ``handler``.

        ``handler``'s cursor must already be on the execution bar — the engine
        advances it before calling, which is what makes this T+1: the orders were
        raised on the previous bar.

        Buys are attempted in order and consume ``available_cash`` as they fill, so
        a portfolio that raised more orders than it can afford gets the earlier ones
        and an ``INSUFFICIENT_CAPITAL`` rejection for the rest, rather than silently
        going into unfunded leverage.
        """

        execution_date = handler.current_date
        if execution_date is None:
            raise RuntimeError(
                "ExecutionHandler.execute called before any bar was served; the "
                "engine must advance the DataHandler first"
            )

        pending = self._collect_pending(orders, execution_date)
        fills: list[Fill] = []
        rejections: list[RejectedOrder] = []
        cash = available_cash
        held = dict(positions or {})

        for order in pending:
            bar = handler.current_bar(order.security_id)
            if bar is None:
                rejections.append(
                    RejectedOrder(
                        date=execution_date,
                        order=order,
                        reason=RejectionReason.NO_BAR,
                        detail=(
                            f"{order.security_id} did not trade on "
                            f"{execution_date.isoformat()}"
                        ),
                    )
                )
                continue

            locked, why = self.circuit_policy.is_locked(
                bar, handler.previous_close(order.security_id)
            )
            if locked:
                rejections.append(
                    RejectedOrder(
                        date=execution_date,
                        order=order,
                        reason=RejectionReason.CIRCUIT_LIMIT,
                        detail=why,
                    )
                )
                log.debug(
                    "execution.circuit_rejected",
                    security_id=order.security_id,
                    date=execution_date.isoformat(),
                    side=order.side.value,
                    quantity=order.quantity,
                    reason=why,
                )
                continue

            quantity = order.quantity
            # A notional buy resolves its share count at the fill price, so
            # "deploy ₹X including costs" lands exactly on ₹X.
            if order.side is Side.BUY and order.notional is not None:
                quantity = self._resolve_notional(order, float(bar["open"]))
                if quantity <= 0:
                    rejections.append(
                        RejectedOrder(
                            date=execution_date,
                            order=order,
                            reason=RejectionReason.ZERO_QUANTITY,
                            detail=(
                                f"₹{order.notional:,.2f} buys less than one share at "
                                f"₹{float(bar['open']):,.2f}"
                            ),
                        )
                    )
                    continue

            # A sell can never exceed the position actually held.
            if order.side is Side.SELL:
                holding = held.get(order.security_id, 0.0)
                if holding <= 0:
                    rejections.append(
                        RejectedOrder(
                            date=execution_date,
                            order=order,
                            reason=RejectionReason.NO_POSITION,
                            detail=f"no position in {order.security_id} to sell",
                        )
                    )
                    continue
                quantity = min(quantity, holding)

            capped, cap_detail = self._apply_volume_cap(quantity, bar)
            if capped <= 0.0:
                rejections.append(
                    RejectedOrder(
                        date=execution_date,
                        order=order,
                        reason=RejectionReason.VOLUME_CAP,
                        detail=cap_detail,
                    )
                )
                self._maybe_carry(order, quantity, execution_date)
                continue

            reference_price = float(bar["open"])
            price = self.slippage.fill_price(
                reference_price=reference_price,
                side=order.side,
                order_quantity=capped,
                bar_volume=float(bar["volume"]),
            )

            if order.side is Side.BUY:
                capped, cash_ok = self._affordable(capped, price, cash, order)
                if not cash_ok:
                    rejections.append(
                        RejectedOrder(
                            date=execution_date,
                            order=order,
                            reason=RejectionReason.INSUFFICIENT_CAPITAL,
                            detail=(
                                f"settled cash ₹{cash:,.2f} cannot fund "
                                f"{order.quantity:g} × ₹{price:,.2f}"
                            ),
                        )
                    )
                    continue

            costs = self.cost_model.charges(
                Trade(
                    security_id=order.security_id,
                    side=order.side,
                    quantity=capped,
                    price=price,
                    segment=self.segment,
                )
            )
            fill = Fill(
                date=execution_date,
                security_id=order.security_id,
                side=order.side,
                quantity=capped,
                price=price,
                costs=costs,
                reference_price=reference_price,
                reason=order.reason,
                strategy_name=order.strategy_name,
                order_date=order.date,
            )
            fills.append(fill)
            if order.side is Side.BUY:
                cash += fill.cash_flow  # negative
                held[order.security_id] = held.get(order.security_id, 0.0) + capped
            else:
                held[order.security_id] = held.get(order.security_id, 0.0) - capped

            # Anything the cap held back is recorded and possibly carried.
            if capped < quantity:
                remainder = quantity - capped
                rejections.append(
                    RejectedOrder(
                        date=execution_date,
                        order=order,
                        reason=RejectionReason.VOLUME_CAP,
                        detail=cap_detail,
                        filled_quantity=capped,
                    )
                )
                self._maybe_carry(order, remainder, execution_date)

        return fills, rejections

    # --- Internals ---

    def _collect_pending(
        self, orders: list[Order], execution_date: date
    ) -> list[Order]:
        """New orders plus any still-valid carried remainders, expiring the rest."""

        if not self._carried:
            return list(orders)

        still_valid: list[tuple[Order, date]] = []
        revived: list[Order] = []
        for order, first_seen in self._carried:
            if (execution_date - first_seen).days > self.max_carry_days:
                log.debug(
                    "execution.carried_order_expired",
                    security_id=order.security_id,
                    quantity=order.quantity,
                    first_seen=first_seen.isoformat(),
                )
                continue
            revived.append(order)
            still_valid.append((order, first_seen))
        self._carried = []
        # Carried remainders go first: they have been waiting longest.
        return [*revived, *orders]

    def _apply_volume_cap(self, quantity: float, bar: pd.Series) -> tuple[float, str]:
        if self.max_participation is None:
            return quantity, ""
        volume = float(bar["volume"])
        if volume <= 0:
            return 0.0, "bar reported zero volume — nothing could be transacted"
        limit = volume * self.max_participation
        if quantity <= limit:
            return quantity, ""
        detail = (
            f"order for {quantity:,.0f} exceeds {self.max_participation * 100:g}% of "
            f"the bar's {volume:,.0f} volume (cap {limit:,.0f})"
        )
        return limit, detail

    def _maybe_carry(
        self, order: Order, remainder: float, execution_date: date
    ) -> None:
        if self.excess_policy is not ExcessPolicy.SPLIT or remainder <= 0:
            return
        self._carried.append((order.with_quantity(remainder), order.date))

    def _cost_rate(self, order: Order, quantity: float, price: float) -> float:
        """Charges as a fraction of turnover, from the actual cost model.

        Measured rather than assumed: a flat headroom would silently reserve
        capital even under :class:`~quantlab.backtesting.costs.ZeroCostModel`, which
        would stop buy-and-hold from reproducing its benchmark and quietly
        understate every strategy's deployed capital.
        """

        gross = quantity * price
        if gross <= 0:
            return 0.0
        costs = self.cost_model.charges(
            Trade(
                security_id=order.security_id,
                side=order.side,
                quantity=quantity,
                price=price,
                segment=self.segment,
            )
        )
        return float(costs.total) / gross

    def _resolve_notional(self, order: Order, price: float) -> float:
        """Shares that spend exactly ``order.notional``, costs included."""

        if price <= 0 or order.notional is None:
            return 0.0
        probe = order.notional / price
        rate = self._cost_rate(order, probe, price)
        quantity = order.notional / (price * (1.0 + rate))
        if not order.allow_fractional:
            quantity = float(int(quantity))
        return max(0.0, quantity)

    def _affordable(
        self, quantity: float, price: float, cash: float, order: Order
    ) -> tuple[float, bool]:
        """Trim a buy to what settled cash can fund, or reject it outright.

        Affordability is checked against the charges the cost model actually
        returns for this trade, not a guess.
        """

        if price <= 0 or cash <= 0:
            return 0.0, False
        rate = self._cost_rate(order, quantity, price)
        if quantity * price * (1.0 + rate) <= cash:
            return quantity, True
        affordable = cash / (price * (1.0 + rate))
        if not order.allow_fractional:
            affordable = float(int(affordable))
        if affordable <= 0:
            return 0.0, False
        return affordable, True
