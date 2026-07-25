"""The four event types the backtester passes between its components.

The separation is the point. A ``Signal`` says "this looks attractive" and knows
nothing about position size or capital; an ``Order`` says "buy 250 shares" and
knows nothing about why; a ``Fill`` says what actually happened, at what price,
after what costs. Collapsing these — letting a strategy decide quantity, or letting
a portfolio see the strategy that produced a signal — is what makes vectorised
backtests untrustworthy, because the boundary where look-ahead enters stops being
visible.

Every event carries the date it was created on, so the ledger can reconstruct the
whole causal chain: signal on day *t* → order on day *t* → fill on day *t+1*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

#: Alias for the ``date`` type. Needed because these dataclasses have a field
#: called ``date``, which shadows the imported type for any annotation that comes
#: after it in the same class body.
_Date = date


class Direction(StrEnum):
    """What a strategy is asserting about a security."""

    LONG = "long"
    SHORT = "short"
    #: Close whatever position exists, without an opinion on direction.
    EXIT = "exit"


class Side(StrEnum):
    """Which way an order transacts."""

    BUY = "buy"
    SELL = "sell"


class RejectionReason(StrEnum):
    """Why an order received no fill.

    Rejections are recorded, never dropped. The circuit-limit and volume-cap
    counts are findings in their own right: a momentum strategy whose orders are
    rejected precisely on the days its signal fires does not have the returns its
    signal implies.
    """

    CIRCUIT_LIMIT = "circuit_limit"
    VOLUME_CAP = "volume_cap"
    NO_BAR = "no_bar"
    INSUFFICIENT_CAPITAL = "insufficient_capital"
    ZERO_QUANTITY = "zero_quantity"
    NO_POSITION = "no_position"


@dataclass(frozen=True, slots=True)
class Signal:
    """A strategy's opinion about one security on one date.

    No quantity, no price, no capital. ``strength`` is an optional conviction in
    ``[0, 1]`` that a sizing policy may use; a strategy that has no view on
    conviction leaves it at 1.0 rather than inventing one.
    """

    date: date
    security_id: str
    direction: Direction
    strategy_name: str
    strength: float = 1.0
    #: Whether this signal means "add to the existing position" rather than "open
    #: one". Default ``False``: a repeated entry signal is a *hold*, so a trend
    #: strategy that keeps saying "long" every bar does not pyramid into one name.
    #:
    #: Accumulation strategies — dollar-cost averaging above all — need the
    #: opposite, and a periodic contribution genuinely is a different instruction
    #: rather than strategy internals leaking into the portfolio. Without this
    #: flag, DCA degenerates into buy-and-hold, which is a different strategy.
    accumulate: bool = False
    #: Free-form diagnostics (the indicator values behind the signal, say). Logged,
    #: never acted upon.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"signal strength must lie in [0, 1], got {self.strength}")


@dataclass(frozen=True, slots=True)
class Order:
    """An instruction to transact a specific quantity, created on ``date``.

    Filled at the *next* bar's open by default, which is why ``date`` (created)
    and the eventual fill date differ. An order that cannot be filled becomes a
    :class:`RejectedOrder` — it never silently disappears.
    """

    date: date
    security_id: str
    side: Side
    quantity: float
    #: Why the portfolio raised it, e.g. ``"entry"``, ``"exit"``, ``"stop_loss"``.
    reason: str = "entry"
    strategy_name: str = ""
    #: Price used for sizing, for auditing slippage against the eventual fill.
    reference_price: float | None = None
    #: Target rupee amount **including costs**, when the portfolio sized by value
    #: rather than by share count. The execution handler resolves the quantity at
    #: the fill price, which is the only price at which "deploy ₹X" is well defined.
    #:
    #: This exists because sizing happens on day *t*'s close and the fill happens at
    #: day *t+1*'s open. A share count fixed at sizing time leaves residual cash
    #: that varies with the overnight gap; a rupee amount does not. ``quantity``
    #: remains populated as the estimate, so an order is always inspectable.
    notional: float | None = None
    #: Whether the resolved quantity may be fractional. False (whole shares) is
    #: correct for Indian equities.
    allow_fractional: bool = False

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(
                f"order quantity must be positive, got {self.quantity}; use "
                "Side.SELL rather than a negative quantity"
            )
        if self.notional is not None and self.notional <= 0:
            raise ValueError(f"order notional must be positive, got {self.notional}")

    def with_quantity(self, quantity: float) -> Order:
        """A copy with a different quantity — used when a volume cap splits it.

        The notional is dropped: once a cap has bound, the order is defined by the
        share count that was actually admitted, not by the original rupee target.
        """

        return Order(
            date=self.date,
            security_id=self.security_id,
            side=self.side,
            quantity=quantity,
            reason=self.reason,
            strategy_name=self.strategy_name,
            reference_price=self.reference_price,
            notional=None,
            allow_fractional=self.allow_fractional,
        )


@dataclass(frozen=True, slots=True)
class Fill:
    """An executed trade, with its costs itemised.

    ``price`` is post-slippage: what was actually paid per share. ``costs`` is the
    itemised breakdown, kept separate from the price so cost attribution stays
    inspectable rather than being buried in an effective price.
    """

    date: date
    security_id: str
    side: Side
    quantity: float
    price: float
    #: Itemised charges. Typed as ``Any`` to avoid a circular import with
    #: :mod:`quantlab.backtesting.costs`; it is always a ``CostBreakdown``.
    costs: Any
    #: The bar's untouched open, before slippage was applied.
    reference_price: float
    reason: str = "entry"
    strategy_name: str = ""
    #: Date the order was raised — one bar before ``date`` under T+1 execution.
    order_date: _Date | None = None

    @property
    def gross_value(self) -> float:
        """Turnover: quantity times executed price, before costs."""

        return self.quantity * self.price

    @property
    def cash_flow(self) -> float:
        """Signed effect on cash: negative for a buy, positive for a sale.

        Costs always reduce cash, on both sides — a sale nets *less* than its
        gross value, a purchase costs *more*.
        """

        total_costs = float(self.costs.total)
        if self.side is Side.BUY:
            return -(self.gross_value + total_costs)
        return self.gross_value - total_costs

    @property
    def slippage_per_share(self) -> float:
        """Signed slippage against the reference open, in currency per share."""

        if self.side is Side.BUY:
            return self.price - self.reference_price
        return self.reference_price - self.price


@dataclass(frozen=True, slots=True)
class RejectedOrder:
    """An order that received no fill, and why.

    Recorded in the ledger and counted in the metrics. PRD Module 8 singles this
    out because the rejections are not random with respect to the signal: circuit
    breakers hit on exactly the large-move days that momentum strategies trade.
    """

    date: date
    order: Order
    reason: RejectionReason
    detail: str = ""
    #: Quantity that *was* filled, when a volume cap allowed a partial fill.
    filled_quantity: float = 0.0

    @property
    def security_id(self) -> str:
        return self.order.security_id

    @property
    def rejected_quantity(self) -> float:
        return self.order.quantity - self.filled_quantity
