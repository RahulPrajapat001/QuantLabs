"""Transaction cost models, itemised (PRD Module 8).

Unrealistic costs are one of the four failure modes this project exists to defeat.
The Indian equity market makes it particularly easy to get wrong, because the
charges are levied on **turnover**, not profit: STT at 0.1% per side on delivery
plus 0.015% stamp duty on the buy side means a round trip costs roughly 0.22% of
notional before brokerage — and a strategy turning over its book weekly pays that
fifty times a year. Applying a single "10 bps" figure hides this entirely.

Three models, pluggable through the :class:`CostModel` interface:

* :class:`ZeroCostModel` — validation tests only. Its whole purpose is to make
  buy-and-hold reproduce the benchmark exactly, which is how we know the engine's
  arithmetic is right before costs enter.
* :class:`GenericBpsModel` — configurable bps per side plus a fixed charge.
* :class:`IndianEquityCostModel` — every statutory component computed separately.

Each component is returned as its own field on :class:`CostBreakdown`, so the
ledger can attribute cost by type. "Costs ate the edge" is a weak finding;
"STT alone ate 60% of the gross return at this turnover" is an actionable one.

Every rate here is a **rate**, not a hardcoded amount, and every default is stated
in one place so a change in statutory rates is a one-line edit with a visible diff.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, replace
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from quantlab.backtesting.events import Side

# --- Statutory rates, India (as of FY2024-25) --------------------------------
#
# Sources are the exchange and SEBI circulars. Stated as fractions of turnover.

#: Securities Transaction Tax, delivery segment: 0.1% on **both** sides.
STT_DELIVERY_RATE = 0.001
#: STT, intraday segment: 0.025% on the **sell** side only.
STT_INTRADAY_RATE = 0.00025
#: NSE exchange transaction charge: ~0.00297% of turnover (cash segment).
NSE_TRANSACTION_RATE = 0.0000297
#: SEBI turnover fee: ₹10 per crore = 1e-6 of turnover.
SEBI_TURNOVER_RATE = 1e-6
#: Stamp duty: 0.015% on the **buy** side only.
STAMP_DUTY_RATE = 0.00015
#: GST: 18% on (brokerage + exchange charges).
GST_RATE = 0.18
#: Discount-broker intraday rate and per-order cap (₹20).
DISCOUNT_BROKERAGE_RATE = 0.0003
DISCOUNT_BROKERAGE_CAP = 20.0

#: One paisa. Charges are rounded to it, as a real broker's contract note is.
PAISA = 0.01


class Segment(StrEnum):
    """Delivery or intraday — they are taxed differently, not slightly differently."""

    DELIVERY = "delivery"
    INTRADAY = "intraday"


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Itemised charges on a single trade.

    Kept as separate fields rather than a total so the ledger can answer "what did
    we actually pay for?" — see
    :meth:`quantlab.backtesting.ledger.Ledger.cost_attribution`.
    """

    brokerage: float = 0.0
    #: Securities Transaction Tax.
    stt: float = 0.0
    #: NSE cash-segment transaction charge.
    exchange_transaction: float = 0.0
    sebi_turnover_fee: float = 0.0
    stamp_duty: float = 0.0
    #: GST on brokerage plus exchange charges.
    gst: float = 0.0
    #: Anything the model cannot attribute to a named statutory head.
    other: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_transaction
            + self.sebi_turnover_fee
            + self.stamp_duty
            + self.gst
            + self.other
        )

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            brokerage=self.brokerage + other.brokerage,
            stt=self.stt + other.stt,
            exchange_transaction=self.exchange_transaction + other.exchange_transaction,
            sebi_turnover_fee=self.sebi_turnover_fee + other.sebi_turnover_fee,
            stamp_duty=self.stamp_duty + other.stamp_duty,
            gst=self.gst + other.gst,
            other=self.other + other.other,
        )

    def as_dict(self) -> dict[str, float]:
        """Every component plus the total, for the ledger and the research log."""

        out = {f.name: float(getattr(self, f.name)) for f in fields(self)}
        out["total"] = self.total
        return out

    def rounded(self, places: int = 2) -> CostBreakdown:
        """Round every component to the paisa, half-up, as a contract note does.

        Deliberately not :func:`round`. Python rounds half to *even* and, worse,
        ``round(7.425, 2)`` yields ``7.42`` because 7.425 is not exactly
        representable in binary — while a broker charges ₹7.43. Using
        ``Decimal`` with ``ROUND_HALF_UP`` makes the charge reproducible by a human
        with a calculator, which is what the "matches to the paisa" test asserts.
        """

        return CostBreakdown(
            **{
                f.name: _round_half_up(float(getattr(self, f.name)), places)
                for f in fields(self)
            }
        )

    def is_zero(self) -> bool:
        return self.total == 0.0


def _round_half_up(value: float, places: int = 2) -> float:
    """Round half away from zero, the convention on a contract note."""

    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP))


@dataclass(frozen=True, slots=True)
class Trade:
    """The transaction a cost model is asked to price."""

    security_id: str
    side: Side
    quantity: float
    price: float
    segment: Segment = Segment.DELIVERY

    @property
    def turnover(self) -> float:
        return self.quantity * self.price


class CostModel(ABC):
    """Interface every cost model implements.

    Deliberately narrow: given a trade, return the itemised charges. A cost model
    has no access to the portfolio, the strategy, or the calendar, so it cannot
    become a place where behaviour hides.
    """

    #: Stable identifier written to ``research_log.cost_model``. Every reported
    #: Sharpe carries it, because a Sharpe under zero costs and a Sharpe under
    #: Indian statutory costs are different quantities.
    name: str = "abstract"

    @abstractmethod
    def charges(self, trade: Trade) -> CostBreakdown:
        """Itemised charges for ``trade``."""

    def describe(self) -> str:
        """One-line description for reports."""

        return self.name


class ZeroCostModel(CostModel):
    """No costs at all. **Validation tests only.**

    Its purpose is diagnostic: under zero costs a buy-and-hold backtest must
    reproduce the benchmark's return to floating-point tolerance. If it does not,
    the engine's accounting is broken and no amount of realistic cost modelling
    will save the numbers. Any *result* produced under this model is marked as
    such in the research log and carries a warning in reports.
    """

    name = "ZeroCostModel"

    def charges(self, trade: Trade) -> CostBreakdown:
        return CostBreakdown()

    def describe(self) -> str:
        return "ZeroCostModel (no costs — validation only, not a tradeable result)"


@dataclass(frozen=True)
class GenericBpsModel(CostModel):
    """Configurable basis points per side plus a fixed per-order charge.

    The conventional simplification. Useful for sensitivity analysis — sweeping
    ``bps_per_side`` shows the cost level at which an edge dies — but it cannot
    reproduce the Indian asymmetry where the buy side pays stamp duty and the sell
    side does not.
    """

    bps_per_side: float = 10.0
    fixed_per_order: float = 0.0
    name: str = "GenericBpsModel"

    def charges(self, trade: Trade) -> CostBreakdown:
        proportional = trade.turnover * self.bps_per_side / 10_000.0
        return CostBreakdown(other=proportional + self.fixed_per_order)

    def describe(self) -> str:
        return (
            f"GenericBpsModel ({self.bps_per_side:g} bps/side"
            f"{f' + ₹{self.fixed_per_order:g}/order' if self.fixed_per_order else ''})"
        )


class BrokerageModel(ABC):
    """Brokerage is the one negotiable component, so it is parameterised."""

    @abstractmethod
    def charge(self, trade: Trade) -> float:
        """Brokerage payable on ``trade``."""

    @abstractmethod
    def describe(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ZeroDeliveryBrokerage(BrokerageModel):
    """Discount broker: free delivery, capped percentage intraday.

    The Zerodha/Groww structure and the realistic default for a delivery-based
    Indian equity strategy.
    """

    intraday_rate: float = DISCOUNT_BROKERAGE_RATE
    cap: float = DISCOUNT_BROKERAGE_CAP

    def charge(self, trade: Trade) -> float:
        if trade.segment is Segment.DELIVERY:
            return 0.0
        return min(trade.turnover * self.intraday_rate, self.cap)

    def describe(self) -> str:
        return (
            f"zero-delivery discount broker (intraday {self.intraday_rate * 100:g}% "
            f"capped at ₹{self.cap:g})"
        )


@dataclass(frozen=True, slots=True)
class PercentCappedBrokerage(BrokerageModel):
    """Percentage of turnover, capped per order — the 0.03% / ₹20 structure."""

    rate: float = DISCOUNT_BROKERAGE_RATE
    cap: float = DISCOUNT_BROKERAGE_CAP

    def charge(self, trade: Trade) -> float:
        return min(trade.turnover * self.rate, self.cap)

    def describe(self) -> str:
        return f"{self.rate * 100:g}% of turnover capped at ₹{self.cap:g} per order"


@dataclass(frozen=True)
class IndianEquityCostModel(CostModel):
    """Every statutory Indian equity charge, computed and reported separately.

    Composition, for a turnover ``T``:

    ==========================  ==========================================
    Component                   Basis
    ==========================  ==========================================
    Brokerage                   per :attr:`brokerage` (free delivery by default)
    STT                         0.1% × T both sides (delivery);
                                0.025% × T sell side only (intraday)
    NSE transaction charge      0.00297% × T
    SEBI turnover fee           ₹10 per crore = 1e-6 × T
    Stamp duty                  0.015% × T, **buy side only**
    GST                         18% × (brokerage + exchange charges)
    ==========================  ==========================================

    "Exchange charges" for the GST base means the NSE transaction charge plus the
    SEBI turnover fee — the convention every Indian broker's contract note uses.
    GST is *not* levied on STT or stamp duty, which are taxes rather than services.

    Components are rounded to the paisa individually before summing, as a real
    contract note does, so a hand-computed charge matches exactly rather than
    approximately.
    """

    brokerage: BrokerageModel = field(default_factory=ZeroDeliveryBrokerage)
    segment: Segment = Segment.DELIVERY
    stt_delivery_rate: float = STT_DELIVERY_RATE
    stt_intraday_rate: float = STT_INTRADAY_RATE
    nse_transaction_rate: float = NSE_TRANSACTION_RATE
    sebi_turnover_rate: float = SEBI_TURNOVER_RATE
    stamp_duty_rate: float = STAMP_DUTY_RATE
    gst_rate: float = GST_RATE
    #: Round each component to the paisa. Off only for analytic comparisons.
    round_to_paisa: bool = True
    name: str = "IndianEquityCostModel"

    def charges(self, trade: Trade) -> CostBreakdown:
        # A trade's own segment wins over the model default, so one model can
        # price a mixed book.
        segment = trade.segment if trade.segment is not None else self.segment
        turnover = trade.turnover
        is_buy = trade.side is Side.BUY

        brokerage = self.brokerage.charge(
            replace(trade, segment=segment) if trade.segment is not segment else trade
        )

        if segment is Segment.DELIVERY:
            stt = turnover * self.stt_delivery_rate
        else:
            stt = 0.0 if is_buy else turnover * self.stt_intraday_rate

        exchange_transaction = turnover * self.nse_transaction_rate
        sebi_fee = turnover * self.sebi_turnover_rate
        stamp_duty = turnover * self.stamp_duty_rate if is_buy else 0.0

        if self.round_to_paisa:
            # A contract note rounds each charge, then levies GST on the rounded
            # figures. Computing GST on unrounded values and rounding afterwards
            # can differ by a paisa, which would make the hand-computed test
            # ambiguous — so the order of operations is fixed here to match.
            brokerage = _round_half_up(brokerage)
            stt = _round_half_up(stt)
            exchange_transaction = _round_half_up(exchange_transaction)
            sebi_fee = _round_half_up(sebi_fee)
            stamp_duty = _round_half_up(stamp_duty)
            gst = _round_half_up(
                self.gst_rate * (brokerage + exchange_transaction + sebi_fee)
            )
        else:
            gst = self.gst_rate * (brokerage + exchange_transaction + sebi_fee)

        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            exchange_transaction=exchange_transaction,
            sebi_turnover_fee=sebi_fee,
            stamp_duty=stamp_duty,
            gst=gst,
        )

    def describe(self) -> str:
        return (
            f"IndianEquityCostModel ({self.segment.value}, "
            f"brokerage: {self.brokerage.describe()})"
        )

    def round_trip_cost_fraction(
        self, price: float = 100.0, quantity: float = 1000.0
    ) -> float:
        """Cost of a round trip as a fraction of notional, for reports.

        Answers "what does one in-and-out cost me?" — the number that decides
        whether a weekly-rebalanced strategy can possibly work.
        """

        buy = self.charges(Trade("X", Side.BUY, quantity, price, self.segment))
        sell = self.charges(Trade("X", Side.SELL, quantity, price, self.segment))
        return (buy.total + sell.total) / (price * quantity)


# --- Slippage ----------------------------------------------------------------


class SlippageModel(ABC):
    """Adverse price movement between the decision and the fill.

    Separate from costs because it is a *price* effect, not a charge: it changes
    what you paid per share, which changes the position's cost basis. Conflating
    the two makes cost attribution wrong.
    """

    name: str = "abstract"

    @abstractmethod
    def fill_price(
        self,
        *,
        reference_price: float,
        side: Side,
        order_quantity: float,
        bar_volume: float,
    ) -> float:
        """The executed price given the bar's reference (open) price."""

    def describe(self) -> str:
        return self.name


class ZeroSlippage(SlippageModel):
    """Fill exactly at the reference price. Validation only."""

    name = "ZeroSlippage"

    def fill_price(
        self,
        *,
        reference_price: float,
        side: Side,
        order_quantity: float,
        bar_volume: float,
    ) -> float:
        return reference_price


@dataclass(frozen=True)
class FixedBpsSlippage(SlippageModel):
    """A fixed number of basis points against the trade, always adverse."""

    bps: float = 5.0
    name: str = "FixedBpsSlippage"

    def fill_price(
        self,
        *,
        reference_price: float,
        side: Side,
        order_quantity: float,
        bar_volume: float,
    ) -> float:
        drift = reference_price * self.bps / 10_000.0
        return reference_price + drift if side is Side.BUY else reference_price - drift

    def describe(self) -> str:
        return f"FixedBpsSlippage ({self.bps:g} bps, always adverse)"


@dataclass(frozen=True)
class VolumeShareSlippage(SlippageModel):
    """Slippage that grows with the share of the bar's volume being demanded.

    ``impact = coefficient × (order_quantity / bar_volume)²``, the square-root-law
    alternative most practitioners use in its quadratic form for small
    participation rates. Larger orders move the price against themselves, which is
    why a strategy that looks good on ₹1 crore may not scale to ₹100 crore.
    """

    coefficient: float = 0.1
    name: str = "VolumeShareSlippage"

    def fill_price(
        self,
        *,
        reference_price: float,
        side: Side,
        order_quantity: float,
        bar_volume: float,
    ) -> float:
        if bar_volume <= 0:
            return reference_price
        participation = order_quantity / bar_volume
        impact = reference_price * self.coefficient * participation**2
        return (
            reference_price + impact if side is Side.BUY else reference_price - impact
        )

    def describe(self) -> str:
        return (
            f"VolumeShareSlippage (coefficient {self.coefficient:g}, "
            "quadratic in participation)"
        )


#: The three required models, by name, for the cost-sensitivity sweep.
def default_cost_models() -> dict[str, CostModel]:
    """The three required cost models (PRD Module 8), keyed by name."""

    return {
        "ZeroCostModel": ZeroCostModel(),
        "GenericBpsModel": GenericBpsModel(bps_per_side=10.0),
        "IndianEquityCostModel": IndianEquityCostModel(),
    }
