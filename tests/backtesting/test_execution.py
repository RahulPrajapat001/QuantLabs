"""ExecutionHandler — priority test 5: a signal on a band-hit day produces no fill."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantlab.backtesting.costs import (
    IndianEquityCostModel,
    ZeroCostModel,
)
from quantlab.backtesting.data_handler import DataHandler
from quantlab.backtesting.events import Order, RejectionReason, Side
from quantlab.backtesting.execution import (
    CircuitLimitPolicy,
    ExcessPolicy,
    ExecutionHandler,
)

from .conftest import make_flat_bars


def _advance(handler: DataHandler, steps: int) -> None:
    """Move the cursor forward, as the engine does before executing."""

    stream = handler.stream()
    for _ in range(steps):
        next(stream)


def _order(
    *,
    side: Side = Side.BUY,
    quantity: float = 100.0,
    on: date = date(2015, 1, 2),
    security_id: str = "SEC-0000",
) -> Order:
    return Order(date=on, security_id=security_id, side=side, quantity=quantity)


# --- PRIORITY TEST 5 ---------------------------------------------------------


def test_order_on_a_circuit_band_day_receives_no_fill() -> None:
    """A security that gapped to its 20% band cannot be bought at that price.

    This is the test that matters most for momentum and breakout strategies: their
    signals fire on exactly these days, so a naive engine books the returns of
    trades that could never have been executed.
    """

    bars = make_flat_bars(n_days=4)
    # Day 3 opens 25% above day 2's close — beyond the 20% band.
    bars.loc[bars.index[2], ["open", "high", "low", "close"]] = [
        125.0,
        125.0,
        125.0,
        125.0,
    ]
    handler = DataHandler(bars)
    handler_execution = ExecutionHandler(
        cost_model=ZeroCostModel(), max_participation=None
    )

    _advance(handler, 3)  # cursor on the band-hit bar
    fills, rejections = handler_execution.execute(
        [_order()], handler, available_cash=1_000_000.0
    )

    assert fills == []
    assert len(rejections) == 1
    assert rejections[0].reason is RejectionReason.CIRCUIT_LIMIT
    assert "price band" in rejections[0].detail or "zero-range" in rejections[0].detail
    # The order is recorded, not discarded — the rejection is the finding.
    assert rejections[0].order.quantity == 100.0
    assert rejections[0].rejected_quantity == 100.0


def test_vendor_circuit_flag_overrides_inference() -> None:
    """Explicit ``circuit_hit`` data is authoritative in both directions."""

    bars = make_flat_bars(n_days=3)
    bars["circuit_hit"] = [False, False, True]
    handler = DataHandler(bars)
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=None)

    _advance(handler, 3)
    fills, rejections = handler_fills = execution.execute(
        [_order()], handler, available_cash=1_000_000.0
    )
    assert fills == []
    assert rejections[0].reason is RejectionReason.CIRCUIT_LIMIT
    assert "vendor circuit_hit" in rejections[0].detail
    assert handler_fills is not None


def test_a_flagged_normal_day_fills_even_after_a_big_move() -> None:
    """When vendor data says the band was not hit, a large gap still fills.

    Inference cannot distinguish a band lock from an ordinary large gap, so real
    data wins.
    """

    bars = make_flat_bars(n_days=3)
    bars.loc[bars.index[2], ["open", "high", "low", "close"]] = [
        130.0,
        131.0,
        129.0,
        130.0,
    ]
    bars["circuit_hit"] = [False, False, False]
    handler = DataHandler(bars)
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=None)

    _advance(handler, 3)
    fills, rejections = execution.execute(
        [_order()], handler, available_cash=1_000_000.0
    )
    assert len(fills) == 1
    assert rejections == []


def test_zero_range_bar_is_treated_as_locked() -> None:
    """A bar whose high equals its low had no two-sided market."""

    bars = make_flat_bars(n_days=3)
    bars.loc[bars.index[2], ["open", "high", "low", "close"]] = [
        101.0,
        101.0,
        101.0,
        101.0,
    ]
    handler = DataHandler(bars)
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=None)

    _advance(handler, 3)
    fills, rejections = execution.execute(
        [_order()], handler, available_cash=1_000_000.0
    )
    assert fills == []
    assert "zero-range" in rejections[0].detail


def test_band_can_be_disabled_for_unbanded_securities() -> None:
    bars = make_flat_bars(n_days=3)
    bars.loc[bars.index[2], ["open", "high", "low", "close"]] = [
        130.0,
        131.0,
        129.0,
        130.0,
    ]
    handler = DataHandler(bars)
    execution = ExecutionHandler(
        cost_model=ZeroCostModel(),
        max_participation=None,
        circuit_policy=CircuitLimitPolicy(band=None),
    )
    _advance(handler, 3)
    fills, _ = execution.execute([_order()], handler, available_cash=1_000_000.0)
    assert len(fills) == 1


# --- T+1 open execution ------------------------------------------------------


def test_fills_happen_at_the_current_bars_open() -> None:
    """The engine advances the cursor first, so 'current' here is T+1."""

    bars = make_flat_bars(n_days=3, price=100.0)
    bars.loc[bars.index[1], "open"] = 102.0
    handler = DataHandler(bars)
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=None)

    _advance(handler, 2)
    fills, _ = execution.execute([_order()], handler, available_cash=1_000_000.0)
    assert fills[0].price == 102.0
    assert fills[0].reference_price == 102.0
    assert fills[0].order_date == date(2015, 1, 2)
    assert fills[0].date == handler.current_date


def test_execute_before_streaming_raises() -> None:
    handler = DataHandler(make_flat_bars(n_days=3))
    execution = ExecutionHandler(cost_model=ZeroCostModel())
    with pytest.raises(RuntimeError, match="before any bar was served"):
        execution.execute([_order()], handler, available_cash=1_000.0)


# --- Volume participation cap ------------------------------------------------


def test_volume_cap_partially_fills_and_records_the_remainder() -> None:
    """1% of a 1,000,000-share bar is 10,000 shares; the rest is rejected."""

    handler = DataHandler(make_flat_bars(n_days=3, volume=1_000_000.0))
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=0.01)

    _advance(handler, 2)
    fills, rejections = execution.execute(
        [_order(quantity=25_000.0)], handler, available_cash=100_000_000.0
    )
    assert len(fills) == 1
    assert fills[0].quantity == 10_000.0
    assert len(rejections) == 1
    assert rejections[0].reason is RejectionReason.VOLUME_CAP
    assert rejections[0].filled_quantity == 10_000.0
    assert rejections[0].rejected_quantity == 15_000.0
    assert "exceeds 1%" in rejections[0].detail


def test_split_policy_carries_the_remainder_to_the_next_bar() -> None:
    handler = DataHandler(make_flat_bars(n_days=5, volume=1_000_000.0))
    execution = ExecutionHandler(
        cost_model=ZeroCostModel(),
        max_participation=0.01,
        excess_policy=ExcessPolicy.SPLIT,
    )

    _advance(handler, 2)
    fills, _ = execution.execute(
        [_order(quantity=25_000.0)], handler, available_cash=100_000_000.0
    )
    assert fills[0].quantity == 10_000.0
    assert len(execution.carried_orders) == 1
    assert execution.carried_orders[0].quantity == 15_000.0

    # Next bar: the carried remainder is attempted again, capped again.
    _advance(handler, 1)
    fills2, _ = execution.execute([], handler, available_cash=100_000_000.0)
    assert len(fills2) == 1
    assert fills2[0].quantity == 10_000.0
    assert execution.carried_orders[0].quantity == 5_000.0


def test_carried_orders_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remainder cannot wait forever, or an illiquid name builds a hidden backlog."""

    handler = DataHandler(make_flat_bars(n_days=20, volume=1_000.0))
    execution = ExecutionHandler(
        cost_model=ZeroCostModel(),
        max_participation=0.01,
        excess_policy=ExcessPolicy.SPLIT,
        max_carry_days=2,
    )
    _advance(handler, 2)
    execution.execute([_order(quantity=5_000.0)], handler, available_cash=10_000_000.0)
    assert execution.carried_orders

    # Advance well past the carry window.
    _advance(handler, 8)
    execution.execute([], handler, available_cash=10_000_000.0)
    assert execution.carried_orders == []


def test_zero_volume_bar_admits_nothing() -> None:
    bars = make_flat_bars(n_days=3)
    bars.loc[bars.index[1], "volume"] = 0.0
    handler = DataHandler(bars)
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=0.01)
    _advance(handler, 2)
    fills, rejections = execution.execute(
        [_order()], handler, available_cash=1_000_000.0
    )
    assert fills == []
    assert rejections[0].reason is RejectionReason.VOLUME_CAP
    assert "zero volume" in rejections[0].detail


def test_invalid_participation_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="max_participation"):
        ExecutionHandler(cost_model=ZeroCostModel(), max_participation=1.5)


# --- Missing bars, capital, positions ----------------------------------------


def test_missing_bar_is_a_documented_rejection() -> None:
    handler = DataHandler(make_flat_bars(n_days=3))
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=None)
    _advance(handler, 2)
    fills, rejections = execution.execute(
        [_order(security_id="DELISTED")], handler, available_cash=1_000_000.0
    )
    assert fills == []
    assert rejections[0].reason is RejectionReason.NO_BAR
    assert "did not trade" in rejections[0].detail


def test_insufficient_capital_trims_rather_than_overdrawing() -> None:
    handler = DataHandler(make_flat_bars(n_days=3, price=100.0))
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=None)
    _advance(handler, 2)
    # ₹5,000 buys 50 shares at ₹100, not the 100 requested.
    fills, _ = execution.execute(
        [_order(quantity=100.0)], handler, available_cash=5_000.0
    )
    assert fills[0].quantity == 50.0


def test_no_capital_at_all_is_rejected() -> None:
    handler = DataHandler(make_flat_bars(n_days=3))
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=None)
    _advance(handler, 2)
    fills, rejections = execution.execute([_order()], handler, available_cash=0.0)
    assert fills == []
    assert rejections[0].reason is RejectionReason.INSUFFICIENT_CAPITAL


def test_a_sell_cannot_exceed_the_position_held() -> None:
    handler = DataHandler(make_flat_bars(n_days=3))
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=None)
    _advance(handler, 2)
    fills, _ = execution.execute(
        [_order(side=Side.SELL, quantity=500.0)],
        handler,
        available_cash=0.0,
        positions={"SEC-0000": 120.0},
    )
    assert fills[0].quantity == 120.0


def test_selling_nothing_is_rejected() -> None:
    handler = DataHandler(make_flat_bars(n_days=3))
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=None)
    _advance(handler, 2)
    fills, rejections = execution.execute(
        [_order(side=Side.SELL)], handler, available_cash=0.0, positions={}
    )
    assert fills == []
    assert rejections[0].reason is RejectionReason.NO_POSITION


# --- Notional orders ---------------------------------------------------------


def test_notional_order_spends_exactly_its_amount_including_costs() -> None:
    """The mechanism that makes buy-and-hold reproduce its benchmark exactly."""

    handler = DataHandler(make_flat_bars(n_days=3, price=100.0))
    execution = ExecutionHandler(
        cost_model=IndianEquityCostModel(), max_participation=None
    )
    _advance(handler, 2)
    order = Order(
        date=date(2015, 1, 2),
        security_id="SEC-0000",
        side=Side.BUY,
        quantity=1_000.0,
        notional=100_000.0,
        allow_fractional=True,
    )
    fills, _ = execution.execute([order], handler, available_cash=1_000_000.0)
    fill = fills[0]
    spent = fill.gross_value + float(fill.costs.total)
    assert spent == pytest.approx(100_000.0, abs=0.5)
    assert fill.quantity < 1_000.0  # costs consumed part of the notional


def test_notional_too_small_for_one_share_is_rejected() -> None:
    handler = DataHandler(make_flat_bars(n_days=3, price=100.0))
    execution = ExecutionHandler(cost_model=ZeroCostModel(), max_participation=None)
    _advance(handler, 2)
    order = Order(
        date=date(2015, 1, 2),
        security_id="SEC-0000",
        side=Side.BUY,
        quantity=1.0,
        notional=50.0,
        allow_fractional=False,
    )
    fills, rejections = execution.execute([order], handler, available_cash=1_000.0)
    assert fills == []
    assert rejections[0].reason is RejectionReason.ZERO_QUANTITY


# --- Costs and slippage on the fill ------------------------------------------


def test_fill_carries_itemised_costs_and_signed_cash_flow() -> None:
    handler = DataHandler(make_flat_bars(n_days=3, price=2500.0))
    execution = ExecutionHandler(
        cost_model=IndianEquityCostModel(), max_participation=None
    )
    _advance(handler, 2)
    fills, _ = execution.execute(
        [_order(quantity=100.0)], handler, available_cash=1_000_000.0
    )
    fill = fills[0]
    assert fill.costs.stt == 250.0
    assert fill.costs.stamp_duty == 37.50
    # A buy reduces cash by gross plus costs.
    assert fill.cash_flow == pytest.approx(-(250_000.0 + 296.56))

    sells, _ = execution.execute(
        [_order(side=Side.SELL, quantity=100.0)],
        handler,
        available_cash=0.0,
        positions={"SEC-0000": 100.0},
    )
    # A sale credits gross minus costs.
    assert sells[0].cash_flow == pytest.approx(250_000.0 - 259.06)


def test_slippage_is_recorded_against_the_reference_price() -> None:
    from quantlab.backtesting.costs import FixedBpsSlippage

    handler = DataHandler(make_flat_bars(n_days=3, price=100.0))
    execution = ExecutionHandler(
        cost_model=ZeroCostModel(),
        slippage=FixedBpsSlippage(bps=10.0),
        max_participation=None,
    )
    _advance(handler, 2)
    fills, _ = execution.execute([_order()], handler, available_cash=1_000_000.0)
    fill = fills[0]
    assert fill.reference_price == 100.0
    assert fill.price == pytest.approx(100.10)
    assert fill.slippage_per_share == pytest.approx(0.10)


def test_reset_clears_carried_state() -> None:
    handler = DataHandler(make_flat_bars(n_days=5, volume=1_000.0))
    execution = ExecutionHandler(
        cost_model=ZeroCostModel(),
        max_participation=0.01,
        excess_policy=ExcessPolicy.SPLIT,
    )
    _advance(handler, 2)
    execution.execute([_order(quantity=5_000.0)], handler, available_cash=10_000_000.0)
    assert execution.carried_orders
    execution.reset()
    assert execution.carried_orders == []


def test_circuit_policy_handles_a_bar_without_the_optional_column() -> None:
    """Panels without ``circuit_hit`` fall back to inference, not to a crash."""

    policy = CircuitLimitPolicy(band=0.20)
    bar = pd.Series({"open": 130.0, "high": 131.0, "low": 129.0, "close": 130.0})
    locked, reason = policy.is_locked(bar, previous_close=100.0)
    assert locked
    assert "price band" in reason
