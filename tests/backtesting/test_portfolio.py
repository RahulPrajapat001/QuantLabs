"""Portfolio: sizing, risk limits, and T+1 capital availability."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quantlab.backtesting.costs import CostBreakdown
from quantlab.backtesting.events import Direction, Fill, Side, Signal
from quantlab.backtesting.portfolio import (
    Portfolio,
    RiskLimits,
    SizingMethod,
)

DAY = date(2016, 1, 4)


def _signal(
    security_id: str, direction: Direction = Direction.LONG, strength: float = 1.0
) -> Signal:
    return Signal(
        date=DAY,
        security_id=security_id,
        direction=direction,
        strategy_name="test",
        strength=strength,
    )


def _fill(
    security_id: str,
    side: Side,
    quantity: float,
    price: float,
    *,
    costs: float = 0.0,
    on: date = DAY,
) -> Fill:
    return Fill(
        date=on,
        security_id=security_id,
        side=side,
        quantity=quantity,
        price=price,
        costs=CostBreakdown(other=costs),
        reference_price=price,
    )


# --- T+1 settlement ----------------------------------------------------------


def test_sale_proceeds_are_not_spendable_until_they_settle() -> None:
    """The point of modelling T+1: capital availability, not just fill timing.

    A backtest that recycles sale proceeds into a same-day purchase is running on
    credit nobody extended, and the error compounds with turnover.
    """

    portfolio = Portfolio(initial_capital=100_000.0, settlement_days=1)
    portfolio.apply_fill(_fill("A", Side.BUY, 500, 100.0))
    assert portfolio.cash == pytest.approx(50_000.0)

    portfolio.apply_fill(_fill("A", Side.SELL, 500, 110.0))
    # ₹55,000 received but not yet spendable.
    assert portfolio.cash == pytest.approx(50_000.0)
    assert portfolio.unsettled_cash == pytest.approx(55_000.0)
    # Equity still counts it — the money exists, it just cannot be deployed.
    assert portfolio.equity({}) == pytest.approx(105_000.0)

    portfolio.settle(DAY)  # same day: nothing releases
    assert portfolio.cash == pytest.approx(50_000.0)

    released = portfolio.settle(DAY + timedelta(days=1))
    assert released == pytest.approx(55_000.0)
    assert portfolio.cash == pytest.approx(105_000.0)
    assert portfolio.unsettled_cash == 0.0


def test_unsettled_cash_cannot_fund_a_new_position() -> None:
    portfolio = Portfolio(initial_capital=100_000.0, settlement_days=1)
    portfolio.apply_fill(_fill("A", Side.BUY, 1000, 100.0))
    assert portfolio.cash == pytest.approx(0.0)
    portfolio.apply_fill(_fill("A", Side.SELL, 1000, 100.0))

    orders = portfolio.generate_orders([_signal("B")], {"B": 50.0}, on=DAY)
    assert orders == [], "settled cash is zero, so no entry is possible today"


def test_longer_settlement_cycles_are_configurable() -> None:
    portfolio = Portfolio(initial_capital=100_000.0, settlement_days=2)
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0))
    portfolio.apply_fill(_fill("A", Side.SELL, 100, 100.0))
    assert portfolio.settle(DAY + timedelta(days=1)) == 0.0
    assert portfolio.settle(DAY + timedelta(days=2)) > 0.0


# --- Sizing ------------------------------------------------------------------


def test_equal_weight_splits_the_budget_across_signals() -> None:
    portfolio = Portfolio(
        initial_capital=100_000.0,
        sizing=SizingMethod.EQUAL_WEIGHT,
        limits=RiskLimits(max_positions=4, cash_reserve=0.0, max_position_weight=1.0),
        target_positions=4,
    )
    signals = [_signal(s) for s in ("A", "B", "C", "D")]
    prices = dict.fromkeys("ABCD", 100.0)
    orders = portfolio.generate_orders(signals, prices, on=DAY)

    assert len(orders) == 4
    for order in orders:
        # ₹25,000 each at ₹100 = 250 shares.
        assert order.quantity == 250.0
        assert order.notional == pytest.approx(25_000.0)


def test_fewer_signals_than_the_target_still_deploy_capital() -> None:
    """Two signals against a cap of ten should not deploy only 20%."""

    portfolio = Portfolio(
        initial_capital=100_000.0,
        limits=RiskLimits(max_positions=10, cash_reserve=0.0, max_position_weight=1.0),
        target_positions=2,
    )
    orders = portfolio.generate_orders(
        [_signal("A"), _signal("B")], {"A": 100.0, "B": 100.0}, on=DAY
    )
    assert sum(o.notional or 0.0 for o in orders) == pytest.approx(100_000.0)


def test_signal_weighted_sizing_follows_conviction() -> None:
    portfolio = Portfolio(
        initial_capital=100_000.0,
        sizing=SizingMethod.SIGNAL_WEIGHTED,
        limits=RiskLimits(max_positions=2, cash_reserve=0.0, max_position_weight=1.0),
    )
    orders = portfolio.generate_orders(
        [_signal("A", strength=0.75), _signal("B", strength=0.25)],
        {"A": 100.0, "B": 100.0},
        on=DAY,
    )
    by_id = {o.security_id: o for o in orders}
    assert by_id["A"].notional == pytest.approx(75_000.0)
    assert by_id["B"].notional == pytest.approx(25_000.0)


def test_inverse_volatility_sizing_favours_the_calmer_name() -> None:
    portfolio = Portfolio(
        initial_capital=100_000.0,
        sizing=SizingMethod.INVERSE_VOLATILITY,
        limits=RiskLimits(max_positions=2, cash_reserve=0.0, max_position_weight=1.0),
    )
    orders = portfolio.generate_orders(
        [_signal("CALM"), _signal("WILD")],
        {"CALM": 100.0, "WILD": 100.0},
        on=DAY,
        volatilities={"CALM": 0.01, "WILD": 0.03},
    )
    by_id = {o.security_id: o.notional or 0.0 for o in orders}
    assert by_id["CALM"] == pytest.approx(75_000.0)  # 1/0.01 vs 1/0.03 → 3:1
    assert by_id["WILD"] == pytest.approx(25_000.0)


def test_whole_share_sizing_floors_the_quantity() -> None:
    portfolio = Portfolio(
        initial_capital=10_000.0,
        fractional_shares=False,
        limits=RiskLimits(max_positions=1, cash_reserve=0.0, max_position_weight=1.0),
        target_positions=1,
    )
    orders = portfolio.generate_orders([_signal("A")], {"A": 333.0}, on=DAY)
    assert orders[0].quantity == 30.0  # floor(10000/333)
    assert orders[0].allow_fractional is False


# --- Risk limits -------------------------------------------------------------


def test_position_weight_cap_binds() -> None:
    portfolio = Portfolio(
        initial_capital=100_000.0,
        limits=RiskLimits(max_position_weight=0.05, max_positions=1, cash_reserve=0.0),
        target_positions=1,
    )
    orders = portfolio.generate_orders([_signal("A")], {"A": 100.0}, on=DAY)
    # Capped at 5% of equity even though one signal would otherwise take it all.
    assert orders[0].notional == pytest.approx(5_000.0)


def test_max_positions_limits_new_entries_by_conviction() -> None:
    portfolio = Portfolio(
        initial_capital=100_000.0,
        sizing=SizingMethod.SIGNAL_WEIGHTED,
        limits=RiskLimits(max_positions=2, cash_reserve=0.0, max_position_weight=1.0),
    )
    signals = [
        _signal("LOW", strength=0.1),
        _signal("HIGH", strength=0.9),
        _signal("MID", strength=0.5),
    ]
    orders = portfolio.generate_orders(
        signals, dict.fromkeys(("LOW", "HIGH", "MID"), 100.0), on=DAY
    )
    assert {o.security_id for o in orders} == {"HIGH", "MID"}


def test_no_new_entries_once_the_position_limit_is_full() -> None:
    portfolio = Portfolio(
        initial_capital=100_000.0, limits=RiskLimits(max_positions=1, cash_reserve=0.0)
    )
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0))
    orders = portfolio.generate_orders([_signal("B")], {"A": 100.0, "B": 100.0}, on=DAY)
    assert orders == []


def test_cash_reserve_is_withheld() -> None:
    portfolio = Portfolio(
        initial_capital=100_000.0,
        limits=RiskLimits(max_positions=1, cash_reserve=0.10, max_position_weight=1.0),
        target_positions=1,
    )
    orders = portfolio.generate_orders([_signal("A")], {"A": 100.0}, on=DAY)
    assert orders[0].notional == pytest.approx(90_000.0)


def test_gross_exposure_cap_prevents_leverage() -> None:
    portfolio = Portfolio(
        initial_capital=100_000.0,
        limits=RiskLimits(
            max_positions=5,
            max_gross_exposure=1.0,
            cash_reserve=0.0,
            max_position_weight=1.0,
        ),
    )
    portfolio.apply_fill(_fill("A", Side.BUY, 1000, 100.0))  # fully invested
    portfolio.apply_fill(_fill("A", Side.SELL, 1000, 100.0))
    portfolio.settle(DAY + timedelta(days=1))
    portfolio.apply_fill(_fill("A", Side.BUY, 1000, 100.0))
    orders = portfolio.generate_orders([_signal("B")], {"A": 100.0, "B": 100.0}, on=DAY)
    assert orders == []


def test_orders_below_the_minimum_value_are_dropped() -> None:
    portfolio = Portfolio(
        initial_capital=10_000.0,
        limits=RiskLimits(max_positions=100, cash_reserve=0.0, min_order_value=5_000.0),
        target_positions=100,
    )
    orders = portfolio.generate_orders([_signal("A")], {"A": 100.0}, on=DAY)
    assert orders == [], "a ₹100 slice is noise once costs are paid"


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_position_weight"):
        RiskLimits(max_position_weight=1.5)
    with pytest.raises(ValueError, match="max_positions"):
        RiskLimits(max_positions=0)
    with pytest.raises(ValueError, match="cash_reserve"):
        RiskLimits(cash_reserve=1.0)


# --- Exits and repeats -------------------------------------------------------


def test_an_exit_signal_sells_the_whole_position() -> None:
    portfolio = Portfolio(initial_capital=100_000.0)
    portfolio.apply_fill(_fill("A", Side.BUY, 250, 100.0))
    orders = portfolio.generate_orders(
        [_signal("A", Direction.EXIT)], {"A": 105.0}, on=DAY
    )
    assert len(orders) == 1
    assert orders[0].side is Side.SELL
    assert orders[0].quantity == 250.0
    assert orders[0].reason == "exit"


def test_an_exit_signal_with_no_position_does_nothing() -> None:
    portfolio = Portfolio(initial_capital=100_000.0)
    assert (
        portfolio.generate_orders([_signal("A", Direction.EXIT)], {"A": 100.0}, on=DAY)
        == []
    )


def test_a_repeated_long_signal_is_a_hold_not_a_pyramid() -> None:
    portfolio = Portfolio(initial_capital=100_000.0)
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0))
    assert portfolio.generate_orders([_signal("A")], {"A": 100.0}, on=DAY) == []


def test_an_accumulate_signal_adds_to_an_existing_position() -> None:
    """What makes dollar-cost averaging actually average in.

    Without this, a periodic contribution to a name already held is swallowed by
    the no-pyramiding rule and DCA silently degenerates into buy-and-hold.
    """

    portfolio = Portfolio(
        initial_capital=100_000.0,
        limits=RiskLimits(max_positions=1, cash_reserve=0.0, max_position_weight=1.0),
        target_positions=1,
    )
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0))
    signal = Signal(
        date=DAY,
        security_id="A",
        direction=Direction.LONG,
        strategy_name="dca",
        accumulate=True,
    )
    orders = portfolio.generate_orders([signal], {"A": 100.0}, on=DAY)
    assert len(orders) == 1
    assert orders[0].side is Side.BUY


def test_an_add_is_capped_by_the_remaining_position_headroom() -> None:
    """The weight cap applies to the resulting position, not to each contribution.

    Otherwise a monthly top-up would restore the position to the cap every month
    and the limit would never bind.
    """

    portfolio = Portfolio(
        initial_capital=100_000.0,
        limits=RiskLimits(
            max_position_weight=0.20,
            max_positions=5,
            cash_reserve=0.0,
            min_order_value=0.0,
        ),
        target_positions=5,
    )
    # Already holding ₹15,000 of A against a ₹20,000 cap.
    portfolio.apply_fill(_fill("A", Side.BUY, 150, 100.0))
    signal = Signal(
        date=DAY,
        security_id="A",
        direction=Direction.LONG,
        strategy_name="dca",
        accumulate=True,
    )
    orders = portfolio.generate_orders([signal], {"A": 100.0}, on=DAY)
    assert len(orders) == 1
    # Only the ₹5,000 of headroom, not another full 20% slice.
    assert orders[0].notional == pytest.approx(5_000.0)


def test_an_add_consumes_no_position_slot() -> None:
    """A contribution to a held name is not blocked by a full position book."""

    portfolio = Portfolio(
        initial_capital=100_000.0,
        limits=RiskLimits(max_positions=1, cash_reserve=0.0, max_position_weight=1.0),
        target_positions=1,
    )
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0))
    adds = Signal(
        date=DAY,
        security_id="A",
        direction=Direction.LONG,
        strategy_name="dca",
        accumulate=True,
    )
    new_name = Signal(
        date=DAY,
        security_id="B",
        direction=Direction.LONG,
        strategy_name="dca",
        accumulate=True,
    )
    orders = portfolio.generate_orders(
        [adds, new_name], {"A": 100.0, "B": 100.0}, on=DAY
    )
    # A gets its add; B is refused because the single slot is taken.
    assert {o.security_id for o in orders} == {"A"}


def test_a_short_signal_closes_a_long_and_is_otherwise_ignored() -> None:
    """Long-only: shorting the cash segment is not available to this strategy type."""

    portfolio = Portfolio(initial_capital=100_000.0)
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0))
    orders = portfolio.generate_orders(
        [_signal("A", Direction.SHORT), _signal("B", Direction.SHORT)],
        {"A": 100.0, "B": 100.0},
        on=DAY,
    )
    assert len(orders) == 1
    assert orders[0].security_id == "A"
    assert orders[0].side is Side.SELL


# --- P&L accounting ----------------------------------------------------------


def test_cost_basis_includes_acquisition_costs() -> None:
    """So realised P&L is net, not flattering."""

    portfolio = Portfolio(initial_capital=100_000.0)
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0, costs=500.0))
    position = portfolio.positions["A"]
    assert position.cost_basis == pytest.approx(105.0)  # (10,000 + 500) / 100


def test_realised_pnl_is_net_of_both_legs() -> None:
    portfolio = Portfolio(initial_capital=100_000.0)
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0, costs=100.0))
    realised = portfolio.apply_fill(_fill("A", Side.SELL, 100, 110.0, costs=100.0))
    # Gross gain 1,000, minus 100 entry and 100 exit costs.
    assert realised == pytest.approx(800.0)
    assert portfolio.realised_pnl == pytest.approx(800.0)


def test_averaging_up_blends_the_cost_basis() -> None:
    portfolio = Portfolio(initial_capital=100_000.0)
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0))
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 120.0))
    assert portfolio.positions["A"].quantity == 200.0
    assert portfolio.positions["A"].cost_basis == pytest.approx(110.0)


def test_a_closed_position_is_cleared() -> None:
    portfolio = Portfolio(initial_capital=100_000.0)
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0))
    portfolio.apply_fill(_fill("A", Side.SELL, 100, 100.0))
    assert portfolio.open_positions() == {}
    assert portfolio.positions["A"].cost_basis == 0.0


def test_liquidation_applies_no_trading_costs() -> None:
    """A delisting is a corporate action, not a trade the strategy chose."""

    portfolio = Portfolio(initial_capital=100_000.0)
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0))
    realised = portfolio.liquidate("A", 90.0, DAY)
    assert realised == pytest.approx(-1_000.0)
    assert portfolio.open_positions() == {}
    assert portfolio.unsettled_cash == pytest.approx(9_000.0)


def test_liquidating_nothing_is_harmless() -> None:
    portfolio = Portfolio(initial_capital=100_000.0)
    assert portfolio.liquidate("MISSING", 100.0, DAY) == 0.0


def test_reset_restores_the_starting_state() -> None:
    portfolio = Portfolio(initial_capital=100_000.0)
    portfolio.apply_fill(_fill("A", Side.BUY, 100, 100.0))
    portfolio.apply_fill(_fill("A", Side.SELL, 100, 120.0))
    portfolio.reset()
    assert portfolio.cash == 100_000.0
    assert portfolio.positions == {}
    assert portfolio.unsettled_cash == 0.0
    assert portfolio.realised_pnl == 0.0


def test_equity_marks_positions_at_supplied_prices() -> None:
    portfolio = Portfolio(initial_capital=100_000.0)
    portfolio.apply_fill(_fill("A", Side.BUY, 500, 100.0))
    assert portfolio.equity({"A": 120.0}) == pytest.approx(50_000.0 + 60_000.0)
    assert portfolio.gross_exposure({"A": 120.0}) == pytest.approx(60_000.0)


def test_zero_capital_is_rejected() -> None:
    with pytest.raises(ValueError, match="initial_capital"):
        Portfolio(initial_capital=0.0)
