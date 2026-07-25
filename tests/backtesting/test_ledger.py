"""Ledger: FIFO round-trip matching, cost attribution, and the daily record."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quantlab.backtesting.costs import CostBreakdown
from quantlab.backtesting.events import (
    Fill,
    Order,
    RejectedOrder,
    RejectionReason,
    Side,
)
from quantlab.backtesting.ledger import DailyMark, Ledger

START = date(2016, 1, 4)


def _fill(
    side: Side,
    quantity: float,
    price: float,
    *,
    day: int = 0,
    costs: float = 0.0,
    security_id: str = "A",
) -> Fill:
    return Fill(
        date=START + timedelta(days=day),
        security_id=security_id,
        side=side,
        quantity=quantity,
        price=price,
        costs=CostBreakdown(other=costs),
        reference_price=price,
    )


def _mark(day: int, equity: float) -> DailyMark:
    return DailyMark(
        date=START + timedelta(days=day),
        cash=equity,
        unsettled_cash=0.0,
        holdings_value=0.0,
        equity=equity,
        positions=0,
        gross_exposure=0.0,
    )


# --- FIFO round trips --------------------------------------------------------


def test_a_buy_then_sell_closes_one_round_trip_net_of_costs() -> None:
    ledger = Ledger()
    ledger.record_fill(_fill(Side.BUY, 100, 100.0, day=0, costs=50.0))
    ledger.record_fill(_fill(Side.SELL, 100, 110.0, day=5, costs=50.0))

    assert len(ledger.closed_trades) == 1
    trade = ledger.closed_trades[0]
    assert trade.quantity == 100
    assert trade.entry_price == 100.0
    assert trade.exit_price == 110.0
    # Gross 1,000 less 100 of costs across both legs.
    assert trade.pnl == pytest.approx(900.0)
    assert trade.costs == pytest.approx(100.0)
    assert trade.is_win
    assert trade.holding_days == 5
    assert trade.return_pct == pytest.approx(0.09)


def test_lots_close_oldest_first() -> None:
    """FIFO, so the ₹100 lot closes before the ₹120 one."""

    ledger = Ledger()
    ledger.record_fill(_fill(Side.BUY, 100, 100.0, day=0))
    ledger.record_fill(_fill(Side.BUY, 100, 120.0, day=1))
    ledger.record_fill(_fill(Side.SELL, 100, 130.0, day=2))

    assert len(ledger.closed_trades) == 1
    assert ledger.closed_trades[0].entry_price == 100.0
    assert ledger.closed_trades[0].pnl == pytest.approx(3_000.0)


def test_one_sale_can_close_several_lots() -> None:
    ledger = Ledger()
    ledger.record_fill(_fill(Side.BUY, 100, 100.0, day=0))
    ledger.record_fill(_fill(Side.BUY, 100, 120.0, day=1))
    ledger.record_fill(_fill(Side.SELL, 200, 130.0, day=2))

    assert len(ledger.closed_trades) == 2
    assert [t.entry_price for t in ledger.closed_trades] == [100.0, 120.0]
    assert sum(t.pnl for t in ledger.closed_trades) == pytest.approx(4_000.0)


def test_a_partial_sale_leaves_the_remainder_open() -> None:
    ledger = Ledger()
    ledger.record_fill(_fill(Side.BUY, 100, 100.0, day=0, costs=100.0))
    ledger.record_fill(_fill(Side.SELL, 40, 110.0, day=1, costs=40.0))

    assert len(ledger.closed_trades) == 1
    trade = ledger.closed_trades[0]
    assert trade.quantity == 40
    # Entry costs are apportioned: 40% of ₹100, plus the ₹40 exit cost.
    assert trade.costs == pytest.approx(80.0)
    assert trade.pnl == pytest.approx(400.0 - 80.0)

    # Selling the rest carries the remaining 60% of entry costs.
    ledger.record_fill(_fill(Side.SELL, 60, 110.0, day=2, costs=60.0))
    assert len(ledger.closed_trades) == 2
    assert ledger.closed_trades[1].costs == pytest.approx(120.0)


def test_a_losing_trade_is_recorded_as_a_loss() -> None:
    ledger = Ledger()
    ledger.record_fill(_fill(Side.BUY, 100, 100.0, day=0))
    ledger.record_fill(_fill(Side.SELL, 100, 95.0, day=1))
    assert not ledger.closed_trades[0].is_win
    assert ledger.closed_trades[0].pnl == pytest.approx(-500.0)


def test_costs_alone_can_turn_a_gross_gain_into_a_net_loss() -> None:
    """The Indian-costs reality for a small move on a high-turnover trade."""

    ledger = Ledger()
    ledger.record_fill(_fill(Side.BUY, 100, 100.0, day=0, costs=150.0))
    ledger.record_fill(_fill(Side.SELL, 100, 101.0, day=1, costs=150.0))
    trade = ledger.closed_trades[0]
    assert trade.exit_price > trade.entry_price, "gross move is positive"
    assert trade.pnl < 0, "net of costs it is a loss"
    assert not trade.is_win


def test_positions_in_different_securities_are_matched_separately() -> None:
    ledger = Ledger()
    ledger.record_fill(_fill(Side.BUY, 100, 100.0, day=0, security_id="A"))
    ledger.record_fill(_fill(Side.BUY, 100, 200.0, day=0, security_id="B"))
    ledger.record_fill(_fill(Side.SELL, 100, 110.0, day=1, security_id="B"))

    assert len(ledger.closed_trades) == 1
    assert ledger.closed_trades[0].security_id == "B"
    assert ledger.closed_trades[0].entry_price == 200.0


# --- Cost attribution --------------------------------------------------------


def test_costs_accumulate_by_component() -> None:
    ledger = Ledger()
    ledger.record_fill(
        Fill(
            date=START,
            security_id="A",
            side=Side.BUY,
            quantity=100,
            price=2500.0,
            costs=CostBreakdown(stt=250.0, stamp_duty=37.5, gst=1.38),
            reference_price=2500.0,
        )
    )
    ledger.record_fill(
        Fill(
            date=START + timedelta(days=1),
            security_id="A",
            side=Side.SELL,
            quantity=100,
            price=2500.0,
            costs=CostBreakdown(stt=250.0, gst=1.38),
            reference_price=2500.0,
        )
    )
    attribution = ledger.cost_attribution()
    assert attribution["stt"] == pytest.approx(500.0)
    assert attribution["stamp_duty"] == pytest.approx(37.5)
    assert attribution["total"] == pytest.approx(540.26)
    # Shares make the dominant head obvious.
    assert attribution["stt_share"] == pytest.approx(500.0 / 540.26)


def test_turnover_counts_both_sides() -> None:
    ledger = Ledger()
    ledger.record_fill(_fill(Side.BUY, 100, 100.0, day=0))
    ledger.record_fill(_fill(Side.SELL, 100, 110.0, day=1))
    assert ledger.total_turnover() == pytest.approx(21_000.0)


# --- Rejections --------------------------------------------------------------


def test_rejections_are_counted_by_reason() -> None:
    ledger = Ledger()
    order = Order(date=START, security_id="A", side=Side.BUY, quantity=100)
    ledger.record_rejection(
        RejectedOrder(date=START, order=order, reason=RejectionReason.CIRCUIT_LIMIT)
    )
    ledger.record_rejection(
        RejectedOrder(date=START, order=order, reason=RejectionReason.CIRCUIT_LIMIT)
    )
    ledger.record_rejection(
        RejectedOrder(date=START, order=order, reason=RejectionReason.VOLUME_CAP)
    )
    counts = ledger.rejection_counts()
    assert counts["circuit_limit"] == 2
    assert counts["volume_cap"] == 1
    assert counts["no_bar"] == 0
    frame = ledger.rejections_frame()
    assert len(frame) == 3
    assert set(frame["reason"]) == {"circuit_limit", "volume_cap"}


# --- Marks and views ---------------------------------------------------------


def test_equity_curve_and_returns_come_from_the_marks() -> None:
    ledger = Ledger()
    for i, equity in enumerate((100.0, 110.0, 99.0)):
        ledger.record_mark(_mark(i, equity))
    curve = ledger.equity_curve
    assert list(curve) == [100.0, 110.0, 99.0]
    returns = ledger.returns
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)


def test_an_empty_ledger_yields_empty_series() -> None:
    ledger = Ledger()
    assert ledger.equity_curve.empty
    assert ledger.returns.empty
    assert ledger.cost_attribution()["total"] == 0.0


def test_frames_expose_itemised_costs_per_fill() -> None:
    ledger = Ledger()
    ledger.record_fill(
        Fill(
            date=START,
            security_id="A",
            side=Side.BUY,
            quantity=100,
            price=101.0,
            costs=CostBreakdown(stt=10.0),
            reference_price=100.0,
        )
    )
    frame = ledger.fills_frame()
    assert frame.loc[0, "cost_stt"] == 10.0
    assert frame.loc[0, "cost_total"] == 10.0
    assert frame.loc[0, "slippage_per_share"] == pytest.approx(1.0)


def test_liquidations_are_kept_out_of_closed_trades() -> None:
    """A forced exit at a delisting price is not evidence about strategy quality."""

    ledger = Ledger()
    ledger.record_fill(_fill(Side.BUY, 100, 100.0, day=0))
    ledger.record_liquidation(START + timedelta(days=1), "A", 50.0, -5_000.0)
    assert ledger.closed_trades == []
    assert len(ledger.liquidations) == 1
    # The open lot is cleared, so a later sale cannot double-count the position.
    ledger.record_fill(_fill(Side.SELL, 100, 60.0, day=2))
    assert ledger.closed_trades == [], "the liquidated lot must not be matched again"


def test_reset_clears_everything() -> None:
    ledger = Ledger()
    ledger.record_fill(_fill(Side.BUY, 100, 100.0, costs=10.0))
    ledger.record_mark(_mark(0, 100.0))
    ledger.reset()
    assert ledger.fills == []
    assert ledger.marks == []
    assert ledger.costs.total == 0.0
    assert ledger.closed_trades == []


def test_mark_reports_invested_fraction() -> None:
    mark = DailyMark(
        date=START,
        cash=40.0,
        unsettled_cash=0.0,
        holdings_value=60.0,
        equity=100.0,
        positions=1,
        gross_exposure=60.0,
    )
    assert mark.invested_fraction == pytest.approx(0.6)
