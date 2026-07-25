"""Cost models — priority test 4: hand-computed Indian charges, to the paisa."""

from __future__ import annotations

import pytest

from quantlab.backtesting.costs import (
    CostBreakdown,
    FixedBpsSlippage,
    GenericBpsModel,
    IndianEquityCostModel,
    PercentCappedBrokerage,
    Segment,
    Trade,
    VolumeShareSlippage,
    ZeroCostModel,
    ZeroDeliveryBrokerage,
    ZeroSlippage,
    default_cost_models,
)
from quantlab.backtesting.events import Side

# --- PRIORITY TEST 4 ---------------------------------------------------------


def test_indian_delivery_buy_matches_hand_computation_to_the_paisa() -> None:
    """Buy 100 RELIANCE at ₹2,500 delivery. Every charge computed by hand.

    Turnover = 100 × 2,500 = ₹250,000.

    ==========================  =====================================  =========
    Charge                      Computation                            Amount
    ==========================  =====================================  =========
    Brokerage                   zero-delivery discount broker          ₹0.00
    STT                         0.1% × 250,000                         ₹250.00
    NSE transaction             0.00297% × 250,000 = 7.425             ₹7.43
    SEBI turnover fee           ₹10/crore = 1e-6 × 250,000             ₹0.25
    Stamp duty (buy only)       0.015% × 250,000                       ₹37.50
    GST                         18% × (0 + 7.43 + 0.25) = 1.3824       ₹1.38
    ==========================  =====================================  =========
    **Total**                                                          **₹296.56**

    The NSE charge is the one that pins the rounding convention: 7.425 must round
    *up* to 7.43, as a contract note does. Python's ``round`` returns 7.42 here,
    both because it rounds half to even and because 7.425 is not exactly
    representable in binary — hence the ``Decimal``/``ROUND_HALF_UP`` implementation.
    """

    model = IndianEquityCostModel()
    costs = model.charges(Trade("RELIANCE", Side.BUY, 100, 2500.0, Segment.DELIVERY))

    assert costs.brokerage == 0.00
    assert costs.stt == 250.00
    assert costs.exchange_transaction == 7.43
    assert costs.sebi_turnover_fee == 0.25
    assert costs.stamp_duty == 37.50
    assert costs.gst == 1.38
    assert costs.other == 0.00
    assert costs.total == 296.56


def test_indian_delivery_sell_matches_hand_computation_to_the_paisa() -> None:
    """Same trade, sell side: no stamp duty, everything else identical.

    Total = 250.00 + 7.43 + 0.25 + 0.00 + 1.38 = **₹259.06**
    """

    model = IndianEquityCostModel()
    costs = model.charges(Trade("RELIANCE", Side.SELL, 100, 2500.0, Segment.DELIVERY))

    assert costs.stamp_duty == 0.00, "stamp duty is buy-side only"
    assert costs.stt == 250.00, "delivery STT applies to both sides"
    assert costs.total == 259.06


def test_indian_intraday_stt_is_sell_side_only_and_lower() -> None:
    """Intraday: STT is 0.025% and only on the sell side.

    Buy 500 at ₹1,000 → turnover ₹500,000.
      brokerage = min(0.03% × 500,000, ₹20) = min(150, 20) = ₹20.00
      STT       = ₹0.00 (buy side, intraday)
      NSE       = 0.00297% × 500,000 = ₹14.85
      SEBI      = ₹0.50
      stamp     = 0.015% × 500,000 = ₹75.00
      GST       = 18% × (20.00 + 14.85 + 0.50) = 6.363 → ₹6.36
      total     = ₹116.71
    """

    model = IndianEquityCostModel(segment=Segment.INTRADAY)
    buy = model.charges(Trade("TCS", Side.BUY, 500, 1000.0, Segment.INTRADAY))

    assert buy.brokerage == 20.00
    assert buy.stt == 0.00
    assert buy.exchange_transaction == 14.85
    assert buy.sebi_turnover_fee == 0.50
    assert buy.stamp_duty == 75.00
    assert buy.gst == 6.36
    assert buy.total == 116.71

    sell = model.charges(Trade("TCS", Side.SELL, 500, 1000.0, Segment.INTRADAY))
    # 0.025% × 500,000 = ₹125.00
    assert sell.stt == 125.00
    assert sell.stamp_duty == 0.00


def test_gst_is_not_levied_on_stt_or_stamp_duty() -> None:
    """GST applies to brokerage and exchange charges only — they are services.

    STT and stamp duty are taxes; taxing them again would be wrong and would
    overstate costs by roughly 18% of the largest component.
    """

    model = IndianEquityCostModel()
    costs = model.charges(Trade("X", Side.BUY, 100, 2500.0))
    gst_base = costs.brokerage + costs.exchange_transaction + costs.sebi_turnover_fee
    assert costs.gst == pytest.approx(round(0.18 * gst_base, 2), abs=0.01)
    # If STT were in the base, GST would be ~₹46 rather than ₹1.38.
    assert costs.gst < 2.0


def test_round_trip_cost_is_about_22_basis_points() -> None:
    """The number that decides whether a high-turnover strategy can work.

    A delivery round trip costs ~0.22% of notional before slippage. A strategy
    rebalancing weekly pays this ~50 times a year.
    """

    model = IndianEquityCostModel()
    fraction = model.round_trip_cost_fraction(price=100.0, quantity=1000.0)
    assert fraction == pytest.approx(0.00222, abs=0.00005)


def test_costs_scale_linearly_with_turnover() -> None:
    """Charges are levied on turnover, not profit — the structural point."""

    model = IndianEquityCostModel()
    small = model.charges(Trade("X", Side.BUY, 100, 1000.0)).total
    large = model.charges(Trade("X", Side.BUY, 1000, 1000.0)).total
    assert large == pytest.approx(small * 10, rel=0.001)


# --- Brokerage models --------------------------------------------------------


def test_zero_delivery_brokerage_is_free_on_delivery_capped_intraday() -> None:
    brokerage = ZeroDeliveryBrokerage()
    assert brokerage.charge(Trade("X", Side.BUY, 100, 2500.0, Segment.DELIVERY)) == 0.0
    # 0.03% of 250,000 = 75, capped at 20.
    assert brokerage.charge(Trade("X", Side.BUY, 100, 2500.0, Segment.INTRADAY)) == 20.0
    # Small trade below the cap pays the percentage: 0.03% of 10,000 = 3.
    assert brokerage.charge(
        Trade("X", Side.BUY, 10, 1000.0, Segment.INTRADAY)
    ) == pytest.approx(3.0)


def test_percent_capped_brokerage_applies_to_delivery_too() -> None:
    brokerage = PercentCappedBrokerage(rate=0.0003, cap=20.0)
    model = IndianEquityCostModel(brokerage=brokerage)
    costs = model.charges(Trade("X", Side.BUY, 100, 2500.0))
    assert costs.brokerage == 20.00
    # GST now includes brokerage: 18% × (20 + 7.43 + 0.25) = 4.9824 → 4.98
    assert costs.gst == 4.98


# --- The other two required models -------------------------------------------


def test_zero_cost_model_charges_nothing() -> None:
    costs = ZeroCostModel().charges(Trade("X", Side.BUY, 100, 2500.0))
    assert costs.total == 0.0
    assert costs.is_zero()
    assert "validation only" in ZeroCostModel().describe()


def test_generic_bps_model_charges_per_side_plus_fixed() -> None:
    model = GenericBpsModel(bps_per_side=10.0, fixed_per_order=20.0)
    costs = model.charges(Trade("X", Side.BUY, 100, 2500.0))
    # 10 bps of 250,000 = 250, plus 20 fixed.
    assert costs.total == pytest.approx(270.0)
    # Symmetric: unlike the Indian model, no side asymmetry.
    sell = model.charges(Trade("X", Side.SELL, 100, 2500.0))
    assert sell.total == costs.total


def test_all_three_required_models_are_registered() -> None:
    models = default_cost_models()
    assert set(models) == {"ZeroCostModel", "GenericBpsModel", "IndianEquityCostModel"}
    for name, model in models.items():
        assert model.name == name


# --- Breakdown mechanics -----------------------------------------------------


def test_breakdowns_add_componentwise() -> None:
    a = CostBreakdown(brokerage=10.0, stt=250.0)
    b = CostBreakdown(brokerage=5.0, gst=1.8)
    total = a + b
    assert total.brokerage == 15.0
    assert total.stt == 250.0
    assert total.gst == 1.8
    assert total.total == pytest.approx(266.8)


def test_breakdown_as_dict_includes_total() -> None:
    payload = CostBreakdown(stt=250.0).as_dict()
    assert payload["stt"] == 250.0
    assert payload["total"] == 250.0
    assert "brokerage" in payload


def test_rounding_is_half_up_not_bankers() -> None:
    """7.425 → 7.43, and 0.125 → 0.13. Python's round gives 7.42 and 0.12."""

    rounded = CostBreakdown(
        exchange_transaction=7.425, sebi_turnover_fee=0.125
    ).rounded()
    assert rounded.exchange_transaction == 7.43
    assert rounded.sebi_turnover_fee == 0.13


def test_unrounded_model_is_available_for_analytic_comparison() -> None:
    exact = IndianEquityCostModel(round_to_paisa=False)
    costs = exact.charges(Trade("X", Side.BUY, 100, 2500.0))
    assert costs.exchange_transaction == pytest.approx(7.425)


# --- Slippage ----------------------------------------------------------------


def test_zero_slippage_fills_at_the_reference() -> None:
    price = ZeroSlippage().fill_price(
        reference_price=100.0, side=Side.BUY, order_quantity=10, bar_volume=1000
    )
    assert price == 100.0


def test_fixed_bps_slippage_is_always_adverse() -> None:
    model = FixedBpsSlippage(bps=10.0)
    buy = model.fill_price(
        reference_price=100.0, side=Side.BUY, order_quantity=10, bar_volume=1000
    )
    sell = model.fill_price(
        reference_price=100.0, side=Side.SELL, order_quantity=10, bar_volume=1000
    )
    assert buy == pytest.approx(100.10)  # buyer pays more
    assert sell == pytest.approx(99.90)  # seller receives less


def test_volume_share_slippage_grows_with_participation() -> None:
    model = VolumeShareSlippage(coefficient=1.0)
    small = model.fill_price(
        reference_price=100.0, side=Side.BUY, order_quantity=10, bar_volume=10_000
    )
    large = model.fill_price(
        reference_price=100.0, side=Side.BUY, order_quantity=1_000, bar_volume=10_000
    )
    assert small < large
    # Quadratic: 10% participation gives 1% impact at coefficient 1.
    assert large == pytest.approx(101.0)
    # Degenerate bar: no impact rather than a division by zero.
    assert (
        model.fill_price(
            reference_price=100.0, side=Side.BUY, order_quantity=10, bar_volume=0
        )
        == 100.0
    )
