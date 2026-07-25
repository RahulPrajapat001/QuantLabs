"""Metrics: computed from marks and round trips, with the awkward cases pinned."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.costs import CostBreakdown
from quantlab.backtesting.events import Fill, Side
from quantlab.backtesting.ledger import DailyMark, Ledger
from quantlab.backtesting.metrics import (
    annualised_return,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

START = date(2016, 1, 4)


def _curve(values: list[float]) -> pd.Series:
    index = pd.DatetimeIndex([START + timedelta(days=i) for i in range(len(values))])
    return pd.Series(values, index=index)


def _ledger_from_equity(values: list[float]) -> Ledger:
    ledger = Ledger()
    for i, equity in enumerate(values):
        ledger.record_mark(
            DailyMark(
                date=START + timedelta(days=i),
                cash=equity,
                unsettled_cash=0.0,
                holdings_value=0.0,
                equity=equity,
                positions=0,
                gross_exposure=0.0,
            )
        )
    return ledger


# --- Drawdown ----------------------------------------------------------------


def test_drawdown_reports_depth_duration_and_recovery() -> None:
    curve = _curve([100.0, 120.0, 90.0, 100.0, 130.0])
    info = max_drawdown(curve)
    assert info.max_drawdown == pytest.approx(-0.25)  # 120 → 90
    assert info.peak_date == START + timedelta(days=1)
    assert info.trough_date == START + timedelta(days=2)
    assert info.recovery_date == START + timedelta(days=4)
    assert info.drawdown_days == 1
    assert info.recovery_days == 3
    assert info.recovered


def test_an_unrecovered_drawdown_reports_none_not_zero() -> None:
    """Reporting 0 would turn 'never recovered' into 'recovered instantly'."""

    info = max_drawdown(_curve([100.0, 150.0, 80.0, 90.0]))
    assert info.max_drawdown == pytest.approx(80.0 / 150.0 - 1.0)  # -46.67%
    assert info.peak_date == START + timedelta(days=1)
    assert info.trough_date == START + timedelta(days=2)
    assert info.recovery_date is None
    assert info.recovery_days is None
    assert not info.recovered


def test_a_monotonic_curve_has_no_drawdown() -> None:
    info = max_drawdown(_curve([100.0, 101.0, 102.0]))
    assert info.max_drawdown == 0.0
    assert info.peak_date is None


def test_drawdown_of_a_short_series_is_defined() -> None:
    assert max_drawdown(pd.Series(dtype=float)).max_drawdown == 0.0
    assert max_drawdown(_curve([100.0])).max_drawdown == 0.0


# --- Ratios ------------------------------------------------------------------


def test_sharpe_annualises_daily_returns() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 2520))
    value = sharpe_ratio(returns)
    # mean/sd × √252 ≈ 0.05 × 15.87 ≈ 0.79
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert value == pytest.approx(expected)


def test_sharpe_accounts_for_the_risk_free_rate() -> None:
    returns = pd.Series([0.001] * 100)
    assert sharpe_ratio(returns) > 0
    # A risk-free rate equal to the return leaves no excess.
    assert sharpe_ratio(returns, risk_free_rate=0.001 * 252) == pytest.approx(
        0.0, abs=1e-9
    )


def test_sharpe_of_a_constant_series_is_zero_not_infinite() -> None:
    assert sharpe_ratio(pd.Series([0.001] * 50)) == 0.0
    assert sharpe_ratio(pd.Series([0.001])) == 0.0


def test_sortino_exceeds_sharpe_when_downside_is_mild() -> None:
    """A series with capped losses and fat gains looks better on Sortino."""

    returns = pd.Series([0.03, -0.005, 0.02, -0.005, 0.04, -0.005] * 20)
    assert sortino_ratio(returns) > sharpe_ratio(returns)


def test_sortino_with_no_losses_is_zero_rather_than_infinite() -> None:
    assert sortino_ratio(pd.Series([0.001] * 50)) == 0.0


def test_annualised_return_compounds() -> None:
    # Exactly one year of 252 observations, doubling.
    curve = pd.Series(
        [100.0 * (2 ** (i / 252)) for i in range(253)],
        index=pd.DatetimeIndex([START + timedelta(days=i) for i in range(253)]),
    )
    assert annualised_return(curve) == pytest.approx(1.0, rel=1e-6)


def test_annualised_return_of_a_wipeout_is_minus_one() -> None:
    assert annualised_return(_curve([100.0, 50.0, 0.0])) == -1.0


# --- Full metric set ---------------------------------------------------------


def test_metrics_from_a_known_equity_curve() -> None:
    ledger = _ledger_from_equity([100_000.0, 110_000.0, 105_000.0, 120_000.0])
    metrics = compute_metrics(ledger, initial_capital=100_000.0)

    assert metrics.total_return == pytest.approx(0.20)
    assert metrics.trading_days == 4
    assert metrics.final_equity == 120_000.0
    assert metrics.start == START
    assert metrics.drawdown.max_drawdown < 0
    assert metrics.trades == 0
    assert metrics.total_costs == 0.0


def test_trade_statistics_come_from_round_trips() -> None:
    ledger = _ledger_from_equity([100_000.0, 101_000.0, 102_000.0])
    # Two wins of ₹1,000, one loss of ₹500.
    for entry, exit_price, day in (
        (100.0, 110.0, 0),
        (100.0, 110.0, 1),
        (100.0, 95.0, 2),
    ):
        ledger.record_fill(
            Fill(
                date=START + timedelta(days=day),
                security_id=f"S{day}",
                side=Side.BUY,
                quantity=100,
                price=entry,
                costs=CostBreakdown(),
                reference_price=entry,
            )
        )
        ledger.record_fill(
            Fill(
                date=START + timedelta(days=day),
                security_id=f"S{day}",
                side=Side.SELL,
                quantity=100,
                price=exit_price,
                costs=CostBreakdown(),
                reference_price=exit_price,
            )
        )
    metrics = compute_metrics(ledger, initial_capital=100_000.0)

    assert metrics.trades == 3
    assert metrics.win_rate == pytest.approx(2 / 3)
    assert metrics.average_win == pytest.approx(1_000.0)
    assert metrics.average_loss == pytest.approx(-500.0)
    # 2,000 of profit against 500 of loss.
    assert metrics.profit_factor == pytest.approx(4.0)
    assert metrics.average_trade == pytest.approx(500.0)
    assert metrics.expectancy == pytest.approx(2 / 3 * 1000.0 + 1 / 3 * -500.0)


def test_profit_factor_with_no_losses_is_infinite_not_invented() -> None:
    ledger = _ledger_from_equity([100_000.0, 101_000.0])
    ledger.record_fill(
        Fill(
            date=START,
            security_id="A",
            side=Side.BUY,
            quantity=100,
            price=100.0,
            costs=CostBreakdown(),
            reference_price=100.0,
        )
    )
    ledger.record_fill(
        Fill(
            date=START + timedelta(days=1),
            security_id="A",
            side=Side.SELL,
            quantity=100,
            price=110.0,
            costs=CostBreakdown(),
            reference_price=110.0,
        )
    )
    metrics = compute_metrics(ledger, initial_capital=100_000.0)
    assert metrics.profit_factor == float("inf")
    assert metrics.win_rate == 1.0


def test_benchmark_comparison_is_rebased() -> None:
    ledger = _ledger_from_equity([100_000.0, 110_000.0, 120_000.0])
    # Benchmark rises 10% total, on a different price scale.
    benchmark = _curve([50.0, 52.5, 55.0])
    metrics = compute_metrics(ledger, initial_capital=100_000.0, benchmark=benchmark)

    assert metrics.benchmark_total_return == pytest.approx(0.10)
    assert metrics.excess_annual_return is not None
    assert metrics.excess_annual_return > 0  # strategy beat it
    assert metrics.benchmark_sharpe is not None


def test_metrics_dict_is_flat_and_log_ready() -> None:
    ledger = _ledger_from_equity([100_000.0, 105_000.0, 103_000.0])
    payload = compute_metrics(ledger, initial_capital=100_000.0).as_dict()
    assert isinstance(payload["sharpe"], float)
    assert "max_drawdown" in payload
    assert "cost_total" in payload
    assert "rejected_circuit_limit" in payload
    # Nothing nested — the JSON column takes a flat mapping.
    assert all(not isinstance(v, dict) for v in payload.values())


def test_render_labels_the_sharpe_as_undeflated() -> None:
    """A rendered Sharpe must never look like a finished result."""

    ledger = _ledger_from_equity([100_000.0, 105_000.0, 103_000.0])
    text = compute_metrics(ledger, initial_capital=100_000.0).render()
    assert "UNDEFLATED" in text
    assert "research deflate" in text


def test_return_moments_are_non_excess_kurtosis() -> None:
    rng = np.random.default_rng(3)
    ledger = _ledger_from_equity(
        list(100_000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 3000)))
    )
    metrics = compute_metrics(ledger, initial_capital=100_000.0)
    assert metrics.returns_kurtosis == pytest.approx(3.0, abs=0.3)
    assert metrics.returns_skew == pytest.approx(0.0, abs=0.2)


def test_an_empty_ledger_produces_defined_metrics() -> None:
    metrics = compute_metrics(Ledger(), initial_capital=100_000.0)
    assert metrics.total_return == 0.0
    assert metrics.sharpe == 0.0
    assert metrics.trading_days == 0
    assert metrics.final_equity == 100_000.0
    assert metrics.start is None
