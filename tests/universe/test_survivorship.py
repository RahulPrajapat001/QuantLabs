"""Survivorship measurement: metrics and the point-in-time vs current delta."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from quantlab.universe.membership import (
    CurrentUniverse,
    PointInTimeUniverse,
    UniverseResolver,
)
from quantlab.universe.survivorship import (
    compute_metrics,
    measure_survivorship_bias,
    write_survivorship_report,
)

from .conftest import seed_synthetic_index

IDX = "TESTIDX"


def test_compute_metrics_basic_shapes() -> None:
    # A steadily rising series: positive return, finite Sharpe, small drawdown.
    returns = pd.Series([0.01, -0.005, 0.008, 0.012, -0.002, 0.006])
    metrics = compute_metrics(returns)
    assert metrics.annual_return > 0
    assert metrics.sharpe != 0
    assert metrics.max_drawdown <= 0


def test_compute_metrics_flat_series_has_zero_sharpe() -> None:
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    metrics = compute_metrics(returns)
    assert metrics.sharpe == 0.0
    assert metrics.max_drawdown == 0.0


def test_compute_metrics_empty_series() -> None:
    metrics = compute_metrics(pd.Series([], dtype=float))
    assert metrics.annual_return == 0.0
    assert metrics.sharpe == 0.0
    assert metrics.max_drawdown == 0.0


def _size_strategy(resolver: UniverseResolver) -> pd.Series[float]:
    """A toy strategy whose daily return scales with the universe size on a date.

    This makes the survivorship effect visible: the point-in-time universe on
    2018-06-01 holds three names (incl. two later-removed), the current universe
    only two — so the two runs differ purely by survivorship treatment.
    """

    size = len(resolver.constituents(date(2018, 6, 1)))
    return pd.Series([0.001 * size] * 252)


def test_measure_survivorship_bias_reports_delta(session: Session) -> None:
    seed_synthetic_index(session, IDX)
    pit = PointInTimeUniverse(session, IDX)
    current = CurrentUniverse(session, IDX)

    comparison = measure_survivorship_bias(
        _size_strategy,
        strategy_name="toy_size",
        point_in_time=pit,
        current=current,
    )

    # PIT sees 3 constituents on 2018-06-01; current sees 2, so PIT return higher.
    assert comparison.point_in_time.annual_return > comparison.current.annual_return
    # The reported delta (current minus PIT) is therefore negative here.
    assert comparison.annual_return_delta < 0
    assert comparison.index_name == IDX


def test_write_survivorship_report(session: Session, tmp_path: Path) -> None:
    seed_synthetic_index(session, IDX)
    comparison = measure_survivorship_bias(
        _size_strategy,
        strategy_name="toy_size",
        point_in_time=PointInTimeUniverse(session, IDX),
        current=CurrentUniverse(session, IDX),
    )
    out = write_survivorship_report(comparison, path=tmp_path / "survivorship_bias.md")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Survivorship bias measurement" in text
    assert "toy_size" in text
    assert IDX in text
