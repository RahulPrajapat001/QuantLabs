"""Golden regression — priority test 3: one strategy's full metric output, pinned.

A committed fixture holding every metric of a fully specified run. Any change to
sizing, execution, cost arithmetic, marking, or metric definitions moves at least
one of these numbers, and this test fails with the exact field and delta.

Its value is that it fails on changes nobody thought to test for. The unit tests
check the behaviours we anticipated; this catches a refactor that quietly shifts a
fill by one bar or double-counts GST.

**When this test fails**, do not regenerate the fixture to make it pass. Work out
which number moved and why. If the change is a deliberate improvement, regenerate
with::

    python -m tests.backtesting.test_golden --regenerate

and commit the fixture *in the same commit* as the change, so the diff shows both
the cause and its effect on every metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.costs import (
    FixedBpsSlippage,
    IndianEquityCostModel,
    Segment,
    ZeroDeliveryBrokerage,
)
from quantlab.backtesting.data_handler import DataHandler
from quantlab.backtesting.engine import BacktestConfig, BacktestEngine
from quantlab.backtesting.execution import CircuitLimitPolicy, ExcessPolicy
from quantlab.backtesting.portfolio import Portfolio, RiskLimits, SizingMethod
from quantlab.db.session import create_all, create_db_engine, create_session_factory
from quantlab.research.log import ResearchLog
from quantlab.strategies.technical import MovingAverageCrossStrategy

FIXTURE = Path(__file__).parent / "fixtures" / "golden_ma_cross.json"

#: Every input is fixed here. Changing any of them invalidates the fixture, which
#: is the point — the run must be reproducible from this specification alone.
SEED = 20240115
N_DAYS = 500
N_SECURITIES = 6
START = "2016-01-04"
INITIAL_CAPITAL = 1_000_000.0


#: Offset of the injected circuit-band day on SEC-0000. Chosen because the pinned
#: strategy raises an order that executes on exactly this bar, so the band lock
#: actually causes a rejection rather than passing unnoticed — verified by
#: :func:`test_the_golden_run_actually_exercises_the_interesting_paths`. The order
#: was raised on the previous bar from data that this injection does not touch, so
#: the rejection is deterministic.
CIRCUIT_LOCK_OFFSET = 134


def build_bars() -> pd.DataFrame:
    """A deterministic synthetic panel with one circuit-band day.

    The band day is deliberate: it exercises the rejection path, so the fixture
    notices if circuit handling silently stops working.
    """

    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START, periods=N_DAYS)
    rows: list[dict[str, Any]] = []
    for i in range(N_SECURITIES):
        security_id = f"SEC-{i:04d}"
        returns = rng.normal(0.0004, 0.016, N_DAYS)
        close = 100.0 * np.cumprod(1.0 + returns)
        for t, day in enumerate(dates):
            price = float(close[t])
            rows.append(
                {
                    "date": day,
                    "security_id": security_id,
                    "open": price * 0.9985,
                    "high": price * 1.011,
                    "low": price * 0.989,
                    "close": price,
                    "adj_close": price,
                    "volume": 250_000.0 + 1_000.0 * ((t * 7 + i) % 50),
                }
            )
    bars = pd.DataFrame(rows)

    # One locked-at-band day on SEC-0000: open == high == low == close, 22% above
    # the previous close, which is both a band breach and a zero-range bar.
    offset = CIRCUIT_LOCK_OFFSET
    mask = (bars["security_id"] == "SEC-0000") & (
        bars["date"] == pd.Timestamp(dates[offset])
    )
    previous = float(
        bars.loc[
            (bars["security_id"] == "SEC-0000")
            & (bars["date"] == pd.Timestamp(dates[offset - 1])),
            "close",
        ].iloc[0]
    )
    locked = previous * 1.22
    bars.loc[mask, ["open", "high", "low", "close", "adj_close"]] = locked
    return bars


def build_engine(research_log: ResearchLog) -> BacktestEngine:
    """The pinned configuration. Every knob is stated explicitly."""

    return BacktestEngine(
        research_log,
        strategy=MovingAverageCrossStrategy(fast=20, slow=50, use_ema=False),
        data_handler=DataHandler(build_bars()),
        portfolio=Portfolio(
            initial_capital=INITIAL_CAPITAL,
            sizing=SizingMethod.EQUAL_WEIGHT,
            limits=RiskLimits(
                max_position_weight=0.25,
                max_positions=4,
                max_gross_exposure=1.0,
                cash_reserve=0.02,
                min_order_value=1_000.0,
            ),
            target_positions=4,
            fractional_shares=False,
            settlement_days=1,
        ),
        config=BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            cost_model=IndianEquityCostModel(
                brokerage=ZeroDeliveryBrokerage(), segment=Segment.DELIVERY
            ),
            slippage=FixedBpsSlippage(bps=5.0),
            max_participation=0.01,
            excess_policy=ExcessPolicy.REJECT,
            circuit_policy=CircuitLimitPolicy(band=0.20),
            segment=Segment.DELIVERY,
            settlement_days=1,
            universe="point-in-time",
            index_name="NIFTY500",
        ),
    )


def run_golden() -> dict[str, Any]:
    """Run the pinned configuration and return its full metric set."""

    engine_db = create_db_engine("sqlite:///:memory:")
    create_all(engine_db)
    session = create_session_factory(engine_db)()
    try:
        engine = build_engine(ResearchLog(session))
        engine.run()
        assert engine.metrics is not None
        payload = dict(engine.metrics.as_dict())
        # Structural counts, so a change in trade or fill handling also shows up.
        payload["_fills"] = len(engine.ledger.fills)
        payload["_rejections"] = len(engine.ledger.rejections)
        payload["_marks"] = len(engine.ledger.marks)
        payload["_signals"] = len(engine.ledger.signals)
        return payload
    finally:
        session.close()


def _normalise(value: Any) -> Any:
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return str(value)
        return round(value, 10)
    if isinstance(value, bool | int | str) or value is None:
        return value
    return str(value)


def regenerate() -> None:
    """Write the fixture. Run deliberately, never to silence a failure."""

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: _normalise(v) for k, v in sorted(run_golden().items())}
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_golden_fixture_exists() -> None:
    assert FIXTURE.exists(), (
        f"{FIXTURE} is missing. Generate it with "
        "'python -m tests.backtesting.test_golden --regenerate' and commit it."
    )


def test_full_metric_output_matches_the_committed_fixture() -> None:
    """Every metric of a fully specified run, pinned to the committed fixture."""

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual = {k: _normalise(v) for k, v in run_golden().items()}

    assert set(actual) == set(expected), (
        "the metric set itself changed — added: "
        f"{sorted(set(actual) - set(expected))}, removed: "
        f"{sorted(set(expected) - set(actual))}"
    )

    mismatches = []
    for key in sorted(expected):
        want, got = expected[key], actual[key]
        if isinstance(want, float) and isinstance(got, float):
            if want != pytest.approx(got, rel=1e-9, abs=1e-9):
                mismatches.append(f"  {key}: fixture={want!r} actual={got!r}")
        elif want != got:
            mismatches.append(f"  {key}: fixture={want!r} actual={got!r}")

    assert not mismatches, (
        "the pinned run's output changed:\n"
        + "\n".join(mismatches)
        + "\n\nWork out which change moved these numbers. Regenerate the fixture "
        "only if the change is deliberate, and commit both together."
    )


def test_the_golden_run_actually_exercises_the_interesting_paths() -> None:
    """A regression fixture over a run that trades nothing would guard nothing."""

    payload = run_golden()
    assert payload["_fills"] > 20, "the pinned run must actually trade"
    assert payload["trades"] > 5, "and must close round trips"
    assert payload["total_costs"] > 0, "and must pay Indian costs"
    assert payload["rejected_circuit_limit"] > 0, (
        "the pinned panel contains band-locked days; if this is zero the circuit "
        "rejection path is no longer being exercised"
    )


if __name__ == "__main__":  # pragma: no cover - manual regeneration
    import sys

    if "--regenerate" in sys.argv:
        regenerate()
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps({k: _normalise(v) for k, v in run_golden().items()}, indent=2))
