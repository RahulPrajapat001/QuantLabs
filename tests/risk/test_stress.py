"""Stress testing: real ingested windows, and a hard refusal to synthesise a shock."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session, sessionmaker

from quantlab.config import (
    DATA_START,
    HOLDOUT_START,
    HoldoutViolationError,
    get_settings,
)
from quantlab.db.models import OHLCVDaily, Security
from quantlab.db.session import create_all, create_db_engine, create_session_factory
from quantlab.risk.stress import (
    STRESS_WINDOWS,
    STRESS_WINDOWS_BY_SLUG,
    StressDataUnavailableError,
    StressWindow,
    get_stress_window,
    load_window_returns,
    replay_window,
    stress_coverage,
    stress_test,
    stress_window_ingest_plan,
)
from quantlab.risk.var import RiskError


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    engine = create_db_engine("sqlite:///:memory:")
    create_all(engine)
    return create_session_factory(engine)


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Session:
    session = session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def seed_window(
    session: Session,
    window: StressWindow,
    *,
    securities: tuple[str, ...] = ("SEC-A", "SEC-B"),
    crash: float = -0.35,
    recover: bool = False,
) -> None:
    """Ingest bars spanning a window, with a decline and an optional recovery.

    Deliberately *real rows in the database*: the whole point of the module is that it
    reads ingested data, so the tests exercise the database path rather than injecting
    a frame.
    """

    dates = pd.bdate_range(window.start, window.end)
    n = len(dates)
    trough = n // 2
    for security_id in securities:
        session.merge(
            Security(security_id=security_id, name=security_id, exchange="NSE")
        )
    session.flush()

    # Compound exactly to the requested decline, so `crash=-0.35` really does produce
    # a 35% drawdown rather than the 29.7% that spreading it arithmetically gives.
    down_steps = max(1, trough)
    up_steps = max(1, n - trough - 1)
    down_rate = (1.0 + crash) ** (1.0 / down_steps) - 1.0
    # Recover past the prior peak so the recovery date is reachable inside the window.
    up_target = (1.0 / (1.0 + crash)) * 1.05 if recover else 1.0
    up_rate = up_target ** (1.0 / up_steps) - 1.0

    for security_id in securities:
        price = 100.0
        for i, day in enumerate(dates):
            step = down_rate if i <= trough else up_rate
            price *= 1.0 + step
            session.add(
                OHLCVDaily(
                    security_id=security_id,
                    date=day.date(),
                    open=price,
                    high=price * 1.01,
                    low=price * 0.99,
                    close=price,
                    adj_close=price,
                    volume=1_000_000.0,
                    source="test",
                    as_of=datetime.now(UTC),
                )
            )
    session.flush()


# --- The window registry -----------------------------------------------------


def test_all_five_required_windows_are_defined() -> None:
    """PRD Module 9 names exactly these five."""

    slugs = {w.slug for w in STRESS_WINDOWS}
    assert slugs == {"gfc", "covid", "rate_shock_2022", "demonetisation", "ilfs"}


@pytest.mark.parametrize(
    ("slug", "start", "end"),
    [
        ("gfc", date(2007, 10, 1), date(2009, 3, 31)),
        ("covid", date(2020, 2, 1), date(2020, 4, 30)),
        ("rate_shock_2022", date(2022, 1, 1), date(2022, 10, 31)),
        ("demonetisation", date(2016, 11, 1), date(2016, 12, 31)),
        ("ilfs", date(2018, 9, 1), date(2018, 10, 31)),
    ],
)
def test_window_dates_match_the_specification(
    slug: str, start: date, end: date
) -> None:
    window = get_stress_window(slug)
    assert window.start == start
    assert window.end == end


def test_india_specific_windows_are_labelled() -> None:
    """An India-only shock says nothing about a US portfolio, so the market matters."""

    assert get_stress_window("demonetisation").market == "india"
    assert get_stress_window("ilfs").market == "india"
    assert get_stress_window("gfc").market == "global"


def test_the_gfc_predates_the_ordinary_sample() -> None:
    """The reason stress data has to be requested explicitly.

    Ordinary ingestion starts at DATA_START (2015-01-01), so the GFC would never be
    fetched by it. Nothing else in the project would notice its absence.
    """

    assert get_stress_window("gfc").predates_sample
    assert get_stress_window("gfc").start < DATA_START
    assert not get_stress_window("covid").predates_sample


def test_no_required_window_reaches_into_the_holdout() -> None:
    for window in STRESS_WINDOWS:
        assert not window.touches_holdout, f"{window.slug} reaches past {HOLDOUT_START}"


def test_unknown_window_lists_the_alternatives() -> None:
    with pytest.raises(RiskError, match="unknown stress window"):
        get_stress_window("dot_com")


def test_windows_are_indexed_by_slug() -> None:
    assert set(STRESS_WINDOWS_BY_SLUG) == {w.slug for w in STRESS_WINDOWS}


# --- Ingestion planning ------------------------------------------------------


def test_the_ingest_plan_covers_every_window_with_padding() -> None:
    plan = stress_window_ingest_plan(padding_days=30)
    assert set(plan.ranges) == {w.slug for w in STRESS_WINDOWS}
    for window in STRESS_WINDOWS:
        start, end = plan.ranges[window.slug]
        assert start == window.start - timedelta(days=30)
        assert end == window.end + timedelta(days=30)


def test_the_plan_reaches_back_before_the_ordinary_data_start() -> None:
    plan = stress_window_ingest_plan()
    assert plan.earliest() < DATA_START
    text = plan.render()
    assert "predates DATA_START" in text
    assert "retained regardless of the backtest window" in text


# --- Coverage ----------------------------------------------------------------


def test_coverage_reports_nothing_ingested(session: Session) -> None:
    for coverage in stress_coverage(session):
        assert not coverage.available
        assert "NOT INGESTED" in coverage.render()


def test_coverage_detects_a_complete_window(session: Session) -> None:
    window = get_stress_window("covid")
    seed_window(session, window)
    coverage = {c.window.slug: c for c in stress_coverage(session)}["covid"]
    assert coverage.available
    assert coverage.spans_window
    assert coverage.securities == 2
    assert "complete" in coverage.render()


def test_coverage_flags_partial_data(session: Session) -> None:
    """Replaying the last fortnight of a crisis and calling it the crisis is worse
    than not testing at all, so partial coverage is called out."""

    partial = StressWindow(
        name="partial",
        slug="partial",
        start=date(2009, 3, 1),
        end=date(2009, 3, 31),
        market="global",
        description="only the tail of the GFC",
    )
    seed_window(session, partial)
    coverage = {c.window.slug: c for c in stress_coverage(session)}["gfc"]
    assert coverage.available
    assert not coverage.spans_window
    assert "PARTIAL" in coverage.render()


# --- THE REFUSAL -------------------------------------------------------------


def test_a_missing_window_raises_rather_than_synthesising(session: Session) -> None:
    """The central guarantee: no synthetic shock is ever substituted.

    A stress test that quietly generated a "-30% shock" when the real data was absent
    would report a reassuring number about nothing. The error names the window and
    prints the exact ingest command instead.
    """

    window = get_stress_window("covid")
    with pytest.raises(StressDataUnavailableError) as exc:
        load_window_returns(session, window)

    message = str(exc.value)
    assert "will not substitute a synthetic" in message
    assert "quantlab ingest" in message
    assert "2020-01-02" in message  # the padded start of the required range


def test_stress_test_records_unavailable_windows_instead_of_skipping(
    session: Session,
) -> None:
    """A report covering two of five windows must say which three are missing."""

    seed_window(session, get_stress_window("covid"))
    report = stress_test(session)

    assert not report.complete
    tested = {r.window.slug for r in report.results}
    unavailable = {slug for slug, _ in report.unavailable}
    assert tested == {"covid"}
    assert unavailable == {"gfc", "rate_shock_2022", "demonetisation", "ilfs"}

    text = report.render()
    assert "could NOT be tested" in text
    assert "This report is INCOMPLETE" in text
    assert "not evidence of resilience" in text


def test_a_complete_report_says_so(session: Session) -> None:
    for window in STRESS_WINDOWS:
        seed_window(session, window)
    report = stress_test(session)
    assert report.complete
    assert len(report.results) == 5
    assert "INCOMPLETE" not in report.render()


def test_weighted_securities_absent_from_the_window_raise(session: Session) -> None:
    seed_window(session, get_stress_window("covid"), securities=("SEC-A",))
    with pytest.raises(StressDataUnavailableError, match="no ingested data"):
        load_window_returns(session, get_stress_window("covid"), security_ids=["SEC-Z"])


# --- Replay ------------------------------------------------------------------


def test_replay_reports_drawdown_worst_day_and_recovery(session: Session) -> None:
    """The three required outputs (PRD Module 9)."""

    window = get_stress_window("covid")
    seed_window(session, window, crash=-0.35, recover=True)
    returns = load_window_returns(session, window)
    result = replay_window(returns, window)

    assert result.max_drawdown > 0.3
    assert result.worst_day < 0
    assert result.worst_day_date is not None
    assert result.peak_date is not None
    assert result.trough_date is not None
    assert result.trough_date > result.peak_date
    assert result.recovered
    assert result.recovery_days is not None and result.recovery_days > 0
    assert result.trading_days == len(returns)

    text = result.render()
    assert "max drawdown" in text
    assert "worst day" in text
    assert "recovery" in text


def test_an_unrecovered_window_reports_none_not_a_number(session: Session) -> None:
    """Reporting the window length would read as a fast recovery."""

    window = get_stress_window("covid")
    seed_window(session, window, crash=-0.40, recover=False)
    result = replay_window(load_window_returns(session, window), window)
    assert not result.recovered
    assert result.recovery_days is None
    assert "NOT RECOVERED" in result.render()


def test_weights_are_applied_and_held_fixed(session: Session) -> None:
    """A crisis is not when a rebalance executes cleanly, so weights do not drift."""

    window = get_stress_window("covid")
    dates = pd.bdate_range(window.start, window.end)
    session.merge(Security(security_id="UP", name="UP", exchange="NSE"))
    session.merge(Security(security_id="DOWN", name="DOWN", exchange="NSE"))
    session.flush()
    for security_id, step in (("UP", 0.002), ("DOWN", -0.01)):
        price = 100.0
        for day in dates:
            price *= 1.0 + step
            session.add(
                OHLCVDaily(
                    security_id=security_id,
                    date=day.date(),
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    adj_close=price,
                    volume=1e6,
                    source="test",
                    as_of=datetime.now(UTC),
                )
            )
    session.flush()

    returns = load_window_returns(session, window)
    mostly_up = replay_window(returns, window, weights={"UP": 0.9, "DOWN": 0.1})
    mostly_down = replay_window(returns, window, weights={"UP": 0.1, "DOWN": 0.9})
    assert mostly_up.total_return > mostly_down.total_return
    assert mostly_down.max_drawdown > mostly_up.max_drawdown


def test_weights_are_renormalised_over_available_securities(session: Session) -> None:
    seed_window(session, get_stress_window("covid"), securities=("SEC-A",))
    returns = load_window_returns(session, get_stress_window("covid"))
    # Only SEC-A has data; the weight on the missing name is dropped and the rest
    # renormalised rather than silently treated as cash.
    result = replay_window(
        returns, get_stress_window("covid"), weights={"SEC-A": 0.5, "MISSING": 0.5}
    )
    assert result.securities == 1


def test_zero_weights_are_refused(session: Session) -> None:
    seed_window(session, get_stress_window("covid"))
    returns = load_window_returns(session, get_stress_window("covid"))
    with pytest.raises(RiskError, match="weights sum to zero"):
        replay_window(returns, get_stress_window("covid"), weights={"SEC-A": 0.0})


def test_correlation_convergence_is_reported(session: Session) -> None:
    """In a crash correlations converge and diversification stops working."""

    window = get_stress_window("covid")
    seed_window(session, window, securities=("SEC-A", "SEC-B", "SEC-C"))
    result = replay_window(load_window_returns(session, window), window)
    assert result.mean_correlation is not None
    # These synthetic series move identically, so correlation is 1.
    assert result.mean_correlation > 0.9
    assert "diversification" in result.render()


def test_replay_accepts_a_preaggregated_series(session: Session) -> None:
    window = get_stress_window("covid")
    series = pd.Series(
        [-0.05] * 20 + [0.03] * 20,
        index=pd.bdate_range(window.start, periods=40),
    )
    result = replay_window(series, window)
    assert result.securities == 1
    assert result.mean_correlation is None
    assert result.max_drawdown > 0.5


def test_an_empty_frame_is_refused() -> None:
    with pytest.raises(StressDataUnavailableError, match="no returns to replay"):
        replay_window(pd.DataFrame(), get_stress_window("covid"))


def test_a_single_observation_cannot_measure_a_drawdown() -> None:
    series = pd.Series([0.01], index=pd.bdate_range("2020-02-03", periods=1))
    with pytest.raises(StressDataUnavailableError, match="not enough to measure"):
        replay_window(series, get_stress_window("covid"))


# --- Report aggregation ------------------------------------------------------


def test_the_worst_window_is_identified(session: Session) -> None:
    seed_window(session, get_stress_window("covid"), crash=-0.40)
    seed_window(session, get_stress_window("ilfs"), crash=-0.10)
    report = stress_test(session)
    worst = report.worst_window
    assert worst is not None
    assert worst.window.slug == "covid"
    assert "Worst window" in report.render()


def test_report_frame_has_a_row_per_tested_window(session: Session) -> None:
    seed_window(session, get_stress_window("covid"))
    seed_window(session, get_stress_window("demonetisation"))
    frame = stress_test(session).frame()
    assert len(frame) == 2
    assert set(frame["window"]) == {"covid", "demonetisation"}
    for column in ("max_drawdown", "worst_day", "recovery_days", "recovered"):
        assert column in frame.columns


# --- The holdout lock still applies ------------------------------------------


def test_a_window_reaching_into_the_holdout_is_refused(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock is not suspended for risk work."""

    get_settings.cache_clear()
    monkeypatch.delenv("QUANTLAB_UNLOCK_HOLDOUT", raising=False)
    future = StressWindow(
        name="hypothetical future crisis",
        slug="future",
        start=date(2024, 6, 1),
        end=date(2024, 9, 30),
        market="global",
        description="inside the locked holdout",
    )
    assert future.touches_holdout
    with pytest.raises(HoldoutViolationError, match="locked holdout"):
        stress_test(session, windows=(future,))
    get_settings.cache_clear()


def test_an_unlocked_holdout_window_is_permitted(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("QUANTLAB_UNLOCK_HOLDOUT", "1")
    future = StressWindow(
        name="hypothetical future crisis",
        slug="future",
        start=date(2024, 6, 1),
        end=date(2024, 9, 30),
        market="global",
        description="inside the locked holdout",
    )
    seed_window(session, future)
    report = stress_test(session, windows=(future,), allow_holdout=True)
    assert len(report.results) == 1
    get_settings.cache_clear()


# --- Real returns, not fabricated ones ---------------------------------------


def test_returns_are_computed_from_ingested_adjusted_closes(session: Session) -> None:
    """A last check that the numbers come from the database rather than a model."""

    window = get_stress_window("demonetisation")
    dates = pd.bdate_range(window.start, window.end)
    session.merge(Security(security_id="SEC-A", name="SEC-A", exchange="NSE"))
    session.flush()
    prices = np.linspace(100.0, 80.0, len(dates))
    for day, price in zip(dates, prices, strict=True):
        session.add(
            OHLCVDaily(
                security_id="SEC-A",
                date=day.date(),
                open=price,
                high=price,
                low=price,
                close=price,
                adj_close=price,
                volume=1e6,
                source="test",
                as_of=datetime.now(UTC),
            )
        )
    session.flush()

    returns = load_window_returns(session, window)
    expected = pd.Series(prices).pct_change().dropna().to_numpy()
    assert np.allclose(returns["SEC-A"].to_numpy(), expected)
