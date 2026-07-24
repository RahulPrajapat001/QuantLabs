"""Shared fixtures for the universe-layer tests.

Offline and deterministic: an in-memory SQLite database with the full schema and
small seed helpers for securities, tickers, membership intervals and bars.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from quantlab.db.models import (
    OHLCVDaily,
    Security,
    TickerHistory,
)
from quantlab.db.session import create_all, create_db_engine, create_session_factory
from quantlab.universe.membership import (
    build_intervals_from_events,
    insert_membership_intervals,
)


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


def add_security(
    session: Session,
    security_id: str,
    *,
    name: str | None = None,
    exchange: str = "NSE",
    delisting_date: date | None = None,
    delisting_reason: str | None = None,
) -> str:
    """Insert a bare security row (FK target for membership / bars)."""

    session.add(
        Security(
            security_id=security_id,
            name=name or security_id,
            exchange=exchange,
            delisting_date=delisting_date,
            delisting_reason=delisting_reason,
        )
    )
    session.flush()
    return security_id


def add_ticker(
    session: Session,
    security_id: str,
    ticker: str,
    *,
    start: date,
    end: date | None = None,
    source: str = "test",
) -> None:
    session.add(
        TickerHistory(
            security_id=security_id,
            ticker=ticker,
            start_date=start,
            end_date=end,
            source=source,
        )
    )
    session.flush()


def add_bar(
    session: Session,
    security_id: str,
    on: date,
    close: float,
    *,
    source: str = "test",
) -> None:
    session.add(
        OHLCVDaily(
            security_id=security_id,
            date=on,
            open=close,
            high=close,
            low=close,
            close=close,
            adj_close=close,
            volume=1_000.0,
            source=source,
            as_of=datetime.now(UTC),
        )
    )
    session.flush()


def seed_synthetic_index(session: Session, index_name: str = "TESTIDX") -> None:
    """A synthetic index with known additions, removals and a re-addition.

    Intervals (half-open ``[start, end)``):

    * ``SEC-A``: 2015-01-01 → current (never leaves)
    * ``SEC-B``: 2016-06-01 → 2019-03-01
    * ``SEC-C``: 2015-01-01 → 2017-09-01, then re-added 2020-03-01 → current
    * ``SEC-D``: 2018-03-01 → 2018-09-01
    """

    for sec in ("SEC-A", "SEC-B", "SEC-C", "SEC-D"):
        add_security(session, sec)

    events = [
        ("SEC-A", "add", date(2015, 1, 1), "seed"),
        ("SEC-C", "add", date(2015, 1, 1), "seed"),
        ("SEC-C", "remove", date(2017, 9, 1), "seed"),
        ("SEC-B", "add", date(2016, 6, 1), "seed"),
        ("SEC-D", "add", date(2018, 3, 1), "seed"),
        ("SEC-D", "remove", date(2018, 9, 1), "seed"),
        ("SEC-B", "remove", date(2019, 3, 1), "seed"),
        ("SEC-C", "add", date(2020, 3, 1), "seed"),
    ]
    intervals = build_intervals_from_events(events)
    insert_membership_intervals(session, index_name, intervals)
