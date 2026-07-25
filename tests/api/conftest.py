"""Fixtures for the API tests.

The database is a **temporary file**, not in-memory SQLite. That is deliberate:
the backtest worker runs on a thread and opens its own session
(:func:`quantlab.api.deps.session_factory`), and two connections to
``sqlite:///:memory:`` are two different empty databases. A file-backed URL is
the only way the job registry's threads see the data the request seeded.

``QUANTLAB_DATABASE_URL`` is set for the session and the settings/session-factory
caches are cleared, so both the dependency-injected path and the worker's
direct-construction path resolve to the same file.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from quantlab.db.models import (
    IndexMembership,
    OHLCVDaily,
    Security,
    TickerHistory,
)
from quantlab.db.session import create_all, create_db_engine, create_session_factory

INDEX_NAME = "TESTIDX"
N_SECURITIES = 8
START = date(2016, 1, 4)
N_DAYS = 320


def _seed(session: Session) -> None:
    """A small but complete world: securities, tickers, prices, membership.

    One security is delisted mid-sample and one leaves the index, so the
    survivorship endpoint has something real to report rather than an empty
    difference.
    """

    import pandas as pd

    rng = np.random.default_rng(11)
    dates = list(pd.bdate_range(START, periods=N_DAYS).date)
    now = datetime.now(UTC)

    # Securities first, flushed, so the foreign keys on every dependent table
    # resolve. SQLite enforces them, and the unit-of-work ordering is not worth
    # relying on for a fixture.
    for i in range(N_SECURITIES):
        session.add(
            Security(
                security_id=f"SEC{i:03d}",
                isin=f"INE{i:09d}",
                name=f"Test Company {i}",
                exchange="NSE",
                listing_date=START,
                delisting_date=(date(2017, 1, 10) if i == N_SECURITIES - 1 else None),
                delisting_reason="merger" if i == N_SECURITIES - 1 else None,
            )
        )
    session.flush()

    for i in range(N_SECURITIES):
        security_id = f"SEC{i:03d}"
        delisted = date(2017, 1, 10) if i == N_SECURITIES - 1 else None
        session.add(
            TickerHistory(
                security_id=security_id,
                ticker=f"TEST{i}",
                start_date=START,
                end_date=None,
                source="test",
            )
        )
        # Leaves the index partway through, so point-in-time membership and
        # today's list genuinely differ.
        session.add(
            IndexMembership(
                security_id=security_id,
                index_name=INDEX_NAME,
                start_date=START,
                end_date=date(2016, 9, 30) if i == 0 else None,
                source_document="test-fixture",
            )
        )

        price = 100.0 + i * 5.0
        for when in dates:
            if delisted is not None and when > delisted:
                break
            step = float(rng.normal(0.0004, 0.012))
            price = max(1.0, price * (1.0 + step))
            high = price * (1.0 + abs(float(rng.normal(0, 0.004))))
            low = price * (1.0 - abs(float(rng.normal(0, 0.004))))
            session.add(
                OHLCVDaily(
                    security_id=security_id,
                    date=when,
                    open=price * (1.0 + float(rng.normal(0, 0.002))),
                    high=max(high, price),
                    low=min(low, price),
                    close=price,
                    adj_close=price,
                    volume=int(rng.integers(50_000, 500_000)),
                    source="test",
                    as_of=now,
                )
            )
    session.commit()


@pytest.fixture(scope="session")
def database_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("api-db") / "quantlab.db"
    return f"sqlite:///{path.as_posix()}"


@pytest.fixture(scope="session", autouse=True)
def _configure(database_url: str) -> Iterator[None]:
    """Point every session-construction path at the same temporary database."""

    import os

    from quantlab.api import deps
    from quantlab.config import get_settings

    previous = os.environ.get("QUANTLAB_DATABASE_URL")
    os.environ["QUANTLAB_DATABASE_URL"] = database_url
    get_settings.cache_clear()
    deps.session_factory.cache_clear()

    engine = create_db_engine(database_url)
    create_all(engine)
    session = create_session_factory(engine)()
    try:
        _seed(session)
    finally:
        session.close()

    yield

    if previous is None:
        os.environ.pop("QUANTLAB_DATABASE_URL", None)
    else:
        os.environ["QUANTLAB_DATABASE_URL"] = previous
    get_settings.cache_clear()
    deps.session_factory.cache_clear()


@pytest.fixture
def session(database_url: str) -> Iterator[Session]:
    factory = create_session_factory(create_db_engine(database_url))
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client() -> Iterator[TestClient]:
    from quantlab.api.app import create_app
    from quantlab.api.jobs import reset_registry

    reset_registry()
    with TestClient(create_app()) as test_client:
        yield test_client
    reset_registry()


@pytest.fixture
def logged_run(session: Session) -> str:
    """A completed ``research_log`` row, for the endpoints that need one."""

    from quantlab.research.log import ResearchLog, RunResult, RunSpec

    rng = np.random.default_rng(3)
    returns = rng.normal(0.0006, 0.011, size=400)
    log = ResearchLog(session)
    spec = RunSpec(
        strategy_name="fixture_strategy",
        universe="point-in-time",
        index_name=INDEX_NAME,
        start=START,
        end=date(2017, 3, 31),
        cost_model="IndianEquityCostModel",
        parameters={"lookback": 20},
    )
    with log.run(spec) as handle:
        handle.record(
            RunResult.from_returns(
                returns,
                metrics={
                    "total_return": 0.184,
                    "annual_return": 0.121,
                    "annual_volatility": 0.174,
                    "sortino": 1.02,
                    "calmar": 0.61,
                    "max_drawdown": 0.198,
                    "drawdown_days": 47,
                    "recovery_days": 88,
                    "trades": 214,
                    "win_rate": 0.53,
                    "profit_factor": 1.24,
                    "average_trade": 812.4,
                    "expectancy": 640.1,
                    "turnover": 1.86,
                    "total_costs": 41_233.0,
                    "cost_drag": 0.021,
                    "final_equity": 1_184_000.0,
                    "trading_days": 400,
                    "returns_skew": -0.21,
                    "returns_kurtosis": 4.1,
                },
            )
        )
    return handle.run_id
