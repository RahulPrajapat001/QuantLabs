"""Fixtures for the backtesting tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session, sessionmaker

from quantlab.db.session import create_all, create_db_engine, create_session_factory
from quantlab.research.log import ResearchLog


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


@pytest.fixture
def research_log(session: Session) -> ResearchLog:
    return ResearchLog(session)


def make_bars(
    *,
    start: str = "2015-01-02",
    n_days: int = 60,
    securities: int = 3,
    seed: int = 0,
    volume: float = 100_000.0,
    drift: float = 0.0,
    volatility: float = 0.01,
) -> pd.DataFrame:
    """A tidy, schema-complete bar panel for the DataHandler."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for i in range(securities):
        security_id = f"SEC-{i:04d}"
        close = 100.0 * np.cumprod(1.0 + rng.normal(drift, volatility, n_days))
        for t, day in enumerate(dates):
            price = float(close[t])
            rows.append(
                {
                    "date": day,
                    "security_id": security_id,
                    "open": price * 0.999,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": volume + t,
                }
            )
    return pd.DataFrame(rows)


def make_flat_bars(
    *,
    start: str = "2015-01-02",
    n_days: int = 10,
    security_id: str = "SEC-0000",
    price: float = 100.0,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """A single security at a constant price — makes arithmetic checkable by hand."""

    dates = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(
        [
            {
                "date": day,
                "security_id": security_id,
                "open": price,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "adj_close": price,
                "volume": volume,
            }
            for day in dates
        ]
    )


def make_trending_bars(
    *,
    start: str = "2016-01-04",
    n_days: int = 120,
    security_id: str = "SEC-0000",
    daily_return: float = 0.004,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """A steadily rising series — trend strategies will take a position in it."""

    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    price = 100.0
    for day in dates:
        rows.append(
            {
                "date": day,
                "security_id": security_id,
                "open": price * 0.999,
                "high": price * 1.006,
                "low": price * 0.996,
                "close": price,
                "adj_close": price,
                "volume": volume,
            }
        )
        price *= 1.0 + daily_return
    return pd.DataFrame(rows)
