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
) -> pd.DataFrame:
    """A tidy, schema-complete bar panel for the DataHandler."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for i in range(securities):
        security_id = f"SEC-{i:04d}"
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n_days))
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
                    "volume": 100_000.0 + t,
                }
            )
    return pd.DataFrame(rows)
