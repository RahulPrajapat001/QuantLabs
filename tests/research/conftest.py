"""Fixtures for the research-integrity tests.

Everything runs against in-memory SQLite with the full schema, so the research log
is exercised as a real table rather than a mock — the point of the log is that it
persists, and a mock cannot fail to persist.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker

from quantlab.db.session import create_all, create_db_engine, create_session_factory
from quantlab.research.log import ResearchLog, RunSpec


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


def make_spec(
    strategy_name: str = "test_strategy",
    *,
    start: date = date(2015, 1, 1),
    end: date = date(2021, 12, 31),
    hypothesis_id: str | None = None,
    holdout: bool = False,
    cost_model: str = "IndianEquityCostModel",
    **parameters: object,
) -> RunSpec:
    """A RunSpec on the train split by default — no holdout involvement."""

    return RunSpec(
        strategy_name=strategy_name,
        universe="point-in-time",
        index_name="NIFTY500",
        start=start,
        end=end,
        cost_model=cost_model,
        parameters=parameters,
        hypothesis_id=hypothesis_id,
        holdout=holdout,
    )


def zero_edge_returns(
    rng: np.random.Generator, *, n_periods: int = 1260, volatility: float = 0.012
) -> np.ndarray:
    """Daily returns drawn with an expected value of **exactly zero** — no edge.

    Deliberately *not* demeaned: the realised mean of each draw is the sampling
    noise the whole exercise is about. Over 1260 days (five years) the annualised
    Sharpe of such a series has a standard deviation near 0.45, so the best of two
    hundred of them reliably looks like a tradeable strategy. It is not one.
    """

    return rng.normal(0.0, volatility, size=n_periods)


def write_hypotheses(tmp_path: Path, body: str) -> Path:
    """Write a throwaway pre-registration document and return its path."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "hypotheses.md"
    path.write_text(body, encoding="utf-8")
    return path
