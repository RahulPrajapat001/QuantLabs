"""Fixtures for the machine-learning tests (PRD Module 11).

The panel is synthetic and deliberately so. A real NSE panel would make these
tests depend on ingested data, and — more importantly — a *known* signal is the
only way to tell a working validation harness from a broken one. Here the forward
return is generated as ``beta * signal + noise``, so a model that recovers a
positive rank IC out of sample is doing what it should, and one that reports a
large in-sample R² with a near-zero out-of-sample one is showing the overfitting
gap the metrics are meant to expose.

Everything sits inside the train split (2015-2021) unless a test is specifically
exercising the holdout lock.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session, sessionmaker

from quantlab.db.session import create_all, create_db_engine, create_session_factory
from quantlab.ml_models.dataset import MLDataset, build_dataset
from quantlab.research.log import ResearchLog

FEATURES = ("signal", "noise_a", "noise_b", "momentum_12_1")


def make_panel(
    *,
    n_securities: int = 12,
    n_days: int = 400,
    start: date = date(2016, 1, 4),
    beta: float = 0.35,
    seed: int = 7,
) -> pd.DataFrame:
    """A long OHLCV+features panel whose next-day return really does follow ``signal``.

    ``beta`` sets how much: 0.35 gives a cross-sectional rank IC around 0.3, far
    stronger than anything in a real equity panel, which is the point — a test
    that needs a subtle effect to pass is a test that fails for unrelated reasons.
    """

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    frames = []
    for i in range(n_securities):
        signal = rng.normal(0.0, 1.0, size=n_days)
        noise_a = rng.normal(0.0, 1.0, size=n_days)
        noise_b = rng.normal(0.0, 1.0, size=n_days)
        # The return realised *after* date t is driven by the signal observed at
        # t, so a correctly aligned dataset finds it and a look-ahead bug does not.
        shocks = rng.normal(0.0, 0.01, size=n_days)
        forward = beta * 0.01 * signal + shocks
        returns = np.concatenate([[0.0], forward[:-1]])
        close = 100.0 * np.cumprod(1.0 + returns)
        momentum = pd.Series(close).pct_change(60).fillna(0.0).to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "security_id": f"SEC{i:03d}",
                    "adj_close": close,
                    "signal": signal,
                    "noise_a": noise_a,
                    "noise_b": noise_b,
                    "momentum_12_1": momentum,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["date", "security_id"])


def full_universe(panel: pd.DataFrame) -> dict[date, set[str]]:
    """Every security a member on every date — the simple case."""

    members = set(panel["security_id"].unique())
    return {d.date(): set(members) for d in pd.to_datetime(panel["date"]).unique()}


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()


@pytest.fixture
def universe(panel: pd.DataFrame) -> dict[date, set[str]]:
    return full_universe(panel)


@pytest.fixture
def dataset(panel: pd.DataFrame, universe: dict[date, set[str]]) -> MLDataset:
    return build_dataset(
        panel, features=FEATURES, universe=universe, index_name="TESTIDX"
    )


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    engine = create_db_engine("sqlite:///:memory:")
    create_all(engine)
    return create_session_factory(engine)


@pytest.fixture
def session(session_factory: sessionmaker[Session]):
    session = session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def research_log(session: Session) -> ResearchLog:
    return ResearchLog(session)
