"""Shared fixtures and fakes for the data-layer tests.

Nothing here touches a network or a real vendor: providers are driven by
in-memory fakes, so the whole suite is deterministic and offline.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy.orm import Session, sessionmaker

from quantlab.data.base import (
    DataProvider,
    DelistingInfo,
    Frequency,
    Security,
    empty_corporate_actions,
    empty_ohlcv,
)
from quantlab.db.session import create_all, create_db_engine, create_session_factory


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    """An in-memory SQLite database with the full schema created."""

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


def make_ohlcv(dates: list[str], *, base: float = 100.0) -> pd.DataFrame:
    """A tidy, schema-valid OHLCV frame for the given ISO dates."""

    rows = []
    for i, d in enumerate(dates):
        close = base + i
        rows.append(
            {
                "date": pd.Timestamp(d),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "adj_close": close,
                "volume": 1_000.0 + i,
            }
        )
    return pd.DataFrame(rows)


class FakeProvider(DataProvider):
    """A DataProvider backed by canned frames keyed on the vendor symbol.

    Used to drive the ingest layer deterministically. It still flows through the
    base class's schema validation, so anything it returns must be schema-valid.
    """

    source = "fake"

    def __init__(
        self,
        ohlcv: dict[str, pd.DataFrame] | None = None,
        actions: dict[str, pd.DataFrame] | None = None,
        delisting: dict[str, DelistingInfo] | None = None,
    ) -> None:
        self._ohlcv = ohlcv or {}
        self._actions = actions or {}
        self._delisting = delisting or {}
        self.calls: list[tuple[str, date, date]] = []

    def _fetch_ohlcv(
        self,
        security: Security,
        start: date,
        end: date,
        frequency: Frequency,
    ) -> pd.DataFrame:
        self.calls.append((security.symbol, start, end))
        frame = self._ohlcv.get(security.symbol, empty_ohlcv()).copy()
        if frame.empty:
            return frame
        mask = (frame["date"] >= pd.Timestamp(start)) & (
            frame["date"] <= pd.Timestamp(end)
        )
        return frame.loc[mask].reset_index(drop=True)

    def _fetch_corporate_actions(self, security: Security) -> pd.DataFrame:
        return self._actions.get(security.symbol, empty_corporate_actions()).copy()

    def _fetch_delisting_info(self, security: Security) -> DelistingInfo:
        return self._delisting.get(security.symbol, DelistingInfo(delisted=False))
