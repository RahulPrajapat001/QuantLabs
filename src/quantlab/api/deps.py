"""Request-scoped dependencies: the database session, and loading price panels.

The session factory is created once per process and cached. Every request gets
its own :class:`~sqlalchemy.orm.Session`, closed when the response is finished —
a session shared across requests accumulates identity-map state and will
eventually serve a stale row.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from functools import lru_cache
from typing import Annotated

import pandas as pd
from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from quantlab.config import Settings, get_settings
from quantlab.db.models import OHLCVDaily
from quantlab.db.session import create_db_engine, create_session_factory

__all__ = [
    "SessionDep",
    "SettingsDep",
    "load_price_panel",
    "session_factory",
]

_OHLCV_COLUMNS = (
    "date",
    "security_id",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)


@lru_cache(maxsize=1)
def session_factory() -> sessionmaker[Session]:
    """Process-wide session factory, built from ``QUANTLAB_DATABASE_URL``."""

    return create_session_factory(create_db_engine())


def get_session() -> Iterator[Session]:
    """One session per request, always closed."""

    session = session_factory()()
    try:
        yield session
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def load_price_panel(
    session: Session,
    index_name: str,
    *,
    start: date | None = None,
    end: date | None = None,
    security_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Long OHLCV for every security that was **ever** in ``index_name``.

    "Ever" is deliberate and is not a survivorship leak: the point-in-time
    restriction is applied per date downstream (by the universe resolver or the
    dataset builder). Loading only *current* members here would bake the leak in
    at the source, where nothing downstream could undo it.
    """

    from quantlab.db.models import IndexMembership

    members = set(
        session.execute(
            select(IndexMembership.security_id)
            .where(IndexMembership.index_name == index_name)
            .distinct()
        ).scalars()
    )
    if security_ids is not None:
        members &= security_ids
    if not members:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"no index membership recorded for {index_name!r}. Build the "
                f"universe first: quantlab universe build --index {index_name}"
            ),
        )

    stmt = select(
        OHLCVDaily.date,
        OHLCVDaily.security_id,
        OHLCVDaily.open,
        OHLCVDaily.high,
        OHLCVDaily.low,
        OHLCVDaily.close,
        OHLCVDaily.adj_close,
        OHLCVDaily.volume,
    ).where(OHLCVDaily.security_id.in_(sorted(members)))
    if start is not None:
        stmt = stmt.where(OHLCVDaily.date >= start)
    if end is not None:
        stmt = stmt.where(OHLCVDaily.date <= end)

    rows = session.execute(stmt).all()
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"no OHLCV rows for {index_name!r} in the requested window. "
                "Ingest data first: quantlab ingest --tickers ..."
            ),
        )
    panel = pd.DataFrame(rows, columns=list(_OHLCV_COLUMNS))
    return panel.sort_values(["date", "security_id"]).reset_index(drop=True)


def returns_matrix(panel: pd.DataFrame, *, lookback: int | None = None) -> pd.DataFrame:
    """Wide daily returns from a long OHLCV panel, on ``adj_close``.

    ``adj_close`` rather than ``close``: a split would otherwise register as a
    50% one-day loss, and every risk number computed from it would be wrong.
    """

    wide = panel.pivot(
        index="date", columns="security_id", values="adj_close"
    ).sort_index()
    returns = wide.pct_change().dropna(how="all")
    if lookback is not None:
        returns = returns.tail(lookback)
    return returns.dropna(axis=1, how="any")
