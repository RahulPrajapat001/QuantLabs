"""Securities and their ticker history.

Securities are identified by a stable internal ``security_id``; a ticker is a
label that a company can change and that an exchange can reassign to somebody
else. ``ticker_history`` records which ticker pointed at which security over
time, so a listing that renamed mid-sample resolves correctly rather than
splitting into two apparent companies or merging two real ones.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from quantlab.api.deps import SessionDep
from quantlab.api.formatting import format_date
from quantlab.db.models import Security, TickerHistory

router = APIRouter(prefix="/securities", tags=["securities"])


class TickerPeriod(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    start_date: date
    end_date: date | None
    start_display: str
    end_display: str
    is_current: bool


class SecurityOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    security_id: str
    name: str | None
    isin: str | None
    exchange: str | None
    listing_date: date | None
    delisting_date: date | None
    delisting_reason: str | None
    listing_display: str
    delisting_display: str
    is_active: bool = Field(description="False once a delisting date is recorded.")
    tickers: list[TickerPeriod] = Field(default_factory=list)


class SecurityPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    limit: int
    offset: int
    items: list[SecurityOut]


def _to_out(security: Security, tickers: list[TickerHistory]) -> SecurityOut:
    return SecurityOut(
        security_id=security.security_id,
        name=security.name,
        isin=security.isin,
        exchange=security.exchange,
        listing_date=security.listing_date,
        delisting_date=security.delisting_date,
        delisting_reason=security.delisting_reason,
        listing_display=format_date(security.listing_date),
        delisting_display=format_date(security.delisting_date),
        is_active=security.delisting_date is None,
        tickers=[
            TickerPeriod(
                ticker=t.ticker,
                start_date=t.start_date,
                end_date=t.end_date,
                start_display=format_date(t.start_date),
                end_display=format_date(t.end_date),
                is_current=t.end_date is None,
            )
            for t in sorted(tickers, key=lambda t: t.start_date)
        ],
    )


@router.get("", response_model=SecurityPage, summary="List securities.")
def list_securities(
    session: SessionDep,
    search: str | None = Query(
        default=None, description="Case-insensitive match on ticker, name or ISIN."
    ),
    active_only: bool = Query(
        default=False,
        description=(
            "Exclude delisted securities. Off by default — a universe that "
            "silently drops delistings is survivorship-biased."
        ),
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> SecurityPage:
    stmt = select(Security)
    if active_only:
        stmt = stmt.where(Security.delisting_date.is_(None))
    if search:
        pattern = f"%{search.lower()}%"
        matching = select(TickerHistory.security_id).where(
            func.lower(TickerHistory.ticker).like(pattern)
        )
        stmt = stmt.where(
            func.lower(func.coalesce(Security.name, "")).like(pattern)
            | func.lower(func.coalesce(Security.isin, "")).like(pattern)
            | Security.security_id.in_(matching)
        )

    total = int(session.scalar(select(func.count()).select_from(stmt.subquery())) or 0)
    rows = list(
        session.scalars(stmt.order_by(Security.security_id).limit(limit).offset(offset))
    )
    history = _ticker_history(session, [r.security_id for r in rows])
    return SecurityPage(
        total=total,
        limit=limit,
        offset=offset,
        items=[_to_out(r, history.get(r.security_id, [])) for r in rows],
    )


@router.get("/{security_id}", response_model=SecurityOut, summary="One security by id.")
def get_security(security_id: str, session: SessionDep) -> SecurityOut:
    security = session.get(Security, security_id)
    if security is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no security with security_id {security_id!r}",
        )
    history = _ticker_history(session, [security_id])
    return _to_out(security, history.get(security_id, []))


def _ticker_history(
    session: SessionDep, security_ids: list[str]
) -> dict[str, list[TickerHistory]]:
    if not security_ids:
        return {}
    rows = session.scalars(
        select(TickerHistory).where(TickerHistory.security_id.in_(security_ids))
    )
    grouped: dict[str, list[TickerHistory]] = {}
    for row in rows:
        grouped.setdefault(row.security_id, []).append(row)
    return grouped
