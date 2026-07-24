"""Ingestion orchestration.

Ties the pieces together for one ticker over one date range:

1. Resolve the ticker to a stable ``security_id`` **as of the requested date**
   (:mod:`quantlab.data.identity`), minting a new identity only if unknown.
2. Fetch validated OHLCV / corporate actions / delisting status from a provider.
3. Land the raw response immutably as partitioned parquet
   (:mod:`quantlab.data.storage`).
4. Upsert cleaned rows into the database idempotently on ``(security_id, date)``.
5. Record the attempt in ``ingest_log`` — success or failure — so incremental
   mode knows where to resume and a partial run never looks complete.

Delisted securities keep every price row; a reported delisting only stamps
``securities.delisting_date`` / ``delisting_reason``, it never drops history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from quantlab.config import get_settings
from quantlab.db.models import (
    CorporateAction,
    IngestLog,
    OHLCVDaily,
    Security,
)

from .base import DataProvider, Frequency
from .base import Security as SecurityRef
from .errors import ProviderError
from .identity import get_or_create_security
from .storage import write_raw_ohlcv

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of ingesting one ticker."""

    ticker: str
    security_id: str
    source: str
    start: date
    end: date
    rows_written: int
    rows_total: int
    corporate_actions_written: int
    status: str
    error: str | None = None


def _upsert_ohlcv(
    session: Session,
    security_id: str,
    df: pd.DataFrame,
    *,
    source: str,
    as_of: datetime,
) -> int:
    """Insert missing bars and update changed ones; return rows written.

    Idempotent on ``(security_id, date)``: a second identical ingest inserts
    nothing and updates nothing. A row whose values changed (e.g. a corrected
    ``adj_close`` after a split) is updated in place and its ``as_of`` bumped —
    that retroactive correction is expected and is not a duplicate.

    Portable across SQLite and PostgreSQL (no dialect-specific upsert).
    """

    if df.empty:
        return 0

    dates = [d.date() for d in pd.to_datetime(df["date"])]
    existing_rows = session.execute(
        select(OHLCVDaily)
        .where(OHLCVDaily.security_id == security_id)
        .where(OHLCVDaily.date.in_(dates))
    ).scalars()
    existing = {row.date: row for row in existing_rows}

    written = 0
    for _, r in df.iterrows():
        row_date = pd.Timestamp(r["date"]).date()
        values = {
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "adj_close": float(r["adj_close"]),
            "volume": float(r["volume"]),
        }
        current = existing.get(row_date)
        if current is None:
            session.add(
                OHLCVDaily(
                    security_id=security_id,
                    date=row_date,
                    source=source,
                    as_of=as_of,
                    **values,
                )
            )
            written += 1
        elif any(getattr(current, k) != v for k, v in values.items()):
            for k, v in values.items():
                setattr(current, k, v)
            current.as_of = as_of
            current.source = source
            written += 1

    session.flush()
    return written


def _upsert_corporate_actions(
    session: Session,
    security_id: str,
    df: pd.DataFrame,
    *,
    source: str,
    as_of: datetime,
) -> int:
    """Idempotently upsert corporate actions on ``(security_id, date, type)``."""

    if df.empty:
        return 0

    keys = [
        (pd.Timestamp(r["date"]).date(), str(r["action_type"]))
        for _, r in df.iterrows()
    ]
    existing_rows = session.execute(
        select(CorporateAction).where(CorporateAction.security_id == security_id)
    ).scalars()
    existing = {(row.date, row.action_type): row for row in existing_rows}

    written = 0
    for (row_date, action_type), (_, r) in zip(keys, df.iterrows(), strict=True):
        value = float(r["value"])
        current = existing.get((row_date, action_type))
        if current is None:
            session.add(
                CorporateAction(
                    security_id=security_id,
                    date=row_date,
                    action_type=action_type,
                    value=value,
                    source=source,
                    as_of=as_of,
                )
            )
            written += 1
        elif current.value != value:
            current.value = value
            current.as_of = as_of
            written += 1

    session.flush()
    return written


def ingest_ticker(
    session: Session,
    provider: DataProvider,
    ticker: str,
    start: date,
    end: date,
    *,
    frequency: Frequency = Frequency.DAILY,
    raw_dir: Path | None = None,
    as_of_date: date | None = None,
) -> IngestResult:
    """Ingest one ticker over ``[start, end]``.

    ``as_of_date`` is the point-in-time reference used to resolve the ticker to a
    ``security_id`` — it defaults to ``end`` (the most recent requested session),
    never "today". Every path writes an ``ingest_log`` row; a provider failure is
    recorded as ``status='failed'`` and re-raised so a partial run cannot pass
    for a complete one.
    """

    settings = get_settings()
    raw_dir = raw_dir or settings.raw_dir
    resolution_date = as_of_date or end
    as_of = datetime.now(UTC)

    security_id = get_or_create_security(
        session, ticker, resolution_date, source=provider.source
    )
    security_ref = SecurityRef(security_id=security_id, symbol=ticker)

    try:
        ohlcv = provider.fetch_ohlcv(security_ref, start, end, frequency)
        actions = provider.fetch_corporate_actions(security_ref)
        delisting = provider.fetch_delisting_info(security_ref)
    except ProviderError as exc:
        session.add(
            IngestLog(
                security_id=security_id,
                ticker=ticker,
                source=provider.source,
                start_date=start,
                end_date=end,
                rows_ingested=0,
                status="failed",
                error=str(exc)[:1024],
            )
        )
        session.flush()
        log.error(
            "ingest.failed",
            ticker=ticker,
            security_id=security_id,
            source=provider.source,
            error=str(exc),
        )
        raise

    # Immutable raw landing first, then the cleaned DB rows.
    write_raw_ohlcv(ohlcv, raw_dir=raw_dir, source=provider.source, ticker=ticker)

    rows_written = _upsert_ohlcv(
        session, security_id, ohlcv, source=provider.source, as_of=as_of
    )
    ca_written = _upsert_corporate_actions(
        session, security_id, actions, source=provider.source, as_of=as_of
    )

    if delisting.delisted:
        security = session.get(Security, security_id)
        if security is not None:
            security.delisting_date = delisting.delisting_date
            security.delisting_reason = delisting.reason

    session.add(
        IngestLog(
            security_id=security_id,
            ticker=ticker,
            source=provider.source,
            start_date=start,
            end_date=end,
            rows_ingested=rows_written,
            status="success",
        )
    )
    session.flush()

    log.info(
        "ingest.ok",
        ticker=ticker,
        security_id=security_id,
        source=provider.source,
        rows_written=rows_written,
        rows_total=len(ohlcv),
        corporate_actions=ca_written,
    )
    return IngestResult(
        ticker=ticker,
        security_id=security_id,
        source=provider.source,
        start=start,
        end=end,
        rows_written=rows_written,
        rows_total=len(ohlcv),
        corporate_actions_written=ca_written,
        status="success",
    )


def last_successful_end(session: Session, security_id: str, source: str) -> date | None:
    """Latest ``end_date`` of a successful ingest for a security + source."""

    stmt = (
        select(IngestLog.end_date)
        .where(IngestLog.security_id == security_id)
        .where(IngestLog.source == source)
        .where(IngestLog.status == "success")
        .order_by(IngestLog.end_date.desc())
        .limit(1)
    )
    return session.execute(stmt).scalars().first()


def ingest_incremental(
    session: Session,
    provider: DataProvider,
    *,
    until: date | None = None,
    raw_dir: Path | None = None,
) -> list[IngestResult]:
    """Fetch only new data since each security's last successful ingest.

    For every security that has a prior successful ingest from ``provider``'s
    source, resume from the day after its recorded ``end_date`` up to ``until``
    (default: today). Securities already current, or with no prior ingest, are
    skipped — there is no range to infer.
    """

    until = until or datetime.now(UTC).date()
    results: list[IngestResult] = []

    security_ids = (
        session.execute(
            select(IngestLog.security_id)
            .where(IngestLog.source == provider.source)
            .where(IngestLog.status == "success")
            .distinct()
        )
        .scalars()
        .all()
    )

    for security_id in security_ids:
        last_end = last_successful_end(session, security_id, provider.source)
        if last_end is None:
            continue
        resume = last_end + timedelta(days=1)
        if resume > until:
            log.info(
                "ingest.incremental.skip",
                security_id=security_id,
                reason="already current",
                last_end=last_end.isoformat(),
            )
            continue

        # Resolve the ticker that was current at the resume date.
        ticker = _current_ticker(session, security_id, resume, provider.source)
        if ticker is None:
            log.warning(
                "ingest.incremental.no_ticker",
                security_id=security_id,
                as_of=resume.isoformat(),
            )
            continue

        results.append(
            ingest_ticker(
                session,
                provider,
                ticker,
                resume,
                until,
                raw_dir=raw_dir,
                as_of_date=resume,
            )
        )

    return results


def _current_ticker(
    session: Session, security_id: str, as_of: date, source: str
) -> str | None:
    from quantlab.db.models import TickerHistory

    stmt = (
        select(TickerHistory.ticker)
        .where(TickerHistory.security_id == security_id)
        .where(TickerHistory.start_date <= as_of)
        .where((TickerHistory.end_date.is_(None)) | (TickerHistory.end_date >= as_of))
        .order_by(TickerHistory.start_date.desc())
        .limit(1)
    )
    ticker = session.execute(stmt).scalars().first()
    if ticker is not None:
        return ticker
    # Fall back to the most recent ticker on record for this security.
    stmt2 = (
        select(TickerHistory.ticker)
        .where(TickerHistory.security_id == security_id)
        .order_by(TickerHistory.start_date.desc())
        .limit(1)
    )
    return session.execute(stmt2).scalars().first()
