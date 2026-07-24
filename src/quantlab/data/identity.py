"""Ticker → ``security_id`` resolution.

This is the easy-to-get-wrong core of the data layer. Two rules govern it:

1. A ticker is resolved to a ``security_id`` **as of the requested date**, using
   ``ticker_history`` — never as of today. Ticker strings are reused across
   companies over time; only a point-in-time lookup is correct.
2. A company that renames keeps **one** ``security_id``. The rename is two rows
   in ``ticker_history`` (old ticker closed, new ticker opened), so both tickers
   resolve to the same security.

When a ticker is entirely unknown, :func:`get_or_create_security` mints a new
stable ``security_id`` and opens a ``ticker_history`` row — it never guesses an
existing identity.
"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from quantlab.db.models import Security, TickerHistory


def _infer_exchange(ticker: str) -> str:
    """Best-effort exchange from a vendor suffix.

    ``.NS`` → NSE (the research exchange). US symbols carry no suffix; they exist
    only to validate the provider abstraction and are labelled ``US``.
    """

    upper = ticker.upper()
    if upper.endswith(".NS"):
        return "NSE"
    if upper.endswith(".BO"):
        return "BSE"
    return "US"


def new_security_id() -> str:
    """Mint a fresh, opaque, stable internal identity."""

    return f"SEC-{uuid.uuid4().hex[:12]}"


def resolve_security_id(session: Session, ticker: str, as_of: date) -> str | None:
    """Return the ``security_id`` a ticker mapped to on ``as_of``, or ``None``.

    A row matches when ``start_date <= as_of`` and the row is either open
    (``end_date IS NULL``) or ``as_of <= end_date``. If multiple rows somehow
    match, the one with the latest ``start_date`` wins (most specific window).
    """

    stmt = (
        select(TickerHistory)
        .where(TickerHistory.ticker == ticker)
        .where(TickerHistory.start_date <= as_of)
        .where((TickerHistory.end_date.is_(None)) | (TickerHistory.end_date >= as_of))
        .order_by(TickerHistory.start_date.desc())
    )
    row = session.execute(stmt).scalars().first()
    return row.security_id if row is not None else None


def get_or_create_security(
    session: Session,
    ticker: str,
    as_of: date,
    *,
    source: str,
    name: str | None = None,
    isin: str | None = None,
    exchange: str | None = None,
    listing_date: date | None = None,
) -> str:
    """Resolve ``ticker`` as of ``as_of``, creating a new security if unknown.

    The lookup is point-in-time (:func:`resolve_security_id`). Only when no
    existing identity matches is a new ``security_id`` minted together with an
    open ``ticker_history`` row starting at ``as_of``.
    """

    existing = resolve_security_id(session, ticker, as_of)
    if existing is not None:
        return existing

    security_id = new_security_id()
    session.add(
        Security(
            security_id=security_id,
            isin=isin,
            name=name or ticker,
            exchange=exchange or _infer_exchange(ticker),
            listing_date=listing_date,
        )
    )
    session.add(
        TickerHistory(
            security_id=security_id,
            ticker=ticker,
            start_date=listing_date or as_of,
            end_date=None,
            source=source,
        )
    )
    session.flush()
    return security_id
