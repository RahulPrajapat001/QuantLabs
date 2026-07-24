"""Delisting handling for the universe and backtester (PRD Module 2).

Rules, verbatim from the specification:

* A security delisted mid-backtest is **liquidated at its last available price**
  on the delisting date.
* If ``delisting_reason`` indicates **bankruptcy**, a configurable **recovery
  rate** applies (default ``0`` — a total loss).
* A delisted price is **never forward-filled**. The last real bar is used; no
  synthetic post-delisting price is ever invented.

This module reads the last actual ``ohlcv_daily`` bar at or before the delisting
date and computes the liquidation proceeds. It deliberately does not fabricate a
price when none exists — a delisting with no prior bar returns ``None`` so the
caller must handle it explicitly rather than silently assuming a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from quantlab.config import get_settings
from quantlab.db.models import OHLCVDaily, Security

#: Substrings (case-folded) in ``delisting_reason`` that mark a bankruptcy /
#: insolvency exit, where the recovery rate applies. Chosen for NSE + US wording
#: ("NCLT" and "IBC" are the Indian insolvency forums / code).
BANKRUPTCY_KEYWORDS: tuple[str, ...] = (
    "bankrupt",
    "insolven",
    "liquidat",
    "wind up",
    "winding up",
    "nclt",
    "ibc",
    "chapter 11",
    "chapter 7",
)


def is_bankruptcy(reason: str | None) -> bool:
    """True when ``reason`` names a bankruptcy / insolvency-style delisting."""

    if not reason:
        return False
    folded = reason.casefold()
    return any(keyword in folded for keyword in BANKRUPTCY_KEYWORDS)


def liquidation_proceeds(
    last_price: float,
    *,
    reason: str | None,
    recovery_rate: float,
) -> float:
    """Proceeds per share from liquidating a delisted position.

    A normal (non-bankruptcy) delisting recovers the full last price. A
    bankruptcy recovers ``last_price * recovery_rate`` (default rate ``0``).
    """

    if is_bankruptcy(reason):
        return last_price * recovery_rate
    return last_price


@dataclass(frozen=True, slots=True)
class LiquidationResult:
    """Outcome of liquidating one delisted position."""

    security_id: str
    delisting_date: date
    #: The date of the actual last bar used (never after ``delisting_date``).
    price_date: date
    #: Last available raw close used as the liquidation price.
    last_price: float
    #: Proceeds per share after any bankruptcy recovery haircut.
    proceeds: float
    bankruptcy: bool
    recovery_rate: float
    reason: str | None


def last_available_bar(
    session: Session, security_id: str, on_or_before: date
) -> tuple[date, float] | None:
    """Return ``(date, close)`` of the latest actual bar at/before ``on_or_before``.

    This is a genuine traded bar, never a forward-filled or synthesised value. If
    no bar exists on or before the date, returns ``None``.
    """

    stmt = (
        select(OHLCVDaily.date, OHLCVDaily.close)
        .where(OHLCVDaily.security_id == security_id)
        .where(OHLCVDaily.date <= on_or_before)
        .order_by(OHLCVDaily.date.desc())
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None
    return (row[0], float(row[1]))


def liquidate_position(
    session: Session,
    security_id: str,
    *,
    delisting_date: date | None = None,
    reason: str | None = None,
    recovery_rate: float | None = None,
) -> LiquidationResult | None:
    """Compute the liquidation of a delisted position at its last real price.

    ``delisting_date`` and ``reason`` default to the values recorded on the
    ``securities`` row; ``recovery_rate`` defaults to
    ``settings.bankruptcy_recovery_rate``. Returns ``None`` when the security is
    not delisted or has no bar at/before the delisting date (the caller must then
    decide — we never invent a price).
    """

    security = session.get(Security, security_id)
    if security is None:
        return None

    effective_date = delisting_date or security.delisting_date
    if effective_date is None:
        return None  # not delisted — nothing to liquidate

    effective_reason = reason if reason is not None else security.delisting_reason
    rate = (
        recovery_rate
        if recovery_rate is not None
        else get_settings().bankruptcy_recovery_rate
    )

    bar = last_available_bar(session, security_id, effective_date)
    if bar is None:
        return None  # no real price to liquidate at; never forward-fill one

    price_date, last_price = bar
    bankruptcy = is_bankruptcy(effective_reason)
    proceeds = liquidation_proceeds(
        last_price, reason=effective_reason, recovery_rate=rate
    )
    return LiquidationResult(
        security_id=security_id,
        delisting_date=effective_date,
        price_date=price_date,
        last_price=last_price,
        proceeds=proceeds,
        bankruptcy=bankruptcy,
        recovery_rate=rate,
        reason=effective_reason,
    )
