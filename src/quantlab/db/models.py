"""SQLAlchemy ORM models for the QuantLab data platform.

These cover Module 1 (ingestion) and Module 2 (universe construction) of the
schema in ``docs/PRD.md`` §Module 14: ``securities``, ``ticker_history``,
``ohlcv_daily``, ``corporate_actions``, ``ingest_log``, ``index_membership`` and
``unresolved_membership_change``. Later milestones extend the same ``Base``.

Design notes:

* Every security is a stable internal ``security_id`` string — never a ticker.
  Tickers live in ``ticker_history`` with date ranges so renames resolve to one
  security_id (PRD §3, Identity).
* ``ohlcv_daily`` is unique on ``(security_id, date)`` — the natural key that
  makes ingestion idempotent.
* ``close`` and ``adj_close`` are stored separately and every price row carries
  an ``as_of`` timestamp, because adjustment factors change retroactively on a
  split and overwriting history makes yesterday's backtest irreproducible.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base shared by every QuantLab table."""


class Security(Base):
    """A stable internal identity for a listed (or delisted) company.

    Delisted securities keep their full price history; ``delisting_date`` and
    ``delisting_reason`` record the exit but the rows in ``ohlcv_daily`` remain.
    """

    __tablename__ = "securities"

    security_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    isin: Mapped[str | None] = mapped_column(String(12), unique=True, nullable=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    exchange: Mapped[str] = mapped_column(String(16), nullable=False)
    listing_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    delisting_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    delisting_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    tickers: Mapped[list[TickerHistory]] = relationship(
        back_populates="security", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Security {self.security_id} {self.name!r} {self.exchange}>"


class TickerHistory(Base):
    """Maps a ticker symbol to a ``security_id`` over a date range.

    A company that renames keeps one ``security_id`` and gains a second row: the
    old ticker's ``end_date`` closes and the new ticker's ``start_date`` opens.
    ``end_date`` is ``NULL`` while a ticker is still current.
    """

    __tablename__ = "ticker_history"
    __table_args__ = (
        UniqueConstraint(
            "ticker", "start_date", "source", name="uq_ticker_history_ticker_start"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    security_id: Mapped[str] = mapped_column(
        ForeignKey("securities.security_id", ondelete="CASCADE"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    #: ``NULL`` means the ticker is still current (open-ended).
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)

    security: Mapped[Security] = relationship(back_populates="tickers")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<TickerHistory {self.ticker} -> {self.security_id} "
            f"[{self.start_date}..{self.end_date or 'open'}]>"
        )


class OHLCVDaily(Base):
    """A single daily bar for one security, unique on ``(security_id, date)``."""

    __tablename__ = "ohlcv_daily"
    __table_args__ = (
        UniqueConstraint("security_id", "date", name="uq_ohlcv_security_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    security_id: Mapped[str] = mapped_column(
        ForeignKey("securities.security_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    #: Raw, unadjusted close — stored separately from ``adj_close`` on purpose.
    close: Mapped[float] = mapped_column(Float, nullable=False)
    #: Split/dividend-adjusted close; changes retroactively after a split.
    adj_close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    #: When this row's values were computed — point-in-time correctness.
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CorporateAction(Base):
    """A split, dividend or bonus for a security."""

    __tablename__ = "corporate_actions"
    __table_args__ = (
        UniqueConstraint(
            "security_id",
            "date",
            "action_type",
            name="uq_corp_action_security_date_type",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    security_id: Mapped[str] = mapped_column(
        ForeignKey("securities.security_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    action_type: Mapped[str] = mapped_column(String(16), nullable=False)
    #: Split ratio or per-share cash amount depending on ``action_type``.
    value: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class IngestLog(Base):
    """One row per ingest attempt, per security, per source.

    Incremental mode reads the latest ``end_date`` of a ``status='success'`` row
    to know where to resume. A failed attempt is recorded too, so a partial run
    never masquerades as complete.
    """

    __tablename__ = "ingest_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    security_id: Mapped[str] = mapped_column(
        ForeignKey("securities.security_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    rows_ingested: Mapped[int] = mapped_column(nullable=False, default=0)
    #: ``"success"`` or ``"failed"``.
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class IndexMembership(Base):
    """Point-in-time membership of a security in an index (PRD Module 2).

    Each row is a half-open interval ``[start_date, end_date)`` during which the
    security was a constituent:

    * ``start_date`` is the first date the security **is** a member (inclusive).
    * ``end_date`` is the first date it is **no longer** a member (exclusive);
      ``NULL`` means it is still a current constituent.

    The half-open convention makes reconstitution clean: on an effective date the
    added name is already in and the removed name is already out. Membership is
    *never* read from present-day constituents for a historical date — see
    :func:`quantlab.universe.membership.universe_as_of`.
    """

    __tablename__ = "index_membership"
    __table_args__ = (
        UniqueConstraint(
            "security_id",
            "index_name",
            "start_date",
            name="uq_index_membership_sec_index_start",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    security_id: Mapped[str] = mapped_column(
        ForeignKey("securities.security_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_name: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    #: ``NULL`` means the security is still a current constituent (open-ended).
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    #: The reconstitution filing / page revision this interval was derived from,
    #: so any membership row is traceable to a primary source (reproducibility).
    source_document: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<IndexMembership {self.security_id} in {self.index_name} "
            f"[{self.start_date}..{self.end_date or 'current'}]>"
        )


class UnresolvedMembershipChange(Base):
    """A reconstitution entry that could not be mapped to a ``security_id``.

    Rather than silently drop an addition or removal whose ticker/name we cannot
    resolve, we record it here (PRD Module 2, "Unresolved changes"). The share of
    entries that resolve is a **reported** data-quality metric, not something
    hidden — see :class:`quantlab.universe.reconstitution.ResolutionReport`.
    """

    __tablename__ = "unresolved_membership_change"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    index_name: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    #: ``"add"`` or ``"remove"``.
    action: Mapped[str] = mapped_column(String(8), nullable=False)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False)
    ticker: Mapped[str | None] = mapped_column(String(32), nullable=True)
    company_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source_document: Mapped[str] = mapped_column(String(512), nullable=False)
    #: Why resolution failed (e.g. "no ticker_history match as of 2018-03-01").
    reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<UnresolvedMembershipChange {self.action} "
            f"{self.ticker or self.company_name!r} {self.index_name} "
            f"@ {self.effective_date}>"
        )
