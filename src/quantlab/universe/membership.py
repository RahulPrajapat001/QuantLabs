"""Point-in-time index membership — the core of a credible backtest.

The whole platform's defence against **survivorship bias** lives here. Membership
is stored as half-open intervals ``[start_date, end_date)`` in ``index_membership``
and queried with :func:`universe_as_of`, which returns *exactly* the constituents
on a given date and **never** consults present-day membership.

Convention (matches :class:`quantlab.db.models.IndexMembership`):

* ``start_date`` — first date the security **is** a member (inclusive).
* ``end_date`` — first date it is **no longer** a member (exclusive); ``NULL``
  while it is still a current constituent.

So a name added effective ``A`` is a member from ``A`` onward, and a name removed
effective ``R`` is a member up to and including ``R - 1`` day. This makes the
"day before / day after a change" boundary unambiguous.

Three resolvers implement one small :class:`UniverseResolver` protocol so the
backtester and the survivorship harness can be pointed at different notions of
"the universe" without knowing how each is computed:

* :class:`PointInTimeUniverse` — the honest, date-aware constituents.
* :class:`FixedUniverse` — constituents frozen at one date (``--universe-mode
  fixed``); removes *selection* bias but not delisting attrition.
* :class:`CurrentUniverse` — today's constituents applied to every date; this is
  the survivorship-biased universe used only as a comparison baseline.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Protocol, runtime_checkable

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from quantlab.db.models import IndexMembership

log = structlog.get_logger(__name__)


class UniverseMode(StrEnum):
    """How constituents are determined over the life of a backtest."""

    #: Date-aware membership from reconstitution history. The only mode valid for
    #: a reported research result.
    POINT_IN_TIME = "point-in-time"
    #: Constituents frozen at a single ``as_of`` date and held constant.
    FIXED = "fixed"


@dataclass(frozen=True, slots=True)
class MembershipInterval:
    """One ``[start_date, end_date)`` membership span for a security."""

    security_id: str
    index_name: str
    start_date: date
    #: ``None`` means still a current constituent (open-ended).
    end_date: date | None
    source_document: str


# --- The point-in-time query -------------------------------------------------


def universe_as_of(session: Session, index_name: str, on: date) -> set[str]:
    """Return the ``security_id`` set constituting ``index_name`` on ``on``.

    A security is in the universe when ``start_date <= on`` and it has not yet
    left (``end_date IS NULL`` or ``on < end_date``). This reads only historical
    interval rows; it can never leak today's membership into a past date.
    """

    stmt = (
        select(IndexMembership.security_id)
        .where(IndexMembership.index_name == index_name)
        .where(IndexMembership.start_date <= on)
        .where((IndexMembership.end_date.is_(None)) | (IndexMembership.end_date > on))
        .distinct()
    )
    return set(session.execute(stmt).scalars().all())


def current_constituents(session: Session, index_name: str) -> set[str]:
    """Return the *present-day* constituents (open intervals only).

    This is the survivorship-biased set — every name that has ever left the index
    is missing. It exists solely to be compared against the point-in-time
    universe in the survivorship measurement; never build a research result on it.
    """

    stmt = (
        select(IndexMembership.security_id)
        .where(IndexMembership.index_name == index_name)
        .where(IndexMembership.end_date.is_(None))
        .distinct()
    )
    return set(session.execute(stmt).scalars().all())


def membership_intervals(session: Session, index_name: str) -> list[MembershipInterval]:
    """All stored intervals for ``index_name``, ordered by security then start."""

    stmt = (
        select(IndexMembership)
        .where(IndexMembership.index_name == index_name)
        .order_by(IndexMembership.security_id, IndexMembership.start_date)
    )
    rows = session.execute(stmt).scalars().all()
    return [
        MembershipInterval(
            security_id=r.security_id,
            index_name=r.index_name,
            start_date=r.start_date,
            end_date=r.end_date,
            source_document=r.source_document,
        )
        for r in rows
    ]


# --- Building intervals from add / remove events -----------------------------


def build_intervals_from_events(
    events: Iterable[tuple[str, str, date, str]],
) -> list[MembershipInterval]:
    """Fold a stream of ``(security_id, action, effective_date, source)`` events
    into half-open membership intervals, per (security, index-less) key.

    ``action`` is ``"add"`` or ``"remove"``. Events are processed in date order.
    An ``add`` while already a member is ignored (idempotent); a ``remove`` while
    not a member is logged and skipped (our history does not reach the matching
    addition). The caller supplies ``index_name`` separately via the returned
    intervals — see :func:`insert_membership_intervals`, which stamps it.

    Note: this helper is index-agnostic on purpose; group events by index before
    calling it. It returns intervals with ``index_name=""`` to be filled in.
    """

    ordered = sorted(events, key=lambda e: (e[0], e[2]))
    intervals: list[MembershipInterval] = []

    open_start: dict[str, tuple[date, str]] = {}
    for security_id, action, effective_date, source in ordered:
        if action == "add":
            if security_id in open_start:
                continue  # already a member; ignore duplicate addition
            open_start[security_id] = (effective_date, source)
        elif action == "remove":
            opened = open_start.pop(security_id, None)
            if opened is None:
                log.warning(
                    "universe.remove_without_open",
                    security_id=security_id,
                    effective_date=effective_date.isoformat(),
                )
                continue
            start, start_source = opened
            intervals.append(
                MembershipInterval(
                    security_id=security_id,
                    index_name="",
                    start_date=start,
                    end_date=effective_date,
                    source_document=start_source,
                )
            )
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown membership action {action!r}")

    # Anything still open at the end is a current constituent (open-ended).
    for security_id, (start, start_source) in open_start.items():
        intervals.append(
            MembershipInterval(
                security_id=security_id,
                index_name="",
                start_date=start,
                end_date=None,
                source_document=start_source,
            )
        )
    return intervals


def insert_membership_intervals(
    session: Session,
    index_name: str,
    intervals: Iterable[MembershipInterval],
) -> int:
    """Insert intervals for ``index_name``, skipping exact ``start_date`` dupes.

    Idempotent on ``(security_id, index_name, start_date)`` — re-ingesting the
    same filings does not create duplicate rows. Returns the number inserted.
    """

    existing = {
        (m.security_id, m.start_date) for m in membership_intervals(session, index_name)
    }
    inserted = 0
    for interval in intervals:
        key = (interval.security_id, interval.start_date)
        if key in existing:
            continue
        session.add(
            IndexMembership(
                security_id=interval.security_id,
                index_name=index_name,
                start_date=interval.start_date,
                end_date=interval.end_date,
                source_document=interval.source_document,
            )
        )
        existing.add(key)
        inserted += 1
    session.flush()
    return inserted


# --- Resolvers ---------------------------------------------------------------


@runtime_checkable
class UniverseResolver(Protocol):
    """Anything that can name the constituents active on a given date."""

    index_name: str

    def constituents(self, on: date) -> set[str]:
        """Return the ``security_id`` set active on ``on``."""
        ...


@dataclass(frozen=True)
class PointInTimeUniverse:
    """Date-aware constituents from reconstitution history (the honest universe)."""

    session: Session
    index_name: str

    def constituents(self, on: date) -> set[str]:
        return universe_as_of(self.session, self.index_name, on)


@dataclass(frozen=True)
class FixedUniverse:
    """Constituents frozen at one date and held constant (``--universe-mode fixed``).

    This removes *selection* bias (the frozen set is a real historical universe,
    not today's winners) but **not** delisting attrition: a name that delists
    during the backtest still must be liquidated, not held. That caveat is
    documented wherever fixed mode is offered — see the CLI ``--universe-mode``
    help and ``docs/research/survivorship_bias.md``.
    """

    index_name: str
    frozen_at: date
    members: frozenset[str]

    def constituents(self, on: date) -> set[str]:
        return set(self.members)


@dataclass(frozen=True)
class CurrentUniverse:
    """Today's constituents applied to *every* date — the biased baseline.

    Used only by the survivorship measurement to quantify how much the bias
    inflates results. It ignores ``on`` entirely, which is precisely the flaw
    being measured.
    """

    session: Session
    index_name: str

    def constituents(self, on: date) -> set[str]:
        return current_constituents(self.session, self.index_name)


def freeze_fixed_universe(
    session: Session, index_name: str, as_of: date
) -> FixedUniverse:
    """Snapshot the point-in-time universe at ``as_of`` into a fixed resolver."""

    members = universe_as_of(session, index_name, as_of)
    return FixedUniverse(
        index_name=index_name, frozen_at=as_of, members=frozenset(members)
    )
