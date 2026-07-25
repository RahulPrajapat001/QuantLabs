"""Point-in-time index membership.

The endpoint that exists to make survivorship bias visible rather than to hide
it. ``/as-of`` answers "who was in the NIFTY 500 on this date", using
reconstitution history — not today's constituent list filtered backwards. The
difference between those two answers is the survivorship comparison the UI's
universe page is built to show, so both are served here and the gap is computed
server-side.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, Field

from quantlab.api.deps import SessionDep
from quantlab.api.formatting import format_date
from quantlab.api.models import Num
from quantlab.universe.membership import (
    current_constituents,
    membership_intervals,
    universe_as_of,
)

router = APIRouter(prefix="/universe", tags=["universe"])


class MembershipOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    index_name: str
    as_of: date
    as_of_display: str
    count: Num
    security_ids: list[str]


class SurvivorshipComparison(BaseModel):
    """Point-in-time membership against today's list, on the same date.

    ``survivorship_only`` are the securities a naive backtest would have traded
    on ``as_of`` purely because they are in the index *now*. Every one of them is
    a company the researcher could not have known to buy at the time.
    """

    model_config = ConfigDict(frozen=True)

    index_name: str
    as_of: date
    as_of_display: str
    point_in_time_count: Num
    current_count: Num
    #: In today's index but not in it on ``as_of`` — the look-ahead names.
    survivorship_only: list[str]
    #: In the index on ``as_of`` but not today — dropped, merged or delisted.
    dropped_since: list[str]
    overlap: Num
    #: Share of the point-in-time universe that survives to today.
    survival_rate: Num


class IntervalOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    security_id: str
    start: date
    end: date | None
    start_display: str
    end_display: str
    is_open: bool = Field(description="True while the security is still a member.")
    source_document: str = Field(
        description="The reconstitution filing this interval was derived from."
    )


@router.get(
    "/{index_name}/as-of",
    response_model=MembershipOut,
    summary="Constituents on a date, from reconstitution history.",
)
def membership_as_of(
    index_name: str,
    session: SessionDep,
    on: date = Query(description="The date to resolve membership for."),
) -> MembershipOut:
    members = sorted(universe_as_of(session, index_name, on))
    return MembershipOut(
        index_name=index_name,
        as_of=on,
        as_of_display=format_date(on),
        count=Num.count(len(members)),
        security_ids=members,
    )


@router.get(
    "/{index_name}/survivorship",
    response_model=SurvivorshipComparison,
    summary="Point-in-time membership vs today's list — the bias, quantified.",
)
def survivorship(
    index_name: str,
    session: SessionDep,
    on: date = Query(description="The historical date to compare against."),
) -> SurvivorshipComparison:
    historical = universe_as_of(session, index_name, on)
    current = current_constituents(session, index_name)
    overlap = historical & current
    return SurvivorshipComparison(
        index_name=index_name,
        as_of=on,
        as_of_display=format_date(on),
        point_in_time_count=Num.count(len(historical)),
        current_count=Num.count(len(current)),
        survivorship_only=sorted(current - historical),
        dropped_since=sorted(historical - current),
        overlap=Num.count(len(overlap)),
        survival_rate=Num.share(len(overlap) / len(historical) if historical else None),
    )


@router.get(
    "/{index_name}/intervals",
    response_model=list[IntervalOut],
    summary="Every membership interval, for the date-scrubber timeline.",
)
def intervals(index_name: str, session: SessionDep) -> list[IntervalOut]:
    return [
        IntervalOut(
            security_id=interval.security_id,
            start=interval.start_date,
            end=interval.end_date,
            start_display=format_date(interval.start_date),
            end_display=format_date(interval.end_date),
            is_open=interval.end_date is None,
            source_document=interval.source_document,
        )
        for interval in membership_intervals(session, index_name)
    ]
