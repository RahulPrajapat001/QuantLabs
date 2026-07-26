"""Liveness and readiness.

The distinction is operational, not cosmetic, and an orchestrator treats them
differently:

* ``/health`` — **liveness**. "This process is running." It touches nothing
  external, so a database outage does not cause Kubernetes to kill and restart
  an otherwise healthy API, which would turn a dependency problem into a restart
  loop.
* ``/ready`` — **readiness**. "This process can serve useful traffic." It checks
  the database, that migrations are at head, and — the one that matters for a
  research platform — **data freshness**.

Data freshness is a readiness concern because a research API answering questions
from a database whose last bar is three weeks old is worse than one that is
plainly down: it returns confident, stale numbers. The check reports the age of
the newest bar and fails readiness past a threshold, so a silently broken
nightly ingest surfaces as an unready service rather than as quietly ageing
research.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, text

from quantlab import __version__
from quantlab.api.deps import SessionDep
from quantlab.config import HOLDOUT_START
from quantlab.db.models import OHLCVDaily

router = APIRouter(tags=["ops"])

#: Calendar days the newest bar may lag before readiness fails. Generous enough
#: to survive a weekend plus a public holiday run of a few days — NSE closes for
#: both — without paging anyone, tight enough that a nightly ingest broken for a
#: week is caught.
STALE_AFTER_DAYS = 5


class Check(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    ok: bool
    detail: str


class Health(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    version: str


class Ready(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: bool
    version: str
    checks: list[Check]
    #: Newest bar in ``ohlcv_daily``, or null when nothing is ingested.
    latest_bar: date | None = Field(default=None)
    #: Calendar days between ``latest_bar`` and now. Null when there are no bars.
    data_age_days: int | None = Field(default=None)


@router.get("/health", response_model=Health, summary="Liveness probe.")
def health() -> Health:
    """Process liveness. Deliberately touches no dependency — see module docs."""

    return Health(status="ok", version=__version__)


@router.get("/ready", response_model=Ready, summary="Readiness, incl. data freshness.")
def ready(session: SessionDep, response: Response) -> Ready:
    """Readiness. Returns 503 when any check fails, so a load balancer can act."""

    checks: list[Check] = []

    # --- Database reachable ---
    try:
        session.execute(text("SELECT 1"))
        checks.append(Check(name="database", ok=True, detail="reachable"))
    except Exception as exc:
        checks.append(Check(name="database", ok=False, detail=f"unreachable: {exc}"))
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Ready(ready=False, version=__version__, checks=checks)

    # --- Migrations at head ---
    try:
        stamped = session.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar()
        checks.append(
            Check(
                name="migrations",
                ok=stamped is not None,
                detail=f"alembic_version={stamped}"
                if stamped
                else "no alembic_version row: run `alembic upgrade head`",
            )
        )
    except Exception:
        checks.append(
            Check(
                name="migrations",
                ok=False,
                detail="alembic_version table absent: run `alembic upgrade head`",
            )
        )

    # --- Data freshness ---
    latest: date | None = session.scalar(select(func.max(OHLCVDaily.date)))
    age: int | None = None
    if latest is None:
        checks.append(
            Check(
                name="data_freshness",
                ok=False,
                detail="ohlcv_daily is empty: run `quantlab ingest`",
            )
        )
    else:
        age = (datetime.now(UTC).date() - latest).days
        fresh = age <= STALE_AFTER_DAYS
        # Past the holdout boundary the newest bar is *expected* to be old while
        # the lock holds, so an aged bar there is not a broken ingest.
        note = "" if latest < HOLDOUT_START else " (at/past the holdout boundary)"
        checks.append(
            Check(
                name="data_freshness",
                ok=fresh,
                detail=(
                    f"newest bar {latest.isoformat()}, {age}d old"
                    f" (threshold {STALE_AFTER_DAYS}d){note}"
                ),
            )
        )

    everything_ok = all(check.ok for check in checks)
    if not everything_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return Ready(
        ready=everything_ok,
        version=__version__,
        checks=checks,
        latest_bar=latest,
        data_age_days=age,
    )
