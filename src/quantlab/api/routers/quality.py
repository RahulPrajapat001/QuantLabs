"""Data quality findings.

**Computed on demand, not read from a table.** PRD Module 3 specifies a
persisted ``data_quality_findings`` table and it has not been built; rather than
serve an endpoint that reads a table that does not exist, this runs the checks
against ``ohlcv_daily`` when asked. The trade-off is honest and worth stating: a
finding here has no id, no history, and no acknowledged/resolved state, so it
cannot be triaged over time. When Module 3 lands, this endpoint should read the
table and these checks should move into the ingest path.

One category *is* persisted already — ``unresolved_membership_change``, the
reconstitution entries the universe builder could not resolve. Those are surfaced
alongside, because a universe with unresolved changes is a universe whose
point-in-time membership is incomplete, which is a data quality problem of the
first order.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from quantlab.api.deps import SessionDep
from quantlab.api.formatting import format_date
from quantlab.api.models import Num
from quantlab.db.models import (
    CorporateAction,
    OHLCVDaily,
    Security,
    UnresolvedMembershipChange,
)

router = APIRouter(prefix="/quality", tags=["quality"])

#: A single-day move beyond this is either a real event, a corporate action the
#: adjustment missed, or a bad print. All three are worth a human looking.
EXTREME_MOVE = 0.20
#: Consecutive identical closes that suggest a stale feed rather than a quiet
#: stock. Five sessions is a trading week.
STALE_RUN = 5


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)

    check: str
    severity: Severity
    security_id: str | None
    date: date | None
    date_display: str
    detail: str
    #: What a researcher should do about it, not just what is wrong.
    remedy: str


class QualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_on_demand: bool = Field(
        default=True,
        description=(
            "These findings are computed per request. There is no persisted "
            "data_quality_findings table yet (PRD Module 3)."
        ),
    )
    checks_run: list[str]
    rows_examined: Num
    counts: dict[str, int]
    findings: list[Finding]


@router.get("/findings", response_model=QualityReport, summary="Run the checks now.")
def findings(
    session: SessionDep,
    security_id: str | None = Query(default=None),
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5_000),
) -> QualityReport:
    stmt = select(
        OHLCVDaily.security_id,
        OHLCVDaily.date,
        OHLCVDaily.open,
        OHLCVDaily.high,
        OHLCVDaily.low,
        OHLCVDaily.close,
        OHLCVDaily.adj_close,
        OHLCVDaily.volume,
    )
    if security_id is not None:
        stmt = stmt.where(OHLCVDaily.security_id == security_id)
    if start is not None:
        stmt = stmt.where(OHLCVDaily.date >= start)
    if end is not None:
        stmt = stmt.where(OHLCVDaily.date <= end)

    rows = session.execute(stmt.order_by(OHLCVDaily.security_id, OHLCVDaily.date)).all()
    panel = pd.DataFrame(
        rows,
        columns=[
            "security_id",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ],
    )

    found: list[Finding] = []
    checks = [
        "non_positive_price",
        "inconsistent_ohlc",
        "zero_volume",
        "extreme_move",
        "stale_price",
        "missing_adjustment",
        "no_price_history",
        "unresolved_membership_change",
    ]

    if not panel.empty:
        action_dates = _corporate_action_dates(session)
        found += _price_checks(panel)
        found += _series_checks(panel, action_dates)
    found += _coverage_checks(session)
    found += _membership_checks(session)

    counts: dict[str, int] = {}
    for finding in found:
        counts[finding.check] = counts.get(finding.check, 0) + 1

    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    found.sort(
        key=lambda f: (
            order[f.severity],
            f.check,
            f.security_id or "",
            f.date or date.min,
        )
    )

    return QualityReport(
        checks_run=checks,
        rows_examined=Num.count(len(panel)),
        counts=counts,
        findings=found[:limit],
    )


def _finding(
    check: str,
    severity: Severity,
    detail: str,
    remedy: str,
    *,
    security_id: str | None = None,
    when: date | None = None,
) -> Finding:
    return Finding(
        check=check,
        severity=severity,
        security_id=security_id,
        date=when,
        date_display=format_date(when),
        detail=detail,
        remedy=remedy,
    )


def _as_date(value: object) -> date | None:
    """``itertuples`` erases column types, so narrow at the boundary."""

    if isinstance(value, date):
        return value
    if value is None:
        return None
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def _price_checks(panel: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []

    bad = panel[(panel[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    for row in bad.itertuples():
        out.append(
            _finding(
                "non_positive_price",
                Severity.ERROR,
                f"close={row.close}, low={row.low} — a price must be positive",
                "Re-ingest this security; the vendor row is corrupt.",
                security_id=str(row.security_id),
                when=_as_date(row.date),
            )
        )

    # high must bound the day; low must floor it. A violation means the bar is
    # internally inconsistent and every derived range/ATR feature is wrong.
    inconsistent = panel[
        (panel["high"] < panel["low"])
        | (panel["high"] < panel["close"])
        | (panel["low"] > panel["close"])
    ]
    for row in inconsistent.itertuples():
        out.append(
            _finding(
                "inconsistent_ohlc",
                Severity.ERROR,
                f"high={row.high}, low={row.low}, close={row.close} do not bracket",
                "Re-ingest; range-based features (ATR, Parkinson) are invalid here.",
                security_id=str(row.security_id),
                when=_as_date(row.date),
            )
        )

    zero_volume = panel[panel["volume"] <= 0]
    for row in zero_volume.itertuples():
        out.append(
            _finding(
                "zero_volume",
                Severity.WARNING,
                "zero volume — the security did not trade",
                "Check for a halt or suspension; a fill here would be fictional.",
                security_id=str(row.security_id),
                when=_as_date(row.date),
            )
        )
    return out


def _series_checks(
    panel: pd.DataFrame, action_dates: set[tuple[str, date]]
) -> list[Finding]:
    out: list[Finding] = []
    for security_id, frame in panel.groupby("security_id", sort=True):
        frame = frame.sort_values("date")
        returns = frame["adj_close"].pct_change()

        for idx, value in returns.items():
            if pd.isna(value) or abs(value) <= EXTREME_MOVE:
                continue
            when = frame.at[idx, "date"]
            if (str(security_id), when) in action_dates:
                continue  # explained by a recorded corporate action
            out.append(
                _finding(
                    "extreme_move",
                    Severity.WARNING,
                    f"{value:+.1%} in one session on adjusted close, with no "
                    "corporate action recorded",
                    "Verify against a second source; an unadjusted split looks "
                    "exactly like this.",
                    security_id=str(security_id),
                    when=_as_date(when),
                )
            )

        # A run of identical closes is a stale feed far more often than it is a
        # genuinely motionless stock.
        closes = frame["close"].to_numpy()
        run = 1
        for i in range(1, len(closes)):
            if closes[i] == closes[i - 1]:
                run += 1
                continue
            if run >= STALE_RUN:
                out.append(
                    _finding(
                        "stale_price",
                        Severity.WARNING,
                        f"close unchanged for {run} consecutive sessions",
                        "Check the vendor feed for a gap filled by carry-forward.",
                        security_id=str(security_id),
                        when=_as_date(frame["date"].to_numpy()[i - 1]),
                    )
                )
            run = 1

        # adj_close identical to close across the whole history means either no
        # corporate action ever occurred, or the adjustment was never applied.
        if len(frame) > 250 and (frame["close"] == frame["adj_close"]).all():
            has_actions = any(sid == str(security_id) for sid, _ in action_dates)
            if has_actions:
                out.append(
                    _finding(
                        "missing_adjustment",
                        Severity.ERROR,
                        "corporate actions are recorded but adj_close never "
                        "differs from close",
                        "The adjustment factor was not applied; every return "
                        "computed from this security is wrong.",
                        security_id=str(security_id),
                    )
                )
    return out


def _corporate_action_dates(session: SessionDep) -> set[tuple[str, date]]:
    rows = session.execute(
        select(CorporateAction.security_id, CorporateAction.date)
    ).all()
    return {(sid, when) for sid, when in rows}


def _coverage_checks(session: SessionDep) -> list[Finding]:
    priced = select(OHLCVDaily.security_id).distinct().subquery()
    missing = session.scalars(
        select(Security.security_id).where(
            Security.security_id.not_in(select(priced.c.security_id))
        )
    )
    return [
        _finding(
            "no_price_history",
            Severity.ERROR,
            "the security exists but has no OHLCV rows at all",
            "Ingest it, or it will silently vanish from every universe.",
            security_id=security_id,
        )
        for security_id in missing
    ]


def _membership_checks(session: SessionDep) -> list[Finding]:
    total = int(
        session.scalar(select(func.count()).select_from(UnresolvedMembershipChange))
        or 0
    )
    if not total:
        return []
    rows = session.scalars(select(UnresolvedMembershipChange).limit(200))
    return [
        _finding(
            "unresolved_membership_change",
            Severity.ERROR,
            f"{row.index_name} {row.action} of "
            f"{row.ticker or row.company_name or 'unknown'!r} could not be "
            f"resolved to a security_id ({row.reason or 'no reason recorded'})",
            "Point-in-time membership is incomplete until this resolves; every "
            "backtest over the affected window is missing a constituent.",
            when=row.effective_date,
        )
        for row in rows
    ]
