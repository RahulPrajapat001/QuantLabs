"""Data quality findings.

Served from the persisted ``data_quality_findings`` table (PRD Module 3) when a
validation run has written one, and computed on demand when it has not. The
response says which happened: ``generated_on_demand`` is no longer a hardcoded
``True`` but a fact about where these rows came from, and a persisted report
carries the ``run_id`` and timestamp of the run that produced it.

The fallback is not a formality — it is the normal state of a database that has
never had :func:`quantlab.data.validation.validate` run against it, and of one
with no defects at all (a clean run writes no rows, so it leaves no trace).

The checks themselves live in :mod:`quantlab.data.validation`. This module is
presentation: filtering, limiting, and the number formatting the frontend needs.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from quantlab.api.deps import SessionDep
from quantlab.api.models import Num
from quantlab.data.validation import (
    DataQualityReport,
    Finding,
    Severity,
    latest_report,
    run_checks,
)
from quantlab.db.models import OHLCVDaily

__all__ = ["Finding", "QualityReport", "Severity", "router"]

router = APIRouter(prefix="/quality", tags=["quality"])


class QualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_on_demand: bool = Field(
        description=(
            "True when these findings were computed for this request rather than "
            "read from data_quality_findings. An on-demand finding has no id and "
            "no history: run `quantlab validate` to persist one."
        ),
    )
    run_id: str | None = Field(
        default=None,
        description="The validation run these findings came from. Null on demand.",
    )
    generated_at: datetime | None = Field(
        default=None,
        description="When the persisted run executed. Null on demand.",
    )
    checks_run: list[str]
    rows_examined: Num
    counts: dict[str, int]
    findings: list[Finding]


class CompletenessCell(BaseModel):
    """One security-month: bars present against NSE sessions expected."""

    model_config = ConfigDict(frozen=True)

    security_id: str
    #: ``YYYY-MM``.
    month: str
    bars: Num
    expected: Num
    #: ``bars / expected``, capped at 1.0. Null when the security was not listed
    #: for any part of the month — an unlisted month is not an incomplete one,
    #: and shading it as a gap would invent a data problem.
    completeness: Num


class CompletenessGrid(BaseModel):
    model_config = ConfigDict(frozen=True)

    securities: list[str]
    months: list[str]
    cells: list[CompletenessCell]
    #: Bars present over bars expected, across every cell.
    overall: Num
    truncated: bool = Field(
        description="True when more securities exist than the requested limit."
    )


@router.get(
    "/completeness",
    response_model=CompletenessGrid,
    summary="Bar coverage per security per month.",
)
def completeness(
    session: SessionDep,
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    limit: int = Query(
        default=40,
        ge=1,
        le=200,
        description="Securities per grid. A heatmap of 500 rows is not readable.",
    ),
) -> CompletenessGrid:
    """Expected sessions come from the observed trading calendar, not a count of
    weekdays: the union of dates on which *any* security traded is the exchange
    calendar as this database knows it, so holidays never read as missing data.
    """

    stmt = select(OHLCVDaily.security_id, OHLCVDaily.date)
    if start is not None:
        stmt = stmt.where(OHLCVDaily.date >= start)
    if end is not None:
        stmt = stmt.where(OHLCVDaily.date <= end)
    rows = session.execute(stmt).all()

    if not rows:
        return CompletenessGrid(
            securities=[],
            months=[],
            cells=[],
            overall=Num.share(None),
            truncated=False,
        )

    frame = pd.DataFrame(rows, columns=["security_id", "date"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.strftime("%Y-%m")

    # Sessions the exchange actually had, per month.
    sessions = frame.groupby("month")["date"].nunique()
    months = sorted(sessions.index)

    all_securities = sorted(frame["security_id"].unique())
    securities = all_securities[:limit]

    # Listed span per security, so months before listing or after delisting are
    # reported as null rather than 0%.
    spans = frame.groupby("security_id")["month"].agg(["min", "max"])
    counted = frame.groupby(["security_id", "month"]).size()

    cells: list[CompletenessCell] = []
    present_total = 0
    expected_total = 0
    for security_id in securities:
        first, last = spans.loc[security_id, "min"], spans.loc[security_id, "max"]
        for month in months:
            expected = int(sessions[month])
            if month < first or month > last:
                cells.append(
                    CompletenessCell(
                        security_id=security_id,
                        month=month,
                        bars=Num.count(0),
                        expected=Num.count(expected),
                        completeness=Num.share(None),
                    )
                )
                continue
            bars = int(counted.get((security_id, month), 0))
            present_total += bars
            expected_total += expected
            cells.append(
                CompletenessCell(
                    security_id=security_id,
                    month=month,
                    bars=Num.count(bars),
                    expected=Num.count(expected),
                    completeness=Num.share(
                        min(bars / expected, 1.0) if expected else None
                    ),
                )
            )

    return CompletenessGrid(
        securities=securities,
        months=months,
        cells=cells,
        overall=Num.share(present_total / expected_total if expected_total else None),
        truncated=len(all_securities) > len(securities),
    )


@router.get(
    "/findings",
    response_model=QualityReport,
    summary="Persisted findings, or the checks run now.",
)
def findings(
    session: SessionDep,
    security_id: str | None = Query(default=None),
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5_000),
    refresh: bool = Query(
        default=False,
        description=(
            "Skip the table and run the checks now. Does not persist — writing "
            "is `quantlab validate`, not a GET."
        ),
    ),
) -> QualityReport:
    report: DataQualityReport | None = None
    if not refresh:
        report = latest_report(session, security_id=security_id, start=start, end=end)
    if report is None:
        report = run_checks(session, security_id=security_id, start=start, end=end)

    return QualityReport(
        generated_on_demand=not report.persisted,
        run_id=report.run_id,
        generated_at=report.generated_at,
        checks_run=report.checks_run,
        rows_examined=Num.count(report.rows_examined),
        counts=report.counts,
        findings=report.findings[:limit],
    )
