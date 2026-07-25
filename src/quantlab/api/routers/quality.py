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

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, Field

from quantlab.api.deps import SessionDep
from quantlab.api.models import Num
from quantlab.data.validation import (
    DataQualityReport,
    Finding,
    Severity,
    latest_report,
    run_checks,
)

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
