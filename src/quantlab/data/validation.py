"""Data validation — PRD Module 3.

Automated checks over cleaned ``ohlcv_daily`` producing structured
:class:`Finding` objects, persisted to ``data_quality_findings``. These checks
previously lived inside the API router and were recomputed per request, which
meant a finding had no identity and no history: the same bad print looked new
every time it was looked at, and nothing could be compared against a previous
run. They live here now, and :func:`validate` writes them down.

The entry points:

* :func:`run_checks` — compute findings, touch nothing. Pure read.
* :func:`validate` — compute *and* persist, returning the run's ``run_id``.
  This is what a refresh should call.
* :func:`latest_report` — read the most recent persisted run back out.

**A clean run leaves evidence.** :func:`validate` writes a row to
``data_quality_runs`` unconditionally, so "validated, nothing wrong" and "never
validated" are distinguishable. That is what lets :func:`latest_report` serve a
clean run instead of falling back to on-demand computation forever, and what lets
:func:`compare_to_previous` detect a clean → dirty regression.

Not yet built from Module 3: calendar gaps against the ``exchange_calendars`` NSE
calendar, duplicate-date detection, MAD-based outliers, split-ratio matching,
circuit-limit days, and the HTML ``DataQualityReport`` rendering with a
completeness percentage. The ``--fail-on-regression`` guard *is* built — see
:func:`compare_to_previous`, which the nightly workflow depends on.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

import pandas as pd
import structlog
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from quantlab.db.models import (
    CorporateAction,
    DataQualityFinding,
    DataQualityRun,
    OHLCVDaily,
    Security,
    UnresolvedMembershipChange,
)

__all__ = [
    "CHECKS",
    "EXTREME_MOVE",
    "STALE_RUN",
    "DataQualityReport",
    "Finding",
    "Regression",
    "Severity",
    "compare_to_previous",
    "latest_report",
    "run_checks",
    "validate",
]

log = structlog.get_logger(__name__)

#: A single-day move beyond this is either a real event, a corporate action the
#: adjustment missed, or a bad print. All three are worth a human looking.
EXTREME_MOVE = 0.20
#: Consecutive identical closes that suggest a stale feed rather than a quiet
#: stock. Five sessions is a trading week.
STALE_RUN = 5

#: Every check a run performs, in the order they are applied. Persisted findings
#: carry a ``check`` from this tuple.
CHECKS: tuple[str, ...] = (
    "non_positive_price",
    "inconsistent_ohlc",
    "zero_volume",
    "extreme_move",
    "stale_price",
    "missing_adjustment",
    "no_price_history",
    "unresolved_membership_change",
)

#: What absence prints as. Mirrors ``quantlab.api.formatting.EM_DASH``; the data
#: layer does not import the API layer to get it.
_EM_DASH = "—"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


class Finding(BaseModel):
    """One defect: what check, how bad, where, and what to do about it."""

    model_config = ConfigDict(frozen=True)

    check: str
    severity: Severity
    security_id: str | None
    date: date | None
    date_display: str
    detail: str
    #: What a researcher should do about it, not just what is wrong.
    remedy: str


class DataQualityReport(BaseModel):
    """The result of one validation run.

    ``run_id`` and ``generated_at`` are set only when the findings came from
    ``data_quality_findings``; a computed-but-unpersisted run has neither, and
    that difference is what the API reports as ``generated_on_demand``.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str | None = None
    generated_at: datetime | None = None
    checks_run: list[str]
    rows_examined: int
    #: Findings per check, over the *whole* run — not just the returned slice.
    counts: dict[str, int]
    findings: list[Finding]

    @property
    def persisted(self) -> bool:
        return self.run_id is not None


# --- Computing -----------------------------------------------------------------


def run_checks(
    session: Session,
    *,
    security_id: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> DataQualityReport:
    """Run every check over the selected slice of ``ohlcv_daily``. Read-only."""

    panel = _load_panel(session, security_id=security_id, start=start, end=end)

    found: list[Finding] = []
    if not panel.empty:
        action_dates = _corporate_action_dates(session)
        found += _price_checks(panel)
        found += _series_checks(panel, action_dates)
    found += _coverage_checks(session)
    found += _membership_checks(session)

    return _report(found, rows_examined=len(panel))


def _load_panel(
    session: Session,
    *,
    security_id: str | None,
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
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
    return pd.DataFrame(
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


def _report(
    findings: list[Finding],
    *,
    rows_examined: int,
    run_id: str | None = None,
    generated_at: datetime | None = None,
) -> DataQualityReport:
    """Count, sort and package. Counts are over every finding, pre-truncation."""

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.check] = counts.get(finding.check, 0) + 1

    ordered = sorted(
        findings,
        key=lambda f: (
            _SEVERITY_ORDER[f.severity],
            f.check,
            f.security_id or "",
            f.date or date.min,
        ),
    )
    return DataQualityReport(
        run_id=run_id,
        generated_at=generated_at,
        checks_run=list(CHECKS),
        rows_examined=rows_examined,
        counts=counts,
        findings=ordered,
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
        date_display=when.isoformat() if when is not None else _EM_DASH,
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


def _corporate_action_dates(session: Session) -> set[tuple[str, date]]:
    rows = session.execute(
        select(CorporateAction.security_id, CorporateAction.date)
    ).all()
    return {(sid, when) for sid, when in rows}


def _coverage_checks(session: Session) -> list[Finding]:
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


def _membership_checks(session: Session) -> list[Finding]:
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


# --- Persisting ----------------------------------------------------------------


def validate(
    session: Session,
    *,
    security_id: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> DataQualityReport:
    """Run every check and write the findings to ``data_quality_findings``.

    Returns the report with ``run_id`` set. Commits — a validation run that
    computed findings and then dropped them on an exception is the situation this
    table exists to prevent.
    """

    report = run_checks(session, security_id=security_id, start=start, end=end)
    run_id = uuid.uuid4().hex

    by_severity: dict[str, int] = {}
    for finding in report.findings:
        by_severity[str(finding.severity)] = (
            by_severity.get(str(finding.severity), 0) + 1
        )

    scope_parts = [
        f"security_id={security_id}" if security_id else None,
        f"start={start.isoformat()}" if start else None,
        f"end={end.isoformat()}" if end else None,
    ]
    scope = ", ".join(p for p in scope_parts if p) or None

    # The run row is written unconditionally — that is the whole point of the
    # table. A clean run must leave evidence, or the regression guard cannot
    # compare against it and the API cannot tell "clean" from "never ran".
    session.add(
        DataQualityRun(
            run_id=run_id,
            rows_examined=report.rows_examined,
            checks_run=list(report.checks_run),
            error_count=by_severity.get("error", 0),
            warning_count=by_severity.get("warning", 0),
            info_count=by_severity.get("info", 0),
            scope=scope,
        )
    )

    session.add_all(
        DataQualityFinding(
            run_id=run_id,
            check=finding.check,
            severity=str(finding.severity),
            security_id=finding.security_id,
            date=finding.date,
            detail=finding.detail,
            remedy=finding.remedy,
            rows_examined=report.rows_examined,
        )
        for finding in report.findings
    )
    session.commit()

    log.info(
        "data_quality.run_persisted",
        run_id=run_id,
        findings=len(report.findings),
        rows_examined=report.rows_examined,
        counts=report.counts,
    )
    return report.model_copy(update={"run_id": run_id})


@dataclass(frozen=True, slots=True)
class Regression:
    """A comparison of one run's severity counts against the previous run's.

    "Worse" is defined per severity, not on a total: ten new `info` findings and
    one new `error` are not the same event, and a total would let the error hide
    behind a wash. Any severity increasing is a regression.
    """

    previous_run_id: str | None
    current: dict[str, int]
    previous: dict[str, int]

    @property
    def worsened(self) -> dict[str, tuple[int, int]]:
        """Severities that increased, as ``{severity: (before, after)}``.

        Empty when there is no previous run. Without that guard a first run on a
        database that already has findings compares against ``{}``, reads every
        count as an increase, and fails — which is the kind of false positive that
        gets a CI gate switched off in week one.
        """

        if self.previous_run_id is None:
            return {}
        return {
            str(severity): (
                self.previous.get(str(severity), 0),
                self.current.get(str(severity), 0),
            )
            for severity in (Severity.ERROR, Severity.WARNING, Severity.INFO)
            if self.current.get(str(severity), 0) > self.previous.get(str(severity), 0)
        }

    @property
    def is_regression(self) -> bool:
        return bool(self.worsened)

    def render(self) -> str:
        if self.previous_run_id is None:
            return (
                "No previous validation run to compare against — this is the "
                "baseline. Nothing can have regressed."
            )
        if not self.is_regression:
            errors_before = self.previous.get("error", 0)
            errors_after = self.current.get("error", 0)
            warn_before = self.previous.get("warning", 0)
            warn_after = self.current.get("warning", 0)
            return (
                f"No regression against run {self.previous_run_id[:8]}: "
                f"errors {errors_before}→{errors_after}, "
                f"warnings {warn_before}→{warn_after}."
            )
        lines = [f"REGRESSION against run {self.previous_run_id[:8]}:"]
        lines += [
            f"  {severity}: {before} → {after}"
            for severity, (before, after) in self.worsened.items()
        ]
        return "\n".join(lines)


def _run_counts(run: DataQualityRun) -> dict[str, int]:
    return {
        "error": run.error_count,
        "warning": run.warning_count,
        "info": run.info_count,
    }


def compare_to_previous(session: Session, run_id: str) -> Regression:
    """Compare ``run_id``'s severity counts against the run before it.

    Reads ``data_quality_runs``, not ``data_quality_findings``. That distinction
    is the whole reason the run table exists: a clean run writes no findings, so
    comparing findings made a clean baseline invisible and let 0 errors → 5 errors
    pass as "first run, nothing to compare".

    A genuinely first run is the baseline and is not a regression — a fresh
    database must not fail the nightly job.
    """

    current = session.get(DataQualityRun, run_id)
    if current is None:
        raise ValueError(
            f"no data_quality_runs row for {run_id!r}; compare_to_previous needs a "
            "persisted run (validate() writes one, run_checks() does not)"
        )

    previous = session.scalars(
        select(DataQualityRun)
        .where(DataQualityRun.run_id != run_id)
        .order_by(DataQualityRun.created_at.desc())
        .limit(1)
    ).first()

    return Regression(
        previous_run_id=previous.run_id if previous else None,
        current=_run_counts(current),
        previous=_run_counts(previous) if previous else {},
    )


def latest_report(
    session: Session,
    *,
    security_id: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> DataQualityReport | None:
    """Read the most recent persisted run back, or ``None`` if there is none.

    ``None`` means "no run has ever written a finding", which includes the happy
    case of a database with no defects — see the module docstring.
    """

    # The run table, not the findings table: a clean run writes no findings, and
    # reading findings here is what made a validated-and-clean database look
    # identical to a never-validated one, so the API fell back to computing on
    # demand forever.
    run = session.scalars(
        select(DataQualityRun).order_by(DataQualityRun.created_at.desc()).limit(1)
    ).first()
    if run is None:
        return None
    run_id = run.run_id

    # A NULL security_id or date is not "outside the filter" — it is a finding
    # that indicts a whole series or a filing rather than one row, and those stay
    # visible however the caller narrows the window. Filtering them out would
    # hide ``missing_adjustment`` from anyone who passed a date range.
    stmt = select(DataQualityFinding).where(DataQualityFinding.run_id == run_id)
    if security_id is not None:
        stmt = stmt.where(
            or_(
                DataQualityFinding.security_id == security_id,
                DataQualityFinding.security_id.is_(None),
            )
        )
    if start is not None:
        stmt = stmt.where(
            or_(DataQualityFinding.date.is_(None), DataQualityFinding.date >= start)
        )
    if end is not None:
        stmt = stmt.where(
            or_(DataQualityFinding.date.is_(None), DataQualityFinding.date <= end)
        )

    rows = list(session.scalars(stmt))
    if not rows:
        # The run exists but nothing in it matches the filter. An empty report is
        # the honest answer; falling back to on-demand would silently answer a
        # different question.
        return _report(
            [],
            rows_examined=run.rows_examined,
            run_id=run_id,
            generated_at=run.created_at,
        )

    return _report(
        [_from_row(row) for row in rows],
        rows_examined=rows[0].rows_examined,
        run_id=run_id,
        generated_at=min(row.created_at for row in rows),
    )


def _from_row(row: DataQualityFinding) -> Finding:
    return _finding(
        row.check,
        Severity(row.severity),
        row.detail,
        row.remedy,
        security_id=row.security_id,
        when=row.date,
    )
