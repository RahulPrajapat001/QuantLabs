"""Module 3 validation: the checks find defects, and the findings survive.

The point of persistence is that a finding outlives the request that produced it,
so the tests here care less about the individual checks (each defect is injected
and asserted below) than about the round trip: what ``validate`` computes is what
``latest_report`` reads back.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from quantlab.data.validation import (
    CHECKS,
    Finding,
    Severity,
    compare_to_previous,
    latest_report,
    run_checks,
    validate,
)
from quantlab.db.models import (
    DataQualityFinding,
    DataQualityRun,
    OHLCVDaily,
    Security,
)

START = date(2020, 1, 1)


def _security(session: Session, security_id: str) -> None:
    session.add(
        Security(
            security_id=security_id,
            isin=f"INE{security_id:>09}"[:12],
            name=f"Company {security_id}",
            exchange="NSE",
            listing_date=START,
        )
    )


def _bar(
    session: Session,
    security_id: str,
    when: date,
    *,
    close: float = 100.0,
    high: float | None = None,
    low: float | None = None,
    volume: float = 10_000.0,
) -> None:
    session.add(
        OHLCVDaily(
            security_id=security_id,
            date=when,
            open=close,
            high=close + 1.0 if high is None else high,
            low=close - 1.0 if low is None else low,
            close=close,
            adj_close=close,
            volume=volume,
            source="test",
            as_of=datetime.now(UTC),
        )
    )


@pytest.fixture
def defective(session: Session) -> Session:
    """A database with one injected defect per check that a small panel can show.

    ``SEC001`` has a bar whose high does not bracket its close and a session that
    did not trade. ``SEC002`` exists with no price history at all.
    """

    _security(session, "SEC001")
    _security(session, "SEC002")
    session.flush()

    _bar(session, "SEC001", date(2020, 1, 2))
    # high below close: the bar does not bracket, so every range feature is wrong.
    _bar(session, "SEC001", date(2020, 1, 3), close=100.0, high=99.0, low=98.0)
    # did not trade.
    _bar(session, "SEC001", date(2020, 1, 6), volume=0.0)
    session.commit()
    return session


def test_checks_find_the_injected_defects(defective: Session) -> None:
    report = run_checks(defective)

    assert report.counts["inconsistent_ohlc"] == 1
    assert report.counts["zero_volume"] == 1
    assert report.counts["no_price_history"] == 1
    assert report.rows_examined == 3
    assert report.checks_run == list(CHECKS)
    # Nothing was written: run_checks is a pure read.
    assert report.run_id is None
    assert defective.scalar(select(DataQualityFinding.id)) is None


def test_findings_persist_and_round_trip(defective: Session) -> None:
    """What validate() computes is what latest_report() reads back."""

    computed = validate(defective)
    assert computed.run_id is not None

    stored = defective.scalars(select(DataQualityFinding)).all()
    assert len(stored) == len(computed.findings)
    assert {row.run_id for row in stored} == {computed.run_id}
    assert all(row.rows_examined == computed.rows_examined for row in stored)

    read_back = latest_report(defective)
    assert read_back is not None
    assert read_back.run_id == computed.run_id
    assert read_back.rows_examined == computed.rows_examined
    assert read_back.counts == computed.counts
    # Finding is frozen and compares by value: this is the round trip.
    assert read_back.findings == computed.findings
    assert read_back.persisted is True
    assert read_back.generated_at is not None


def test_a_persisted_finding_keeps_its_severity_and_remedy(defective: Session) -> None:
    validate(defective)
    report = latest_report(defective)
    assert report is not None

    by_check = {f.check: f for f in report.findings}
    assert by_check["inconsistent_ohlc"].severity is Severity.ERROR
    assert by_check["zero_volume"].severity is Severity.WARNING
    assert by_check["inconsistent_ohlc"].date == date(2020, 1, 3)
    assert by_check["inconsistent_ohlc"].date_display == "2020-01-03"
    # A finding without a remedy is a complaint.
    assert all(f.remedy for f in report.findings)


def test_undated_findings_survive_a_date_filter(defective: Session) -> None:
    """``no_price_history`` indicts a whole series, so a window cannot exclude it."""

    validate(defective)
    report = latest_report(defective, start=date(2020, 1, 6), end=date(2020, 1, 6))

    assert report is not None
    checks = {f.check for f in report.findings}
    assert "no_price_history" in checks
    assert "zero_volume" in checks  # dated 2020-01-06, inside the window
    assert "inconsistent_ohlc" not in checks  # dated 2020-01-03, outside it


def test_the_latest_run_is_the_one_served(defective: Session) -> None:
    first = validate(defective)
    _bar(defective, "SEC001", date(2020, 1, 7), volume=0.0)
    defective.commit()
    second = validate(defective)

    assert second.run_id != first.run_id
    report = latest_report(defective)
    assert report is not None
    assert report.run_id == second.run_id
    assert report.counts["zero_volume"] == 2


def test_a_never_validated_database_has_no_report(session: Session) -> None:
    """`latest_report` is None only when validation genuinely never ran.

    This previously also covered "validated and clean", because a clean run wrote
    nothing. It no longer does: `data_quality_runs` records every run, so the two
    states are distinguishable — see `test_a_clean_run_still_records_itself`.
    """

    _security(session, "SEC001")
    session.flush()
    _bar(session, "SEC001", date(2020, 1, 2))
    session.commit()

    assert latest_report(session) is None, "nothing has run yet"

    validate(session)
    assert latest_report(session) is not None, "a clean run is still a run"


def test_finding_is_frozen() -> None:
    finding = Finding(
        check="zero_volume",
        severity=Severity.WARNING,
        security_id="SEC001",
        date=date(2020, 1, 6),
        date_display="2020-01-06",
        detail="zero volume",
        remedy="check for a halt",
    )
    with pytest.raises(ValidationError):
        finding.check = "other"  # type: ignore[misc]


# --- Regression guard (--fail-on-regression) ---------------------------------


def test_a_clean_run_still_records_itself(session: Session) -> None:
    """The whole reason `data_quality_runs` exists.

    A clean run writes no findings. Before the run table, that made "validated,
    nothing wrong" indistinguishable from "never validated" — so the API fell back
    to on-demand forever and the regression guard could not compare against a
    clean baseline.
    """

    _security(session, "SEC001")
    session.flush()
    _bar(session, "SEC001", date(2020, 1, 2))
    session.commit()

    report = validate(session)
    assert report.findings == []

    run = session.get(DataQualityRun, report.run_id)
    assert run is not None, "a clean run must still leave a row"
    assert run.error_count == 0
    assert run.rows_examined == 1

    # And it is now servable, rather than looking like "never validated".
    assert latest_report(session) is not None


def test_first_run_is_a_baseline_not_a_regression(defective: Session) -> None:
    report = validate(defective)
    assert report.run_id is not None

    regression = compare_to_previous(defective, report.run_id)
    assert regression.previous_run_id is None
    assert regression.is_regression is False
    assert "baseline" in regression.render()


def test_clean_then_dirty_is_a_regression(session: Session) -> None:
    """0 errors -> 1 error must fail. This is the case the guard exists for."""

    _security(session, "SEC001")
    session.flush()
    _bar(session, "SEC001", date(2020, 1, 2))
    session.commit()

    clean = validate(session)
    assert compare_to_previous(session, clean.run_id or "").is_regression is False

    # A bar whose high does not bracket its close: one new ERROR.
    _bar(session, "SEC001", date(2020, 1, 3), close=100.0, high=99.0, low=98.0)
    session.commit()
    dirty = validate(session)

    regression = compare_to_previous(session, dirty.run_id or "")
    assert regression.previous_run_id == clean.run_id
    assert regression.is_regression is True
    assert regression.worsened[Severity.ERROR] == (0, 1)
    assert "REGRESSION" in regression.render()


def test_unchanged_findings_are_not_a_regression(defective: Session) -> None:
    first = validate(defective)
    second = validate(defective)

    regression = compare_to_previous(defective, second.run_id or "")
    assert regression.previous_run_id == first.run_id
    assert regression.is_regression is False


def test_fewer_findings_is_not_a_regression(defective: Session) -> None:
    """Fixing data must not fail the guard."""

    before = validate(defective)
    assert before.counts.get("zero_volume") == 1

    # Repair the zero-volume bar.
    bar = defective.scalars(select(OHLCVDaily).where(OHLCVDaily.volume == 0.0)).one()
    bar.volume = 12_345.0
    defective.commit()

    after = validate(defective)
    regression = compare_to_previous(defective, after.run_id or "")
    assert regression.is_regression is False


def test_comparing_an_unpersisted_run_raises(defective: Session) -> None:
    """`run_checks` writes nothing, so there is no run to compare."""

    with pytest.raises(ValueError, match="no data_quality_runs row"):
        compare_to_previous(defective, "never-persisted")
