"""Reconstitution parsing and resolution: intervals, unresolved rows, rate."""

from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from quantlab.db.models import UnresolvedMembershipChange
from quantlab.universe.membership import universe_as_of
from quantlab.universe.reconstitution import (
    ReconstitutionEvent,
    ingest_reconstitution,
    parse_nifty_reconstitution,
    parse_sp500_changes,
)

from .conftest import add_security, add_ticker

# A minimal fixture mirroring Wikipedia's S&P 500 changes table: a two-level
# header with grouped Added / Removed → Ticker / Security columns.
SP500_HTML = """
<table>
  <tr>
    <th rowspan="2">Date</th>
    <th colspan="2">Added</th>
    <th colspan="2">Removed</th>
    <th rowspan="2">Reason</th>
  </tr>
  <tr>
    <th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th>
  </tr>
  <tr>
    <td>September 20, 2019</td>
    <td>AAA</td><td>Alpha Inc</td>
    <td>ZZZ</td><td>Zeta Corp</td>
    <td>Market cap change</td>
  </tr>
  <tr>
    <td>March 15, 2020</td>
    <td>BBB</td><td>Beta Inc</td>
    <td></td><td></td>
    <td>Addition only</td>
  </tr>
</table>
"""


def test_parse_sp500_changes_extracts_add_and_remove() -> None:
    events = parse_sp500_changes(SP500_HTML, source_document="wiki.html")
    actions = {(e.action, e.ticker, e.effective_date) for e in events}
    assert ("add", "AAA", date(2019, 9, 20)) in actions
    assert ("remove", "ZZZ", date(2019, 9, 20)) in actions
    assert ("add", "BBB", date(2020, 3, 15)) in actions
    # No spurious removal from the empty second-row cells.
    assert not any(
        e.action == "remove" and e.effective_date == date(2020, 3, 15) for e in events
    )


def test_parse_nifty_reconstitution_buckets_include_exclude() -> None:
    text = (
        "NSE Indices Semi-Annual Reconstitution\n"
        "Securities to be included\n"
        "RELIANCE TCS\n"
        "Securities to be excluded\n"
        "YESBANK\n"
    )
    events = parse_nifty_reconstitution(
        text,
        index_name="NIFTY50",
        effective_date=date(2020, 3, 27),
        source_document="nifty50_2020-03-27.pdf",
    )
    by_action = {(e.action, e.ticker) for e in events}
    assert ("add", "RELIANCE") in by_action
    assert ("add", "TCS") in by_action
    assert ("remove", "YESBANK") in by_action


def test_ingest_resolves_and_reports_rate(session: Session) -> None:
    add_security(session, "SEC-A", name="Alpha Inc")
    add_ticker(session, "SEC-A", "AAA", start=date(2015, 1, 1))

    events = [
        ReconstitutionEvent(
            index_name="SP500",
            action="add",
            effective_date=date(2016, 1, 4),
            source_document="wiki.html",
            ticker="AAA",
        ),
        ReconstitutionEvent(
            index_name="SP500",
            action="remove",
            effective_date=date(2018, 6, 1),
            source_document="wiki.html",
            ticker="AAA",
        ),
        # GHOST has no ticker_history → cannot resolve → recorded, not dropped.
        ReconstitutionEvent(
            index_name="SP500",
            action="add",
            effective_date=date(2016, 1, 4),
            source_document="wiki.html",
            ticker="GHOST",
        ),
    ]

    report = ingest_reconstitution(session, "SP500", events)

    assert report.total_events == 3
    assert report.resolved_events == 2
    assert report.unresolved_events == 1
    assert report.resolution_rate == 2 / 3
    assert report.intervals_inserted == 1  # AAA add+remove → one interval

    # The resolved interval is queryable point-in-time.
    assert universe_as_of(session, "SP500", date(2017, 1, 1)) == {"SEC-A"}
    assert universe_as_of(session, "SP500", date(2019, 1, 1)) == set()

    # The unresolved change is persisted rather than silently dropped.
    unresolved = session.execute(select(UnresolvedMembershipChange)).scalars().all()
    assert len(unresolved) == 1
    assert unresolved[0].ticker == "GHOST"
    assert unresolved[0].action == "add"


def test_ingest_is_idempotent(session: Session) -> None:
    add_security(session, "SEC-A")
    add_ticker(session, "SEC-A", "AAA", start=date(2015, 1, 1))
    events = [
        ReconstitutionEvent(
            index_name="SP500",
            action="add",
            effective_date=date(2016, 1, 4),
            source_document="wiki.html",
            ticker="AAA",
        ),
    ]
    first = ingest_reconstitution(session, "SP500", events)
    second = ingest_reconstitution(session, "SP500", events)
    assert first.intervals_inserted == 1
    assert second.intervals_inserted == 0  # no duplicate rows on re-ingest


def test_nse_symbol_resolves_via_ns_suffix(session: Session) -> None:
    """A bare NSE symbol resolves against the ``.NS`` vendor ticker."""

    add_security(session, "SEC-REL", name="Reliance")
    add_ticker(session, "SEC-REL", "RELIANCE.NS", start=date(2015, 1, 1))

    events = [
        ReconstitutionEvent(
            index_name="NIFTY50",
            action="add",
            effective_date=date(2016, 3, 1),
            source_document="nifty50_2016-03-01.pdf",
            ticker="RELIANCE",  # bare symbol, as printed in the filing
        ),
    ]
    report = ingest_reconstitution(session, "NIFTY50", events)
    assert report.resolved_events == 1
    assert universe_as_of(session, "NIFTY50", date(2017, 1, 1)) == {"SEC-REL"}
