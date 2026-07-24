"""Ticker → security_id resolution, including renames and ticker reuse."""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from quantlab.data.identity import (
    get_or_create_security,
    resolve_security_id,
)
from quantlab.db.models import Security, TickerHistory


def _seed_renamed_company(session: Session) -> str:
    """One security that traded as OLD.NS then renamed to NEW.NS mid-history."""

    sec_id = "SEC-fixed-0001"
    session.add(Security(security_id=sec_id, name="Acme", exchange="NSE"))
    session.add_all(
        [
            TickerHistory(
                security_id=sec_id,
                ticker="OLD.NS",
                start_date=date(2015, 1, 1),
                end_date=date(2019, 6, 30),
                source="fake",
            ),
            TickerHistory(
                security_id=sec_id,
                ticker="NEW.NS",
                start_date=date(2019, 7, 1),
                end_date=None,
                source="fake",
            ),
        ]
    )
    session.flush()
    return sec_id


def test_rename_resolves_to_single_security_id(session: Session) -> None:
    sec_id = _seed_renamed_company(session)

    # Old ticker resolves for dates within its window.
    assert resolve_security_id(session, "OLD.NS", date(2016, 1, 1)) == sec_id
    # New ticker resolves for dates within its window.
    assert resolve_security_id(session, "NEW.NS", date(2020, 1, 1)) == sec_id
    # Both point at the same identity — the whole point.
    assert resolve_security_id(
        session, "OLD.NS", date(2016, 1, 1)
    ) == resolve_security_id(session, "NEW.NS", date(2020, 1, 1))


def test_point_in_time_resolution_not_today(session: Session) -> None:
    _seed_renamed_company(session)
    # OLD.NS was retired in mid-2019; it must NOT resolve after its end_date.
    assert resolve_security_id(session, "OLD.NS", date(2020, 1, 1)) is None
    # NEW.NS did not exist before mid-2019.
    assert resolve_security_id(session, "NEW.NS", date(2016, 1, 1)) is None


def test_get_or_create_reuses_existing_identity(session: Session) -> None:
    sec_id = _seed_renamed_company(session)
    resolved = get_or_create_security(
        session, "NEW.NS", date(2021, 1, 1), source="fake"
    )
    assert resolved == sec_id
    # No new security row was created.
    assert session.query(Security).count() == 1


def test_get_or_create_mints_new_identity_for_unknown(session: Session) -> None:
    sec_id = get_or_create_security(
        session, "FRESH.NS", date(2020, 1, 1), source="fake"
    )
    assert sec_id.startswith("SEC-")
    sec = session.get(Security, sec_id)
    assert sec is not None
    assert sec.exchange == "NSE"  # inferred from .NS suffix
    history = session.query(TickerHistory).filter_by(security_id=sec_id).all()
    assert len(history) == 1
    assert history[0].ticker == "FRESH.NS"


def test_ticker_reuse_across_companies(session: Session) -> None:
    """The same ticker string can belong to different companies over time."""

    old = Security(security_id="SEC-A", name="First Co", exchange="NSE")
    new = Security(security_id="SEC-B", name="Second Co", exchange="NSE")
    session.add_all([old, new])
    session.add_all(
        [
            TickerHistory(
                security_id="SEC-A",
                ticker="REUSED.NS",
                start_date=date(2015, 1, 1),
                end_date=date(2017, 12, 31),
                source="fake",
            ),
            TickerHistory(
                security_id="SEC-B",
                ticker="REUSED.NS",
                start_date=date(2019, 1, 1),
                end_date=None,
                source="fake",
            ),
        ]
    )
    session.flush()

    assert resolve_security_id(session, "REUSED.NS", date(2016, 1, 1)) == "SEC-A"
    assert resolve_security_id(session, "REUSED.NS", date(2020, 1, 1)) == "SEC-B"
    # Gap between the two listings resolves to neither.
    assert resolve_security_id(session, "REUSED.NS", date(2018, 6, 1)) is None
