"""Delisting handling: liquidate at last real price; bankruptcy recovery; no fill."""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from quantlab.universe.delisting import (
    is_bankruptcy,
    last_available_bar,
    liquidate_position,
    liquidation_proceeds,
)

from .conftest import add_bar, add_security


def test_is_bankruptcy_keywords() -> None:
    assert is_bankruptcy("NCLT insolvency resolution")
    assert is_bankruptcy("Delisted under IBC")
    assert is_bankruptcy("Chapter 11 filing")
    assert not is_bankruptcy("Merger with parent")
    assert not is_bankruptcy("Acquired")
    assert not is_bankruptcy(None)


def test_liquidation_proceeds_non_bankruptcy_is_full_price() -> None:
    assert liquidation_proceeds(120.0, reason="Merger", recovery_rate=0.0) == 120.0


def test_liquidation_proceeds_bankruptcy_applies_recovery_rate() -> None:
    # Default recovery of 0 → total loss.
    assert liquidation_proceeds(120.0, reason="NCLT", recovery_rate=0.0) == 0.0
    # A configured recovery rate scales the last price.
    assert liquidation_proceeds(120.0, reason="NCLT", recovery_rate=0.3) == 36.0


def test_last_available_bar_never_forward_fills(session: Session) -> None:
    add_security(session, "SEC-X", delisting_date=date(2020, 6, 15))
    add_bar(session, "SEC-X", date(2020, 6, 10), 100.0)
    add_bar(session, "SEC-X", date(2020, 6, 12), 105.0)  # last real bar

    # The last bar at/before the delisting date is the genuine 2020-06-12 one;
    # no synthetic bar is invented on 2020-06-15.
    bar = last_available_bar(session, "SEC-X", date(2020, 6, 15))
    assert bar == (date(2020, 6, 12), 105.0)


def test_liquidate_non_bankruptcy_uses_last_close(session: Session) -> None:
    add_security(
        session,
        "SEC-M",
        delisting_date=date(2021, 3, 20),
        delisting_reason="Merger with acquirer",
    )
    add_bar(session, "SEC-M", date(2021, 3, 18), 250.0)

    result = liquidate_position(session, "SEC-M")
    assert result is not None
    assert result.bankruptcy is False
    assert result.last_price == 250.0
    assert result.proceeds == 250.0
    assert result.price_date == date(2021, 3, 18)


def test_liquidate_bankruptcy_default_zero_recovery(session: Session) -> None:
    add_security(
        session,
        "SEC-B",
        delisting_date=date(2019, 9, 30),
        delisting_reason="NCLT insolvency proceedings",
    )
    add_bar(session, "SEC-B", date(2019, 9, 27), 12.0)

    result = liquidate_position(session, "SEC-B")
    assert result is not None
    assert result.bankruptcy is True
    assert result.recovery_rate == 0.0
    assert result.proceeds == 0.0  # total loss


def test_liquidate_bankruptcy_with_recovery_rate(session: Session) -> None:
    add_security(
        session,
        "SEC-B2",
        delisting_date=date(2019, 9, 30),
        delisting_reason="Winding up under IBC",
    )
    add_bar(session, "SEC-B2", date(2019, 9, 27), 40.0)

    result = liquidate_position(session, "SEC-B2", recovery_rate=0.25)
    assert result is not None
    assert result.proceeds == 10.0  # 40 * 0.25


def test_liquidate_returns_none_without_a_price(session: Session) -> None:
    """A delisting with no prior bar yields None — a price is never fabricated."""

    add_security(session, "SEC-NO", delisting_date=date(2020, 1, 10))
    assert liquidate_position(session, "SEC-NO") is None


def test_liquidate_returns_none_for_live_security(session: Session) -> None:
    add_security(session, "SEC-LIVE")  # no delisting_date
    add_bar(session, "SEC-LIVE", date(2020, 1, 2), 99.0)
    assert liquidate_position(session, "SEC-LIVE") is None
