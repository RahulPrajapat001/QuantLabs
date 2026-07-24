"""Point-in-time membership: the constituent set on each side of every change.

The synthetic index (see ``conftest.seed_synthetic_index``) has known additions,
a removal, a re-addition and a name that never leaves. We assert
``universe_as_of`` returns exactly the right set on the day before and the day
after each change — the boundary a survivorship-safe backtest depends on.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.orm import Session

from quantlab.universe.membership import (
    build_intervals_from_events,
    current_constituents,
    freeze_fixed_universe,
    universe_as_of,
)

from .conftest import seed_synthetic_index

IDX = "TESTIDX"


@pytest.mark.parametrize(
    ("on", "expected"),
    [
        # Inception: A and C are founding members; B, D not yet in.
        (date(2015, 1, 1), {"SEC-A", "SEC-C"}),
        # --- B added effective 2016-06-01 ---
        (date(2016, 5, 31), {"SEC-A", "SEC-C"}),  # day before
        (date(2016, 6, 1), {"SEC-A", "SEC-B", "SEC-C"}),  # day of / after
        # --- C removed effective 2017-09-01 ---
        (date(2017, 8, 31), {"SEC-A", "SEC-B", "SEC-C"}),  # last day as member
        (date(2017, 9, 1), {"SEC-A", "SEC-B"}),  # first day out
        # --- D added 2018-03-01, removed 2018-09-01 ---
        (date(2018, 2, 28), {"SEC-A", "SEC-B"}),
        (date(2018, 3, 1), {"SEC-A", "SEC-B", "SEC-D"}),
        (date(2018, 8, 31), {"SEC-A", "SEC-B", "SEC-D"}),
        (date(2018, 9, 1), {"SEC-A", "SEC-B"}),
        # --- B removed effective 2019-03-01 ---
        (date(2019, 2, 28), {"SEC-A", "SEC-B"}),
        (date(2019, 3, 1), {"SEC-A"}),
        # --- C re-added effective 2020-03-01 ---
        (date(2020, 2, 29), {"SEC-A"}),
        (date(2020, 3, 1), {"SEC-A", "SEC-C"}),
        # Far future: only the open intervals remain.
        (date(2030, 1, 1), {"SEC-A", "SEC-C"}),
    ],
)
def test_universe_as_of_boundaries(
    session: Session, seeded_index: None, on: date, expected: set[str]
) -> None:
    assert universe_as_of(session, IDX, on) == expected


@pytest.fixture
def seeded_index(session: Session) -> None:
    seed_synthetic_index(session, IDX)


def test_current_constituents_are_survivorship_biased(
    session: Session, seeded_index: None
) -> None:
    """Today's constituents drop every name that ever left — the bias itself."""

    # On 2018-06-01 the honest universe holds B and D (both later removed).
    assert universe_as_of(session, IDX, date(2018, 6, 1)) == {
        "SEC-A",
        "SEC-B",
        "SEC-D",
    }
    # The current set silently excludes them — proving the PIT query does not
    # consult present-day membership.
    assert current_constituents(session, IDX) == {"SEC-A", "SEC-C"}


def test_empty_before_any_membership(session: Session, seeded_index: None) -> None:
    assert universe_as_of(session, IDX, date(2010, 1, 1)) == set()


def test_unknown_index_is_empty(session: Session, seeded_index: None) -> None:
    assert universe_as_of(session, "NOSUCH", date(2018, 1, 1)) == set()


def test_fixed_mode_freezes_and_holds_constant(
    session: Session, seeded_index: None
) -> None:
    """Fixed mode returns the frozen set for every date, ignoring later changes."""

    frozen = freeze_fixed_universe(session, IDX, date(2016, 6, 1))
    assert frozen.members == frozenset({"SEC-A", "SEC-B", "SEC-C"})
    # Held constant regardless of the query date — even far past real changes.
    assert frozen.constituents(date(2016, 6, 1)) == {"SEC-A", "SEC-B", "SEC-C"}
    assert frozen.constituents(date(2025, 1, 1)) == {"SEC-A", "SEC-B", "SEC-C"}


def test_build_intervals_handles_readdition() -> None:
    """A remove then a later add produces two disjoint intervals for one name."""

    events = [
        ("X", "add", date(2015, 1, 1), "s"),
        ("X", "remove", date(2017, 1, 1), "s"),
        ("X", "add", date(2019, 1, 1), "s"),
    ]
    intervals = sorted(build_intervals_from_events(events), key=lambda i: i.start_date)
    assert len(intervals) == 2
    assert (intervals[0].start_date, intervals[0].end_date) == (
        date(2015, 1, 1),
        date(2017, 1, 1),
    )
    # The still-open second interval has no end.
    assert (intervals[1].start_date, intervals[1].end_date) == (
        date(2019, 1, 1),
        None,
    )


def test_build_intervals_ignores_duplicate_add() -> None:
    events = [
        ("X", "add", date(2015, 1, 1), "s"),
        ("X", "add", date(2016, 1, 1), "s"),  # duplicate while already a member
        ("X", "remove", date(2018, 1, 1), "s"),
    ]
    intervals = build_intervals_from_events(events)
    assert len(intervals) == 1
    assert intervals[0].start_date == date(2015, 1, 1)
    assert intervals[0].end_date == date(2018, 1, 1)
