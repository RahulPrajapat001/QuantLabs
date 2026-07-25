"""Rebalancing: a security that left the index is exited, never held silently."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session, sessionmaker

from quantlab.db.session import create_all, create_db_engine, create_session_factory
from quantlab.portfolio.covariance import ledoit_wolf_covariance
from quantlab.portfolio.optimisers import (
    OptimisationError,
    PortfolioConstraints,
    minimum_variance,
)
from quantlab.portfolio.rebalance import (
    RebalanceHistory,
    rebalance_plan,
    rebalance_schedule,
)
from quantlab.universe.membership import (
    build_intervals_from_events,
    insert_membership_intervals,
)
from tests.universe.conftest import add_security

INDEX = "TESTIDX"


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    engine = create_db_engine("sqlite:///:memory:")
    create_all(engine)
    return create_session_factory(engine)


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Session:
    session = session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def indexed(session: Session) -> Session:
    """An index where SEC-B is removed on 2019-03-01 and SEC-D on 2018-09-01.

    Both are held in the test portfolios, so a rebalance after their removal must
    exit them.
    """

    for security_id in ("SEC-A", "SEC-B", "SEC-C", "SEC-D"):
        add_security(session, security_id)
    events = [
        ("SEC-A", "add", date(2015, 1, 1), "seed"),
        ("SEC-B", "add", date(2016, 6, 1), "seed"),
        ("SEC-C", "add", date(2015, 1, 1), "seed"),
        ("SEC-D", "add", date(2018, 3, 1), "seed"),
        ("SEC-D", "remove", date(2018, 9, 1), "seed"),
        ("SEC-B", "remove", date(2019, 3, 1), "seed"),
    ]
    insert_membership_intervals(session, INDEX, build_intervals_from_events(events))
    session.flush()
    return session


def make_returns(
    securities: list[str], *, n_obs: int = 300, seed: int = 5
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0004, 0.015, (n_obs, len(securities))),
        columns=securities,
        index=pd.bdate_range("2017-01-02", periods=n_obs),
    )


# --- THE RULE ----------------------------------------------------------------


def test_a_security_that_left_the_index_is_exited(indexed: Session) -> None:
    """The survivorship trap this module exists to close.

    SEC-D left the index on 2018-09-01. The portfolio still holds it. On a rebalance
    after that date it must be assigned a target of zero and recorded as a forced
    exit — not passed over because it is absent from the candidate set, which is how
    a naive implementation keeps holding it.
    """

    held = {"SEC-A": 0.3, "SEC-C": 0.3, "SEC-D": 0.4}
    plan = rebalance_plan(indexed, INDEX, date(2018, 10, 1), current_weights=held)

    assert "SEC-D" in plan.forced_exits
    assert "SEC-D" not in plan.target_weights.index
    # The exit is a real trade, sized at the weight that was held.
    assert plan.trades["SEC-D"] == pytest.approx(-0.4)
    assert plan.forced_exit_weight == pytest.approx(0.4)
    assert "no longer index members" in plan.render()


def test_a_still_member_is_not_forced_out(indexed: Session) -> None:
    """SEC-D is a member on 2018-06-01, so it stays."""

    held = {"SEC-A": 0.3, "SEC-C": 0.3, "SEC-D": 0.4}
    plan = rebalance_plan(indexed, INDEX, date(2018, 6, 1), current_weights=held)
    assert plan.forced_exits == ()
    assert "SEC-D" in plan.target_weights.index


def test_several_exits_accumulate(indexed: Session) -> None:
    """By 2019-04-01 both SEC-B and SEC-D have left."""

    held = {"SEC-A": 0.25, "SEC-B": 0.25, "SEC-C": 0.25, "SEC-D": 0.25}
    plan = rebalance_plan(indexed, INDEX, date(2019, 4, 1), current_weights=held)
    assert set(plan.forced_exits) == {"SEC-B", "SEC-D"}
    assert plan.forced_exit_weight == pytest.approx(0.5)
    assert set(plan.target_weights.index) == {"SEC-A", "SEC-C"}


def test_forced_exits_are_counted_in_turnover(indexed: Session) -> None:
    """An exit is a trade. If it were silently held, turnover would understate it."""

    held = {"SEC-A": 0.5, "SEC-D": 0.5}
    plan = rebalance_plan(indexed, INDEX, date(2018, 10, 1), current_weights=held)
    # Sell all of SEC-D, and rebalance A and C to equal weight.
    assert plan.turnover > 0.0
    assert plan.trades["SEC-D"] == pytest.approx(-0.5)
    assert plan.trades.sum() == pytest.approx(0.0, abs=1e-9)


def test_an_empty_portfolio_has_no_forced_exits(indexed: Session) -> None:
    plan = rebalance_plan(indexed, INDEX, date(2018, 10, 1))
    assert plan.forced_exits == ()
    assert set(plan.entries) == set(plan.target_weights.index)


def test_a_zero_weight_holding_is_not_treated_as_held(indexed: Session) -> None:
    """A closed position is not a holding, so it is not a forced exit."""

    plan = rebalance_plan(
        indexed, INDEX, date(2019, 4, 1), current_weights={"SEC-B": 0.0, "SEC-A": 1.0}
    )
    assert "SEC-B" not in plan.forced_exits


# --- Point-in-time universe --------------------------------------------------


def test_the_universe_is_read_at_the_rebalance_date_not_today(indexed: Session) -> None:
    """Membership is resolved as of the rebalance date, never from current members."""

    early = rebalance_plan(indexed, INDEX, date(2016, 1, 1))
    mid = rebalance_plan(indexed, INDEX, date(2018, 6, 1))
    late = rebalance_plan(indexed, INDEX, date(2019, 4, 1))

    assert set(early.target_weights.index) == {"SEC-A", "SEC-C"}
    assert set(mid.target_weights.index) == {"SEC-A", "SEC-B", "SEC-C", "SEC-D"}
    assert set(late.target_weights.index) == {"SEC-A", "SEC-C"}


def test_an_empty_universe_raises_rather_than_falling_back(indexed: Session) -> None:
    """No members means no rebalance — never a silent fallback to today's index."""

    with pytest.raises(OptimisationError, match="never fall back to current"):
        rebalance_plan(indexed, INDEX, date(2010, 1, 1))


# --- Investability -----------------------------------------------------------


def test_members_without_enough_history_are_excluded_and_reported(
    indexed: Session,
) -> None:
    """A variance estimated from four observations is not an estimate."""

    returns = make_returns(["SEC-A", "SEC-C"], n_obs=300)
    returns["SEC-B"] = np.nan
    returns.iloc[-5:, returns.columns.get_loc("SEC-B")] = 0.01

    plan = rebalance_plan(
        indexed,
        INDEX,
        date(2018, 6, 1),
        returns=returns,
        min_observations=60,
    )
    assert "SEC-B" in plan.unpriced
    assert "SEC-B" not in plan.target_weights.index
    # SEC-D is a member on this date but has no column at all.
    assert "SEC-D" in plan.unpriced
    assert "no usable price history" in plan.render()


def test_nothing_investable_raises(indexed: Session) -> None:
    empty = pd.DataFrame(index=pd.bdate_range("2017-01-02", periods=10))
    with pytest.raises(OptimisationError, match="nothing investable"):
        rebalance_plan(indexed, INDEX, date(2018, 6, 1), returns=empty)


# --- Optimiser integration ---------------------------------------------------


def test_a_supplied_optimiser_sets_the_target_weights(indexed: Session) -> None:
    returns = make_returns(["SEC-A", "SEC-B", "SEC-C", "SEC-D"])

    def optimise(frame: pd.DataFrame):
        return minimum_variance(
            ledoit_wolf_covariance(frame),
            constraints=PortfolioConstraints(max_weight=0.5),
        )

    plan = rebalance_plan(
        indexed, INDEX, date(2018, 6, 1), returns=returns, optimiser=optimise
    )
    assert plan.result is not None
    assert plan.result.method == "minimum_variance"
    assert plan.target_weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert plan.target_weights.max() <= 0.5 + 1e-6


def test_the_default_optimiser_is_equal_weight(indexed: Session) -> None:
    plan = rebalance_plan(indexed, INDEX, date(2018, 6, 1))
    assert plan.result is not None
    assert plan.result.method == "equal_weight"
    assert np.allclose(plan.target_weights.to_numpy(), 0.25)


def test_optimiser_exits_are_distinguished_from_forced_exits(
    indexed: Session,
) -> None:
    """Two reasons to sell, reported separately because they mean different things."""

    returns = make_returns(["SEC-A", "SEC-B", "SEC-C", "SEC-D"])

    def only_first(frame: pd.DataFrame):
        from quantlab.portfolio.optimisers import equal_weight

        return equal_weight([frame.columns[0]])

    plan = rebalance_plan(
        indexed,
        INDEX,
        date(2019, 4, 1),
        current_weights={"SEC-A": 0.25, "SEC-B": 0.25, "SEC-C": 0.25, "SEC-D": 0.25},
        returns=returns,
        optimiser=only_first,
    )
    # SEC-B and SEC-D left the index; SEC-C was dropped by the optimiser.
    assert set(plan.forced_exits) == {"SEC-B", "SEC-D"}
    assert "SEC-C" in plan.optimiser_exits


# --- Schedule and history ----------------------------------------------------


def test_schedule_frequencies() -> None:
    monthly = rebalance_schedule(date(2020, 1, 1), date(2020, 12, 31))
    assert len(monthly) == 12
    quarterly = rebalance_schedule(
        date(2020, 1, 1), date(2020, 12, 31), frequency="quarterly"
    )
    assert len(quarterly) == 4
    annual = rebalance_schedule(
        date(2018, 1, 1), date(2020, 12, 31), frequency="annual"
    )
    assert len(annual) == 3
    with pytest.raises(OptimisationError, match="unknown frequency"):
        rebalance_schedule(date(2020, 1, 1), date(2020, 2, 1), frequency="hourly")


def test_history_aggregates_forced_exits_across_rebalances(indexed: Session) -> None:
    """The measurable size of the survivorship effect, accumulated over time."""

    held = {"SEC-A": 0.25, "SEC-B": 0.25, "SEC-C": 0.25, "SEC-D": 0.25}
    plans = [
        rebalance_plan(indexed, INDEX, when, current_weights=held)
        for when in (date(2018, 6, 1), date(2018, 10, 1), date(2019, 4, 1))
    ]
    history = RebalanceHistory(plans=tuple(plans))

    assert history.total_forced_exits == 0 + 1 + 2
    frame = history.frame()
    assert len(frame) == 3
    assert frame["forced_exits"].tolist() == [0, 1, 2]
    assert np.isfinite(history.mean_turnover)
    assert "survivorship bias" in history.render()


def test_history_with_no_exits_says_nothing_about_survivorship(
    indexed: Session,
) -> None:
    plans = (rebalance_plan(indexed, INDEX, date(2018, 6, 1)),)
    history = RebalanceHistory(plans=plans)
    assert history.total_forced_exits == 0
    assert "survivorship bias" not in history.render()
