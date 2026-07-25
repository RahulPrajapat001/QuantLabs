"""Cost sensitivity: the required deliverable, and its honesty properties."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantlab.backtesting.cost_sensitivity import (
    SURVIVAL_SHARPE,
    render_markdown,
    run_cost_sensitivity,
    write_report,
)
from quantlab.research.log import ResearchLog
from quantlab.strategies.systematic import BuyAndHoldStrategy
from quantlab.strategies.technical import (
    BollingerBandStrategy,
    MovingAverageCrossStrategy,
)

from .conftest import make_bars


@pytest.fixture(scope="module")
def swept() -> tuple[object, ResearchLog]:
    """Run the sweep **once** for the whole module.

    Nine backtests is expensive, and every test here interrogates the same sweep.
    A function-scoped fixture would re-run it thirteen times for no additional
    coverage. The fixture owns its own database so it does not depend on the
    function-scoped session.
    """

    from quantlab.db.session import (
        create_all,
        create_db_engine,
        create_session_factory,
    )

    engine = create_db_engine("sqlite:///:memory:")
    create_all(engine)
    session = create_session_factory(engine)()
    research_log = ResearchLog(session)
    strategies = {
        "ma_cross": MovingAverageCrossStrategy(fast=5, slow=20),
        "bollinger_reversion": BollingerBandStrategy(window=10),
        "buy_and_hold": BuyAndHoldStrategy(),
    }
    bars = make_bars(n_days=120, securities=3, seed=8, drift=0.0006, volume=500_000.0)
    try:
        yield (
            run_cost_sensitivity(
                research_log, strategies, bars, initial_capital=1_000_000.0
            ),
            research_log,
        )
    finally:
        session.close()


@pytest.fixture
def report(swept):
    return swept[0]


@pytest.fixture
def sweep_log(swept) -> ResearchLog:
    return swept[1]


def test_every_strategy_runs_under_every_cost_model(report) -> None:
    assert len(report.rows) == 9  # 3 strategies × 3 cost models
    assert {row.cost_model for row in report.rows} == {
        "ZeroCostModel",
        "GenericBpsModel",
        "IndianEquityCostModel",
    }
    assert {row.strategy for row in report.rows} == {
        "ma_cross",
        "bollinger_reversion",
        "buy_and_hold",
    }


def test_every_combination_is_logged_as_a_trial(report, sweep_log: ResearchLog) -> None:
    """Sweeping cost models is a search, and it counts toward the trials total."""

    assert sweep_log.trials() == 9
    assert report.trials_after == 9
    for row in report.rows:
        entry = sweep_log.entry(row.run_id)
        assert entry.status == "complete"
        assert entry.cost_model == row.cost_model
        assert "cost sensitivity sweep" in (entry.notes or "")


def test_zero_costs_never_look_worse_than_indian_costs(report) -> None:
    """The gross column is an upper bound on the net one — a sanity invariant.

    If a strategy ever earned *more* after paying STT, the cost plumbing is wired
    backwards somewhere.
    """

    pivot = report.pivot("annual_return")
    for strategy in pivot.index:
        gross = pivot.loc[strategy, "ZeroCostModel"]
        net = pivot.loc[strategy, "IndianEquityCostModel"]
        assert net <= gross + 1e-12, f"{strategy} earned more after costs than before"


def test_costs_paid_are_zero_only_under_the_zero_model(report) -> None:
    for row in report.rows:
        if row.cost_model == "ZeroCostModel":
            assert row.total_costs == 0.0
        elif row.trades > 0 or row.turnover > 0:
            assert row.total_costs > 0.0, f"{row.strategy} traded but paid nothing"


def test_higher_turnover_costs_more(report) -> None:
    """Turnover is what Indian taxes are levied on, so it drives the drag."""

    frame = report.frame()
    indian = frame[frame["cost_model"] == "IndianEquityCostModel"]
    traded = indian[indian["turnover"] > 0].sort_values("turnover")
    if len(traded) >= 2:
        assert traded.iloc[0]["cost_drag"] <= traded.iloc[-1]["cost_drag"] + 1e-9


def test_survival_verdict_uses_the_net_sharpe(report) -> None:
    pivot = report.pivot("sharpe")
    for strategy, survives in report.survives.items():
        net = pivot.loc[strategy, "IndianEquityCostModel"]
        assert survives == (net >= SURVIVAL_SHARPE)
    # Survivors and casualties partition the strategy set.
    assert set(report.survivors()) | set(report.casualties()) == set(report.survives)
    assert not set(report.survivors()) & set(report.casualties())


# --- The report itself -------------------------------------------------------


def test_markdown_reports_casualties_as_findings(report) -> None:
    markdown = render_markdown(report)
    assert "# Cost sensitivity" in markdown
    assert "finding, not a" in markdown
    if report.casualties():
        assert "Do not survive realistic costs" in markdown
        assert "reported rather than discarded" in markdown


def test_markdown_labels_zero_costs_as_not_a_result(report) -> None:
    """The gross column must never read as tradeable."""

    markdown = render_markdown(report)
    assert "validation instrument, not a result" in markdown
    assert "No number in that column is tradeable" in markdown


def test_markdown_carries_the_trials_count_and_deflation_instruction(report) -> None:
    """PRD §4.6: a Sharpe is not reportable without its trials count."""

    markdown = render_markdown(report)
    assert "undeflated" in markdown.lower()
    assert str(report.trials_after) in markdown
    assert "quantlab research deflate" in markdown


def test_markdown_states_its_own_limitations(report) -> None:
    markdown = render_markdown(report)
    assert "What this table cannot tell you" in markdown
    assert "Slippage is not modelled" in markdown
    assert "holdout" in markdown.lower()


def test_markdown_shows_the_gross_to_net_gap(report) -> None:
    """The gap is the interesting quantity, so it gets its own column."""

    markdown = render_markdown(report)
    assert "Gross → net gap" in markdown


def test_report_writes_to_the_expected_path(report, tmp_path: Path) -> None:
    target = tmp_path / "research" / "cost_sensitivity.md"
    written = write_report(report, target)
    assert written == target
    assert target.exists()
    assert "Cost sensitivity" in target.read_text(encoding="utf-8")


def test_pivot_orders_columns_gross_to_net(report) -> None:
    markdown = render_markdown(report)
    header_line = next(
        line for line in markdown.splitlines() if line.startswith("| Strategy |")
    )
    assert header_line.index("ZeroCostModel") < header_line.index("GenericBpsModel")
    assert header_line.index("GenericBpsModel") < header_line.index(
        "IndianEquityCostModel"
    )
