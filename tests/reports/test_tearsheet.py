"""The tearsheet, and the page it cannot ship without.

The PDF byte stream needs WeasyPrint's system libraries (Pango/Cairo); everything
above it does not. So the content rules — the mandatory methodology page, the
Sharpe contract, the warning markers — are tested against the HTML, which is the
same document. Only the byte-generation tests skip when the GTK stack is absent.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker

from quantlab.db.session import create_all, create_db_engine, create_session_factory
from quantlab.reports.tearsheet import (
    REQUIRED_METHODOLOGY_FIELDS,
    Methodology,
    MethodologyIncompleteError,
    build_methodology,
    from_run,
    render_html,
)
from quantlab.research.log import ResearchLog, RunResult, RunSpec


def _weasyprint_available() -> bool:
    try:
        import weasyprint  # noqa: F401
    except Exception:
        return False
    return True


requires_weasyprint = pytest.mark.skipif(
    not _weasyprint_available(),
    reason="WeasyPrint's system libraries (Pango/Cairo) are not installed",
)


@pytest.fixture
def session() -> Session:
    engine = create_db_engine("sqlite:///:memory:")
    create_all(engine)
    factory: sessionmaker[Session] = create_session_factory(engine)
    session = factory()
    try:
        yield session
    finally:
        session.close()


def _log_run(
    session: Session,
    *,
    cost_model: str = "IndianEquityCostModel",
    hypothesis_id: str | None = None,
    end: date = date(2021, 12, 31),
) -> str:
    log = ResearchLog(session)
    rng = np.random.default_rng(5)
    spec = RunSpec(
        strategy_name="momentum_12_1",
        universe="point-in-time",
        index_name="NIFTY500",
        start=date(2016, 1, 4),
        end=end,
        cost_model=cost_model,
        hypothesis_id=hypothesis_id,
    )
    with log.run(spec) as handle:
        handle.record(
            RunResult.from_returns(
                rng.normal(0.0005, 0.012, size=500),
                metrics={
                    "total_return": 0.31,
                    "annual_return": 0.114,
                    "annual_volatility": 0.19,
                    "sortino": 1.1,
                    "calmar": 0.58,
                    "max_drawdown": 0.196,
                    "trades": 180,
                    "win_rate": 0.51,
                    "turnover": 2.1,
                    "total_costs": 38_000.0,
                    "cost_drag": 0.019,
                    "final_equity": 1_310_000.0,
                    "trading_days": 500,
                },
            )
        )
    return handle.run_id


# --- The methodology page is mandatory ---------------------------------------


def test_the_methodology_page_states_all_five_required_things(session):
    run_id = _log_run(session)
    methodology = build_methodology(ResearchLog(session), run_id)
    for field in REQUIRED_METHODOLOGY_FIELDS:
        assert getattr(methodology, field), f"{field} is empty"


def test_a_methodology_missing_any_field_cannot_be_constructed():
    with pytest.raises(MethodologyIncompleteError, match="universe_construction"):
        Methodology(
            universe_construction="",
            cost_model="x",
            splits="x",
            trials="x",
            limitations=("y",),
        )


def test_a_methodology_with_no_limitations_cannot_be_constructed():
    with pytest.raises(MethodologyIncompleteError, match="limitations"):
        Methodology(
            universe_construction="x",
            cost_model="x",
            splits="x",
            trials="x",
            limitations=(),
        )


def test_rendering_refuses_a_tearsheet_whose_methodology_was_hollowed_out(session):
    """The check is repeated at render time, not trusted from construction."""

    import dataclasses

    data = from_run(ResearchLog(session), _log_run(session))
    # Bypass __post_init__ the way a careless caller might.
    broken = dataclasses.replace(
        data,
        methodology=object.__new__(Methodology),  # type: ignore[arg-type]
    )
    with pytest.raises((MethodologyIncompleteError, AttributeError)):
        render_html(broken)


def test_the_trials_count_comes_from_the_log_not_the_caller(session):
    """There is no parameter that supplies it, and it reflects the whole log."""

    first = _log_run(session)
    _log_run(session)
    _log_run(session)
    methodology = build_methodology(ResearchLog(session), first)
    assert "3 completed run(s)" in methodology.trials


def test_the_methodology_page_appears_in_the_rendered_document(session):
    html = render_html(from_run(ResearchLog(session), _log_run(session)))
    assert "Methodology" in html
    assert "Universe construction" in html
    assert "Cost model" in html
    assert "Data splits" in html
    assert "Trials" in html
    assert "Limitations" in html


def test_the_methodology_page_starts_a_new_page(session):
    """It cannot be reduced to a footnote under the trades table."""

    html = render_html(from_run(ResearchLog(session), _log_run(session)))
    assert 'class="methodology"' in html
    css = html[html.index("<style>") : html.index("</style>")]
    assert "break-before: page" in css or "page-break-before: always" in css


# --- The Sharpe contract survives into the document --------------------------


def test_the_sharpe_renders_with_its_trials_count_adjacent(session):
    html = render_html(from_run(ResearchLog(session), _log_run(session)))
    assert "Sharpe ratio" in html
    assert "Trials" in html
    assert "Deflated" in html
    # Not hidden behind an interaction — this is a PDF; there are none.
    assert "E[max] under null" in html


def test_an_insignificant_result_says_so_in_words(session):
    for _ in range(40):
        _log_run(session)
    run_id = _log_run(session)
    data = from_run(ResearchLog(session), run_id)
    html = render_html(data)
    if not data.sharpe.significant:
        assert "Not significant after deflation" in html
        assert "consistent with noise" in html


def test_a_gross_run_is_marked_as_not_tradeable(session):
    run_id = _log_run(session, cost_model="ZeroCostModel")
    html = render_html(from_run(ResearchLog(session), run_id))
    assert "Costs are not realistic" in html
    assert "NO COSTS APPLIED" in html


def test_a_realistic_run_carries_no_cost_warning(session):
    html = render_html(from_run(ResearchLog(session), _log_run(session)))
    assert "Costs are not realistic" not in html


def test_an_exploratory_run_is_labelled(session):
    html = render_html(from_run(ResearchLog(session), _log_run(session)))
    assert "EXPLORATORY" in html


def test_the_split_badge_is_present_and_decomposed(session):
    run_id = _log_run(session, end=date(2022, 6, 30))
    data = from_run(ResearchLog(session), run_id)
    assert data.split_parts == ("train", "validation")
    html = render_html(data)
    assert "TRAIN" in html and "VALIDATION" in html


def test_every_number_is_in_the_tabular_figures_class(session):
    html = render_html(from_run(ResearchLog(session), _log_run(session)))
    assert 'class="ql-num"' in html
    css = html[html.index("<style>") : html.index("</style>")]
    assert "tabular-nums" in css


def test_the_document_is_traceable_to_its_run(session):
    run_id = _log_run(session)
    html = render_html(from_run(ResearchLog(session), run_id))
    assert run_id in html
    assert "traceable to that row" in html


# --- PDF generation ----------------------------------------------------------


@requires_weasyprint
def test_the_pdf_renders(session):
    from quantlab.reports.tearsheet import render_pdf

    pdf = render_pdf(from_run(ResearchLog(session), _log_run(session)))
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 5_000


def test_the_pdf_error_names_the_missing_libraries(session):
    """When the GTK stack is absent the message has to be actionable."""

    if _weasyprint_available():
        pytest.skip("WeasyPrint is available, so the failure path cannot fire")
    from quantlab.reports.tearsheet import WeasyPrintUnavailableError, render_pdf

    with pytest.raises(WeasyPrintUnavailableError, match="Pango"):
        render_pdf(from_run(ResearchLog(session), _log_run(session)))
