"""PDF tearsheets (PRD Module 15), with a methodology page that cannot be dropped.

A tearsheet is the artefact that leaves the building. It gets emailed, printed and
screenshotted, and by the time it is being read the reader has no access to the
research log, the cost model, or the knowledge that the Sharpe on page one was the
best of three hundred. Everything needed to judge the number has to travel *with*
the number.

So the methodology page is not a section that a caller can switch off. It is
assembled by :func:`build_methodology`, which raises if any of its five required
statements — universe construction, cost model, splits, trials count, limitations
— is missing, and :func:`render_html` refuses to produce a document without it.
The check is on the *content*, not on a flag: a methodology page whose trials
count is absent is not a methodology page.

Two enforcement points are inherited rather than reimplemented:
:class:`~quantlab.research.reporting.SharpeReport` cannot exist without a trials
count, and :func:`~quantlab.research.reporting.assert_no_bare_sharpe` catches a
Sharpe arriving through a metrics mapping. The tearsheet adds no new way to print
one.

**WeasyPrint is imported lazily.** It binds to Pango and Cairo at import time, and
those are system libraries rather than Python packages. Everything up to the PDF
byte stream — the model, the HTML, the methodology checks — works without them,
which is also what makes it all testable on a machine that has no GTK runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import structlog

from quantlab.api.formatting import format_date
from quantlab.research.log import ResearchLog
from quantlab.research.reporting import SharpeReport, assert_no_bare_sharpe

log = structlog.get_logger(__name__)

__all__ = [
    "Methodology",
    "MethodologyIncompleteError",
    "TearsheetData",
    "build_methodology",
    "render_html",
    "render_pdf",
]

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: The five statements PRD Module 15 requires. Absent any one of them, the
#: document is not publishable and this module will not produce it.
REQUIRED_METHODOLOGY_FIELDS = (
    "universe_construction",
    "cost_model",
    "splits",
    "trials",
    "limitations",
)


class MethodologyIncompleteError(RuntimeError):
    """Raised when a tearsheet would ship without its methodology page."""


class WeasyPrintUnavailableError(RuntimeError):
    """Raised when the PDF backend's system libraries are missing."""


@dataclass(frozen=True, slots=True)
class Methodology:
    """The mandatory page. Every field is required and non-empty."""

    universe_construction: str
    cost_model: str
    splits: str
    trials: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        missing = [
            name for name in REQUIRED_METHODOLOGY_FIELDS if not getattr(self, name)
        ]
        if missing:
            raise MethodologyIncompleteError(
                f"the methodology page is missing {missing}. PRD Module 15 makes "
                "it mandatory: a tearsheet without it states a result while "
                "withholding what the reader needs to judge it."
            )


@dataclass(frozen=True, slots=True)
class TearsheetData:
    """Everything one tearsheet renders. Assembled by :func:`from_run`."""

    run_id: str
    strategy_name: str
    generated_at: datetime
    window: tuple[date, date]
    split: str
    split_parts: tuple[str, ...]
    holdout_touched: bool
    exploratory: bool
    hypothesis_id: str | None
    sharpe: SharpeReport
    #: Everything except the Sharpe, which is rendered from ``sharpe``.
    metrics: dict[str, Any]
    methodology: Methodology
    #: ``(label, formatted_value)`` pairs for the performance table.
    performance_rows: tuple[tuple[str, str], ...] = ()
    risk_rows: tuple[tuple[str, str], ...] = ()
    trade_rows: tuple[tuple[str, str], ...] = ()
    #: Inline SVG chart markup, if any. Embedded rather than linked so the PDF is
    #: self-contained.
    charts: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def costs_realistic(self) -> bool:
        return self.sharpe.cost_model == "IndianEquityCostModel"


def build_methodology(
    research_log: ResearchLog,
    run_id: str,
    *,
    limitations: tuple[str, ...] | None = None,
) -> Methodology:
    """Assemble the methodology page from the log, not from the caller.

    The trials count in particular is read from ``research_log``; there is no
    parameter that supplies it. A methodology page whose trials count was typed
    in by the person reporting the result is decoration.
    """

    from quantlab.config import HOLDOUT_START, TRAIN_END, VALIDATION_END

    entry = research_log.entry(run_id)
    trials = research_log.trials()

    cost_model = entry.cost_model
    cost_sentence = {
        "IndianEquityCostModel": (
            "Itemised Indian statutory costs: STT, stamp duty (buy side only), "
            "NSE transaction charge, SEBI turnover fee and GST on brokerage and "
            "exchange charges. A delivery round trip costs roughly 0.22% of "
            "notional before slippage."
        ),
        "ZeroCostModel": (
            "NO COSTS APPLIED. This is a validation instrument, not a tradeable "
            "result — the figures overstate what any account would have earned."
        ),
        "GenericBpsModel": (
            "A flat basis-point charge per side. A conventional simplification "
            "that does not reflect Indian statutory costs; STT alone exceeds it "
            "on a delivery round trip."
        ),
    }.get(cost_model, f"{cost_model} — see the cost model's own documentation.")

    default_limitations = (
        "No intraday data: every fill is at a daily open or close.",
        "No point-in-time fundamentals, so no valuation or quality factors.",
        "No borrow costs or short availability modelling.",
        "Universe resolution is below 100%; unresolved reconstitution entries "
        "are recorded and reduce point-in-time membership completeness.",
        "NSE corporate-action adjustment is self-built, not vendor-supplied.",
        "Single market, single asset class — nothing here generalises to another "
        "exchange without re-validation.",
    )

    return Methodology(
        universe_construction=(
            f"Point-in-time membership of {entry.index_name}, reconstructed from "
            "index reconstitution filings. Securities enter and leave on their "
            "actual effective dates, and delisted securities remain in the "
            "universe for the period they were members. Current constituents are "
            "never used to define a historical universe."
        ),
        cost_model=f"{cost_model}. {cost_sentence}",
        splits=(
            f"Train {'≤'} {TRAIN_END.isoformat()}; validation "
            f"{TRAIN_END.year + 1}-01-01 to {VALIDATION_END.isoformat()}; holdout "
            f"{HOLDOUT_START.isoformat()} onward, locked and evaluated once at "
            f"the end of the project. This run covers: {entry.split}."
        ),
        trials=(
            f"{trials} completed run(s) recorded in research_log at the time of "
            f"generation. The Sharpe on this tearsheet is deflated against that "
            f"count, which spans every strategy and model configuration tried on "
            f"this project — not only those sharing this run's name."
        ),
        limitations=limitations or default_limitations,
    )


def from_run(
    research_log: ResearchLog,
    run_id: str,
    *,
    limitations: tuple[str, ...] | None = None,
    charts: tuple[tuple[str, str], ...] = (),
) -> TearsheetData:
    """Build tearsheet data from a logged run."""

    from quantlab.api.formatting import (
        format_count,
        format_currency,
        format_percent,
        format_ratio,
        format_share,
    )

    entry = research_log.entry(run_id)
    report = SharpeReport.from_log(research_log, run_id)
    metrics = dict(entry.metrics or {})

    # The same guard the markdown generator uses. A Sharpe arriving through the
    # metrics mapping is caught here rather than printed.
    assert_no_bare_sharpe(metrics, report)

    def get(key: str) -> Any:
        return metrics.get(key)

    performance = (
        ("Total return", format_percent(get("total_return"))),
        ("Annualised return", format_percent(get("annual_return"))),
        ("Annualised volatility", format_percent(get("annual_volatility"))),
        ("Sortino", format_ratio(get("sortino"))),
        ("Calmar", format_ratio(get("calmar"))),
        ("Final equity", format_currency(get("final_equity"))),
        ("Trading days", format_count(get("trading_days"))),
    )
    risk = (
        ("Max drawdown", format_percent(_negative(get("max_drawdown")))),
        ("Drawdown days", format_count(get("drawdown_days"))),
        ("Recovery days", format_count(get("recovery_days"))),
        ("Benchmark return", format_percent(get("benchmark_annual_return"))),
        ("Excess return", format_percent(get("excess_annual_return"))),
        ("Return skew", format_ratio(get("returns_skew"))),
        ("Return kurtosis", format_ratio(get("returns_kurtosis"))),
    )
    trades = (
        ("Trades", format_count(get("trades"))),
        ("Win rate", format_share(get("win_rate"))),
        ("Profit factor", format_ratio(get("profit_factor"))),
        ("Average trade", format_currency(get("average_trade"))),
        ("Expectancy", format_currency(get("expectancy"))),
        ("Turnover", format_share(get("turnover"))),
        ("Total costs", format_currency(get("total_costs"))),
        ("Cost drag", format_percent(_negative(get("cost_drag")))),
    )

    return TearsheetData(
        run_id=run_id,
        strategy_name=entry.strategy_name,
        generated_at=datetime.now(UTC),
        window=(entry.start_date, entry.end_date),
        split=entry.split,
        split_parts=tuple(p for p in entry.split.split("+") if p),
        holdout_touched=bool(entry.holdout_touched),
        exploratory=bool(entry.exploratory),
        hypothesis_id=entry.hypothesis_id,
        sharpe=report,
        metrics=metrics,
        methodology=build_methodology(research_log, run_id, limitations=limitations),
        performance_rows=performance,
        risk_rows=risk,
        trade_rows=trades,
        charts=charts,
    )


def _negative(value: Any) -> float | None:
    """Drawdowns and cost drag are losses; render them signed."""

    if value is None:
        return None
    return -abs(float(value))


def render_html(data: TearsheetData) -> str:
    """Render the tearsheet to HTML.

    Refuses to produce a document whose methodology page is incomplete. The check
    is repeated here rather than trusted from construction because
    :class:`TearsheetData` can be built by hand in a test or a notebook.
    """

    from jinja2 import Environment, FileSystemLoader, select_autoescape

    missing = [
        name
        for name in REQUIRED_METHODOLOGY_FIELDS
        if not getattr(data.methodology, name, None)
    ]
    if missing:
        raise MethodologyIncompleteError(
            f"refusing to render a tearsheet whose methodology page is missing "
            f"{missing}"
        )

    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["isodate"] = format_date
    template = environment.get_template("tearsheet.html")
    return template.render(
        data=data,
        sharpe=data.sharpe,
        methodology=data.methodology,
        stylesheet=(TEMPLATE_DIR / "tearsheet.css").read_text(encoding="utf-8"),
    )


def render_pdf(data: TearsheetData) -> bytes:
    """Render to PDF bytes via WeasyPrint.

    Raises :class:`WeasyPrintUnavailableError` when WeasyPrint's system
    dependencies (Pango, Cairo, GDK-PixBuf — the GTK stack) are not installed.
    That is a deployment concern rather than a Python one: the Docker image in M4
    carries them, and a bare Windows host does not.
    """

    html = render_html(data)
    try:
        # Imported here rather than at module scope: importing weasyprint
        # binds to Pango and Cairo, which a bare Windows host does not have.
        from weasyprint import HTML
    except (ImportError, OSError) as exc:  # pragma: no cover - environment
        raise WeasyPrintUnavailableError(
            "WeasyPrint could not load its system libraries (Pango/Cairo). The "
            "HTML tearsheet rendered fine; only the PDF conversion needs them. "
            "On Debian/Ubuntu: apt-get install libpango-1.0-0 libpangoft2-1.0-0; "
            "on Windows install the GTK3 runtime. See "
            "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html"
        ) from exc

    pdf = HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf()
    log.info(
        "reports.tearsheet.rendered",
        run_id=data.run_id,
        strategy=data.strategy_name,
        bytes=len(pdf),
    )
    assert isinstance(pdf, bytes)
    return pdf
