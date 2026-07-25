"""Cost sensitivity — a required deliverable, not an optional check (PRD Module 8).

Every strategy runs under all three cost models. The output is a table of which
strategies survive realistic Indian transaction costs and which do not.

**A strategy that dies under realistic costs is a finding.** It is reported as one,
in plain language, and it is not deleted from the table. The interesting quantity is
usually not the surviving Sharpe but the *gap*: a strategy earning 1.4 gross and
0.1 net is telling you its edge is smaller than the statutory cost of harvesting it,
which is a fact about the strategy, not a failure of the exercise.

Every run here writes to ``research_log`` like any other, so the trials count grows
by the number of (strategy × cost model) combinations. That is correct and
deliberate: sweeping cost models *is* a search, and a Sharpe picked from the survivors
must be deflated against the full count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

from quantlab.backtesting.costs import CostModel, default_cost_models
from quantlab.backtesting.data_handler import DataHandler
from quantlab.backtesting.engine import BacktestConfig, BacktestEngine
from quantlab.backtesting.portfolio import Portfolio, RiskLimits
from quantlab.research.log import ResearchLog

if TYPE_CHECKING:
    # Type-only, to keep backtesting and strategies acyclic — see engine.py.
    from quantlab.strategies.base import Strategy

log = structlog.get_logger(__name__)

#: A strategy must clear this net Sharpe under the Indian model to be called a
#: survivor. Not a claim that 0.5 is good — a threshold below which the result is
#: indistinguishable from noise once costs are real.
SURVIVAL_SHARPE = 0.5


@dataclass(frozen=True, slots=True)
class CostSensitivityRow:
    """One strategy under one cost model."""

    strategy: str
    cost_model: str
    run_id: str
    sharpe: float
    annual_return: float
    total_costs: float
    cost_drag: float
    turnover: float
    trades: int
    max_drawdown: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "cost_model": self.cost_model,
            "run_id": self.run_id,
            "sharpe": self.sharpe,
            "annual_return": self.annual_return,
            "total_costs": self.total_costs,
            "cost_drag": self.cost_drag,
            "turnover": self.turnover,
            "trades": self.trades,
            "max_drawdown": self.max_drawdown,
        }


@dataclass(frozen=True, slots=True)
class CostSensitivityReport:
    """The full sweep, plus the survival verdict per strategy."""

    rows: list[CostSensitivityRow]
    #: Strategy → whether it clears :data:`SURVIVAL_SHARPE` under Indian costs.
    survives: dict[str, bool]
    trials_after: int
    universe: str
    index_name: str
    start: str
    end: str

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([row.as_dict() for row in self.rows])

    def pivot(self, field: str = "sharpe") -> pd.DataFrame:
        """Strategies as rows, cost models as columns."""

        frame = self.frame()
        if frame.empty:
            return frame
        return frame.pivot(index="strategy", columns="cost_model", values=field)

    def survivors(self) -> list[str]:
        return sorted(name for name, ok in self.survives.items() if ok)

    def casualties(self) -> list[str]:
        return sorted(name for name, ok in self.survives.items() if not ok)


def run_cost_sensitivity(
    research_log: ResearchLog,
    strategies: dict[str, Strategy],
    bars: pd.DataFrame,
    *,
    cost_models: dict[str, CostModel] | None = None,
    initial_capital: float = 1_000_000.0,
    universe: str = "point-in-time",
    index_name: str = "NIFTY500",
    limits: RiskLimits | None = None,
) -> CostSensitivityReport:
    """Run every strategy under every cost model, logging each combination."""

    models = cost_models or default_cost_models()
    rows: list[CostSensitivityRow] = []
    sharpes: dict[tuple[str, str], float] = {}

    for strategy_name, strategy in strategies.items():
        for model_name, model in models.items():
            strategy.reset()
            handler = DataHandler(bars)
            portfolio = Portfolio(
                initial_capital=initial_capital,
                limits=limits or RiskLimits(max_positions=10, max_position_weight=0.15),
                target_positions=10,
            )
            engine = BacktestEngine(
                research_log,
                strategy=strategy,
                data_handler=handler,
                portfolio=portfolio,
                config=BacktestConfig(
                    initial_capital=initial_capital,
                    cost_model=model,
                    universe=universe,
                    index_name=index_name,
                    notes=f"cost sensitivity sweep: {strategy_name} under {model_name}",
                ),
            )
            backtest = engine.run()
            metrics = engine.metrics
            if metrics is None:  # pragma: no cover - defensive
                continue
            row = CostSensitivityRow(
                strategy=strategy_name,
                cost_model=model_name,
                run_id=backtest.run_id,
                sharpe=metrics.sharpe,
                annual_return=metrics.annual_return,
                total_costs=metrics.total_costs,
                cost_drag=metrics.cost_drag,
                turnover=metrics.turnover,
                trades=metrics.trades,
                max_drawdown=metrics.drawdown.max_drawdown,
            )
            rows.append(row)
            sharpes[(strategy_name, model_name)] = metrics.sharpe
            log.info(
                "cost_sensitivity.run",
                strategy=strategy_name,
                cost_model=model_name,
                sharpe=metrics.sharpe,
                cost_drag=metrics.cost_drag,
            )

    survives = {
        name: sharpes.get((name, "IndianEquityCostModel"), 0.0) >= SURVIVAL_SHARPE
        for name in strategies
    }
    dates = pd.to_datetime(bars["date"])
    return CostSensitivityReport(
        rows=rows,
        survives=survives,
        trials_after=research_log.trials(),
        universe=universe,
        index_name=index_name,
        start=str(dates.min().date()),
        end=str(dates.max().date()),
    )


def render_markdown(report: CostSensitivityReport) -> str:
    """Render the report for ``docs/research/cost_sensitivity.md``."""

    frame = report.frame()
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Cost sensitivity",
        "",
        f"*Generated {generated}.*",
        "",
        "Every strategy run under all three cost models. This is a required "
        "deliverable (PRD Module 8), and its purpose is to establish which "
        "strategies survive realistic Indian transaction costs — **not** to find "
        "the best Sharpe.",
        "",
        "## How to read this",
        "",
        "- `ZeroCostModel` is a **validation instrument, not a result.** Its column "
        "exists to show what the strategy would earn in a frictionless world, which "
        "is the gross edge. No number in that column is tradeable.",
        "- `GenericBpsModel` applies a flat 10 bps per side — the conventional "
        "simplification, shown for comparison.",
        "- `IndianEquityCostModel` is the honest one: itemised STT, stamp duty, "
        "exchange and SEBI charges, and GST. A round trip costs roughly **0.22% of "
        "notional** before slippage.",
        "- **A strategy that dies under realistic costs is a finding, not a "
        "failure.** The gap between the gross and net columns measures how much of "
        "the apparent edge was actually a subsidy from ignoring costs.",
        "",
        f"**Every Sharpe below is undeflated.** These runs contributed to a "
        f"`research_log` trials count now standing at **{report.trials_after}**. "
        "Deflate before drawing any conclusion:",
        "",
        "```",
        "quantlab research deflate --strategy <name>",
        "```",
        "",
        "## Run specification",
        "",
        f"| Universe | {report.index_name}, {report.universe} |",
        "| --- | --- |",
        f"| Period | {report.start} → {report.end} |",
        f"| Survival threshold | net Sharpe ≥ {SURVIVAL_SHARPE} under "
        "`IndianEquityCostModel` |",
        f"| Combinations | {len(report.rows)} (strategy × cost model) |",
        "",
    ]

    if not frame.empty:
        lines += [
            "## Sharpe by cost model",
            "",
            _pivot_table(report, "sharpe", "{:.3f}"),
        ]
        lines += [
            "",
            "## Annualised return by cost model",
            "",
            _pivot_table(report, "annual_return", "{:.2%}"),
        ]
        lines += [
            "",
            "## Cost drag under Indian costs",
            "",
            "Cost drag is total charges as a fraction of initial capital. Turnover "
            "is annualised two-way turnover — the quantity Indian taxes are levied "
            "on, and therefore the best single predictor of whether a strategy "
            "survives.",
            "",
            _indian_table(report),
        ]

    lines += ["", "## Verdict", ""]
    survivors = report.survivors()
    casualties = report.casualties()

    if survivors:
        lines.append(
            f"**Survive realistic costs** (net Sharpe ≥ {SURVIVAL_SHARPE}): "
            + ", ".join(f"`{name}`" for name in survivors)
        )
    else:
        lines.append(
            f"**No strategy clears a net Sharpe of {SURVIVAL_SHARPE} under Indian "
            "statutory costs.** This is the result, and it is reported as the "
            "result. It says that on this universe and period, the tested rules do "
            "not produce an edge large enough to pay STT, stamp duty and GST and "
            "still compensate for the risk taken."
        )
    lines.append("")
    if casualties:
        lines.append(
            "**Do not survive realistic costs**: "
            + ", ".join(f"`{name}`" for name in casualties)
        )
        lines.append("")
        lines.append(
            "These are reported rather than discarded. A strategy whose gross edge "
            "is real and whose net edge is not has told you something specific: the "
            "edge is smaller than the cost of harvesting it at this turnover. The "
            "response is lower turnover, not a different cost model."
        )
    lines += [
        "",
        "## What this table cannot tell you",
        "",
        "- It is a **cost** sweep, not a validation of any strategy. A strategy that "
        "survives here has cleared one hurdle out of several; it still has to clear "
        "deflation against the full trials count, and it still has to hold up out of "
        "sample.",
        "- Slippage is not modelled in this sweep (zero slippage), so the Indian "
        "column is a **floor** on costs rather than a full estimate. Real execution "
        "adds market impact on top.",
        "- Every number here comes from the train and/or validation split. The "
        "holdout (2024-01-01 onward) is locked and is not touched by this sweep.",
        "",
        "---",
        "",
        "Every row above is traceable to a `research_log` run_id:",
        "",
        "```",
        "quantlab research log --strategy <name>",
        "```",
    ]
    return "\n".join(lines) + "\n"


def _pivot_table(report: CostSensitivityReport, field: str, fmt: str) -> str:
    pivot = report.pivot(field)
    # Gross first, then the conventional simplification, then the honest one.
    order = [
        c
        for c in ("ZeroCostModel", "GenericBpsModel", "IndianEquityCostModel")
        if c in pivot.columns
    ]
    pivot = pivot[order]
    header = "| Strategy | " + " | ".join(order) + " | Gross → net gap |"
    divider = "| --- " * (len(order) + 2) + "|"
    rows = [header, divider]
    for strategy in pivot.index:
        values = [pivot.loc[strategy, column] for column in order]
        cells = [fmt.format(v) for v in values]
        gap = ""
        if "ZeroCostModel" in order and "IndianEquityCostModel" in order:
            gross = pivot.loc[strategy, "ZeroCostModel"]
            net = pivot.loc[strategy, "IndianEquityCostModel"]
            gap = fmt.format(net - gross)
        rows.append(f"| `{strategy}` | " + " | ".join(cells) + f" | {gap} |")
    return "\n".join(rows)


def _indian_table(report: CostSensitivityReport) -> str:
    frame = report.frame()
    indian = frame[frame["cost_model"] == "IndianEquityCostModel"]
    rows = [
        "| Strategy | Trades | Turnover (x/yr) | Costs paid | Cost drag "
        "| Net Sharpe | Survives |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in indian.sort_values("cost_drag", ascending=False).iterrows():
        survives = "yes" if report.survives.get(str(row["strategy"])) else "**no**"
        rows.append(
            f"| `{row['strategy']}` | {int(row['trades'])} | "
            f"{row['turnover']:.2f} | ₹{row['total_costs']:,.0f} | "
            f"{row['cost_drag']:.2%} | {row['sharpe']:.3f} | {survives} |"
        )
    return "\n".join(rows)


def write_report(report: CostSensitivityReport, path: Path | None = None) -> Path:
    """Write ``docs/research/cost_sensitivity.md`` and return the path."""

    target = path or Path("docs/research/cost_sensitivity.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(report), encoding="utf-8")
    log.info("cost_sensitivity.report_written", path=str(target), rows=len(report.rows))
    return target
