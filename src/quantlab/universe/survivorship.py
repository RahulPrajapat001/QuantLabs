"""Survivorship-bias measurement — a required deliverable, not a check.

The routine here runs *any* strategy twice — once on the point-in-time universe,
once on today's constituents — and reports the delta in annual return, Sharpe,
and max drawdown (PRD §4.7). The gap is the fingerprint of survivorship bias: the
current-constituents run silently excludes every name that fell out of the index,
which are disproportionately the losers, so it flatters the strategy.

A strategy is modelled as a callable ``StrategyFn`` that, given a
:class:`~quantlab.universe.membership.UniverseResolver`, returns a daily-returns
series. That is all the harness needs, so it works before the M2 backtester
exists and stays decoupled from it: point the same strategy at a
:class:`~quantlab.universe.membership.PointInTimeUniverse` and a
:class:`~quantlab.universe.membership.CurrentUniverse` and compare.

The performance metrics here are deliberately minimal and local to the
measurement; the authoritative risk/return analytics live in the M2 ``risk``
module. Sharpe here is a plain, un-deflated ratio used only for the *relative*
survivorship delta — it is never a reported headline number (headline Sharpes
must carry trial count + deflated Sharpe, per CLAUDE.md §3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .membership import UniverseResolver

#: A strategy: given a universe resolver, produce a daily-returns series indexed
#: by date. Simple, backtester-agnostic, and enough to measure the bias.
StrategyFn = Callable[[UniverseResolver], "pd.Series[float]"]

#: NSE trades ~250 sessions a year; used to annualise.
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """The three headline numbers the survivorship delta compares."""

    annual_return: float
    sharpe: float
    #: Max drawdown as a negative fraction (e.g. -0.42 for a 42% peak-to-trough).
    max_drawdown: float


def compute_metrics(
    returns: pd.Series[float],
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Annualised return, Sharpe, and max drawdown from a daily-returns series.

    * annual return — geometric: ``prod(1+r) ** (ppy/n) - 1``.
    * Sharpe — ``mean(excess) / std(excess) * sqrt(ppy)`` with sample std; ``0``
      when volatility is zero (a flat series has no risk-adjusted signal).
    * max drawdown — the minimum of the equity curve's peak-to-trough decline,
      as a negative fraction.
    """

    clean = returns.dropna().astype(float)
    if clean.empty:
        return PerformanceMetrics(annual_return=0.0, sharpe=0.0, max_drawdown=0.0)

    n = len(clean)
    growth = float(np.prod(clean.to_numpy(dtype=float) + 1.0))
    annual_return = growth ** (periods_per_year / n) - 1.0 if growth > 0 else -1.0

    per_period_rf = risk_free_rate / periods_per_year
    excess = clean - per_period_rf
    std = float(excess.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(excess.mean()) / std * (periods_per_year**0.5) if std > 0 else 0.0

    equity = (1.0 + clean).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_drawdown = float(drawdown.min())

    return PerformanceMetrics(
        annual_return=annual_return, sharpe=sharpe, max_drawdown=max_drawdown
    )


@dataclass(frozen=True, slots=True)
class SurvivorshipComparison:
    """Point-in-time vs current-constituents performance for one strategy."""

    strategy_name: str
    index_name: str
    point_in_time: PerformanceMetrics
    current: PerformanceMetrics

    @property
    def annual_return_delta(self) -> float:
        """Current minus point-in-time annual return (the inflation, if positive)."""

        return self.current.annual_return - self.point_in_time.annual_return

    @property
    def sharpe_delta(self) -> float:
        return self.current.sharpe - self.point_in_time.sharpe

    @property
    def max_drawdown_delta(self) -> float:
        return self.current.max_drawdown - self.point_in_time.max_drawdown


def measure_survivorship_bias(
    strategy: StrategyFn,
    *,
    strategy_name: str,
    point_in_time: UniverseResolver,
    current: UniverseResolver,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> SurvivorshipComparison:
    """Run ``strategy`` on both universes and return the metric comparison.

    ``point_in_time`` should be a
    :class:`~quantlab.universe.membership.PointInTimeUniverse` and ``current`` a
    :class:`~quantlab.universe.membership.CurrentUniverse` over the same index,
    so the only difference between the two runs is the survivorship treatment.
    """

    pit_returns = strategy(point_in_time)
    cur_returns = strategy(current)
    return SurvivorshipComparison(
        strategy_name=strategy_name,
        index_name=getattr(point_in_time, "index_name", "?"),
        point_in_time=compute_metrics(
            pit_returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
        current=compute_metrics(
            cur_returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
    )


def render_survivorship_report(comparison: SurvivorshipComparison) -> str:
    """Render a :class:`SurvivorshipComparison` as a Markdown document."""

    pit = comparison.point_in_time
    cur = comparison.current
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    def pct(x: float) -> str:
        return f"{x * 100:+.2f}%"

    ret_row = (
        f"| Annualised return | {pct(pit.annual_return)} "
        f"| {pct(cur.annual_return)} | {pct(comparison.annual_return_delta)} |"
    )
    sharpe_row = (
        f"| Sharpe ratio | {pit.sharpe:.2f} | {cur.sharpe:.2f} "
        f"| {comparison.sharpe_delta:+.2f} |"
    )
    dd_row = (
        f"| Max drawdown | {pct(pit.max_drawdown)} "
        f"| {pct(cur.max_drawdown)} | {pct(comparison.max_drawdown_delta)} |"
    )

    return f"""# Survivorship bias measurement

_Generated {generated} by `quantlab.universe.survivorship`._

**Strategy:** `{comparison.strategy_name}`
**Index:** `{comparison.index_name}`

The same strategy is run twice: once on the **point-in-time** universe (the honest
universe, reconstructed from index reconstitution history) and once on the
**current constituents** (today's index members applied to every historical date).
The difference is the survivorship bias: the current-constituents run cannot see
any company that was dropped from the index, which are disproportionately the
failures, so it overstates performance.

| Metric | Point-in-time | Current constituents | Delta (current minus PIT) |
|---|---:|---:|---:|
{ret_row}
{sharpe_row}
{dd_row}

A positive return/Sharpe delta or a shallower (less negative) current-constituents
drawdown means survivorship bias is inflating the biased backtest by that amount.

> The Sharpe figures here are plain, un-deflated ratios used only to size the
> survivorship gap. Any headline Sharpe reported elsewhere must be accompanied by
> its trial count and deflated Sharpe (CLAUDE.md §3).
"""


def write_survivorship_report(
    comparison: SurvivorshipComparison,
    *,
    path: Path = Path("docs/research/survivorship_bias.md"),
) -> Path:
    """Write the rendered survivorship report to ``path`` and return it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_survivorship_report(comparison), encoding="utf-8")
    return path
