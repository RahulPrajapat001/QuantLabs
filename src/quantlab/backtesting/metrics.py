"""Performance metrics computed from the ledger (PRD Module 8).

Every number here is derived from the daily marks and closed trades — nothing is
estimated or annualised from a shortcut. Three choices worth stating, because they
are the ones that differ between honest and flattering implementations:

* **Sharpe uses the sample standard deviation of daily returns** (``ddof=1``) and a
  configurable risk-free rate, annualised by ``√252``. It is *not* computed from
  monthly or annual aggregates, which reduces the apparent volatility and inflates
  the ratio.
* **Win rate, profit factor, average trade and expectancy come from closed round
  trips**, not from daily equity changes. A strategy that grinds out small daily
  gains and occasionally loses a lot has a high daily "win rate" and a poor trade
  win rate; the second is the honest one.
* **Turnover is annualised two-way turnover** as a fraction of average equity. This
  is the number that determines whether Indian transaction taxes are survivable, so
  it is reported alongside every cost figure rather than buried.

Nothing here reports a Sharpe on its own. :class:`BacktestMetrics` carries the raw
value, and the reporting layer refuses to emit it without the trials count and
deflated Sharpe from ``research_log`` (see
:mod:`quantlab.research.reporting`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtesting.ledger import Ledger

#: Indian equity trading days per year.
TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class DrawdownInfo:
    """The worst peak-to-trough decline, and how long it lasted."""

    max_drawdown: float
    peak_date: date | None
    trough_date: date | None
    recovery_date: date | None
    #: Days from peak to trough.
    drawdown_days: int
    #: Days from peak back to a new high. ``None`` when never recovered — which is
    #: itself the finding, and must not be reported as zero.
    recovery_days: int | None

    @property
    def recovered(self) -> bool:
        return self.recovery_date is not None


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """The full metric set for one run."""

    # Returns
    total_return: float
    annual_return: float
    annual_volatility: float
    #: Annualised Sharpe. Never report without trials count + deflated Sharpe.
    sharpe: float
    sortino: float
    calmar: float

    # Drawdown
    drawdown: DrawdownInfo

    # Trades
    trades: int
    win_rate: float
    profit_factor: float
    average_trade: float
    average_win: float
    average_loss: float
    expectancy: float

    # Activity
    turnover: float
    total_costs: float
    cost_drag: float
    cost_attribution: dict[str, float]
    rejections: dict[str, int]

    # Context
    start: date | None
    end: date | None
    trading_days: int
    initial_capital: float
    final_equity: float
    periods_per_year: int = TRADING_DAYS

    # Benchmark
    benchmark_total_return: float | None = None
    benchmark_annual_return: float | None = None
    benchmark_sharpe: float | None = None
    excess_annual_return: float | None = None

    #: Return moments, carried so the deflated Sharpe can be computed without
    #: recomputing them from a series the log does not store.
    returns_skew: float = 0.0
    returns_kurtosis: float = 3.0

    def as_dict(self) -> dict[str, Any]:
        """Flat mapping for ``research_log.metrics``.

        Deliberately includes ``sharpe`` — the log records everything. The
        *reporting* layer is what refuses to display it bare.
        """

        out: dict[str, Any] = {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.drawdown.max_drawdown,
            "max_drawdown_days": self.drawdown.drawdown_days,
            "recovery_days": self.drawdown.recovery_days,
            "recovered": self.drawdown.recovered,
            "trades": self.trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "average_trade": self.average_trade,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "expectancy": self.expectancy,
            "turnover": self.turnover,
            "total_costs": self.total_costs,
            "cost_drag": self.cost_drag,
            "trading_days": self.trading_days,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
        }
        out.update({f"cost_{k}": v for k, v in self.cost_attribution.items()})
        out.update({f"rejected_{k}": v for k, v in self.rejections.items()})
        if self.benchmark_total_return is not None:
            out["benchmark_total_return"] = self.benchmark_total_return
            out["benchmark_annual_return"] = self.benchmark_annual_return
            out["benchmark_sharpe"] = self.benchmark_sharpe
            out["excess_annual_return"] = self.excess_annual_return
        return out

    def render(self) -> str:
        """Human-readable summary. Sharpe is labelled as undeflated."""

        lines = [
            f"Period:            {self.start} → {self.end} "
            f"({self.trading_days} trading days)",
            f"Equity:            ₹{self.initial_capital:,.0f} → "
            f"₹{self.final_equity:,.0f}",
            f"Total return:      {self.total_return * 100:8.2f}%",
            f"Annual return:     {self.annual_return * 100:8.2f}%",
            f"Annual volatility: {self.annual_volatility * 100:8.2f}%",
            f"Sharpe:            {self.sharpe:8.3f}   (UNDEFLATED — see "
            "'quantlab research deflate')",
            f"Sortino:           {self.sortino:8.3f}",
            f"Calmar:            {self.calmar:8.3f}",
            f"Max drawdown:      {self.drawdown.max_drawdown * 100:8.2f}%  "
            f"({self.drawdown.drawdown_days}d to trough, "
            + (
                f"{self.drawdown.recovery_days}d to recover)"
                if self.drawdown.recovered
                else "never recovered)"
            ),
            f"Trades:            {self.trades:8d}",
            f"Win rate:          {self.win_rate * 100:8.2f}%",
            f"Profit factor:     {self.profit_factor:8.3f}",
            f"Expectancy:        ₹{self.expectancy:,.2f} per trade",
            f"Turnover:          {self.turnover:8.2f}x per year",
            f"Total costs:       ₹{self.total_costs:,.2f} "
            f"({self.cost_drag * 100:.2f}% of initial capital)",
        ]
        if self.benchmark_total_return is not None:
            lines.append(
                f"Benchmark:         {self.benchmark_total_return * 100:8.2f}% total, "
                f"{(self.benchmark_annual_return or 0) * 100:.2f}% annual "
                f"(excess {(self.excess_annual_return or 0) * 100:+.2f}%)"
            )
        if self.rejections.get("circuit_limit"):
            lines.append(
                f"Circuit rejections:{self.rejections['circuit_limit']:8d}  "
                "— orders that received no fill because the security was at its "
                "price band"
            )
        return "\n".join(lines)


def max_drawdown(curve: pd.Series) -> DrawdownInfo:
    """Worst peak-to-trough decline, with duration and recovery.

    ``recovery_days`` is ``None`` when equity never regained the peak. Reporting
    that as zero, or as the length of the series, would turn a strategy that never
    recovered into one that recovered instantly.
    """

    if curve.empty or len(curve) < 2:
        return DrawdownInfo(0.0, None, None, None, 0, None)

    running_peak = curve.cummax()
    drawdowns = curve / running_peak - 1.0
    trough_idx = drawdowns.idxmin()
    worst = float(drawdowns.loc[trough_idx])

    if worst == 0.0:
        return DrawdownInfo(0.0, None, None, None, 0, None)

    # The peak that preceded the trough.
    before_trough = curve.loc[:trough_idx]
    peak_idx = before_trough.idxmax()
    peak_value = float(curve.loc[peak_idx])

    after = curve.loc[trough_idx:]
    recovered = after[after >= peak_value]
    recovery_idx = recovered.index[0] if len(recovered) else None

    peak_date = pd.Timestamp(peak_idx).date()
    trough_date = pd.Timestamp(trough_idx).date()
    recovery_date = (
        pd.Timestamp(recovery_idx).date() if recovery_idx is not None else None
    )

    return DrawdownInfo(
        max_drawdown=worst,
        peak_date=peak_date,
        trough_date=trough_date,
        recovery_date=recovery_date,
        drawdown_days=(trough_date - peak_date).days,
        recovery_days=(recovery_date - peak_date).days if recovery_date else None,
    )


def sharpe_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe from daily returns, using the sample standard deviation."""

    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    sd = float(excess.std(ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino: excess return over *downside* deviation.

    Downside deviation is computed over the full sample with upside set to zero,
    not over only the negative observations — dividing by the count of losing days
    alone would make a strategy look better simply for having few of them.
    """

    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = excess.clip(upper=0.0)
    downside_deviation = float(np.sqrt((downside**2).mean()))
    if downside_deviation == 0.0:
        return 0.0
    return float(excess.mean() / downside_deviation * np.sqrt(periods_per_year))


def annualised_return(
    curve: pd.Series, *, periods_per_year: int = TRADING_DAYS
) -> float:
    """Compound annual growth rate from the equity curve.

    Uses the number of observations rather than calendar span, so a series with
    gaps is not credited with growth over days it did not trade.
    """

    if len(curve) < 2:
        return 0.0
    start, end = float(curve.iloc[0]), float(curve.iloc[-1])
    if start <= 0:
        return 0.0
    if end <= 0:
        return -1.0
    years = (len(curve) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)


def compute_metrics(
    ledger: Ledger,
    *,
    initial_capital: float,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> BacktestMetrics:
    """Compute the full metric set from a ledger.

    ``benchmark`` is an equity or price series on the same dates; it is rebased to
    ``initial_capital`` so the comparison is like-for-like.
    """

    curve = ledger.equity_curve
    returns = ledger.returns
    trades = ledger.closed_trades

    total_return = (
        float(curve.iloc[-1] / curve.iloc[0] - 1.0) if len(curve) >= 2 else 0.0
    )
    annual = annualised_return(curve, periods_per_year=periods_per_year)
    volatility = (
        float(returns.std(ddof=1) * np.sqrt(periods_per_year))
        if len(returns) >= 2
        else 0.0
    )
    drawdown = max_drawdown(curve)
    calmar = annual / abs(drawdown.max_drawdown) if drawdown.max_drawdown < 0 else 0.0

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    win_rate = len(wins) / len(trades) if trades else 0.0
    average_win = gross_profit / len(wins) if wins else 0.0
    average_loss = -gross_loss / len(losses) if losses else 0.0
    # Profit factor is undefined with no losses; reporting infinity is honest,
    # and downstream formatting handles it, whereas substituting a large finite
    # number would invent a value.
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )
    average_trade = sum(t.pnl for t in trades) / len(trades) if trades else 0.0
    expectancy = win_rate * average_win + (1 - win_rate) * average_loss

    average_equity = float(curve.mean()) if len(curve) else initial_capital
    years = max(len(curve) - 1, 1) / periods_per_year
    turnover = (
        ledger.total_turnover() / average_equity / years if average_equity > 0 else 0.0
    )
    total_costs = ledger.costs.total

    skew, kurtosis = _moments(returns)

    metrics = BacktestMetrics(
        total_return=total_return,
        annual_return=annual,
        annual_volatility=volatility,
        sharpe=sharpe_ratio(
            returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        sortino=sortino_ratio(
            returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        calmar=calmar,
        drawdown=drawdown,
        trades=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        average_trade=average_trade,
        average_win=average_win,
        average_loss=average_loss,
        expectancy=expectancy,
        turnover=turnover,
        total_costs=total_costs,
        cost_drag=total_costs / initial_capital if initial_capital else 0.0,
        cost_attribution=ledger.cost_attribution(),
        rejections=ledger.rejection_counts(),
        start=pd.Timestamp(curve.index[0]).date() if len(curve) else None,
        end=pd.Timestamp(curve.index[-1]).date() if len(curve) else None,
        trading_days=len(curve),
        initial_capital=initial_capital,
        final_equity=float(curve.iloc[-1]) if len(curve) else initial_capital,
        periods_per_year=periods_per_year,
        returns_skew=skew,
        returns_kurtosis=kurtosis,
    )

    if benchmark is not None and len(benchmark) >= 2:
        metrics = _attach_benchmark(
            metrics,
            benchmark,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
    return metrics


def _attach_benchmark(
    metrics: BacktestMetrics,
    benchmark: pd.Series,
    *,
    initial_capital: float,
    risk_free_rate: float,
    periods_per_year: int,
) -> BacktestMetrics:
    """Rebase the benchmark to the same capital and compare like for like."""

    from dataclasses import replace

    rebased = benchmark / float(benchmark.iloc[0]) * initial_capital
    bench_returns = rebased.pct_change().dropna()
    bench_total = float(rebased.iloc[-1] / rebased.iloc[0] - 1.0)
    bench_annual = annualised_return(rebased, periods_per_year=periods_per_year)
    return replace(
        metrics,
        benchmark_total_return=bench_total,
        benchmark_annual_return=bench_annual,
        benchmark_sharpe=sharpe_ratio(
            bench_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        excess_annual_return=metrics.annual_return - bench_annual,
    )


def _moments(returns: pd.Series) -> tuple[float, float]:
    """Population skew and **non-excess** kurtosis (3.0 for a normal)."""

    if len(returns) < 3:
        return 0.0, 3.0
    values = returns.to_numpy(dtype=float)
    sd = float(values.std(ddof=0))
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0, 3.0
    z = (values - values.mean()) / sd
    return float((z**3).mean()), float((z**4).mean())
