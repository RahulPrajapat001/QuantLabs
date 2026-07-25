"""Stress testing on real crisis windows (PRD Module 9).

**Real ingested data, never synthetic shocks.** A synthetic "-30% shock" is a
statement about the modeller's imagination, not about the market. It has no
correlation structure, no sequencing, no liquidity collapse, and no recovery path —
and every one of those is the part that hurts. In the COVID crash the NIFTY fell 38%
in 33 sessions with correlations converging on 1 and circuit breakers halting trade;
a uniform percentage haircut captures none of that.

So this module replays the portfolio through windows that actually happened:

======================  ===========================  ==================================
Window                  Period                       What it tests
======================  ===========================  ==================================
GFC                     2007-10-01 → 2009-03-31      A long grinding bear market
COVID crash             2020-02-01 → 2020-04-30      A fast crash with correlation → 1
Rate shock              2022-01-01 → 2022-10-31      Sustained multiple compression
Demonetisation          2016-11-01 → 2016-12-31      An India-specific liquidity shock
IL&FS credit event      2018-09-01 → 2018-10-31      Indian credit and NBFC contagion
======================  ===========================  ==================================

These windows are **ingested and retained regardless of the backtest range**. The GFC
predates ``DATA_START`` (2015-01-01) by seven years, so it would never be fetched by
ordinary ingestion — which is precisely why it has to be requested explicitly, and
why :func:`stress_window_ingest_plan` exists.

**When the data is not there, this module says so and refuses to run.** It never
falls back to a simulated shock. A stress test that quietly substitutes synthetic
data for a missing window reports a reassuring number about nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from quantlab.config import HOLDOUT_START, assert_holdout_allowed
from quantlab.db.models import OHLCVDaily
from quantlab.risk.var import RiskError

log = structlog.get_logger(__name__)


class StressDataUnavailableError(RiskError):
    """Raised when a stress window has no ingested data.

    Deliberately fatal rather than a fallback. The whole value of stress testing is
    that the shock is real; substituting a synthetic one would produce a number that
    looks like evidence and is not.
    """


@dataclass(frozen=True, slots=True)
class StressWindow:
    """A historical crisis period, defined once and referenced everywhere."""

    name: str
    slug: str
    start: date
    end: date
    #: ``"global"`` or ``"india"`` — an India-only shock says nothing about a
    #: portfolio of US equities and vice versa.
    market: str
    description: str

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    @property
    def predates_sample(self) -> bool:
        """Whether this window falls before the project's ordinary data start.

        These are the windows that ordinary ingestion will never fetch, and the
        reason an explicit stress ingestion exists.
        """

        from quantlab.config import DATA_START

        return self.start < DATA_START

    @property
    def touches_holdout(self) -> bool:
        return self.end >= HOLDOUT_START

    def render(self) -> str:
        return (
            f"{self.slug:16s} {self.start.isoformat()} → {self.end.isoformat()} "
            f"({self.market:6s}) {self.description}"
        )


#: The five required windows (PRD Module 9). Defined once; every stress test reads
#: them from here so a window cannot be quietly redefined mid-project.
STRESS_WINDOWS: tuple[StressWindow, ...] = (
    StressWindow(
        name="Global Financial Crisis",
        slug="gfc",
        start=date(2007, 10, 1),
        end=date(2009, 3, 31),
        market="global",
        description=(
            "An 18-month bear market rather than a crash: tests whether a strategy "
            "survives sustained losses, not just a single bad week"
        ),
    ),
    StressWindow(
        name="COVID crash",
        slug="covid",
        start=date(2020, 2, 1),
        end=date(2020, 4, 30),
        market="global",
        description=(
            "The fastest major drawdown on record, with cross-sectional correlations "
            "converging on 1 and diversification failing when it was needed"
        ),
    ),
    StressWindow(
        name="2022 rate shock",
        slug="rate_shock_2022",
        start=date(2022, 1, 1),
        end=date(2022, 10, 31),
        market="global",
        description=(
            "Sustained multiple compression as rates rose: punishes long-duration "
            "growth exposure that momentum strategies accumulate"
        ),
    ),
    StressWindow(
        name="Demonetisation",
        slug="demonetisation",
        start=date(2016, 11, 1),
        end=date(2016, 12, 31),
        market="india",
        description=(
            "India-specific liquidity shock: 86% of currency in circulation "
            "withdrawn overnight, hitting cash-economy sectors hardest"
        ),
    ),
    StressWindow(
        name="IL&FS credit event",
        slug="ilfs",
        start=date(2018, 9, 1),
        end=date(2018, 10, 31),
        market="india",
        description=(
            "Indian credit contagion: an NBFC default cascading into financials and "
            "mid-caps, with a sector-specific rather than market-wide shock"
        ),
    ),
)

STRESS_WINDOWS_BY_SLUG: dict[str, StressWindow] = {w.slug: w for w in STRESS_WINDOWS}


def get_stress_window(slug: str) -> StressWindow:
    """Look up a window by slug, listing the alternatives if it is unknown."""

    window = STRESS_WINDOWS_BY_SLUG.get(slug)
    if window is None:
        raise RiskError(
            f"unknown stress window {slug!r}; available: "
            f"{', '.join(sorted(STRESS_WINDOWS_BY_SLUG))}"
        )
    return window


# --- Ingestion planning ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StressIngestPlan:
    """The date range that must be ingested to cover every stress window.

    A single contiguous range per window, padded so that returns and any recovery
    measurement have context on both sides. Ordinary ingestion starts at
    ``DATA_START``; these ranges reach back to 2007, which is why they are requested
    separately and retained afterwards.
    """

    windows: tuple[StressWindow, ...]
    #: ``(start, end)`` per window slug, padded.
    ranges: dict[str, tuple[date, date]]
    #: Trading days of padding either side.
    padding_days: int

    def earliest(self) -> date:
        return min(start for start, _ in self.ranges.values())

    def latest(self) -> date:
        return max(end for _, end in self.ranges.values())

    def render(self) -> str:
        lines = [
            "Stress windows to ingest and retain (independent of any backtest range):",
            "",
        ]
        for window in self.windows:
            start, end = self.ranges[window.slug]
            flag = " [predates DATA_START]" if window.predates_sample else ""
            lines.append(
                f"  {window.slug:16s} {start.isoformat()} → {end.isoformat()}{flag}"
            )
        lines += [
            "",
            f"  Full span required: {self.earliest().isoformat()} → "
            f"{self.latest().isoformat()}",
            "",
            "  These ranges are retained regardless of the backtest window. Ordinary "
            "ingestion starts at DATA_START (2015-01-01) and would never fetch the "
            "GFC, so it must be requested explicitly.",
        ]
        return "\n".join(lines)


def stress_window_ingest_plan(
    *, padding_days: int = 30, windows: tuple[StressWindow, ...] | None = None
) -> StressIngestPlan:
    """Date ranges to ingest so every stress window can be replayed.

    Padding gives the replay a return series that starts inside the window rather
    than at its first bar, and leaves room to measure a recovery that completes just
    after the window closes.
    """

    selected = windows or STRESS_WINDOWS
    ranges = {
        window.slug: (
            window.start - timedelta(days=padding_days),
            window.end + timedelta(days=padding_days),
        )
        for window in selected
    }
    return StressIngestPlan(windows=selected, ranges=ranges, padding_days=padding_days)


# --- Coverage ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StressCoverage:
    """How much of a window is actually present in the database."""

    window: StressWindow
    bars: int
    securities: int
    first_bar: date | None
    last_bar: date | None

    @property
    def available(self) -> bool:
        return self.bars > 0

    @property
    def spans_window(self) -> bool:
        """Whether ingested data covers the window from end to end.

        Partial coverage is worse than none if it goes unnoticed: replaying the last
        two weeks of the GFC and calling it a GFC stress test understates the
        drawdown by an order of magnitude.
        """

        if self.first_bar is None or self.last_bar is None:
            return False
        # Allow a week of slack for holidays at either boundary.
        slack = timedelta(days=7)
        return (
            self.first_bar <= self.window.start + slack
            and self.last_bar >= self.window.end - slack
        )

    def render(self) -> str:
        if not self.available:
            return f"  {self.window.slug:16s} NOT INGESTED"
        status = "complete" if self.spans_window else "PARTIAL"
        return (
            f"  {self.window.slug:16s} {self.bars:>7,} bars, "
            f"{self.securities:>4} securities, "
            f"{self.first_bar} → {self.last_bar}  [{status}]"
        )


def stress_coverage(
    session: Session, *, windows: tuple[StressWindow, ...] | None = None
) -> list[StressCoverage]:
    """Report what stress data is present. Run this before trusting a stress test."""

    results: list[StressCoverage] = []
    for window in windows or STRESS_WINDOWS:
        row = session.execute(
            select(
                func.count(OHLCVDaily.id),
                func.count(func.distinct(OHLCVDaily.security_id)),
                func.min(OHLCVDaily.date),
                func.max(OHLCVDaily.date),
            ).where(
                OHLCVDaily.date >= window.start,
                OHLCVDaily.date <= window.end,
            )
        ).one()
        results.append(
            StressCoverage(
                window=window,
                bars=int(row[0] or 0),
                securities=int(row[1] or 0),
                first_bar=row[2],
                last_bar=row[3],
            )
        )
    return results


def load_window_returns(
    session: Session,
    window: StressWindow,
    security_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Load real adjusted-close returns for a window, wide by security.

    Raises :class:`StressDataUnavailableError` when nothing is ingested. There is no
    synthetic fallback, by design.
    """

    stmt = select(OHLCVDaily.date, OHLCVDaily.security_id, OHLCVDaily.adj_close).where(
        OHLCVDaily.date >= window.start, OHLCVDaily.date <= window.end
    )
    if security_ids:
        stmt = stmt.where(OHLCVDaily.security_id.in_(sorted(security_ids)))
    rows = session.execute(stmt).all()

    if not rows:
        plan = stress_window_ingest_plan(windows=(window,))
        start, end = plan.ranges[window.slug]
        raise StressDataUnavailableError(
            f"no ingested data for the {window.name} window "
            f"({window.start.isoformat()} → {window.end.isoformat()}). Stress "
            "testing uses real market data and will not substitute a synthetic "
            f"shock. Ingest it first:\n\n"
            f"    quantlab ingest --tickers <symbols> --start {start.isoformat()} "
            f"--end {end.isoformat()}\n\n"
            "See 'quantlab risk stress-plan' for every window's required range."
        )

    frame = pd.DataFrame(rows, columns=["date", "security_id", "adj_close"])
    frame["date"] = pd.to_datetime(frame["date"])
    wide = frame.pivot(
        index="date", columns="security_id", values="adj_close"
    ).sort_index()
    return wide.pct_change().dropna(how="all")


# --- Replay ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StressResult:
    """What the portfolio did through one real crisis window."""

    window: StressWindow
    #: Positive fraction: 0.38 means a 38% peak-to-trough loss.
    max_drawdown: float
    #: Worst single-day return, as a signed fraction.
    worst_day: float
    worst_day_date: date | None
    total_return: float
    annualised_volatility: float
    #: Trading days from the trough back to the pre-crisis peak. ``None`` means the
    #: portfolio had not recovered by the end of the window — reported as ``None``
    #: rather than as the window length, which would read as a fast recovery.
    recovery_days: int | None
    peak_date: date | None
    trough_date: date | None
    trading_days: int
    securities: int
    #: Mean pairwise correlation during the window. In a crash this converges toward
    #: 1 and diversification stops working.
    mean_correlation: float | None = None

    @property
    def recovered(self) -> bool:
        return self.recovery_days is not None

    def render(self) -> str:
        lines = [
            f"{self.window.name} ({self.window.start} → {self.window.end})",
            f"  max drawdown:      {self.max_drawdown:7.2%}",
            f"  worst day:         {self.worst_day:7.2%}"
            + (f" on {self.worst_day_date}" if self.worst_day_date else ""),
            f"  total return:      {self.total_return:7.2%}",
            f"  volatility:        {self.annualised_volatility:7.2%} annualised",
            "  recovery:          "
            + (
                f"{self.recovery_days} trading days"
                if self.recovered
                else "NOT RECOVERED within the window"
            ),
            f"  trading days:      {self.trading_days}, {self.securities} securities",
        ]
        if self.mean_correlation is not None:
            lines.append(f"  mean correlation:  {self.mean_correlation:7.3f}")
            if self.mean_correlation > 0.7:
                lines.append(
                    "  Correlations converged: the portfolio's diversification "
                    "was not available during this window."
                )
        return "\n".join(lines)


def replay_window(
    returns: pd.DataFrame | pd.Series,
    window: StressWindow,
    *,
    weights: dict[str, float] | pd.Series | None = None,
    periods_per_year: int = 252,
) -> StressResult:
    """Replay a portfolio through a window's real returns.

    ``returns`` is either a per-security frame (with ``weights``) or an
    already-aggregated portfolio return series. Weights are held fixed through the
    window — a crisis is not the time a rebalance would have been executed cleanly,
    and assuming otherwise flatters the result.
    """

    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise StressDataUnavailableError(
                f"no returns to replay for the {window.name} window"
            )
        if weights is None:
            series = returns.mean(axis=1)
            securities = int(returns.shape[1])
        else:
            weight_series = pd.Series(weights, dtype=float)
            common = [c for c in returns.columns if c in weight_series.index]
            if not common:
                raise StressDataUnavailableError(
                    f"none of the {len(weight_series)} weighted securities have "
                    f"ingested data in the {window.name} window. The portfolio "
                    "cannot be replayed through a period it has no prices for."
                )
            aligned = weight_series.loc[common]
            total = float(aligned.sum())
            if total == 0:
                raise RiskError("weights sum to zero")
            series = (returns[common] * (aligned / total)).sum(axis=1)
            securities = len(common)
        mean_correlation = _mean_correlation(returns)
    else:
        series = returns.dropna()
        securities = 1
        mean_correlation = None

    series = series.dropna()
    if len(series) < 2:
        raise StressDataUnavailableError(
            f"only {len(series)} return observations in the {window.name} window — "
            "not enough to measure a drawdown"
        )

    # Positional rather than index-based throughout. Prepending the opening level
    # with the first return's own date would create a duplicate index label, and
    # every subsequent .loc lookup would silently return a two-row Series.
    values = series.to_numpy(dtype=float)
    equity = np.cumprod(1.0 + values)
    # Position 0 is the opening level, so a portfolio that only ever falls still
    # records a drawdown from where it started.
    with_start = np.concatenate([[1.0], equity])
    running_peak = np.maximum.accumulate(with_start)
    drawdowns = with_start / running_peak - 1.0

    trough_position = int(drawdowns.argmin())
    max_drawdown = -float(drawdowns[trough_position])
    peak_position = int(with_start[: trough_position + 1].argmax())
    peak_value = float(with_start[peak_position])

    after = with_start[trough_position:]
    recovered_positions = np.nonzero(after >= peak_value)[0]
    recovery_days: int | None = (
        int(recovered_positions[0]) if recovered_positions.size else None
    )

    def date_at(position: int) -> date:
        """Map an equity position back to a date; position 0 is the opening level."""

        return pd.Timestamp(series.index[max(0, position - 1)]).date()

    peak_index = date_at(peak_position)
    trough_index = date_at(trough_position)
    worst_position = int(values.argmin())
    return StressResult(
        window=window,
        max_drawdown=max_drawdown,
        worst_day=float(series.iloc[worst_position]),
        worst_day_date=pd.Timestamp(series.index[worst_position]).date(),
        total_return=float(equity[-1] - 1.0),
        annualised_volatility=float(series.std(ddof=1) * np.sqrt(periods_per_year)),
        recovery_days=recovery_days,
        peak_date=peak_index,
        trough_date=trough_index,
        trading_days=len(series),
        securities=securities,
        mean_correlation=mean_correlation,
    )


def _mean_correlation(returns: pd.DataFrame) -> float | None:
    """Mean off-diagonal correlation, or ``None`` when it is undefined.

    Undefined happens for real reasons — a suspended security with a flat return
    series has no correlation with anything — so it returns ``None`` rather than
    reporting a NaN that would print as a number.
    """

    if returns.shape[1] < 2:
        return None
    matrix = returns.corr().to_numpy(dtype=float)
    n = matrix.shape[0]
    off_diagonal = matrix[~np.eye(n, dtype=bool)]
    if off_diagonal.size == 0 or np.all(np.isnan(off_diagonal)):
        return None
    value = float(np.nanmean(off_diagonal))
    return value if np.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class StressTestReport:
    """Every window the portfolio was replayed through, and every one it was not."""

    results: tuple[StressResult, ...]
    #: Windows with no ingested data, as ``(slug, reason)``. Reported, never hidden:
    #: a stress report covering two of five windows must say which three are missing.
    unavailable: tuple[tuple[str, str], ...]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "window": r.window.slug,
                    "market": r.window.market,
                    "max_drawdown": r.max_drawdown,
                    "worst_day": r.worst_day,
                    "total_return": r.total_return,
                    "recovery_days": r.recovery_days,
                    "recovered": r.recovered,
                    "trading_days": r.trading_days,
                }
                for r in self.results
            ]
        )

    @property
    def worst_window(self) -> StressResult | None:
        return max(self.results, key=lambda r: r.max_drawdown, default=None)

    @property
    def complete(self) -> bool:
        """Whether every required window was replayable."""

        return not self.unavailable

    def render(self) -> str:
        lines = ["Stress test — real historical windows, no synthetic shocks", ""]
        for result in self.results:
            lines.append(result.render())
            lines.append("")
        if self.unavailable:
            lines.append("Windows that could NOT be tested (no ingested data):")
            for slug, reason in self.unavailable:
                lines.append(f"  {slug:16s} {reason}")
            lines += [
                "",
                "This report is INCOMPLETE. The missing windows are not evidence of "
                "resilience; they are absent data. Run 'quantlab risk stress-plan' "
                "for the ranges to ingest.",
                "",
            ]
        worst = self.worst_window
        if worst is not None:
            lines.append(
                f"Worst window: {worst.window.name}, {worst.max_drawdown:.2%} "
                "drawdown"
                + (
                    f", recovered in {worst.recovery_days} trading days"
                    if worst.recovered
                    else ", never recovered within the window"
                )
            )
        return "\n".join(lines)


def stress_test(
    session: Session,
    *,
    weights: dict[str, float] | pd.Series | None = None,
    windows: tuple[StressWindow, ...] | None = None,
    allow_holdout: bool = False,
) -> StressTestReport:
    """Replay a portfolio through every stress window with ingested data.

    Windows without data are recorded in :attr:`StressTestReport.unavailable` rather
    than skipped silently, so the report cannot read as a pass when it is really an
    absence.

    A window reaching into the locked holdout is refused unless explicitly unlocked —
    the lock applies here as everywhere else.
    """

    selected = windows or STRESS_WINDOWS
    results: list[StressResult] = []
    unavailable: list[tuple[str, str]] = []

    for window in selected:
        if window.touches_holdout:
            assert_holdout_allowed(
                window.end,
                holdout=allow_holdout,
                context=f"stress window {window.slug!r}",
            )
        try:
            returns = load_window_returns(
                session,
                window,
                security_ids=list(weights) if weights is not None else None,
            )
            results.append(replay_window(returns, window, weights=weights))
        except StressDataUnavailableError as exc:
            unavailable.append((window.slug, str(exc).split(".")[0]))
            log.warning(
                "stress.window_unavailable", window=window.slug, reason=str(exc)[:200]
            )

    return StressTestReport(results=tuple(results), unavailable=tuple(unavailable))
