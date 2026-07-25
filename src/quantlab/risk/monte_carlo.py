"""Monte Carlo simulation from bootstrapped returns (PRD Module 9).

**Bootstrapped, not Gaussian.** Drawing from a fitted normal distribution produces
paths that are smoother than any market has ever been: no fat tails, no volatility
clustering, no skew. Resampling the portfolio's own realised returns keeps every
feature of the empirical distribution that a parametric fit throws away.

Two resampling schemes, and the difference matters:

* **IID bootstrap** — draw single days with replacement. Preserves the return
  distribution exactly; destroys autocorrelation and volatility clustering. Drawdowns
  come out systematically *shallower* than reality, because real losses arrive in
  runs and shuffled ones do not.
* **Stationary block bootstrap** — draw contiguous blocks of geometrically
  distributed length (Politis & Romano). Preserves short-range dependence, so the
  simulated drawdowns are believable. This is the default for anything involving a
  drawdown or a path-dependent statistic.

What this can and cannot tell you: it estimates the distribution of outcomes *given
that the future resembles the sampled past*. It cannot produce a crisis the sample
never contained. That is what the stress module is for, and why it uses real crisis
windows rather than simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from quantlab.risk.var import TRADING_DAYS, RiskError

BootstrapMethod = Literal["iid", "block"]

#: Percentiles reported by default.
DEFAULT_PERCENTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """The distribution of simulated outcomes."""

    #: Terminal value per path, as a multiple of starting capital.
    terminal_values: np.ndarray = field(repr=False)
    #: Maximum drawdown per path, as a positive fraction.
    max_drawdowns: np.ndarray = field(repr=False)
    #: Annualised return per path.
    annual_returns: np.ndarray = field(repr=False)
    n_simulations: int
    horizon_days: int
    method: BootstrapMethod
    block_size: int | None
    #: Observations in the source sample the paths were drawn from.
    source_observations: int
    periods_per_year: int = TRADING_DAYS
    #: Equity quantile *bands* over the horizon — the fan chart — indexed by day
    #: with one column per requested percentile. ``None`` unless ``monte_carlo``
    #: was called with ``path_percentiles``.
    #:
    #: Bands, not paths: retaining 10,000 × 252 floats to plot five lines is a
    #: waste, and a band is what a fan chart draws anyway. Note that a band is
    #: not a path — the 5th-percentile line is the 5th percentile *on each day*,
    #: which no single simulation traced.
    path_bands: pd.DataFrame | None = field(default=None, repr=False)

    def percentiles(
        self, percentiles: tuple[float, ...] = DEFAULT_PERCENTILES
    ) -> pd.DataFrame:
        """Terminal value, annual return and max drawdown at each percentile."""

        return pd.DataFrame(
            {
                "terminal_value": [
                    float(np.quantile(self.terminal_values, p)) for p in percentiles
                ],
                "annual_return": [
                    float(np.quantile(self.annual_returns, p)) for p in percentiles
                ],
                # Drawdown percentiles are inverted: the 95th percentile of *risk*
                # is the deep one, so report the deep tail at high percentiles.
                "max_drawdown": [
                    float(np.quantile(self.max_drawdowns, p)) for p in percentiles
                ],
            },
            index=pd.Index(percentiles, name="percentile"),
        )

    @property
    def median_terminal(self) -> float:
        return float(np.median(self.terminal_values))

    @property
    def probability_of_loss(self) -> float:
        """Share of paths ending below the starting capital."""

        return float(np.count_nonzero(self.terminal_values < 1.0) / self.n_simulations)

    def probability_of_drawdown(self, threshold: float) -> float:
        """Share of paths whose worst drawdown exceeds ``threshold`` (e.g. 0.20)."""

        if not 0.0 < threshold < 1.0:
            raise RiskError(f"threshold must lie in (0, 1), got {threshold}")
        return float(
            np.count_nonzero(self.max_drawdowns >= threshold) / self.n_simulations
        )

    def probability_of_ruin(self, threshold: float = 0.5) -> float:
        """Share of paths losing at least ``threshold`` of capital at the trough."""

        return self.probability_of_drawdown(threshold)

    def value_at_risk(self, confidence: float = 0.95) -> float:
        """Terminal-value VaR: the loss exceeded by ``1-confidence`` of paths."""

        quantile = float(np.quantile(self.terminal_values, 1.0 - confidence))
        return max(0.0, 1.0 - quantile)

    def render(self) -> str:
        table = self.percentiles()
        lines = [
            f"Monte Carlo ({self.n_simulations:,} paths, {self.horizon_days}-day "
            f"horizon, {self.method} bootstrap"
            + (f", block≈{self.block_size}" if self.block_size else "")
            + f", drawn from {self.source_observations} observations)",
            "",
            f"  {'percentile':>10}  {'terminal':>9}  {'annual':>8}  {'max DD':>8}",
        ]
        for percentile, row in table.iterrows():
            lines.append(
                f"  {float(str(percentile)):>10.0%}  {row['terminal_value']:>9.3f}  "
                f"{row['annual_return']:>8.2%}  {row['max_drawdown']:>8.2%}"
            )
        lines += [
            "",
            f"  probability of ending down:      {self.probability_of_loss:6.1%}",
            f"  probability of a 20% drawdown:   "
            f"{self.probability_of_drawdown(0.20):6.1%}",
            f"  probability of a 50% drawdown:   {self.probability_of_ruin(0.50):6.1%}",
        ]
        if self.method == "iid":
            lines += [
                "",
                "  NOTE: the IID bootstrap shuffles days independently, which breaks "
                "volatility clustering. Real losses arrive in runs, so these "
                "drawdowns are optimistic. Use method='block' for drawdown work.",
            ]
        lines += [
            "",
            "  This describes futures that resemble the sampled past. It cannot "
            "produce a crisis the sample does not contain — see the stress module.",
        ]
        return "\n".join(lines)


def _prepare(
    returns: pd.Series | np.ndarray, minimum: int = 60
) -> npt.NDArray[np.float64]:
    raw = np.asarray(returns, dtype=np.float64).ravel()
    values: npt.NDArray[np.float64] = raw[np.isfinite(raw)]
    if values.size < minimum:
        raise RiskError(
            f"bootstrapping needs at least {minimum} finite observations, got "
            f"{values.size}. Resampling a shorter history mostly reproduces the "
            "same few days."
        )
    return values


def _iid_paths(
    values: np.ndarray, rng: np.random.Generator, n_simulations: int, horizon: int
) -> np.ndarray:
    indices = rng.integers(0, values.size, size=(n_simulations, horizon))
    return values[indices]


def _block_paths(
    values: np.ndarray,
    rng: np.random.Generator,
    n_simulations: int,
    horizon: int,
    block_size: int,
) -> np.ndarray:
    """Stationary bootstrap: geometric block lengths, wrapping at the sample end.

    Wrapping (rather than truncating) is what makes the resampled series stationary
    — every observation is equally likely to appear, whereas truncation
    under-samples the end of the history.
    """

    n = values.size
    probability = 1.0 / max(1, block_size)
    paths = np.empty((n_simulations, horizon), dtype=float)
    # Vectorised across paths: at each step either continue the current block or
    # jump to a fresh random start.
    position = rng.integers(0, n, size=n_simulations)
    for step in range(horizon):
        paths[:, step] = values[position]
        restart = rng.random(n_simulations) < probability
        position = np.where(
            restart, rng.integers(0, n, size=n_simulations), (position + 1) % n
        )
    return paths


def monte_carlo(
    returns: pd.Series | np.ndarray,
    *,
    n_simulations: int = 10_000,
    horizon_days: int = TRADING_DAYS,
    method: BootstrapMethod = "block",
    block_size: int | None = None,
    periods_per_year: int = TRADING_DAYS,
    seed: int | None = 0,
    path_percentiles: tuple[float, ...] | None = None,
) -> MonteCarloResult:
    """Simulate forward paths by resampling ``returns``.

    ``block_size`` defaults to ``⌈n^(1/3)⌉``, the standard rule for the stationary
    bootstrap — about 12 days on five years of data, which is the right order of
    magnitude for equity volatility clustering.

    ``seed`` is fixed by default so a reported percentile is reproducible. Pass
    ``None`` for genuine randomness, and expect the numbers to move.

    ``path_percentiles`` additionally retains the equity quantile bands over the
    horizon (:attr:`MonteCarloResult.path_bands`), which is what a fan chart
    plots. It is off by default because the summary statistics do not need it.
    """

    values = _prepare(returns)
    if n_simulations < 1:
        raise RiskError(f"n_simulations must be at least 1, got {n_simulations}")
    if horizon_days < 1:
        raise RiskError(f"horizon_days must be at least 1, got {horizon_days}")

    rng = np.random.default_rng(seed)
    resolved_block: int | None = None
    if method == "iid":
        paths = _iid_paths(values, rng, n_simulations, horizon_days)
    elif method == "block":
        resolved_block = block_size or max(2, int(np.ceil(values.size ** (1.0 / 3.0))))
        paths = _block_paths(values, rng, n_simulations, horizon_days, resolved_block)
    else:
        raise RiskError(f"unknown bootstrap method {method!r}; use iid | block")

    # Equity paths, starting at 1.0.
    equity = np.cumprod(1.0 + paths, axis=1)
    terminal = equity[:, -1]

    # Drawdown against the running peak, including the starting value so a path
    # that only ever falls still reports a drawdown.
    with_start = np.concatenate([np.ones((n_simulations, 1)), equity], axis=1)
    running_peak = np.maximum.accumulate(with_start, axis=1)
    drawdowns = 1.0 - with_start / running_peak
    max_drawdowns = drawdowns.max(axis=1)

    years = horizon_days / periods_per_year
    with np.errstate(invalid="ignore", divide="ignore"):
        annual = np.where(
            terminal > 0, np.power(np.maximum(terminal, 1e-12), 1.0 / years) - 1.0, -1.0
        )

    bands: pd.DataFrame | None = None
    if path_percentiles:
        bands = pd.DataFrame(
            {
                f"p{round(p * 100):02d}": np.quantile(equity, p, axis=0)
                for p in path_percentiles
            },
            index=pd.RangeIndex(1, horizon_days + 1, name="day"),
        )

    return MonteCarloResult(
        terminal_values=terminal,
        max_drawdowns=max_drawdowns,
        annual_returns=annual,
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        method=method,
        block_size=resolved_block,
        source_observations=int(values.size),
        periods_per_year=periods_per_year,
        path_bands=bands,
    )
