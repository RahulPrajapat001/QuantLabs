"""Multiple-testing corrections: deflated Sharpe, null max Sharpe, and PBO.

The best of two hundred backtests has a high Sharpe whether or not any of them
has an edge. That is not a subtlety — it is the single most common reason a
published retail strategy fails out of sample. Three corrections, all required
by PRD §4.4:

* :func:`deflated_sharpe_ratio` — Bailey & López de Prado (2014). Given the
  number of trials, the spread of Sharpes across those trials, and the shape of
  the return distribution, it returns the probability that the true Sharpe
  exceeds zero. A Sharpe of 1.8 selected from 300 trials routinely deflates to a
  probability near a coin flip.
* :func:`expected_max_sharpe_null` — simulates the selection process itself:
  ``N`` strategies with *no edge*, matched on length and volatility. The
  distribution of the maximum is what a real strategy has to beat. Failing to
  clear it means the result is indistinguishable from noise.
* :func:`probability_of_backtest_overfitting` — PBO via combinatorially
  symmetric cross-validation (Bailey, Borwein, López de Prado & Zhu, 2015): the
  probability that the configuration chosen in-sample lands below median
  out-of-sample.

**Trials counts come from the research log.** Every function that needs one takes
a :class:`~quantlab.research.log.ResearchLog` and counts rows itself. There is no
parameter to pass ``N=12`` when the log says 300; that is the entire point.

Units convention: every Sharpe crossing this module's public API is
**annualised**, paired with ``periods_per_year``. Conversion to the per-period
units the formulae require happens internally, once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import structlog
from scipy import stats

from quantlab.research.log import ResearchLog, ResearchLogError

log = structlog.get_logger(__name__)

#: Euler–Mascheroni constant, from the Gumbel approximation to E[max] of N
#: independent standard normals (Bailey & López de Prado 2014, eq. 5).
_EULER_MASCHERONI = 0.5772156649015329

#: Default probability threshold at which a deflated Sharpe is called
#: significant: the true Sharpe exceeds zero with 95% probability.
DEFAULT_SIGNIFICANCE = 0.95


class DeflationError(ValueError):
    """Raised when a deflated Sharpe cannot be computed from what the log holds."""


# --- Probabilistic and deflated Sharpe ---------------------------------------


def _psr_per_period(
    *,
    sharpe: float,
    benchmark: float,
    n_periods: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Probabilistic Sharpe ratio, all Sharpes in **per-period** units.

    ``Ẑ = (SR - SR*) · √(T-1) / √(1 - γ3·SR + (γ4-1)/4 · SR²)``, and
    ``PSR = Φ(Ẑ)``. ``γ4`` is non-excess kurtosis (3.0 for a normal): negative
    skew and fat tails both *widen* the denominator, lowering the probability —
    which is why a strategy that sells volatility looks worse here than its
    Sharpe alone suggests.
    """

    if n_periods < 2:
        raise DeflationError(
            f"need at least 2 return observations for a probabilistic Sharpe, "
            f"got {n_periods}"
        )
    variance_term = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if variance_term <= 0.0:
        raise DeflationError(
            "the Sharpe estimator's variance term is non-positive "
            f"(skew={skew:.4f}, kurtosis={kurtosis:.4f}, per-period SR="
            f"{sharpe:.4f}); the return distribution is too degenerate for the "
            "probabilistic Sharpe ratio to be meaningful"
        )
    z = (sharpe - benchmark) * math.sqrt(n_periods - 1) / math.sqrt(variance_term)
    return float(stats.norm.cdf(z))


def probabilistic_sharpe_ratio(
    *,
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    n_periods: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = 252,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark_sharpe``.

    Both Sharpes annualised. This is the deflated Sharpe's building block: with
    ``benchmark_sharpe=0`` it corrects only for track-record length and return
    shape, *not* for multiple testing. Use
    :func:`deflated_sharpe_ratio` for anything that was selected from a search.
    """

    scale = math.sqrt(float(periods_per_year))
    return _psr_per_period(
        sharpe=observed_sharpe / scale,
        benchmark=benchmark_sharpe / scale,
        n_periods=n_periods,
        skew=skew,
        kurtosis=kurtosis,
    )


def expected_max_sharpe_analytic(
    *,
    trials: int,
    sharpe_variance: float,
    periods_per_year: int = 252,
) -> float:
    """Expected maximum **annualised** Sharpe under the null, in closed form.

    ``E[max SR] ≈ √V[SR] · ((1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)))``

    This is the benchmark the deflated Sharpe measures against: the Sharpe you
    would expect to see from the best of ``N`` genuinely edgeless trials, given
    how much Sharpes varied across those trials. With ``trials <= 1`` there was no
    selection, so the benchmark is 0.
    """

    if trials < 1:
        raise DeflationError(f"trials must be at least 1, got {trials}")
    if sharpe_variance < 0.0:
        raise DeflationError(
            f"sharpe_variance must be non-negative, got {sharpe_variance}"
        )
    if trials == 1 or sharpe_variance == 0.0:
        return 0.0

    n = float(trials)
    gamma = _EULER_MASCHERONI
    quantile_1 = float(stats.norm.ppf(1.0 - 1.0 / n))
    quantile_2 = float(stats.norm.ppf(1.0 - 1.0 / (n * math.e)))
    return math.sqrt(sharpe_variance) * (
        (1.0 - gamma) * quantile_1 + gamma * quantile_2
    )


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    """A deflated Sharpe and every input that produced it.

    Carries its provenance deliberately: this object is what the reporting layer
    accepts in place of a bare float, so the trials count travels with the number
    instead of being reconstructed later from memory.
    """

    observed_sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    trials: int
    sharpe_variance: float
    n_periods: int
    skew: float
    kurtosis: float
    periods_per_year: int
    #: ``run_id`` of the logged run this deflation describes, when it came from
    #: the log (it always does outside of unit tests).
    run_id: str | None = None
    #: Probabilistic Sharpe against a zero benchmark — the same number with the
    #: multiple-testing correction switched off, for comparison.
    probabilistic_sharpe: float | None = None

    @property
    def significant(self) -> bool:
        """Whether the true Sharpe exceeds zero with ≥95% probability."""

        return self.deflated_sharpe >= DEFAULT_SIGNIFICANCE

    def clears_selection_bias(self) -> bool:
        """Whether the observed Sharpe even exceeds the best-of-N null benchmark."""

        return self.observed_sharpe > self.expected_max_sharpe

    def render(self) -> str:
        """One-line-per-fact rendering, for the CLI and the report generator."""

        verdict = (
            "SIGNIFICANT at 95%"
            if self.significant
            else "NOT significant — consistent with noise from the search"
        )
        lines = [
            f"Observed Sharpe (annualised):   {self.observed_sharpe:8.3f}",
            f"Trials (from research_log):     {self.trials:8d}",
            f"Variance of trial Sharpes:      {self.sharpe_variance:8.4f}",
            f"Expected max Sharpe under null: {self.expected_max_sharpe:8.3f}",
            f"Deflated Sharpe (P[SR > 0]):    {self.deflated_sharpe:8.4f}   {verdict}",
        ]
        if self.probabilistic_sharpe is not None:
            lines.append(
                f"  (undeflated PSR, 1 trial:     {self.probabilistic_sharpe:8.4f})"
            )
        lines.append(
            f"Return shape: T={self.n_periods}, skew={self.skew:.3f}, "
            f"kurtosis={self.kurtosis:.3f} (non-excess)"
        )
        if not self.clears_selection_bias():
            lines.append(
                "  The observed Sharpe does not even exceed the expected maximum "
                "of an edgeless search of this size."
            )
        if self.run_id is not None:
            lines.append(f"run_id: {self.run_id}")
        return "\n".join(lines)


def deflated_sharpe_ratio(
    *,
    observed_sharpe: float,
    trials: int,
    sharpe_variance: float,
    n_periods: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = 252,
    run_id: str | None = None,
) -> DeflatedSharpeResult:
    """The deflated Sharpe ratio: P[true Sharpe > 0] after correcting for search.

    Mechanically, the probabilistic Sharpe ratio benchmarked against the expected
    maximum of ``trials`` edgeless attempts rather than against zero. Prefer
    :func:`deflate_run` or :func:`deflate_strategy`, which read ``trials`` and
    ``sharpe_variance`` out of the research log rather than trusting a caller to
    remember them.
    """

    expected_max = expected_max_sharpe_analytic(
        trials=trials,
        sharpe_variance=sharpe_variance,
        periods_per_year=periods_per_year,
    )
    scale = math.sqrt(float(periods_per_year))
    deflated = _psr_per_period(
        sharpe=observed_sharpe / scale,
        benchmark=expected_max / scale,
        n_periods=n_periods,
        skew=skew,
        kurtosis=kurtosis,
    )
    undeflated = _psr_per_period(
        sharpe=observed_sharpe / scale,
        benchmark=0.0,
        n_periods=n_periods,
        skew=skew,
        kurtosis=kurtosis,
    )
    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        deflated_sharpe=deflated,
        expected_max_sharpe=expected_max,
        trials=trials,
        sharpe_variance=sharpe_variance,
        n_periods=n_periods,
        skew=skew,
        kurtosis=kurtosis,
        periods_per_year=periods_per_year,
        run_id=run_id,
        probabilistic_sharpe=undeflated,
    )


# --- Log-backed entry points -------------------------------------------------


def deflate_run(
    research_log: ResearchLog,
    run_id: str,
    *,
    scope_strategy: str | None = None,
) -> DeflatedSharpeResult:
    """Deflate one logged run, reading the trials count from the log.

    ``scope_strategy`` narrows the trial universe to a single strategy name. The
    default — the **whole log** — is the honest one: strategy search and model
    search are one selection process, so a result chosen after 300 runs across
    five families is deflated by 300 (PRD §Module 12, "Trials accounting").
    """

    entry = research_log.entry(run_id)
    if entry.sharpe is None:
        raise DeflationError(
            f"run {run_id} has no recorded Sharpe (status={entry.status!r}); there "
            "is nothing to deflate"
        )
    if entry.n_periods is None:
        raise DeflationError(
            f"run {run_id} recorded no observation count, so its Sharpe cannot be "
            "deflated. Log results via RunResult.from_returns so the return "
            "moments are captured with the Sharpe."
        )
    trials = research_log.trials(strategy_name=scope_strategy)
    variance = research_log.sharpe_variance(strategy_name=scope_strategy)
    result = deflated_sharpe_ratio(
        observed_sharpe=float(entry.sharpe),
        trials=trials,
        sharpe_variance=variance,
        n_periods=int(entry.n_periods),
        skew=float(entry.returns_skew or 0.0),
        kurtosis=float(entry.returns_kurtosis or 3.0),
        periods_per_year=int(entry.periods_per_year),
        run_id=run_id,
    )
    log.info(
        "research.deflated",
        run_id=run_id,
        strategy=entry.strategy_name,
        observed_sharpe=result.observed_sharpe,
        trials=trials,
        deflated_sharpe=result.deflated_sharpe,
        significant=result.significant,
    )
    return result


def deflate_strategy(
    research_log: ResearchLog,
    strategy_name: str,
    *,
    scope_strategy: str | None = None,
) -> DeflatedSharpeResult:
    """Deflate a strategy's **best** logged run.

    The best run is the one a researcher is tempted to report, so it is the one
    that must carry the correction.
    """

    entry = research_log.best_run(strategy_name=strategy_name)
    if entry is None:
        raise ResearchLogError(
            f"no completed run of {strategy_name!r} in research_log; nothing to "
            "deflate. Run the backtest first — results are only reportable once "
            "they are logged."
        )
    return deflate_run(research_log, entry.run_id, scope_strategy=scope_strategy)


# --- Expected maximum Sharpe under the null (simulated) ----------------------


@dataclass(frozen=True, slots=True)
class NullMaxSharpeDistribution:
    """Distribution of the best Sharpe from ``trials`` edgeless strategies."""

    trials: int
    n_periods: int
    periods_per_year: int
    volatility: float
    #: One simulated best-of-``trials`` annualised Sharpe per simulation.
    max_sharpes: np.ndarray = field(repr=False)
    #: ``"gaussian"`` or ``"bootstrap"`` (resampled from the strategy's own
    #: demeaned returns, which preserves skew and fat tails).
    method: str = "gaussian"

    @property
    def mean(self) -> float:
        return float(self.max_sharpes.mean())

    @property
    def std(self) -> float:
        return float(self.max_sharpes.std(ddof=1))

    def quantile(self, q: float) -> float:
        """The ``q``-quantile of the simulated maxima (annualised Sharpe)."""

        return float(np.quantile(self.max_sharpes, q))

    def p_value(self, observed_sharpe: float) -> float:
        """Fraction of edgeless searches that beat ``observed_sharpe``.

        This is a one-sided empirical p-value for "my best strategy is better
        than the best of an equally large search over nothing".
        """

        exceed = int(np.count_nonzero(self.max_sharpes >= observed_sharpe))
        return (exceed + 1) / (self.max_sharpes.size + 1)

    def clears(self, observed_sharpe: float, *, level: float = 0.95) -> bool:
        """Whether ``observed_sharpe`` exceeds the ``level`` quantile of the null."""

        return observed_sharpe > self.quantile(level)

    def render(self, observed_sharpe: float | None = None) -> str:
        lines = [
            f"Expected max Sharpe under the null "
            f"({self.method}, {self.trials} zero-edge strategies, "
            f"T={self.n_periods}, vol={self.volatility:.4f}/period)",
            f"  mean of maxima:   {self.mean:8.3f}",
            f"  std of maxima:    {self.std:8.3f}",
            f"  median:           {self.quantile(0.5):8.3f}",
            f"  95th percentile:  {self.quantile(0.95):8.3f}",
            f"  99th percentile:  {self.quantile(0.99):8.3f}",
        ]
        if observed_sharpe is not None:
            verdict = "clears the null" if self.clears(observed_sharpe) else "IS NOISE"
            lines.append(
                f"  observed {observed_sharpe:.3f}: empirical p="
                f"{self.p_value(observed_sharpe):.4f} — {verdict}"
            )
        return "\n".join(lines)


def expected_max_sharpe_null(
    *,
    trials: int,
    n_periods: int,
    volatility: float = 0.01,
    periods_per_year: int = 252,
    n_simulations: int = 1000,
    returns: Sequence[float] | np.ndarray | None = None,
    seed: int | None = 0,
) -> NullMaxSharpeDistribution:
    """Simulate ``trials`` zero-edge strategies and return the max-Sharpe distribution.

    Each simulated strategy has exactly zero expected return, ``n_periods``
    observations and the given per-period ``volatility`` — matched to the real
    strategy on everything except edge. Repeating the whole search
    ``n_simulations`` times gives the distribution of the best Sharpe such a
    search produces by luck alone.

    Pass ``returns`` to bootstrap from the strategy's own demeaned returns
    instead of drawing Gaussians. That matters: fat tails and skew change the
    distribution of the maximum, and a Gaussian null is the optimistic one.
    """

    if trials < 1:
        raise DeflationError(f"trials must be at least 1, got {trials}")
    if n_periods < 2:
        raise DeflationError(f"n_periods must be at least 2, got {n_periods}")
    if n_simulations < 1:
        raise DeflationError(f"n_simulations must be at least 1, got {n_simulations}")

    rng = np.random.default_rng(seed)
    annualiser = math.sqrt(float(periods_per_year))

    pool: np.ndarray | None = None
    method = "gaussian"
    if returns is not None:
        pool = np.asarray(returns, dtype=float).ravel()
        pool = pool[np.isfinite(pool)]
        if pool.size < 2:
            raise DeflationError(
                "bootstrap null needs at least 2 finite return observations"
            )
        # Demean: the null is zero edge by construction, whatever the real
        # strategy earned.
        pool = pool - pool.mean()
        volatility = float(pool.std(ddof=1))
        method = "bootstrap"
    elif volatility <= 0.0:
        raise DeflationError(f"volatility must be positive, got {volatility}")

    # Chunk over simulations so the working array stays bounded regardless of
    # trials x n_periods x n_simulations.
    budget = 4_000_000
    per_sim = max(1, trials * n_periods)
    chunk = max(1, min(n_simulations, budget // per_sim))

    maxima = np.empty(n_simulations, dtype=float)
    done = 0
    while done < n_simulations:
        size = min(chunk, n_simulations - done)
        if pool is None:
            draws = rng.normal(0.0, volatility, size=(size, trials, n_periods))
        else:
            idx = rng.integers(0, pool.size, size=(size, trials, n_periods))
            draws = pool[idx]
        means = draws.mean(axis=2)
        sds = draws.std(axis=2, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpes = np.where(sds > 0.0, means / sds * annualiser, 0.0)
        maxima[done : done + size] = sharpes.max(axis=1)
        done += size

    return NullMaxSharpeDistribution(
        trials=trials,
        n_periods=n_periods,
        periods_per_year=periods_per_year,
        volatility=volatility,
        max_sharpes=maxima,
        method=method,
    )


def null_max_sharpe_for_run(
    research_log: ResearchLog,
    run_id: str,
    *,
    scope_strategy: str | None = None,
    n_simulations: int = 1000,
    returns: Sequence[float] | np.ndarray | None = None,
    seed: int | None = 0,
) -> NullMaxSharpeDistribution:
    """Simulate the null for a logged run, with ``trials`` read from the log.

    Volatility is recovered from the run's own logged Sharpe and returns
    (``vol = |mean| / |SR_period|`` is not identifiable from the Sharpe alone, so
    the run's ``volatility`` metric is used when present, else a unit-scale
    default — the max-Sharpe distribution of zero-mean draws is scale-invariant,
    so this only matters for the bootstrap variant).
    """

    entry = research_log.entry(run_id)
    if entry.n_periods is None:
        raise DeflationError(
            f"run {run_id} recorded no observation count; the null cannot be "
            "matched on length"
        )
    trials = research_log.trials(strategy_name=scope_strategy)
    metrics: Mapping[str, object] = entry.metrics or {}
    raw_vol = metrics.get("volatility")
    volatility = float(raw_vol) if isinstance(raw_vol, (int, float)) else 0.01
    return expected_max_sharpe_null(
        trials=max(trials, 1),
        n_periods=int(entry.n_periods),
        volatility=volatility,
        periods_per_year=int(entry.periods_per_year),
        n_simulations=n_simulations,
        returns=returns,
        seed=seed,
    )


# --- Probability of backtest overfitting (CSCV) ------------------------------


@dataclass(frozen=True, slots=True)
class PBOResult:
    """Probability of backtest overfitting, via combinatorially symmetric CV."""

    pbo: float
    n_configurations: int
    n_splits: int
    n_combinations: int
    #: Logit of the OOS relative rank of the IS-best configuration, per split.
    logits: np.ndarray = field(repr=False)
    #: In-sample Sharpe of the selected configuration, per split.
    is_sharpes: np.ndarray = field(repr=False)
    #: Out-of-sample Sharpe of that same configuration, per split.
    oos_sharpes: np.ndarray = field(repr=False)

    @property
    def performance_degradation(self) -> float:
        """OLS slope of OOS Sharpe on IS Sharpe for the selected configurations.

        A slope at or below zero is the signature of overfitting: picking a
        *better* in-sample configuration buys nothing, or costs, out of sample.
        """

        if self.is_sharpes.size < 2:
            return float("nan")
        x = self.is_sharpes
        y = self.oos_sharpes
        var = float(x.var())
        if var == 0.0:
            return float("nan")
        return float(((x - x.mean()) * (y - y.mean())).mean() / var)

    @property
    def probability_of_loss(self) -> float:
        """Share of splits where the IS-best configuration lost money OOS."""

        if self.oos_sharpes.size == 0:
            return float("nan")
        return float(np.count_nonzero(self.oos_sharpes < 0.0) / self.oos_sharpes.size)

    def render(self) -> str:
        lines = [
            f"PBO (CSCV, S={self.n_splits}, {self.n_combinations} splits, "
            f"{self.n_configurations} configurations)",
            f"  probability of backtest overfitting: {self.pbo:6.3f}",
            f"  performance degradation (slope):    "
            f"{self.performance_degradation:6.3f}",
            f"  probability of OOS loss:            {self.probability_of_loss:6.3f}",
        ]
        if self.pbo > 0.5:
            lines.append(
                "  The in-sample winner lands below median out-of-sample more "
                "often than not: this selection procedure is overfitting."
            )
        return "\n".join(lines)


def probability_of_backtest_overfitting(
    returns: np.ndarray,
    *,
    n_splits: int = 16,
    max_combinations: int = 50_000,
) -> PBOResult:
    """PBO via combinatorially symmetric cross-validation.

    ``returns`` is a ``T x N`` matrix: one column per configuration tried, one row
    per period, aligned on the same periods. The rows are cut into ``n_splits``
    contiguous blocks; for each of the ``C(S, S/2)`` ways of splitting those
    blocks into equal in-sample and out-of-sample halves, the best configuration
    in-sample is selected and its *rank* out-of-sample is recorded. PBO is the
    frequency with which that rank falls below the median.

    Sharpes over block unions are computed from per-block sums and sums of
    squares, so all ``C(S, S/2)`` splits cost one matrix multiply rather than a
    reslice each — 12,870 splits over 200 configurations runs in well under a
    second.
    """

    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise DeflationError(
            f"returns must be a 2-D T x N matrix of per-configuration returns, "
            f"got shape {matrix.shape}"
        )
    n_periods, n_configs = matrix.shape
    if n_configs < 2:
        raise DeflationError(
            f"PBO compares configurations against each other; got {n_configs}. "
            "Run more than one configuration, or there is no selection to assess."
        )
    if n_splits < 2 or n_splits % 2 != 0:
        raise DeflationError(f"n_splits must be even and at least 2, got {n_splits}")
    if n_periods < n_splits * 2:
        raise DeflationError(
            f"need at least {n_splits * 2} periods to cut {n_splits} usable blocks, "
            f"got {n_periods}"
        )
    if not np.isfinite(matrix).all():
        raise DeflationError("returns matrix contains non-finite values")

    n_combinations = math.comb(n_splits, n_splits // 2)
    if n_combinations > max_combinations:
        raise DeflationError(
            f"C({n_splits}, {n_splits // 2}) = {n_combinations} splits exceeds "
            f"max_combinations={max_combinations}; lower n_splits (16 is the "
            "conventional choice) or raise the cap deliberately"
        )

    # Per-block sufficient statistics: count, sum, sum of squares per config.
    bounds = np.linspace(0, n_periods, n_splits + 1).astype(int)
    counts = np.empty(n_splits, dtype=float)
    sums = np.empty((n_splits, n_configs), dtype=float)
    sums_sq = np.empty((n_splits, n_configs), dtype=float)
    for b in range(n_splits):
        block = matrix[bounds[b] : bounds[b + 1]]
        counts[b] = float(block.shape[0])
        sums[b] = block.sum(axis=0)
        sums_sq[b] = (block**2).sum(axis=0)

    half = n_splits // 2
    selector = np.zeros((n_combinations, n_splits), dtype=float)
    for i, combo in enumerate(combinations(range(n_splits), half)):
        selector[i, list(combo)] = 1.0
    complement = 1.0 - selector

    is_sharpes_all = _sharpe_from_moments(selector, counts, sums, sums_sq)
    oos_sharpes_all = _sharpe_from_moments(complement, counts, sums, sums_sq)

    chosen = np.argmax(is_sharpes_all, axis=1)
    rows = np.arange(n_combinations)
    # Relative rank of the chosen configuration among OOS performances.
    ranks = np.apply_along_axis(stats.rankdata, 1, oos_sharpes_all)
    chosen_rank = ranks[rows, chosen]
    omega = chosen_rank / (n_configs + 1.0)
    logits = np.log(omega / (1.0 - omega))
    pbo = float(np.count_nonzero(logits <= 0.0) / n_combinations)

    return PBOResult(
        pbo=pbo,
        n_configurations=n_configs,
        n_splits=n_splits,
        n_combinations=n_combinations,
        logits=logits,
        is_sharpes=is_sharpes_all[rows, chosen],
        oos_sharpes=oos_sharpes_all[rows, chosen],
    )


def _sharpe_from_moments(
    selector: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray,
    sums_sq: np.ndarray,
) -> np.ndarray:
    """Per-period Sharpe of each block-union implied by ``selector``.

    ``selector`` is ``C x S`` of 0/1. Pooling blocks by their sufficient
    statistics gives exactly the Sharpe of the concatenated series, without
    materialising it: ``var = (Σx² - n·mean²) / (n-1)``.
    """

    n = selector @ counts
    total = selector @ sums
    total_sq = selector @ sums_sq
    mean = total / n[:, None]
    var = (total_sq - n[:, None] * mean**2) / (n[:, None] - 1.0)
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0.0, mean / sd, 0.0)


def pbo_from_log(
    research_log: ResearchLog,
    returns_by_run_id: Mapping[str, Sequence[float] | np.ndarray],
    *,
    strategy_name: str | None = None,
    n_splits: int = 16,
) -> PBOResult:
    """PBO over the logged configurations of a strategy.

    The set of configurations is taken from ``research_log``, not from whatever
    the caller happened to pass: every completed run must appear in
    ``returns_by_run_id``, and a missing one raises. Computing PBO over a
    hand-picked subset of the trials would measure the overfitting of a smaller,
    imaginary search.
    """

    entries = research_log.entries(strategy_name=strategy_name)
    if not entries:
        raise ResearchLogError(
            f"no completed runs{f' of {strategy_name!r}' if strategy_name else ''} "
            "in research_log; there is nothing to assess for overfitting"
        )
    missing = [e.run_id for e in entries if e.run_id not in returns_by_run_id]
    if missing:
        raise DeflationError(
            f"{len(missing)} of {len(entries)} logged runs have no return series "
            f"(first missing: {missing[0]}). PBO must be computed over every trial "
            "in the log — a subset measures a search that never happened."
        )

    series = [
        np.asarray(returns_by_run_id[e.run_id], dtype=float).ravel() for e in entries
    ]
    lengths = {s.size for s in series}
    if len(lengths) != 1:
        raise DeflationError(
            f"CSCV requires all configurations to cover the same periods; got "
            f"{len(lengths)} distinct series lengths {sorted(lengths)}"
        )
    matrix = np.column_stack(series)
    return probability_of_backtest_overfitting(matrix, n_splits=n_splits)
