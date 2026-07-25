"""Covariance estimation, with Ledoit-Wolf shrinkage (PRD Module 10).

**Sample covariance is unstable when assets outnumber observations, and reviewers
check for this.** The failure is not subtle. With ``N`` assets and ``T``
observations, the sample covariance matrix has ``N(N+1)/2`` parameters estimated from
``NT`` numbers: 100 securities over 60 days means 5,050 parameters from 6,000
observations. The matrix is near-singular, its smallest eigenvalues are mostly
estimation error, and a mean-variance optimiser will find them — it inverts the
matrix, so the noisiest directions get the largest weights. The result looks
optimal in-sample and falls apart out of sample. This is the single most common way a
portfolio optimiser produces nonsense while reporting a beautiful backtest.

Ledoit-Wolf shrinkage pulls the sample matrix toward a structured target:

    ``Σ_shrunk = (1-δ)·Σ_sample + δ·F``

where ``F`` is a low-parameter target and ``δ`` is chosen analytically to minimise
expected squared error. The shrinkage intensity is *derived*, not tuned, which is why
it is trustworthy. :attr:`CovarianceEstimate.condition_number` shows what it buys:
typically several orders of magnitude of conditioning.

Three targets are offered because the right structure depends on the universe, and
:attr:`CovarianceEstimate.shrinkage` is always reported so the reader can see how much
of the answer came from the data and how much from the assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

CovarianceMethod = Literal[
    "sample", "ledoit_wolf", "constant_correlation", "single_index"
]

#: Trading days per year.
TRADING_DAYS = 252

#: Above this, the matrix is numerically unreliable to invert and any optimiser
#: output built on it should be treated as noise.
ILL_CONDITIONED = 1e10


class CovarianceError(ValueError):
    """Raised when a covariance matrix cannot be estimated from the data supplied."""


@dataclass(frozen=True, slots=True)
class CovarianceEstimate:
    """A covariance matrix with the diagnostics needed to judge it."""

    matrix: pd.DataFrame
    method: CovarianceMethod
    #: Shrinkage intensity in ``[0, 1]``. 0.0 is the raw sample matrix; 1.0 is the
    #: structured target alone. Around 0.2–0.5 is typical for equity panels.
    shrinkage: float
    n_observations: int
    n_assets: int
    periods_per_year: int = TRADING_DAYS

    @property
    def annualised(self) -> pd.DataFrame:
        return self.matrix * self.periods_per_year

    @property
    def volatilities(self) -> pd.Series:
        """Annualised volatility per asset, from the diagonal."""

        return pd.Series(
            np.sqrt(np.diag(self.matrix) * self.periods_per_year),
            index=self.matrix.index,
            name="volatility",
        )

    @property
    def condition_number(self) -> float:
        """Ratio of largest to smallest eigenvalue.

        The direct measure of how dangerous this matrix is to invert. Above roughly
        1e10 the optimiser's weights are dominated by estimation error.
        """

        eigenvalues = np.linalg.eigvalsh(self.matrix.to_numpy(dtype=float))
        smallest = float(eigenvalues.min())
        largest = float(eigenvalues.max())
        if smallest <= 0:
            return float("inf")
        return largest / smallest

    @property
    def is_positive_definite(self) -> bool:
        eigenvalues = np.linalg.eigvalsh(self.matrix.to_numpy(dtype=float))
        return bool(eigenvalues.min() > 0)

    @property
    def observations_per_asset(self) -> float:
        """``T/N``. Below about 2 the sample matrix should not be used unshrunk."""

        return self.n_observations / self.n_assets if self.n_assets else float("nan")

    @property
    def needs_shrinkage(self) -> bool:
        """Whether this estimate is unsafe to hand to an optimiser.

        Two separate concerns, deliberately not conflated. An ill-conditioned matrix
        is dangerous whatever produced it. Thin data (``T/N < 2``) is dangerous only
        while it remains unaddressed — once shrinkage has been applied the thinness
        is a fact about the panel, not a defect in the estimate, so a shrunk matrix
        with good conditioning answers ``False`` here.
        """

        if self.condition_number > ILL_CONDITIONED:
            return True
        return self.shrinkage == 0.0 and self.observations_per_asset < 2.0

    def correlation(self) -> pd.DataFrame:
        """Implied correlation matrix."""

        sd = np.sqrt(np.diag(self.matrix.to_numpy(dtype=float)))
        outer = np.outer(sd, sd)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(outer > 0, self.matrix.to_numpy(dtype=float) / outer, 0.0)
        return pd.DataFrame(corr, index=self.matrix.index, columns=self.matrix.columns)

    def render(self) -> str:
        lines = [
            f"Covariance estimate ({self.method}, {self.n_assets} assets, "
            f"{self.n_observations} observations)",
            f"  observations per asset: {self.observations_per_asset:8.2f}",
            f"  shrinkage intensity:    {self.shrinkage:8.4f}",
            f"  condition number:       {self.condition_number:8.2e}",
            f"  positive definite:      {'yes' if self.is_positive_definite else 'NO'}",
        ]
        if self.method == "sample" and self.needs_shrinkage:
            lines += [
                "",
                "  WARNING: this sample matrix is too thin or too ill-conditioned to "
                "invert safely. An optimiser will load onto its noisiest "
                "eigenvectors and the weights will not survive out of sample. Use "
                "method='ledoit_wolf'.",
            ]
        return "\n".join(lines)


def _prepare(returns: pd.DataFrame, minimum: int = 20) -> pd.DataFrame:
    if returns.shape[1] < 1:
        raise CovarianceError("no asset columns supplied")
    clean = returns.dropna(axis=1, how="all").dropna()
    if clean.shape[0] < minimum:
        raise CovarianceError(
            f"need at least {minimum} complete observations, got {clean.shape[0]}. "
            "Covariance estimated from fewer rows than assets is not a covariance "
            "matrix, it is noise with a shape."
        )
    if clean.shape[1] < 1:
        raise CovarianceError("no asset has a complete return series")
    return clean


def sample_covariance(
    returns: pd.DataFrame, *, periods_per_year: int = TRADING_DAYS
) -> CovarianceEstimate:
    """Plain sample covariance. Unstable when ``N`` approaches ``T``.

    Provided because it is the right answer for a handful of assets with long
    histories, and because the closed-form tests need an unshrunk matrix to compare
    against. Not the default for anything wide.
    """

    clean = _prepare(returns)
    return CovarianceEstimate(
        matrix=clean.cov(),
        method="sample",
        shrinkage=0.0,
        n_observations=int(clean.shape[0]),
        n_assets=int(clean.shape[1]),
        periods_per_year=periods_per_year,
    )


def ledoit_wolf_covariance(
    returns: pd.DataFrame, *, periods_per_year: int = TRADING_DAYS
) -> CovarianceEstimate:
    """Ledoit-Wolf shrinkage toward scaled identity, with analytic intensity.

    Uses ``sklearn.covariance.LedoitWolf``, which implements the 2004 estimator and
    returns the derived shrinkage coefficient. The target is a scaled identity: equal
    variances, zero correlations. That target is deliberately crude — its job is to be
    well-conditioned, not to be true, and the analytic ``δ`` decides how much of it to
    use.
    """

    from sklearn.covariance import LedoitWolf

    clean = _prepare(returns)
    estimator = LedoitWolf(assume_centered=False).fit(clean.to_numpy(dtype=float))
    matrix = pd.DataFrame(
        estimator.covariance_, index=clean.columns, columns=clean.columns
    )
    return CovarianceEstimate(
        matrix=matrix,
        method="ledoit_wolf",
        shrinkage=float(estimator.shrinkage_),
        n_observations=int(clean.shape[0]),
        n_assets=int(clean.shape[1]),
        periods_per_year=periods_per_year,
    )


def constant_correlation_covariance(
    returns: pd.DataFrame,
    *,
    shrinkage: float | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> CovarianceEstimate:
    """Shrinkage toward the constant-correlation target (Ledoit-Wolf 2003).

    The target keeps each asset's own variance and replaces every pairwise
    correlation with the average one. For equities this is a far better structural
    guess than the identity — the average correlation is genuinely positive — so less
    shrinkage is needed to condition the matrix, and less of the sample's real
    structure is discarded.

    With ``shrinkage=None`` the intensity is estimated by the Ledoit-Wolf asymptotic
    formula against this target.
    """

    clean = _prepare(returns)
    sample = clean.cov().to_numpy(dtype=float)
    n_assets = sample.shape[0]
    if n_assets < 2:
        return CovarianceEstimate(
            matrix=clean.cov(),
            method="constant_correlation",
            shrinkage=0.0,
            n_observations=int(clean.shape[0]),
            n_assets=n_assets,
            periods_per_year=periods_per_year,
        )

    variances = np.diag(sample)
    sd = np.sqrt(variances)
    outer = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(outer > 0, sample / outer, 0.0)
    off_diagonal = correlation[~np.eye(n_assets, dtype=bool)]
    mean_correlation = float(np.nanmean(off_diagonal))

    target = mean_correlation * outer
    np.fill_diagonal(target, variances)

    intensity = (
        _honey_shrinkage(clean.to_numpy(dtype=float), sample, target)
        if shrinkage is None
        else float(shrinkage)
    )
    intensity = min(1.0, max(0.0, intensity))
    shrunk = (1.0 - intensity) * sample + intensity * target
    return CovarianceEstimate(
        matrix=pd.DataFrame(shrunk, index=clean.columns, columns=clean.columns),
        method="constant_correlation",
        shrinkage=intensity,
        n_observations=int(clean.shape[0]),
        n_assets=n_assets,
        periods_per_year=periods_per_year,
    )


def _honey_shrinkage(
    observations: np.ndarray, sample: np.ndarray, target: np.ndarray
) -> float:
    """Optimal shrinkage intensity toward ``target``, Ledoit-Wolf style.

    ``δ* = (π - ρ) / γ / T``, where ``π`` estimates the sum of asymptotic variances
    of the sample entries, ``γ`` the squared distance from sample to target, and ``ρ``
    the covariance between estimation errors. ``ρ`` is approximated by its diagonal
    contribution — the standard simplification, since the full off-diagonal term is
    expensive and contributes little.
    """

    t = observations.shape[0]
    centred = observations - observations.mean(axis=0)

    # π: sum over entries of the asymptotic variance of the sample covariance.
    products = np.einsum("ti,tj->tij", centred, centred)
    pi_matrix = ((products - sample) ** 2).mean(axis=0)
    pi = float(pi_matrix.sum())

    # γ: squared Frobenius distance between sample and target.
    gamma = float(((target - sample) ** 2).sum())
    if gamma <= 0:
        return 0.0

    # ρ: diagonal contribution only.
    rho = float(np.trace(pi_matrix))

    intensity = float((pi - rho) / gamma / t)
    return min(1.0, max(0.0, intensity))


def single_index_covariance(
    returns: pd.DataFrame,
    market: pd.Series,
    *,
    shrinkage: float = 0.5,
    periods_per_year: int = TRADING_DAYS,
) -> CovarianceEstimate:
    """Shrinkage toward a single-index (market model) target.

    ``F = ββ'σ²_m + D``, where ``β`` are market betas, ``σ²_m`` the market variance,
    and ``D`` the diagonal of residual variances. This encodes the economically
    motivated structure — assets covary because they share market exposure — and is
    the target that reviewers expect for equities.
    """

    clean = _prepare(returns)
    aligned = pd.concat(
        [clean, market.rename("_market")], axis=1, join="inner"
    ).dropna()
    if aligned.shape[0] < 20:
        raise CovarianceError(
            f"only {aligned.shape[0]} observations overlap with the market series"
        )
    asset_columns = [c for c in aligned.columns if c != "_market"]
    assets = aligned[asset_columns]
    market_returns = aligned["_market"]

    market_variance = float(market_returns.var(ddof=1))
    if market_variance <= 0:
        raise CovarianceError("the market series has zero variance")

    betas = np.array(
        [float(assets[c].cov(market_returns) / market_variance) for c in asset_columns]
    )
    systematic = np.outer(betas, betas) * market_variance
    sample = assets.cov().to_numpy(dtype=float)
    residual_variance = np.maximum(0.0, np.diag(sample) - np.diag(systematic))
    target = systematic.copy()
    np.fill_diagonal(target, np.diag(systematic) + residual_variance)

    intensity = min(1.0, max(0.0, float(shrinkage)))
    shrunk = (1.0 - intensity) * sample + intensity * target
    return CovarianceEstimate(
        matrix=pd.DataFrame(shrunk, index=asset_columns, columns=asset_columns),
        method="single_index",
        shrinkage=intensity,
        n_observations=int(assets.shape[0]),
        n_assets=len(asset_columns),
        periods_per_year=periods_per_year,
    )


def estimate_covariance(
    returns: pd.DataFrame,
    *,
    method: CovarianceMethod = "ledoit_wolf",
    market: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> CovarianceEstimate:
    """Dispatch to a covariance estimator. Defaults to Ledoit-Wolf.

    The default is shrunk rather than sample on purpose: an optimiser handed a
    near-singular matrix produces confident nonsense, and the person running it
    usually cannot tell.
    """

    if method == "sample":
        return sample_covariance(returns, periods_per_year=periods_per_year)
    if method == "ledoit_wolf":
        return ledoit_wolf_covariance(returns, periods_per_year=periods_per_year)
    if method == "constant_correlation":
        return constant_correlation_covariance(
            returns, periods_per_year=periods_per_year
        )
    if method == "single_index":
        if market is None:
            raise CovarianceError(
                "the single-index target needs a market return series; pass market="
            )
        return single_index_covariance(
            returns, market, periods_per_year=periods_per_year
        )
    raise CovarianceError(
        f"unknown covariance method {method!r}; use sample | ledoit_wolf | "
        "constant_correlation | single_index"
    )
