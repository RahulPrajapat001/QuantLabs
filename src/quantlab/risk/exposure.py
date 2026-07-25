"""Rolling beta, alpha, volatility, and factor exposure (PRD Module 9).

Everything here is **rolling**, not full-sample, because a single beta over ten
years describes a portfolio that no longer exists. A strategy whose beta drifts from
0.3 to 1.1 has quietly become a different strategy, and a static regression reports
0.7 and hides it.

Two choices worth stating:

* **Beta and alpha come from one regression per window**, on excess returns over the
  risk-free rate. Alpha is annualised so it reads as a return; beta is
  dimensionless.
* **Factor exposures use a multivariate regression with Newey-West standard errors.**
  Factor returns are autocorrelated and heteroskedastic; ordinary OLS standard errors
  on them are too small, which turns an insignificant exposure into a significant one.
  That correction is the difference between "this strategy is long momentum" and "this
  strategy might be long momentum".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from quantlab.risk.var import TRADING_DAYS, RiskError

#: Minimum observations in a rolling window before a statistic is reported. Below
#: this, a beta is noise with a decimal point.
MIN_WINDOW = 20


@dataclass(frozen=True, slots=True)
class RollingExposure:
    """Rolling beta, alpha, correlation and R² against a benchmark."""

    beta: pd.Series
    #: Annualised alpha, so it reads as a return rather than a daily intercept.
    alpha: pd.Series
    correlation: pd.Series
    r_squared: pd.Series
    window: int
    periods_per_year: int = TRADING_DAYS

    @property
    def latest_beta(self) -> float:
        return (
            float(self.beta.dropna().iloc[-1])
            if self.beta.notna().any()
            else float("nan")
        )

    @property
    def latest_alpha(self) -> float:
        return (
            float(self.alpha.dropna().iloc[-1])
            if self.alpha.notna().any()
            else float("nan")
        )

    @property
    def beta_range(self) -> tuple[float, float]:
        """Minimum and maximum rolling beta — the drift a static beta would hide."""

        clean = self.beta.dropna()
        if clean.empty:
            return (float("nan"), float("nan"))
        return (float(clean.min()), float(clean.max()))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "beta": self.beta,
                "alpha": self.alpha,
                "correlation": self.correlation,
                "r_squared": self.r_squared,
            }
        )

    def render(self) -> str:
        low, high = self.beta_range
        lines = [
            f"Rolling exposure ({self.window}-day window)",
            f"  latest beta:      {self.latest_beta:7.3f}",
            f"  beta range:       {low:7.3f} to {high:7.3f}",
            f"  latest alpha:     {self.latest_alpha:7.2%} annualised",
        ]
        if np.isfinite(low) and np.isfinite(high) and high - low > 0.5:
            lines.append(
                "  Beta moved by more than 0.5 over the sample: a single "
                "full-sample beta would describe none of these periods."
            )
        return "\n".join(lines)


def rolling_beta_alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    window: int = 60,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> RollingExposure:
    """Rolling beta and annualised alpha of ``returns`` against ``benchmark``.

    Both series are aligned on their common dates first — a mismatched index would
    otherwise silently regress a strategy against a shifted benchmark, which
    produces a plausible-looking and completely wrong beta.
    """

    if window < MIN_WINDOW:
        raise RiskError(
            f"window must be at least {MIN_WINDOW} observations, got {window}"
        )
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < window:
        raise RiskError(
            f"only {len(aligned)} overlapping observations between the strategy and "
            f"the benchmark, need at least {window}. Check that both series are "
            "indexed by date."
        )

    daily_rf = risk_free_rate / periods_per_year
    strategy_excess = aligned["strategy"] - daily_rf
    benchmark_excess = aligned["benchmark"] - daily_rf

    covariance = strategy_excess.rolling(window).cov(benchmark_excess)
    variance = benchmark_excess.rolling(window).var()
    beta = covariance / variance.replace(0.0, np.nan)

    strategy_mean = strategy_excess.rolling(window).mean()
    benchmark_mean = benchmark_excess.rolling(window).mean()
    alpha = (strategy_mean - beta * benchmark_mean) * periods_per_year

    correlation = strategy_excess.rolling(window).corr(benchmark_excess)
    return RollingExposure(
        beta=beta.rename("beta"),
        alpha=alpha.rename("alpha"),
        correlation=correlation.rename("correlation"),
        r_squared=(correlation**2).rename("r_squared"),
        window=window,
        periods_per_year=periods_per_year,
    )


def rolling_volatility(
    returns: pd.Series,
    *,
    window: int = 60,
    periods_per_year: int = TRADING_DAYS,
    annualise: bool = True,
) -> pd.Series:
    """Rolling annualised volatility.

    Uses the sample standard deviation (``ddof=1``). Annualising by ``√252``
    assumes independence; volatility clusters, so a crisis window's annualised
    figure overstates the year ahead and a calm window's understates it. Reported
    as-is because the alternative is a model, and a model belongs in a module that
    says so.
    """

    if window < MIN_WINDOW:
        raise RiskError(f"window must be at least {MIN_WINDOW}, got {window}")
    volatility = returns.rolling(window).std(ddof=1)
    if annualise:
        volatility = volatility * np.sqrt(periods_per_year)
    return volatility.rename(f"volatility_{window}d")


@dataclass(frozen=True, slots=True)
class FactorLoading:
    """One factor's exposure, with an honest standard error."""

    factor: str
    beta: float
    std_error: float
    t_stat: float
    p_value: float

    @property
    def significant(self) -> bool:
        """Whether the exposure clears |t| ≥ 2 on Newey-West standard errors."""

        return abs(self.t_stat) >= 2.0

    def render(self) -> str:
        marker = "*" if self.significant else " "
        return (
            f"  {self.factor:20s} {self.beta:8.4f}  (t={self.t_stat:6.2f}, "
            f"p={self.p_value:.4f}){marker}"
        )


@dataclass(frozen=True, slots=True)
class FactorExposure:
    """A portfolio's exposures to a set of factor return series.

    ``alpha`` is what the factors do **not** explain, annualised. A strategy whose
    alpha vanishes once momentum and size are controlled for is a repackaging of
    those factors, not a new source of return — and that is a finding worth
    reporting rather than burying.
    """

    loadings: tuple[FactorLoading, ...]
    alpha: float
    alpha_t_stat: float
    alpha_p_value: float
    r_squared: float
    adjusted_r_squared: float
    n_observations: int
    #: Newey-West lag length used for the standard errors.
    newey_west_lags: int

    def betas(self) -> dict[str, float]:
        return {loading.factor: loading.beta for loading in self.loadings}

    def significant_factors(self) -> list[str]:
        return [loading.factor for loading in self.loadings if loading.significant]

    @property
    def perfectly_explained(self) -> bool:
        """Whether the factors explain the returns exactly, leaving no residual.

        Happens when the portfolio *is* a linear combination of the factors — a
        replication, or a factor regressed on itself. The regression then has
        residuals near machine epsilon, so every standard error collapses toward zero
        and every t-statistic explodes. The resulting "significance" is arithmetic
        noise, not evidence.
        """

        return self.r_squared > 1.0 - 1e-9

    @property
    def alpha_significant(self) -> bool:
        """Whether alpha clears |t| ≥ 2 on Newey-West standard errors.

        Always ``False`` for a degenerate perfect fit: an alpha of 1e-19 with a
        standard error of 1e-16 is not a finding, and reporting it as one would be
        exactly the kind of spurious claim this module exists to prevent.
        """

        if self.perfectly_explained:
            return False
        return abs(self.alpha_t_stat) >= 2.0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "factor": loading.factor,
                    "beta": loading.beta,
                    "std_error": loading.std_error,
                    "t_stat": loading.t_stat,
                    "p_value": loading.p_value,
                    "significant": loading.significant,
                }
                for loading in self.loadings
            ]
        )

    def render(self) -> str:
        lines = [
            f"Factor exposure ({self.n_observations} observations, "
            f"Newey-West lags={self.newey_west_lags})",
            "",
        ]
        lines.extend(loading.render() for loading in self.loadings)
        lines += [
            "",
            f"  {'alpha (annualised)':20s} {self.alpha:8.2%}  "
            f"(t={self.alpha_t_stat:6.2f}, p={self.alpha_p_value:.4f})"
            f"{'*' if self.alpha_significant else ' '}",
            f"  R² {self.r_squared:.4f}  adjusted R² {self.adjusted_r_squared:.4f}",
        ]
        if not self.alpha_significant:
            lines.append("")
            lines.append(
                "  Alpha is not distinguishable from zero once these factors are "
                "controlled for: the return is factor exposure, not skill."
            )
        if self.perfectly_explained:
            lines.append(
                "  The factors explain the returns exactly: this portfolio is a "
                "replication of them. Standard errors are degenerate here, so no "
                "t-statistic above should be read as evidence of anything."
            )
        elif self.r_squared > 0.9:
            lines.append(
                f"  R² of {self.r_squared:.2f} means the factors explain almost all "
                "of the variation — this portfolio is close to a factor combination."
            )
        return "\n".join(lines)


def factor_exposure(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    newey_west_lags: int | None = None,
) -> FactorExposure:
    """Regress portfolio returns on factor returns, with Newey-West errors.

    ``factors`` is a frame of factor return series indexed by date, one column per
    factor. Alignment is on the intersection of dates.

    Standard errors are Newey-West (HAC) corrected. The default lag length follows
    the usual ``⌊4·(T/100)^(2/9)⌋`` rule. Using ordinary OLS errors here would
    overstate every t-statistic, because factor returns are autocorrelated.
    """

    aligned = pd.concat(
        [returns.rename("_portfolio"), factors], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise RiskError(
            "no overlapping dates between the return series and the factor frame"
        )
    n = len(aligned)
    factor_names = [c for c in aligned.columns if c != "_portfolio"]
    if not factor_names:
        raise RiskError("no factor columns supplied")
    if n <= len(factor_names) + 1:
        raise RiskError(
            f"{n} observations cannot identify {len(factor_names)} factor loadings "
            "plus an intercept"
        )

    daily_rf = risk_free_rate / periods_per_year
    y = (aligned["_portfolio"] - daily_rf).to_numpy(dtype=float)
    x = sm.add_constant(aligned[factor_names].to_numpy(dtype=float))

    lags = newey_west_lags
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, lags)

    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    loadings = tuple(
        FactorLoading(
            factor=name,
            beta=float(model.params[i + 1]),
            std_error=float(model.bse[i + 1]),
            t_stat=float(model.tvalues[i + 1]),
            p_value=float(model.pvalues[i + 1]),
        )
        for i, name in enumerate(factor_names)
    )
    return FactorExposure(
        loadings=loadings,
        alpha=float(model.params[0]) * periods_per_year,
        alpha_t_stat=float(model.tvalues[0]),
        alpha_p_value=float(model.pvalues[0]),
        r_squared=float(model.rsquared),
        adjusted_r_squared=float(model.rsquared_adj),
        n_observations=n,
        newey_west_lags=lags,
    )


@dataclass(frozen=True, slots=True)
class RollingFactorExposure:
    """Factor betas through time — the drift a single regression cannot show."""

    betas: pd.DataFrame
    r_squared: pd.Series
    window: int
    factors: tuple[str, ...] = field(default_factory=tuple)

    def drift(self) -> pd.Series:
        """Range of each factor beta over the sample."""

        clean = self.betas.dropna(how="all")
        if clean.empty:
            return pd.Series(dtype=float)
        return (clean.max() - clean.min()).rename("beta_range")


def rolling_factor_exposure(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    window: int = 126,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> RollingFactorExposure:
    """Rolling multivariate factor betas.

    Ordinary OLS per window (no HAC): the standard errors are not reported here, and
    computing HAC errors on several hundred overlapping windows costs a great deal
    for numbers nobody reads. Use :func:`factor_exposure` when the significance of
    an exposure matters.
    """

    aligned = pd.concat(
        [returns.rename("_portfolio"), factors], axis=1, join="inner"
    ).dropna()
    factor_names = [c for c in aligned.columns if c != "_portfolio"]
    if len(aligned) < window:
        raise RiskError(
            f"only {len(aligned)} aligned observations, need at least {window}"
        )

    daily_rf = risk_free_rate / periods_per_year
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    r_squared: dict[pd.Timestamp, float] = {}
    values = aligned.to_numpy(dtype=float)

    for end in range(window, len(aligned) + 1):
        chunk = values[end - window : end]
        y = chunk[:, 0] - daily_rf
        x = sm.add_constant(chunk[:, 1:])
        try:
            fit = sm.OLS(y, x).fit()
        except (
            np.linalg.LinAlgError,
            ValueError,
        ):  # pragma: no cover - singular window
            continue
        index = aligned.index[end - 1]
        rows[index] = {
            name: float(fit.params[i + 1]) for i, name in enumerate(factor_names)
        }
        r_squared[index] = float(fit.rsquared)

    betas = pd.DataFrame.from_dict(rows, orient="index").reindex(aligned.index)
    return RollingFactorExposure(
        betas=betas,
        r_squared=pd.Series(r_squared).reindex(aligned.index).rename("r_squared"),
        window=window,
        factors=tuple(factor_names),
    )
