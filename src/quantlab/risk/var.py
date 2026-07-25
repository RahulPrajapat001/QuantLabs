"""Value at Risk and Expected Shortfall (PRD Module 9).

Three estimators, because they disagree and the disagreement is the information:

* **Historical** — the empirical quantile. Makes no distributional assumption, so
  it captures whatever the sample actually did, but it cannot produce a loss larger
  than the worst one observed. With two years of daily data the 99% VaR is
  interpolated from about five observations.
* **Parametric (Gaussian)** — mean plus z-score times volatility. Closed-form and
  stable, and wrong in the direction that matters: equity returns have fat tails
  and negative skew, so a Gaussian VaR understates tail loss precisely when it is
  being relied upon.
* **Cornish-Fisher** — expands the Gaussian quantile using the sample's skew and
  excess kurtosis. Recovers most of the gap between the other two without needing
  the sample to contain a tail event.

**Expected Shortfall is reported alongside VaR, always.** VaR says "you will lose
at least this much on the worst 1% of days" and is silent about how much worse it
gets. ES answers that, and it is the coherent risk measure — VaR is not subadditive,
so a VaR-optimised portfolio can look safer than the sum of its parts.

Sign convention: losses are reported as **positive** numbers. A 99% VaR of 0.031
means a 3.1% loss. The alternative (negative for loss) reads naturally in a return
series and then inverts confusingly in every comparison and report, so the
conversion happens once, here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

#: Confidence levels reported by default. 95% for routine monitoring, 99% for tail
#: work, 99.5% because a regulator will ask.
DEFAULT_CONFIDENCES: tuple[float, ...] = (0.95, 0.99, 0.995)

#: Trading days per year, for annualising.
TRADING_DAYS = 252

VaRMethod = Literal["historical", "parametric", "cornish_fisher"]


class RiskError(ValueError):
    """Raised when a risk estimate cannot be computed from the data supplied."""


@dataclass(frozen=True, slots=True)
class VaRResult:
    """A VaR and ES pair at one confidence level, with its provenance.

    Both are **positive** magnitudes of loss, expressed as a fraction of the
    portfolio value over one period.
    """

    method: VaRMethod
    confidence: float
    var: float
    expected_shortfall: float
    n_observations: int
    #: Sample moments the estimate was built from — the reason the three methods
    #: differ, kept attached so the difference is explicable.
    mean: float
    volatility: float
    skew: float
    #: Excess kurtosis (0.0 for a normal), the convention Cornish-Fisher uses.
    excess_kurtosis: float

    @property
    def var_pct(self) -> float:
        return self.var * 100.0

    @property
    def es_pct(self) -> float:
        return self.expected_shortfall * 100.0

    @property
    def tail_ratio(self) -> float:
        """ES divided by VaR. Above ~1.2 signals a fat left tail.

        For a normal distribution at 99% this sits near 1.15; materially higher
        means the losses beyond VaR are much worse than VaR suggests.
        """

        return self.expected_shortfall / self.var if self.var > 0 else float("nan")

    def scaled(self, days: int) -> float:
        """VaR scaled to a ``days``-day horizon by the square root of time.

        The square-root rule assumes independent, identically distributed returns.
        Both assumptions fail in a crisis — volatility clusters and returns
        autocorrelate — so a 10-day VaR obtained this way is optimistic. Use the
        stress module for horizons that matter.
        """

        if days < 1:
            raise RiskError(f"horizon must be at least 1 day, got {days}")
        return self.var * math.sqrt(days)

    def render(self) -> str:
        return (
            f"{self.method:15s} {self.confidence:.1%}  "
            f"VaR {self.var_pct:6.2f}%  ES {self.es_pct:6.2f}%  "
            f"(ES/VaR {self.tail_ratio:.2f}, n={self.n_observations})"
        )


def _clean(
    returns: pd.Series | np.ndarray, *, minimum: int = 30
) -> npt.NDArray[np.float64]:
    raw = np.asarray(returns, dtype=np.float64).ravel()
    values: npt.NDArray[np.float64] = raw[np.isfinite(raw)]
    if values.size < minimum:
        raise RiskError(
            f"need at least {minimum} finite return observations for a risk "
            f"estimate, got {values.size}. A VaR computed from a handful of "
            "observations is a number, not an estimate."
        )
    return values


def _moments(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(mean, sd, skew, excess kurtosis)`` — sd sample, moments population."""

    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd == 0.0:
        return mean, 0.0, 0.0, 0.0
    z = (values - mean) / values.std(ddof=0)
    skew = float((z**3).mean())
    excess_kurtosis = float((z**4).mean()) - 3.0
    return mean, sd, skew, excess_kurtosis


def historical_var(
    returns: pd.Series | np.ndarray, *, confidence: float = 0.99
) -> VaRResult:
    """Empirical-quantile VaR and the mean loss beyond it.

    No distributional assumption. Its limitation is structural: it cannot report a
    loss worse than the worst observation, so on a sample containing no crisis it
    will say the tail is thin.
    """

    _check_confidence(confidence)
    values = _clean(returns)
    quantile = float(np.quantile(values, 1.0 - confidence))
    tail = values[values <= quantile]
    # The mean of the tail; if the quantile lands on the minimum the tail is one
    # observation, which is honest but worth knowing about.
    es = float(tail.mean()) if tail.size else quantile
    mean, sd, skew, excess = _moments(values)
    return VaRResult(
        method="historical",
        confidence=confidence,
        var=max(0.0, -quantile),
        expected_shortfall=max(0.0, -es),
        n_observations=int(values.size),
        mean=mean,
        volatility=sd,
        skew=skew,
        excess_kurtosis=excess,
    )


def parametric_var(
    returns: pd.Series | np.ndarray, *, confidence: float = 0.99
) -> VaRResult:
    """Gaussian VaR: ``-(μ + z·σ)``, with the closed-form Gaussian ES.

    Included because it is the industry default and because comparing it to the
    other two quantifies how much the normality assumption is hiding. On Indian
    equity indices the gap at 99% is routinely 20–40%.
    """

    _check_confidence(confidence)
    values = _clean(returns)
    mean, sd, skew, excess = _moments(values)
    z = float(stats.norm.ppf(1.0 - confidence))
    var = -(mean + z * sd)
    # Gaussian ES: μ - σ·φ(z)/(1-c), where φ is the standard normal density.
    density = float(stats.norm.pdf(z))
    es = -(mean - sd * density / (1.0 - confidence))
    return VaRResult(
        method="parametric",
        confidence=confidence,
        var=max(0.0, var),
        expected_shortfall=max(0.0, es),
        n_observations=int(values.size),
        mean=mean,
        volatility=sd,
        skew=skew,
        excess_kurtosis=excess,
    )


def cornish_fisher_var(
    returns: pd.Series | np.ndarray, *, confidence: float = 0.99
) -> VaRResult:
    """VaR from a Cornish-Fisher expansion of the Gaussian quantile.

    The expansion adjusts the normal quantile ``z`` for the sample's skew ``S`` and
    excess kurtosis ``K``::

        z_cf = z + (z² - 1)·S/6 + (z³ - 3z)·K/24 - (2z³ - 5z)·S²/36

    This recovers most of the fat-tail correction without requiring the sample to
    contain a tail event, which is exactly where historical VaR is weakest.

    It is an expansion, not a distribution: for large ``|S|`` or ``K`` the adjusted
    quantile can stop being monotonic in the confidence level, at which point the
    number is meaningless. That case is detected and raised rather than returned.
    """

    _check_confidence(confidence)
    values = _clean(returns)
    mean, sd, skew, excess = _moments(values)
    z = float(stats.norm.ppf(1.0 - confidence))
    z_cf = _cornish_fisher_quantile(z, skew, excess)

    # Monotonicity check: a valid expansion must give a more extreme quantile at a
    # higher confidence level. If it does not, the moments are too large for the
    # expansion to be trusted here.
    probe = float(stats.norm.ppf(1.0 - min(0.9999, confidence + 0.004)))
    if _cornish_fisher_quantile(probe, skew, excess) > z_cf:
        raise RiskError(
            f"the Cornish-Fisher expansion is not monotonic at {confidence:.1%} "
            f"for skew={skew:.2f}, excess kurtosis={excess:.2f}. The sample's "
            "moments are too extreme for the expansion to be valid; use historical "
            "VaR, or a longer sample."
        )

    var = -(mean + z_cf * sd)
    # ES by integrating the expansion over the tail. Numerical, because the
    # expansion has no closed-form tail expectation.
    grid = np.linspace(1e-6, 1.0 - confidence, 512)
    quantiles = np.array(
        [
            mean + _cornish_fisher_quantile(float(stats.norm.ppf(p)), skew, excess) * sd
            for p in grid
        ]
    )
    es = -float(quantiles.mean())
    return VaRResult(
        method="cornish_fisher",
        confidence=confidence,
        var=max(0.0, var),
        expected_shortfall=max(0.0, es),
        n_observations=int(values.size),
        mean=mean,
        volatility=sd,
        skew=skew,
        excess_kurtosis=excess,
    )


def _cornish_fisher_quantile(z: float, skew: float, excess_kurtosis: float) -> float:
    """The Cornish-Fisher adjusted standard-normal quantile."""

    return (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )


def expected_shortfall(
    returns: pd.Series | np.ndarray,
    *,
    confidence: float = 0.99,
    method: VaRMethod = "historical",
) -> float:
    """Expected Shortfall alone, when only the tail mean is wanted."""

    return value_at_risk(
        returns, confidence=confidence, method=method
    ).expected_shortfall


def value_at_risk(
    returns: pd.Series | np.ndarray,
    *,
    confidence: float = 0.99,
    method: VaRMethod = "historical",
) -> VaRResult:
    """Dispatch to one of the three estimators."""

    if method == "historical":
        return historical_var(returns, confidence=confidence)
    if method == "parametric":
        return parametric_var(returns, confidence=confidence)
    if method == "cornish_fisher":
        return cornish_fisher_var(returns, confidence=confidence)
    raise RiskError(
        f"unknown VaR method {method!r}; use historical | parametric | cornish_fisher"
    )


@dataclass(frozen=True, slots=True)
class VaRComparison:
    """All three estimators at several confidence levels, side by side.

    The point of the table is the disagreement. When parametric VaR is materially
    below the other two, the portfolio's tail risk is worse than a normal
    approximation admits, and any downstream sizing that used the Gaussian number
    is too aggressive.
    """

    results: tuple[VaRResult, ...]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "method": r.method,
                    "confidence": r.confidence,
                    "var": r.var,
                    "expected_shortfall": r.expected_shortfall,
                    "es_over_var": r.tail_ratio,
                }
                for r in self.results
            ]
        )

    def pivot(self, field: str = "var") -> pd.DataFrame:
        return self.frame().pivot(index="confidence", columns="method", values=field)

    def gaussian_understatement(self, confidence: float = 0.99) -> float:
        """How much lower Gaussian VaR is than the fattest-tailed estimate.

        Reported as a fraction: 0.30 means the Gaussian number is 30% below the
        worst of the other two. This is the number that says whether normality is
        a safe simplification for this portfolio.
        """

        at_level = [r for r in self.results if r.confidence == confidence]
        parametric = next((r for r in at_level if r.method == "parametric"), None)
        others = [r.var for r in at_level if r.method != "parametric"]
        if parametric is None or not others or parametric.var <= 0:
            return float("nan")
        worst = max(others)
        return (worst - parametric.var) / worst if worst > 0 else float("nan")

    def render(self) -> str:
        lines = ["Value at Risk / Expected Shortfall (positive = loss)", ""]
        lines.extend(f"  {r.render()}" for r in self.results)
        understatement = self.gaussian_understatement()
        if np.isfinite(understatement):
            lines.append("")
            lines.append(
                f"  Gaussian VaR sits {understatement:.1%} below the fattest-tailed "
                "estimate at 99%."
            )
            if understatement > 0.15:
                lines.append(
                    "  Normality is not a safe simplification for this return "
                    "series; size on the historical or Cornish-Fisher figure."
                )
        return "\n".join(lines)


def var_comparison(
    returns: pd.Series | np.ndarray,
    *,
    confidences: tuple[float, ...] = DEFAULT_CONFIDENCES,
) -> VaRComparison:
    """Every estimator at every confidence level.

    A Cornish-Fisher estimate that fails its own validity check is omitted rather
    than substituted with a Gaussian one — a silently downgraded estimate is worse
    than a missing row.
    """

    results: list[VaRResult] = []
    for confidence in confidences:
        results.append(historical_var(returns, confidence=confidence))
        results.append(parametric_var(returns, confidence=confidence))
        try:
            results.append(cornish_fisher_var(returns, confidence=confidence))
        except RiskError:
            continue
    return VaRComparison(results=tuple(results))


def _check_confidence(confidence: float) -> None:
    if not 0.5 < confidence < 1.0:
        raise RiskError(
            f"confidence must lie in (0.5, 1), got {confidence}. Pass 0.99 for a "
            "99% level, not 0.01."
        )
