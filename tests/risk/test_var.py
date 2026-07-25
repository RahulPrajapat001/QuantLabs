"""VaR and Expected Shortfall: three estimators, and where they disagree."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantlab.risk.var import (
    RiskError,
    cornish_fisher_var,
    expected_shortfall,
    historical_var,
    parametric_var,
    value_at_risk,
    var_comparison,
)

from .conftest import fat_tailed_returns, normal_returns

# --- Against known analytical values -----------------------------------------


def test_parametric_var_matches_the_closed_form() -> None:
    """Gaussian VaR is ``-(μ + z·σ)`` — checkable exactly."""

    returns = normal_returns(n=2000, mean=0.0005, sd=0.010)
    result = parametric_var(returns, confidence=0.99)

    z = float(stats.norm.ppf(0.01))
    expected = -(returns.mean() + z * returns.std(ddof=1))
    assert result.var == pytest.approx(expected, abs=1e-12)


def test_parametric_es_matches_the_gaussian_closed_form() -> None:
    """Gaussian ES is ``-(μ - σ·φ(z)/(1-c))``."""

    returns = normal_returns(n=2000, mean=0.0, sd=0.010)
    result = parametric_var(returns, confidence=0.99)

    z = float(stats.norm.ppf(0.01))
    expected = -(returns.mean() - returns.std(ddof=1) * float(stats.norm.pdf(z)) / 0.01)
    assert result.expected_shortfall == pytest.approx(expected, abs=1e-12)


def test_historical_var_is_the_empirical_quantile() -> None:
    returns = normal_returns(n=1000)
    result = historical_var(returns, confidence=0.95)
    assert result.var == pytest.approx(-float(np.quantile(returns, 0.05)), abs=1e-12)


def test_historical_es_is_the_mean_of_the_tail() -> None:
    returns = normal_returns(n=1000)
    result = historical_var(returns, confidence=0.95)
    quantile = float(np.quantile(returns, 0.05))
    tail = returns[returns <= quantile]
    assert result.expected_shortfall == pytest.approx(-float(tail.mean()), abs=1e-12)


def test_cornish_fisher_reduces_to_gaussian_when_moments_are_normal() -> None:
    """With zero skew and zero excess kurtosis the expansion adds nothing.

    Every correction term carries a factor of S or K, so a normal sample must give
    back the Gaussian answer. This is what pins the expansion's algebra.
    """

    from quantlab.risk.var import _cornish_fisher_quantile

    z = float(stats.norm.ppf(0.01))
    assert _cornish_fisher_quantile(z, 0.0, 0.0) == pytest.approx(z, abs=1e-15)


def test_cornish_fisher_and_parametric_agree_on_gaussian_data() -> None:
    returns = normal_returns(n=5000, sd=0.011)
    gaussian = parametric_var(returns, confidence=0.99)
    expanded = cornish_fisher_var(returns, confidence=0.99)
    # Sample moments are near zero, so the two land within a few percent.
    assert expanded.var == pytest.approx(gaussian.var, rel=0.10)


# --- The disagreement that matters -------------------------------------------


def test_gaussian_var_understates_a_fat_tail() -> None:
    """The reason all three estimators exist.

    On a fat-tailed, negatively skewed series the Gaussian estimate is materially
    below both the empirical quantile and the moment-corrected one. A portfolio sized
    on the Gaussian number is levered on a false assumption.
    """

    returns = fat_tailed_returns(n=3000, df=3)
    gaussian = parametric_var(returns, confidence=0.99)
    empirical = historical_var(returns, confidence=0.99)
    expanded = cornish_fisher_var(returns, confidence=0.99)

    assert gaussian.var < empirical.var
    assert gaussian.var < expanded.var
    assert expanded.excess_kurtosis > 1.0, "fixture should be genuinely fat-tailed"


def test_comparison_quantifies_the_understatement() -> None:
    returns = fat_tailed_returns(n=3000, df=3)
    comparison = var_comparison(returns)
    understatement = comparison.gaussian_understatement(0.99)
    assert understatement > 0.10
    text = comparison.render()
    assert "below the fattest-tailed" in text
    assert "Normality is not a safe simplification" in text


def test_comparison_stays_quiet_on_well_behaved_data() -> None:
    comparison = var_comparison(normal_returns(n=3000))
    assert abs(comparison.gaussian_understatement(0.99)) < 0.15
    assert "not a safe simplification" not in comparison.render()


def test_es_always_exceeds_var() -> None:
    """ES is the mean loss beyond VaR, so it cannot be smaller. True by definition,
    and a useful guard against a sign error."""

    for returns in (normal_returns(), fat_tailed_returns()):
        for method in ("historical", "parametric", "cornish_fisher"):
            result = value_at_risk(returns, confidence=0.99, method=method)  # type: ignore[arg-type]
            assert result.expected_shortfall >= result.var


def test_tail_ratio_is_larger_for_fat_tails() -> None:
    normal_ratio = parametric_var(normal_returns(n=3000), confidence=0.99).tail_ratio
    fat_ratio = historical_var(
        fat_tailed_returns(n=3000, df=3), confidence=0.99
    ).tail_ratio
    assert fat_ratio > normal_ratio
    # A Gaussian at 99% sits near 1.15.
    assert normal_ratio == pytest.approx(1.15, abs=0.05)


# --- Monotonicity and scaling ------------------------------------------------


@pytest.mark.parametrize("method", ["historical", "parametric", "cornish_fisher"])
def test_var_increases_with_confidence(method: str) -> None:
    returns = normal_returns(n=3000)
    at_95 = value_at_risk(returns, confidence=0.95, method=method)  # type: ignore[arg-type]
    at_99 = value_at_risk(returns, confidence=0.99, method=method)  # type: ignore[arg-type]
    assert at_99.var > at_95.var


def test_var_scales_with_the_square_root_of_time() -> None:
    result = parametric_var(normal_returns(), confidence=0.99)
    assert result.scaled(4) == pytest.approx(result.var * 2.0)
    assert result.scaled(1) == pytest.approx(result.var)
    with pytest.raises(RiskError, match="at least 1 day"):
        result.scaled(0)


def test_var_scales_with_volatility() -> None:
    """Doubling the volatility doubles Gaussian VaR, with a zero mean."""

    quiet = normal_returns(n=3000, mean=0.0, sd=0.01, seed=5)
    loud = quiet * 2.0
    assert parametric_var(loud, confidence=0.99).var == pytest.approx(
        parametric_var(quiet, confidence=0.99).var * 2.0, rel=1e-9
    )


# --- Losses are positive -----------------------------------------------------


def test_losses_are_reported_as_positive_numbers() -> None:
    """The sign convention, asserted so it cannot silently flip."""

    result = historical_var(normal_returns(), confidence=0.99)
    assert result.var > 0
    assert result.expected_shortfall > 0
    assert result.var_pct == pytest.approx(result.var * 100)


def test_a_series_that_never_loses_reports_zero_not_negative() -> None:
    always_up = pd.Series([0.001] * 200)
    result = historical_var(always_up, confidence=0.99)
    assert result.var == 0.0
    assert result.expected_shortfall == 0.0


# --- Validation --------------------------------------------------------------


def test_too_few_observations_is_refused() -> None:
    with pytest.raises(RiskError, match="at least 30 finite return observations"):
        historical_var(pd.Series([0.01, -0.02, 0.03]))


def test_confidence_outside_the_valid_range_is_refused() -> None:
    returns = normal_returns()
    for bad in (0.01, 0.5, 1.0, 1.5):
        with pytest.raises(RiskError, match="confidence must lie"):
            historical_var(returns, confidence=bad)


def test_unknown_method_is_refused() -> None:
    with pytest.raises(RiskError, match="unknown VaR method"):
        value_at_risk(normal_returns(), method="magic")  # type: ignore[arg-type]


def test_non_finite_values_are_dropped_not_propagated() -> None:
    returns = normal_returns(n=500).copy()
    returns.iloc[10] = np.nan
    returns.iloc[20] = np.inf
    result = historical_var(returns, confidence=0.95)
    assert result.n_observations == 498
    assert np.isfinite(result.var)


def test_cornish_fisher_refuses_non_monotonic_moments() -> None:
    """An expansion that is not monotonic in confidence is meaningless, and says so.

    Rather than returning a number that would sit below the 95% figure at 99%.
    """

    rng = np.random.default_rng(1)
    # Extreme positive skew and huge kurtosis: a few enormous gains.
    values = rng.normal(0, 0.005, 500)
    values[:5] = 0.9
    with pytest.raises(RiskError, match="not monotonic"):
        cornish_fisher_var(pd.Series(values), confidence=0.99)


def test_comparison_omits_an_invalid_cornish_fisher_rather_than_substituting() -> None:
    """A silently downgraded estimate is worse than a missing row."""

    rng = np.random.default_rng(2)
    values = rng.normal(0, 0.005, 500)
    values[:5] = 0.9
    comparison = var_comparison(pd.Series(values), confidences=(0.99,))
    methods = {r.method for r in comparison.results}
    assert "historical" in methods
    assert "parametric" in methods
    assert "cornish_fisher" not in methods


# --- Convenience -------------------------------------------------------------


def test_expected_shortfall_helper_matches_the_full_result() -> None:
    returns = normal_returns()
    assert expected_shortfall(returns, confidence=0.99) == pytest.approx(
        historical_var(returns, confidence=0.99).expected_shortfall
    )


def test_comparison_pivot_has_a_row_per_confidence(normal) -> None:
    comparison = var_comparison(normal)
    pivot = comparison.pivot("var")
    assert list(pivot.index) == [0.95, 0.99, 0.995]
    assert "historical" in pivot.columns
    frame = comparison.frame()
    assert "es_over_var" in frame.columns


def test_result_carries_the_moments_that_explain_the_difference(fat_tailed) -> None:
    result = historical_var(fat_tailed, confidence=0.99)
    assert result.skew < 0, "the fixture is negatively skewed"
    assert result.excess_kurtosis > 0
    assert result.n_observations == len(fat_tailed)
    assert "ES/VaR" in result.render()
