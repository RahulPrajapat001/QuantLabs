"""Rolling beta/alpha, rolling volatility, and factor exposure regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.risk.exposure import (
    factor_exposure,
    rolling_beta_alpha,
    rolling_factor_exposure,
    rolling_volatility,
)
from quantlab.risk.var import RiskError

from .conftest import normal_returns


def make_pair(
    *,
    beta: float = 1.2,
    alpha_daily: float = 0.0002,
    n: int = 500,
    seed: int = 21,
    noise_sd: float = 0.004,
) -> tuple[pd.Series, pd.Series]:
    """A strategy built with a **known** beta and alpha against a benchmark.

    ``noise_sd`` matters for test power. The standard error of an alpha estimate is
    roughly ``noise_sd/√window``, so at the default 0.004 a 250-day window gives an
    alpha standard error near 2.5e-4 — larger than the 2e-4 signal itself. Tests that
    assert alpha is *recovered* therefore pass a smaller ``noise_sd``, so the quantity
    they are checking is actually identifiable. Beta is estimated far more precisely
    and needs no such care.
    """

    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2016-01-04", periods=n)
    benchmark = pd.Series(rng.normal(0.0004, 0.011, n), index=index, name="benchmark")
    noise = pd.Series(rng.normal(0.0, noise_sd, n), index=index)
    strategy = (alpha_daily + beta * benchmark + noise).rename("strategy")
    return strategy, benchmark


# --- Rolling beta and alpha --------------------------------------------------


def test_rolling_beta_recovers_a_known_beta() -> None:
    """Constructed with beta 1.2, so the rolling estimate must centre there."""

    strategy, benchmark = make_pair(beta=1.2, n=800)
    exposure = rolling_beta_alpha(strategy, benchmark, window=120)
    betas = exposure.beta.dropna()
    assert len(betas) > 500
    assert float(betas.mean()) == pytest.approx(1.2, abs=0.05)


def test_rolling_alpha_is_centred_on_a_known_alpha() -> None:
    """A 0.0002/day alpha annualises to ~5%, and the rolling series brackets it.

    The tolerance is wide on purpose. Each window's alpha has a standard error of
    roughly ``σ_ε·√252/√window`` — about 1.3 percentage points here — and the windows
    overlap heavily, so the mean of the rolling series is a far noisier statistic than
    its length suggests. Asserting it to within a fraction of a percent would be
    testing this particular seed, not the estimator.

    Exactness is covered by the per-window OLS comparison below, which checks the
    implementation directly and holds at any noise level.
    """

    strategy, benchmark = make_pair(alpha_daily=0.0002, n=1500, noise_sd=0.0008)
    exposure = rolling_beta_alpha(strategy, benchmark, window=250)
    alphas = exposure.alpha.dropna()

    expected = 0.0002 * 252
    assert float(alphas.mean()) == pytest.approx(expected, abs=0.035)
    # The sign and order of magnitude are what a reader actually relies on.
    assert float(alphas.median()) > 0.0
    assert 0.5 * expected < float(alphas.median()) < 2.0 * expected


def test_rolling_alpha_is_zero_when_there_is_none() -> None:
    """No constructed alpha, so the rolling series must straddle zero."""

    strategy, benchmark = make_pair(alpha_daily=0.0, n=1500, noise_sd=0.0008)
    alphas = rolling_beta_alpha(strategy, benchmark, window=250).alpha.dropna()
    assert float(alphas.mean()) == pytest.approx(0.0, abs=0.035)


def test_rolling_alpha_matches_a_per_window_ols_intercept() -> None:
    """The exact check: the closed-form rolling alpha *is* the OLS intercept.

    ``α = ȳ - β·x̄`` is what OLS with a constant produces by construction, so this
    verifies the implementation rather than the estimator's statistical properties —
    and it holds regardless of how noisy the data is.
    """

    import statsmodels.api as sm

    strategy, benchmark = make_pair(n=400, seed=23)
    window = 120
    exposure = rolling_beta_alpha(strategy, benchmark, window=window)

    for end in (window, 250, 400):
        y = strategy.iloc[end - window : end].to_numpy()
        x = benchmark.iloc[end - window : end].to_numpy()
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        index = strategy.index[end - 1]
        assert exposure.beta.loc[index] == pytest.approx(
            float(fit.params[1]), abs=1e-10
        )
        assert exposure.alpha.loc[index] == pytest.approx(
            float(fit.params[0]) * 252, abs=1e-10
        )


def test_beta_of_a_series_against_itself_is_one() -> None:
    returns = normal_returns(n=400)
    exposure = rolling_beta_alpha(returns, returns, window=100)
    betas = exposure.beta.dropna()
    assert np.allclose(betas, 1.0, atol=1e-9)
    assert np.allclose(exposure.r_squared.dropna(), 1.0, atol=1e-9)


def test_rolling_beta_detects_drift_that_a_static_beta_would_hide() -> None:
    """The reason everything here is rolling.

    A strategy whose beta moves from 0.3 to 1.3 is described by neither number, and a
    full-sample regression reports the average and hides the change.
    """

    rng = np.random.default_rng(31)
    n = 600
    index = pd.bdate_range("2016-01-04", periods=n)
    benchmark = pd.Series(rng.normal(0.0004, 0.011, n), index=index)
    betas = np.concatenate([np.full(n // 2, 0.3), np.full(n - n // 2, 1.3)])
    strategy = pd.Series(
        betas * benchmark.to_numpy() + rng.normal(0, 0.003, n), index=index
    )

    exposure = rolling_beta_alpha(strategy, benchmark, window=100)
    low, high = exposure.beta_range
    assert low == pytest.approx(0.3, abs=0.15)
    assert high == pytest.approx(1.3, abs=0.15)
    assert "Beta moved by more than 0.5" in exposure.render()


def test_misaligned_series_are_intersected_not_silently_shifted() -> None:
    """A shifted index would give a plausible and completely wrong beta."""

    strategy, benchmark = make_pair(n=400)
    shifted = benchmark.iloc[50:]
    exposure = rolling_beta_alpha(strategy, shifted, window=100)
    # Only the overlapping dates are used.
    assert len(exposure.beta.dropna()) <= len(shifted) - 100 + 1


def test_insufficient_overlap_raises() -> None:
    strategy, benchmark = make_pair(n=400)
    with pytest.raises(RiskError, match="overlapping observations"):
        rolling_beta_alpha(strategy.iloc[:50], benchmark.iloc[300:], window=60)


def test_window_below_the_floor_is_refused() -> None:
    strategy, benchmark = make_pair()
    with pytest.raises(RiskError, match="at least 20"):
        rolling_beta_alpha(strategy, benchmark, window=5)


def test_exposure_frame_has_all_four_series() -> None:
    strategy, benchmark = make_pair()
    frame = rolling_beta_alpha(strategy, benchmark, window=100).frame()
    assert list(frame.columns) == ["beta", "alpha", "correlation", "r_squared"]


# --- Rolling volatility ------------------------------------------------------


def test_rolling_volatility_annualises() -> None:
    returns = normal_returns(n=500, sd=0.01)
    volatility = rolling_volatility(returns, window=100)
    # 1% daily annualises to about 15.9%.
    assert float(volatility.dropna().mean()) == pytest.approx(
        0.01 * np.sqrt(252), abs=0.01
    )


def test_rolling_volatility_can_stay_unannualised() -> None:
    returns = normal_returns(n=300, sd=0.01)
    raw = rolling_volatility(returns, window=60, annualise=False)
    assert float(raw.dropna().mean()) == pytest.approx(0.01, abs=0.002)


def test_rolling_volatility_tracks_a_regime_change() -> None:
    rng = np.random.default_rng(41)
    calm = rng.normal(0, 0.005, 300)
    stormy = rng.normal(0, 0.025, 300)
    returns = pd.Series(
        np.concatenate([calm, stormy]), index=pd.bdate_range("2016-01-04", periods=600)
    )
    volatility = rolling_volatility(returns, window=60).dropna()
    assert float(volatility.iloc[50]) < float(volatility.iloc[-1])
    assert float(volatility.iloc[-1]) / float(volatility.iloc[50]) > 3.0


def test_rolling_volatility_rejects_a_tiny_window() -> None:
    with pytest.raises(RiskError, match="at least 20"):
        rolling_volatility(normal_returns(), window=3)


# --- Factor exposure ---------------------------------------------------------


def make_factors(n: int = 600, seed: int = 51) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2016-01-04", periods=n)
    return pd.DataFrame(
        {
            "market": rng.normal(0.0004, 0.011, n),
            "momentum": rng.normal(0.0002, 0.008, n),
            "value": rng.normal(0.0001, 0.007, n),
        },
        index=index,
    )


def test_factor_exposure_recovers_known_loadings() -> None:
    """Built with market 0.9 and momentum 0.5, so those are what must come out."""

    factors = make_factors(n=800)
    rng = np.random.default_rng(61)
    returns = (
        0.9 * factors["market"]
        + 0.5 * factors["momentum"]
        + rng.normal(0.0, 0.003, len(factors))
    ).rename("strategy")

    exposure = factor_exposure(returns, factors)
    betas = exposure.betas()
    assert betas["market"] == pytest.approx(0.9, abs=0.05)
    assert betas["momentum"] == pytest.approx(0.5, abs=0.05)
    assert betas["value"] == pytest.approx(0.0, abs=0.06)
    assert set(exposure.significant_factors()) >= {"market", "momentum"}
    assert "value" not in exposure.significant_factors()


def test_a_mostly_factor_portfolio_has_no_alpha() -> None:
    """The finding that matters: a repackaged factor is not a source of return."""

    factors = make_factors(n=800)
    rng = np.random.default_rng(101)
    # Overwhelmingly factor exposure, with a little genuine residual — a realistic
    # replication rather than an exact one.
    returns = (
        0.7 * factors["market"]
        + 0.3 * factors["momentum"]
        + rng.normal(0.0, 0.0008, len(factors))
    ).rename("strategy")

    exposure = factor_exposure(returns, factors)
    assert exposure.r_squared > 0.9
    assert not exposure.perfectly_explained
    assert not exposure.alpha_significant
    text = exposure.render()
    assert "factor exposure, not skill" in text
    assert "close to a factor combination" in text


def test_an_exact_factor_replication_is_flagged_as_degenerate() -> None:
    """A perfect fit produces meaningless t-statistics, and must not claim otherwise.

    With zero residual, every standard error collapses toward machine epsilon and the
    intercept's t-statistic explodes even though the intercept is ~1e-19. Reporting
    that as a significant alpha would be a wrong claim manufactured by arithmetic.
    """

    factors = make_factors(n=800)
    returns = (0.7 * factors["market"] + 0.3 * factors["momentum"]).rename("strategy")

    exposure = factor_exposure(returns, factors)
    assert exposure.r_squared == pytest.approx(1.0)
    assert exposure.perfectly_explained
    assert not exposure.alpha_significant, "a 1e-19 alpha is not a finding"
    assert abs(exposure.alpha) < 1e-10
    assert "degenerate" in exposure.render()


def test_genuine_alpha_survives_the_regression() -> None:
    factors = make_factors(n=1000)
    rng = np.random.default_rng(71)
    # A real 0.0004/day edge on top of factor exposure.
    returns = (
        0.0004 + 0.8 * factors["market"] + rng.normal(0.0, 0.002, len(factors))
    ).rename("strategy")

    exposure = factor_exposure(returns, factors)
    assert exposure.alpha == pytest.approx(0.0004 * 252, abs=0.03)
    assert exposure.alpha_significant


def test_newey_west_lags_default_to_the_standard_rule() -> None:
    factors = make_factors(n=1000)
    returns = factors["market"].rename("strategy")
    exposure = factor_exposure(returns, factors)
    expected = int(np.floor(4.0 * (len(factors) / 100.0) ** (2.0 / 9.0)))
    assert exposure.newey_west_lags == expected
    assert f"lags={expected}" in exposure.render()


def test_newey_west_widens_errors_relative_to_plain_ols() -> None:
    """The correction's purpose: autocorrelated factors inflate OLS t-statistics.

    With an autocorrelated factor, HAC standard errors must exceed the naive ones,
    which is what stops a spurious exposure looking significant.
    """

    import statsmodels.api as sm

    rng = np.random.default_rng(81)
    n = 600
    # An AR(1) factor with strong persistence.
    factor = np.zeros(n)
    for t in range(1, n):
        factor[t] = 0.85 * factor[t - 1] + rng.normal(0, 0.01)
    index = pd.bdate_range("2016-01-04", periods=n)
    factors = pd.DataFrame({"persistent": factor}, index=index)
    returns = pd.Series(0.6 * factor + rng.normal(0, 0.01, n), index=index)

    hac = factor_exposure(returns, factors, newey_west_lags=10)
    plain = sm.OLS(returns.to_numpy(), sm.add_constant(factor)).fit()
    assert hac.loadings[0].std_error > float(plain.bse[1])


def test_factor_exposure_validates_its_inputs() -> None:
    factors = make_factors(n=200)
    returns = factors["market"]
    with pytest.raises(RiskError, match="no factor columns"):
        factor_exposure(returns, pd.DataFrame(index=factors.index))
    with pytest.raises(RiskError, match="no overlapping dates"):
        factor_exposure(
            pd.Series([0.01] * 10, index=pd.bdate_range("2030-01-01", periods=10)),
            factors,
        )
    with pytest.raises(RiskError, match="cannot identify"):
        factor_exposure(returns.iloc[:3], factors.iloc[:3])


def test_exposure_frame_lists_every_factor() -> None:
    factors = make_factors()
    exposure = factor_exposure(factors["market"], factors)
    frame = exposure.frame()
    assert set(frame["factor"]) == {"market", "momentum", "value"}
    assert "significant" in frame.columns


# --- Rolling factor exposure -------------------------------------------------


def test_rolling_factor_exposure_tracks_changing_loadings() -> None:
    factors = make_factors(n=600)
    rng = np.random.default_rng(91)
    loadings = np.concatenate([np.full(300, 0.2), np.full(300, 1.4)])
    returns = pd.Series(
        loadings * factors["market"].to_numpy() + rng.normal(0, 0.002, 600),
        index=factors.index,
    )

    rolling = rolling_factor_exposure(returns, factors, window=120)
    market = rolling.betas["market"].dropna()
    assert float(market.iloc[0]) == pytest.approx(0.2, abs=0.2)
    assert float(market.iloc[-1]) == pytest.approx(1.4, abs=0.2)
    drift = rolling.drift()
    assert float(drift["market"]) > 0.8


def test_rolling_factor_exposure_needs_enough_history() -> None:
    factors = make_factors(n=50)
    with pytest.raises(RiskError, match="aligned observations"):
        rolling_factor_exposure(factors["market"], factors, window=200)


def test_rolling_factor_exposure_reports_r_squared() -> None:
    factors = make_factors(n=400)
    rolling = rolling_factor_exposure(factors["market"], factors, window=120)
    assert rolling.r_squared.dropna().between(0.0, 1.0).all()
    assert set(rolling.factors) == {"market", "momentum", "value"}
