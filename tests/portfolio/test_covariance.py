"""Covariance estimation, and the instability Ledoit-Wolf shrinkage exists to fix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio.covariance import (
    ILL_CONDITIONED,
    CovarianceError,
    constant_correlation_covariance,
    estimate_covariance,
    ledoit_wolf_covariance,
    sample_covariance,
    single_index_covariance,
)

from .conftest import factor_panel


def wide_panel(n_assets: int = 60, n_obs: int = 70, seed: int = 3) -> pd.DataFrame:
    """More assets than is comfortable for the observations available.

    ``T/N ≈ 1.2``: the regime where sample covariance stops being usable and an
    optimiser handed it produces confident nonsense.
    """

    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.012, n_obs)
    betas = rng.uniform(0.4, 1.7, n_assets)
    data = np.outer(market, betas) + rng.normal(0, 0.02, (n_obs, n_assets))
    return pd.DataFrame(data, columns=[f"S{i:03d}" for i in range(n_assets)])


# --- The instability being fixed ---------------------------------------------


def test_sample_covariance_is_ill_conditioned_when_assets_approach_observations() -> (
    None
):
    """The failure mode, demonstrated rather than asserted in a docstring.

    With T/N near 1 the smallest eigenvalues of the sample matrix are almost entirely
    estimation error. An optimiser inverts the matrix, so those directions receive the
    largest weights — which is how a mean-variance portfolio ends up loaded onto the
    noisiest linear combinations in the panel.
    """

    panel = wide_panel(n_assets=60, n_obs=70)
    sample = sample_covariance(panel)

    assert sample.observations_per_asset < 2.0
    assert sample.condition_number > 1e3
    assert sample.needs_shrinkage
    assert "ill-conditioned" in sample.render()
    assert "ledoit_wolf" in sample.render()


def test_shrinkage_conditions_the_matrix() -> None:
    panel = wide_panel(n_assets=60, n_obs=70)
    sample = sample_covariance(panel)
    shrunk = ledoit_wolf_covariance(panel)

    assert shrunk.condition_number < sample.condition_number
    assert shrunk.shrinkage > 0.0
    assert shrunk.is_positive_definite
    assert not shrunk.needs_shrinkage


def test_shrunk_matrix_is_invertible_where_the_sample_one_is_hopeless() -> None:
    """More assets than observations: the sample matrix is singular by construction."""

    panel = wide_panel(n_assets=80, n_obs=40)
    sample = sample_covariance(panel)
    # Rank deficient: at least one eigenvalue is numerically zero.
    assert sample.condition_number > ILL_CONDITIONED
    assert not sample.is_positive_definite

    shrunk = ledoit_wolf_covariance(panel)
    assert shrunk.is_positive_definite
    inverse = np.linalg.inv(shrunk.matrix.to_numpy())
    # A genuine inverse: A·A⁻¹ = I.
    assert np.allclose(shrunk.matrix.to_numpy() @ inverse, np.eye(80), atol=1e-6)


# --- Shrinkage behaviour -----------------------------------------------------


def test_shrinkage_intensity_is_reported_and_bounded(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    assert 0.0 <= estimate.shrinkage <= 1.0
    assert f"{estimate.shrinkage:.4f}" in estimate.render()


def test_shrinkage_is_higher_when_data_is_thinner() -> None:
    """Less data, less trust in the sample, more weight on the target."""

    thick = ledoit_wolf_covariance(factor_panel(n_assets=15, n_obs=1000))
    thin = ledoit_wolf_covariance(factor_panel(n_assets=15, n_obs=60))
    assert thin.shrinkage > thick.shrinkage


def test_iid_data_shrinks_all_the_way_to_the_target() -> None:
    """A sanity check on the estimator, not a pathology.

    When the data genuinely is uncorrelated with equal variances, the scaled-identity
    target *is* the truth, and the analytic intensity correctly goes to 1.
    """

    rng = np.random.default_rng(1)
    iid = pd.DataFrame(rng.normal(0, 0.015, (500, 8)), columns=list("ABCDEFGH"))
    estimate = ledoit_wolf_covariance(iid)
    assert estimate.shrinkage > 0.9


def test_factor_structure_shrinks_only_partially(panel) -> None:
    """With real covariance structure to estimate, shrinkage stays moderate."""

    estimate = ledoit_wolf_covariance(panel)
    assert 0.0 < estimate.shrinkage < 0.5


def test_sample_covariance_has_no_shrinkage(panel) -> None:
    assert sample_covariance(panel).shrinkage == 0.0


# --- Alternative targets -----------------------------------------------------


def test_constant_correlation_target_shrinks_less_than_identity(panel) -> None:
    """A better structural guess needs less of itself to condition the matrix.

    The average equity correlation is genuinely positive, so a constant-correlation
    target is closer to the truth than a zero-correlation identity — and less of the
    sample's real structure has to be discarded.
    """

    identity = ledoit_wolf_covariance(panel)
    constant = constant_correlation_covariance(panel)
    assert constant.method == "constant_correlation"
    assert 0.0 <= constant.shrinkage <= 1.0
    # Both condition the matrix relative to the raw sample.
    sample = sample_covariance(panel)
    assert constant.condition_number <= sample.condition_number + 1e-6
    assert identity.condition_number <= sample.condition_number + 1e-6


def test_constant_correlation_accepts_an_explicit_intensity(panel) -> None:
    fixed = constant_correlation_covariance(panel, shrinkage=0.4)
    assert fixed.shrinkage == pytest.approx(0.4)
    full = constant_correlation_covariance(panel, shrinkage=1.0)
    # At full shrinkage the diagonal is untouched but correlations are all equal.
    correlation = full.correlation().to_numpy()
    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
    assert np.allclose(off_diagonal, off_diagonal[0], atol=1e-8)


def test_single_index_target_uses_market_betas(panel) -> None:
    market = panel.mean(axis=1).rename("market")
    estimate = single_index_covariance(panel, market, shrinkage=0.5)
    assert estimate.method == "single_index"
    assert estimate.is_positive_definite
    assert estimate.shrinkage == pytest.approx(0.5)


def test_single_index_needs_a_market_series(panel) -> None:
    with pytest.raises(CovarianceError, match="needs a market return series"):
        estimate_covariance(panel, method="single_index")


def test_single_index_rejects_a_constant_market(panel) -> None:
    flat = pd.Series(0.0, index=panel.index, name="market")
    with pytest.raises(CovarianceError, match="zero variance"):
        single_index_covariance(panel, flat)


# --- Diagnostics -------------------------------------------------------------


def test_volatilities_come_off_the_diagonal_annualised(panel) -> None:
    estimate = sample_covariance(panel)
    volatilities = estimate.volatilities
    expected = panel.std(ddof=1) * np.sqrt(252)
    pd.testing.assert_series_equal(
        volatilities, expected.rename("volatility"), atol=1e-10, check_names=False
    )


def test_correlation_is_recovered_from_the_covariance(panel) -> None:
    estimate = sample_covariance(panel)
    pd.testing.assert_frame_equal(
        estimate.correlation(), panel.corr(), atol=1e-10, check_names=False
    )


def test_annualised_matrix_scales_by_periods_per_year(panel) -> None:
    estimate = sample_covariance(panel, periods_per_year=252)
    assert np.allclose(estimate.annualised.to_numpy(), estimate.matrix.to_numpy() * 252)


def test_observations_per_asset_is_reported(panel) -> None:
    estimate = sample_covariance(panel)
    assert estimate.observations_per_asset == pytest.approx(
        panel.shape[0] / panel.shape[1]
    )


# --- Input validation --------------------------------------------------------


def test_too_few_observations_is_refused() -> None:
    rng = np.random.default_rng(0)
    tiny = pd.DataFrame(rng.normal(0, 0.01, (8, 5)))
    with pytest.raises(CovarianceError, match="at least 20 complete observations"):
        sample_covariance(tiny)


def test_empty_input_is_refused() -> None:
    with pytest.raises(CovarianceError):
        sample_covariance(pd.DataFrame())


def test_unknown_method_is_refused(panel) -> None:
    with pytest.raises(CovarianceError, match="unknown covariance method"):
        estimate_covariance(panel, method="magic")  # type: ignore[arg-type]


def test_dispatch_defaults_to_ledoit_wolf(panel) -> None:
    """The default is shrunk on purpose: a raw sample matrix is the dangerous choice."""

    assert estimate_covariance(panel).method == "ledoit_wolf"
