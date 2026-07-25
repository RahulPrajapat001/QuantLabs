"""Fixtures for the portfolio-optimisation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def two_asset_covariance(
    sigma_a: float = 0.20, sigma_b: float = 0.30, rho: float = 0.25
) -> pd.DataFrame:
    """A 2×2 covariance matrix built from volatilities and a correlation.

    Annualised units throughout, so ``periods_per_year=1`` in the optimiser keeps the
    reported volatility directly comparable to the closed-form answer.
    """

    covariance = rho * sigma_a * sigma_b
    return pd.DataFrame(
        [[sigma_a**2, covariance], [covariance, sigma_b**2]],
        index=["A", "B"],
        columns=["A", "B"],
    )


def closed_form_min_variance_weight(
    sigma_a: float = 0.20, sigma_b: float = 0.30, rho: float = 0.25
) -> float:
    """The analytical minimum-variance weight on asset A, for two assets.

    Minimising ``w²σ_a² + (1-w)²σ_b² + 2w(1-w)ρσ_aσ_b`` over ``w`` and solving the
    first-order condition gives::

        w_a = (σ_b² - ρσ_aσ_b) / (σ_a² + σ_b² - 2ρσ_aσ_b)

    Derived independently of the implementation, which is the point of the test.
    """

    covariance = rho * sigma_a * sigma_b
    return (sigma_b**2 - covariance) / (sigma_a**2 + sigma_b**2 - 2.0 * covariance)


def factor_panel(
    *,
    n_assets: int = 10,
    n_obs: int = 400,
    seed: int = 7,
    market_vol: float = 0.012,
) -> pd.DataFrame:
    """A return panel with genuine factor structure.

    Shared market factor with dispersed betas plus dispersed idiosyncratic
    volatility. Unlike an IID panel — which *is* the Ledoit-Wolf shrinkage target and
    therefore shrinks all the way to it — this has real covariance structure to
    estimate, so shrinkage lands somewhere informative.
    """

    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, market_vol, n_obs)
    betas = rng.uniform(0.5, 1.6, n_assets)
    idiosyncratic_vol = rng.uniform(0.008, 0.030, n_assets)
    data = (
        np.outer(market, betas)
        + rng.normal(0, 1, (n_obs, n_assets)) * idiosyncratic_vol
    )
    return pd.DataFrame(
        data,
        columns=[f"SEC-{i:04d}" for i in range(n_assets)],
        index=pd.bdate_range("2016-01-04", periods=n_obs),
    )


@pytest.fixture
def panel() -> pd.DataFrame:
    return factor_panel()


@pytest.fixture
def two_assets() -> pd.DataFrame:
    return two_asset_covariance()
