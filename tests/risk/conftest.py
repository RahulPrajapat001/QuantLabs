"""Fixtures for the risk-analytics tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def normal_returns(
    *, n: int = 1000, mean: float = 0.0004, sd: float = 0.012, seed: int = 11
) -> pd.Series:
    """A well-behaved Gaussian return series — the case where the three VaR
    estimators should agree."""

    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(mean, sd, n),
        index=pd.bdate_range("2016-01-04", periods=n),
        name="return",
    )


def fat_tailed_returns(
    *, n: int = 1000, sd: float = 0.012, df: int = 3, seed: int = 12
) -> pd.Series:
    """Student-t returns with negative skew — where Gaussian VaR understates the tail.

    Scaled so the sample standard deviation is comparable to the normal fixture, and
    given a negative skew by dampening the upside, which is what equity returns
    actually look like.
    """

    rng = np.random.default_rng(seed)
    raw = rng.standard_t(df, n)
    raw = raw / raw.std() * sd
    # Negative skew: compress gains, leave losses alone.
    skewed = np.where(raw > 0, raw * 0.6, raw)
    return pd.Series(
        skewed, index=pd.bdate_range("2016-01-04", periods=n), name="return"
    )


def correlated_panel(
    *,
    n_assets: int = 8,
    n_obs: int = 500,
    market_beta_low: float = 0.6,
    market_beta_high: float = 1.4,
    seed: int = 13,
) -> pd.DataFrame:
    """A panel with a shared market factor, so correlations are genuinely positive."""

    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.011, n_obs)
    betas = rng.uniform(market_beta_low, market_beta_high, n_assets)
    idiosyncratic = rng.normal(0, 0.013, (n_obs, n_assets))
    data = np.outer(market, betas) + idiosyncratic
    return pd.DataFrame(
        data,
        columns=[f"SEC-{i:04d}" for i in range(n_assets)],
        index=pd.bdate_range("2016-01-04", periods=n_obs),
    )


def blocked_panel(
    *, groups: int = 3, per_group: int = 4, n_obs: int = 400, seed: int = 14
) -> pd.DataFrame:
    """A panel with clear block structure: names within a group move together.

    Gives hierarchical clustering a known right answer to recover.
    """

    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}
    for group in range(groups):
        driver = rng.normal(0.0, 0.015, n_obs)
        for member in range(per_group):
            columns[f"G{group}-{member}"] = driver + rng.normal(0.0, 0.004, n_obs)
    return pd.DataFrame(columns, index=pd.bdate_range("2016-01-04", periods=n_obs))


@pytest.fixture
def normal() -> pd.Series:
    return normal_returns()


@pytest.fixture
def fat_tailed() -> pd.Series:
    return fat_tailed_returns()


@pytest.fixture
def panel() -> pd.DataFrame:
    return correlated_panel()
