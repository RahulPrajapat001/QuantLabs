"""Statistical wrappers: correct direction on data with a known answer.

Each test feeds a series whose true nature is known (white noise, a random walk,
a cointegrated pair, ...) and asserts the wrapper reaches the right conclusion and
sets a sensible interpretation string.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quantlab.stats as st


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240724)


def _random_walk(rng: np.random.Generator, n: int = 600) -> pd.Series:
    return pd.Series(np.cumsum(rng.normal(size=n)))


def _white_noise(rng: np.random.Generator, n: int = 600) -> pd.Series:
    return pd.Series(rng.normal(size=n))


def _ar1(rng: np.random.Generator, phi: float = 0.8, n: int = 600) -> pd.Series:
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return pd.Series(x)


# --- ADF / KPSS (complementary) ---------------------------------------------


def test_adf_stationary_on_white_noise(rng: np.random.Generator) -> None:
    result = st.adf(_white_noise(rng))
    assert result.reject_null  # rejects unit root -> stationary
    assert "stationary" in result.interpretation


def test_adf_nonstationary_on_random_walk(rng: np.random.Generator) -> None:
    result = st.adf(_random_walk(rng))
    assert not result.reject_null
    assert "non-stationary" in result.interpretation


def test_kpss_is_complement_of_adf(rng: np.random.Generator) -> None:
    # KPSS null is stationarity: white noise does NOT reject; a walk DOES.
    assert not st.kpss_test(_white_noise(rng)).reject_null
    assert st.kpss_test(_random_walk(rng)).reject_null


# --- Jarque-Bera -------------------------------------------------------------


def test_jarque_bera_accepts_normal_rejects_heavy_tail(
    rng: np.random.Generator,
) -> None:
    assert not st.jarque_bera(pd.Series(rng.normal(size=2000))).reject_null
    assert st.jarque_bera(pd.Series(rng.standard_t(3, size=2000))).reject_null


# --- Ljung-Box ---------------------------------------------------------------


def test_ljung_box_detects_autocorrelation(rng: np.random.Generator) -> None:
    assert not st.ljung_box(_white_noise(rng), lags=10).reject_null
    assert st.ljung_box(_ar1(rng), lags=10).reject_null


# --- ACF / PACF --------------------------------------------------------------


def test_acf_pacf_flag_ar1_lag_one(rng: np.random.Generator) -> None:
    ar1 = _ar1(rng)
    assert 1 in st.acf(ar1, nlags=10).significant_lags
    # For an AR(1), PACF cuts off after lag 1.
    assert 1 in st.pacf(ar1, nlags=10).significant_lags


# --- Cointegration -----------------------------------------------------------


def test_engle_granger_finds_cointegration(rng: np.random.Generator) -> None:
    x = np.cumsum(rng.normal(size=500))
    y = 2.0 * x + rng.normal(scale=0.5, size=500)  # stationary spread
    result = st.engle_granger(pd.Series(y), pd.Series(x))
    assert result.cointegrated
    assert result.hedge_ratio == pytest.approx(2.0, abs=0.15)


def test_engle_granger_rejects_independent_walks(rng: np.random.Generator) -> None:
    x = np.cumsum(rng.normal(size=500))
    y = np.cumsum(rng.normal(size=500))  # unrelated walk
    assert not st.engle_granger(pd.Series(y), pd.Series(x)).cointegrated


def test_johansen_finds_one_relation(rng: np.random.Generator) -> None:
    x = np.cumsum(rng.normal(size=500))
    y = 1.5 * x + rng.normal(scale=0.5, size=500)
    result = st.johansen(pd.DataFrame({"y": y, "x": x}))
    assert result.n_cointegrating >= 1


# --- Multivariate ------------------------------------------------------------


def test_vif_flags_collinear_feature(rng: np.random.Generator) -> None:
    a = rng.normal(size=400)
    df = pd.DataFrame(
        {
            "a": a,
            "b": a * 0.95 + rng.normal(scale=0.1, size=400),
            "c": rng.normal(size=400),
        }
    )
    result = st.vif(df)
    assert "a" in result.high_collinearity or "b" in result.high_collinearity
    assert result.vif["c"] < result.vif["a"]


def test_pca_reports_variance(rng: np.random.Generator) -> None:
    a = rng.normal(size=400)
    df = pd.DataFrame(
        {"a": a, "b": a + rng.normal(scale=0.05, size=400), "c": rng.normal(size=400)}
    )
    result = st.pca(df)
    assert sum(result.explained_variance_ratio) == pytest.approx(1.0, abs=1e-6)
    assert result.n_components_for_90pct <= 3
    assert result.explained_variance_ratio[0] >= result.explained_variance_ratio[1]


def test_correlation_and_covariance(rng: np.random.Generator) -> None:
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "b": -a + rng.normal(scale=0.01, size=300)})
    corr = st.correlation_matrix(df)
    assert corr.matrix.loc["a", "b"] == pytest.approx(-1.0, abs=0.05)
    cov = st.covariance_matrix(df)
    assert cov.matrix.shape == (2, 2)
