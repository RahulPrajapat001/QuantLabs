"""Hypothesis tests wrapping statsmodels: ADF, KPSS, Jarque-Bera, Ljung-Box.

Thin wrappers — statsmodels does the maths. Each returns a
:class:`HypothesisTestResult` whose ``interpretation`` states the *conclusion*
in plain English, so callers need not remember that ADF's null is a unit root
while KPSS's null is stationarity (opposite directions, complementary tests).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera as _jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss

from .results import HypothesisTestResult

SIGNIFICANCE = 0.05


def _as_array(series: pd.Series | Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    return cast("np.ndarray", arr[~np.isnan(arr)])


def adf(
    series: pd.Series | Sequence[float],
    *,
    regression: str = "c",
    significance: float = SIGNIFICANCE,
) -> HypothesisTestResult:
    """Augmented Dickey-Fuller test. Null: the series has a unit root.

    Rejecting the null (p < ``significance``) is evidence the series is
    **stationary**.
    """

    stat, p_value, _usedlag, _nobs, crit, _icbest = adfuller(
        _as_array(series), regression=regression, autolag="AIC"
    )
    reject = p_value < significance
    interpretation = (
        "stationary (rejects the unit-root null)"
        if reject
        else "non-stationary — a unit root cannot be rejected"
    )
    return HypothesisTestResult(
        name="Augmented Dickey-Fuller",
        statistic=float(stat),
        p_value=float(p_value),
        null_hypothesis="series has a unit root (non-stationary)",
        reject_null=reject,
        interpretation=interpretation,
        details={f"crit_{k}": float(v) for k, v in crit.items()},
    )


def kpss_test(
    series: pd.Series | Sequence[float],
    *,
    regression: str = "c",
    significance: float = SIGNIFICANCE,
) -> HypothesisTestResult:
    """KPSS test. Null: the series is stationary.

    Rejecting the null (p < ``significance``) is evidence of
    **non-stationarity** — the complement of :func:`adf`.
    """

    with warnings.catch_warnings():
        # statsmodels warns when the p-value is outside the interpolation table;
        # the clipped p-value is still the correct side of the threshold.
        warnings.simplefilter("ignore")
        stat, p_value, _lags, crit = kpss(
            _as_array(series), regression=regression, nlags="auto"
        )
    reject = p_value < significance
    interpretation = (
        "non-stationary (rejects the stationarity null)"
        if reject
        else "stationary — cannot reject the stationarity null"
    )
    return HypothesisTestResult(
        name="KPSS",
        statistic=float(stat),
        p_value=float(p_value),
        null_hypothesis="series is stationary",
        reject_null=reject,
        interpretation=interpretation,
        details={f"crit_{k}": float(v) for k, v in crit.items()},
    )


def jarque_bera(
    series: pd.Series | Sequence[float],
    *,
    significance: float = SIGNIFICANCE,
) -> HypothesisTestResult:
    """Jarque-Bera normality test. Null: the data are normally distributed."""

    arr = _as_array(series)
    stat, p_value, skew, kurtosis = _jarque_bera(arr)
    reject = p_value < significance
    interpretation = (
        f"not normal (skew={skew:.2f}, kurtosis={kurtosis:.2f})"
        if reject
        else "consistent with normality"
    )
    return HypothesisTestResult(
        name="Jarque-Bera",
        statistic=float(stat),
        p_value=float(p_value),
        null_hypothesis="data are normally distributed",
        reject_null=reject,
        interpretation=interpretation,
        details={"skew": float(skew), "kurtosis": float(kurtosis)},
    )


def ljung_box(
    series: pd.Series | Sequence[float],
    *,
    lags: int = 10,
    significance: float = SIGNIFICANCE,
) -> HypothesisTestResult:
    """Ljung-Box test for autocorrelation up to ``lags``.

    Null: the series is independently distributed (no autocorrelation).
    Reports the statistic and p-value at the largest lag; per-lag p-values are in
    ``details``.
    """

    result = acorr_ljungbox(_as_array(series), lags=lags, return_df=True)
    stat = float(result["lb_stat"].iloc[-1])
    p_value = float(result["lb_pvalue"].iloc[-1])
    reject = p_value < significance
    interpretation = (
        f"significant autocorrelation up to lag {lags}"
        if reject
        else f"no significant autocorrelation up to lag {lags}"
    )
    return HypothesisTestResult(
        name="Ljung-Box",
        statistic=stat,
        p_value=p_value,
        null_hypothesis="no autocorrelation (series is white noise)",
        reject_null=reject,
        interpretation=interpretation,
        details={
            f"lag_{int(lag)}_pvalue": float(pv)
            for lag, pv in zip(result.index, result["lb_pvalue"], strict=True)
        },
    )
