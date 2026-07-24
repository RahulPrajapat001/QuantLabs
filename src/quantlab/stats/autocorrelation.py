"""ACF and PACF wrappers with confidence bands (PRD Module 5)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf as _acf
from statsmodels.tsa.stattools import pacf as _pacf

from .results import AutocorrelationResult


def _as_array(series: pd.Series | Sequence[float]) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    return cast("np.ndarray", arr[~np.isnan(arr)])


def _significant_lags(values: np.ndarray, confint: np.ndarray) -> tuple[int, ...]:
    # confint are absolute bounds around each coefficient; a lag is significant
    # when the coefficient sits outside the band centred on it. Lag 0 (== 1.0) is
    # skipped.
    lower = confint[:, 0] - values
    upper = confint[:, 1] - values
    significant = []
    for lag in range(1, len(values)):
        if values[lag] < lower[lag] or values[lag] > upper[lag]:
            significant.append(lag)
    return tuple(significant)


def acf(
    series: pd.Series | Sequence[float], *, nlags: int = 20, alpha: float = 0.05
) -> AutocorrelationResult:
    """Autocorrelation function up to ``nlags`` with ``(1-alpha)`` bands."""

    values, confint = _acf(_as_array(series), nlags=nlags, alpha=alpha, fft=True)
    sig = _significant_lags(values, confint)
    interpretation = (
        f"significant autocorrelation at lag(s) {', '.join(map(str, sig))}"
        if sig
        else "no autocorrelation significant beyond the confidence band"
    )
    return AutocorrelationResult(
        name="ACF",
        values=tuple(float(v) for v in values),
        confint_lower=tuple(float(v) for v in confint[:, 0]),
        confint_upper=tuple(float(v) for v in confint[:, 1]),
        significant_lags=sig,
        interpretation=interpretation,
    )


def pacf(
    series: pd.Series | Sequence[float], *, nlags: int = 20, alpha: float = 0.05
) -> AutocorrelationResult:
    """Partial autocorrelation function up to ``nlags`` with ``(1-alpha)`` bands."""

    values, confint = _pacf(_as_array(series), nlags=nlags, alpha=alpha)
    sig = _significant_lags(values, confint)
    interpretation = (
        f"significant partial autocorrelation at lag(s) {', '.join(map(str, sig))}"
        if sig
        else "no partial autocorrelation significant beyond the confidence band"
    )
    return AutocorrelationResult(
        name="PACF",
        values=tuple(float(v) for v in values),
        confint_lower=tuple(float(v) for v in confint[:, 0]),
        confint_upper=tuple(float(v) for v in confint[:, 1]),
        significant_lags=sig,
        interpretation=interpretation,
    )
