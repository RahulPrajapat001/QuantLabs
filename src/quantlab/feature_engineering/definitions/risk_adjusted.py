"""Risk-adjusted features: rolling Sharpe, Sortino, Calmar, beta and alpha.

Beta and alpha read the ``market_return`` base column (the cross-sectional mean
daily return, supplied by the build layer); alpha reuses the ``beta_*`` column so
the rolling regression slope is computed once.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from quantlab.feature_engineering.registry import register_feature

ANNUALISATION: float = float(np.sqrt(252.0))


@register_feature(
    name="rolling_sharpe_60",
    dependencies=["return_1d"],
    lookback=60,
    category="risk_adjusted",
)
def rolling_sharpe_60(df: pd.DataFrame) -> pd.Series[float]:
    r = df["return_1d"]
    return ANNUALISATION * r.rolling(60).mean() / r.rolling(60).std()


@register_feature(
    name="rolling_sharpe_120",
    dependencies=["return_1d"],
    lookback=120,
    category="risk_adjusted",
)
def rolling_sharpe_120(df: pd.DataFrame) -> pd.Series[float]:
    r = df["return_1d"]
    return ANNUALISATION * r.rolling(120).mean() / r.rolling(120).std()


def _downside_std(returns: pd.Series[float], window: int) -> pd.Series[float]:
    downside = returns.clip(upper=0.0)
    return cast("pd.Series[float]", np.sqrt((downside**2).rolling(window).mean()))


@register_feature(
    name="rolling_sortino_60",
    dependencies=["return_1d"],
    lookback=60,
    category="risk_adjusted",
)
def rolling_sortino_60(df: pd.DataFrame) -> pd.Series[float]:
    r = df["return_1d"]
    return ANNUALISATION * r.rolling(60).mean() / _downside_std(r, 60)


@register_feature(
    name="rolling_sortino_120",
    dependencies=["return_1d"],
    lookback=120,
    category="risk_adjusted",
)
def rolling_sortino_120(df: pd.DataFrame) -> pd.Series[float]:
    r = df["return_1d"]
    return ANNUALISATION * r.rolling(120).mean() / _downside_std(r, 120)


@register_feature(
    name="calmar_252",
    dependencies=["return_1d"],
    lookback=252,
    category="risk_adjusted",
)
def calmar_252(df: pd.DataFrame) -> pd.Series[float]:
    def _calmar(window: np.ndarray) -> float:
        equity = np.cumprod(1.0 + window)
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1.0
        max_dd = drawdown.min()
        if max_dd >= 0.0:
            return np.nan
        annual = equity[-1] ** (252.0 / len(window)) - 1.0
        return float(annual / abs(max_dd))

    return df["return_1d"].rolling(252).apply(_calmar, raw=True)


def _rolling_beta(
    ret: pd.Series[float], mkt: pd.Series[float], window: int
) -> pd.Series[float]:
    covariance = ret.rolling(window).cov(mkt)
    variance = mkt.rolling(window).var()
    return covariance / variance


@register_feature(
    name="beta_60",
    dependencies=["return_1d", "market_return"],
    lookback=60,
    category="risk_adjusted",
)
def beta_60(df: pd.DataFrame) -> pd.Series[float]:
    return _rolling_beta(df["return_1d"], df["market_return"], 60)


@register_feature(
    name="beta_120",
    dependencies=["return_1d", "market_return"],
    lookback=120,
    category="risk_adjusted",
)
def beta_120(df: pd.DataFrame) -> pd.Series[float]:
    return _rolling_beta(df["return_1d"], df["market_return"], 120)


@register_feature(
    name="alpha_60",
    dependencies=["return_1d", "market_return", "beta_60"],
    lookback=60,
    category="risk_adjusted",
)
def alpha_60(df: pd.DataFrame) -> pd.Series[float]:
    # Rolling CAPM intercept: E[r] - beta * E[r_mkt] over the window.
    mean_r = df["return_1d"].rolling(60).mean()
    mean_m = df["market_return"].rolling(60).mean()
    return mean_r - df["beta_60"] * mean_m


@register_feature(
    name="alpha_120",
    dependencies=["return_1d", "market_return", "beta_120"],
    lookback=120,
    category="risk_adjusted",
)
def alpha_120(df: pd.DataFrame) -> pd.Series[float]:
    mean_r = df["return_1d"].rolling(120).mean()
    mean_m = df["market_return"].rolling(120).mean()
    return mean_r - df["beta_120"] * mean_m
