"""Technical indicators: RSI, MACD, ADX (+DI/-DI), OBV, VWAP.

MACD reads the shared ``ema_12`` / ``ema_26`` columns rather than recomputing
them, and ADX reads the ``plus_di_14`` / ``minus_di_14`` columns — the DAG keeps
each intermediate single-computed. RSI, MACD and ADX are hand-verified against a
reference implementation in the tests.
"""

from __future__ import annotations

import pandas as pd

from quantlab.feature_engineering._indicators import (
    directional_indicators,
    ema,
    obv,
    rolling_vwap,
    rsi,
)
from quantlab.feature_engineering.registry import register_feature


@register_feature(
    name="rsi_14", dependencies=["close"], lookback=14, category="technical"
)
def rsi_14(df: pd.DataFrame) -> pd.Series[float]:
    return rsi(df["close"], 14)


@register_feature(
    name="macd_line",
    dependencies=["ema_12", "ema_26"],
    lookback=26,
    category="technical",
)
def macd_line(df: pd.DataFrame) -> pd.Series[float]:
    # Shared EMAs already computed upstream in the DAG.
    return df["ema_12"] - df["ema_26"]


@register_feature(
    name="macd_signal",
    dependencies=["macd_line"],
    lookback=35,
    category="technical",
)
def macd_signal(df: pd.DataFrame) -> pd.Series[float]:
    return ema(df["macd_line"], 9)


@register_feature(
    name="macd_hist",
    dependencies=["macd_line", "macd_signal"],
    lookback=35,
    category="technical",
)
def macd_hist(df: pd.DataFrame) -> pd.Series[float]:
    return df["macd_line"] - df["macd_signal"]


@register_feature(
    name="plus_di_14",
    dependencies=["high", "low", "close"],
    lookback=14,
    category="technical",
)
def plus_di_14(df: pd.DataFrame) -> pd.Series[float]:
    return directional_indicators(df, 14)[0]


@register_feature(
    name="minus_di_14",
    dependencies=["high", "low", "close"],
    lookback=14,
    category="technical",
)
def minus_di_14(df: pd.DataFrame) -> pd.Series[float]:
    return directional_indicators(df, 14)[1]


@register_feature(
    name="adx_14",
    dependencies=["high", "low", "close"],
    lookback=28,
    category="technical",
)
def adx_14(df: pd.DataFrame) -> pd.Series[float]:
    return directional_indicators(df, 14)[2]


@register_feature(
    name="obv", dependencies=["close", "volume"], lookback=1, category="technical"
)
def obv_feature(df: pd.DataFrame) -> pd.Series[float]:
    return obv(df["close"], df["volume"])


@register_feature(
    name="vwap_14",
    dependencies=["high", "low", "close", "volume"],
    lookback=14,
    category="technical",
)
def vwap_14(df: pd.DataFrame) -> pd.Series[float]:
    return rolling_vwap(df, 14)
