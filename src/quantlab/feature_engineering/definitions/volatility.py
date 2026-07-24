"""Volatility features: rolling return std, ATR, Parkinson, Bollinger width."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from quantlab.feature_engineering._indicators import atr
from quantlab.feature_engineering.registry import add_feature, register_feature

VOL_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)


def _make_vol(window: int) -> Callable[[pd.DataFrame], pd.Series[float]]:
    def _vol(df: pd.DataFrame, window: int = window) -> pd.Series[float]:
        # Rolling std of daily log returns; trailing window only.
        return df["log_return_1d"].rolling(window).std()

    return _vol


for _w in VOL_WINDOWS:
    add_feature(
        f"volatility_{_w}d",
        _make_vol(_w),
        dependencies=["log_return_1d"],
        lookback=_w,
        category="volatility",
    )


@register_feature(
    name="atr_14",
    dependencies=["high", "low", "close"],
    lookback=14,
    category="volatility",
)
def atr_14(df: pd.DataFrame) -> pd.Series[float]:
    return atr(df, 14)


@register_feature(
    name="parkinson_20",
    dependencies=["high", "low"],
    lookback=20,
    category="volatility",
)
def parkinson_20(df: pd.DataFrame) -> pd.Series[float]:
    # Parkinson high-low range volatility estimator.
    log_hl_sq = cast("pd.Series[float]", np.log(df["high"] / df["low"])) ** 2
    scaled = log_hl_sq.rolling(20).mean() / (4.0 * np.log(2.0))
    return cast("pd.Series[float]", np.sqrt(scaled))


@register_feature(
    name="bollinger_width_20",
    dependencies=["close"],
    lookback=20,
    category="volatility",
)
def bollinger_width_20(df: pd.DataFrame) -> pd.Series[float]:
    # (upper - lower) / mid with k=2 == 4*std/mean; a normalised band width.
    mean = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    return 4.0 * std / mean
