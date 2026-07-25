"""Moving-average features: SMA, EMA and WMA over several windows.

The EMA family deliberately includes spans 12 and 26 so MACD depends on
``ema_12`` / ``ema_26`` rather than recomputing them — the canonical shared
intermediate in the DAG.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from quantlab.feature_engineering._indicators import ema, wma
from quantlab.feature_engineering.registry import add_feature

SMA_WINDOWS: tuple[int, ...] = (5, 10, 20, 50, 100, 200)
EMA_WINDOWS: tuple[int, ...] = (5, 10, 12, 20, 26, 50, 100, 200)
WMA_WINDOWS: tuple[int, ...] = (10, 20, 50)


def _make_sma(window: int) -> Callable[[pd.DataFrame], pd.Series[float]]:
    def _sma(df: pd.DataFrame, window: int = window) -> pd.Series[float]:
        return df["close"].rolling(window).mean()

    return _sma


def _make_ema(window: int) -> Callable[[pd.DataFrame], pd.Series[float]]:
    def _ema(df: pd.DataFrame, window: int = window) -> pd.Series[float]:
        return ema(df["close"], window)

    return _ema


def _make_wma(window: int) -> Callable[[pd.DataFrame], pd.Series[float]]:
    def _wma(df: pd.DataFrame, window: int = window) -> pd.Series[float]:
        return wma(df["close"], window)

    return _wma


for _w in SMA_WINDOWS:
    add_feature(
        f"sma_{_w}",
        _make_sma(_w),
        dependencies=["close"],
        lookback=_w,
        category="moving_average",
        description=(
            f"Simple moving average of close over {_w} sessions; every bar "
            "weighted equally."
        ),
    )

for _w in EMA_WINDOWS:
    add_feature(
        f"ema_{_w}",
        _make_ema(_w),
        dependencies=["close"],
        lookback=_w,
        category="moving_average",
        description=(
            f"Exponential moving average of close, span {_w}; recent bars "
            "weighted more heavily."
        ),
    )

for _w in WMA_WINDOWS:
    add_feature(
        f"wma_{_w}",
        _make_wma(_w),
        dependencies=["close"],
        lookback=_w,
        category="moving_average",
        description=(f"Linearly weighted moving average of close over {_w} sessions."),
    )
