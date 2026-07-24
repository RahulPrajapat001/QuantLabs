"""Microstructure features: volume shock, gap %, dollar volume, lags, ranges.

Also the time-series *rolling rank* features — the percentile of the current
value within its own trailing window (distinct from the cross-sectional rank,
which ranks across securities on a date).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from quantlab.feature_engineering.registry import add_feature, register_feature


@register_feature(
    name="volume_shock_5",
    dependencies=["volume"],
    lookback=5,
    category="microstructure",
)
def volume_shock_5(df: pd.DataFrame) -> pd.Series[float]:
    return df["volume"] / df["volume"].rolling(5).mean()


@register_feature(
    name="volume_shock_20",
    dependencies=["volume"],
    lookback=20,
    category="microstructure",
)
def volume_shock_20(df: pd.DataFrame) -> pd.Series[float]:
    return df["volume"] / df["volume"].rolling(20).mean()


@register_feature(
    name="gap_pct",
    dependencies=["open", "close"],
    lookback=1,
    category="microstructure",
)
def gap_pct(df: pd.DataFrame) -> pd.Series[float]:
    # Overnight gap: today's open vs yesterday's close.
    return df["open"] / df["close"].shift(1) - 1.0


@register_feature(
    name="dollar_volume",
    dependencies=["close", "volume"],
    lookback=1,
    category="microstructure",
)
def dollar_volume(df: pd.DataFrame) -> pd.Series[float]:
    return df["close"] * df["volume"]


LAG_STEPS: tuple[int, ...] = (1, 2, 3, 5)


def _make_return_lag(step: int) -> Callable[[pd.DataFrame], pd.Series[float]]:
    def _lag(df: pd.DataFrame, step: int = step) -> pd.Series[float]:
        return df["return_1d"].shift(step)

    return _lag


for _step in LAG_STEPS:
    add_feature(
        f"return_lag_{_step}",
        _make_return_lag(_step),
        dependencies=["return_1d"],
        lookback=_step + 1,
        category="microstructure",
    )


@register_feature(
    name="close_to_high_20",
    dependencies=["close", "high"],
    lookback=20,
    category="microstructure",
)
def close_to_high_20(df: pd.DataFrame) -> pd.Series[float]:
    return df["close"] / df["high"].rolling(20).max()


@register_feature(
    name="close_to_low_20",
    dependencies=["close", "low"],
    lookback=20,
    category="microstructure",
)
def close_to_low_20(df: pd.DataFrame) -> pd.Series[float]:
    return df["close"] / df["low"].rolling(20).min()


def _rolling_pct_rank(series: pd.Series[float], window: int) -> pd.Series[float]:
    """Percentile of the current value within its own trailing ``window``."""

    def _rank(values: np.ndarray) -> float:
        return float((values <= values[-1]).mean())

    return series.rolling(window).apply(_rank, raw=True)


@register_feature(
    name="rolling_rank_return_20d",
    dependencies=["return_1d"],
    lookback=60,
    category="microstructure",
)
def rolling_rank_return_20d(df: pd.DataFrame) -> pd.Series[float]:
    return _rolling_pct_rank(df["return_1d"], 60)


@register_feature(
    name="rolling_rank_rsi_14",
    dependencies=["rsi_14"],
    lookback=74,
    category="microstructure",
)
def rolling_rank_rsi_14(df: pd.DataFrame) -> pd.Series[float]:
    return _rolling_pct_rank(df["rsi_14"], 60)
