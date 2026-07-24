"""Momentum features: price change over 5/10/20/50/100/200 sessions."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from quantlab.feature_engineering.registry import add_feature

MOMENTUM_WINDOWS: tuple[int, ...] = (5, 10, 20, 50, 100, 200)


def _make_momentum(window: int) -> Callable[[pd.DataFrame], pd.Series[float]]:
    def _momentum(df: pd.DataFrame, window: int = window) -> pd.Series[float]:
        # close_t / close_{t-window} - 1: uses only trailing data.
        return df["close"] / df["close"].shift(window) - 1.0

    return _momentum


for _window in MOMENTUM_WINDOWS:
    add_feature(
        f"momentum_{_window}d",
        _make_momentum(_window),
        dependencies=["close"],
        lookback=_window,
        category="momentum",
    )
