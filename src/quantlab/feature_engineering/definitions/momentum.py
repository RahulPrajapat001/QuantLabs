"""Momentum features: price change over 5/10/20/50/100/200 sessions.

Also the classic **12-1 momentum**: the trailing-12-month return skipping the most
recent month, which avoids the well-documented short-term reversal that
contaminates a raw 12-month signal. It is the base of the momentum baseline
factor (PRD Module 6).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from quantlab.feature_engineering.registry import add_feature

MOMENTUM_WINDOWS: tuple[int, ...] = (5, 10, 20, 50, 100, 200)

#: ~12 months and ~1 month in trading days, for the 12-1 momentum feature.
_TWELVE_MONTHS = 252
_ONE_MONTH = 21


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
        description=f"Price change over the trailing {_window} sessions.",
    )


def _momentum_12_1(df: pd.DataFrame) -> pd.Series[float]:
    # Return from t-252 to t-21: skips the most recent month (reversal), causal.
    return df["close"].shift(_ONE_MONTH) / df["close"].shift(_TWELVE_MONTHS) - 1.0


add_feature(
    "momentum_12_1",
    _momentum_12_1,
    dependencies=["close"],
    lookback=_TWELVE_MONTHS,
    category="momentum",
    description=(
        "Trailing 12-month return skipping the most recent month, which avoids "
        "the short-term reversal that contaminates a raw 12-month signal."
    ),
)
