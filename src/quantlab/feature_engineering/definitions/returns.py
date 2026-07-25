"""Return features: simple, log, multi-day, rolling and cumulative.

``return_1d`` and ``log_return_1d`` are the shared roots that momentum,
volatility and the risk-adjusted family all build on — computed once per security
and reused via the DAG.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from quantlab.feature_engineering.registry import register_feature


@register_feature(
    name="return_1d",
    dependencies=["close"],
    lookback=1,
    category="returns",
    description="Simple one-session close-to-close return.",
)
def return_1d(df: pd.DataFrame) -> pd.Series[float]:
    return df["close"].pct_change()


@register_feature(
    name="log_return_1d",
    dependencies=["close"],
    lookback=1,
    category="returns",
    description="Natural log of the one-session close ratio; additive across time.",
)
def log_return_1d(df: pd.DataFrame) -> pd.Series[float]:
    ratio = df["close"] / df["close"].shift(1)
    return cast("pd.Series[float]", np.log(ratio))


@register_feature(
    name="return_5d",
    dependencies=["close"],
    lookback=5,
    category="returns",
    description="Simple return over the trailing 5 sessions (about one week).",
)
def return_5d(df: pd.DataFrame) -> pd.Series[float]:
    return df["close"].pct_change(5)


@register_feature(
    name="return_10d",
    dependencies=["close"],
    lookback=10,
    category="returns",
    description="Simple return over the trailing 10 sessions (about two weeks).",
)
def return_10d(df: pd.DataFrame) -> pd.Series[float]:
    return df["close"].pct_change(10)


@register_feature(
    name="return_20d",
    dependencies=["close"],
    lookback=20,
    category="returns",
    description="Simple return over the trailing 20 sessions (about one month).",
)
def return_20d(df: pd.DataFrame) -> pd.Series[float]:
    return df["close"].pct_change(20)


@register_feature(
    name="rolling_return_20d",
    dependencies=["log_return_1d"],
    lookback=20,
    category="returns",
    description=(
        "Sum of log returns over 20 sessions — a compounding-safe cumulative return."
    ),
)
def rolling_return_20d(df: pd.DataFrame) -> pd.Series[float]:
    return df["log_return_1d"].rolling(20).sum()


@register_feature(
    name="rolling_return_60d",
    dependencies=["log_return_1d"],
    lookback=60,
    category="returns",
    description="Sum of log returns over 60 sessions (about one quarter).",
)
def rolling_return_60d(df: pd.DataFrame) -> pd.Series[float]:
    return df["log_return_1d"].rolling(60).sum()


@register_feature(
    name="cum_log_return",
    dependencies=["log_return_1d"],
    lookback=1,
    category="returns",
    description="Cumulative sum of log returns since the series began.",
)
def cum_log_return(df: pd.DataFrame) -> pd.Series[float]:
    # Cumulative from the series start; causal and unaffected by future rows.
    return df["log_return_1d"].cumsum()
