"""Pure, causal indicator math shared by the feature definitions.

Every function here uses only trailing data — no row's output depends on a later
row — so features built on them pass the registry-wide look-ahead test. The
Wilder-smoothed indicators (RSI, ATR, ADX) use exponential smoothing with
``alpha = 1/period`` via :func:`wilder_rma`; the tests verify each against an
independent explicit-recursion reference and against convention-independent
truths (a monotonically rising close gives RSI 100, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series[float], span: int) -> pd.Series[float]:
    """Exponential moving average (standard TA recursion, ``adjust=False``)."""

    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wma(series: pd.Series[float], window: int) -> pd.Series[float]:
    """Linearly weighted moving average (most recent bar weighted highest)."""

    weights = np.arange(1, window + 1, dtype=float)
    denom = weights.sum()

    def _dot(values: np.ndarray) -> float:
        return float(np.dot(values, weights) / denom)

    return series.rolling(window).apply(_dot, raw=True)


def wilder_rma(series: pd.Series[float], period: int) -> pd.Series[float]:
    """Wilder's running moving average: EWMA with ``alpha = 1/period``.

    Causal by construction; the value at *t* is a function of the seed and every
    prior value up to *t*, never a future one.
    """

    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series[float]:
    """Wilder's True Range: ``max(H-L, |H-Cprev|, |L-Cprev|)``."""

    prev_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - prev_close).abs()
    low_close = (df["low"] - prev_close).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series[float]:
    """Average True Range: Wilder-smoothed True Range."""

    return wilder_rma(true_range(df), period)


def rsi(close: pd.Series[float], period: int = 14) -> pd.Series[float]:
    """Relative Strength Index (Wilder). Rising series -> 100, falling -> 0."""

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)
    rs = avg_gain / avg_loss
    result = 100.0 - 100.0 / (1.0 + rs)
    # All-gains window: avg_loss == 0 -> RS = inf -> RSI = 100 (not NaN).
    return result.where(avg_loss != 0.0, 100.0).where(~avg_gain.isna(), np.nan)


def macd(
    close: pd.Series[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series[float], pd.Series[float], pd.Series[float]]:
    """MACD line, signal line and histogram.

    Line = EMA(fast) - EMA(slow); signal = EMA(line, signal); hist = line - signal.
    """

    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line.dropna(), signal).reindex(close.index)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def directional_indicators(
    df: pd.DataFrame, period: int = 14
) -> tuple[pd.Series[float], pd.Series[float], pd.Series[float]]:
    """Wilder's ``+DI``, ``-DI`` and ``ADX`` over ``period``.

    ``+DM``/``-DM`` are the up/down moves that dominate their opposite; each is
    Wilder-smoothed, divided by the smoothed True Range to give the directional
    indicators, and ADX is the Wilder-smoothed absolute DI spread.
    """

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    atr_ = wilder_rma(true_range(df), period)
    plus_di = 100.0 * wilder_rma(plus_dm, period) / atr_
    minus_di = 100.0 * wilder_rma(minus_dm, period) / atr_

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum != 0.0, np.nan)
    adx = wilder_rma(dx, period)
    return plus_di, minus_di, adx


def obv(close: pd.Series[float], volume: pd.Series[float]) -> pd.Series[float]:
    """On-Balance Volume: signed cumulative volume by daily price direction."""

    direction = pd.Series(np.sign(close.diff().fillna(0.0)), index=close.index)
    return (direction * volume).cumsum()


def rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series[float]:
    """Rolling volume-weighted average price over ``window`` sessions."""

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).rolling(window).sum()
    vol = df["volume"].rolling(window).sum()
    return pv / vol
