"""Hand-verification of RSI, MACD, ATR and ADX against reference implementations.

Two kinds of check per indicator:

* **Reference recursion.** An independent, explicit iterative implementation of
  the same definition (EMA / Wilder recursion). Because pandas ``ewm(adjust=False)``
  seeds at the first observation and there are no interior NaNs in these inputs,
  the vectorised feature must match the loop exactly wherever the feature is
  defined.
* **Convention-independent truths.** A monotonically rising close gives RSI 100
  and a falling one RSI 0; ATR of constant-range bars approaches that range; ADX
  stays within [0, 100]. These hold regardless of implementation details.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.feature_engineering._indicators import (
    atr,
    directional_indicators,
    ema,
    macd,
    rsi,
    true_range,
)


def _ema_ref(values: np.ndarray, span: int) -> np.ndarray:
    """Explicit EMA recursion (adjust=False), seeded at the first non-NaN value."""

    alpha = 2.0 / (span + 1.0)
    out = np.full(len(values), np.nan)
    prev: float | None = None
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        prev = v if prev is None else alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _wilder_ref(values: np.ndarray, period: int) -> np.ndarray:
    """Explicit Wilder recursion (alpha = 1/period), seeded at first non-NaN."""

    alpha = 1.0 / period
    out = np.full(len(values), np.nan)
    prev: float | None = None
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        prev = v if prev is None else alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _sample_ohlcv(n: int = 200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.02, n))
    intraday = np.abs(rng.normal(0.0, 0.015, n))
    return pd.DataFrame(
        {
            "high": close * (1.0 + intraday),
            "low": close * (1.0 - intraday),
            "close": close,
        }
    )


def _compare_where_defined(feature: pd.Series, reference: np.ndarray) -> None:
    mask = feature.notna().to_numpy()
    assert mask.sum() > 20  # a meaningful overlap was actually compared
    np.testing.assert_allclose(
        feature.to_numpy()[mask], reference[mask], rtol=1e-9, atol=1e-9
    )


# --- EMA (the shared building block) ----------------------------------------


def test_ema_matches_reference_recursion() -> None:
    df = _sample_ohlcv()
    _compare_where_defined(ema(df["close"], 12), _ema_ref(df["close"].to_numpy(), 12))


# --- RSI ---------------------------------------------------------------------


def test_rsi_matches_reference() -> None:
    df = _sample_ohlcv()
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0).to_numpy()
    loss = (-delta).clip(lower=0.0).to_numpy()
    avg_gain = _wilder_ref(gain, 14)
    avg_loss = _wilder_ref(loss, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi_ref = 100.0 - 100.0 / (1.0 + rs)
    _compare_where_defined(rsi(close, 14), rsi_ref)


def test_rsi_bounds_on_monotone_series() -> None:
    rising = pd.Series(np.linspace(100, 200, 80))
    falling = pd.Series(np.linspace(200, 100, 80))
    assert np.isclose(rsi(rising, 14).iloc[-1], 100.0)
    assert np.isclose(rsi(falling, 14).iloc[-1], 0.0)


# --- MACD --------------------------------------------------------------------


def test_macd_matches_reference() -> None:
    df = _sample_ohlcv()
    close = df["close"]
    line, signal, hist = macd(close, 12, 26, 9)

    ref_line = _ema_ref(close.to_numpy(), 12) - _ema_ref(close.to_numpy(), 26)
    _compare_where_defined(line, ref_line)

    # Histogram is exactly line minus signal.
    valid = hist.notna()
    np.testing.assert_allclose(
        hist[valid].to_numpy(), (line - signal)[valid].to_numpy(), rtol=1e-12
    )


# --- ATR ---------------------------------------------------------------------


def test_atr_matches_reference() -> None:
    df = _sample_ohlcv()
    tr = true_range(df).to_numpy()
    _compare_where_defined(atr(df, 14), _wilder_ref(tr, 14))


def test_atr_of_constant_range_bars() -> None:
    # Flat closes with a fixed 4-point high/low band -> ATR converges to 4.
    n = 100
    close = pd.Series(np.full(n, 50.0))
    df = pd.DataFrame({"high": close + 2.0, "low": close - 2.0, "close": close})
    assert np.isclose(atr(df, 14).iloc[-1], 4.0, atol=1e-6)


# --- ADX ---------------------------------------------------------------------


def test_adx_matches_reference() -> None:
    df = _sample_ohlcv()
    period = 14
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_ref = _wilder_ref(true_range(df).to_numpy(), period)
    plus_di_ref = 100.0 * _wilder_ref(plus_dm, period) / atr_ref
    minus_di_ref = 100.0 * _wilder_ref(minus_dm, period) / atr_ref
    # Mirror the feature's min_periods warm-up so the DX smoothing seeds at the
    # same index (EWM has long memory; a different seed leaves a decaying offset).
    plus_di_ref[: period - 1] = np.nan
    minus_di_ref[: period - 1] = np.nan
    di_sum = plus_di_ref + minus_di_ref
    with np.errstate(divide="ignore", invalid="ignore"):
        dx_ref = 100.0 * np.abs(plus_di_ref - minus_di_ref) / di_sum
    adx_ref = _wilder_ref(dx_ref, period)

    _, _, adx = directional_indicators(df, period)
    _compare_where_defined(adx, adx_ref)


def test_adx_within_bounds() -> None:
    df = _sample_ohlcv()
    _, _, adx = directional_indicators(df, 14)
    valid = adx.dropna()
    assert (valid >= 0.0).all() and (valid <= 100.0).all()
