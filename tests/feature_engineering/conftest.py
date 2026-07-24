"""Deterministic synthetic OHLCV fixtures for the feature-engineering tests.

Offline and reproducible: geometric random walks with a fixed seed, long enough
(600 sessions) that every feature — including the 252-day ones — has non-NaN
values well before the interior points the look-ahead test probes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.feature_engineering import (
    REGISTRY,
    FeatureKind,
    add_market_return,
    compute_time_series_features,
)


def make_security(n: int = 600, *, seed: int = 0) -> pd.DataFrame:
    """One security's OHLCV history plus a ``market_return`` base column."""

    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.018, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    intraday = np.abs(rng.normal(0.0, 0.012, n))
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2015-01-01", periods=n),
            "open": close * (1.0 + rng.normal(0.0, 0.004, n)),
            "high": close * (1.0 + intraday),
            "low": close * (1.0 - intraday),
            "close": close,
            "adj_close": close,
            "volume": rng.integers(50_000, 900_000, n).astype(float),
        }
    )
    # A plausible, causal market proxy (a smoothed version of this security's own
    # return is fine for a single-security look-ahead test).
    df["market_return"] = df["close"].pct_change().rolling(3).mean().fillna(0.0)
    return df


def make_panel(
    n_securities: int = 6, n_days: int = 80, *, seed: int = 0
) -> pd.DataFrame:
    """A long panel with time-series feature columns already computed.

    Used by the cross-sectional look-ahead test and the build tests. Each
    security's time-series features are computed independently, then the frames
    are concatenated into one ``(date, security_id, features…)`` panel.
    """

    ts_names = [s.name for s in REGISTRY.by_kind(FeatureKind.TIME_SERIES)]
    frames = []
    for i in range(n_securities):
        sec = make_security(n_days, seed=seed + i)
        sec["security_id"] = f"SEC-{i:04d}"
        frames.append(sec)
    raw = pd.concat(frames, ignore_index=True)
    raw = add_market_return(raw.drop(columns="market_return"))

    computed = []
    for sid, frame in raw.groupby("security_id", sort=True):
        out = compute_time_series_features(
            frame.sort_values("date").reset_index(drop=True), ts_names
        )
        out["security_id"] = sid
        computed.append(out)
    return pd.concat(computed, ignore_index=True)


@pytest.fixture(scope="session")
def security_df() -> pd.DataFrame:
    return make_security(600, seed=0)


@pytest.fixture(scope="session")
def panel_df() -> pd.DataFrame:
    return make_panel(6, 80, seed=100)
