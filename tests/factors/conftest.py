"""Synthetic panels for factor tests.

``make_signal_panel`` builds prices where a hidden per-(date, security) signal
predicts the *next-day* return with a controllable strength, plus the signal
column itself — so the IC machinery has a known answer to recover. With
``strength=0`` the signal is pure noise, uncorrelated with returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_signal_panel(
    *,
    n_days: int = 300,
    n_securities: int = 40,
    strength: float = 0.02,
    noise: float = 0.02,
    seed: int = 0,
    start: str = "2016-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(panel, prices)`` with a known signal->next-day-return relationship.

    ``panel`` has ``[date, security_id, sig]``; ``prices`` has
    ``[date, security_id, close]``. The return realised on day ``t+1`` is
    ``strength * sig_t + noise``, so ``sig`` at date ``t`` predicts the next day's
    return (a genuine, causal, cross-sectional signal).
    """

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for i in range(n_securities):
        sid = f"SEC-{i:04d}"
        sig = rng.normal(size=n_days)
        fwd = strength * sig + rng.normal(0.0, noise, n_days)
        ret = np.empty(n_days)
        ret[0] = 0.0
        ret[1:] = fwd[:-1]  # signal at t drives return at t+1
        close = 100.0 * np.cumprod(1.0 + ret)
        for t in range(n_days):
            rows.append(
                {
                    "date": dates[t],
                    "security_id": sid,
                    "close": float(close[t]),
                    "sig": float(sig[t]),
                }
            )
    panel = pd.DataFrame(rows)
    prices = panel[["date", "security_id", "close"]].copy()
    return panel, prices
