"""The IC machinery recovers a known signal and rejects pure noise."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.factors import evaluate_factor, is_eligible
from quantlab.factors.protocol import BaseFactor

from .conftest import make_signal_panel


class _SignalFactor(BaseFactor):
    """Scores the panel's hidden ``sig`` column directly."""

    name = "known_signal"

    def _signal(self, panel: pd.DataFrame) -> pd.Series:
        return panel["sig"]


class _NoiseFactor(BaseFactor):
    """Scores are independent random draws — uncorrelated with returns."""

    name = "pure_noise"

    def __init__(self, seed: int = 123) -> None:
        self._seed = seed

    def _signal(self, panel: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(self._seed)
        return pd.Series(rng.normal(size=len(panel)), index=panel.index)


def test_ic_recovers_known_signal() -> None:
    panel, prices = make_signal_panel(strength=0.02, noise=0.02, seed=1)
    ev = evaluate_factor(_SignalFactor(), panel, prices, primary_horizon=1)

    # A real predictive signal: positive, strongly significant IC at horizon 1.
    assert ev.primary_ic.mean_ic > 0.05
    assert ev.primary_ic.t_stat > 5.0
    # Spearman (rank) IC recovers it too.
    assert ev.spearman_ic[1].mean_ic > 0.05


def test_known_signal_has_monotonic_quantiles_and_decays() -> None:
    panel, prices = make_signal_panel(strength=0.02, noise=0.02, seed=2)
    ev = evaluate_factor(_SignalFactor(), panel, prices, primary_horizon=1)

    # Top quantile out-earns the bottom; the spread is positive.
    assert ev.quantile_returns[-1] > ev.quantile_returns[0]
    assert ev.long_short_spread > 0
    # The next-day signal decays as the horizon lengthens.
    assert ev.ic_decay[1] > ev.ic_decay[20]
    assert is_eligible(ev)


def test_ic_is_zero_on_pure_noise() -> None:
    panel, prices = make_signal_panel(strength=0.02, noise=0.02, seed=3)
    ev = evaluate_factor(_NoiseFactor(seed=7), panel, prices, primary_horizon=1)

    # Indistinguishable from zero: tiny mean IC, insignificant t-stat.
    assert abs(ev.primary_ic.mean_ic) < 0.03
    assert abs(ev.primary_ic.t_stat) < 1.96
    # A factor that predicts nothing is not eligible for the strategy builder.
    assert not is_eligible(ev)


def test_forward_returns_are_causal_targets() -> None:
    _panel, prices = make_signal_panel(n_days=20, n_securities=3, seed=4)
    from quantlab.factors import forward_returns

    fwd = forward_returns(prices, horizon=5)
    # A 5-day forward return needs 5 future bars, so the last 5 dates per security
    # have no target and are dropped.
    per_sec = fwd.groupby("security_id").size()
    assert (per_sec == 20 - 5).all()
