"""Baseline factors (PRD Module 6).

Four well-known cross-sectional signals, each a :class:`FeatureFactor` over a
registered feature. They are the reference alphas the evaluation suite and the
IC gate are exercised against — not novel research, but the honest baselines a
new factor must beat.

* **12-1 momentum** — trailing-year return skipping the last month; winners keep
  winning at the 1-12 month horizon. Direction ``+1``.
* **Short-term reversal** — last week's return, sign-flipped; recent winners tend
  to give back over days. Direction ``-1`` on ``return_5d``.
* **Low volatility** — trailing volatility, sign-flipped; the low-vol anomaly pays
  calmer names. Direction ``-1`` on ``volatility_20d``.
* **Volume** — recent volume shock, sign-flipped; attention/turnover spikes tend
  to precede weaker returns. Direction ``-1`` on ``volume_shock_20``.
"""

from __future__ import annotations

from .protocol import FeatureFactor


def momentum_12_1_factor() -> FeatureFactor:
    return FeatureFactor("momentum_12_1", "momentum_12_1", direction=1)


def short_term_reversal_factor() -> FeatureFactor:
    return FeatureFactor("short_term_reversal", "return_5d", direction=-1)


def low_volatility_factor() -> FeatureFactor:
    return FeatureFactor("low_volatility", "volatility_20d", direction=-1)


def volume_factor() -> FeatureFactor:
    return FeatureFactor("volume", "volume_shock_20", direction=-1)


def baseline_factors() -> list[FeatureFactor]:
    """The four baseline factors as a list, in a stable order."""

    return [
        momentum_12_1_factor(),
        short_term_reversal_factor(),
        low_volatility_factor(),
        volume_factor(),
    ]
