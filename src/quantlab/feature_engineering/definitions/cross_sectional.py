"""Cross-sectional features: z-score and rank across the point-in-time universe.

Each function receives the panel rows for a **single date** (already restricted
to that date's index constituents by the engine) and returns a per-security
series. Because a cross-section only ever touches same-date rows, these features
carry no time-dimension look-ahead; the engine's universe restriction ensures a
security is never ranked against names it was not in the index alongside.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from quantlab.feature_engineering.registry import FeatureKind, add_feature


def _make_zscore(base: str) -> Callable[[pd.DataFrame], pd.Series[float]]:
    def _zscore(df: pd.DataFrame, base: str = base) -> pd.Series[float]:
        x = df[base]
        std = x.std(ddof=0)
        if not np.isfinite(std) or std == 0.0:
            return pd.Series(np.nan, index=x.index)
        return ((x - x.mean()) / std).replace([np.inf, -np.inf], np.nan)

    return _zscore


def _make_rank(base: str) -> Callable[[pd.DataFrame], pd.Series[float]]:
    def _rank(df: pd.DataFrame, base: str = base) -> pd.Series[float]:
        # Percentile rank across securities on this date, in (0, 1].
        return df[base].rank(pct=True)

    return _rank


# Z-scores for a subset; ranks for a subset — together six cross-sectional features.
for _suffix, _base in (
    ("momentum_20d", "momentum_20d"),
    ("return_20d", "return_20d"),
    ("volatility_20d", "volatility_20d"),
):
    add_feature(
        f"xs_zscore_{_suffix}",
        _make_zscore(_base),
        dependencies=[_base],
        lookback=0,
        category="cross_sectional",
        kind=FeatureKind.CROSS_SECTIONAL,
        description=(
            f"Cross-sectional z-score of {_base} across the universe on each date. "
            "Lookback is 0: it needs no history of its own, only its input's."
        ),
    )

for _suffix, _base in (
    ("momentum_20d", "momentum_20d"),
    ("return_20d", "return_20d"),
    ("rsi_14", "rsi_14"),
):
    add_feature(
        f"xs_rank_{_suffix}",
        _make_rank(_base),
        dependencies=[_base],
        lookback=0,
        category="cross_sectional",
        kind=FeatureKind.CROSS_SECTIONAL,
        description=(
            f"Cross-sectional percentile rank of {_base} across the universe on "
            "each date, in (0, 1]."
        ),
    )
