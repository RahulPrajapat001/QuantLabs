"""The factor protocol (PRD Module 6).

A **factor** takes a feature panel and returns cross-sectional **scores** per
``(date, security_id)`` — the standardised strength of a signal *relative to the
other securities in the index on that date*. Two rules are structural:

* Scores are computed **within the point-in-time universe** for each date: a
  security absent from the index on a date is neither scored nor used to
  standardise its peers (no look-ahead into future membership, no comparison
  against names it never traded alongside).
* Standardisation is **cross-sectional**, per date — a z-score across securities,
  never across time — so the score is comparable across dates and regimes.

:class:`FeatureFactor` covers the common case: take one feature column, optionally
flip its sign (``direction``), and z-score it cross-sectionally. The baseline
factors in :mod:`quantlab.factors.baselines` are all instances of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

#: A mapping from date to the set of ``security_id`` in the index on that date.
UniverseByDate = Mapping[date, set[str]]

#: Long score frame columns.
SCORE_COLUMNS = ("date", "security_id", "score")


@runtime_checkable
class Factor(Protocol):
    """Anything that turns a feature panel into cross-sectional scores."""

    name: str

    def compute(
        self, panel: pd.DataFrame, *, universe: UniverseByDate | None = None
    ) -> pd.DataFrame:
        """Return a long ``[date, security_id, score]`` frame."""
        ...


def _to_date(value: object) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def cross_sectional_zscore(
    frame: pd.DataFrame,
    signal_col: str,
    *,
    universe: UniverseByDate | None = None,
) -> pd.DataFrame:
    """Z-score ``signal_col`` across securities within each date's universe.

    ``frame`` must have ``date`` and ``security_id`` columns. Rows whose signal is
    missing, or whose security is outside the universe on that date, are dropped.
    Returns a long ``[date, security_id, score]`` frame.
    """

    pieces: list[pd.DataFrame] = []
    for date_val, rows in frame.groupby("date", sort=True):
        if universe is not None:
            members = universe.get(_to_date(date_val))
            if members is not None:
                rows = rows[rows["security_id"].isin(members)]
        signal = rows[signal_col]
        valid = signal.notna()
        signal = signal[valid]
        if signal.empty:
            continue
        std = signal.std(ddof=0)
        if not np.isfinite(std) or std == 0.0:
            continue  # a degenerate cross-section carries no rankable information
        z = (signal - signal.mean()) / std
        pieces.append(
            pd.DataFrame(
                {
                    "date": rows.loc[valid, "date"].to_numpy(),
                    "security_id": rows.loc[valid, "security_id"].to_numpy(),
                    "score": z.to_numpy(),
                }
            )
        )
    if not pieces:
        return pd.DataFrame({c: [] for c in SCORE_COLUMNS})
    return pd.concat(pieces, ignore_index=True)


class BaseFactor(ABC):
    """A factor that standardises a per-security signal cross-sectionally."""

    name: str

    @abstractmethod
    def _signal(self, panel: pd.DataFrame) -> pd.Series:
        """Return the raw (unstandardised) per-row signal aligned to ``panel``."""

    def compute(
        self, panel: pd.DataFrame, *, universe: UniverseByDate | None = None
    ) -> pd.DataFrame:
        work = panel[["date", "security_id"]].copy()
        work["_signal"] = self._signal(panel).to_numpy()
        return cross_sectional_zscore(work, "_signal", universe=universe)


class FeatureFactor(BaseFactor):
    """A factor built from a single feature column, optionally sign-flipped.

    ``direction=+1`` scores high-feature securities high; ``direction=-1`` inverts
    it (e.g. low-volatility investing scores *low* volatility high).
    """

    def __init__(self, name: str, feature: str, *, direction: int = 1) -> None:
        if direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")
        self.name = name
        self.feature = feature
        self.direction = direction

    def _signal(self, panel: pd.DataFrame) -> pd.Series:
        if self.feature not in panel.columns:
            raise KeyError(
                f"factor {self.name!r} needs feature column {self.feature!r}, "
                f"which is not in the panel"
            )
        return self.direction * panel[self.feature]
