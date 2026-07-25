"""Pairs trading on a cointegrated spread (PRD Module 7).

The one strategy here that requires a statistical precondition rather than just a
rule. Two securities whose prices are cointegrated have a mean-reverting spread; the
strategy trades deviations from that mean. Without cointegration the spread is a
random walk and "reversion" is a coin flip that occasionally looks like alpha.

**The hedge ratio is estimated on a rolling window ending at the current bar**, never
on the full sample. Fitting it once over all history and trading the resulting spread
is look-ahead: the ratio embeds knowledge of how the relationship evolved. That single
shortcut is responsible for a large share of published pairs-trading results.

Long-only constraint: the cash segment of Indian equities does not permit shorting for
a delivery strategy, so this trades the long leg only when the spread says that leg is
cheap. A market-neutral version would need the F&O segment and a different cost model,
which is out of scope (CLAUDE.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantlab.backtesting.data_handler import BarBatch, DataHandler
from quantlab.backtesting.events import Direction, Signal
from quantlab.strategies.base import Strategy


@dataclass
class PairsTradingStrategy(Strategy):
    """Trade the spread between two securities when it deviates from its mean.

    Parameters
    ----------
    security_a, security_b:
        The pair. ``security_a`` is the leg traded; ``security_b`` is the hedge
        reference.
    window:
        Rolling window for the hedge ratio and the spread's mean and deviation.
    entry_z, exit_z:
        Z-score thresholds. Entry at 2.0 and exit at 0.5 is conventional: exiting at
        zero means waiting for a mean that keeps moving, and most round trips never
        quite reach it.
    """

    security_a: str
    security_b: str
    window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    name: str = "pairs_trading"

    _in_position: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.entry_z <= self.exit_z:
            raise ValueError("entry_z must exceed exit_z, or entry and exit collide")
        if self.window < 20:
            raise ValueError(
                f"window {self.window} is too short to estimate a hedge ratio "
                "meaningfully; 20 observations is the floor"
            )

    @property
    def warmup(self) -> int:
        return self.window + 1

    def parameters(self) -> dict[str, Any]:
        return {
            "security_a": self.security_a,
            "security_b": self.security_b,
            "window": self.window,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
        }

    def reset(self) -> None:
        self._in_position = False

    def on_bar(self, batch: BarBatch, handler: DataHandler) -> list[Signal]:
        a = self._closes(handler, self.security_a, self.window)
        b = self._closes(handler, self.security_b, self.window)
        if len(a) < self.window or len(b) < self.window:
            return []
        n = min(len(a), len(b))
        a_values = a.tail(n).to_numpy(dtype=float)
        b_values = b.tail(n).to_numpy(dtype=float)

        z = self._zscore(a_values, b_values)
        if z is None:
            return []

        # Spread below its mean means leg A is cheap relative to the hedge.
        if not self._in_position and z <= -self.entry_z:
            self._in_position = True
            return [self._signal(batch, self.security_a, Direction.LONG, zscore=z)]
        if self._in_position and z >= -self.exit_z:
            self._in_position = False
            return [self._signal(batch, self.security_a, Direction.EXIT, zscore=z)]
        return []

    def _zscore(self, a: np.ndarray, b: np.ndarray) -> float | None:
        """Current spread z-score, with the hedge ratio fit on this window only."""

        if np.allclose(b, b[0]):
            return None
        # OLS through the origin-inclusive form: a = alpha + beta*b + spread.
        design = np.column_stack([np.ones_like(b), b])
        try:
            coefficients, *_ = np.linalg.lstsq(design, a, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate window
            return None
        spread = a - design @ coefficients
        sd = float(spread.std(ddof=1))
        if sd == 0.0 or not np.isfinite(sd):
            return None
        return float((spread[-1] - spread.mean()) / sd)
