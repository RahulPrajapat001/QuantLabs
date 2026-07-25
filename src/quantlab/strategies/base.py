"""``Strategy`` — consumes bars, emits signals. Nothing else.

A strategy sees the current bar and whatever history the
:class:`~quantlab.backtesting.data_handler.DataHandler` will show it, and returns
:class:`~quantlab.backtesting.events.Signal` objects. It does **not** decide
position size, does not know the portfolio's capital, and cannot place an order.
That separation is what makes the look-ahead guarantee meaningful: the only window
onto the data is the handler, and the handler refuses to serve anything past the
current bar.

Two obligations on implementors:

* Declare :attr:`warmup` — the number of bars of history the signal needs. The
  engine will not call :meth:`on_bar` before that many bars have been served, so a
  strategy never emits a signal computed from a half-filled indicator. Getting this
  wrong is a subtle source of bias: an RSI computed on 3 of its 14 days is not a
  noisy RSI, it is a different and much more volatile signal.
* Declare :meth:`parameters` — every tunable value. These are written to
  ``research_log``, and they are what makes a run reproducible and what the trials
  count is counting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from quantlab.backtesting.data_handler import BarBatch, DataHandler
from quantlab.backtesting.events import Direction, Signal


class Strategy(ABC):
    """Base class for every strategy."""

    #: Stable name, written to ``research_log.strategy_name``.
    name: str = "abstract"

    @property
    def warmup(self) -> int:
        """Bars of history required before this strategy may signal."""

        return 0

    @abstractmethod
    def on_bar(self, batch: BarBatch, handler: DataHandler) -> list[Signal]:
        """Return signals for this bar. May be empty — usually is."""

    def parameters(self) -> dict[str, Any]:
        """Every tunable parameter, for the research log."""

        return {}

    def reset(self) -> None:  # noqa: B027 - intentionally optional
        """Clear any internal state between runs.

        Not abstract on purpose: most strategies are stateless functions of the
        visible history and have nothing to reset. Forcing every one of them to
        write an empty override would add noise and make the stateful ones — which
        genuinely must implement this — harder to spot.
        """

    # --- Helpers for implementors ---

    def _signal(
        self,
        batch: BarBatch,
        security_id: str,
        direction: Direction,
        *,
        strength: float = 1.0,
        accumulate: bool = False,
        **metadata: Any,
    ) -> Signal:
        return Signal(
            date=batch.date,
            security_id=security_id,
            direction=direction,
            strategy_name=self.name,
            strength=strength,
            accumulate=accumulate,
            metadata=metadata,
        )

    @staticmethod
    def _closes(handler: DataHandler, security_id: str, lookback: int) -> pd.Series:
        """Adjusted closes for one security, newest last, cursor-bounded.

        Uses ``adj_close``: a signal computed on raw closes treats a 1:10 split as a
        90% crash and will fire on it. Execution uses the raw price; signals use the
        adjusted one.
        """

        history = handler.history(security_id, lookback=lookback, columns=["adj_close"])
        return history["adj_close"].reset_index(drop=True)

    @staticmethod
    def _ohlcv(handler: DataHandler, security_id: str, lookback: int) -> pd.DataFrame:
        """Full adjusted-price history for one security, cursor-bounded."""

        return handler.history(security_id, lookback=lookback)


class SingleNameStrategy(Strategy):
    """Convenience base for strategies that evaluate each security independently.

    Implement :meth:`evaluate` for one security; the loop over the cross-section and
    the warmup check are handled here. Most technical strategies are of this shape;
    cross-sectional ones (momentum ranking, pairs) are not and should subclass
    :class:`Strategy` directly.
    """

    def on_bar(self, batch: BarBatch, handler: DataHandler) -> list[Signal]:
        signals: list[Signal] = []
        for security_id in batch.security_ids:
            direction = self.evaluate(security_id, batch, handler)
            if direction is not None:
                signals.append(self._signal(batch, security_id, direction))
        return signals

    @abstractmethod
    def evaluate(
        self, security_id: str, batch: BarBatch, handler: DataHandler
    ) -> Direction | None:
        """Return a direction for this security, or ``None`` for no opinion."""
