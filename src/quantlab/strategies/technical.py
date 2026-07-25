"""Technical strategies: MA cross, RSI, MACD, Bollinger, breakout (PRD Module 7).

Each evaluates one security at a time against its own history. Indicator maths is
reused from :mod:`quantlab.feature_engineering._indicators` rather than
reimplemented — a second RSI in the codebase would eventually disagree with the
first, and then no result is reproducible.

All of these compute on ``adj_close``. On raw closes a 1:10 split reads as a 90%
crash, and every one of these strategies would fire on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantlab.backtesting.data_handler import BarBatch, DataHandler
from quantlab.backtesting.events import Direction
from quantlab.feature_engineering._indicators import ema, macd, rsi
from quantlab.strategies.base import SingleNameStrategy


@dataclass
class MovingAverageCrossStrategy(SingleNameStrategy):
    """Long when the fast average is above the slow one, flat otherwise.

    The canonical trend-following rule. Signals on the *state* rather than the
    crossing event, so a position that should be held is re-signalled each bar and
    the portfolio treats the repeat as a hold — which keeps the strategy's intent
    stateless and therefore auditable.
    """

    fast: int = 50
    slow: int = 200
    use_ema: bool = False
    name: str = "ma_cross"

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError(
                f"fast window ({self.fast}) must be shorter than slow ({self.slow})"
            )

    @property
    def warmup(self) -> int:
        return self.slow

    def parameters(self) -> dict[str, Any]:
        return {"fast": self.fast, "slow": self.slow, "use_ema": self.use_ema}

    def evaluate(
        self, security_id: str, batch: BarBatch, handler: DataHandler
    ) -> Direction | None:
        closes = self._closes(handler, security_id, self.slow + 1)
        if len(closes) < self.slow:
            return None
        if self.use_ema:
            fast_value = float(ema(closes, self.fast).iloc[-1])
            slow_value = float(ema(closes, self.slow).iloc[-1])
        else:
            fast_value = float(closes.tail(self.fast).mean())
            slow_value = float(closes.tail(self.slow).mean())
        return Direction.LONG if fast_value > slow_value else Direction.EXIT


@dataclass
class RSIStrategy(SingleNameStrategy):
    """Buy oversold, exit when the bounce is done.

    Mean reversion on Wilder's RSI. Exits at ``exit_level`` rather than at the
    overbought line, because waiting for 70 on a stock that bounced to 55 and rolled
    over gives the gain back — and a strategy that only exits on a strong signal has
    no exit at all in a downtrend.
    """

    period: int = 14
    oversold: float = 30.0
    exit_level: float = 55.0
    name: str = "rsi_reversion"

    @property
    def warmup(self) -> int:
        return self.period * 2

    def parameters(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "oversold": self.oversold,
            "exit_level": self.exit_level,
        }

    def evaluate(
        self, security_id: str, batch: BarBatch, handler: DataHandler
    ) -> Direction | None:
        closes = self._closes(handler, security_id, self.period * 3)
        if len(closes) < self.period + 1:
            return None
        value = float(rsi(closes, self.period).iloc[-1])
        if value != value:  # NaN during warmup
            return None
        if value <= self.oversold:
            return Direction.LONG
        if value >= self.exit_level:
            return Direction.EXIT
        return None


@dataclass
class MACDStrategy(SingleNameStrategy):
    """Long while the MACD histogram is positive.

    Momentum via the convergence of two EMAs. The histogram (MACD minus signal)
    turns before the lines cross, which is the point of using it.
    """

    fast: int = 12
    slow: int = 26
    signal: int = 9
    name: str = "macd"

    @property
    def warmup(self) -> int:
        return self.slow + self.signal

    def parameters(self) -> dict[str, Any]:
        return {"fast": self.fast, "slow": self.slow, "signal": self.signal}

    def evaluate(
        self, security_id: str, batch: BarBatch, handler: DataHandler
    ) -> Direction | None:
        closes = self._closes(handler, security_id, (self.slow + self.signal) * 3)
        if len(closes) < self.slow + self.signal:
            return None
        _line, _signal_line, histogram = macd(
            closes, fast=self.fast, slow=self.slow, signal=self.signal
        )
        value = float(histogram.iloc[-1])
        if value != value:
            return None
        return Direction.LONG if value > 0 else Direction.EXIT


@dataclass
class BollingerBandStrategy(SingleNameStrategy):
    """Buy the lower band, exit at the middle.

    Mean reversion on volatility-scaled bands. Exiting at the moving average rather
    than the upper band is the same reasoning as the RSI exit: the middle band is
    where the reversion thesis has played out, and holding for the upper band is a
    different (trend) bet the strategy did not make.
    """

    window: int = 20
    num_std: float = 2.0
    name: str = "bollinger_reversion"

    @property
    def warmup(self) -> int:
        return self.window

    def parameters(self) -> dict[str, Any]:
        return {"window": self.window, "num_std": self.num_std}

    def evaluate(
        self, security_id: str, batch: BarBatch, handler: DataHandler
    ) -> Direction | None:
        closes = self._closes(handler, security_id, self.window + 1)
        if len(closes) < self.window:
            return None
        window = closes.tail(self.window)
        middle = float(window.mean())
        sd = float(window.std(ddof=0))
        price = float(closes.iloc[-1])
        if sd == 0.0:
            return None
        if price <= middle - self.num_std * sd:
            return Direction.LONG
        if price >= middle:
            return Direction.EXIT
        return None


@dataclass
class BreakoutStrategy(SingleNameStrategy):
    """Long on a new N-day high, exit on a new M-day low — Donchian channels.

    The strategy most exposed to circuit limits, and deliberately included for that
    reason: a breakout to a new high is exactly the kind of move that locks a
    security at its price band, and the execution handler will reject the fill. The
    gap between this strategy's signal count and its fill count is the finding.
    """

    entry_window: int = 55
    exit_window: int = 20
    name: str = "breakout"

    @property
    def warmup(self) -> int:
        return max(self.entry_window, self.exit_window) + 1

    def parameters(self) -> dict[str, Any]:
        return {"entry_window": self.entry_window, "exit_window": self.exit_window}

    def evaluate(
        self, security_id: str, batch: BarBatch, handler: DataHandler
    ) -> Direction | None:
        lookback = max(self.entry_window, self.exit_window) + 1
        closes = self._closes(handler, security_id, lookback)
        if len(closes) < lookback:
            return None
        price = float(closes.iloc[-1])
        # Exclude today from the channel: comparing today's close to a window that
        # contains today makes a new high tautological.
        prior = closes.iloc[:-1]
        if price >= float(prior.tail(self.entry_window).max()):
            return Direction.LONG
        if price <= float(prior.tail(self.exit_window).min()):
            return Direction.EXIT
        return None
