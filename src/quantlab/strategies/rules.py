"""Custom-rule strategy interface (PRD Module 7).

Composes a strategy from callables: a buy condition, a sell condition, an exit, a
stop loss, a take profit, and a position-size hint. Each receives a
:class:`RuleContext` — the current bar plus cursor-bounded history — and returns a
decision. The rules cannot reach past the current bar because the context is built
from the same :class:`~quantlab.backtesting.data_handler.DataHandler` guarantees
everything else uses.

Stop loss and take profit are evaluated **against the entry price this strategy
recorded when the position opened**, and they emit exit *signals* rather than
executing. That matters: a stop that fires intrabar at exactly the stop price is
fiction — the fill happens at the next open like any other order, and in a gap it
happens well below the stop. Modelling stops as guaranteed intrabar exits at the
trigger price is one of the most common ways a backtest manufactures returns it
could not have earned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantlab.backtesting.data_handler import BarBatch, DataHandler
from quantlab.backtesting.events import Direction, Signal
from quantlab.strategies.base import Strategy


@dataclass(frozen=True, slots=True)
class RuleContext:
    """What a rule is allowed to see: this bar, and history up to it."""

    security_id: str
    bar: pd.Series
    history: pd.DataFrame
    #: Entry price if a position is open, else ``None``.
    entry_price: float | None

    @property
    def close(self) -> float:
        return float(self.bar["close"])

    @property
    def adj_close(self) -> float:
        return float(self.bar["adj_close"])

    @property
    def is_open(self) -> bool:
        return self.entry_price is not None

    @property
    def unrealised_return(self) -> float:
        """Return since entry, or 0.0 when flat."""

        if self.entry_price is None or self.entry_price <= 0:
            return 0.0
        return self.close / self.entry_price - 1.0

    def closes(self, lookback: int | None = None) -> pd.Series:
        """Adjusted closes, oldest first, optionally the last ``lookback`` bars."""

        series = self.history["adj_close"]
        return series.tail(lookback) if lookback else series


#: A rule maps a context to a boolean.
Rule = Callable[[RuleContext], bool]
#: A sizing hint maps a context to a conviction in [0, 1].
SizeRule = Callable[[RuleContext], float]


@dataclass
class CustomRuleStrategy(Strategy):
    """A strategy assembled from callables.

    Parameters
    ----------
    buy, sell:
        Entry and discretionary-exit conditions.
    exit_rule:
        An additional exit (a time stop, say), evaluated alongside ``sell``.
    stop_loss, take_profit:
        Fractions, e.g. ``0.08`` for an 8% stop. Evaluated on the close against the
        recorded entry price, emitting an exit signal filled at the next open.
    position_size:
        Returns a conviction in ``[0, 1]``, which the portfolio's
        ``SIGNAL_WEIGHTED`` sizing consumes. It is a *hint*, not an instruction: a
        strategy still cannot set its own position size.
    lookback:
        History depth handed to the rules.
    """

    buy: Rule
    sell: Rule | None = None
    exit_rule: Rule | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size: SizeRule | None = None
    lookback: int = 252
    name: str = "custom_rule"

    #: Entry prices, so stops and targets have a reference. Cleared on reset.
    _entries: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("stop_loss", self.stop_loss),
            ("take_profit", self.take_profit),
        ):
            if value is not None and not 0.0 < value < 1.0:
                raise ValueError(
                    f"{label} must be a fraction in (0, 1), got {value}; 0.08 means 8%"
                )

    @property
    def warmup(self) -> int:
        return min(self.lookback, 20)

    def parameters(self) -> dict[str, Any]:
        return {
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "lookback": self.lookback,
            "has_sell_rule": self.sell is not None,
            "has_exit_rule": self.exit_rule is not None,
            "has_size_rule": self.position_size is not None,
        }

    def reset(self) -> None:
        self._entries.clear()

    def on_bar(self, batch: BarBatch, handler: DataHandler) -> list[Signal]:
        signals: list[Signal] = []
        for row in batch.bars.itertuples(index=False):
            security_id = str(row.security_id)
            history = handler.history(security_id, lookback=self.lookback)
            if history.empty:
                continue
            context = RuleContext(
                security_id=security_id,
                bar=history.iloc[-1],
                history=history,
                entry_price=self._entries.get(security_id),
            )

            if context.is_open:
                reason = self._exit_reason(context)
                if reason is not None:
                    self._entries.pop(security_id, None)
                    signals.append(
                        self._signal(
                            batch, security_id, Direction.EXIT, exit_reason=reason
                        )
                    )
                continue

            if self._safe(self.buy, context):
                self._entries[security_id] = context.close
                strength = 1.0
                if self.position_size is not None:
                    strength = min(1.0, max(0.0, float(self.position_size(context))))
                signals.append(
                    self._signal(batch, security_id, Direction.LONG, strength=strength)
                )
        return signals

    def _exit_reason(self, context: RuleContext) -> str | None:
        """Why this position should close, or ``None`` to hold.

        Stops are checked before discretionary rules: risk control is not
        negotiable against a signal that says hold.
        """

        move = context.unrealised_return
        if self.stop_loss is not None and move <= -self.stop_loss:
            return "stop_loss"
        if self.take_profit is not None and move >= self.take_profit:
            return "take_profit"
        if self.sell is not None and self._safe(self.sell, context):
            return "sell_rule"
        if self.exit_rule is not None and self._safe(self.exit_rule, context):
            return "exit_rule"
        return None

    @staticmethod
    def _safe(rule: Rule, context: RuleContext) -> bool:
        """Evaluate a user rule, letting look-ahead errors through.

        A rule that raises :class:`~quantlab.backtesting.data_handler.LookAheadError`
        must not be swallowed — that is a bug in the rule and the run should fail
        loudly. Anything else (a short history, a division by zero on a flat series)
        is treated as "no signal", because a user rule crashing on bar 3 of 2,000
        should not abandon the backtest.
        """

        from quantlab.backtesting.data_handler import LookAheadError

        try:
            return bool(rule(context))
        except LookAheadError:
            raise
        except (IndexError, KeyError, ValueError, ZeroDivisionError):
            return False
