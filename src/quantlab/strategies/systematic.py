"""Cross-sectional and schedule-driven strategies (PRD Module 7).

Unlike the technical strategies, momentum and mean reversion here are
**cross-sectional**: they rank the whole universe on each rebalance date and hold
the top (or bottom) slice. That is the form the academic literature documents and
the form H1 and H2 in ``docs/hypotheses.md`` pre-register, so it is the form
implemented.

Also here: dollar-cost averaging, and buy-and-hold — which exists as the engine's
own validation instrument. Under ``ZeroCostModel`` with fractional shares,
buy-and-hold must reproduce the benchmark's return to floating-point tolerance. If
it does not, the engine's accounting is wrong and nothing else it reports means
anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from quantlab.backtesting.data_handler import BarBatch, DataHandler
from quantlab.backtesting.events import Direction, Signal
from quantlab.strategies.base import Strategy


def _month_changed(previous: date | None, current: date) -> bool:
    return previous is None or (previous.year, previous.month) != (
        current.year,
        current.month,
    )


def _week_changed(previous: date | None, current: date) -> bool:
    if previous is None:
        return True
    return previous.isocalendar()[:2] != current.isocalendar()[:2]


@dataclass
class CrossSectionalMomentumStrategy(Strategy):
    """Hold the top decile by trailing return, skipping the most recent month.

    The 12-1 construction of H1: rank on the cumulative return over ``lookback``
    days *excluding* the last ``skip`` days. The skip matters — the most recent month
    carries short-horizon reversal, which works against the twelve-month signal, so
    including it measures the two effects netted against each other rather than
    momentum.

    Monthly rebalancing is not incidental either: at Indian statutory costs, weekly
    rebalancing of a decile portfolio pays roughly 0.22% of notional per round trip,
    which is most of what the signal earns.
    """

    lookback: int = 252
    skip: int = 21
    top_fraction: float = 0.1
    rebalance: str = "monthly"
    name: str = "momentum_12_1"

    _last_rebalance: date | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must lie in (0, 1]")
        if self.skip >= self.lookback:
            raise ValueError("skip must be shorter than lookback")

    @property
    def warmup(self) -> int:
        return self.lookback + 1

    def parameters(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "skip": self.skip,
            "top_fraction": self.top_fraction,
            "rebalance": self.rebalance,
        }

    def reset(self) -> None:
        self._last_rebalance = None

    def _due(self, today: date) -> bool:
        if self.rebalance == "monthly":
            return _month_changed(self._last_rebalance, today)
        if self.rebalance == "weekly":
            return _week_changed(self._last_rebalance, today)
        return True

    def on_bar(self, batch: BarBatch, handler: DataHandler) -> list[Signal]:
        if not self._due(batch.date):
            return []
        self._last_rebalance = batch.date

        scores: dict[str, float] = {}
        for security_id in batch.security_ids:
            closes = self._closes(handler, security_id, self.lookback + 1)
            if len(closes) < self.lookback + 1:
                continue
            # Return from (t - lookback) to (t - skip): the skip window is excluded.
            start = float(closes.iloc[0])
            end = (
                float(closes.iloc[-1 - self.skip])
                if self.skip
                else float(closes.iloc[-1])
            )
            if start <= 0:
                continue
            scores[security_id] = end / start - 1.0

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        keep = max(1, int(len(ranked) * self.top_fraction))
        winners = {security_id for security_id, _ in ranked[:keep]}

        signals = []
        for security_id, score in ranked:
            direction = Direction.LONG if security_id in winners else Direction.EXIT
            signals.append(self._signal(batch, security_id, direction, momentum=score))
        return signals


@dataclass
class CrossSectionalReversalStrategy(Strategy):
    """Hold the *worst* recent performers — short-horizon reversal (H2).

    Pre-registered as an expected negative result: the effect is real gross and is
    expected to die net of Indian transaction taxes, because harvesting it requires
    the turnover that those taxes are levied on. The interesting output is the cost
    level at which it crosses over, not a headline Sharpe.
    """

    lookback: int = 5
    bottom_fraction: float = 0.1
    rebalance: str = "weekly"
    name: str = "reversal_5d"

    _last_rebalance: date | None = field(default=None, init=False, repr=False)

    @property
    def warmup(self) -> int:
        return self.lookback + 1

    def parameters(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "bottom_fraction": self.bottom_fraction,
            "rebalance": self.rebalance,
        }

    def reset(self) -> None:
        self._last_rebalance = None

    def on_bar(self, batch: BarBatch, handler: DataHandler) -> list[Signal]:
        due = (
            _week_changed(self._last_rebalance, batch.date)
            if self.rebalance == "weekly"
            else True
        )
        if not due:
            return []
        self._last_rebalance = batch.date

        scores: dict[str, float] = {}
        for security_id in batch.security_ids:
            closes = self._closes(handler, security_id, self.lookback + 1)
            if len(closes) < self.lookback + 1:
                continue
            start = float(closes.iloc[0])
            if start <= 0:
                continue
            scores[security_id] = float(closes.iloc[-1]) / start - 1.0

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1])  # worst first
        keep = max(1, int(len(ranked) * self.bottom_fraction))
        losers = {security_id for security_id, _ in ranked[:keep]}
        return [
            self._signal(
                batch,
                security_id,
                Direction.LONG if security_id in losers else Direction.EXIT,
                trailing_return=score,
            )
            for security_id, score in ranked
        ]


@dataclass
class BuyAndHoldStrategy(Strategy):
    """Buy everything available on the first eligible bar, then never trade again.

    The engine's validation instrument. With :class:`ZeroCostModel`, zero slippage
    and fractional shares, its return must equal the underlying's return to
    floating-point tolerance — the test that proves the accounting is sound before
    any friction is layered on.
    """

    name: str = "buy_and_hold"
    _entered: bool = field(default=False, init=False, repr=False)

    def parameters(self) -> dict[str, Any]:
        return {}

    def reset(self) -> None:
        self._entered = False

    def on_bar(self, batch: BarBatch, handler: DataHandler) -> list[Signal]:
        if self._entered:
            return []
        self._entered = True
        return [
            self._signal(batch, security_id, Direction.LONG)
            for security_id in batch.security_ids
        ]


@dataclass
class DollarCostAveragingStrategy(Strategy):
    """Buy a fixed slice on a schedule, regardless of price.

    The honest baseline for a retail investor, and the one every active strategy
    should be compared against. It signals on the first bar of each period; the
    portfolio's sizing decides the amount, so "fixed rupee amount" is expressed as a
    position-weight cap rather than being hardcoded here.

    Its signals set ``accumulate=True``. That flag is what makes this dollar-cost
    *averaging* rather than buy-and-hold: without it the portfolio reads a repeated
    entry signal as a hold, the second contribution never happens, and the two
    strategies produce identical results — which is how this was caught.
    """

    interval: str = "monthly"
    name: str = "dca"

    _last_purchase: date | None = field(default=None, init=False, repr=False)

    def parameters(self) -> dict[str, Any]:
        return {"interval": self.interval}

    def reset(self) -> None:
        self._last_purchase = None

    def on_bar(self, batch: BarBatch, handler: DataHandler) -> list[Signal]:
        due = (
            _month_changed(self._last_purchase, batch.date)
            if self.interval == "monthly"
            else _week_changed(self._last_purchase, batch.date)
        )
        if not due:
            return []
        self._last_purchase = batch.date
        return [
            self._signal(batch, security_id, Direction.LONG, accumulate=True)
            for security_id in batch.security_ids
        ]
