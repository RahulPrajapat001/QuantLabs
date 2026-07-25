"""Rebalancing against the point-in-time universe (PRD Module 10).

**A security that left the index is exited at the rebalance date, not held silently.**

This is the survivorship trap in its subtlest form. An optimiser handed today's index
members and yesterday's weights will quietly keep any name that dropped out — it never
appears in the new candidate set, so it never gets a target weight, and a naive
implementation reads "no target" as "leave it alone". The position persists, and it
persists in a name that was removed from the index, which usually means it fell. The
backtest then holds the losers it should have sold and reports the result as a
portfolio return.

:func:`rebalance_plan` closes that hole by construction: every security in the current
book that is absent from the point-in-time universe is assigned an explicit target of
zero and recorded in :attr:`RebalancePlan.forced_exits`. The exits are visible in the
plan, counted in the turnover, and reported.

Delistings are the same problem with a harder edge — see
:mod:`quantlab.universe.delisting`, which resolves the last real traded price rather
than forward-filling one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import pandas as pd
import structlog
from sqlalchemy.orm import Session

from quantlab.portfolio.optimisers import (
    OptimisationError,
    OptimisationResult,
    equal_weight,
)
from quantlab.universe.membership import universe_as_of

log = structlog.get_logger(__name__)


class Optimiser(Protocol):
    """Anything that turns a return panel into weights.

    Kept as a protocol so the rebalancer does not care which optimiser is in use, and
    so a test can substitute a trivial one.
    """

    def __call__(self, returns: pd.DataFrame) -> OptimisationResult: ...


@dataclass(frozen=True, slots=True)
class RebalancePlan:
    """The trades that move the book to its new target weights."""

    as_of: date
    index_name: str
    target_weights: pd.Series
    current_weights: pd.Series
    #: Securities exited because they are no longer index members. The whole point
    #: of this module: these would otherwise be held silently.
    forced_exits: tuple[str, ...]
    #: Securities exited because the optimiser gave them no weight.
    optimiser_exits: tuple[str, ...]
    entries: tuple[str, ...]
    #: Universe members with no usable price history, so not investable today.
    unpriced: tuple[str, ...]
    result: OptimisationResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def trades(self) -> pd.Series:
        """Weight change per security. Negative is a sale."""

        assets = self.target_weights.index.union(self.current_weights.index)
        target = self.target_weights.reindex(assets).fillna(0.0)
        current = self.current_weights.reindex(assets).fillna(0.0)
        return (target - current).rename("weight_change")

    @property
    def turnover(self) -> float:
        """One-way turnover: half the sum of absolute weight changes."""

        return float(self.trades.abs().sum() / 2.0)

    @property
    def forced_exit_weight(self) -> float:
        """How much of the book is being sold purely because of index exits.

        A large number here on a historical rebalance is the measure of how much a
        current-constituents backtest would have quietly held.
        """

        if not self.forced_exits:
            return 0.0
        return float(
            self.current_weights.reindex(list(self.forced_exits))
            .fillna(0.0)
            .abs()
            .sum()
        )

    def render(self) -> str:
        lines = [
            f"Rebalance {self.index_name} as of {self.as_of.isoformat()}",
            f"  target holdings:   {int((self.target_weights.abs() > 0).sum())}",
            f"  turnover:          {self.turnover:7.2%}",
            f"  entries:           {len(self.entries)}",
            f"  optimiser exits:   {len(self.optimiser_exits)}",
            f"  forced exits:      {len(self.forced_exits)}"
            f"  ({self.forced_exit_weight:.2%} of the book)",
        ]
        if self.forced_exits:
            shown = ", ".join(self.forced_exits[:8])
            more = (
                f" (+{len(self.forced_exits) - 8} more)"
                if len(self.forced_exits) > 8
                else ""
            )
            lines += [
                "",
                "  Exited because they are no longer index members:",
                f"    {shown}{more}",
                "  These are sold at this rebalance. A backtest built on current "
                "constituents would have held them and never noticed.",
            ]
        if self.unpriced:
            lines += [
                "",
                f"  {len(self.unpriced)} universe members have no usable price "
                "history and were not investable at this date.",
            ]
        return "\n".join(lines)


def rebalance_plan(
    session: Session,
    index_name: str,
    as_of: date,
    *,
    current_weights: dict[str, float] | pd.Series | None = None,
    returns: pd.DataFrame | None = None,
    optimiser: Optimiser | None = None,
    min_observations: int = 60,
) -> RebalancePlan:
    """Build a rebalance plan against the point-in-time universe at ``as_of``.

    ``returns`` is a wide return panel (one column per security) whose history must
    end at or before ``as_of`` — the caller is responsible for that, and passing
    future returns here would be a look-ahead the optimiser cannot detect.

    ``optimiser`` defaults to equal weight over the investable members. Every member
    without ``min_observations`` of history is excluded and reported, rather than
    being handed to an optimiser that would estimate its variance from four days.
    """

    members = set(universe_as_of(session, index_name, as_of))
    if not members:
        raise OptimisationError(
            f"{index_name} has no point-in-time members as of {as_of.isoformat()}. "
            "Build the universe first — never fall back to current constituents."
        )

    current = pd.Series(current_weights or {}, dtype=float)
    held = {str(k) for k, v in current.items() if abs(float(v)) > 1e-12}

    # The rule this module exists for: anything held that is no longer a member is
    # exited here, explicitly.
    forced_exits = tuple(sorted(held - members))

    investable: list[str] = []
    unpriced: list[str] = []
    if returns is not None:
        for security_id in sorted(members):
            if security_id not in returns.columns:
                unpriced.append(security_id)
                continue
            if int(returns[security_id].notna().sum()) < min_observations:
                unpriced.append(security_id)
                continue
            investable.append(security_id)
    else:
        investable = sorted(members)

    if not investable:
        raise OptimisationError(
            f"none of {len(members)} members of {index_name} as of "
            f"{as_of.isoformat()} have {min_observations} observations of history; "
            "there is nothing investable to rebalance into"
        )

    result: OptimisationResult | None = None
    if optimiser is not None and returns is not None:
        result = optimiser(returns[investable])
        target = result.weights
    else:
        result = equal_weight(investable)
        target = result.weights

    target = target[target.abs() > 1e-12]
    optimiser_exits = tuple(sorted((held & members) - set(target.index)))
    entries = tuple(sorted(set(target.index) - held))

    if forced_exits:
        log.info(
            "rebalance.forced_exits",
            index=index_name,
            as_of=as_of.isoformat(),
            count=len(forced_exits),
            securities=list(forced_exits[:20]),
        )

    return RebalancePlan(
        as_of=as_of,
        index_name=index_name,
        target_weights=target,
        current_weights=current,
        forced_exits=forced_exits,
        optimiser_exits=optimiser_exits,
        entries=entries,
        unpriced=tuple(unpriced),
        result=result,
        diagnostics={
            "members": len(members),
            "investable": len(investable),
            "min_observations": min_observations,
        },
    )


def rebalance_schedule(
    start: date, end: date, *, frequency: str = "monthly"
) -> list[date]:
    """Rebalance dates between ``start`` and ``end``.

    ``monthly`` | ``quarterly`` | ``annual``. Frequency is the dominant driver of
    cost: at Indian statutory rates a monthly rebalance of a decile portfolio pays
    roughly 12 times the round-trip cost of an annual one, which is usually more
    than the optimiser's theoretical improvement.
    """

    offsets = {"monthly": "MS", "quarterly": "QS", "annual": "YS"}
    if frequency not in offsets:
        raise OptimisationError(
            f"unknown frequency {frequency!r}; use {' | '.join(offsets)}"
        )
    dates = pd.date_range(start=start, end=end, freq=offsets[frequency])
    return [pd.Timestamp(d).date() for d in dates]


@dataclass(frozen=True, slots=True)
class RebalanceHistory:
    """A sequence of plans, for measuring turnover and index-exit drag over time."""

    plans: tuple[RebalancePlan, ...]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "as_of": plan.as_of,
                    "holdings": int((plan.target_weights.abs() > 0).sum()),
                    "turnover": plan.turnover,
                    "entries": len(plan.entries),
                    "forced_exits": len(plan.forced_exits),
                    "forced_exit_weight": plan.forced_exit_weight,
                    "unpriced": len(plan.unpriced),
                }
                for plan in self.plans
            ]
        )

    @property
    def total_forced_exits(self) -> int:
        return sum(len(plan.forced_exits) for plan in self.plans)

    @property
    def mean_turnover(self) -> float:
        frame = self.frame()
        return float(frame["turnover"].mean()) if not frame.empty else float("nan")

    def render(self) -> str:
        frame = self.frame()
        lines = [
            f"Rebalance history ({len(self.plans)} dates)",
            f"  mean turnover per rebalance: {self.mean_turnover:7.2%}",
            f"  total forced index exits:    {self.total_forced_exits:7d}",
        ]
        if not frame.empty:
            lines.append(
                f"  mean weight forced out:      "
                f"{float(frame['forced_exit_weight'].mean()):7.2%}"
            )
        if self.total_forced_exits:
            lines += [
                "",
                "  Every forced exit above is a position a current-constituents "
                "backtest would have continued to hold. That difference is the "
                "survivorship bias, and it is measurable rather than hypothetical.",
            ]
        return "\n".join(lines)
