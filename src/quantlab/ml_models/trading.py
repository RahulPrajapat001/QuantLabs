"""Turning out-of-sample predictions into a return series, honestly.

A model's MAE says nothing about whether it makes money, so PRD Module 11 also
requires a *traded* comparison against buy-and-hold and a naive momentum rule.
This module builds the return series that comparison runs on: on each rebalance
date the cross-section is ranked by prediction, the top slice is held long (and
the bottom slice shorted, for the long-short rule), and the realised forward
returns are weighted up.

Three things this is deliberately **not**:

* It is **not the backtester**. There is no order book, no circuit-limit
  rejection, no volume participation cap, no T+1 settlement — all of which
  :mod:`quantlab.backtesting` models and all of which move the answer. Costs here
  are a single turnover-proportional charge derived from
  :meth:`IndianEquityCostModel.round_trip_cost_fraction`. It is the right
  instrument for *ranking model configurations against each other and against
  baselines under identical assumptions*, and the wrong instrument for a headline
  Sharpe. A strategy that survives here goes to the event-driven engine next.
* It is **not gross by default**. Rebalancing a decile portfolio daily at Indian
  statutory rates costs roughly 0.22% of notional per round trip; a gross Sharpe
  computed over daily rebalancing is a number about a market that does not charge
  STT. Pass ``cost_per_unit_traded=0.0`` to see the gross figure — it is reported
  alongside, never instead.
* It does **not overlap its return observations**. With a horizon of *h* days,
  rebalancing every day would produce a series whose consecutive observations
  share *h-1* days of price movement; its standard deviation is understated and
  its Sharpe correspondingly inflated. Rebalancing therefore happens every *h*
  dates and the annualisation factor is ``252 / h``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "TradedResult",
    "TradingRule",
    "default_one_way_cost",
    "trade_predictions",
]


def default_one_way_cost() -> float:
    """Cost of trading one unit of notional one way, under the Indian model.

    Half the round trip, because ``sum |Δw|`` already counts both legs of a
    switch: selling 10% of the book out of one name and into another moves 20%
    of notional and pays one one-way charge on each 10%.
    """

    from quantlab.backtesting.costs import IndianEquityCostModel

    return IndianEquityCostModel().round_trip_cost_fraction() / 2.0


@dataclass(frozen=True, slots=True)
class TradingRule:
    """How a cross-section of predictions becomes a set of weights."""

    #: ``"long_short"`` holds the top slice and shorts the bottom one at equal
    #: gross exposure; ``"long_only"`` holds the top slice and nothing else.
    mode: str = "long_short"
    #: Fraction of the cross-section in each leg. 0.2 is a quintile.
    quantile: float = 0.2
    #: Cost per unit of one-way traded notional. Defaults to the Indian
    #: statutory model; set 0.0 for the gross comparison.
    cost_per_unit_traded: float | None = None
    #: Name recorded in ``research_log``. Says plainly that this is a
    #: turnover approximation, not the itemised ledger.
    cost_model: str = "IndianEquityCostModel(turnover-approx)"

    def __post_init__(self) -> None:
        if self.mode not in {"long_short", "long_only"}:
            raise ValueError(
                f"mode must be 'long_short' or 'long_only', got {self.mode!r}"
            )
        if not 0.0 < self.quantile <= 0.5:
            raise ValueError(
                f"quantile must lie in (0, 0.5], got {self.quantile}: both legs of "
                "a long-short book have to fit in the cross-section"
            )

    @property
    def one_way_cost(self) -> float:
        if self.cost_per_unit_traded is not None:
            return float(self.cost_per_unit_traded)
        return default_one_way_cost()

    @property
    def effective_cost_model(self) -> str:
        return "ZeroCostModel" if self.one_way_cost == 0.0 else self.cost_model

    def describe(self) -> str:
        return (
            f"{self.mode} top/bottom {self.quantile:.0%}, "
            f"cost {self.one_way_cost * 1e4:.1f} bps per unit traded "
            f"({self.effective_cost_model})"
        )

    def parameters(self) -> dict[str, Any]:
        return {
            "trading_mode": self.mode,
            "trading_quantile": self.quantile,
            "one_way_cost": self.one_way_cost,
        }


@dataclass(frozen=True, slots=True)
class TradedResult:
    """The return series a set of predictions produced, gross and net.

    Exposes :attr:`annualised_sharpe` as a plain float on purpose — it is the
    quantity fed to the deflation and to the baseline comparison. It is not a
    reportable result: the only path to a reportable Sharpe runs through
    :class:`~quantlab.research.reporting.SharpeReport`, which cannot be built
    without a trials count read from ``research_log``.
    """

    label: str
    dates: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    #: ``sum |Δw|`` per rebalance — the notional actually traded.
    traded_notional: np.ndarray
    periods_per_year: int
    rule: TradingRule
    n_names: np.ndarray

    def __len__(self) -> int:
        return int(self.net_returns.size)

    @property
    def cost_model(self) -> str:
        return self.rule.effective_cost_model

    @property
    def total_return(self) -> float:
        return float(np.prod(1.0 + self.net_returns) - 1.0)

    @property
    def annualised_return(self) -> float:
        n = len(self)
        if n == 0:
            return float("nan")
        growth = float(np.prod(1.0 + self.net_returns))
        if growth <= 0.0:
            return -1.0
        return float(growth ** (self.periods_per_year / n) - 1.0)

    @property
    def volatility(self) -> float:
        if len(self) < 2:
            return float("nan")
        return float(self.net_returns.std(ddof=1) * np.sqrt(self.periods_per_year))

    @property
    def annualised_sharpe(self) -> float:
        """Annualised Sharpe of the **net** series. An input, not a result."""

        if len(self) < 2:
            return float("nan")
        sd = float(self.net_returns.std(ddof=1))
        if sd == 0.0:
            return 0.0
        mean = float(self.net_returns.mean())
        return mean / sd * float(np.sqrt(self.periods_per_year))

    @property
    def gross_annualised_sharpe(self) -> float:
        if len(self) < 2:
            return float("nan")
        sd = float(self.gross_returns.std(ddof=1))
        if sd == 0.0:
            return 0.0
        return float(self.gross_returns.mean() / sd * np.sqrt(self.periods_per_year))

    @property
    def max_drawdown(self) -> float:
        if len(self) == 0:
            return float("nan")
        equity = np.cumprod(1.0 + self.net_returns)
        peak = np.maximum.accumulate(equity)
        return float((equity / peak - 1.0).min())

    @property
    def hit_rate(self) -> float:
        if len(self) == 0:
            return float("nan")
        return float(np.count_nonzero(self.net_returns > 0.0) / len(self))

    @property
    def mean_turnover(self) -> float:
        """Average notional traded per rebalance, as a fraction of the book."""

        if self.traded_notional.size == 0:
            return float("nan")
        return float(self.traded_notional.mean())

    @property
    def total_cost(self) -> float:
        """Total cost paid, as a fraction of notional, over the whole series."""

        return float((self.gross_returns - self.net_returns).sum())

    @property
    def start(self) -> date:
        return pd.Timestamp(self.dates.min()).date()

    @property
    def end(self) -> date:
        return pd.Timestamp(self.dates.max()).date()

    def as_dict(self) -> Mapping[str, Any]:
        """Metrics for the log. Always stored **nested**, never at top level.

        ``annualised_sharpe`` is a Sharpe-like key, and
        :func:`quantlab.research.reporting.assert_no_bare_sharpe` scans the top
        level of a metrics mapping. Nesting is not evasion — it is the accurate
        statement that this is not the run's own Sharpe. Flattening it would make
        a baseline's number indistinguishable from the model's.
        """

        return {
            "label": self.label,
            "n_periods": len(self),
            "annualised_sharpe": self.annualised_sharpe,
            "gross_annualised_sharpe": self.gross_annualised_sharpe,
            "annualised_return": self.annualised_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "mean_turnover": self.mean_turnover,
            "total_cost": self.total_cost,
            "cost_model": self.cost_model,
            "periods_per_year": self.periods_per_year,
        }

    def render(self) -> str:
        return "\n".join(
            [
                f"{self.label} — {self.rule.describe()}",
                f"  periods:            {len(self):8d} "
                f"({self.start.isoformat()}..{self.end.isoformat()}, "
                f"{self.periods_per_year}/yr)",
                f"  annualised return:  {self.annualised_return:8.2%}",
                f"  volatility:         {self.volatility:8.2%}",
                f"  Sharpe (net):       {self.annualised_sharpe:8.3f}",
                f"  Sharpe (gross):     {self.gross_annualised_sharpe:8.3f}",
                f"  max drawdown:       {self.max_drawdown:8.2%}",
                f"  mean turnover:      {self.mean_turnover:8.2%} per rebalance",
                f"  total cost paid:    {self.total_cost:8.2%} of notional",
            ]
        )


def _weights_for_date(predictions: np.ndarray, rule: TradingRule) -> np.ndarray:
    """Equal-weight the top (and bottom) slice of one cross-section."""

    n = predictions.size
    keep = max(1, int(n * rule.quantile))
    order = np.argsort(-predictions, kind="stable")
    weights = np.zeros(n, dtype=float)

    if rule.mode == "long_only":
        weights[order[:keep]] = 1.0 / keep
        return weights

    # Long-short at unit gross exposure: half the book each side, so the series
    # is comparable to the long-only and buy-and-hold ones without rescaling.
    if 2 * keep > n:
        keep = max(1, n // 2)
    weights[order[:keep]] = 0.5 / keep
    weights[order[-keep:]] = -0.5 / keep
    return weights


def trade_predictions(
    frame: pd.DataFrame,
    *,
    label: str,
    rule: TradingRule | None = None,
    horizon: int = 1,
    periods_per_year: int = 252,
    equal_weight_all: bool = False,
) -> TradedResult:
    """Build the return series implied by ``frame``'s per-date predictions.

    ``frame`` is long ``[date, security_id, prediction, forward_return]`` where
    ``forward_return`` is the realised return over the same ``horizon`` the
    prediction refers to. Rebalancing happens every ``horizon`` dates so the
    return observations do not overlap.

    ``equal_weight_all`` ignores the predictions and holds the whole
    cross-section equally — that is buy-and-hold, and routing it through the same
    function is what makes the baseline comparison mechanically identical rather
    than merely similar.
    """

    trading_rule = rule or TradingRule()
    required = {"date", "security_id", "prediction", "forward_return"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"prediction frame is missing column(s): {missing}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["date", "security_id"])

    # Non-overlapping observations: hold each book for the full label horizon, so
    # consecutive returns never share a day of price movement.
    cross_sections = list(work.groupby("date", sort=True))[::horizon]

    dates: list[Any] = []
    gross: list[float] = []
    traded: list[float] = []
    names: list[int] = []
    previous: dict[str, float] = {}

    for when, rows in cross_sections:
        if rows.empty:  # pragma: no cover - defensive
            continue
        securities = rows["security_id"].to_numpy()
        forward = rows["forward_return"].to_numpy(dtype=float)

        if equal_weight_all:
            weights = np.full(securities.size, 1.0 / securities.size)
        else:
            weights = _weights_for_date(
                rows["prediction"].to_numpy(dtype=float), trading_rule
            )

        current = {s: float(w) for s, w in zip(securities, weights, strict=True) if w}
        # Notional traded includes closing out names that left the book entirely.
        moved = sum(
            abs(current.get(s, 0.0) - previous.get(s, 0.0))
            for s in set(current) | set(previous)
        )

        dates.append(when)
        gross.append(float(np.dot(weights, forward)))
        traded.append(float(moved))
        names.append(int(securities.size))
        previous = current

    if not dates:
        raise ValueError("no rebalance dates produced a tradeable cross-section")

    gross_arr = np.asarray(gross, dtype=float)
    traded_arr = np.asarray(traded, dtype=float)
    net_arr = gross_arr - traded_arr * trading_rule.one_way_cost

    return TradedResult(
        label=label,
        dates=np.asarray(dates),
        gross_returns=gross_arr,
        net_returns=net_arr,
        traded_notional=traded_arr,
        periods_per_year=max(1, round(periods_per_year / horizon)),
        rule=trading_rule,
        n_names=np.asarray(names, dtype=int),
    )
