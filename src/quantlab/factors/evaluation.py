"""Factor evaluation: IC, decay, quantiles, turnover, fitness (PRD Module 6).

The evaluation answers one question — *does this factor's cross-sectional score
predict forward returns?* — and answers it honestly:

* **Information coefficient (IC).** Per date, the cross-sectional correlation
  between the factor score and the forward return. Both Pearson and Spearman
  (rank) IC. The IC time series carries a mean, a **Newey-West** t-statistic (the
  overlapping forward returns induce autocorrelation the HAC standard error
  corrects for), and its information ratio.
* **IC decay.** Mean IC at forward horizons 1 / 5 / 10 / 20 days — how fast the
  signal fades.
* **Quantile buckets.** Mean forward return per score quantile and the top-minus-
  bottom **long-short spread**.
* **Turnover** of the top quantile and the factor's **autocorrelation** — a signal
  that predicts but churns every day is expensive to trade.
* **Fitness.** A single quality number (the IC information ratio) the gate reads.

Evaluation runs on the **train split**; :func:`quantlab.factors.split.guard_holdout`
raises if handed locked holdout data without the explicit unlock.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .protocol import Factor
from .split import guard_holdout

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 20)
DEFAULT_QUANTILES = 5


@dataclass(frozen=True, slots=True)
class ICStats:
    """Information-coefficient statistics for one horizon and method."""

    horizon: int
    method: str
    mean_ic: float
    std_ic: float
    ic_ir: float
    t_stat: float
    p_value: float
    newey_west_se: float
    n_periods: int
    interpretation: str


@dataclass(frozen=True, slots=True)
class FactorEvaluation:
    """The full evaluation of one factor over the train split."""

    factor_name: str
    primary_horizon: int
    horizons: tuple[int, ...]
    pearson_ic: Mapping[int, ICStats]
    spearman_ic: Mapping[int, ICStats]
    ic_decay: Mapping[int, float]
    quantile_returns: tuple[float, ...]
    long_short_spread: float
    turnover: float
    factor_autocorrelation: float
    fitness: float
    n_quantiles: int

    @property
    def primary_ic(self) -> ICStats:
        return self.pearson_ic[self.primary_horizon]

    @property
    def is_eligible(self) -> bool:
        """Whether the factor clears the IC gate (see :mod:`quantlab.factors.gate`)."""

        from .gate import is_eligible

        return is_eligible(self)


# --- forward returns and IC --------------------------------------------------


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return a long ``[date, security_id, fwd_ret]`` frame for ``horizon`` days.

    ``prices`` is long ``[date, security_id, close]``. The forward return at date
    *t* is ``close_{t+h}/close_t - 1`` per security — it uses future prices (it is
    the prediction *target*, not a feature), computed per security so horizons
    never bleed across securities.
    """

    out = prices.sort_values(["security_id", "date"]).copy()
    grouped = out.groupby("security_id", sort=False)["close"]
    out["fwd_ret"] = grouped.shift(-horizon) / out["close"] - 1.0
    return out[["date", "security_id", "fwd_ret"]].dropna(subset=["fwd_ret"])


def _ic_series(
    scores: pd.DataFrame,
    fwd: pd.DataFrame,
    *,
    method: Literal["pearson", "spearman"],
    min_names: int = 5,
) -> pd.Series:
    """Per-date cross-sectional correlation of score vs forward return."""

    merged = scores.merge(fwd, on=["date", "security_id"], how="inner")
    ics: dict[object, float] = {}
    for date_val, rows in merged.groupby("date", sort=True):
        if len(rows) < min_names:
            continue
        corr = rows["score"].corr(rows["fwd_ret"], method=method)
        if pd.notna(corr):
            ics[date_val] = float(corr)
    return pd.Series(ics, dtype=float).sort_index()


def _summarise_ic(ic: pd.Series, *, horizon: int, method: str) -> ICStats:
    """Mean IC with a Newey-West t-stat accounting for overlap up to ``horizon``."""

    values = ic.to_numpy(dtype=float)
    n = len(values)
    if n < 3:
        return ICStats(
            horizon=horizon,
            method=method,
            mean_ic=float(np.nanmean(values)) if n else 0.0,
            std_ic=float(np.nanstd(values, ddof=1)) if n > 1 else 0.0,
            ic_ir=0.0,
            t_stat=0.0,
            p_value=1.0,
            newey_west_se=0.0,
            n_periods=n,
            interpretation="too few periods to assess significance",
        )

    maxlags = max(1, horizon)
    model = sm.OLS(values, np.ones(n)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    mean_ic = float(model.params[0])
    se = float(model.bse[0])
    t_stat = float(model.tvalues[0])
    p_value = float(model.pvalues[0])
    std_ic = float(np.std(values, ddof=1))
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0

    direction = "positive" if mean_ic > 0 else "negative"
    significant = abs(t_stat) >= 1.96
    interpretation = f"{direction} mean IC {mean_ic:.3f} " + (
        f"significant (Newey-West t={t_stat:.2f})"
        if significant
        else f"not significant (Newey-West t={t_stat:.2f})"
    )
    return ICStats(
        horizon=horizon,
        method=method,
        mean_ic=mean_ic,
        std_ic=std_ic,
        ic_ir=ic_ir,
        t_stat=t_stat,
        p_value=p_value,
        newey_west_se=se,
        n_periods=n,
        interpretation=interpretation,
    )


# --- quantiles, turnover, autocorrelation ------------------------------------


def _quantile_returns(
    scores: pd.DataFrame, fwd: pd.DataFrame, *, n_quantiles: int
) -> tuple[list[float], float]:
    """Mean forward return per score quantile (low->high) and long-short spread."""

    merged = scores.merge(fwd, on=["date", "security_id"], how="inner")
    per_bucket: list[list[float]] = [[] for _ in range(n_quantiles)]
    for _date_val, rows in merged.groupby("date", sort=False):
        if len(rows) < n_quantiles:
            continue
        # Rank before cutting so tied scores never collapse the bucket edges.
        ranks = rows["score"].rank(method="first")
        buckets = pd.qcut(ranks, n_quantiles, labels=False)
        for q in range(n_quantiles):
            mask = buckets == q
            if mask.any():
                per_bucket[q].append(float(rows.loc[mask.to_numpy(), "fwd_ret"].mean()))
    means = [float(np.mean(b)) if b else float("nan") for b in per_bucket]
    spread = means[-1] - means[0]
    return means, spread


def _top_quantile_turnover(scores: pd.DataFrame, *, n_quantiles: int) -> float:
    """Average daily membership turnover of the top score quantile."""

    # groupby(sort=True) yields dates in order, so insertion order is chronological.
    ordered: list[set[str]] = []
    for _date_val, rows in scores.groupby("date", sort=True):
        if len(rows) < n_quantiles:
            continue
        threshold = rows["score"].quantile(1.0 - 1.0 / n_quantiles)
        members = set(rows.loc[rows["score"] >= threshold, "security_id"])
        if members:
            ordered.append(members)

    if len(ordered) < 2:
        return float("nan")
    turnovers = [
        len(curr - prev) / len(curr)
        for prev, curr in itertools.pairwise(ordered)
        if curr
    ]
    return float(np.mean(turnovers)) if turnovers else float("nan")


def _factor_autocorrelation(scores: pd.DataFrame) -> float:
    """Mean day-over-day Spearman autocorrelation of the cross-sectional scores."""

    wide = scores.pivot_table(
        index="date", columns="security_id", values="score"
    ).sort_index()
    autocorrs: list[float] = []
    for i in range(1, len(wide)):
        corr = wide.iloc[i].corr(wide.iloc[i - 1], method="spearman")
        if pd.notna(corr):
            autocorrs.append(float(corr))
    return float(np.mean(autocorrs)) if autocorrs else float("nan")


# --- top-level evaluation ----------------------------------------------------


def evaluate_factor(
    factor: Factor,
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    universe: Mapping[date, set[str]] | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    primary_horizon: int | None = None,
    n_quantiles: int = DEFAULT_QUANTILES,
    allow_holdout: bool = False,
) -> FactorEvaluation:
    """Evaluate ``factor`` over the (train-split) ``panel`` and ``prices``.

    ``panel`` is the long feature panel the factor scores; ``prices`` is long
    ``[date, security_id, close]`` for the forward returns. Raises
    :class:`~quantlab.factors.split.HoldoutViolationError` if the scored dates reach
    the locked holdout without the explicit unlock.
    """

    scores = factor.compute(panel, universe=universe)
    guard_holdout(scores["date"].tolist(), allow_holdout=allow_holdout)

    horizons = tuple(horizons)
    primary = primary_horizon if primary_horizon is not None else horizons[0]

    pearson: dict[int, ICStats] = {}
    spearman: dict[int, ICStats] = {}
    decay: dict[int, float] = {}
    for h in horizons:
        fwd = forward_returns(prices, h)
        p_series = _ic_series(scores, fwd, method="pearson")
        s_series = _ic_series(scores, fwd, method="spearman")
        pearson[h] = _summarise_ic(p_series, horizon=h, method="pearson")
        spearman[h] = _summarise_ic(s_series, horizon=h, method="spearman")
        decay[h] = pearson[h].mean_ic

    fwd_primary = forward_returns(prices, primary)
    quantile_returns, spread = _quantile_returns(
        scores, fwd_primary, n_quantiles=n_quantiles
    )
    turnover = _top_quantile_turnover(scores, n_quantiles=n_quantiles)
    autocorr = _factor_autocorrelation(scores)
    fitness = pearson[primary].ic_ir

    return FactorEvaluation(
        factor_name=factor.name,
        primary_horizon=primary,
        horizons=horizons,
        pearson_ic=pearson,
        spearman_ic=spearman,
        ic_decay=decay,
        quantile_returns=tuple(quantile_returns),
        long_short_spread=spread,
        turnover=turnover,
        factor_autocorrelation=autocorr,
        fitness=fitness,
        n_quantiles=n_quantiles,
    )
