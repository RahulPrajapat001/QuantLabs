"""Cointegration tests: Engle-Granger and Johansen (PRD Module 5)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .results import CointegrationResult, JohansenResult

SIGNIFICANCE = 0.05


def engle_granger(
    y: pd.Series | Sequence[float],
    x: pd.Series | Sequence[float],
    *,
    significance: float = SIGNIFICANCE,
) -> CointegrationResult:
    """Engle-Granger two-step cointegration test of ``y`` on ``x``.

    Null: the two series are **not** cointegrated. Rejecting it (p < significance)
    means a stationary linear combination exists — the basis for a pairs trade.
    The reported ``hedge_ratio`` is the OLS slope of ``y`` on ``x``.
    """

    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    stat, p_value, crit = coint(y_arr, x_arr)

    # Hedge ratio from a simple OLS slope (with intercept), computed via lstsq to
    # avoid another dependency.
    design = np.column_stack([np.ones_like(x_arr), x_arr])
    coeffs, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    hedge_ratio = float(coeffs[1])

    cointegrated = p_value < significance
    interpretation = (
        f"cointegrated (hedge ratio {hedge_ratio:.3f}) — a stationary spread exists"
        if cointegrated
        else "not cointegrated — no stationary linear combination found"
    )
    return CointegrationResult(
        name="Engle-Granger",
        statistic=float(stat),
        p_value=float(p_value),
        reject_null=cointegrated,
        cointegrated=cointegrated,
        hedge_ratio=hedge_ratio,
        critical_values={
            "1%": float(crit[0]),
            "5%": float(crit[1]),
            "10%": float(crit[2]),
        },
        interpretation=interpretation,
    )


def johansen(
    df: pd.DataFrame, *, det_order: int = 0, k_ar_diff: int = 1
) -> JohansenResult:
    """Johansen cointegration test over the columns of ``df``.

    Infers the cointegration rank from the **trace** statistic against the 95%
    critical values: the number of relations is the count of trace statistics
    that exceed their critical value, reading from ``r=0`` upward.
    """

    frame = df.dropna()
    result = coint_johansen(frame.to_numpy(dtype=float), det_order, k_ar_diff)
    trace_stats = result.lr1
    trace_crit_95 = result.cvt[:, 1]
    max_eig_stats = result.lr2
    max_eig_crit_95 = result.cvm[:, 1]

    n_cointegrating = 0
    for stat, crit in zip(trace_stats, trace_crit_95, strict=True):
        if stat > crit:
            n_cointegrating += 1
        else:
            break

    if n_cointegrating == 0:
        interpretation = "no cointegrating relationships at the 95% level"
    elif n_cointegrating == frame.shape[1]:
        interpretation = (
            "full rank — the series are jointly stationary, not merely cointegrated"
        )
    else:
        interpretation = (
            f"{n_cointegrating} cointegrating relationship(s) at the 95% level"
        )
    return JohansenResult(
        name="Johansen",
        n_cointegrating=n_cointegrating,
        trace_stats=tuple(float(v) for v in trace_stats),
        trace_crit_95=tuple(float(v) for v in trace_crit_95),
        max_eig_stats=tuple(float(v) for v in max_eig_stats),
        max_eig_crit_95=tuple(float(v) for v in max_eig_crit_95),
        interpretation=interpretation,
    )
