"""Typed results for the statistical tests (PRD Module 5).

Every test in this package returns one of these dataclasses. Each carries the raw
``statistic`` and ``p_value`` where applicable, plus a plain-English
``interpretation`` string so a caller (or a UI) never has to know the direction
of a particular test's null hypothesis to describe the outcome — KPSS and ADF, for
instance, have opposite nulls, and the interpretation states the conclusion, not
the mechanics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True, slots=True)
class HypothesisTestResult:
    """A single hypothesis test with a statistic, p-value and conclusion."""

    name: str
    statistic: float
    p_value: float
    null_hypothesis: str
    reject_null: bool
    interpretation: str
    details: Mapping[str, float] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - display aid
        return (
            f"{self.name}: stat={self.statistic:.4f} p={self.p_value:.4g} "
            f"-> {self.interpretation}"
        )


@dataclass(frozen=True, slots=True)
class AutocorrelationResult:
    """ACF or PACF values with confidence bands and the significant lags."""

    name: str
    values: tuple[float, ...]
    confint_lower: tuple[float, ...]
    confint_upper: tuple[float, ...]
    significant_lags: tuple[int, ...]
    interpretation: str


@dataclass(frozen=True, slots=True)
class CointegrationResult:
    """Engle-Granger two-step cointegration outcome."""

    name: str
    statistic: float
    p_value: float
    reject_null: bool
    cointegrated: bool
    hedge_ratio: float
    critical_values: Mapping[str, float]
    interpretation: str


@dataclass(frozen=True, slots=True)
class JohansenResult:
    """Johansen cointegration: rank inferred from the trace statistic."""

    name: str
    n_cointegrating: int
    trace_stats: tuple[float, ...]
    trace_crit_95: tuple[float, ...]
    max_eig_stats: tuple[float, ...]
    max_eig_crit_95: tuple[float, ...]
    interpretation: str


@dataclass(frozen=True, slots=True)
class VIFResult:
    """Variance Inflation Factor per feature, with the collinear offenders."""

    name: str
    vif: Mapping[str, float]
    high_collinearity: tuple[str, ...]
    threshold: float
    interpretation: str


@dataclass(frozen=True, slots=True)
class PCAResult:
    """Principal component decomposition summary."""

    name: str
    explained_variance_ratio: tuple[float, ...]
    cumulative_variance: tuple[float, ...]
    n_components_for_90pct: int
    loadings: pd.DataFrame
    interpretation: str


@dataclass(frozen=True, slots=True)
class MatrixResult:
    """A correlation or covariance matrix with a one-line summary."""

    name: str
    matrix: pd.DataFrame
    method: str
    interpretation: str
