"""Statistical analysis wrapping statsmodels (PRD Module 5).

Every test returns a typed result carrying its statistic, p-value and a
plain-English interpretation. statsmodels does the maths; this package is a thin,
consistent surface over it.
"""

from __future__ import annotations

from .autocorrelation import acf, pacf
from .cointegration import engle_granger, johansen
from .hypothesis import adf, jarque_bera, kpss_test, ljung_box
from .multivariate import correlation_matrix, covariance_matrix, pca, vif
from .results import (
    AutocorrelationResult,
    CointegrationResult,
    HypothesisTestResult,
    JohansenResult,
    MatrixResult,
    PCAResult,
    VIFResult,
)

__all__ = [
    "AutocorrelationResult",
    "CointegrationResult",
    "HypothesisTestResult",
    "JohansenResult",
    "MatrixResult",
    "PCAResult",
    "VIFResult",
    "acf",
    "adf",
    "correlation_matrix",
    "covariance_matrix",
    "engle_granger",
    "jarque_bera",
    "johansen",
    "kpss_test",
    "ljung_box",
    "pacf",
    "pca",
    "vif",
]
