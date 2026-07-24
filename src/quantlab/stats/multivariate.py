"""Multivariate diagnostics: VIF, PCA, correlation and covariance (PRD Module 5)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .results import MatrixResult, PCAResult, VIFResult

#: Conventional VIF cut-off above which multicollinearity is a concern.
VIF_THRESHOLD = 5.0


def vif(df: pd.DataFrame, *, threshold: float = VIF_THRESHOLD) -> VIFResult:
    """Variance Inflation Factor for each column of ``df``.

    A constant is added internally (VIF is undefined without an intercept). A VIF
    above ``threshold`` flags a feature whose variance is inflated by collinearity
    with the others.
    """

    frame = df.dropna()
    design = np.column_stack([np.ones(len(frame)), frame.to_numpy(dtype=float)])
    scores: dict[str, float] = {}
    for i, column in enumerate(frame.columns):
        # +1 offset skips the intercept column at index 0.
        scores[str(column)] = float(variance_inflation_factor(design, i + 1))

    high = tuple(name for name, value in scores.items() if value > threshold)
    interpretation = (
        f"high multicollinearity in: {', '.join(high)} (VIF > {threshold:g})"
        if high
        else f"no feature exceeds a VIF of {threshold:g}"
    )
    return VIFResult(
        name="Variance Inflation Factor",
        vif=scores,
        high_collinearity=high,
        threshold=threshold,
        interpretation=interpretation,
    )


def pca(
    df: pd.DataFrame, *, n_components: int | None = None, standardize: bool = True
) -> PCAResult:
    """Principal component analysis of ``df`` (wraps ``statsmodels`` PCA).

    Reports the explained-variance ratio per component, the cumulative curve, the
    number of components needed to reach 90% variance, and the loadings.
    """

    frame = df.dropna()
    ncomp = n_components or frame.shape[1]
    model = PCA(
        frame.to_numpy(dtype=float),
        ncomp=ncomp,
        standardize=standardize,
        method="svd",
    )
    eigenvalues = np.asarray(model.eigenvals, dtype=float)
    ratio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(ratio)
    n_for_90 = int(np.searchsorted(cumulative, 0.90) + 1)

    loadings = pd.DataFrame(
        np.asarray(model.loadings),
        index=frame.columns,
        columns=[f"PC{i + 1}" for i in range(ncomp)],
    )
    interpretation = (
        f"{n_for_90} of {frame.shape[1]} components explain 90% of the variance; "
        f"PC1 alone explains {ratio[0] * 100:.1f}%"
    )
    return PCAResult(
        name="PCA",
        explained_variance_ratio=tuple(float(v) for v in ratio),
        cumulative_variance=tuple(float(v) for v in cumulative),
        n_components_for_90pct=n_for_90,
        loadings=loadings,
        interpretation=interpretation,
    )


def correlation_matrix(
    df: pd.DataFrame,
    *,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
) -> MatrixResult:
    """Correlation matrix of ``df`` (``pearson`` / ``spearman`` / ``kendall``)."""

    matrix = df.corr(method=method)
    off_diag = matrix.where(~np.eye(len(matrix), dtype=bool))
    strongest = off_diag.abs().max().max()
    interpretation = (
        f"strongest pairwise |{method}| correlation is {strongest:.2f}"
        if not np.isnan(strongest)
        else "correlation matrix is undefined (insufficient data)"
    )
    return MatrixResult(
        name="Correlation matrix",
        matrix=matrix,
        method=method,
        interpretation=interpretation,
    )


def covariance_matrix(df: pd.DataFrame) -> MatrixResult:
    """Covariance matrix of ``df`` columns."""

    matrix = df.cov()
    interpretation = f"largest variance on the diagonal is {np.diag(matrix).max():.4g}"
    return MatrixResult(
        name="Covariance matrix",
        matrix=matrix,
        method="covariance",
        interpretation=interpretation,
    )
