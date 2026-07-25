"""Correlation structure and hierarchical clustering (PRD Module 9).

A correlation matrix over 500 securities has 124,750 entries and communicates
nothing by inspection. Hierarchical clustering reorders it so the block structure
becomes visible: which names move together, and how many genuinely independent bets
a portfolio holds.

That last question is the one that matters for risk. A portfolio of thirty Indian
banks is not thirty positions — under stress it is close to one, and the correlation
that made it look diversified in calm markets is the correlation that fails when it
is needed. :attr:`ClusteredCorrelation.effective_bets` puts a number on this.

Distance metric: ``d = √(2(1-ρ))``, the standard transformation that makes a
correlation matrix into a proper Euclidean distance. Using ``1-ρ`` directly is
common and wrong — it does not satisfy the triangle inequality, so the resulting
dendrogram is not a valid hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from quantlab.risk.var import RiskError

LinkageMethod = Literal["single", "complete", "average", "ward"]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]


@dataclass(frozen=True, slots=True)
class ClusteredCorrelation:
    """A correlation matrix with its cluster structure resolved."""

    #: Correlation matrix in the original column order.
    correlation: pd.DataFrame
    #: The same matrix reordered so clustered names are adjacent — the readable form.
    ordered: pd.DataFrame
    #: Cluster label per security, at the chosen cut.
    labels: pd.Series
    #: SciPy linkage matrix, for plotting a dendrogram.
    linkage: np.ndarray
    method: LinkageMethod
    correlation_method: CorrelationMethod
    n_clusters: int

    @property
    def mean_correlation(self) -> float:
        """Mean off-diagonal correlation — the crude diversification headline."""

        values = self.correlation.to_numpy(dtype=float)
        n = values.shape[0]
        if n < 2:
            return float("nan")
        off_diagonal = values[~np.eye(n, dtype=bool)]
        return float(np.nanmean(off_diagonal))

    @property
    def effective_bets(self) -> float:
        """How many independent positions this portfolio really holds.

        ``(Σλ)² / Σλ²`` over the correlation matrix's eigenvalues — the
        participation ratio. For ``n`` uncorrelated assets it equals ``n``; for ``n``
        perfectly correlated assets it equals 1. Thirty bank stocks with average
        correlation 0.8 give roughly 1.5, which is the honest position count.
        """

        eigenvalues = np.linalg.eigvalsh(self.correlation.to_numpy(dtype=float))
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if eigenvalues.size == 0:
            return float("nan")
        return float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum())

    def clusters(self) -> dict[int, list[str]]:
        """Members of each cluster."""

        grouped: dict[int, list[str]] = {}
        for security_id, label in self.labels.items():
            grouped.setdefault(int(label), []).append(str(security_id))
        return grouped

    def largest_cluster(self) -> list[str]:
        groups = self.clusters()
        if not groups:
            return []
        return max(groups.values(), key=len)

    def most_correlated_pair(self) -> tuple[str, str, float]:
        """The pair that moves together most — usually a near-duplicate holding."""

        matrix = self.correlation.to_numpy(dtype=float).copy()
        np.fill_diagonal(matrix, np.nan)
        if np.all(np.isnan(matrix)):
            return ("", "", float("nan"))
        flat = int(np.nanargmax(matrix))
        raw_i, raw_j = np.unravel_index(flat, matrix.shape)
        i, j = int(raw_i), int(raw_j)
        return (
            str(self.correlation.index[i]),
            str(self.correlation.columns[j]),
            float(matrix[i, j]),
        )

    def render(self) -> str:
        n = len(self.correlation)
        left, right, rho = self.most_correlated_pair()
        lines = [
            f"Correlation structure ({n} securities, {self.correlation_method}, "
            f"{self.method} linkage)",
            f"  mean off-diagonal correlation: {self.mean_correlation:6.3f}",
            f"  clusters at this cut:          {self.n_clusters:6d}",
            f"  effective independent bets:    {self.effective_bets:6.2f}",
        ]
        if left:
            lines.append(
                f"  most correlated pair:          {left} / {right} = {rho:.3f}"
            )
        if np.isfinite(self.effective_bets) and n > 0:
            ratio = self.effective_bets / n
            if ratio < 0.3:
                lines.append(
                    f"  {n} holdings behave like {self.effective_bets:.1f} independent "
                    "positions. This portfolio is far less diversified than its "
                    "position count suggests."
                )
        largest = self.largest_cluster()
        if len(largest) > max(2, n // 4):
            lines.append(
                f"  Largest cluster holds {len(largest)} of {n} names — a single "
                "shared driver dominates the book."
            )
        return "\n".join(lines)


def correlation_distance(correlation: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Convert a correlation matrix to a valid Euclidean distance matrix.

    ``d = √(2(1-ρ))``. Unlike ``1-ρ``, this satisfies the triangle inequality, which
    is a precondition for the linkage to represent a genuine hierarchy rather than an
    arbitrary tree.
    """

    values = correlation.to_numpy(dtype=float)
    if not np.allclose(values, values.T, equal_nan=True):
        raise RiskError("correlation matrix is not symmetric")
    values = np.clip(np.nan_to_num(values, nan=0.0), -1.0, 1.0)
    distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - values)))
    np.fill_diagonal(distance, 0.0)
    # Force exact symmetry: squareform rejects matrices that differ by 1e-16.
    symmetric: npt.NDArray[np.float64] = (distance + distance.T) / 2.0
    return symmetric


def cluster_correlations(
    returns: pd.DataFrame,
    *,
    method: LinkageMethod = "average",
    correlation_method: CorrelationMethod = "pearson",
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    min_observations: int = 30,
) -> ClusteredCorrelation:
    """Cluster a return panel's correlation matrix hierarchically.

    ``returns`` is wide: one column per security, indexed by date.

    Exactly one of ``n_clusters`` or ``distance_threshold`` selects the cut. With
    neither, the cut defaults to ``0.7 × max distance``, which tends to separate
    sectors on Indian equity data without splitting every name into its own group.
    """

    if returns.shape[1] < 2:
        raise RiskError(
            f"clustering needs at least 2 securities, got {returns.shape[1]}"
        )
    usable = returns.dropna(axis=1, thresh=min_observations)
    if usable.shape[1] < 2:
        raise RiskError(
            f"fewer than 2 securities have {min_observations} observations; "
            "correlations computed from shorter overlaps are not comparable"
        )

    correlation = usable.corr(method=correlation_method)
    if correlation.isna().all().all():
        raise RiskError("correlation matrix is entirely undefined")

    distance = correlation_distance(correlation)
    condensed = squareform(distance, checks=False)
    linkage = hierarchy.linkage(condensed, method=method)

    if n_clusters is not None and distance_threshold is not None:
        raise RiskError("pass n_clusters or distance_threshold, not both")
    if n_clusters is not None:
        labels = hierarchy.fcluster(linkage, t=n_clusters, criterion="maxclust")
    else:
        threshold = (
            distance_threshold
            if distance_threshold is not None
            else 0.7 * float(distance.max())
        )
        labels = hierarchy.fcluster(linkage, t=threshold, criterion="distance")

    order = hierarchy.leaves_list(linkage)
    names = list(correlation.columns)
    ordered_names = [names[i] for i in order]

    return ClusteredCorrelation(
        correlation=correlation,
        ordered=correlation.loc[ordered_names, ordered_names],
        labels=pd.Series(labels, index=names, name="cluster"),
        linkage=linkage,
        method=method,
        correlation_method=correlation_method,
        n_clusters=len(set(labels)),
    )


def rolling_mean_correlation(returns: pd.DataFrame, *, window: int = 60) -> pd.Series:
    """Mean pairwise correlation through time.

    The single most useful correlation statistic for risk: it rises sharply in a
    crisis, which is exactly when a diversified-looking portfolio stops being one.
    A backtest whose risk model used the calm-period average was never holding the
    portfolio it thought it held.
    """

    if returns.shape[1] < 2:
        raise RiskError("need at least 2 securities")
    if window < 10:
        raise RiskError(f"window must be at least 10 observations, got {window}")

    n = returns.shape[1]
    means: dict[pd.Timestamp, float] = {}
    values = returns.to_numpy(dtype=float)
    for end in range(window, len(returns) + 1):
        chunk = values[end - window : end]
        # Columns with no variation give NaN correlations; ignore them rather than
        # letting one flat series void the whole window.
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.corrcoef(chunk, rowvar=False)
        if matrix.ndim != 2 or matrix.shape[0] != n:
            continue
        off_diagonal = matrix[~np.eye(n, dtype=bool)]
        means[returns.index[end - 1]] = float(np.nanmean(off_diagonal))
    return pd.Series(means).reindex(returns.index).rename(f"mean_correlation_{window}d")
