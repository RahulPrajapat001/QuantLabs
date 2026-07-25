"""Correlation clustering, and the effective number of independent bets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import squareform

from quantlab.risk.correlation import (
    cluster_correlations,
    correlation_distance,
    rolling_mean_correlation,
)
from quantlab.risk.var import RiskError

from .conftest import blocked_panel, correlated_panel

# --- The distance metric -----------------------------------------------------


def test_distance_is_the_square_root_transform() -> None:
    """``d = √(2(1-ρ))``, not ``1-ρ``."""

    correlation = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    distance = correlation_distance(correlation)
    assert distance[0, 1] == pytest.approx(np.sqrt(2.0 * 0.5))
    assert distance[0, 0] == 0.0


def test_distance_satisfies_the_triangle_inequality() -> None:
    """The property ``1-ρ`` lacks, and the reason for the transform.

    Without it the linkage is not a valid hierarchy, and the dendrogram it produces is
    an arbitrary tree rather than a nested structure.
    """

    panel = correlated_panel(n_assets=12, n_obs=400)
    distance = correlation_distance(panel.corr())
    n = distance.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert distance[i, j] <= distance[i, k] + distance[k, j] + 1e-9


def test_distance_is_exactly_symmetric_for_squareform() -> None:
    """``squareform`` rejects matrices asymmetric by even 1e-16."""

    panel = correlated_panel(n_assets=10)
    distance = correlation_distance(panel.corr())
    squareform(distance, checks=True)  # raises if not perfectly symmetric


def test_perfect_correlation_gives_zero_distance() -> None:
    identical = pd.DataFrame(
        [[1.0, 1.0], [1.0, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    assert correlation_distance(identical)[0, 1] == pytest.approx(0.0)


def test_perfect_anticorrelation_gives_maximum_distance() -> None:
    opposed = pd.DataFrame(
        [[1.0, -1.0], [-1.0, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    assert correlation_distance(opposed)[0, 1] == pytest.approx(2.0)


def test_asymmetric_correlation_input_is_refused() -> None:
    bad = pd.DataFrame([[1.0, 0.5], [0.2, 1.0]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(RiskError, match="not symmetric"):
        correlation_distance(bad)


# --- Clustering recovers known structure -------------------------------------


def test_clustering_recovers_known_blocks() -> None:
    """Three groups of four names, each driven by its own factor.

    The clustering must put each group together — a known right answer rather than a
    plausible-looking dendrogram.
    """

    panel = blocked_panel(groups=3, per_group=4)
    clustered = cluster_correlations(panel, n_clusters=3)

    assert clustered.n_clusters == 3
    groups = clustered.clusters()
    assert len(groups) == 3
    for members in groups.values():
        prefixes = {name.split("-")[0] for name in members}
        assert len(prefixes) == 1, f"cluster mixed groups: {members}"
        assert len(members) == 4


def test_ordered_matrix_puts_clustered_names_adjacent() -> None:
    panel = blocked_panel(groups=3, per_group=4)
    clustered = cluster_correlations(panel, n_clusters=3)
    order = [str(c) for c in clustered.ordered.columns]
    # Reordering groups the prefixes into contiguous runs.
    prefixes = [name.split("-")[0] for name in order]
    runs = [prefixes[0]]
    for prefix in prefixes[1:]:
        if prefix != runs[-1]:
            runs.append(prefix)
    assert len(runs) == 3, f"groups are not contiguous after reordering: {order}"


def test_ordered_matrix_holds_the_same_values() -> None:
    """Reordering is cosmetic — it must not change any correlation."""

    panel = correlated_panel(n_assets=8)
    clustered = cluster_correlations(panel)
    for a in clustered.correlation.index:
        for b in clustered.correlation.columns:
            assert clustered.ordered.loc[a, b] == pytest.approx(
                clustered.correlation.loc[a, b]
            )


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_every_linkage_method_works(method: str) -> None:
    panel = blocked_panel(groups=2, per_group=3)
    clustered = cluster_correlations(panel, method=method, n_clusters=2)
    assert clustered.method == method
    assert clustered.n_clusters == 2


def test_distance_threshold_selects_the_cut() -> None:
    panel = blocked_panel(groups=4, per_group=3)
    tight = cluster_correlations(panel, distance_threshold=0.1)
    loose = cluster_correlations(panel, distance_threshold=1.9)
    assert tight.n_clusters >= loose.n_clusters


def test_passing_both_cut_criteria_is_refused() -> None:
    panel = correlated_panel(n_assets=6)
    with pytest.raises(RiskError, match="not both"):
        cluster_correlations(panel, n_clusters=2, distance_threshold=0.5)


# --- Effective bets ----------------------------------------------------------


def test_uncorrelated_assets_give_effective_bets_equal_to_the_count() -> None:
    rng = np.random.default_rng(5)
    independent = pd.DataFrame(rng.normal(0, 0.01, (2000, 6)), columns=list("ABCDEF"))
    clustered = cluster_correlations(independent)
    assert clustered.effective_bets == pytest.approx(6.0, abs=0.4)


def test_highly_correlated_assets_collapse_to_about_one_bet() -> None:
    """Thirty bank stocks are not thirty positions.

    The participation ratio is what makes that quantitative rather than rhetorical.
    """

    rng = np.random.default_rng(6)
    driver = rng.normal(0, 0.015, 1000)
    panel = pd.DataFrame(
        {f"BANK{i}": driver + rng.normal(0, 0.002, 1000) for i in range(10)}
    )
    clustered = cluster_correlations(panel)
    assert clustered.mean_correlation > 0.9
    assert clustered.effective_bets < 2.0
    text = clustered.render()
    assert "independent positions" in text
    assert "less diversified than its position count" in text


def test_effective_bets_lies_between_one_and_the_asset_count() -> None:
    panel = correlated_panel(n_assets=8)
    clustered = cluster_correlations(panel)
    assert 1.0 <= clustered.effective_bets <= 8.0 + 1e-9


def test_most_correlated_pair_is_identified() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.01, 500)
    panel = pd.DataFrame(
        {
            "A": base,
            "B": base + rng.normal(0, 0.0005, 500),  # near-duplicate of A
            "C": rng.normal(0, 0.01, 500),
            "D": rng.normal(0, 0.01, 500),
        }
    )
    left, right, rho = cluster_correlations(panel).most_correlated_pair()
    assert {left, right} == {"A", "B"}
    assert rho > 0.95


def test_a_dominant_cluster_is_called_out() -> None:
    rng = np.random.default_rng(8)
    driver = rng.normal(0, 0.015, 600)
    columns = {f"SAME{i}": driver + rng.normal(0, 0.001, 600) for i in range(8)}
    columns["OTHER"] = rng.normal(0, 0.015, 600)
    clustered = cluster_correlations(pd.DataFrame(columns), n_clusters=2)
    assert len(clustered.largest_cluster()) == 8
    assert "single shared driver dominates" in clustered.render()


# --- Rolling correlation -----------------------------------------------------


def test_rolling_mean_correlation_rises_in_a_crisis() -> None:
    """The statistic that matters most: diversification fails when it is needed.

    Calm period with independent names, then a regime where everything is driven by
    one factor. A risk model built on the calm average was never describing the
    portfolio that existed in the second half.
    """

    rng = np.random.default_rng(9)
    n_assets, calm_n, crisis_n = 6, 400, 200
    calm = rng.normal(0, 0.01, (calm_n, n_assets))
    shock = rng.normal(0, 0.03, crisis_n)
    crisis = np.column_stack(
        [shock + rng.normal(0, 0.003, crisis_n) for _ in range(n_assets)]
    )
    panel = pd.DataFrame(
        np.vstack([calm, crisis]),
        columns=[f"S{i}" for i in range(n_assets)],
        index=pd.bdate_range("2016-01-04", periods=calm_n + crisis_n),
    )

    rolling = rolling_mean_correlation(panel, window=60).dropna()
    assert float(rolling.iloc[100]) < 0.2, "calm period should be uncorrelated"
    assert float(rolling.iloc[-1]) > 0.8, "crisis period should converge"


def test_rolling_correlation_needs_two_securities() -> None:
    single = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.01, 100)})
    with pytest.raises(RiskError, match="at least 2 securities"):
        rolling_mean_correlation(single)


def test_rolling_correlation_rejects_a_tiny_window() -> None:
    panel = correlated_panel(n_assets=4, n_obs=100)
    with pytest.raises(RiskError, match="at least 10"):
        rolling_mean_correlation(panel, window=3)


# --- Validation --------------------------------------------------------------


def test_clustering_needs_at_least_two_securities() -> None:
    single = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.01, 100)})
    with pytest.raises(RiskError, match="at least 2 securities"):
        cluster_correlations(single)


def test_securities_with_too_little_history_are_dropped() -> None:
    panel = correlated_panel(n_assets=4, n_obs=200)
    panel["SPARSE"] = np.nan
    panel.iloc[-5:, panel.columns.get_loc("SPARSE")] = 0.01
    clustered = cluster_correlations(panel, min_observations=30)
    assert "SPARSE" not in clustered.correlation.columns


def test_all_sparse_input_is_refused() -> None:
    panel = pd.DataFrame({"A": [np.nan] * 100, "B": [np.nan] * 100})
    with pytest.raises(RiskError, match="fewer than 2 securities"):
        cluster_correlations(panel)


@pytest.mark.parametrize("correlation_method", ["pearson", "spearman", "kendall"])
def test_every_correlation_method_works(correlation_method: str) -> None:
    panel = blocked_panel(groups=2, per_group=3, n_obs=200)
    clustered = cluster_correlations(
        panel, correlation_method=correlation_method, n_clusters=2
    )
    assert clustered.correlation_method == correlation_method
    assert correlation_method in clustered.render()


def test_labels_cover_every_security() -> None:
    panel = correlated_panel(n_assets=7)
    clustered = cluster_correlations(panel)
    assert set(clustered.labels.index) == set(panel.columns)
    assert clustered.linkage.shape == (6, 4)  # n-1 merges, 4 columns
