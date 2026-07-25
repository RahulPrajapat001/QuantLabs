"""Optimisers — the required test is the two-asset closed-form min-variance check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio.covariance import ledoit_wolf_covariance, sample_covariance
from quantlab.portfolio.optimisers import (
    OptimisationError,
    PortfolioConstraints,
    analytic_minimum_variance,
    efficient_frontier,
    equal_weight,
    inverse_volatility,
    maximum_sharpe,
    minimum_variance,
    risk_parity,
)

from .conftest import (
    closed_form_min_variance_weight,
    factor_panel,
    two_asset_covariance,
)

#: Unconstrained bounds, so the solver reproduces the unconstrained closed form.
UNBOUNDED = PortfolioConstraints(long_only=False, min_weight=-10.0, max_weight=10.0)


# --- THE REQUIRED TEST -------------------------------------------------------


@pytest.mark.parametrize(
    ("sigma_a", "sigma_b", "rho"),
    [
        (0.20, 0.30, 0.25),  # the canonical case
        (0.20, 0.30, 0.00),  # uncorrelated: weights are inverse-variance
        (0.20, 0.30, -0.50),  # negatively correlated: both weights still positive
        (0.15, 0.15, 0.60),  # equal volatility: must split exactly 50/50
        (0.10, 0.40, 0.30),  # very dispersed volatility
    ],
)
def test_min_variance_matches_the_closed_form_two_asset_solution(
    sigma_a: float, sigma_b: float, rho: float
) -> None:
    """Two assets with known correlation, verified against the analytical solution.

    For two assets the minimum-variance weight has a closed form::

        w_a = (σ_b² - ρσ_aσ_b) / (σ_a² + σ_b² - 2ρσ_aσ_b)

    derived in ``conftest.closed_form_min_variance_weight`` independently of the
    implementation. If the cvxpy problem is set up wrongly — wrong objective, a sign
    error, a stray constraint — this is where it shows, because there is an exact
    answer to compare against rather than a plausible-looking one.
    """

    covariance = two_asset_covariance(sigma_a, sigma_b, rho)
    expected_a = closed_form_min_variance_weight(sigma_a, sigma_b, rho)

    result = minimum_variance(covariance, constraints=UNBOUNDED, periods_per_year=1)

    assert result.status == "optimal"
    assert result.weights["A"] == pytest.approx(expected_a, abs=1e-8)
    assert result.weights["B"] == pytest.approx(1.0 - expected_a, abs=1e-8)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)

    # And the resulting variance matches the closed-form variance.
    w = np.array([expected_a, 1.0 - expected_a])
    expected_variance = float(w @ covariance.to_numpy() @ w)
    assert result.volatility == pytest.approx(np.sqrt(expected_variance), abs=1e-8)


def test_equal_volatility_splits_exactly_in_half() -> None:
    """A specific closed-form case worth its own assertion.

    When both assets have identical volatility the correlation cancels out of the
    formula entirely and the answer is 50/50 for any ρ. An implementation that got
    the correlation term wrong would still pass on asymmetric inputs but fail here.
    """

    for rho in (-0.9, -0.3, 0.0, 0.3, 0.9):
        result = minimum_variance(
            two_asset_covariance(0.2, 0.2, rho),
            constraints=UNBOUNDED,
            periods_per_year=1,
        )
        assert result.weights["A"] == pytest.approx(0.5, abs=1e-7), (
            f"failed at rho={rho}"
        )


def test_uncorrelated_assets_get_inverse_variance_weights() -> None:
    """With ρ=0 the closed form collapses to ``σ_b²/(σ_a²+σ_b²)``."""

    result = minimum_variance(
        two_asset_covariance(0.10, 0.20, 0.0), constraints=UNBOUNDED, periods_per_year=1
    )
    expected = 0.20**2 / (0.10**2 + 0.20**2)
    assert result.weights["A"] == pytest.approx(expected, abs=1e-8)
    assert result.weights["A"] == pytest.approx(0.8, abs=1e-8)


def test_solver_agrees_with_the_analytic_helper_on_many_assets() -> None:
    """The unconstrained solution is ``Σ⁻¹1/1'Σ⁻¹1`` for any number of assets."""

    covariance = sample_covariance(factor_panel(n_assets=6, n_obs=500)).matrix
    analytic = analytic_minimum_variance(covariance)
    solved = minimum_variance(covariance, constraints=UNBOUNDED, periods_per_year=1)
    pd.testing.assert_series_equal(
        solved.weights, analytic, check_names=False, atol=1e-6
    )


def test_analytic_helper_refuses_a_singular_matrix() -> None:
    """A singular matrix is exactly what shrinkage exists to fix, and it says so."""

    from quantlab.portfolio.covariance import CovarianceError

    singular = pd.DataFrame(
        [[0.04, 0.04], [0.04, 0.04]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(CovarianceError, match="shrinkage"):
        analytic_minimum_variance(singular)


# --- Equal weight ------------------------------------------------------------


def test_equal_weight_is_exactly_equal() -> None:
    result = equal_weight(["A", "B", "C", "D"])
    assert result.weights.tolist() == [0.25] * 4
    assert result.effective_positions == pytest.approx(4.0)
    assert result.status == "analytic"


def test_equal_weight_reports_volatility_when_given_a_covariance(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    result = equal_weight(panel.columns, covariance=estimate)
    assert result.volatility > 0
    assert result.shrinkage == pytest.approx(estimate.shrinkage)


def test_equal_weight_needs_assets() -> None:
    with pytest.raises(OptimisationError, match="no assets"):
        equal_weight([])


# --- Inverse volatility ------------------------------------------------------


def test_inverse_volatility_weights_are_proportional_to_one_over_sigma() -> None:
    covariance = two_asset_covariance(0.10, 0.30, 0.0)
    result = inverse_volatility(
        covariance,
        constraints=PortfolioConstraints(max_weight=1.0),
        periods_per_year=1,
    )
    # 1/0.10 : 1/0.30 = 3 : 1
    assert result.weights["A"] == pytest.approx(0.75, abs=1e-9)
    assert result.weights["B"] == pytest.approx(0.25, abs=1e-9)


def test_inverse_volatility_uses_only_the_diagonal() -> None:
    """Correlations are ignored by construction — the same diagonal, same answer."""

    low = inverse_volatility(
        two_asset_covariance(0.2, 0.3, -0.8),
        constraints=PortfolioConstraints(max_weight=1.0),
    )
    high = inverse_volatility(
        two_asset_covariance(0.2, 0.3, 0.8),
        constraints=PortfolioConstraints(max_weight=1.0),
    )
    pd.testing.assert_series_equal(low.weights, high.weights)


def test_inverse_volatility_respects_the_weight_cap(panel) -> None:
    result = inverse_volatility(
        ledoit_wolf_covariance(panel), constraints=PortfolioConstraints(max_weight=0.15)
    )
    assert result.weights.max() <= 0.15 + 1e-9


def test_inverse_volatility_rejects_zero_variance() -> None:
    flat = pd.DataFrame([[0.0, 0.0], [0.0, 0.04]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(OptimisationError, match="zero or negative variance"):
        inverse_volatility(flat)


# --- Risk parity -------------------------------------------------------------


def test_risk_parity_equalises_risk_contributions(panel) -> None:
    """The defining property: every asset contributes the same share of variance."""

    result = risk_parity(ledoit_wolf_covariance(panel))
    contributions = np.array(result.diagnostics["risk_contributions"])
    assert contributions.sum() == pytest.approx(1.0, abs=1e-6)
    expected = 1.0 / len(contributions)
    assert np.allclose(contributions, expected, atol=1e-4)
    assert result.diagnostics["max_contribution_gap"] < 1e-3


def test_risk_parity_is_long_only_by_construction(panel) -> None:
    """The log barrier forces strictly positive weights."""

    result = risk_parity(ledoit_wolf_covariance(panel))
    assert (result.weights > 0).all()
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_risk_parity_sits_between_equal_weight_and_min_variance(panel) -> None:
    """It uses more information than equal weight, less aggressively than min-var."""

    estimate = ledoit_wolf_covariance(panel)
    equal = equal_weight(panel.columns, covariance=estimate)
    parity = risk_parity(estimate)
    minvar = minimum_variance(
        estimate, constraints=PortfolioConstraints(max_weight=1.0)
    )
    assert minvar.volatility <= parity.volatility + 1e-9
    assert parity.volatility <= equal.volatility + 1e-9


def test_risk_parity_honours_an_unequal_risk_budget(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    assets = list(panel.columns)
    budget = dict.fromkeys(assets, 1.0)
    budget[assets[0]] = 3.0  # triple risk allocation to the first asset
    result = risk_parity(estimate, risk_budget=budget)
    contributions = np.array(result.diagnostics["risk_contributions"])
    total = sum(budget.values())
    assert contributions[0] == pytest.approx(3.0 / total, abs=1e-3)


def test_risk_parity_rejects_a_bad_budget(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    with pytest.raises(OptimisationError, match="strictly positive"):
        risk_parity(estimate, risk_budget=dict.fromkeys(panel.columns, 0.0))
    with pytest.raises(OptimisationError, match="no risk budget"):
        risk_parity(estimate, risk_budget={next(iter(panel.columns)): 1.0})


# --- Constraints -------------------------------------------------------------


def test_weight_bounds_bind(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    result = minimum_variance(
        estimate, constraints=PortfolioConstraints(max_weight=0.12, min_weight=0.05)
    )
    assert result.weights.min() >= 0.05 - 1e-8
    assert result.weights.max() <= 0.12 + 1e-8


def test_long_only_forbids_shorts(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    result = minimum_variance(
        estimate, constraints=PortfolioConstraints(long_only=True, max_weight=1.0)
    )
    assert (result.weights >= -1e-9).all()


def test_long_short_permits_them() -> None:
    """With a negative floor the optimiser may short, and on this input it does."""

    covariance = two_asset_covariance(0.10, 0.11, 0.98)
    result = minimum_variance(
        covariance,
        constraints=PortfolioConstraints(
            long_only=False, min_weight=-1.0, max_weight=2.0
        ),
        periods_per_year=1,
    )
    # Two near-identical assets with different vols: the closed form shorts the
    # higher-volatility one.
    assert result.weights.min() < 0


def test_infeasible_bounds_raise_before_solving() -> None:
    """Caught arithmetically, with a message that says how to fix it."""

    with pytest.raises(OptimisationError, match="caps total weight"):
        minimum_variance(
            two_asset_covariance(), constraints=PortfolioConstraints(max_weight=0.1)
        )


def test_impossible_floor_raises() -> None:
    with pytest.raises(OptimisationError, match="forces at least"):
        minimum_variance(
            two_asset_covariance(),
            constraints=PortfolioConstraints(min_weight=0.8, max_weight=1.0),
        )


def test_constraints_validate_themselves() -> None:
    with pytest.raises(OptimisationError, match="max_weight must be positive"):
        PortfolioConstraints(max_weight=0.0)
    with pytest.raises(OptimisationError, match="turnover_penalty"):
        PortfolioConstraints(turnover_penalty=-1.0)
    with pytest.raises(OptimisationError, match="contradicts long_only"):
        PortfolioConstraints(long_only=True, min_weight=-0.1)


# --- Turnover ----------------------------------------------------------------


def test_turnover_penalty_reduces_trading(panel) -> None:
    """A real rebalance costs money; the penalty is what makes the optimiser care."""

    estimate = ledoit_wolf_covariance(panel)
    previous = dict.fromkeys(panel.columns, 1.0 / len(panel.columns))

    unpenalised = minimum_variance(
        estimate,
        constraints=PortfolioConstraints(max_weight=0.30),
        previous_weights=previous,
    )
    penalised = minimum_variance(
        estimate,
        constraints=PortfolioConstraints(max_weight=0.30, turnover_penalty=5.0),
        previous_weights=previous,
    )
    assert penalised.turnover is not None and unpenalised.turnover is not None
    assert penalised.turnover < unpenalised.turnover
    # And it costs variance — that is the trade-off, not a bug.
    assert penalised.volatility >= unpenalised.volatility - 1e-9


def test_a_hard_turnover_cap_binds(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    previous = dict.fromkeys(panel.columns, 1.0 / len(panel.columns))
    result = minimum_variance(
        estimate,
        constraints=PortfolioConstraints(max_weight=0.30, max_turnover=0.10),
        previous_weights=previous,
    )
    assert result.turnover is not None
    assert result.turnover <= 0.10 + 1e-6


def test_turnover_is_zero_against_an_identical_portfolio(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    equal = equal_weight(panel.columns, covariance=estimate)
    again = equal_weight(
        panel.columns, covariance=estimate, previous_weights=equal.weights
    )
    assert again.turnover == pytest.approx(0.0, abs=1e-12)


# --- Efficient frontier and max Sharpe ---------------------------------------


def test_frontier_is_monotonic_in_risk_and_return(panel) -> None:
    """Higher target return must cost more volatility, or it is not a frontier."""

    estimate = ledoit_wolf_covariance(panel)
    frontier = efficient_frontier(
        panel.mean(),
        estimate,
        constraints=PortfolioConstraints(max_weight=0.30),
        n_points=15,
    )
    assert len(frontier) >= 10
    assert frontier["expected_return"].is_monotonic_increasing
    # Volatility is non-decreasing up to solver tolerance.
    volatility = frontier["volatility"].to_numpy()
    assert np.all(np.diff(volatility) > -1e-6)


def test_frontier_minimum_matches_the_min_variance_portfolio(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    constraints = PortfolioConstraints(max_weight=0.30)
    frontier = efficient_frontier(
        panel.mean(), estimate, constraints=constraints, n_points=15
    )
    minvar = minimum_variance(estimate, constraints=constraints)
    assert float(frontier["volatility"].min()) == pytest.approx(
        minvar.volatility, rel=1e-4
    )


def test_frontier_rejects_missing_expected_returns(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    partial = panel.mean().iloc[:-2]
    with pytest.raises(OptimisationError, match="no expected return"):
        efficient_frontier(partial, estimate)


def test_max_sharpe_beats_every_frontier_point_it_scanned(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    constraints = PortfolioConstraints(max_weight=0.30)
    best = maximum_sharpe(panel.mean(), estimate, constraints=constraints, n_points=40)
    frontier = efficient_frontier(
        panel.mean(), estimate, constraints=constraints, n_points=40
    )
    assert best.sharpe is not None
    assert best.sharpe >= float(frontier["sharpe"].max()) - 1e-9


def test_max_sharpe_respects_bounds_which_the_rotation_trick_would_break(panel) -> None:
    """The reason max Sharpe scans the frontier instead of rescaling a solution.

    The usual ``min w'Σw s.t. μ'w=1`` rotation then renormalises, which can push a
    weight past ``max_weight``. Scanning cannot, and this asserts it.
    """

    estimate = ledoit_wolf_covariance(panel)
    result = maximum_sharpe(
        panel.mean(),
        estimate,
        constraints=PortfolioConstraints(max_weight=0.15),
        n_points=40,
    )
    assert result.weights.max() <= 0.15 + 1e-6
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_max_sharpe_accounts_for_the_risk_free_rate(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    constraints = PortfolioConstraints(max_weight=0.30)
    zero_rf = maximum_sharpe(
        panel.mean(), estimate, constraints=constraints, n_points=30
    )
    high_rf = maximum_sharpe(
        panel.mean(),
        estimate,
        constraints=constraints,
        risk_free_rate=0.06,
        n_points=30,
    )
    assert zero_rf.sharpe is not None and high_rf.sharpe is not None
    # A higher hurdle pushes the optimum along the frontier, not backwards.
    assert high_rf.volatility >= zero_rf.volatility - 1e-9


# --- Result diagnostics ------------------------------------------------------


def test_result_reports_effective_positions_and_exposure(panel) -> None:
    estimate = ledoit_wolf_covariance(panel)
    equal = equal_weight(panel.columns, covariance=estimate)
    concentrated = minimum_variance(
        estimate, constraints=PortfolioConstraints(max_weight=0.40)
    )
    assert equal.effective_positions == pytest.approx(len(panel.columns))
    assert concentrated.effective_positions < equal.effective_positions
    assert equal.gross_exposure == pytest.approx(1.0)
    assert equal.net_exposure == pytest.approx(1.0)


def test_render_labels_the_sharpe_as_ex_ante(panel) -> None:
    """An optimiser's Sharpe is an input-derived estimate, not a backtest result."""

    estimate = ledoit_wolf_covariance(panel)
    text = maximum_sharpe(
        panel.mean(),
        estimate,
        constraints=PortfolioConstraints(max_weight=0.30),
        n_points=20,
    ).render()
    assert "EX ANTE" in text
    assert "not be reported as performance" in text


def test_render_warns_on_an_ill_conditioned_matrix() -> None:
    """The warning has to appear on the output, not only on the input diagnostics."""

    # Two almost perfectly correlated assets: invertible, but barely.
    covariance = two_asset_covariance(0.20, 0.20, 0.9999999999)
    result = minimum_variance(covariance, constraints=UNBOUNDED, periods_per_year=1)
    assert result.condition_number is not None
    assert result.condition_number > 1e9
    assert "ill-conditioned" in result.render()


def test_asymmetric_or_non_finite_covariance_is_rejected() -> None:
    asymmetric = pd.DataFrame(
        [[0.04, 0.01], [0.02, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(OptimisationError, match="not symmetric"):
        minimum_variance(asymmetric, constraints=UNBOUNDED)

    with_nan = pd.DataFrame(
        [[0.04, np.nan], [np.nan, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(OptimisationError, match="non-finite"):
        minimum_variance(with_nan, constraints=UNBOUNDED)


def test_optimisers_accept_a_covariance_estimate_or_a_frame(panel) -> None:
    """Either input type, same answer — the estimate just carries diagnostics too."""

    estimate = ledoit_wolf_covariance(panel)
    constraints = PortfolioConstraints(max_weight=0.30)
    from_estimate = minimum_variance(estimate, constraints=constraints)
    from_frame = minimum_variance(estimate.matrix, constraints=constraints)
    pd.testing.assert_series_equal(from_estimate.weights, from_frame.weights, atol=1e-8)
    assert from_estimate.shrinkage is not None
    assert from_frame.shrinkage is None
