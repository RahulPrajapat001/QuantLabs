"""Portfolio optimisation (PRD Module 10).

* :mod:`quantlab.portfolio.covariance` — sample and shrunk covariance estimation.
  **Ledoit-Wolf shrinkage is the default**, because sample covariance is unstable
  when assets approach observations and an optimiser will find that instability.
* :mod:`quantlab.portfolio.optimisers` — equal weight, inverse volatility, risk
  parity, minimum variance, maximum Sharpe, and the efficient frontier, with weight
  bounds, long-only or long-short, and a turnover penalty.
* :mod:`quantlab.portfolio.rebalance` — rebalancing against the point-in-time
  universe, so a security that left the index is exited rather than held silently.
"""

from quantlab.portfolio.covariance import (
    ILL_CONDITIONED,
    CovarianceError,
    CovarianceEstimate,
    constant_correlation_covariance,
    estimate_covariance,
    ledoit_wolf_covariance,
    sample_covariance,
    single_index_covariance,
)
from quantlab.portfolio.optimisers import (
    OptimisationError,
    OptimisationResult,
    PortfolioConstraints,
    analytic_minimum_variance,
    efficient_frontier,
    equal_weight,
    inverse_volatility,
    maximum_sharpe,
    minimum_variance,
    risk_parity,
)
from quantlab.portfolio.rebalance import (
    RebalanceHistory,
    RebalancePlan,
    rebalance_plan,
    rebalance_schedule,
)

__all__ = [
    "ILL_CONDITIONED",
    "CovarianceError",
    "CovarianceEstimate",
    "OptimisationError",
    "OptimisationResult",
    "PortfolioConstraints",
    "RebalanceHistory",
    "RebalancePlan",
    "analytic_minimum_variance",
    "constant_correlation_covariance",
    "efficient_frontier",
    "equal_weight",
    "estimate_covariance",
    "inverse_volatility",
    "ledoit_wolf_covariance",
    "maximum_sharpe",
    "minimum_variance",
    "rebalance_plan",
    "rebalance_schedule",
    "risk_parity",
    "sample_covariance",
    "single_index_covariance",
]
