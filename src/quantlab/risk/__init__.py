"""Risk analytics (PRD Module 9).

* :mod:`quantlab.risk.var` — historical, parametric and Cornish-Fisher VaR, with
  Expected Shortfall reported alongside every one of them.
* :mod:`quantlab.risk.exposure` — rolling beta and alpha, rolling volatility, and
  factor exposure by regression with Newey-West standard errors.
* :mod:`quantlab.risk.correlation` — correlation matrices with hierarchical
  clustering, and the effective number of independent bets.
* :mod:`quantlab.risk.monte_carlo` — forward simulation from bootstrapped returns,
  block-resampled by default so drawdowns keep their clustering.
* :mod:`quantlab.risk.stress` — replay through real crisis windows. Synthetic shocks
  are never substituted for missing data.
"""

from quantlab.risk.correlation import (
    ClusteredCorrelation,
    cluster_correlations,
    correlation_distance,
    rolling_mean_correlation,
)
from quantlab.risk.exposure import (
    FactorExposure,
    FactorLoading,
    RollingExposure,
    RollingFactorExposure,
    factor_exposure,
    rolling_beta_alpha,
    rolling_factor_exposure,
    rolling_volatility,
)
from quantlab.risk.monte_carlo import MonteCarloResult, monte_carlo
from quantlab.risk.stress import (
    STRESS_WINDOWS,
    STRESS_WINDOWS_BY_SLUG,
    StressCoverage,
    StressDataUnavailableError,
    StressIngestPlan,
    StressResult,
    StressTestReport,
    StressWindow,
    get_stress_window,
    load_window_returns,
    replay_window,
    stress_coverage,
    stress_test,
    stress_window_ingest_plan,
)
from quantlab.risk.var import (
    DEFAULT_CONFIDENCES,
    RiskError,
    VaRComparison,
    VaRResult,
    cornish_fisher_var,
    expected_shortfall,
    historical_var,
    parametric_var,
    value_at_risk,
    var_comparison,
)

__all__ = [
    "DEFAULT_CONFIDENCES",
    "STRESS_WINDOWS",
    "STRESS_WINDOWS_BY_SLUG",
    "ClusteredCorrelation",
    "FactorExposure",
    "FactorLoading",
    "MonteCarloResult",
    "RiskError",
    "RollingExposure",
    "RollingFactorExposure",
    "StressCoverage",
    "StressDataUnavailableError",
    "StressIngestPlan",
    "StressResult",
    "StressTestReport",
    "StressWindow",
    "VaRComparison",
    "VaRResult",
    "cluster_correlations",
    "cornish_fisher_var",
    "correlation_distance",
    "expected_shortfall",
    "factor_exposure",
    "get_stress_window",
    "historical_var",
    "load_window_returns",
    "monte_carlo",
    "parametric_var",
    "replay_window",
    "rolling_beta_alpha",
    "rolling_factor_exposure",
    "rolling_mean_correlation",
    "rolling_volatility",
    "stress_coverage",
    "stress_test",
    "stress_window_ingest_plan",
    "value_at_risk",
    "var_comparison",
]
