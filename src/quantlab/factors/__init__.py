"""Alpha research: factor protocol, baselines, evaluation and the IC gate.

PRD Module 6. A factor turns a feature panel into cross-sectional scores within
the point-in-time universe; the evaluation suite measures whether those scores
predict forward returns; and the gate refuses any factor that does not clear the
IC threshold — enforced in code, on the train split only.
"""

from __future__ import annotations

from .baselines import (
    baseline_factors,
    low_volatility_factor,
    momentum_12_1_factor,
    short_term_reversal_factor,
    volume_factor,
)
from .evaluation import (
    FactorEvaluation,
    ICStats,
    evaluate_factor,
    forward_returns,
)
from .gate import (
    FactorNotEligibleError,
    assert_eligible,
    eligibility_reason,
    is_eligible,
)
from .protocol import (
    BaseFactor,
    Factor,
    FeatureFactor,
    cross_sectional_zscore,
)
from .split import HoldoutViolationError, guard_holdout

__all__ = [
    "BaseFactor",
    "Factor",
    "FactorEvaluation",
    "FactorNotEligibleError",
    "FeatureFactor",
    "HoldoutViolationError",
    "ICStats",
    "assert_eligible",
    "baseline_factors",
    "cross_sectional_zscore",
    "eligibility_reason",
    "evaluate_factor",
    "forward_returns",
    "guard_holdout",
    "is_eligible",
    "low_volatility_factor",
    "momentum_12_1_factor",
    "short_term_reversal_factor",
    "volume_factor",
]
