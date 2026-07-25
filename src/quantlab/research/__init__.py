"""Research integrity layer: the log, multiple-testing corrections, pre-registration.

This subpackage is what makes the project's numbers defensible. Nothing here
computes alpha; everything here constrains what may be claimed about it.

* :mod:`quantlab.research.log` — every backtest run, written from inside the
  engine. Trials counts come from here.
* :mod:`quantlab.research.deflated` — deflated Sharpe, expected maximum Sharpe
  under the null, and probability of backtest overfitting.
* :mod:`quantlab.research.preregister` — hypotheses hashed and committed before
  they are tested.
* :mod:`quantlab.research.reporting` — the type that makes a bare Sharpe
  unrepresentable.
"""

from quantlab.research.deflated import (
    DeflatedSharpeResult,
    DeflationError,
    NullMaxSharpeDistribution,
    PBOResult,
    deflate_run,
    deflate_strategy,
    deflated_sharpe_ratio,
    expected_max_sharpe_analytic,
    expected_max_sharpe_null,
    null_max_sharpe_for_run,
    pbo_from_log,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from quantlab.research.log import (
    GitProvenance,
    LoggedRun,
    LogSummary,
    ResearchLog,
    ResearchLogError,
    RunResult,
    RunSpec,
)
from quantlab.research.preregister import (
    Hypothesis,
    PreRegistrationError,
    Threshold,
    ValidationReport,
    load_hypotheses,
    registration_commit,
    require_hypothesis,
    validate,
    validate_against_log,
)
from quantlab.research.reporting import (
    ReportingRuleError,
    SharpeReport,
    assert_no_bare_sharpe,
)

__all__ = [
    "DeflatedSharpeResult",
    "DeflationError",
    "GitProvenance",
    "Hypothesis",
    "LogSummary",
    "LoggedRun",
    "NullMaxSharpeDistribution",
    "PBOResult",
    "PreRegistrationError",
    "ReportingRuleError",
    "ResearchLog",
    "ResearchLogError",
    "RunResult",
    "RunSpec",
    "SharpeReport",
    "Threshold",
    "ValidationReport",
    "assert_no_bare_sharpe",
    "deflate_run",
    "deflate_strategy",
    "deflated_sharpe_ratio",
    "expected_max_sharpe_analytic",
    "expected_max_sharpe_null",
    "load_hypotheses",
    "null_max_sharpe_for_run",
    "pbo_from_log",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "registration_commit",
    "require_hypothesis",
    "validate",
    "validate_against_log",
]
