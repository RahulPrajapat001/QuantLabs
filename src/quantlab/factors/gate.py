"""The IC gate (PRD Module 6).

A factor is eligible for the strategy builder only if it clears an IC threshold —
enforced **in code**, not documentation. Eligibility requires, at the primary
horizon, both:

* a mean Pearson IC whose magnitude is at least ``settings.factor_min_abs_ic``, and
* a Newey-West t-statistic whose magnitude is at least ``settings.factor_min_ic_tstat``.

The strategy builder calls :func:`assert_eligible`, which raises
:class:`FactorNotEligibleError` for any factor that fails — so an unproven alpha cannot
reach a backtest by omission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantlab.config import get_settings

if TYPE_CHECKING:
    from .evaluation import FactorEvaluation


class FactorNotEligibleError(RuntimeError):
    """Raised when a factor that fails the IC gate is used downstream."""


def eligibility_reason(evaluation: FactorEvaluation) -> str | None:
    """Return ``None`` if eligible, else a plain-English reason it failed."""

    settings = get_settings()
    ic = evaluation.primary_ic
    if abs(ic.mean_ic) < settings.factor_min_abs_ic:
        return (
            f"|mean IC| {abs(ic.mean_ic):.4f} < threshold "
            f"{settings.factor_min_abs_ic:.4f} at horizon {ic.horizon}"
        )
    if abs(ic.t_stat) < settings.factor_min_ic_tstat:
        return (
            f"|Newey-West t| {abs(ic.t_stat):.2f} < threshold "
            f"{settings.factor_min_ic_tstat:.2f} at horizon {ic.horizon}"
        )
    return None


def is_eligible(evaluation: FactorEvaluation) -> bool:
    """Whether ``evaluation`` clears the IC gate."""

    return eligibility_reason(evaluation) is None


def assert_eligible(evaluation: FactorEvaluation) -> None:
    """Raise :class:`FactorNotEligibleError` unless ``evaluation`` clears the gate."""

    reason = eligibility_reason(evaluation)
    if reason is not None:
        raise FactorNotEligibleError(
            f"factor {evaluation.factor_name!r} is not eligible for the strategy "
            f"builder: {reason}"
        )
