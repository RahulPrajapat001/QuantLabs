"""The IC gate is enforced in code, not documentation."""

from __future__ import annotations

import pytest

from quantlab.factors import (
    FactorNotEligibleError,
    assert_eligible,
    evaluate_factor,
    is_eligible,
)
from quantlab.factors.protocol import BaseFactor

from .conftest import make_signal_panel
from .test_evaluation import _NoiseFactor, _SignalFactor


def test_strong_factor_passes_gate() -> None:
    panel, prices = make_signal_panel(strength=0.02, noise=0.02, seed=5)
    ev = evaluate_factor(_SignalFactor(), panel, prices, primary_horizon=1)
    assert is_eligible(ev)
    assert_eligible(ev)  # does not raise


def test_noise_factor_fails_gate() -> None:
    panel, prices = make_signal_panel(strength=0.02, noise=0.02, seed=6)
    ev = evaluate_factor(_NoiseFactor(seed=11), panel, prices, primary_horizon=1)
    assert not is_eligible(ev)
    with pytest.raises(FactorNotEligibleError, match="not eligible"):
        assert_eligible(ev)


def test_eligibility_reason_names_the_failing_criterion() -> None:
    panel, prices = make_signal_panel(strength=0.02, noise=0.02, seed=8)
    ev = evaluate_factor(_NoiseFactor(seed=12), panel, prices, primary_horizon=1)
    from quantlab.factors import eligibility_reason

    reason = eligibility_reason(ev)
    assert reason is not None
    assert "IC" in reason


def test_evaluation_is_eligible_property_matches_gate() -> None:
    panel, prices = make_signal_panel(strength=0.02, noise=0.02, seed=9)

    class _Weak(BaseFactor):
        name = "weak"

        def _signal(self, panel):  # type: ignore[no-untyped-def]
            return panel["close"] * 0.0  # constant -> degenerate -> no IC

    ev = evaluate_factor(_Weak(), panel, prices, primary_horizon=1)
    assert ev.is_eligible == is_eligible(ev)
