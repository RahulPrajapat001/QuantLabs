"""Holdout enforcement: evaluation refuses locked data unless explicitly unlocked."""

from __future__ import annotations

from datetime import date

import pytest

from quantlab.config import HOLDOUT_START, get_settings
from quantlab.factors import HoldoutViolationError, evaluate_factor, guard_holdout

from .conftest import make_signal_panel
from .test_evaluation import _SignalFactor


def test_guard_allows_train_dates() -> None:
    # Well before the holdout: no raise.
    guard_holdout([date(2018, 1, 1), date(2021, 12, 31)])


def test_guard_raises_on_holdout_dates() -> None:
    with pytest.raises(HoldoutViolationError, match="locked holdout"):
        guard_holdout([date(2018, 1, 1), HOLDOUT_START])


def test_guard_still_raises_with_flag_but_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # allow_holdout alone is not enough without QUANTLAB_UNLOCK_HOLDOUT.
    get_settings.cache_clear()
    monkeypatch.delenv("QUANTLAB_UNLOCK_HOLDOUT", raising=False)
    with pytest.raises(HoldoutViolationError, match="allow_holdout=True and"):
        guard_holdout([HOLDOUT_START], allow_holdout=True)
    get_settings.cache_clear()


def test_guard_permits_holdout_when_fully_unlocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("QUANTLAB_UNLOCK_HOLDOUT", "1")
    guard_holdout([HOLDOUT_START], allow_holdout=True)  # no raise
    get_settings.cache_clear()


def test_evaluate_factor_refuses_holdout_data() -> None:
    # A panel whose dates run into the holdout must be rejected by the evaluator.
    panel, prices = make_signal_panel(n_days=60, n_securities=10, start="2023-12-01")
    with pytest.raises(HoldoutViolationError):
        evaluate_factor(_SignalFactor(), panel, prices, primary_horizon=1)
