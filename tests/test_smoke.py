"""Smoke tests: the package imports and core configuration is wired correctly.

No business logic is under test yet — these guard the scaffold itself.
"""

from __future__ import annotations

from datetime import date

import quantlab
from quantlab.config import (
    DATA_START,
    HOLDOUT_START,
    TRAIN_END,
    VALIDATION_END,
    get_settings,
)


def test_package_imports() -> None:
    assert quantlab.__version__


def test_split_boundaries_are_ordered() -> None:
    assert DATA_START < TRAIN_END < VALIDATION_END < HOLDOUT_START


def test_holdout_is_locked_to_2024() -> None:
    assert date(2024, 1, 1) == HOLDOUT_START


def test_settings_default_to_locked_holdout() -> None:
    settings = get_settings()
    assert settings.unlock_holdout is False
