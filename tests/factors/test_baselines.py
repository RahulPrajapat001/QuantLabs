"""Baseline factors reference the right features with the right sign."""

from __future__ import annotations

from datetime import date

import pandas as pd

from quantlab.factors.baselines import (
    baseline_factors,
    low_volatility_factor,
    momentum_12_1_factor,
    short_term_reversal_factor,
    volume_factor,
)


def test_baseline_set_is_the_four_named_factors() -> None:
    names = {f.name for f in baseline_factors()}
    assert names == {
        "momentum_12_1",
        "short_term_reversal",
        "low_volatility",
        "volume",
    }


def test_baselines_reference_expected_features_and_directions() -> None:
    assert momentum_12_1_factor().feature == "momentum_12_1"
    assert momentum_12_1_factor().direction == 1
    # Reversal, low-vol and volume are all contrarian (sign-flipped).
    assert short_term_reversal_factor().direction == -1
    assert short_term_reversal_factor().feature == "return_5d"
    assert low_volatility_factor().direction == -1
    assert low_volatility_factor().feature == "volatility_20d"
    assert volume_factor().direction == -1


def test_low_volatility_scores_calm_names_high() -> None:
    # Two dates, four names with increasing volatility feature values.
    rows = []
    for d in (date(2018, 1, 1), date(2018, 1, 2)):
        for i, vol in enumerate([0.1, 0.2, 0.3, 0.4]):
            rows.append({"date": d, "security_id": f"SEC-{i}", "volatility_20d": vol})
    panel = pd.DataFrame(rows)

    scores = low_volatility_factor().compute(panel)
    day = scores[scores["date"] == date(2018, 1, 1)].set_index("security_id")
    # Lowest volatility (SEC-0) gets the highest score; highest vol the lowest.
    assert day.loc["SEC-0", "score"] > day.loc["SEC-3", "score"]


def test_momentum_12_1_is_a_registered_feature() -> None:
    from quantlab.feature_engineering import REGISTRY

    assert "momentum_12_1" in REGISTRY
    spec = REGISTRY.get("momentum_12_1")
    assert spec.category == "momentum"
    assert spec.lookback == 252
