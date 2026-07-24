"""Factor protocol: cross-sectional standardisation within the universe."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.factors.protocol import FeatureFactor, cross_sectional_zscore


def _panel() -> pd.DataFrame:
    dates = [date(2018, 1, 1), date(2018, 1, 2)]
    rows = []
    for d in dates:
        for i, val in enumerate([1.0, 2.0, 3.0, 100.0]):
            rows.append({"date": d, "security_id": f"SEC-{i}", "feat": val})
    return pd.DataFrame(rows)


def test_scores_are_zero_mean_unit_std_per_date() -> None:
    factor = FeatureFactor("f", "feat", direction=1)
    scores = factor.compute(_panel())
    for _d, rows in scores.groupby("date"):
        assert rows["score"].mean() == pytest.approx(0.0, abs=1e-9)
        assert rows["score"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_direction_flips_sign() -> None:
    up = FeatureFactor("u", "feat", direction=1).compute(_panel())
    down = FeatureFactor("d", "feat", direction=-1).compute(_panel())
    merged = up.merge(down, on=["date", "security_id"], suffixes=("_up", "_down"))
    assert np.allclose(merged["score_up"], -merged["score_down"])


def test_universe_restriction_excludes_nonmembers() -> None:
    factor = FeatureFactor("f", "feat", direction=1)
    universe = {
        date(2018, 1, 1): {"SEC-0", "SEC-1", "SEC-2"},  # SEC-3 excluded
        date(2018, 1, 2): {"SEC-0", "SEC-1", "SEC-2", "SEC-3"},
    }
    scores = factor.compute(_panel(), universe=universe)
    day1 = scores[scores["date"] == date(2018, 1, 1)]
    assert set(day1["security_id"]) == {"SEC-0", "SEC-1", "SEC-2"}
    # The excluded outlier does not distort day-1 standardisation.
    assert day1["score"].max() < 2.0


def test_missing_feature_column_raises() -> None:
    factor = FeatureFactor("f", "not_here", direction=1)
    with pytest.raises(KeyError, match="not_here"):
        factor.compute(_panel())


def test_degenerate_cross_section_is_dropped() -> None:
    # All-equal signal on a date -> zero std -> that date carries no scores.
    frame = pd.DataFrame(
        {
            "date": [date(2018, 1, 1)] * 3,
            "security_id": ["A", "B", "C"],
            "feat": [5.0, 5.0, 5.0],
        }
    )
    scores = cross_sectional_zscore(frame, "feat")
    assert scores.empty
