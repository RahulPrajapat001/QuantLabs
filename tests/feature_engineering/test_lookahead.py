"""Look-ahead protection — non-negotiable, over the ENTIRE registry.

For every registered feature, the value at time *t* computed on the full series
must equal the value computed on the series truncated at *t*. Any feature that
reads a future row changes when the future is removed, and fails here. This is a
parametrised sweep over the whole registry (PRD Module 3), not a spot check:

* time-series features are truncated by row on one security's history;
* cross-sectional features are truncated by date on a multi-security panel.

A failure fails the build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.feature_engineering import (
    REGISTRY,
    FeatureKind,
    compute_cross_sectional_features,
    compute_time_series_features,
)

TIME_SERIES_SPECS = REGISTRY.by_kind(FeatureKind.TIME_SERIES)
CROSS_SECTIONAL_SPECS = REGISTRY.by_kind(FeatureKind.CROSS_SECTIONAL)

# Interior points well past the deepest lookback (252), so values are non-NaN and
# each truncation genuinely discards a large future tail.
TS_PROBES = (300, 450, 599)


def _values_equal(a: float, b: float) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True))


@pytest.mark.parametrize(
    "spec", TIME_SERIES_SPECS, ids=[s.name for s in TIME_SERIES_SPECS]
)
def test_time_series_feature_has_no_lookahead(spec, security_df: pd.DataFrame) -> None:
    full = compute_time_series_features(security_df, [spec.name])[spec.name]
    for t in TS_PROBES:
        truncated = compute_time_series_features(
            security_df.iloc[: t + 1], [spec.name]
        )[spec.name]
        full_val = float(full.iloc[t])
        trunc_val = float(truncated.iloc[t])
        assert _values_equal(full_val, trunc_val), (
            f"{spec.name} look-ahead at t={t}: full={full_val} truncated={trunc_val}"
        )


@pytest.mark.parametrize(
    "spec", CROSS_SECTIONAL_SPECS, ids=[s.name for s in CROSS_SECTIONAL_SPECS]
)
def test_cross_sectional_feature_has_no_lookahead(spec, panel_df: pd.DataFrame) -> None:
    dates = sorted(pd.to_datetime(panel_df["date"]).unique())
    full = compute_cross_sectional_features(panel_df, [spec.name])

    for date_idx in (40, len(dates) - 1):
        cutoff = dates[date_idx]
        truncated_panel = panel_df[pd.to_datetime(panel_df["date"]) <= cutoff]
        truncated = compute_cross_sectional_features(truncated_panel, [spec.name])

        full_at_t = (
            full[pd.to_datetime(full["date"]) == cutoff]
            .set_index("security_id")[spec.name]
            .sort_index()
        )
        trunc_at_t = (
            truncated[pd.to_datetime(truncated["date"]) == cutoff]
            .set_index("security_id")[spec.name]
            .sort_index()
        )
        assert list(full_at_t.index) == list(trunc_at_t.index)
        for sec in full_at_t.index:
            assert _values_equal(full_at_t[sec], trunc_at_t[sec]), (
                f"{spec.name} cross-sectional look-ahead at {cutoff} for {sec}"
            )
