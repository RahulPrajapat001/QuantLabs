"""Dataset assembly: targets, label dates, the universe restriction, the lock."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.config import HoldoutViolationError, Split
from quantlab.ml_models.dataset import (
    TargetKind,
    TargetSpec,
    build_dataset,
    join_prices,
    restrict_to_universe,
)
from quantlab.ml_models.models import Task

from .conftest import FEATURES, full_universe, make_panel


def tiny_panel(*, n_days: int = 12, n_securities: int = 6, start="2016-01-04"):
    """A panel with prices that step by exactly 1% a day, so targets are checkable."""

    dates = pd.bdate_range(start=start, periods=n_days)
    frames = []
    for i in range(n_securities):
        close = 100.0 * (1.01 ** np.arange(n_days))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "security_id": f"SEC{i:03d}",
                    "adj_close": close,
                    "signal": np.arange(n_days, dtype=float) + i,
                    "momentum_12_1": np.linspace(-1.0, 1.0, n_days) + i,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- Targets -----------------------------------------------------------------


def test_next_day_return_is_the_realised_forward_return():
    panel = tiny_panel()
    dataset = build_dataset(
        panel,
        features=["signal", "momentum_12_1"],
        universe=full_universe(panel),
        index_name="T",
    )
    assert dataset.task is Task.REGRESSION
    assert np.allclose(dataset.y.to_numpy(), 0.01)


def test_direction_target_is_the_sign_of_that_return():
    panel = tiny_panel()
    dataset = build_dataset(
        panel,
        features=["signal", "momentum_12_1"],
        universe=full_universe(panel),
        index_name="T",
        target=TargetSpec(kind=TargetKind.DIRECTION),
    )
    assert dataset.task is Task.CLASSIFICATION
    assert set(dataset.y.unique()) == {1}
    # The forward return is retained even for the classification target: the
    # direction is what the model predicts, the return is what a book earns.
    assert np.allclose(dataset.forward_return, 0.01)


def test_label_date_is_the_security_s_own_next_trading_date():
    panel = tiny_panel(n_days=8)
    dataset = build_dataset(
        panel,
        features=["signal", "momentum_12_1"],
        universe=full_universe(panel),
        index_name="T",
    )
    frame = dataset.frame()
    dates = sorted(pd.to_datetime(panel["date"]).unique())
    for _, row in frame.iterrows():
        position = dates.index(pd.Timestamp(row["date"]))
        assert pd.Timestamp(row["label_date"]) == dates[position + 1]


def test_a_longer_horizon_pushes_the_label_date_out():
    panel = tiny_panel(n_days=20)
    dataset = build_dataset(
        panel,
        features=["signal", "momentum_12_1"],
        universe=full_universe(panel),
        index_name="T",
        target=TargetSpec(horizon=5),
    )
    assert np.allclose(dataset.y.to_numpy(), 1.01**5 - 1.0)
    gaps = (dataset.times.t1 - dataset.times.t0).astype("timedelta64[D]").astype(int)
    assert gaps.min() >= 5


def test_the_last_rows_have_no_label_and_are_dropped():
    panel = tiny_panel(n_days=10, n_securities=6)
    dataset = build_dataset(
        panel,
        features=["signal", "momentum_12_1"],
        universe=full_universe(panel),
        index_name="T",
        target=TargetSpec(horizon=3),
    )
    # 10 dates, 3-day horizon: the last 3 cross-sections cannot be labelled.
    assert dataset.end == pd.Timestamp(sorted(panel["date"].unique())[-4]).date()


def test_horizon_must_be_positive():
    with pytest.raises(ValueError, match="at least 1 trading day"):
        TargetSpec(horizon=0)


# --- The point-in-time universe ----------------------------------------------


def test_a_security_absent_from_the_index_is_never_a_sample():
    panel = tiny_panel()
    universe = full_universe(panel)
    excluded = "SEC003"
    dropped_dates = sorted(universe)[:5]
    for day in dropped_dates:
        universe[day] = universe[day] - {excluded}

    dataset = build_dataset(
        panel,
        features=["signal", "momentum_12_1"],
        universe=universe,
        index_name="T",
    )
    frame = dataset.frame()
    for day in dropped_dates:
        rows = frame[frame["date"] == pd.Timestamp(day)]
        assert excluded not in set(rows["security_id"])
    # …and is still present on the dates it *was* a member.
    assert excluded in set(frame["security_id"])


def test_a_date_with_no_membership_record_raises_rather_than_keeping_everything():
    panel = tiny_panel()
    universe = full_universe(panel)
    del universe[sorted(universe)[3]]
    with pytest.raises(ValueError, match="no point-in-time universe record"):
        restrict_to_universe(panel, universe)


def test_the_forward_return_is_computed_before_the_universe_restriction():
    """A security's return is a property of its price series, not its membership.

    Restricting first would compute the "next day" return across the gap left by
    a security's absence from the index, which is a different number from the one
    the security actually returned.
    """

    panel = tiny_panel(n_days=10)
    universe = full_universe(panel)
    missing_day = sorted(universe)[4]
    universe[missing_day] = universe[missing_day] - {"SEC000"}

    dataset = build_dataset(
        panel,
        features=["signal", "momentum_12_1"],
        universe=universe,
        index_name="T",
    )
    rows = dataset.frame()
    theirs = rows[rows["security_id"] == "SEC000"]
    assert np.allclose(theirs["forward_return"].to_numpy(), 0.01)


def test_a_date_with_too_few_names_is_dropped():
    panel = tiny_panel(n_days=10, n_securities=6)
    universe = full_universe(panel)
    thin_day = sorted(universe)[2]
    universe[thin_day] = set(sorted(universe[thin_day])[:2])
    dataset = build_dataset(
        panel,
        features=["signal", "momentum_12_1"],
        universe=universe,
        index_name="T",
        min_names_per_date=5,
    )
    assert pd.Timestamp(thin_day) not in set(dataset.times.t0.tolist())


# --- The holdout lock --------------------------------------------------------


def test_a_label_reaching_into_the_holdout_is_refused():
    """The lock is checked against the label date, not the feature date.

    Features stop in 2023 here. The one-day target still reads a 2024 price, and
    2024 is locked.
    """

    panel = tiny_panel(n_days=11, start="2023-12-18")
    panel_dates = sorted(panel["date"].unique())
    # The last date that can carry a label is 2023-12-29 — comfortably outside the
    # lock. Its label is read off 2024-01-01, which is not.
    assert pd.Timestamp(panel_dates[-2]) < pd.Timestamp("2024-01-01")
    assert pd.Timestamp(panel_dates[-1]) >= pd.Timestamp("2024-01-01")
    with pytest.raises(HoldoutViolationError, match="holdout"):
        build_dataset(
            panel,
            features=["signal", "momentum_12_1"],
            universe=full_universe(panel),
            index_name="T",
        )


def test_a_dataset_stopping_short_of_the_boundary_is_allowed():
    panel = tiny_panel(n_days=10, start="2023-12-01")
    dataset = build_dataset(
        panel,
        features=["signal", "momentum_12_1"],
        universe=full_universe(panel),
        index_name="T",
    )
    assert dataset.label_end < date(2024, 1, 1)


# --- Split masks -------------------------------------------------------------


def test_a_sample_straddling_a_split_boundary_belongs_to_neither():
    dates = pd.bdate_range("2021-12-27", periods=6)  # crosses into 2022
    frames = []
    for i in range(6):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "security_id": f"SEC{i:03d}",
                    "adj_close": 100.0 * (1.01 ** np.arange(len(dates))),
                    "signal": np.arange(len(dates), dtype=float),
                    "momentum_12_1": np.arange(len(dates), dtype=float),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    dataset = build_dataset(
        panel,
        features=["signal", "momentum_12_1"],
        universe=full_universe(panel),
        index_name="T",
    )

    train = dataset.split_mask(Split.TRAIN)
    validation = dataset.split_mask(Split.VALIDATION)
    straddling = (dataset.times.t0 <= np.datetime64("2021-12-31")) & (
        dataset.times.t1 >= np.datetime64("2022-01-01")
    )
    assert straddling.any()
    assert not (train & straddling).any()
    assert not (validation & straddling).any()
    assert not (train & validation).any()


def test_split_of_an_empty_range_raises_with_the_covered_window():
    panel = make_panel(n_days=120, start=date(2016, 1, 4))
    dataset = build_dataset(
        panel, features=FEATURES, universe=full_universe(panel), index_name="T"
    )
    with pytest.raises(ValueError, match="validation split of this dataset is empty"):
        dataset.split(Split.VALIDATION)


# --- Assembly errors ---------------------------------------------------------


def test_missing_feature_columns_are_named():
    panel = tiny_panel()
    with pytest.raises(KeyError, match="nonexistent"):
        build_dataset(
            panel,
            features=["signal", "nonexistent"],
            universe=full_universe(panel),
            index_name="T",
        )


def test_a_dataset_needs_at_least_one_feature():
    panel = tiny_panel()
    with pytest.raises(ValueError, match="at least one feature"):
        build_dataset(panel, features=[], universe=full_universe(panel), index_name="T")


def test_join_prices_aligns_features_to_the_price_panel():
    panel = tiny_panel(n_days=6)
    features = panel[["date", "security_id", "signal", "momentum_12_1"]]
    prices = panel[["date", "security_id", "adj_close"]]
    joined = join_prices(features, prices)
    assert set(joined.columns) == {
        "date",
        "security_id",
        "signal",
        "momentum_12_1",
        "adj_close",
    }
    assert len(joined) == len(features)


def test_subset_preserves_alignment(dataset):
    subset = dataset.subset(np.arange(0, len(dataset), 3))
    assert len(subset) == len(subset.X) == len(subset.times)
    assert subset.security_ids.size == len(subset)
    assert subset.target == dataset.target


def test_describe_states_the_feature_and_label_windows(dataset):
    text = dataset.describe()
    assert "labels through" in text
    assert dataset.label_end.isoformat() in text
