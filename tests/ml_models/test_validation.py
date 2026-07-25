"""The validation tests — the ones that decide whether Module 11 means anything.

A purged CV that silently fails to purge produces exactly the results a working
one produces, only better, which is why these tests check the *mechanism* rather
than the output: no shared dates, no overlapping label windows, an embargo that
actually removes rows, and a self-check that catches a fold built wrongly by hand.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.ml_models.dataset import SampleTimes
from quantlab.ml_models.validation import (
    Fold,
    LeakageError,
    PurgedKFold,
    WalkForward,
    assert_no_label_leakage,
    render_folds,
)


def make_times(*, n_dates: int = 200, n_securities: int = 5, horizon: int = 5):
    """Panel sample times: every security observed on every date."""

    dates = pd.bdate_range("2016-01-04", periods=n_dates + horizon)
    t0: list[np.datetime64] = []
    t1: list[np.datetime64] = []
    for i in range(n_dates):
        for _ in range(n_securities):
            t0.append(dates[i].to_datetime64())
            t1.append(dates[i + horizon].to_datetime64())
    return SampleTimes(
        t0=np.array(t0, dtype="datetime64[ns]"),
        t1=np.array(t1, dtype="datetime64[ns]"),
    )


# --- The core guarantees -----------------------------------------------------


def test_no_date_appears_in_both_train_and_test():
    """Folds are blocks of dates, not blocks of rows.

    On a panel, splitting by row puts Reliance's Tuesday in training and Infosys's
    Tuesday in test. The cross-section is correlated by construction, so that is a
    leak even though no single row is shared.
    """

    times = make_times()
    for fold in PurgedKFold(n_splits=5).split(times):
        train_dates = set(times.t0[fold.train_idx].tolist())
        test_dates = set(times.t0[fold.test_idx].tolist())
        assert not (train_dates & test_dates)


def test_purging_removes_samples_whose_labels_overlap_the_test_block():
    """The whole point: a five-day label spanning the fold boundary must go.

    The unpurged split — everything outside the test block — is computed here as
    the counterfactual, so the test proves purging removed something real rather
    than asserting that a number is non-zero for its own sake.
    """

    horizon = 5
    times = make_times(horizon=horizon)
    folds = PurgedKFold(n_splits=5, embargo_days=0).split(times)

    for fold in folds:
        test_lo = np.datetime64(fold.test_start)
        test_hi = np.datetime64(fold.test_label_end)

        naive_train = np.setdiff1d(np.arange(len(times)), fold.test_idx)
        naive_overlap = int(
            np.count_nonzero(
                (times.t1[naive_train] >= test_lo) & (times.t0[naive_train] <= test_hi)
            )
        )
        assert naive_overlap > 0, "the counterfactual must actually leak"
        assert fold.n_purged == naive_overlap

        purged_overlap = int(
            np.count_nonzero(
                (times.t1[fold.train_idx] >= test_lo)
                & (times.t0[fold.train_idx] <= test_hi)
            )
        )
        assert purged_overlap == 0


def test_horizon_one_still_purges_the_boundary_sample():
    """Even a next-day target leaks one cross-section across each boundary."""

    times = make_times(horizon=1)
    folds = PurgedKFold(n_splits=4, embargo_days=0).split(times)
    assert sum(f.n_purged for f in folds) > 0


def test_embargo_excludes_dates_following_the_test_block():
    times = make_times(horizon=5)
    embargo = 10
    folds = PurgedKFold(n_splits=5, embargo_days=embargo).split(times)
    dates = times.unique_dates

    assert sum(f.n_embargoed for f in folds) > 0
    for fold in folds:
        label_end = np.datetime64(fold.test_label_end)
        start = int(np.searchsorted(dates, label_end, side="right"))
        stop = min(start + embargo, dates.size)
        if start >= stop:
            continue
        embargoed = set(dates[start:stop].tolist())
        assert not (set(times.t0[fold.train_idx].tolist()) & embargoed)


def test_zero_embargo_embargoes_nothing():
    times = make_times()
    folds = PurgedKFold(n_splits=5, embargo_days=0).split(times)
    assert all(f.n_embargoed == 0 for f in folds)


def test_embargo_pct_is_a_fraction_of_the_date_grid():
    times = make_times(n_dates=200)
    folds = PurgedKFold(n_splits=5, embargo_pct=0.05).split(times)
    # 5% of 205 distinct dates = 11 dates embargoed after each test block, except
    # the final block, which has nothing after it.
    assert sum(f.n_embargoed for f in folds) > 0
    assert folds[-1].n_embargoed == 0


def test_folds_are_deterministic():
    times = make_times()
    first = PurgedKFold(n_splits=5).split(times)
    second = PurgedKFold(n_splits=5).split(times)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.train_idx, b.train_idx)
        assert np.array_equal(a.test_idx, b.test_idx)


def test_every_sample_is_tested_exactly_once():
    times = make_times()
    folds = PurgedKFold(n_splits=5).split(times)
    tested = np.concatenate([f.test_idx for f in folds])
    assert np.array_equal(np.sort(tested), np.arange(len(times)))


# --- The self-check ----------------------------------------------------------


def test_assert_no_label_leakage_rejects_a_hand_built_leaky_fold():
    """The guard the splitters run on their own output has to actually fire."""

    times = make_times(n_dates=50, n_securities=2, horizon=5)
    leaky = Fold(
        index=0,
        train_idx=np.arange(len(times)),  # everything, including the test window
        test_idx=np.arange(10),
        train_start=date(2016, 1, 4),
        train_end=date(2016, 3, 1),
        test_start=pd.Timestamp(times.t0[0]).date(),
        test_end=pd.Timestamp(times.t0[9]).date(),
        test_label_end=pd.Timestamp(times.t1[9]).date(),
        n_purged=0,
        n_embargoed=0,
    )
    with pytest.raises(LeakageError, match="label window overlaps"):
        assert_no_label_leakage(leaky, times)


# --- Configuration errors ----------------------------------------------------


def test_too_few_dates_for_the_requested_folds_raises():
    times = make_times(n_dates=3, n_securities=50)
    with pytest.raises(ValueError, match="blocks of dates, not blocks of rows"):
        PurgedKFold(n_splits=5).split(times)


def test_an_embargo_that_empties_a_training_fold_raises_rather_than_skipping():
    times = make_times(n_dates=40, n_securities=2, horizon=5)
    with pytest.raises(ValueError, match="training samples after purging"):
        PurgedKFold(n_splits=2, embargo_days=100, min_train_samples=1).split(times)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_splits": 1}, "n_splits must be at least 2"),
        ({"embargo_pct": 1.5}, "embargo_pct"),
        ({"embargo_days": -1}, "embargo_days must be non-negative"),
    ],
)
def test_purged_kfold_validates_its_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PurgedKFold(**kwargs)


# --- Walk-forward ------------------------------------------------------------


def test_walk_forward_never_trains_on_the_future():
    times = make_times()
    for mode, kwargs in (("expanding", {}), ("rolling", {"train_days": 60})):
        folds = WalkForward(mode=mode, n_splits=4, **kwargs).split(times)
        assert folds
        for fold in folds:
            assert fold.train_end < fold.test_start


def test_walk_forward_expanding_training_window_grows():
    times = make_times()
    folds = WalkForward(mode="expanding", n_splits=4).split(times)
    sizes = [f.n_train for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]
    # Expanding means the window keeps its original start.
    assert len({f.train_start for f in folds}) == 1


def test_walk_forward_rolling_window_stays_a_fixed_length():
    times = make_times()
    folds = WalkForward(mode="rolling", n_splits=4, train_days=60).split(times)
    starts = [f.train_start for f in folds]
    assert len(set(starts)) == len(starts), "a rolling window must move its start"
    spans = {(f.train_end - f.train_start).days for f in folds}
    # Every window covers the same calendar span up to the purge at its far end.
    assert max(spans) - min(spans) <= 10


def test_walk_forward_purges_the_boundary():
    times = make_times(horizon=5)
    folds = WalkForward(mode="expanding", n_splits=4).split(times)
    assert all(f.n_purged > 0 for f in folds)
    for fold in folds:
        assert_no_label_leakage(fold, times)


def test_walk_forward_gap_days_widens_the_buffer():
    times = make_times(horizon=5)
    without = WalkForward(mode="expanding", n_splits=4).split(times)
    with_gap = WalkForward(mode="expanding", n_splits=4, gap_days=20).split(times)
    assert sum(f.n_embargoed for f in with_gap) > sum(f.n_embargoed for f in without)
    assert all(a.n_train > b.n_train for a, b in zip(without, with_gap, strict=True))


def test_walk_forward_covers_every_date_to_the_end():
    times = make_times()
    folds = WalkForward(mode="expanding", n_splits=4).split(times)
    assert folds[-1].test_end == pd.Timestamp(times.t0.max()).date()


def test_rolling_walk_forward_requires_a_window_length():
    with pytest.raises(ValueError, match="needs train_days"):
        WalkForward(mode="rolling")


def test_unknown_walk_forward_mode_raises():
    with pytest.raises(ValueError, match=r"expanding.*rolling"):
        WalkForward(mode="sliding")


# --- Reporting ---------------------------------------------------------------


def test_render_folds_reports_the_purge_counts():
    times = make_times()
    text = render_folds(PurgedKFold(n_splits=3).split(times))
    assert "purged" in text
    assert "embargoed" in text
    assert "fold 0" in text


def test_splitters_expose_their_configuration_for_the_log():
    for splitter in (PurgedKFold(n_splits=3), WalkForward(mode="expanding")):
        params = splitter.parameters()
        assert params["splitter"] in {"PurgedKFold", "WalkForward"}
        assert splitter.describe()
