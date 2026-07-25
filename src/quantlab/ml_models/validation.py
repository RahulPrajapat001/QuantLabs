"""Purged cross-validation and walk-forward splitting — the point of Module 11.

Standard k-fold on a time series is not a slightly optimistic estimate; it is a
broken one. Two mechanisms break it, and they are different:

1. **Shuffling.** Randomly assigning rows to folds trains on Thursday and tests
   on Wednesday. Every splitter here cuts **contiguous blocks of dates**, never
   rows, so a fold boundary is a moment in time. On a panel this also means a
   single date never appears in both train and test — otherwise the model learns
   Reliance's Tuesday and is scored on Infosys's Tuesday, in a market where the
   cross-section is correlated by construction.

2. **Label overlap.** A sample dated *t* carries a label that resolves at
   *t + h*. If *t + h* lands inside the test block, that training sample and the
   test block share a price. This survives contiguous splitting untouched, and it
   is what **purging** removes: any training sample whose label window
   ``[t0, t1]`` intersects the test window is dropped
   (López de Prado, *Advances in Financial Machine Learning*, ch. 7).

**Embargo** handles what purging cannot. Features are serially correlated, so a
sample drawn immediately *after* the test block still carries information from
inside it even though its label window does not overlap. A fixed number of dates
following each test block is therefore also excluded from training. Note the
asymmetry: in :class:`WalkForward` the training data lies entirely before the test
block, so there is nothing after it to embargo — the equivalent protection is the
purge at the boundary plus an optional ``gap_days`` buffer, and calling that an
"embargo" would be borrowing the word for a different mechanism.

Every fold reports ``n_purged`` and ``n_embargoed``. A purged CV that removed zero
samples has not protected anything, and the counts are how that gets noticed
instead of assumed. Each splitter also re-checks its own output before returning:
if any training sample's label window overlaps its test window, :meth:`split`
raises rather than handing back a fold that leaks.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, runtime_checkable

import numpy as np
import structlog

from quantlab.ml_models.dataset import SampleTimes

log = structlog.get_logger(__name__)

__all__ = [
    "Fold",
    "LeakageError",
    "PurgedKFold",
    "Splitter",
    "WalkForward",
    "assert_no_label_leakage",
]


class LeakageError(RuntimeError):
    """Raised when a produced fold would train on data overlapping its test set."""


def _as_date(value: np.datetime64) -> date:
    converted = value.astype("datetime64[D]").astype(date)
    assert isinstance(converted, date)
    return converted


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/test split, with the evidence that it was purged."""

    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    #: Latest date any *test* label resolves on — the true end of the test
    #: window, later than ``test_end`` by up to the target horizon.
    test_label_end: date
    #: Samples dropped from training because their label window overlapped the
    #: test window.
    n_purged: int
    #: Samples dropped from training because they fell in the embargo (or gap)
    #: period, over and above those already purged.
    n_embargoed: int

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_test(self) -> int:
        return int(self.test_idx.size)

    def render(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start.isoformat()}"
            f"..{self.train_end.isoformat()} ({self.n_train:,}) | "
            f"test {self.test_start.isoformat()}..{self.test_end.isoformat()} "
            f"({self.n_test:,}, labels to {self.test_label_end.isoformat()}) | "
            f"purged {self.n_purged:,}, embargoed {self.n_embargoed:,}"
        )


@runtime_checkable
class Splitter(Protocol):
    """Anything that cuts a dataset's sample times into leak-free folds."""

    def split(self, times: SampleTimes) -> list[Fold]: ...

    def describe(self) -> str: ...

    def parameters(self) -> dict[str, Any]: ...


def _overlap_mask(times: SampleTimes, test_lo: Any, test_label_hi: Any) -> np.ndarray:
    """True for samples whose label window intersects ``[test_lo, test_label_hi]``.

    Interval intersection, not membership: a sample dated before the test block
    whose label resolves inside it overlaps, and so does one dated inside the
    block's label tail whose own label resolves after it.
    """

    return np.asarray((times.t1 >= test_lo) & (times.t0 <= test_label_hi), dtype=bool)


def assert_no_label_leakage(fold: Fold, times: SampleTimes) -> None:
    """Raise unless every training sample's label window clears the test window.

    Called by each splitter on its own output. Cheap (one vectorised comparison
    per fold) and worth it: a purge implemented with an off-by-one on the
    inequality still produces plausible-looking folds and quietly restores most
    of the leak it was written to remove.
    """

    if fold.train_idx.size == 0:
        return
    test_lo = np.datetime64(fold.test_start)
    test_hi = np.datetime64(fold.test_label_end)
    train_t0 = times.t0[fold.train_idx]
    train_t1 = times.t1[fold.train_idx]
    overlapping = int(np.count_nonzero((train_t1 >= test_lo) & (train_t0 <= test_hi)))
    if overlapping:
        raise LeakageError(
            f"fold {fold.index} leaves {overlapping} training sample(s) whose "
            f"label window overlaps the test window "
            f"[{fold.test_start.isoformat()}, {fold.test_label_end.isoformat()}]. "
            "The fold is not usable: any metric computed on it is contaminated."
        )


def _fold_dates(times: SampleTimes, idx: np.ndarray) -> tuple[date, date]:
    selected = times.t0[idx]
    return _as_date(selected.min()), _as_date(selected.max())


@dataclass(frozen=True, slots=True)
class PurgedKFold:
    """K contiguous date blocks, with label-overlap purging and an embargo.

    ``embargo_pct`` is expressed as a fraction of the number of **unique dates**
    in the sample, following the convention in López de Prado ch. 7 — 1% of a
    seven-year daily panel is roughly two trading weeks. Pass ``embargo_days`` to
    state it in trading days instead, which is easier to reason about when the
    features have a known lookback.

    The folds are returned in chronological order, which makes the per-fold
    metrics readable as a time series. That is presentation only: unlike
    :class:`WalkForward`, a k-fold trains on data that *follows* its test block,
    so its estimate is not a simulation of live trading. It is a more efficient
    use of a short sample, and the two answer different questions — run both.
    """

    n_splits: int = 5
    embargo_pct: float = 0.01
    embargo_days: int | None = None
    min_train_samples: int = 1

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {self.n_splits}")
        if not 0.0 <= self.embargo_pct < 1.0:
            raise ValueError(
                f"embargo_pct is a fraction of the date range and must lie in "
                f"[0, 1), got {self.embargo_pct}"
            )
        if self.embargo_days is not None and self.embargo_days < 0:
            raise ValueError(
                f"embargo_days must be non-negative, got {self.embargo_days}"
            )

    def describe(self) -> str:
        embargo = (
            f"{self.embargo_days} days"
            if self.embargo_days is not None
            else f"{self.embargo_pct:.1%} of dates"
        )
        return f"PurgedKFold(n_splits={self.n_splits}, embargo={embargo})"

    def parameters(self) -> dict[str, Any]:
        return {
            "splitter": "PurgedKFold",
            "n_splits": self.n_splits,
            "embargo_pct": self.embargo_pct,
            "embargo_days": self.embargo_days,
        }

    def _embargo_length(self, n_dates: int) -> int:
        if self.embargo_days is not None:
            return int(self.embargo_days)
        return math.ceil(self.embargo_pct * n_dates)

    def split(self, times: SampleTimes) -> list[Fold]:
        dates = times.unique_dates
        n_dates = int(dates.size)
        if n_dates < self.n_splits:
            raise ValueError(
                f"cannot cut {self.n_splits} folds from {n_dates} distinct dates; "
                "folds are blocks of dates, not blocks of rows"
            )

        embargo_n = self._embargo_length(n_dates)
        blocks = np.array_split(np.arange(n_dates), self.n_splits)
        folds: list[Fold] = []

        for i, block in enumerate(blocks):
            test_lo = dates[block[0]]
            test_hi = dates[block[-1]]
            test_mask = (times.t0 >= test_lo) & (times.t0 <= test_hi)
            test_idx = np.flatnonzero(test_mask)
            if test_idx.size == 0:  # pragma: no cover - blocks come from t0 itself
                continue
            label_hi = times.t1[test_idx].max()

            overlap = _overlap_mask(times, test_lo, label_hi)
            purge_mask = ~test_mask & overlap

            embargo_mask = np.zeros(len(times), dtype=bool)
            if embargo_n > 0:
                start = int(np.searchsorted(dates, label_hi, side="right"))
                stop = min(start + embargo_n, n_dates)
                if start < stop:
                    embargo_mask = (
                        ~test_mask
                        & ~overlap
                        & (times.t0 >= dates[start])
                        & (times.t0 <= dates[stop - 1])
                    )

            train_mask = ~test_mask & ~overlap & ~embargo_mask
            train_idx = np.flatnonzero(train_mask)
            if train_idx.size < self.min_train_samples:
                raise ValueError(
                    f"fold {i} has {train_idx.size} training samples after purging "
                    f"and embargo (minimum {self.min_train_samples}). Reduce "
                    "n_splits, shorten the embargo, or extend the sample — a fold "
                    "is not silently skipped, because dropping it would change "
                    "what the reported average is an average of."
                )

            train_start, train_end = _fold_dates(times, train_idx)
            fold = Fold(
                index=i,
                train_idx=train_idx,
                test_idx=test_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=_as_date(test_lo),
                test_end=_as_date(test_hi),
                test_label_end=_as_date(label_hi),
                n_purged=int(purge_mask.sum()),
                n_embargoed=int(embargo_mask.sum()),
            )
            assert_no_label_leakage(fold, times)
            folds.append(fold)

        log.info(
            "ml.validation.purged_kfold",
            n_splits=len(folds),
            embargo_dates=embargo_n,
            total_purged=sum(f.n_purged for f in folds),
            total_embargoed=sum(f.n_embargoed for f in folds),
        )
        return folds


@dataclass(frozen=True, slots=True)
class WalkForward:
    """Expanding or rolling walk-forward: train on the past, test on what follows.

    This is the simulation of live trading. ``expanding`` re-fits on everything
    known so far — the natural choice when the sample is short and the data
    generating process is assumed stable. ``rolling`` holds the training window
    at a fixed length, which is the right assumption when it is not: a model
    fitted through 2015–2019 and applied in 2021 is being asked to generalise
    across a regime change that a rolling window would have forgotten.

    Which one is correct is an empirical question about the market, so both are
    implemented and PRD Module 11 requires both to be run.

    ``gap_days`` inserts an explicit buffer between the training window and the
    test block, on top of the mandatory label purge. Purging is not optional and
    is not configurable — a walk-forward that trains on a sample whose label
    resolves after the test block begins is simply wrong.
    """

    mode: str = "expanding"
    n_splits: int = 5
    #: Fraction of the date grid reserved for the first training window. Ignored
    #: when ``train_days`` is given.
    initial_train_frac: float = 0.5
    #: Length of the training window in trading days. Required for ``rolling``;
    #: for ``expanding`` it sets the size of the first window only.
    train_days: int | None = None
    gap_days: int = 0
    min_train_samples: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"expanding", "rolling"}:
            raise ValueError(
                f"mode must be 'expanding' or 'rolling', got {self.mode!r}"
            )
        if self.n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {self.n_splits}")
        if self.mode == "rolling" and self.train_days is None:
            raise ValueError(
                "rolling walk-forward needs train_days: the whole difference from "
                "the expanding variant is that the window has a fixed length"
            )
        if not 0.0 < self.initial_train_frac < 1.0:
            raise ValueError(
                f"initial_train_frac must lie in (0, 1), got {self.initial_train_frac}"
            )
        if self.gap_days < 0:
            raise ValueError(f"gap_days must be non-negative, got {self.gap_days}")

    def describe(self) -> str:
        window = (
            f"rolling {self.train_days}d"
            if self.mode == "rolling"
            else f"expanding from {self.initial_train_frac:.0%}"
        )
        return f"WalkForward({window}, n_splits={self.n_splits}, gap={self.gap_days}d)"

    def parameters(self) -> dict[str, Any]:
        return {
            "splitter": "WalkForward",
            "mode": self.mode,
            "n_splits": self.n_splits,
            "initial_train_frac": self.initial_train_frac,
            "train_days": self.train_days,
            "gap_days": self.gap_days,
        }

    def split(self, times: SampleTimes) -> list[Fold]:
        dates = times.unique_dates
        n_dates = int(dates.size)

        initial = (
            int(self.train_days)
            if self.train_days is not None
            else round(self.initial_train_frac * n_dates)
        )
        initial = max(1, min(initial, n_dates - 1))
        remaining = n_dates - initial
        test_size = remaining // self.n_splits
        if test_size < 1:
            raise ValueError(
                f"{remaining} dates remain after the initial training window of "
                f"{initial}, which cannot be cut into {self.n_splits} test blocks. "
                "Shorten the training window, reduce n_splits, or extend the sample."
            )

        folds: list[Fold] = []
        for i in range(self.n_splits):
            lo_pos = initial + i * test_size
            # The final block absorbs the remainder so no date goes unused.
            hi_pos = n_dates - 1 if i == self.n_splits - 1 else lo_pos + test_size - 1
            test_lo = dates[lo_pos]
            test_hi = dates[hi_pos]

            test_mask = (times.t0 >= test_lo) & (times.t0 <= test_hi)
            test_idx = np.flatnonzero(test_mask)
            if test_idx.size == 0:  # pragma: no cover - defensive
                continue
            label_hi = times.t1[test_idx].max()

            # Candidate training window: strictly before the test block.
            train_lo_pos = (
                max(0, lo_pos - int(self.train_days))
                if self.mode == "rolling" and self.train_days is not None
                else 0
            )
            candidates = (times.t0 >= dates[train_lo_pos]) & (times.t0 < test_lo)

            overlap = _overlap_mask(times, test_lo, label_hi)
            purge_mask = candidates & overlap

            gap_mask = np.zeros(len(times), dtype=bool)
            if self.gap_days > 0:
                gap_start = dates[max(train_lo_pos, lo_pos - self.gap_days)]
                gap_mask = candidates & ~overlap & (times.t0 >= gap_start)

            train_mask = candidates & ~overlap & ~gap_mask
            train_idx = np.flatnonzero(train_mask)
            if train_idx.size < self.min_train_samples:
                raise ValueError(
                    f"walk-forward fold {i} has {train_idx.size} training samples "
                    f"after purging (minimum {self.min_train_samples}); the "
                    "initial window is too short for the target horizon"
                )

            train_start, train_end = _fold_dates(times, train_idx)
            fold = Fold(
                index=i,
                train_idx=train_idx,
                test_idx=test_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=_as_date(test_lo),
                test_end=_as_date(test_hi),
                test_label_end=_as_date(label_hi),
                n_purged=int(purge_mask.sum()),
                n_embargoed=int(gap_mask.sum()),
            )
            assert_no_label_leakage(fold, times)
            folds.append(fold)

        log.info(
            "ml.validation.walk_forward",
            mode=self.mode,
            n_splits=len(folds),
            total_purged=sum(f.n_purged for f in folds),
        )
        return folds


def render_folds(folds: Sequence[Fold]) -> str:
    """A readable fold table — what ``quantlab ml folds`` prints."""

    if not folds:
        return "(no folds)"
    lines = [f.render() for f in folds]
    lines.append(
        f"total: {sum(f.n_train for f in folds):,} train / "
        f"{sum(f.n_test for f in folds):,} test samples, "
        f"{sum(f.n_purged for f in folds):,} purged, "
        f"{sum(f.n_embargoed for f in folds):,} embargoed"
    )
    return "\n".join(lines)


def iter_folds(splitter: Splitter, times: SampleTimes) -> Iterator[Fold]:
    """Yield folds one at a time, for callers that do not want the list."""

    yield from splitter.split(times)
