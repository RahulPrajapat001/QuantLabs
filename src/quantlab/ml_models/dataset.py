"""Supervised datasets built from the point-in-time feature panel.

Two targets, per PRD Module 11: the **next-day return** (regression) and its
**direction** (classification). Everything hard about this module is in the
bookkeeping around those two trivial definitions.

**Every sample carries two timestamps, not one.** ``t0`` is the date whose
features the model sees; ``t1`` is the date its label resolves on. A sample with
``t0 = 2021-12-31`` and horizon 1 is *not* a training observation confined to
2021 — its label is read off 2022's first trading day. Carrying ``t1`` explicitly
is what makes purging possible (:mod:`quantlab.ml_models.validation`), what makes
the split masks honest (a sample straddling the train/validation boundary belongs
to neither), and what makes the holdout guard correct: the lock is checked against
``max(t1)``, because a dataset ending 2023-12-29 with a five-day horizon reads
prices well inside the locked period.

**Rows are restricted to the point-in-time universe.** ``universe`` is required,
not optional. A model that trains on a security which was not in the index on that
date has been handed the index committee's future decision, and the resulting
backtest is measuring survivorship, not skill. Passing ``None`` is not available;
a date with no membership record raises rather than defaulting to "keep
everything".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
import structlog

from quantlab.config import (
    DATA_START,
    HOLDOUT_START,
    TRAIN_END,
    VALIDATION_END,
    Split,
    assert_holdout_allowed,
)
from quantlab.factors.protocol import UniverseByDate
from quantlab.ml_models.models import Task

log = structlog.get_logger(__name__)

__all__ = [
    "MLDataset",
    "SampleTimes",
    "TargetKind",
    "TargetSpec",
    "build_dataset",
    "restrict_to_universe",
]


class TargetKind(StrEnum):
    """The two permitted prediction targets."""

    #: Simple return over the next ``horizon`` trading days.
    RETURN = "return"
    #: 1 if that return is strictly positive, else 0.
    DIRECTION = "direction"

    @property
    def task(self) -> Task:
        return Task.REGRESSION if self is TargetKind.RETURN else Task.CLASSIFICATION


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """What is being predicted, and over what horizon."""

    kind: TargetKind = TargetKind.RETURN
    horizon: int = 1

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(
                f"horizon must be at least 1 trading day, got {self.horizon}"
            )

    @property
    def task(self) -> Task:
        return self.kind.task

    def describe(self) -> str:
        noun = "return" if self.kind is TargetKind.RETURN else "direction"
        window = "next-day" if self.horizon == 1 else f"{self.horizon}-day forward"
        return f"{window} {noun} ({self.task.value})"

    def parameters(self) -> dict[str, Any]:
        return {"target": self.kind.value, "horizon": self.horizon}


@dataclass(frozen=True, slots=True)
class SampleTimes:
    """Per-sample ``(t0, t1)`` pairs — feature date and label resolution date.

    Both are ``datetime64[ns]`` arrays of equal length, aligned to the dataset's
    row order. This is the only input the splitters need: purging is a statement
    about label windows, not about feature values.
    """

    t0: np.ndarray
    t1: np.ndarray

    def __post_init__(self) -> None:
        if self.t0.shape != self.t1.shape:
            raise ValueError(
                f"t0 has {self.t0.size} entries and t1 has {self.t1.size}; every "
                "sample needs both a feature date and a label date"
            )
        if self.t0.size and bool((self.t1 < self.t0).any()):
            raise ValueError(
                "at least one sample has its label resolving before its features "
                "are observed, which is not a forward-looking target"
            )

    def __len__(self) -> int:
        return int(self.t0.size)

    @property
    def unique_dates(self) -> np.ndarray:
        """Sorted unique feature dates — the grid the folds are cut on."""

        return np.unique(self.t0)


@dataclass(frozen=True, slots=True)
class MLDataset:
    """A model-ready panel: features, labels, and the timestamps that police them.

    Rows are ordered by ``(t0, security_id)``. The frame is long — one row per
    ``(date, security)`` — because the cross-section is the unit of prediction:
    the model scores every index member on a date and the strategy ranks them.
    """

    X: pd.DataFrame
    y: pd.Series
    times: SampleTimes
    security_ids: np.ndarray
    target: TargetSpec
    index_name: str
    #: The realised forward return for every row, kept even for the
    #: classification target — the direction label is what the model predicts,
    #: but the *return* is what a portfolio built from it earns, and the traded
    #: comparison in :mod:`quantlab.ml_models.trading` needs both.
    forward_return: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.X)
        if not (len(self.y) == n == len(self.times) == self.security_ids.size):
            raise ValueError(
                "dataset arrays disagree on length: "
                f"X={n}, y={len(self.y)}, times={len(self.times)}, "
                f"securities={self.security_ids.size}"
            )

    def __len__(self) -> int:
        return len(self.X)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.X.columns)

    @property
    def task(self) -> Task:
        return self.target.task

    @property
    def dates(self) -> np.ndarray:
        return self.times.t0

    @property
    def start(self) -> date:
        return pd.Timestamp(self.times.t0.min()).date()

    @property
    def end(self) -> date:
        """The last **feature** date. See :attr:`label_end` for the data touched."""

        return pd.Timestamp(self.times.t0.max()).date()

    @property
    def label_end(self) -> date:
        """The last date this dataset reads a price from — what the lock checks."""

        return pd.Timestamp(self.times.t1.max()).date()

    def split_mask(self, split: Split) -> np.ndarray:
        """Boolean mask of rows that lie **entirely** within ``split``.

        Entirely: both ``t0`` and ``t1`` must fall inside the split's bounds. A
        sample whose features are the last training day and whose label resolves
        on the first validation day belongs to neither and is excluded from both.
        Assigning it to train would leak a validation-period price into training;
        assigning it to validation would score the model on a feature date it was
        trained on. Dropping it costs one cross-section per boundary.
        """

        lo, hi = _split_bounds(split)
        t0 = self.times.t0
        t1 = self.times.t1
        mask = t0 >= np.datetime64(lo)
        if hi is not None:
            mask &= t1 <= np.datetime64(hi)
        return np.asarray(mask, dtype=bool)

    def subset(self, mask: np.ndarray | Sequence[int]) -> MLDataset:
        """A new dataset holding only the selected rows, in the same order."""

        idx = np.asarray(mask)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return MLDataset(
            X=self.X.iloc[idx].reset_index(drop=True),
            y=self.y.iloc[idx].reset_index(drop=True),
            times=SampleTimes(t0=self.times.t0[idx], t1=self.times.t1[idx]),
            security_ids=self.security_ids[idx],
            target=self.target,
            index_name=self.index_name,
            forward_return=self.forward_return[idx],
        )

    def split(self, split: Split) -> MLDataset:
        """The subset lying entirely within ``split`` (see :meth:`split_mask`)."""

        subset = self.subset(self.split_mask(split))
        if len(subset) == 0:
            raise ValueError(
                f"the {split.value} split of this dataset is empty: the panel "
                f"covers {self.start.isoformat()}..{self.end.isoformat()}"
            )
        return subset

    def frame(self) -> pd.DataFrame:
        """A long ``[date, security_id, label_date, y, forward_return]`` frame."""

        return pd.DataFrame(
            {
                "date": self.times.t0,
                "security_id": self.security_ids,
                "label_date": self.times.t1,
                "y": self.y.to_numpy(),
                "forward_return": self.forward_return,
            }
        )

    def describe(self) -> str:
        return (
            f"{len(self):,} samples x {len(self.feature_names)} features | "
            f"{self.target.describe()} | {self.index_name} | "
            f"features {self.start.isoformat()}..{self.end.isoformat()}, "
            f"labels through {self.label_end.isoformat()}"
        )


def _split_bounds(split: Split) -> tuple[date, date | None]:
    if split is Split.TRAIN:
        return DATA_START, TRAIN_END
    if split is Split.VALIDATION:
        return date(TRAIN_END.year + 1, 1, 1), VALIDATION_END
    return HOLDOUT_START, None


def _to_date(value: object) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def restrict_to_universe(
    panel: pd.DataFrame, universe: UniverseByDate, *, strict: bool = True
) -> pd.DataFrame:
    """Keep only rows whose security was in the index on that row's date.

    With ``strict=True`` (the default) a date absent from ``universe`` raises.
    The alternative — treating a missing membership record as "everything is a
    member" — is exactly the survivorship leak this function exists to prevent,
    and it would fail silently.
    """

    if panel.empty:
        return panel.copy()

    dates = pd.to_datetime(panel["date"])
    as_dates = np.array([d.date() for d in dates], dtype=object)

    if strict:
        missing = sorted({d for d in as_dates if d not in universe})
        if missing:
            shown = ", ".join(d.isoformat() for d in missing[:5])
            raise ValueError(
                f"{len(missing)} panel date(s) have no point-in-time universe "
                f"record (first: {shown}). Build index membership for the full "
                "range before assembling a dataset — a model must never see a "
                "security that was not in the index on that date, and a missing "
                "record cannot be distinguished from an empty index."
            )

    pairs = pd.MultiIndex.from_arrays([as_dates, panel["security_id"].to_numpy()])
    members = {
        (day, security) for day, holdings in universe.items() for security in holdings
    }
    mask = np.asarray(pairs.isin(members), dtype=bool)
    dropped = int((~mask).sum())
    if dropped:
        log.info(
            "ml.dataset.universe_restricted",
            dropped_rows=dropped,
            kept_rows=int(mask.sum()),
        )
    return panel.loc[mask].reset_index(drop=True)


def _forward_target(
    panel: pd.DataFrame, *, horizon: int, price_column: str
) -> pd.DataFrame:
    """Attach the forward return and the date it resolves on, per security.

    Both are computed with a per-security ``shift(-horizon)`` so a horizon never
    bleeds across the boundary between one security's last bar and the next
    security's first.
    """

    out = panel.sort_values(["security_id", "date"]).reset_index(drop=True)
    grouped = out.groupby("security_id", sort=False)
    future_price = grouped[price_column].shift(-horizon)
    out["forward_return"] = future_price / out[price_column] - 1.0
    out["label_date"] = grouped["date"].shift(-horizon)
    return out


def build_dataset(
    panel: pd.DataFrame,
    *,
    features: Sequence[str],
    universe: UniverseByDate,
    index_name: str,
    target: TargetSpec | None = None,
    price_column: str = "adj_close",
    allow_holdout: bool = False,
    min_names_per_date: int = 5,
) -> MLDataset:
    """Assemble a model-ready dataset from a long feature panel.

    ``panel`` is long ``[date, security_id, <price_column>, <features…>]`` — the
    output of :func:`quantlab.feature_engineering.build.build_feature_panel`
    joined to prices. The cross-sectional features in it must already have been
    computed *within* ``universe``; this function restricts the surviving rows to
    the same membership, but it cannot retro-fix a z-score that was taken across
    a cross-section including non-members.

    Rows with any missing feature or a missing label are dropped. That is a
    stricter rule than the tree models need — they tolerate NaN — but a model
    comparison in which ridge sees 400,000 rows and LightGBM sees 480,000 is not
    a comparison of models.
    """

    spec = target or TargetSpec()
    required = {"date", "security_id", price_column}
    missing_cols = sorted(required - set(panel.columns))
    if missing_cols:
        raise KeyError(f"panel is missing required column(s): {missing_cols}")
    unknown = [f for f in features if f not in panel.columns]
    if unknown:
        raise KeyError(f"panel is missing requested feature(s): {unknown}")
    if not features:
        raise ValueError("a dataset needs at least one feature column")

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])

    # Order matters: the forward return is a property of the *price series*, so
    # it is computed on the unrestricted panel. Restricting first would compute
    # a return across the gap left by a security's absence from the index.
    work = _forward_target(work, horizon=spec.horizon, price_column=price_column)
    work = restrict_to_universe(work, universe)

    columns = ["date", "security_id", "label_date", "forward_return", *features]
    work = work.loc[:, columns].dropna()
    if work.empty:
        raise ValueError(
            "no complete rows survive: every sample was missing a feature or a "
            f"{spec.horizon}-day forward return. Check the feature warmup against "
            "the requested date range."
        )

    counts = work.groupby("date")["security_id"].transform("size")
    work = work.loc[counts >= min_names_per_date].reset_index(drop=True)
    if work.empty:
        raise ValueError(
            f"no date has at least {min_names_per_date} securities; a "
            "cross-sectional model has nothing to rank"
        )

    work = work.sort_values(["date", "security_id"]).reset_index(drop=True)

    t0 = work["date"].to_numpy(dtype="datetime64[ns]")
    t1 = work["label_date"].to_numpy(dtype="datetime64[ns]")
    forward = work["forward_return"].to_numpy(dtype=float)

    # The lock is checked against the *label* date. A dataset whose features stop
    # short of the boundary still reads prices `horizon` days later, and those
    # prices are the holdout.
    assert_holdout_allowed(
        pd.Timestamp(t1.max()).date(),
        holdout=allow_holdout,
        context=f"ML dataset for {index_name!r} ({spec.describe()})",
    )

    y_values = (
        forward if spec.kind is TargetKind.RETURN else (forward > 0.0).astype(int)
    )
    dataset = MLDataset(
        X=work.loc[:, list(features)].astype(float).reset_index(drop=True),
        y=pd.Series(y_values, name=spec.kind.value),
        times=SampleTimes(t0=t0, t1=t1),
        security_ids=work["security_id"].to_numpy(),
        target=spec,
        index_name=index_name,
        forward_return=forward,
    )
    log.info(
        "ml.dataset.built",
        index=index_name,
        target=spec.kind.value,
        horizon=spec.horizon,
        samples=len(dataset),
        features=len(dataset.feature_names),
        start=dataset.start.isoformat(),
        end=dataset.end.isoformat(),
        label_end=dataset.label_end.isoformat(),
    )
    return dataset


def join_prices(
    features: pd.DataFrame, prices: pd.DataFrame, *, price_column: str = "adj_close"
) -> pd.DataFrame:
    """Inner-join a feature panel to prices on ``(date, security_id)``.

    A convenience for the common case where features come from the parquet
    dataset and prices from ``ohlcv_daily``. Inner because a feature row without
    a price cannot produce a label, and a price without features cannot produce a
    prediction.
    """

    left = features.copy()
    right = prices.loc[:, ["date", "security_id", price_column]].copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    return left.merge(right, on=["date", "security_id"], how="inner")
