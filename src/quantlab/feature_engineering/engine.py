"""The DAG compute engine: resolve order once, compute each node once.

Two entry points mirror the two feature kinds:

* :func:`compute_time_series_features` runs a single security's date-sorted frame
  through the topological order, writing each feature as a column so a downstream
  feature reads its dependency straight off the frame — the shared-intermediate
  caching the performance deliverable relies on (an EMA needed by five features
  is computed once).
* :func:`compute_cross_sectional_features` runs the assembled panel through the
  cross-sectional nodes, computing each **per date within the point-in-time
  universe** so a security absent from the index on a date gets ``NaN``, never a
  rank against names it could not have been compared to.

Neither ever reads a future row: time-series features use only trailing windows
and the cross-sectional pass groups strictly by date. That is what the
registry-wide look-ahead test in ``tests/feature_engineering`` proves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date

import pandas as pd

from .registry import REGISTRY, FeatureKind, FeatureRegistry


def compute_time_series_features(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    registry: FeatureRegistry = REGISTRY,
) -> pd.DataFrame:
    """Compute the time-series features (and their deps) for one security.

    ``df`` must be one security's OHLCV history sorted ascending by date and hold
    every base column each requested feature transitively needs (raw OHLCV and,
    for beta/alpha, ``market_return``). Returns a copy with one column added per
    computed time-series feature. Cross-sectional features in the closure are
    skipped here — they are computed later on the full panel.
    """

    order = registry.topological_order(feature_names)
    work = df.copy()
    for name in order:
        spec = registry.get(name)
        if spec.kind is not FeatureKind.TIME_SERIES:
            continue
        result = spec.func(work)
        work[name] = pd.Series(result, index=work.index).astype(float)
    return work


def _apply_cross_sectional(
    panel: pd.DataFrame,
    name: str,
    *,
    registry: FeatureRegistry,
    universe_by_date: Mapping[date, set[str]] | None,
) -> pd.Series[float]:
    """Compute one cross-sectional feature per date, within the universe."""

    spec = registry.get(name)
    out = pd.Series(index=panel.index, dtype=float)
    for date_val, rows in panel.groupby("date", sort=False):
        if universe_by_date is not None:
            members = universe_by_date.get(_as_date(date_val))
            rows = rows if members is None else rows[rows["security_id"].isin(members)]
        if rows.empty:
            continue
        values = spec.func(rows)
        out.loc[values.index] = values.astype(float)
    return out


def compute_cross_sectional_features(
    panel: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    registry: FeatureRegistry = REGISTRY,
    universe_by_date: Mapping[date, set[str]] | None = None,
) -> pd.DataFrame:
    """Add cross-sectional feature columns to a long ``(date, security_id, …)`` panel.

    ``universe_by_date`` maps a date to the set of ``security_id`` in the index on
    that date; when supplied, the cross-section on each date is restricted to
    those members (PRD Module 3). Time-series dependency columns must already be
    present on the panel.
    """

    order = registry.topological_order(feature_names)
    result = panel.copy()
    for name in order:
        if registry.get(name).kind is not FeatureKind.CROSS_SECTIONAL:
            continue
        result[name] = _apply_cross_sectional(
            result, name, registry=registry, universe_by_date=universe_by_date
        )
    return result


def add_market_return(
    panel: pd.DataFrame,
    *,
    universe_by_date: Mapping[date, set[str]] | None = None,
) -> pd.DataFrame:
    """Attach a ``market_return`` base column: the cross-sectional mean return.

    Computed as the equal-weighted mean of each security's daily simple return on
    each date, restricted to universe members when ``universe_by_date`` is given.
    It is a *same-date* aggregate, so it introduces no look-ahead; beta/alpha
    features then read it as a base column.
    """

    out = panel.sort_values(["security_id", "date"]).copy()
    out["_ret"] = out.groupby("security_id", sort=False)["close"].pct_change()

    if universe_by_date is None:
        market = out.groupby("date", sort=False)["_ret"].transform("mean")
    else:
        market = pd.Series(index=out.index, dtype=float)
        for date_val, rows in out.groupby("date", sort=False):
            members = universe_by_date.get(_as_date(date_val))
            member_rows = (
                rows if members is None else rows[rows["security_id"].isin(members)]
            )
            market.loc[rows.index] = member_rows["_ret"].mean()

    out["market_return"] = market.fillna(0.0)
    return out.drop(columns="_ret")


def _as_date(value: object) -> date:
    """Coerce a groupby key (Timestamp or date) to a plain ``date`` for lookup."""

    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def required_base_columns(
    feature_names: Iterable[str], *, registry: FeatureRegistry = REGISTRY
) -> set[str]:
    """The base columns a set of features transitively needs to be computable."""

    from .registry import BASE_COLUMNS

    closure = registry.dependency_closure(feature_names)
    needed: set[str] = set()
    for name in closure:
        for dep in registry.get(name).dependencies:
            if dep in BASE_COLUMNS:
                needed.add(dep)
    return needed
