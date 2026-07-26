"""Full-universe feature build: DB -> per-security compute -> panel -> parquet.

Orchestration only; the per-security maths lives in :mod:`engine`. The build:

1. loads the OHLCV panel for an index over a date range from ``ohlcv_daily``,
2. attaches the ``market_return`` base column (same-date cross-sectional mean),
3. computes time-series features **per security in parallel** (joblib) — the
   per-ticker parallelism the performance deliverable calls for,
4. assembles the long panel and computes cross-sectional features within the
   point-in-time universe on each date, and
5. writes the result to ``data/features/`` as parquet partitioned by year, so a
   reader can push a date/security predicate down to the file level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
import structlog
from joblib import Parallel, delayed
from sqlalchemy import select
from sqlalchemy.orm import Session

from quantlab.config import get_settings
from quantlab.db.models import IndexMembership, OHLCVDaily
from quantlab.universe.membership import universe_as_of

from .engine import (
    add_market_return,
    compute_cross_sectional_features,
    compute_time_series_features,
)
from .registry import REGISTRY, FeatureKind, FeatureRegistry

log = structlog.get_logger(__name__)

_RAW_COLUMNS = ("date", "open", "high", "low", "close", "adj_close", "volume")


def resolve_feature_names(
    selector: str | Sequence[str] | None,
    *,
    registry: FeatureRegistry = REGISTRY,
) -> list[str]:
    """Turn a CLI ``--features`` selector into concrete feature names.

    Accepts ``"all"``/``None`` (everything), a single category name, or a
    comma-separated / sequence of explicit feature names.
    """

    if selector is None or selector == "all":
        return registry.names()
    items = selector.split(",") if isinstance(selector, str) else list(selector)
    items = [i.strip() for i in items if i.strip()]
    categories = {s.category for s in registry.all()}
    if len(items) == 1 and items[0] in categories:
        return [s.name for s in registry.by_category(items[0])]
    unknown = [i for i in items if i not in registry]
    if unknown:
        raise ValueError(f"unknown feature(s): {', '.join(unknown)}")
    return items


def universe_by_date(
    session: Session, index_name: str, dates: Sequence[date]
) -> dict[date, set[str]]:
    """Map each date to its point-in-time constituent set (for cross-section)."""

    return {d: universe_as_of(session, index_name, d) for d in dates}


def load_ohlcv_panel(
    session: Session,
    index_name: str,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Load a long OHLCV panel for every security ever in ``index_name``.

    Membership is used to scope *which* securities to load (any that were ever in
    the index over history); the point-in-time restriction for cross-sectional
    features is applied later per date via :func:`universe_by_date`.
    """

    member_ids = set(
        session.execute(
            select(IndexMembership.security_id)
            .where(IndexMembership.index_name == index_name)
            .distinct()
        ).scalars()
    )
    if not member_ids:
        return pd.DataFrame(columns=["security_id", *_RAW_COLUMNS])

    stmt = select(
        OHLCVDaily.security_id,
        OHLCVDaily.date,
        OHLCVDaily.open,
        OHLCVDaily.high,
        OHLCVDaily.low,
        OHLCVDaily.close,
        OHLCVDaily.adj_close,
        OHLCVDaily.volume,
    ).where(OHLCVDaily.security_id.in_(member_ids))
    if start is not None:
        stmt = stmt.where(OHLCVDaily.date >= start)
    if end is not None:
        stmt = stmt.where(OHLCVDaily.date <= end)

    rows = session.execute(stmt).all()
    panel = pd.DataFrame(rows, columns=["security_id", *_RAW_COLUMNS])
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["security_id", "date"]).reset_index(drop=True)


def _compute_one_security(
    security_id: str,
    frame: pd.DataFrame,
    ts_names: list[str],
    registry: FeatureRegistry,
) -> pd.DataFrame:
    result = compute_time_series_features(
        frame.sort_values("date").reset_index(drop=True),
        ts_names,
        registry=registry,
    )
    result["security_id"] = security_id
    return result


def build_feature_panel(
    panel: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    universe: Mapping[date, set[str]] | None = None,
    registry: FeatureRegistry = REGISTRY,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute ``feature_names`` for a raw OHLCV panel and return the long result.

    ``panel`` is long ``(security_id, date, OHLCV…)``. Time-series features are
    computed per security (parallel when ``n_jobs != 1``); cross-sectional
    features are then computed on the assembled panel, restricted to ``universe``
    members per date when supplied.
    """

    if panel.empty:
        return panel.copy()

    closure = registry.dependency_closure(feature_names)
    ts_names = [n for n in closure if registry.get(n).kind is FeatureKind.TIME_SERIES]
    xs_requested = [
        n
        for n in feature_names
        if n in registry and registry.get(n).kind is FeatureKind.CROSS_SECTIONAL
    ]

    with_market = add_market_return(panel, universe_by_date=universe)
    groups = list(with_market.groupby("security_id", sort=True))

    computed: list[pd.DataFrame] = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_security)(sid, frame, ts_names, registry)
        for sid, frame in groups
    )
    result = pd.concat(computed, ignore_index=True)

    if xs_requested:
        result = compute_cross_sectional_features(
            result, xs_requested, registry=registry, universe_by_date=universe
        )

    # Return the identifying columns plus exactly the requested features.
    keep = ["security_id", "date", *[n for n in feature_names if n in result.columns]]
    return (
        result.loc[:, keep].sort_values(["date", "security_id"]).reset_index(drop=True)
    )


def write_feature_panel(
    panel: pd.DataFrame,
    *,
    features_dir: Path | None = None,
    partition_by_year: bool = True,
) -> Path:
    """Write the feature panel to partitioned parquet with predicate pushdown.

    Partitioned by ``year`` (derived from ``date``) so a reader filtering on a
    date range only touches the relevant files. Returns the dataset root.
    """

    import pyarrow as pa
    import pyarrow.dataset as ds

    settings = get_settings()
    root = features_dir or settings.features_dir
    root.mkdir(parents=True, exist_ok=True)

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year

    table = pa.Table.from_pandas(out, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(root),
        format="parquet",
        partitioning=["year"] if partition_by_year else None,
        partitioning_flavor="hive",
        existing_data_behavior="overwrite_or_ignore",
    )
    log.info("features.written", root=str(root), rows=len(out))
    return root


def read_feature_panel(
    *,
    features_dir: Path | None = None,
    start: date | None = None,
    end: date | None = None,
    securities: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read the feature dataset back, pushing date/security filters to parquet.

    Raises :class:`FileNotFoundError` when the dataset has not been built. That
    includes the case where the directory *exists but is empty*, which is the
    normal state of a fresh checkout: ``features_dir`` is created eagerly, so an
    "is there a directory" check is not the same question as "is there data".
    Without this, pyarrow raises ``ArrowInvalid: No match for FieldRef.Name(date)``
    from inside the filter — which callers cannot reasonably be expected to
    recognise, and which reached the API as a 500 where a 404 telling the
    researcher to run ``quantlab features build`` was intended.
    """

    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    settings = get_settings()
    root = features_dir or settings.features_dir
    if not root.exists() or not any(root.rglob("*.parquet")):
        raise FileNotFoundError(
            f"no feature parquet files under {root}. "
            "Build the panel first: quantlab features build --universe <name>"
        )
    dataset = ds.dataset(str(root), format="parquet", partitioning="hive")

    predicate = None
    if start is not None:
        predicate = pc.field("date") >= pd.Timestamp(start)
    if end is not None:
        end_pred = pc.field("date") <= pd.Timestamp(end)
        predicate = end_pred if predicate is None else predicate & end_pred
    if securities is not None:
        sec_pred = pc.field("security_id").isin(list(securities))
        predicate = sec_pred if predicate is None else predicate & sec_pred

    table = dataset.to_table(
        filter=predicate, columns=list(columns) if columns else None
    )
    return cast("pd.DataFrame", table.to_pandas())


def build_and_write(
    session: Session,
    index_name: str,
    *,
    features: str | Sequence[str] | None = "all",
    start: date | None = None,
    end: date | None = None,
    features_dir: Path | None = None,
    registry: FeatureRegistry = REGISTRY,
    n_jobs: int = 1,
) -> tuple[int, Path]:
    """End-to-end build for the CLI: load, compute, write. Returns (rows, root)."""

    registry.validate()
    names = resolve_feature_names(features, registry=registry)
    panel = load_ohlcv_panel(session, index_name, start=start, end=end)
    if panel.empty:
        raise ValueError(
            f"no OHLCV rows for any member of {index_name!r} in the given range; "
            "ingest data and build the universe first"
        )
    dates = sorted({d.date() for d in pd.to_datetime(panel["date"])})
    uni = universe_by_date(session, index_name, dates)
    result = build_feature_panel(
        panel, names, universe=uni, registry=registry, n_jobs=n_jobs
    )
    root = write_feature_panel(result, features_dir=features_dir)
    return len(result), root
