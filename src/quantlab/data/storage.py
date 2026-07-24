"""Immutable raw landing: partitioned, idempotent parquet.

Raw vendor responses land at ``{raw_dir}/{source}/{ticker}/{year}.parquet`` and
are treated as immutable — a research result is never reproduced by
re-downloading (PRD §3, Storage). "Immutable" here means *stable*: re-ingesting
an overlapping range must neither duplicate rows nor corrupt the file. Writes
are therefore a per-year read-merge-dedupe-replace, done atomically via a temp
file + ``os.replace`` so a crash mid-write never leaves a torn parquet.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from .base import OHLCV_COLUMNS, empty_ohlcv


def _sanitize(component: str) -> str:
    """Make a path component filesystem-safe (tickers contain ``.``, ``/`` etc.)."""

    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in component)


def ticker_dir(raw_dir: Path, source: str, ticker: str) -> Path:
    """Directory holding a ticker's per-year parquet partitions."""

    return raw_dir / _sanitize(source) / _sanitize(ticker)


def raw_path(raw_dir: Path, source: str, ticker: str, year: int) -> Path:
    """Path to one ``{year}.parquet`` partition."""

    return ticker_dir(raw_dir, source, ticker) / f"{year}.parquet"


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, path)  # atomic on POSIX and Windows for same-volume paths


def write_raw_ohlcv(
    df: pd.DataFrame,
    *,
    raw_dir: Path,
    source: str,
    ticker: str,
) -> list[Path]:
    """Persist ``df`` to per-year partitions idempotently.

    For each calendar year present, existing rows are read back, concatenated
    with the new rows, de-duplicated on ``date`` (new rows win, so a corrected
    adjusted close overwrites a stale one), sorted, and atomically rewritten.
    Returns the partition paths touched.
    """

    if df.empty:
        return []

    frame = df.loc[:, list(OHLCV_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    written: list[Path] = []

    for year, chunk in frame.groupby(frame["date"].dt.year):
        path = raw_path(raw_dir, source, ticker, int(year))
        if path.exists():
            existing = pd.read_parquet(path)
            existing["date"] = pd.to_datetime(existing["date"])
            merged = pd.concat([existing, chunk], ignore_index=True)
        else:
            merged = chunk
        merged = (
            merged.drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        _atomic_write_parquet(merged, path)
        written.append(path)

    return written


def read_raw_ohlcv(
    *,
    raw_dir: Path,
    source: str,
    ticker: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Read a ticker's raw OHLCV back, optionally filtered to ``[start, end]``.

    Returns an empty, correctly-typed frame when nothing has been ingested.
    """

    directory = ticker_dir(raw_dir, source, ticker)
    if not directory.exists():
        return empty_ohlcv()

    parts = sorted(directory.glob("*.parquet"))
    if not parts:
        return empty_ohlcv()

    frames = [pd.read_parquet(p) for p in parts]
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    if start is not None:
        combined = combined[combined["date"] >= pd.Timestamp(start)]
    if end is not None:
        combined = combined[combined["date"] <= pd.Timestamp(end)]

    result: pd.DataFrame = (
        combined.loc[:, list(OHLCV_COLUMNS)]
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return result
