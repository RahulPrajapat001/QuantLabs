"""Raw parquet landing is partitioned by year and idempotent."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from quantlab.data.storage import raw_path, read_raw_ohlcv, write_raw_ohlcv

from .conftest import make_ohlcv


def test_partitions_by_year(tmp_path: Path) -> None:
    df = make_ohlcv(["2019-12-31", "2020-01-02", "2021-06-01"])
    written = write_raw_ohlcv(df, raw_dir=tmp_path, source="fake", ticker="ABC.NS")
    years = {p.stem for p in written}
    assert years == {"2019", "2020", "2021"}
    assert raw_path(tmp_path, "fake", "ABC.NS", 2020).exists()


def test_reingest_same_range_is_idempotent(tmp_path: Path) -> None:
    df = make_ohlcv(["2020-01-01", "2020-01-02", "2020-01-03"])
    write_raw_ohlcv(df, raw_dir=tmp_path, source="fake", ticker="ABC.NS")
    write_raw_ohlcv(df, raw_dir=tmp_path, source="fake", ticker="ABC.NS")

    back = read_raw_ohlcv(raw_dir=tmp_path, source="fake", ticker="ABC.NS")
    assert len(back) == 3  # no duplication


def test_overlapping_ranges_merge_without_duplicates(tmp_path: Path) -> None:
    first = make_ohlcv(["2020-01-01", "2020-01-02"])
    second = make_ohlcv(["2020-01-02", "2020-01-03"])
    write_raw_ohlcv(first, raw_dir=tmp_path, source="fake", ticker="ABC.NS")
    write_raw_ohlcv(second, raw_dir=tmp_path, source="fake", ticker="ABC.NS")

    back = read_raw_ohlcv(raw_dir=tmp_path, source="fake", ticker="ABC.NS")
    assert list(back["date"].dt.strftime("%Y-%m-%d")) == [
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
    ]


def test_new_rows_win_on_conflict(tmp_path: Path) -> None:
    original = make_ohlcv(["2020-01-01"], base=100.0)
    corrected = make_ohlcv(["2020-01-01"], base=200.0)  # same date, new adj_close
    write_raw_ohlcv(original, raw_dir=tmp_path, source="fake", ticker="ABC.NS")
    write_raw_ohlcv(corrected, raw_dir=tmp_path, source="fake", ticker="ABC.NS")

    back = read_raw_ohlcv(raw_dir=tmp_path, source="fake", ticker="ABC.NS")
    assert len(back) == 1
    assert back.loc[0, "adj_close"] == 200.0


def test_read_range_filter(tmp_path: Path) -> None:
    df = make_ohlcv(["2020-01-01", "2020-06-01", "2020-12-01"])
    write_raw_ohlcv(df, raw_dir=tmp_path, source="fake", ticker="ABC.NS")

    back = read_raw_ohlcv(
        raw_dir=tmp_path,
        source="fake",
        ticker="ABC.NS",
        start=date(2020, 5, 1),
        end=date(2020, 7, 1),
    )
    assert len(back) == 1
    assert back.loc[0, "date"].strftime("%Y-%m-%d") == "2020-06-01"


def test_read_missing_ticker_returns_empty(tmp_path: Path) -> None:
    back = read_raw_ohlcv(raw_dir=tmp_path, source="fake", ticker="NOPE.NS")
    assert back.empty
