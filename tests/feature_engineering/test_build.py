"""Build orchestration: universe restriction, parquet pushdown, selection, parallel."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.feature_engineering.build import (
    build_feature_panel,
    read_feature_panel,
    resolve_feature_names,
    write_feature_panel,
)

from .conftest import make_panel


def _raw_panel(n_secs: int = 4, n_days: int = 120, seed: int = 5) -> pd.DataFrame:
    from .conftest import make_security

    frames = []
    for i in range(n_secs):
        sec = make_security(n_days, seed=seed + i).drop(columns="market_return")
        sec["security_id"] = f"SEC-{i:04d}"
        frames.append(sec)
    return pd.concat(frames, ignore_index=True)


def test_resolve_feature_names_all() -> None:
    from quantlab.feature_engineering import REGISTRY

    assert resolve_feature_names("all") == REGISTRY.names()


def test_resolve_feature_names_category() -> None:
    names = resolve_feature_names("momentum")
    assert set(names) == {
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "momentum_50d",
        "momentum_100d",
        "momentum_200d",
        "momentum_12_1",
    }


def test_resolve_feature_names_explicit_list() -> None:
    assert resolve_feature_names("rsi_14,atr_14") == ["rsi_14", "atr_14"]


def test_resolve_feature_names_unknown_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown feature"):
        resolve_feature_names("not_a_feature")


def test_build_panel_produces_requested_features() -> None:
    panel = _raw_panel()
    result = build_feature_panel(panel, ["rsi_14", "momentum_20d"], n_jobs=1)
    assert set(result.columns) == {"security_id", "date", "rsi_14", "momentum_20d"}
    assert len(result) == len(panel)


def test_cross_sectional_restricted_to_universe() -> None:
    """A security outside the universe on a date gets NaN for cross-sectional."""

    panel = _raw_panel(n_secs=4, n_days=60)
    all_dates = sorted({d.date() for d in pd.to_datetime(panel["date"])})
    # Exclude SEC-0003 from the universe on every date.
    universe = {d: {"SEC-0000", "SEC-0001", "SEC-0002"} for d in all_dates}

    result = build_feature_panel(
        panel, ["xs_rank_momentum_20d"], universe=universe, n_jobs=1
    )
    excluded = result[result["security_id"] == "SEC-0003"]
    assert excluded["xs_rank_momentum_20d"].isna().all()
    included = result[result["security_id"] == "SEC-0000"]
    # The in-universe names do get ranked (once past the momentum lookback).
    assert included["xs_rank_momentum_20d"].notna().any()


def test_parallel_matches_serial() -> None:
    panel = _raw_panel(n_secs=4, n_days=120)
    names = ["rsi_14", "macd_hist", "beta_60", "volatility_20d"]
    serial = build_feature_panel(panel, names, n_jobs=1)
    parallel = build_feature_panel(panel, names, n_jobs=2)
    pd.testing.assert_frame_equal(
        serial.reset_index(drop=True), parallel.reset_index(drop=True)
    )


def test_parquet_roundtrip_with_predicate_pushdown(tmp_path: Path) -> None:
    panel = _raw_panel(n_secs=3, n_days=400)
    result = build_feature_panel(panel, ["rsi_14", "atr_14"], n_jobs=1)
    write_feature_panel(result, features_dir=tmp_path)

    # Partitioned by year.
    partitions = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert partitions == ["year=2015", "year=2016"]

    # A date + security predicate is pushed down to the read.
    back = read_feature_panel(
        features_dir=tmp_path,
        start=date(2016, 1, 1),
        securities=["SEC-0001"],
        columns=["security_id", "date", "rsi_14"],
    )
    assert set(back["security_id"].unique()) == {"SEC-0001"}
    assert (pd.to_datetime(back["date"]) >= pd.Timestamp("2016-01-01")).all()
    assert set(back.columns) == {"security_id", "date", "rsi_14"}


def test_reading_an_unbuilt_panel_raises_file_not_found(tmp_path: Path) -> None:
    """An empty features directory is "not built", not a corrupt dataset.

    ``features_dir`` is created eagerly, so "the directory exists" and "there is
    data" are different questions. Answering the first was letting pyarrow raise
    ``ArrowInvalid: No match for FieldRef.Name(date)`` from inside the predicate
    — which callers cannot recognise, and which surfaced at the API as a 500
    where the actionable 404 telling the researcher to build the panel was
    intended.
    """

    empty = tmp_path / "features"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="features build"):
        read_feature_panel(features_dir=empty)

    # And the same for a directory that does not exist at all.
    with pytest.raises(FileNotFoundError):
        read_feature_panel(features_dir=tmp_path / "absent")


def test_shared_intermediate_computed_once() -> None:
    """macd_line/signal/hist share ema_12/ema_26; the build still yields them all."""

    panel = _raw_panel(n_secs=2, n_days=80)
    result = build_feature_panel(
        panel, ["macd_line", "macd_signal", "macd_hist"], n_jobs=1
    )
    # hist == line - signal exactly, confirming the shared EMAs flowed through.
    valid = result["macd_hist"].notna()
    np.testing.assert_allclose(
        result.loc[valid, "macd_hist"].to_numpy(),
        (result.loc[valid, "macd_line"] - result.loc[valid, "macd_signal"]).to_numpy(),
        rtol=1e-12,
    )


def test_make_panel_helper_has_ts_columns() -> None:
    panel = make_panel(3, 40)
    assert "momentum_20d" in panel.columns
    assert "rsi_14" in panel.columns
