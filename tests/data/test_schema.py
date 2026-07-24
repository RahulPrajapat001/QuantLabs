"""The canonical OHLCV schema rejects malformed provider output."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.data.base import validate_ohlcv
from quantlab.data.errors import PermanentProviderError

from .conftest import make_ohlcv


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    return validate_ohlcv(df, source="test", symbol="X")


def test_valid_frame_passes_and_is_sorted() -> None:
    df = make_ohlcv(["2020-01-03", "2020-01-01", "2020-01-02"])
    out = _validate(df)
    assert list(out["date"]) == sorted(out["date"])
    assert list(out.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]


def test_high_below_low_is_rejected() -> None:
    df = make_ohlcv(["2020-01-01"])
    df.loc[0, "high"] = df.loc[0, "low"] - 1
    with pytest.raises(PermanentProviderError):
        _validate(df)


def test_close_outside_high_low_is_rejected() -> None:
    df = make_ohlcv(["2020-01-01"])
    df.loc[0, "close"] = df.loc[0, "high"] + 5
    with pytest.raises(PermanentProviderError):
        _validate(df)


def test_negative_price_is_rejected() -> None:
    df = make_ohlcv(["2020-01-01"])
    df.loc[0, "open"] = -1.0
    with pytest.raises(PermanentProviderError):
        _validate(df)


def test_negative_volume_is_rejected() -> None:
    df = make_ohlcv(["2020-01-01"])
    df.loc[0, "volume"] = -1.0
    with pytest.raises(PermanentProviderError):
        _validate(df)


def test_zero_volume_is_allowed() -> None:
    df = make_ohlcv(["2020-01-01"])
    df.loc[0, "volume"] = 0.0
    out = _validate(df)
    assert out.loc[0, "volume"] == 0.0


def test_duplicate_dates_are_rejected() -> None:
    df = make_ohlcv(["2020-01-01", "2020-01-01"])
    with pytest.raises(PermanentProviderError):
        _validate(df)


def test_extra_column_is_rejected() -> None:
    df = make_ohlcv(["2020-01-01"])
    df["surprise"] = 1
    with pytest.raises(PermanentProviderError):
        _validate(df)
