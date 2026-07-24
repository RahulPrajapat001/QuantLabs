"""Provider tests. The HTTP / vendor layer is mocked — no real API is hit."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from quantlab.data.base import Frequency, Security
from quantlab.data.errors import (
    PermanentProviderError,
    TransientProviderError,
)
from quantlab.data.providers.alpha_vantage import AlphaVantageProvider
from quantlab.data.providers.yfinance import YFinanceProvider
from quantlab.data.rate_limit import RateLimiter, RetryPolicy

SEC = Security(security_id="SEC-1", symbol="ABC.NS")


def _no_wait_limiter() -> RateLimiter:
    return RateLimiter(1000.0, sleep=lambda _s: None, clock=lambda: 0.0)


def _no_wait_policy() -> RetryPolicy:
    return RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0.0)


# --- yfinance ----------------------------------------------------------------


class FakeYFTicker:
    """Mimics the slice of yfinance.Ticker the provider uses."""

    def __init__(
        self,
        history_df: pd.DataFrame,
        actions_df: pd.DataFrame | None = None,
        *,
        raise_exc: Exception | None = None,
    ) -> None:
        self._history = history_df
        self._actions = actions_df
        self._raise = raise_exc

    def history(self, **_kwargs: Any) -> pd.DataFrame:
        if self._raise is not None:
            raise self._raise
        return self._history

    @property
    def actions(self) -> pd.DataFrame | None:
        return self._actions

    def get_history_metadata(self) -> dict[str, Any]:
        return {"symbol": "ABC.NS"}


def _yf_history_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")], name="Date"
    )
    return pd.DataFrame(
        {
            "Open": [99.5, 100.5],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.0, 101.0],
            "Adj Close": [100.0, 101.0],
            "Volume": [1000, 1100],
            "Dividends": [0.0, 1.5],
            "Stock Splits": [0.0, 2.0],
        },
        index=idx,
    )


def _make_yf(
    ticker_obj: FakeYFTicker,
) -> YFinanceProvider:
    return YFinanceProvider(
        ticker_factory=lambda _symbol: ticker_obj,
        rate_limiter=_no_wait_limiter(),
        retry_policy=_no_wait_policy(),
    )


def test_yfinance_fetch_ohlcv_conforms_to_schema() -> None:
    provider = _make_yf(FakeYFTicker(_yf_history_frame()))
    df = provider.fetch_ohlcv(SEC, date(2020, 1, 1), date(2020, 1, 2))
    assert list(df.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert len(df) == 2
    assert df.loc[0, "close"] == 100.0


def test_yfinance_corporate_actions() -> None:
    actions = pd.DataFrame(
        {"Dividends": [0.0, 1.5], "Stock Splits": [0.0, 2.0]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")], name="Date"
        ),
    )
    provider = _make_yf(FakeYFTicker(_yf_history_frame(), actions))
    ca = provider.fetch_corporate_actions(SEC)
    kinds = set(ca["action_type"])
    assert kinds == {"dividend", "split"}


def test_yfinance_empty_history_returns_empty_frame() -> None:
    provider = _make_yf(FakeYFTicker(pd.DataFrame()))
    df = provider.fetch_ohlcv(SEC, date(2020, 1, 1), date(2020, 1, 2))
    assert df.empty


def test_yfinance_timeout_is_transient() -> None:
    provider = _make_yf(
        FakeYFTicker(_yf_history_frame(), raise_exc=Exception("Read timed out"))
    )
    with pytest.raises(TransientProviderError):
        provider.fetch_ohlcv(SEC, date(2020, 1, 1), date(2020, 1, 2))


def test_yfinance_unknown_symbol_is_permanent() -> None:
    provider = _make_yf(
        FakeYFTicker(
            _yf_history_frame(), raise_exc=Exception("No data found, symbol delisted")
        )
    )
    with pytest.raises(PermanentProviderError):
        provider.fetch_ohlcv(SEC, date(2020, 1, 1), date(2020, 1, 2))


# --- Alpha Vantage -----------------------------------------------------------


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    """Returns queued responses in order; records calls."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: int) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self._responses.pop(0)


def _av_payload() -> dict[str, Any]:
    return {
        "Time Series (Daily)": {
            "2020-01-02": {
                "1. open": "100.5",
                "2. high": "102.0",
                "3. low": "100.0",
                "4. close": "101.0",
                "5. adjusted close": "101.0",
                "6. volume": "1100",
                "7. dividend amount": "1.5",
                "8. split coefficient": "2.0",
            },
            "2020-01-01": {
                "1. open": "99.5",
                "2. high": "101.0",
                "3. low": "99.0",
                "4. close": "100.0",
                "5. adjusted close": "100.0",
                "6. volume": "1000",
                "7. dividend amount": "0.0",
                "8. split coefficient": "1.0",
            },
        }
    }


def _make_av(session: FakeSession) -> AlphaVantageProvider:
    return AlphaVantageProvider(
        api_key="TEST",
        session=session,
        rate_limiter=_no_wait_limiter(),
        retry_policy=_no_wait_policy(),
    )


def test_alpha_vantage_fetch_ohlcv() -> None:
    provider = _make_av(FakeSession([FakeResponse(_av_payload())]))
    df = provider.fetch_ohlcv(SEC, date(2020, 1, 1), date(2020, 1, 2))
    assert len(df) == 2
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-02"]
    assert df.loc[1, "adj_close"] == 101.0


def test_alpha_vantage_corporate_actions() -> None:
    provider = _make_av(FakeSession([FakeResponse(_av_payload())]))
    ca = provider.fetch_corporate_actions(SEC)
    assert set(ca["action_type"]) == {"dividend", "split"}


def test_alpha_vantage_missing_key_raises() -> None:
    with pytest.raises(PermanentProviderError):
        AlphaVantageProvider(api_key="")


def test_alpha_vantage_rate_limit_note_is_transient_then_retried() -> None:
    # First response is a throttle note; second is real data.
    session = FakeSession(
        [
            FakeResponse({"Note": "Thank you for using Alpha Vantage!"}),
            FakeResponse(_av_payload()),
        ]
    )
    provider = _make_av(session)
    df = provider.fetch_ohlcv(SEC, date(2020, 1, 1), date(2020, 1, 2))
    assert len(df) == 2
    assert len(session.calls) == 2  # retried once


def test_alpha_vantage_error_message_is_permanent() -> None:
    session = FakeSession([FakeResponse({"Error Message": "Invalid API call"})])
    provider = _make_av(session)
    with pytest.raises(PermanentProviderError):
        provider.fetch_ohlcv(SEC, date(2020, 1, 1), date(2020, 1, 2))


def test_alpha_vantage_5xx_is_transient() -> None:
    session = FakeSession(
        [
            FakeResponse({}, status_code=503),
            FakeResponse(_av_payload()),
        ]
    )
    provider = _make_av(session)
    df = provider.fetch_ohlcv(SEC, date(2020, 1, 1), date(2020, 1, 2))
    assert len(df) == 2


def test_alpha_vantage_non_daily_rejected() -> None:
    provider = _make_av(FakeSession([]))
    with pytest.raises(PermanentProviderError):
        provider.fetch_ohlcv(SEC, date(2020, 1, 1), date(2020, 1, 2), Frequency.WEEKLY)
