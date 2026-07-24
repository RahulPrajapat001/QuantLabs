"""Alpha Vantage-backed provider (backup source).

The HTTP seam is a :class:`requests.Session`; tests mock ``session.get`` and
never reach the network. The free tier is ~5 requests/minute, so the default
rate limiter is conservative. Alpha Vantage signals problems in the JSON body
rather than via status codes:

* ``"Note"`` / ``"Information"`` → throttled → :class:`TransientProviderError`.
* ``"Error Message"`` → bad symbol / bad call → :class:`PermanentProviderError`.

The API key comes from :class:`quantlab.config.Settings`, never hardcoded.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import structlog

from quantlab.config import get_settings
from quantlab.data.base import (
    CorporateActionType,
    DataProvider,
    DelistingInfo,
    Frequency,
    Security,
    empty_corporate_actions,
    empty_ohlcv,
)
from quantlab.data.errors import PermanentProviderError, TransientProviderError
from quantlab.data.rate_limit import RateLimiter, RetryPolicy, call_with_retry

log = structlog.get_logger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"
_TIMEOUT = 30


class AlphaVantageProvider(DataProvider):
    """Fetch adjusted daily OHLCV and corporate actions from Alpha Vantage."""

    source = "alphavantage"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        session: Any | None = None,
        rate_limiter: RateLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
        calls_per_minute: float = 5.0,
    ) -> None:
        key = api_key if api_key is not None else get_settings().alpha_vantage_api_key
        if not key:
            raise PermanentProviderError(
                "Alpha Vantage API key is not configured "
                "(set QUANTLAB_ALPHA_VANTAGE_API_KEY)",
                source=self.source,
            )
        self._api_key = key
        self._session = session if session is not None else self._default_session()
        self._rate_limiter = rate_limiter or RateLimiter(calls_per_minute)
        self._retry_policy = retry_policy or RetryPolicy()

    @staticmethod
    def _default_session() -> Any:
        import requests

        return requests.Session()

    # -- HTTP ----------------------------------------------------------------

    def _get(self, params: dict[str, str], *, symbol: str) -> dict[str, Any]:
        def _call() -> dict[str, Any]:
            self._rate_limiter.acquire()
            query = {**params, "apikey": self._api_key}
            try:
                response = self._session.get(_BASE_URL, params=query, timeout=_TIMEOUT)
            except Exception as exc:  # connection errors, timeouts
                raise TransientProviderError(
                    str(exc), source=self.source, symbol=symbol
                ) from exc

            status = getattr(response, "status_code", 200)
            if status == 429 or 500 <= status < 600:
                raise TransientProviderError(
                    f"HTTP {status}", source=self.source, symbol=symbol
                )
            if status >= 400:
                raise PermanentProviderError(
                    f"HTTP {status}", source=self.source, symbol=symbol
                )

            try:
                payload: dict[str, Any] = response.json()
            except Exception as exc:
                raise PermanentProviderError(
                    f"invalid JSON: {exc}", source=self.source, symbol=symbol
                ) from exc

            # Alpha Vantage throttling and errors are in-band.
            if "Note" in payload or "Information" in payload:
                raise TransientProviderError(
                    str(payload.get("Note") or payload.get("Information")),
                    source=self.source,
                    symbol=symbol,
                )
            if "Error Message" in payload:
                raise PermanentProviderError(
                    str(payload["Error Message"]),
                    source=self.source,
                    symbol=symbol,
                )
            return payload

        return call_with_retry(
            _call,
            policy=self._retry_policy,
            description=f"alphavantage {symbol}",
        )

    # -- DataProvider hooks --------------------------------------------------

    def _fetch_ohlcv(
        self,
        security: Security,
        start: date,
        end: date,
        frequency: Frequency,
    ) -> pd.DataFrame:
        if frequency is not Frequency.DAILY:
            raise PermanentProviderError(
                f"Alpha Vantage provider supports daily bars only, got {frequency}",
                source=self.source,
                symbol=security.symbol,
            )

        payload = self._get(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": security.symbol,
                "outputsize": "full",
            },
            symbol=security.symbol,
        )
        series = payload.get("Time Series (Daily)")
        if not series:
            return empty_ohlcv()

        records: list[dict[str, object]] = []
        for day, values in series.items():
            when = pd.Timestamp(day)
            if when < pd.Timestamp(start) or when > pd.Timestamp(end):
                continue
            records.append(
                {
                    "date": when,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "adj_close": float(values["5. adjusted close"]),
                    "volume": float(values["6. volume"]),
                }
            )
        if not records:
            return empty_ohlcv()
        return pd.DataFrame.from_records(records)

    def _fetch_corporate_actions(self, security: Security) -> pd.DataFrame:
        # The adjusted daily series carries per-day dividend and split-coefficient
        # columns; derive actions from them rather than making a second call.
        payload = self._get(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": security.symbol,
                "outputsize": "full",
            },
            symbol=security.symbol,
        )
        series = payload.get("Time Series (Daily)")
        if not series:
            return empty_corporate_actions()

        records: list[dict[str, object]] = []
        for day, values in series.items():
            when = pd.Timestamp(day)
            dividend = float(values.get("7. dividend amount", 0.0) or 0.0)
            split = float(values.get("8. split coefficient", 1.0) or 1.0)
            if dividend > 0:
                records.append(
                    {
                        "date": when,
                        "action_type": CorporateActionType.DIVIDEND.value,
                        "value": dividend,
                    }
                )
            if split not in (0.0, 1.0):
                records.append(
                    {
                        "date": when,
                        "action_type": CorporateActionType.SPLIT.value,
                        "value": split,
                    }
                )
        if not records:
            return empty_corporate_actions()
        return pd.DataFrame.from_records(records)

    def _fetch_delisting_info(self, security: Security) -> DelistingInfo:
        # LISTING_STATUS returns a CSV of active/delisted symbols. Absent a hit
        # we treat the security as listed; authoritative Indian delisting comes
        # from the NSE feed (PRD Module 1).
        return DelistingInfo(delisted=False)
