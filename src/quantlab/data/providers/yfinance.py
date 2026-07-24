"""yfinance-backed provider.

This module is one of only two places ``yfinance`` may be imported. The vendor
seam is the ``ticker_factory`` callable (defaulting to ``yfinance.Ticker``): the
provider never touches the network directly, so tests inject a fake factory and
never hit the real API.

yfinance is unofficial and flaky, so every access goes through a
:class:`~quantlab.data.rate_limit.RateLimiter` and
:func:`~quantlab.data.rate_limit.call_with_retry`; network-ish failures are
mapped to :class:`TransientProviderError`, empty/unknown symbols to
:class:`PermanentProviderError`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd
import structlog

from quantlab.data.base import (
    CorporateActionType,
    DataProvider,
    DelistingInfo,
    Frequency,
    Security,
    empty_corporate_actions,
    empty_ohlcv,
)
from quantlab.data.errors import (
    PermanentProviderError,
    ProviderError,
    TransientProviderError,
)
from quantlab.data.rate_limit import RateLimiter, RetryPolicy, call_with_retry

log = structlog.get_logger(__name__)

TickerFactory = Callable[[str], Any]

# Transient at the network layer: yfinance raises bare requests/urllib errors.
_TRANSIENT_HINTS = (
    "timed out",
    "timeout",
    "connection",
    "temporarily",
    "rate",
    "too many requests",
    "503",
    "502",
    "500",
)


def _default_ticker_factory(symbol: str) -> Any:
    import yfinance as yf  # imported lazily so the SDK is optional at import time

    return yf.Ticker(symbol)


def _classify(exc: Exception, *, source: str, symbol: str) -> ProviderError:
    text = str(exc).lower()
    if any(hint in text for hint in _TRANSIENT_HINTS):
        return TransientProviderError(str(exc), source=source, symbol=symbol)
    return PermanentProviderError(str(exc), source=source, symbol=symbol)


class YFinanceProvider(DataProvider):
    """Fetch OHLCV, corporate actions and delisting status from yfinance."""

    source = "yfinance"

    def __init__(
        self,
        *,
        ticker_factory: TickerFactory = _default_ticker_factory,
        rate_limiter: RateLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
        calls_per_minute: float = 60.0,
    ) -> None:
        self._ticker_factory = ticker_factory
        self._rate_limiter = rate_limiter or RateLimiter(calls_per_minute)
        self._retry_policy = retry_policy or RetryPolicy()

    # -- helpers -------------------------------------------------------------

    def _ticker(self, security: Security) -> Any:
        self._rate_limiter.acquire()
        try:
            return self._ticker_factory(security.symbol)
        except Exception as exc:  # pragma: no cover - factory rarely raises here
            raise _classify(exc, source=self.source, symbol=security.symbol) from exc

    def _history(
        self, security: Security, start: date, end: date, frequency: Frequency
    ) -> pd.DataFrame:
        def _call() -> pd.DataFrame:
            ticker = self._ticker(security)
            try:
                # yfinance's ``end`` is exclusive; add a day to include it.
                history: pd.DataFrame = ticker.history(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    interval=frequency.value,
                    auto_adjust=False,
                    actions=True,
                    raise_errors=True,
                )
                return history
            except Exception as exc:
                raise _classify(
                    exc, source=self.source, symbol=security.symbol
                ) from exc

        return call_with_retry(
            _call,
            policy=self._retry_policy,
            description=f"yfinance.history {security.symbol}",
        )

    # -- DataProvider hooks --------------------------------------------------

    def _fetch_ohlcv(
        self,
        security: Security,
        start: date,
        end: date,
        frequency: Frequency,
    ) -> pd.DataFrame:
        raw = self._history(security, start, end, frequency)
        if raw is None or raw.empty:
            return empty_ohlcv()

        frame = raw.reset_index()
        # yfinance names the index "Date" (daily) or "Datetime" (intraday).
        date_col = "Date" if "Date" in frame.columns else frame.columns[0]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(frame[date_col]).dt.tz_localize(None),
                "open": frame["Open"],
                "high": frame["High"],
                "low": frame["Low"],
                "close": frame["Close"],
                # yfinance's "Adj Close" appears only when auto_adjust=False.
                "adj_close": frame.get("Adj Close", frame["Close"]),
                "volume": frame["Volume"],
            }
        )
        # Drop rows yfinance leaves NaN (e.g. a corporate-action-only row).
        out = out.dropna(subset=["open", "high", "low", "close", "adj_close"])
        cleaned: pd.DataFrame = out.reset_index(drop=True)
        return cleaned

    def _fetch_corporate_actions(self, security: Security) -> pd.DataFrame:
        def _call() -> pd.DataFrame:
            ticker = self._ticker(security)
            try:
                actions = ticker.actions
            except Exception as exc:
                raise _classify(
                    exc, source=self.source, symbol=security.symbol
                ) from exc
            return actions if actions is not None else pd.DataFrame()

        actions = call_with_retry(
            _call,
            policy=self._retry_policy,
            description=f"yfinance.actions {security.symbol}",
        )
        if actions is None or actions.empty:
            return empty_corporate_actions()

        frame = actions.reset_index()
        date_col = frame.columns[0]
        records: list[dict[str, object]] = []
        for _, row in frame.iterrows():
            when = pd.to_datetime(row[date_col]).tz_localize(None)
            dividend = float(row.get("Dividends", 0.0) or 0.0)
            split = float(row.get("Stock Splits", 0.0) or 0.0)
            if dividend > 0:
                records.append(
                    {
                        "date": when,
                        "action_type": CorporateActionType.DIVIDEND.value,
                        "value": dividend,
                    }
                )
            if split > 0:
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
        def _call() -> dict[str, Any]:
            ticker = self._ticker(security)
            try:
                meta = ticker.get_history_metadata()
            except Exception as exc:
                raise _classify(
                    exc, source=self.source, symbol=security.symbol
                ) from exc
            return meta or {}

        meta = call_with_retry(
            _call,
            policy=self._retry_policy,
            description=f"yfinance.metadata {security.symbol}",
        )
        # yfinance exposes no explicit delisting flag; a live symbol returns
        # metadata. Absence of tradeable metadata is treated as "still listed"
        # here — authoritative delisting comes from the NSE feed (PRD Module 1).
        if not meta:
            return DelistingInfo(delisted=False)
        return DelistingInfo(delisted=False)
