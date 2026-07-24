"""QuantLab data subpackage: providers, identity, storage, ingestion.

The only supported way to obtain a provider is :func:`get_provider` — callers
name a source string, never import a vendor SDK (PRD §3, Provider abstraction).
"""

from __future__ import annotations

from .base import (
    CORPORATE_ACTION_SCHEMA,
    OHLCV_SCHEMA,
    CorporateActionType,
    DataProvider,
    DelistingInfo,
    Frequency,
    Security,
)
from .errors import (
    PermanentProviderError,
    ProviderError,
    TransientProviderError,
)

__all__ = [
    "CORPORATE_ACTION_SCHEMA",
    "OHLCV_SCHEMA",
    "CorporateActionType",
    "DataProvider",
    "DelistingInfo",
    "Frequency",
    "PermanentProviderError",
    "ProviderError",
    "Security",
    "TransientProviderError",
    "get_provider",
]

#: Registered provider sources. Keys are the strings accepted by the CLI and by
#: :func:`get_provider`; values are the ``source`` recorded on ingested rows.
SUPPORTED_SOURCES: tuple[str, ...] = ("yfinance", "alphavantage")


def get_provider(source: str, **kwargs: object) -> DataProvider:
    """Construct a provider by source name.

    Imports the concrete provider lazily so a missing vendor SDK or API key only
    matters when that provider is actually requested.
    """

    normalized = source.strip().lower()
    if normalized == "yfinance":
        from .providers.yfinance import YFinanceProvider

        return YFinanceProvider(**kwargs)  # type: ignore[arg-type]
    if normalized in {"alphavantage", "alpha_vantage", "alpha-vantage"}:
        from .providers.alpha_vantage import AlphaVantageProvider

        return AlphaVantageProvider(**kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"unknown data source {source!r}; supported: {', '.join(SUPPORTED_SOURCES)}"
    )
