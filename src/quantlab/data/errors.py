"""Typed provider errors.

Callers distinguish *transient* failures (worth retrying — rate limits, 5xx,
timeouts, connection resets) from *permanent* failures (a bad symbol, an
authentication problem, a delisted-with-no-data response). The retry machinery
in :mod:`quantlab.data.rate_limit` only retries :class:`TransientProviderError`;
everything else surfaces immediately.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for all data-provider failures.

    Carries the vendor ``symbol`` and provider ``source`` for context so that a
    failure deep inside a retry loop still names what it was fetching.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        symbol: str | None = None,
    ) -> None:
        self.source = source
        self.symbol = symbol
        prefix = ""
        if source or symbol:
            prefix = f"[{source or '?'}:{symbol or '?'}] "
        super().__init__(f"{prefix}{message}")


class TransientProviderError(ProviderError):
    """A retryable failure — rate limiting, a 5xx, a timeout, a dropped socket.

    Retrying the same request after a backoff may succeed.
    """


class PermanentProviderError(ProviderError):
    """A non-retryable failure — unknown symbol, bad API key, malformed payload.

    Retrying will not help; the caller must fix the request or give up.
    """
