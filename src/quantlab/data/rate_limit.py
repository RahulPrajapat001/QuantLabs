"""Client-side rate limiting and retry-with-backoff.

Both providers share this machinery. It is deliberately dependency-free and the
sleep function is injectable so tests exercise the full retry path without
actually sleeping or hitting a network.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import structlog

from .errors import TransientProviderError

log = structlog.get_logger(__name__)

T = TypeVar("T")

Sleeper = Callable[[float], None]
Clock = Callable[[], float]


class RateLimiter:
    """Token-free minimum-interval limiter.

    Guarantees at least ``60 / calls_per_minute`` seconds between successive
    :meth:`acquire` calls. Not thread-safe by design — ingestion is per-process
    and sequential per provider instance.
    """

    def __init__(
        self,
        calls_per_minute: float,
        *,
        sleep: Sleeper = time.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive")
        self._min_interval = 60.0 / calls_per_minute
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None

    def acquire(self) -> None:
        """Block until the minimum inter-call interval has elapsed."""

        now = self._clock()
        if self._last_call is not None:
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_call = now


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential-backoff parameters for transient failures."""

    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    #: Multiplicative jitter drawn from ``[1 - jitter, 1 + jitter]``.
    jitter: float = 0.25

    def delay_for(
        self, attempt: int, *, rand: Callable[[], float] = random.random
    ) -> float:
        """Backoff before the ``attempt``-th retry (1-indexed)."""

        raw = self.base_delay * (2.0 ** (attempt - 1))
        capped = min(raw, self.max_delay)
        spread = self.jitter * capped
        return max(0.0, capped + (rand() * 2 - 1) * spread)


def call_with_retry(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    sleep: Sleeper = time.sleep,
    rand: Callable[[], float] = random.random,
    description: str = "provider call",
) -> T:
    """Call ``fn``, retrying only :class:`TransientProviderError`.

    Permanent errors (and any non-provider exception) propagate immediately. The
    last transient error is re-raised once ``max_attempts`` is exhausted.
    """

    last_error: TransientProviderError | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except TransientProviderError as exc:
            last_error = exc
            if attempt == policy.max_attempts:
                break
            delay = policy.delay_for(attempt, rand=rand)
            log.warning(
                "provider.retry",
                description=description,
                attempt=attempt,
                max_attempts=policy.max_attempts,
                delay=round(delay, 3),
                error=str(exc),
            )
            sleep(delay)
    assert last_error is not None  # only reached after a transient failure
    raise last_error
