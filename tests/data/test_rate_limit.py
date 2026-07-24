"""Rate limiting and retry-with-backoff behaviour (no real sleeping)."""

from __future__ import annotations

import pytest

from quantlab.data.errors import (
    PermanentProviderError,
    TransientProviderError,
)
from quantlab.data.rate_limit import RateLimiter, RetryPolicy, call_with_retry


def test_rate_limiter_waits_min_interval() -> None:
    now = [0.0]
    slept: list[float] = []
    limiter = RateLimiter(60.0, sleep=lambda s: slept.append(s), clock=lambda: now[0])
    limiter.acquire()  # first call: no wait
    limiter.acquire()  # immediately after: must wait ~1s (60/min)
    assert slept
    assert slept[0] == pytest.approx(1.0, abs=1e-6)


def test_retry_succeeds_after_transient_failures() -> None:
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientProviderError("temporary")
        return "ok"

    result = call_with_retry(
        flaky,
        policy=RetryPolicy(max_attempts=5, base_delay=0.0, jitter=0.0),
        sleep=lambda _s: None,
    )
    assert result == "ok"
    assert attempts["n"] == 3


def test_retry_gives_up_after_max_attempts() -> None:
    attempts = {"n": 0}

    def always_transient() -> str:
        attempts["n"] += 1
        raise TransientProviderError("still down")

    with pytest.raises(TransientProviderError):
        call_with_retry(
            always_transient,
            policy=RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0.0),
            sleep=lambda _s: None,
        )
    assert attempts["n"] == 3


def test_permanent_error_is_not_retried() -> None:
    attempts = {"n": 0}

    def permanent() -> str:
        attempts["n"] += 1
        raise PermanentProviderError("bad symbol")

    with pytest.raises(PermanentProviderError):
        call_with_retry(
            permanent,
            policy=RetryPolicy(max_attempts=5, base_delay=0.0),
            sleep=lambda _s: None,
        )
    assert attempts["n"] == 1


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=0.0)
    assert policy.delay_for(1) == pytest.approx(1.0)
    assert policy.delay_for(2) == pytest.approx(2.0)
    assert policy.delay_for(3) == pytest.approx(4.0)
    assert policy.delay_for(10) == pytest.approx(10.0)  # capped
