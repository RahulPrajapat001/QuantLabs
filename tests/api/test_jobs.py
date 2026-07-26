"""Backtest submission, polling, and the gates that fire before either."""

from __future__ import annotations

import time

import pytest

from quantlab.api.jobs import JobRegistry, JobState

from .conftest import INDEX_NAME

SUBMIT = {
    "strategy": "buy_and_hold",
    "index_name": INDEX_NAME,
    "start": "2016-01-04",
    "end": "2017-01-31",
    "cost_model": "IndianEquityCostModel",
    "max_positions": 8,
    "max_position_weight": 0.2,
}


def _await(client, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/backtests/jobs/{job_id}").json()
        if body["state"] in ("complete", "failed"):
            return body
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not settle within {timeout}s")


# --- The registry itself ------------------------------------------------------


def test_a_job_runs_and_reports_its_result():
    registry = JobRegistry(max_workers=1)
    try:
        job = registry.submit("test", lambda: 41 + 1)
        deadline = time.monotonic() + 10
        while registry.get(job.id).state != JobState.COMPLETE:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert registry.get(job.id).result == 42
    finally:
        registry.shutdown()


def test_a_failing_job_records_the_error_not_a_traceback():
    registry = JobRegistry(max_workers=1)
    try:

        def explode() -> None:
            raise ValueError("deliberate")

        job = registry.submit("test", explode)
        deadline = time.monotonic() + 10
        while registry.get(job.id).state != JobState.FAILED:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        settled = registry.get(job.id)
        assert settled.error == "ValueError: deliberate"
        assert "Traceback" not in (settled.error or "")
    finally:
        registry.shutdown()


def test_eviction_never_drops_a_running_job():
    """The invariant in the name: a live job is never evicted.

    Capacity is enforced *opportunistically, at submit time* — `_evict_locked`
    runs from `submit` and can only drop jobs that have reached a terminal state.
    So a burst of submissions onto one worker legitimately exceeds capacity while
    those jobs are still queued: there is nothing evictable yet. That is the
    design (never drop live work), not a bug, but it means "len(jobs) <= capacity+1"
    only holds once a later submission arrives to do the sweeping.

    This test asserts both halves separately instead of assuming the burst alone
    settles: first that no live job is ever dropped, then that capacity is honoured
    once a submission occurs with terminal jobs available to evict. The previous
    version asserted the second half immediately after the burst and passed only
    when the machine happened to be idle enough for jobs to finish between
    submissions.
    """

    registry = JobRegistry(max_workers=1, capacity=2)
    try:
        submitted = [registry.submit("test", lambda: None) for _ in range(6)]

        # Nothing live may be evicted, whatever the capacity pressure.
        for job in submitted:
            assert registry.get(job.id) is not None or job.state.terminal, (
                f"job {job.id[:8]} was evicted while not terminal"
            )

        # Wait for the burst to drain.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            jobs = registry.list(kind="test", limit=50)
            if jobs and all(j.state.terminal for j in jobs):
                break
            time.sleep(0.02)
        assert all(j.state.terminal for j in registry.list(kind="test", limit=50)), (
            "jobs did not finish within 10s"
        )

        # One more submission gives eviction terminal jobs to sweep.
        registry.submit("test", lambda: None)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if (
                len(registry.list(kind="test", limit=50))
                <= registry_capacity(registry) + 1
            ):
                break
            time.sleep(0.02)

        jobs = registry.list(kind="test", limit=50)
        assert len(jobs) <= registry_capacity(registry) + 1, (
            f"capacity {registry_capacity(registry)} not enforced after a sweeping "
            f"submit: {len(jobs)} jobs retained"
        )
    finally:
        registry.shutdown()


def registry_capacity(registry: JobRegistry) -> int:
    """Read the configured capacity without asserting on a private name inline."""

    # Reading a private attribute deliberately: the test asserts on the capacity
    # the registry was *configured* with, not on a duplicated literal that could
    # drift from the constructor call above.
    return registry._capacity


# --- Submission gates ---------------------------------------------------------


def test_an_unknown_strategy_is_rejected_at_submission(client):
    r = client.post("/backtests", json={**SUBMIT, "strategy": "nope"})
    assert r.status_code == 404
    assert "unknown strategy" in r.json()["detail"]


def test_an_unknown_cost_model_is_rejected_at_submission(client):
    r = client.post("/backtests", json={**SUBMIT, "cost_model": "FreeModel"})
    assert r.status_code == 422


def test_a_holdout_window_is_refused_before_the_job_is_queued(client):
    """403 now, not a failed job in thirty seconds."""

    r = client.post("/backtests", json={**SUBMIT, "end": "2024-06-30"})
    assert r.status_code == 403
    assert r.json()["error"] == "holdout_locked"
    assert "holdout" in r.json()["detail"].lower()
    # Nothing was queued.
    assert client.get("/backtests/jobs").json() == []


def test_a_universe_with_no_data_is_404_at_submission(client):
    r = client.post("/backtests", json={**SUBMIT, "index_name": "NOSUCHINDEX"})
    assert r.status_code == 404


def test_polling_an_unknown_job_explains_that_jobs_are_in_process(client):
    r = client.get("/backtests/jobs/deadbeef")
    assert r.status_code == 404
    assert "research_log" in r.json()["detail"]


# --- The full lifecycle -------------------------------------------------------


@pytest.mark.slow
def test_submit_poll_and_read_a_result(client):
    submitted = client.post("/backtests", json=SUBMIT)
    assert submitted.status_code == 202, submitted.text
    job = submitted.json()
    assert job["state"] in ("queued", "running")
    assert job["poll_url"] == f"/backtests/jobs/{job['job_id']}"

    settled = _await(client, job["job_id"])
    assert settled["state"] == "complete", settled["error"]

    result = settled["result"]
    assert result["run_id"]
    # The Sharpe is an object carrying its own trials count.
    assert result["sharpe"]["trials"] >= 1
    assert result["sharpe"]["cost_model"] == "IndianEquityCostModel"
    assert result["costs"]["realistic"] is True
    assert result["costs"]["warning"] is None
    assert result["badge"]["parts"] == ["train"]
    # The equity curve is computed server-side and ready to plot.
    assert len(result["equity_curve"]) > 100
    assert all(point["drawdown"] <= 1e-9 for point in result["equity_curve"])


@pytest.mark.slow
def test_a_zero_cost_run_is_marked_as_not_tradeable(client):
    submitted = client.post(
        "/backtests", json={**SUBMIT, "cost_model": "ZeroCostModel"}
    )
    settled = _await(client, submitted.json()["job_id"])
    assert settled["state"] == "complete", settled["error"]
    costs = settled["result"]["costs"]
    assert costs["realistic"] is False
    assert "GROSS" in costs["warning"]


@pytest.mark.slow
def test_the_run_appears_in_the_research_log_afterwards(client):
    submitted = client.post("/backtests", json=SUBMIT)
    settled = _await(client, submitted.json()["job_id"])
    run_id = settled["result"]["run_id"]

    entry = client.get(f"/research/log/{run_id}").json()
    assert entry["status"] == "complete"
    assert entry["strategy_name"] == "buy_and_hold"
    # No hypothesis was supplied, so it is labelled exploratory rather than
    # presented with the authority of a prediction.
    assert entry["exploratory"] is True
