"""Job registry for work too slow to hold an HTTP connection open.

PRD Module 12: *"Long-running backtests return a job id and are polled."* A
backtest over a decade of the NIFTY 500 streams a few hundred thousand bars
through an event loop; that is seconds to minutes, which is past what a request
should hold.

Deliberately in-process and deliberately simple: a thread pool, a dict, and a
lock. There is no Redis, no Celery and no broker, because this is a
single-operator research platform and a queue that needs its own infrastructure
is a second thing to keep running. The consequence is stated rather than hidden:
**jobs do not survive a restart.** A backtest that was running when the process
died is lost from the registry — but not from the audit trail, because
:class:`~quantlab.research.log.ResearchLog` committed its ``running`` row before
the work began and marks it ``failed`` on the way out. The research record is
durable even though the job handle is not.

Each worker opens **its own database session**. SQLAlchemy sessions are not
thread-safe, and sharing the request's session with a background thread is the
kind of bug that appears only under concurrent load.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import structlog

log = structlog.get_logger(__name__)

__all__ = ["Job", "JobRegistry", "JobState", "get_registry"]

#: Retained finished jobs. Old entries are evicted oldest-first; the research log
#: is the durable record, this is only a handle for polling.
DEFAULT_CAPACITY = 256


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (JobState.COMPLETE, JobState.FAILED)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Job:
    """One unit of submitted work and everything a poller needs to see."""

    id: str
    kind: str
    state: JobState = JobState.QUEUED
    submitted_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: Whatever the worker returned. Shape depends on ``kind``.
    result: Any = None
    #: Exception summary, when the job failed. Never a traceback for the client.
    error: str | None = None
    #: Echo of the submitted parameters, so a poller can identify its own job.
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or _now()
        return (end - self.started_at).total_seconds()


class JobRegistry:
    """Submit callables, poll them by id. Thread-safe."""

    def __init__(self, *, max_workers: int = 2, capacity: int = DEFAULT_CAPACITY):
        # Two workers by default: backtests are CPU-bound Python, so more threads
        # buy queueing rather than throughput, and a research platform running
        # dozens of concurrent backtests is inflating its own trials count.
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="quantlab-job"
        )
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._capacity = capacity

    def submit(
        self,
        kind: str,
        work: Callable[[], Any],
        *,
        parameters: dict[str, Any] | None = None,
    ) -> Job:
        """Queue ``work`` and return the handle immediately."""

        job = Job(
            id=uuid.uuid4().hex[:32], kind=kind, parameters=dict(parameters or {})
        )
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        self._executor.submit(self._run, job.id, work)
        log.info("api.job.submitted", job_id=job.id, kind=kind)
        return job

    def _run(self, job_id: str, work: Callable[[], Any]) -> None:
        self._transition(job_id, JobState.RUNNING, started_at=_now())
        try:
            result = work()
        except Exception as exc:
            # The client gets the type and message; the full traceback goes to
            # the structured log, where it belongs.
            log.warning(
                "api.job.failed",
                job_id=job_id,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            self._transition(
                job_id,
                JobState.FAILED,
                finished_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        self._transition(job_id, JobState.COMPLETE, finished_at=_now(), result=result)
        log.info("api.job.complete", job_id=job_id)

    def _transition(self, job_id: str, state: JobState, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:  # pragma: no cover - evicted mid-flight
                return
            job.state = state
            for name, value in fields.items():
                setattr(job, name, value)

    def _evict_locked(self) -> None:
        """Drop the oldest terminal jobs once over capacity. Never drops a live one."""

        while len(self._jobs) > self._capacity:
            for job_id, job in self._jobs.items():
                if job.state.terminal:
                    del self._jobs[job_id]
                    break
            else:  # pragma: no cover - every job still running
                return

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, *, kind: str | None = None, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if kind is not None:
            jobs = [j for j in jobs if j.kind == kind]
        jobs.sort(key=lambda j: j.submitted_at, reverse=True)
        return jobs[:limit]

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


_registry: JobRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> JobRegistry:
    """The process-wide registry, created on first use."""

    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = JobRegistry()
        return _registry


def reset_registry() -> None:
    """Drop the registry. For tests, so one test's jobs cannot leak into another."""

    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.shutdown(wait=False)
        _registry = None
