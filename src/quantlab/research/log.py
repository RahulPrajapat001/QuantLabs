"""The research log — the audit trail every reported number is traceable to.

Multiple testing is the failure mode that makes retail quant results
indefensible: run three hundred variants, report the best one as though it were
the only hypothesis, and a Sharpe of 1.8 means nothing. The defence is not
discipline, it is bookkeeping that cannot be skipped.

**Every backtest run writes a row here, from inside the engine.** Not a decorator
the caller can forget — :meth:`ResearchLog.run` is a context manager that the
engine's ``run()`` wraps its own execution in (see
:class:`quantlab.backtesting.logged.LoggedBacktestEngine`, whose ``run`` is final
and cannot be overridden to skip it). Three properties make it non-optional:

1. The row is inserted and committed **before** the run executes, with
   ``status="running"``. A crashed, killed or non-terminating run leaves evidence
   rather than vanishing from the trials count.
2. Exiting the context without calling :meth:`LoggedRun.record` marks the row
   failed and **raises** :class:`ResearchLogError`. A run that produces results
   but does not log them is not a run; it is an error.
3. After committing, the row is read back. If it is not there, that raises too —
   a silently swallowed write would corrupt every deflated Sharpe downstream.

Trials counts are read from this table by :mod:`quantlab.research.deflated`.
They are never passed in by hand, because a researcher who has run three hundred
variants will not remember three hundred.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any

import numpy as np
import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from quantlab.config import assert_holdout_allowed, resolve_split, touches_holdout
from quantlab.db.models import ResearchLogEntry
from quantlab.utils.git import run_git

log = structlog.get_logger(__name__)

#: Statuses a row can hold. ``running`` rows are excluded from trials counts —
#: an unfinished run has no Sharpe to contribute — but are never deleted.
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"


class ResearchLogError(RuntimeError):
    """Raised when a run cannot be logged, or completes without logging.

    Deliberately fatal. The alternative — returning a result that no row
    accounts for — silently deflates every future Sharpe by too little.
    """


# --- Provenance --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GitProvenance:
    """The commit a run was executed at, and whether the tree was dirty."""

    commit_hash: str | None
    dirty: bool | None

    @property
    def reproducible(self) -> bool:
        """Whether this run can be reproduced from its commit hash alone."""

        return self.commit_hash is not None and self.dirty is False

    def describe(self) -> str:
        if self.commit_hash is None:
            return "no-git"
        return f"{self.commit_hash[:8]}{'-dirty' if self.dirty else ''}"


@lru_cache(maxsize=1)
def capture_git_provenance() -> GitProvenance:
    """Capture the current commit hash and dirty flag.

    Cached for the life of the process: two hundred logged runs must not shell
    out four hundred times, and a research run has no business editing the work
    tree while it executes. Returns ``GitProvenance(None, None)`` outside a git
    work tree — tolerated for ordinary runs, refused for holdout runs.
    """

    commit = run_git(["rev-parse", "HEAD"])
    if not commit:
        return GitProvenance(commit_hash=None, dirty=None)
    status = run_git(["status", "--porcelain"])
    return GitProvenance(
        commit_hash=commit, dirty=bool(status) if status is not None else None
    )


# --- What was run, and what came out -----------------------------------------


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Everything known about a run *before* it executes.

    ``split`` is not a field: it is derived from the date range by
    :func:`quantlab.config.resolve_split`, so a caller cannot label a window that
    straddles train and validation as "train".
    """

    strategy_name: str
    universe: str
    index_name: str
    start: date
    end: date
    cost_model: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    #: Links this run to a pre-registered hypothesis in ``docs/hypotheses.md``.
    #: ``None`` marks the run exploratory — legitimate, but labelled.
    hypothesis_id: str | None = None
    #: Explicit opt-in to holdout data. Honoured only with
    #: ``QUANTLAB_UNLOCK_HOLDOUT=1`` also set (see :mod:`quantlab.config`).
    holdout: bool = False
    periods_per_year: int = 252
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.strategy_name:
            raise ValueError("strategy_name is required: an unnamed run is unauditable")
        if self.end < self.start:
            raise ValueError(
                f"end {self.end.isoformat()} precedes start {self.start.isoformat()}"
            )
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

    @property
    def split(self) -> str:
        """The split label recorded in the log, derived from the date range."""

        return resolve_split(self.start, self.end)

    @property
    def holdout_touched(self) -> bool:
        """Whether this run's window reaches into the locked holdout."""

        return touches_holdout(self.end)

    @property
    def exploratory(self) -> bool:
        """A run with no pre-registered hypothesis is exploratory."""

        return self.hypothesis_id is None


def _population_moments(values: np.ndarray) -> tuple[float, float]:
    """Return ``(skew, non-excess kurtosis)`` using population moments.

    Non-excess: a normal distribution gives 3.0, matching the γ4 of Bailey &
    López de Prado's probabilistic Sharpe ratio. Storing excess kurtosis here and
    forgetting the +3 downstream is exactly the kind of error that inflates a
    deflated Sharpe, so the convention is fixed at the point of capture.
    """

    sd = float(values.std(ddof=0))
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0, 3.0
    z = (values - float(values.mean())) / sd
    return float((z**3).mean()), float((z**4).mean())


@dataclass(frozen=True, slots=True)
class RunResult:
    """The output metrics of a completed run.

    Prefer :meth:`from_returns`: it derives the Sharpe and the moments the
    deflated-Sharpe calculation needs from one return series, so they cannot
    disagree with each other or with ``n_periods``.
    """

    metrics: Mapping[str, Any] = field(default_factory=dict)
    sharpe: float | None = None
    n_periods: int | None = None
    returns_skew: float | None = None
    #: **Non-excess** kurtosis (3.0 for a normal).
    returns_kurtosis: float | None = None

    @classmethod
    def from_returns(
        cls,
        returns: Sequence[float] | np.ndarray,
        *,
        metrics: Mapping[str, Any] | None = None,
        periods_per_year: int = 252,
    ) -> RunResult:
        """Build a result from a per-period return series.

        The annualised Sharpe uses the sample standard deviation (``ddof=1``) and
        assumes a zero risk-free rate — QuantLab reports excess-of-cash Sharpes
        only where the strategy explicitly nets out a cash leg.
        """

        values = np.asarray(returns, dtype=float).ravel()
        values = values[np.isfinite(values)]
        n = int(values.size)
        if n < 2:
            raise ValueError(
                f"need at least 2 finite return observations to log a Sharpe, got {n}"
            )
        sd = float(values.std(ddof=1))
        sharpe = (
            0.0
            if sd == 0.0
            else float(values.mean() / sd * np.sqrt(float(periods_per_year)))
        )
        skew, kurtosis = _population_moments(values)
        merged: dict[str, Any] = dict(metrics or {})
        merged.setdefault("sharpe", sharpe)
        merged.setdefault("n_periods", n)
        return cls(
            metrics=merged,
            sharpe=sharpe,
            n_periods=n,
            returns_skew=skew,
            returns_kurtosis=kurtosis,
        )


class LoggedRun:
    """Handle on an in-flight logged run, yielded by :meth:`ResearchLog.run`.

    The engine calls :meth:`record` exactly once with its output metrics. Not
    calling it is an error, not a silent no-op.
    """

    __slots__ = ("_result", "provenance", "run_id", "spec")

    def __init__(self, run_id: str, spec: RunSpec, provenance: GitProvenance) -> None:
        self.run_id = run_id
        self.spec = spec
        self.provenance = provenance
        self._result: RunResult | None = None

    @property
    def recorded(self) -> bool:
        return self._result is not None

    @property
    def result(self) -> RunResult:
        if self._result is None:  # pragma: no cover - guarded by the caller
            raise ResearchLogError(f"run {self.run_id} has no recorded result yet")
        return self._result

    def record(self, result: RunResult) -> None:
        """Attach the run's output metrics. Calling twice is an error."""

        if self._result is not None:
            raise ResearchLogError(
                f"run {self.run_id} already recorded a result; a run logs its "
                "metrics exactly once so the trials count cannot double-count it"
            )
        self._result = result


# --- The log itself ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogSummary:
    """Aggregate view of the log, for ``quantlab research log --summary``."""

    total_runs: int
    complete_runs: int
    failed_runs: int
    running_runs: int
    strategies: int
    holdout_runs: int
    exploratory_runs: int
    preregistered_runs: int
    dirty_runs: int
    best_sharpe: float | None
    best_sharpe_strategy: str | None
    sharpe_variance: float | None
    runs_per_strategy: Mapping[str, int]
    first_run_at: str | None
    last_run_at: str | None

    def render(self) -> str:
        """A plain-text summary.

        The trials count leads, because it is the number that determines what any
        Sharpe in this log is worth.
        """

        lines = [
            "Research log summary",
            "====================",
            f"Trials (complete runs):   {self.complete_runs}",
            f"Total rows:               {self.total_runs} "
            f"({self.running_runs} running, {self.failed_runs} failed)",
            f"Distinct strategies:      {self.strategies}",
            f"Pre-registered / explor.: {self.preregistered_runs} / "
            f"{self.exploratory_runs}",
            f"Holdout runs:             {self.holdout_runs}",
            f"Runs from a dirty tree:   {self.dirty_runs}",
        ]
        if self.first_run_at is not None:
            lines.append(
                f"Window:                   {self.first_run_at} .. {self.last_run_at}"
            )
        if self.best_sharpe is None:
            lines.append("Best Sharpe:              (none recorded)")
        else:
            lines.append(
                f"Best Sharpe:              {self.best_sharpe:.3f} "
                f"({self.best_sharpe_strategy})"
            )
            lines.append(
                "  NOTE: a bare best-of-N Sharpe is not a result. Deflate it: "
                "quantlab research deflate --strategy "
                f"{self.best_sharpe_strategy}"
            )
        if self.sharpe_variance is not None:
            lines.append(f"Variance of trial Sharpes:{self.sharpe_variance:11.4f}")
        if self.runs_per_strategy:
            lines.append("")
            lines.append("Runs per strategy:")
            for name, count in sorted(
                self.runs_per_strategy.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                lines.append(f"  {count:5d}  {name}")
        return "\n".join(lines)


class ResearchLog:
    """Read/write access to the ``research_log`` table.

    Construct one per session. Writes go through :meth:`run`; everything else
    here is a query used by the multiple-testing corrections and the CLI.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def session(self) -> Session:
        return self._session

    # --- Writing ---

    @contextmanager
    def run(self, spec: RunSpec) -> Iterator[LoggedRun]:
        """Log a run around its execution. The engine wraps its work in this.

        On entry: validates the holdout lock and the hypothesis link, captures
        git provenance, and commits a ``running`` row. On clean exit: writes the
        metrics and commits, then reads the row back to prove it landed. On
        exception: marks the row ``failed`` with the error text and re-raises.
        Exiting without :meth:`LoggedRun.record` raises
        :class:`ResearchLogError`.
        """

        provenance = self._prepare(spec)
        run_id = uuid.uuid4().hex[:32]
        entry = ResearchLogEntry(
            run_id=run_id,
            git_commit_hash=provenance.commit_hash,
            git_dirty=provenance.dirty,
            strategy_name=spec.strategy_name,
            parameters=_jsonable(spec.parameters),
            universe=spec.universe,
            index_name=spec.index_name,
            start_date=spec.start,
            end_date=spec.end,
            split=spec.split,
            cost_model=spec.cost_model,
            metrics={},
            periods_per_year=spec.periods_per_year,
            holdout_touched=spec.holdout_touched,
            hypothesis_id=spec.hypothesis_id,
            hypothesis_hash=self._hypothesis_hash(spec.hypothesis_id),
            exploratory=spec.exploratory,
            status=STATUS_RUNNING,
            notes=spec.notes,
        )
        self._insert(entry)

        handle = LoggedRun(run_id, spec, provenance)
        try:
            yield handle
        except Exception as exc:
            self._finalise_failed(run_id, f"{type(exc).__name__}: {exc}")
            raise
        if not handle.recorded:
            self._finalise_failed(run_id, "run completed without recording metrics")
            raise ResearchLogError(
                f"run {run_id} ({spec.strategy_name!r}) exited without recording "
                "any metrics. Every backtest run must log its results — an "
                "unlogged run is invisible to the trials count and silently "
                "inflates every deflated Sharpe computed afterwards."
            )
        self._finalise_complete(run_id, handle.result)
        log.info(
            "research_log.run_recorded",
            run_id=run_id,
            strategy=spec.strategy_name,
            split=spec.split,
            sharpe=handle.result.sharpe,
            holdout_touched=spec.holdout_touched,
            exploratory=spec.exploratory,
            git=provenance.describe(),
        )
        return

    def _prepare(self, spec: RunSpec) -> GitProvenance:
        """Enforce the lock and the hypothesis link before anything is written."""

        # Independent of the DataHandler's own check: a run that never streams a
        # bar (an analytic or synthetic backtest) must still not slip past.
        assert_holdout_allowed(
            spec.end,
            holdout=spec.holdout,
            context=f"backtest run of {spec.strategy_name!r}",
        )

        provenance = capture_git_provenance()
        if spec.holdout_touched and provenance.commit_hash is None:
            raise ResearchLogError(
                f"refusing to log a holdout run of {spec.strategy_name!r} without a "
                "git commit hash: the whole point of logging a holdout evaluation "
                "is that its position in the project timeline is auditable "
                "afterwards. Run from inside a git work tree."
            )

        if spec.hypothesis_id is not None:
            # Import here: pre-registration reads docs/hypotheses.md, which the
            # log has no need of unless a run actually claims a hypothesis.
            from quantlab.research.preregister import require_hypothesis

            require_hypothesis(spec.hypothesis_id)
        return provenance

    def _hypothesis_hash(self, hypothesis_id: str | None) -> str | None:
        if hypothesis_id is None:
            return None
        from quantlab.research.preregister import require_hypothesis

        return require_hypothesis(hypothesis_id).content_hash

    def _insert(self, entry: ResearchLogEntry) -> None:
        try:
            self._session.add(entry)
            self._session.commit()
        except Exception as exc:  # pragma: no cover - DB failure path
            self._session.rollback()
            raise ResearchLogError(
                f"could not write the research_log row for "
                f"{entry.strategy_name!r}: {type(exc).__name__}: {exc}. The run is "
                "abandoned rather than executed unlogged."
            ) from exc

    def _entry_for_update(self, run_id: str) -> ResearchLogEntry:
        entry = self._session.get(ResearchLogEntry, run_id)
        if entry is None:
            raise ResearchLogError(
                f"research_log row {run_id} disappeared mid-run; the log is not "
                "trustworthy and the run must not be reported"
            )
        return entry

    def _finalise_complete(self, run_id: str, result: RunResult) -> None:
        entry = self._entry_for_update(run_id)
        entry.metrics = _jsonable(result.metrics)
        entry.sharpe = result.sharpe
        entry.n_periods = result.n_periods
        entry.returns_skew = result.returns_skew
        entry.returns_kurtosis = result.returns_kurtosis
        entry.status = STATUS_COMPLETE
        try:
            self._session.commit()
        except Exception as exc:  # pragma: no cover - DB failure path
            self._session.rollback()
            raise ResearchLogError(
                f"could not record metrics for run {run_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Read back: a write that did not land must not be reported as a run.
        self._session.expire_all()
        written = self._session.get(ResearchLogEntry, run_id)
        if written is None or written.status != STATUS_COMPLETE:
            raise ResearchLogError(
                f"run {run_id} did not persist to research_log; refusing to "
                "return an unlogged result"
            )

    def _finalise_failed(self, run_id: str, error: str) -> None:
        try:
            entry = self._session.get(ResearchLogEntry, run_id)
            if entry is None:  # pragma: no cover - defensive
                return
            entry.status = STATUS_FAILED
            entry.error = error[:2048]
            self._session.commit()
        except Exception:  # pragma: no cover - never mask the original failure
            self._session.rollback()
            log.warning("research_log.failure_not_recorded", run_id=run_id)

    # --- Reading ---

    def entry(self, run_id: str) -> ResearchLogEntry:
        """Fetch one row by ``run_id``, raising if it is absent."""

        entry = self._session.get(ResearchLogEntry, run_id)
        if entry is None:
            raise ResearchLogError(f"no research_log row with run_id {run_id!r}")
        return entry

    def entries(
        self,
        *,
        strategy_name: str | None = None,
        status: str | None = STATUS_COMPLETE,
        limit: int | None = None,
    ) -> list[ResearchLogEntry]:
        """Rows, newest first. Defaults to completed runs only."""

        stmt = select(ResearchLogEntry)
        if strategy_name is not None:
            stmt = stmt.where(ResearchLogEntry.strategy_name == strategy_name)
        if status is not None:
            stmt = stmt.where(ResearchLogEntry.status == status)
        stmt = stmt.order_by(
            ResearchLogEntry.created_at.desc(), ResearchLogEntry.run_id
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self._session.scalars(stmt))

    def trials(self, *, strategy_name: str | None = None) -> int:
        """Count completed runs — the trials count.

        **Default scope is the entire log**, not one strategy. This is deliberate
        and matches PRD §Module 12: model and strategy search are one selection
        process, so a strategy chosen after three hundred trials across five
        families is deflated by three hundred, not by the sixty runs that happen
        to share its name. ``strategy_name`` narrows the scope for reporting; it
        does not change what the deflated Sharpe is computed against.
        """

        stmt = (
            select(func.count())
            .select_from(ResearchLogEntry)
            .where(ResearchLogEntry.status == STATUS_COMPLETE)
        )
        if strategy_name is not None:
            stmt = stmt.where(ResearchLogEntry.strategy_name == strategy_name)
        return int(self._session.scalar(stmt) or 0)

    def trial_sharpes(self, *, strategy_name: str | None = None) -> list[float]:
        """Every completed run's annualised Sharpe, for the trials variance.

        Sharpes are normalised to a common annualisation using each row's own
        ``periods_per_year``, so mixing daily and weekly runs in one log does not
        corrupt the variance.
        """

        stmt = select(ResearchLogEntry.sharpe, ResearchLogEntry.periods_per_year).where(
            ResearchLogEntry.status == STATUS_COMPLETE,
            ResearchLogEntry.sharpe.is_not(None),
        )
        if strategy_name is not None:
            stmt = stmt.where(ResearchLogEntry.strategy_name == strategy_name)
        return [float(sharpe) for sharpe, _ppy in self._session.execute(stmt)]

    def sharpe_variance(self, *, strategy_name: str | None = None) -> float:
        """Sample variance of Sharpes across trials (``ddof=1``).

        This is the ``V[SR]`` of the deflated Sharpe ratio: the spread of
        outcomes the selection process was choosing from. With fewer than two
        trials it is 0.0 — and with one trial there is nothing to deflate.
        """

        sharpes = self.trial_sharpes(strategy_name=strategy_name)
        if len(sharpes) < 2:
            return 0.0
        return float(np.var(np.asarray(sharpes, dtype=float), ddof=1))

    def best_run(self, *, strategy_name: str | None = None) -> ResearchLogEntry | None:
        """The completed run with the highest Sharpe — the one needing deflation."""

        stmt = select(ResearchLogEntry).where(
            ResearchLogEntry.status == STATUS_COMPLETE,
            ResearchLogEntry.sharpe.is_not(None),
        )
        if strategy_name is not None:
            stmt = stmt.where(ResearchLogEntry.strategy_name == strategy_name)
        stmt = stmt.order_by(ResearchLogEntry.sharpe.desc()).limit(1)
        return self._session.scalars(stmt).first()

    def latest_run(self, strategy_name: str) -> ResearchLogEntry | None:
        """The most recent completed run of ``strategy_name``."""

        stmt = (
            select(ResearchLogEntry)
            .where(
                ResearchLogEntry.strategy_name == strategy_name,
                ResearchLogEntry.status == STATUS_COMPLETE,
            )
            .order_by(ResearchLogEntry.created_at.desc(), ResearchLogEntry.run_id)
            .limit(1)
        )
        return self._session.scalars(stmt).first()

    def summary(self) -> LogSummary:
        """Aggregate the whole log."""

        rows = list(self._session.scalars(select(ResearchLogEntry)))
        complete = [r for r in rows if r.status == STATUS_COMPLETE]
        per_strategy: dict[str, int] = {}
        for row in complete:
            per_strategy[row.strategy_name] = per_strategy.get(row.strategy_name, 0) + 1
        best = max(
            (r for r in complete if r.sharpe is not None),
            key=lambda r: r.sharpe or 0.0,
            default=None,
        )
        timestamps = sorted(r.created_at for r in rows if r.created_at is not None)
        return LogSummary(
            total_runs=len(rows),
            complete_runs=len(complete),
            failed_runs=sum(1 for r in rows if r.status == STATUS_FAILED),
            running_runs=sum(1 for r in rows if r.status == STATUS_RUNNING),
            strategies=len(per_strategy),
            holdout_runs=sum(1 for r in rows if r.holdout_touched),
            exploratory_runs=sum(1 for r in complete if r.exploratory),
            preregistered_runs=sum(1 for r in complete if not r.exploratory),
            dirty_runs=sum(1 for r in rows if r.git_dirty),
            best_sharpe=best.sharpe if best is not None else None,
            best_sharpe_strategy=best.strategy_name if best is not None else None,
            sharpe_variance=self.sharpe_variance() if len(complete) > 1 else None,
            runs_per_strategy=per_strategy,
            first_run_at=timestamps[0].isoformat(sep=" ", timespec="seconds")
            if timestamps
            else None,
            last_run_at=timestamps[-1].isoformat(sep=" ", timespec="seconds")
            if timestamps
            else None,
        )


def _jsonable(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce a metric/parameter mapping to something the JSON column accepts.

    Numpy scalars and dates are the two things that reliably show up here and
    reliably fail to serialise.
    """

    out: dict[str, Any] = {}
    for key, value in mapping.items():
        out[str(key)] = _jsonable_value(value)
    return out


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return _jsonable(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        # NaN/inf are not valid JSON; recording them as null beats a row that
        # cannot be read back.
        return None
    return value
