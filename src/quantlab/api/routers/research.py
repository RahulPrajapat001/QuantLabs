"""The research log, trials counts and deflated Sharpe ratios.

The endpoint group the honesty rules run through. Two things are worth stating
about what it will and will not do:

* **Trials counts are never accepted as input.** Every deflation reads its count
  from ``research_log``. There is no query parameter that sets it, because a
  researcher who has run three hundred variants will not remember three hundred,
  and one who does remember is not the failure mode worth defending against.
* **The default deflation scope is the whole log**, not one strategy. Strategy
  search and model search are one selection process, so a result chosen after 300
  runs across five families is deflated by 300. ``scope_strategy`` narrows it for
  reporting and is documented as doing exactly that.
"""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from quantlab.api.deps import SessionDep
from quantlab.api.formatting import format_date
from quantlab.api.models import CostModelStatus, Num, SharpeObject, SplitBadge
from quantlab.research.deflated import deflate_run
from quantlab.research.log import ResearchLog
from quantlab.research.reporting import SharpeReport

router = APIRouter(prefix="/research", tags=["research"])


class RunOut(BaseModel):
    """One ``research_log`` row.

    ``sharpe`` is a full :class:`SharpeObject` when the run recorded one, and
    ``null`` when it did not (a failed or still-running row). There is no shape
    in which it is a number.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    strategy_name: str
    universe: str
    index_name: str
    start_date: date
    end_date: date
    window_display: str
    badge: SplitBadge
    costs: CostModelStatus
    status: str
    #: True when no hypothesis was pre-registered. Rendered distinctly in the UI.
    exploratory: bool
    hypothesis_id: str | None
    git_commit_hash: str | None
    git_dirty: bool | None
    reproducible: bool = Field(
        description="Has a commit hash and a clean tree — reproducible from git alone."
    )
    created_at: datetime | None
    n_periods: Num
    parameters: dict[str, object]
    error: str | None
    sharpe: SharpeObject | None


class LogPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_trials: Num = Field(description="Completed runs in the whole log.")
    returned: Num
    items: list[RunOut]


class LogSummaryOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_runs: Num
    complete_runs: Num
    failed_runs: Num
    running_runs: Num
    strategies: Num
    holdout_runs: Num
    exploratory_runs: Num
    preregistered_runs: Num
    dirty_runs: Num
    sharpe_variance: Num
    runs_per_strategy: dict[str, int]
    first_run_at: str | None
    last_run_at: str | None
    #: The best Sharpe in the log, deflated. Null when nothing has been recorded.
    best: SharpeObject | None
    rendered: str = Field(description="Plain-text summary, as the CLI prints it.")


def _run_out(entry: object, log: ResearchLog) -> RunOut:
    from quantlab.db.models import ResearchLogEntry

    assert isinstance(entry, ResearchLogEntry)
    sharpe: SharpeObject | None = None
    if entry.sharpe is not None and entry.n_periods is not None:
        # Built through SharpeReport, so the trials count comes from the log.
        sharpe = SharpeObject.from_report(SharpeReport.from_log(log, entry.run_id))
    return RunOut(
        run_id=entry.run_id,
        strategy_name=entry.strategy_name,
        universe=entry.universe,
        index_name=entry.index_name,
        start_date=entry.start_date,
        end_date=entry.end_date,
        window_display=(
            f"{format_date(entry.start_date)} → {format_date(entry.end_date)}"
        ),
        badge=SplitBadge.from_label(
            entry.split, holdout_touched=bool(entry.holdout_touched)
        ),
        costs=CostModelStatus.from_name(entry.cost_model),
        status=entry.status,
        exploratory=bool(entry.exploratory),
        hypothesis_id=entry.hypothesis_id,
        git_commit_hash=entry.git_commit_hash,
        git_dirty=entry.git_dirty,
        reproducible=entry.git_commit_hash is not None and entry.git_dirty is False,
        created_at=entry.created_at,
        n_periods=Num.count(entry.n_periods),
        parameters=dict(entry.parameters or {}),
        error=entry.error,
        sharpe=sharpe,
    )


@router.get("/log", response_model=LogPage, summary="Query the research log.")
def query_log(
    session: SessionDep,
    strategy: str | None = Query(default=None),
    status_filter: str | None = Query(
        default="complete",
        alias="status",
        description="complete | running | failed. Omit for all statuses.",
    ),
    limit: int = Query(default=50, ge=1, le=500),
) -> LogPage:
    log = ResearchLog(session)
    entries = log.entries(strategy_name=strategy, status=status_filter, limit=limit)
    return LogPage(
        total_trials=Num.count(log.trials()),
        returned=Num.count(len(entries)),
        items=[_run_out(e, log) for e in entries],
    )


@router.get("/log/summary", response_model=LogSummaryOut, summary="Aggregate the log.")
def log_summary(session: SessionDep) -> LogSummaryOut:
    log = ResearchLog(session)
    summary = log.summary()
    best_entry = log.best_run()
    best: SharpeObject | None = None
    if best_entry is not None and best_entry.n_periods is not None:
        best = SharpeObject.from_report(SharpeReport.from_log(log, best_entry.run_id))
    return LogSummaryOut(
        total_runs=Num.count(summary.total_runs),
        complete_runs=Num.count(summary.complete_runs),
        failed_runs=Num.count(summary.failed_runs),
        running_runs=Num.count(summary.running_runs),
        strategies=Num.count(summary.strategies),
        holdout_runs=Num.count(summary.holdout_runs),
        exploratory_runs=Num.count(summary.exploratory_runs),
        preregistered_runs=Num.count(summary.preregistered_runs),
        dirty_runs=Num.count(summary.dirty_runs),
        sharpe_variance=Num.ratio(summary.sharpe_variance, decimals=4),
        runs_per_strategy=dict(summary.runs_per_strategy),
        first_run_at=summary.first_run_at,
        last_run_at=summary.last_run_at,
        best=best,
        rendered=summary.render(),
    )


@router.get("/log/{run_id}", response_model=RunOut, summary="One run by id.")
def get_run(run_id: str, session: SessionDep) -> RunOut:
    log = ResearchLog(session)
    entry = log.entry(run_id)
    return _run_out(entry, log)


@router.get(
    "/deflated/{run_id}",
    response_model=SharpeObject,
    summary="Deflate a run's Sharpe against the log's trials count.",
)
def deflated(
    run_id: str,
    session: SessionDep,
    scope_strategy: str | None = Query(
        default=None,
        description=(
            "Narrow the trial universe to one strategy name, for reporting. The "
            "default — the whole log — is the honest scope: model search and "
            "strategy search are one selection process."
        ),
    ),
) -> SharpeObject:
    log = ResearchLog(session)
    # deflate_run raises DeflationError (-> 422) when the run has no Sharpe or no
    # observation count. There is no fallback to an undeflated number.
    deflation = deflate_run(log, run_id, scope_strategy=scope_strategy)
    entry = log.entry(run_id)
    return SharpeObject.from_report(
        SharpeReport.from_deflation(
            deflation,
            split=entry.split,
            cost_model=entry.cost_model,
            run_id=run_id,
            exploratory=bool(entry.exploratory),
            holdout_touched=bool(entry.holdout_touched),
            hypothesis_id=entry.hypothesis_id,
        )
    )


class HypothesisOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    subject: str
    rationale: str
    prediction: str
    metric: str
    threshold: str
    registered_on: str
    status: str
    #: Changes if the hypothesis is edited after a run tested it, which is how a
    #: quietly relaxed threshold becomes detectable.
    content_hash: str


@router.get(
    "/hypotheses",
    response_model=list[HypothesisOut],
    summary="Pre-registered hypotheses.",
)
def hypotheses() -> list[HypothesisOut]:
    from quantlab.research.preregister import load_hypotheses

    try:
        registered = load_hypotheses()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="docs/hypotheses.md not found",
        ) from exc
    return [
        HypothesisOut(
            id=h.id,
            title=h.title,
            subject=h.subject,
            rationale=h.rationale,
            prediction=h.prediction,
            metric=h.metric,
            threshold=f"{h.threshold.operator}{h.threshold.value:g}",
            registered_on=h.registered_on.isoformat() if h.registered_on else "—",
            status=h.status,
            content_hash=h.content_hash,
        )
        for h in registered.values()
    ]
