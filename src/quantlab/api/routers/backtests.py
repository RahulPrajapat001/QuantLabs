"""Backtest submission and polling.

A backtest streams a decade of bars through an event loop; that is past what a
request should hold open, so submission returns a job id and the client polls
(PRD Module 12).

Two things happen at **submission** time rather than in the worker, deliberately:

* **The holdout lock is checked.** A window reaching into locked data is refused
  with 403 immediately rather than accepted, queued, and failed thirty seconds
  later. The engine and the research log check it again independently — this is
  the third gate, not the only one.
* **The strategy and cost model are resolved.** An unknown name is a client error
  and should read as one, not as a job that mysteriously fails.

The result carries a ``SharpeObject``, never a number. Because the run is logged
from inside the engine, its trials count is already correct by the time the poll
returns.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from quantlab.api.deps import SessionDep, load_price_panel, session_factory
from quantlab.api.formatting import EM_DASH
from quantlab.api.jobs import Job, JobState, get_registry
from quantlab.api.models import (
    CostModelStatus,
    DateRange,
    Num,
    ReferenceRatio,
    SharpeObject,
    SplitBadge,
)
from quantlab.config import assert_holdout_allowed

router = APIRouter(prefix="/backtests", tags=["backtests"])


class StrategySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    warmup: int = Field(description="Bars required before the first signal.")
    parameters: dict[str, Any]


class BacktestRequest(BaseModel):
    strategy: str
    index_name: str = "NIFTY500"
    start: date | None = None
    end: date | None = None
    cost_model: str = Field(
        default="IndianEquityCostModel",
        description="ZeroCostModel | GenericBpsModel | IndianEquityCostModel.",
    )
    initial_capital: float = Field(default=1_000_000.0, gt=0.0)
    max_positions: int = Field(default=10, ge=1, le=200)
    max_position_weight: float = Field(default=0.15, gt=0.0, le=1.0)
    hypothesis_id: str | None = Field(
        default=None,
        description=(
            "Pre-registered hypothesis id. Omitted, the run is logged exploratory "
            "— legitimate, but labelled."
        ),
    )


class DrawdownOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_drawdown: Num
    peak_on: date | None
    trough_on: date | None
    recovered_on: date | None
    drawdown_days: Num
    #: Null means never recovered — which is the finding, not a zero.
    recovery_days: Num
    recovered: bool


class EquityPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: date
    equity: float
    drawdown: float = Field(description="Negative fraction from the running peak.")


#: Trades returned with a result. A five-year daily strategy closes thousands of
#: round trips and no one reads past the first screen; the count is reported
#: exactly regardless.
TRADE_LOG_LIMIT = 500


class TradeOut(BaseModel):
    """One closed round trip."""

    model_config = ConfigDict(frozen=True)

    security_id: str
    entry_date: date
    entry_display: str
    exit_date: date | None
    exit_display: str
    holding_days: Num
    quantity: Num
    entry_price: Num
    exit_price: Num
    #: Costs charged across both legs, already deducted from ``pnl``.
    costs: Num
    pnl: Num
    return_pct: Num
    is_win: bool


class BacktestResult(BaseModel):
    """A completed backtest. The Sharpe is an object; there is no bare float."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    strategy: str
    window: DateRange
    badge: SplitBadge
    costs: CostModelStatus
    sharpe: SharpeObject
    #: Sortino and Calmar are not Sharpe ratios, so the deflation rule does not
    #: reach them — but they are still selected results, so they are labelled.
    sortino: Num
    calmar: Num
    benchmark_sharpe: ReferenceRatio
    total_return: Num
    annual_return: Num
    annual_volatility: Num
    drawdown: DrawdownOut
    trades: Num
    win_rate: Num
    profit_factor: Num
    expectancy: Num
    turnover: Num
    total_costs: Num
    cost_drag: Num
    cost_attribution: dict[str, Num]
    rejections: dict[str, int]
    initial_capital: Num
    final_equity: Num
    trading_days: Num
    #: Ready to plot. The frontend computes nothing.
    equity_curve: list[EquityPoint]
    #: Closed round trips, most recent first, capped at ``TRADE_LOG_LIMIT``.
    #: ``trades`` above is the true count — the log is a window onto it, and
    #: ``trade_log_truncated`` says when the two differ.
    trade_log: list[TradeOut]
    trade_log_truncated: bool


class JobOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    kind: str
    state: JobState
    submitted_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None
    parameters: dict[str, Any]
    error: str | None
    result: BacktestResult | None
    poll_url: str


def _job_out(job: Job) -> JobOut:
    return JobOut(
        job_id=job.id,
        kind=job.kind,
        state=job.state,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        duration_seconds=job.duration_seconds,
        parameters=job.parameters,
        error=job.error,
        result=job.result if isinstance(job.result, BacktestResult) else None,
        poll_url=f"/backtests/jobs/{job.id}",
    )


@router.get(
    "/strategies", response_model=list[StrategySummary], summary="Available strategies."
)
def strategies() -> list[StrategySummary]:
    from quantlab.strategies import default_strategies

    return [
        StrategySummary(
            name=name, warmup=strategy.warmup, parameters=strategy.parameters()
        )
        for name, strategy in sorted(default_strategies().items())
    ]


@router.post(
    "",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a backtest. Returns a job id to poll.",
)
def submit(request: BacktestRequest, session: SessionDep) -> JobOut:
    from quantlab.backtesting.costs import default_cost_models
    from quantlab.strategies import default_strategies

    available = default_strategies()
    if request.strategy not in available:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"unknown strategy {request.strategy!r}; available: {sorted(available)}"
            ),
        )
    models = default_cost_models()
    if request.cost_model not in models:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"unknown cost model {request.cost_model!r}; available: "
                f"{sorted(models)}"
            ),
        )

    # Fail fast on the lock. Accepting a job that is guaranteed to be refused
    # wastes the caller's time and buries the reason in a poll response.
    if request.end is not None:
        assert_holdout_allowed(
            request.end,
            holdout=False,
            context=f"backtest of {request.strategy!r} via the API",
        )

    # Load the panel on the request's session so a data problem is a 404 now,
    # not a failed job later.
    load_price_panel(session, request.index_name, start=request.start, end=request.end)

    job = get_registry().submit(
        "backtest",
        lambda: _run_backtest(request),
        parameters=request.model_dump(mode="json"),
    )
    return _job_out(job)


@router.get("/jobs", response_model=list[JobOut], summary="Recent jobs.")
def list_jobs(limit: int = Query(default=25, ge=1, le=200)) -> list[JobOut]:
    return [_job_out(j) for j in get_registry().list(kind="backtest", limit=limit)]


@router.get("/jobs/{job_id}", response_model=JobOut, summary="Poll one job.")
def poll(job_id: str) -> JobOut:
    job = get_registry().get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"no job {job_id!r}. Jobs are in-process and do not survive a "
                "restart; the run itself is still in research_log."
            ),
        )
    return _job_out(job)


def _run_backtest(request: BacktestRequest) -> BacktestResult:
    """The worker. Opens its own session — sessions are not thread-safe."""

    from quantlab.backtesting.costs import default_cost_models
    from quantlab.backtesting.data_handler import DataHandler
    from quantlab.backtesting.engine import BacktestConfig, BacktestEngine
    from quantlab.backtesting.portfolio import Portfolio, RiskLimits
    from quantlab.research.log import ResearchLog
    from quantlab.research.reporting import SharpeReport
    from quantlab.strategies import default_strategies

    session = session_factory()()
    try:
        bars = load_price_panel(
            session, request.index_name, start=request.start, end=request.end
        )
        research_log = ResearchLog(session)
        engine = BacktestEngine(
            research_log,
            strategy=default_strategies()[request.strategy],
            data_handler=DataHandler(bars, start=request.start, end=request.end),
            portfolio=Portfolio(
                initial_capital=request.initial_capital,
                limits=RiskLimits(
                    max_positions=request.max_positions,
                    max_position_weight=request.max_position_weight,
                ),
                target_positions=request.max_positions,
            ),
            config=BacktestConfig(
                initial_capital=request.initial_capital,
                cost_model=default_cost_models()[request.cost_model],
                universe="point-in-time",
                index_name=request.index_name,
                hypothesis_id=request.hypothesis_id,
            ),
        )
        backtest = engine.run()
        metrics = engine.metrics
        if metrics is None:  # pragma: no cover - engine always sets it
            raise RuntimeError("the engine produced no metrics")

        report = SharpeReport.from_log(research_log, backtest.run_id)
        entry = research_log.entry(backtest.run_id)
        return _to_result(request, backtest.run_id, metrics, report, entry, engine)
    finally:
        session.close()


def _to_result(
    request: BacktestRequest,
    run_id: str,
    metrics: Any,
    report: Any,
    entry: Any,
    engine: Any,
) -> BacktestResult:
    curve = _equity_curve(engine)
    trades, trades_truncated = _trade_log(engine)
    return BacktestResult(
        run_id=run_id,
        strategy=request.strategy,
        window=DateRange.of(metrics.start, metrics.end),
        badge=SplitBadge.from_label(
            entry.split, holdout_touched=bool(entry.holdout_touched)
        ),
        costs=CostModelStatus.from_name(request.cost_model),
        sharpe=SharpeObject.from_report(report),
        sortino=Num.ratio(metrics.sortino),
        calmar=Num.ratio(metrics.calmar),
        benchmark_sharpe=ReferenceRatio.benchmark_sharpe(metrics.benchmark_sharpe),
        total_return=Num.percent(metrics.total_return),
        annual_return=Num.percent(metrics.annual_return),
        annual_volatility=Num.percent(metrics.annual_volatility, decimals=2),
        drawdown=DrawdownOut(
            max_drawdown=Num.percent(-abs(metrics.drawdown.max_drawdown)),
            peak_on=metrics.drawdown.peak_date,
            trough_on=metrics.drawdown.trough_date,
            recovered_on=metrics.drawdown.recovery_date,
            drawdown_days=Num.count(metrics.drawdown.drawdown_days),
            recovery_days=Num.count(metrics.drawdown.recovery_days),
            recovered=metrics.drawdown.recovered,
        ),
        trades=Num.count(metrics.trades),
        win_rate=Num.share(metrics.win_rate),
        profit_factor=Num.ratio(metrics.profit_factor),
        expectancy=Num.currency(metrics.expectancy),
        turnover=Num.share(metrics.turnover),
        total_costs=Num.currency(metrics.total_costs),
        cost_drag=Num.percent(-abs(metrics.cost_drag)),
        cost_attribution={
            name: Num.currency(value)
            for name, value in metrics.cost_attribution.items()
        },
        rejections=dict(metrics.rejections),
        initial_capital=Num.currency(metrics.initial_capital),
        final_equity=Num.currency(metrics.final_equity),
        trading_days=Num.count(metrics.trading_days),
        equity_curve=curve,
        trade_log=trades,
        trade_log_truncated=trades_truncated,
    )


def _trade_log(engine: Any) -> tuple[list[TradeOut], bool]:
    """Closed round trips, newest first.

    Reads the ledger's own trade objects rather than ``trades_frame()``: the
    frame is for analysis in a notebook, and round-tripping through pandas here
    would turn typed dates into timestamps for no benefit.
    """

    ledger = getattr(engine, "ledger", None)
    closed = list(getattr(ledger, "closed_trades", []) or [])
    if not closed:
        return [], False

    closed.sort(key=lambda t: t.exit_date or t.entry_date, reverse=True)
    window = closed[:TRADE_LOG_LIMIT]
    return [
        TradeOut(
            security_id=t.security_id,
            entry_date=t.entry_date,
            entry_display=t.entry_date.isoformat(),
            exit_date=t.exit_date,
            exit_display=t.exit_date.isoformat() if t.exit_date else EM_DASH,
            holding_days=Num.count(t.holding_days),
            quantity=Num.count(t.quantity),
            entry_price=Num.currency(t.entry_price),
            exit_price=Num.currency(t.exit_price),
            costs=Num.currency(t.costs),
            pnl=Num.currency(t.pnl),
            return_pct=Num.percent(t.return_pct),
            is_win=bool(t.is_win),
        )
        for t in window
    ], len(closed) > len(window)


def _equity_curve(engine: Any) -> list[EquityPoint]:
    """Equity and drawdown per day, computed here so the client does not."""

    import numpy as np

    ledger = getattr(engine, "ledger", None)
    if ledger is None:
        return []
    series = ledger.equity_curve
    if series is None or len(series) == 0:
        return []
    values = series.to_numpy(dtype=float)
    peak = np.maximum.accumulate(values)
    drawdown = np.divide(values, peak, out=np.ones_like(values), where=peak > 0) - 1.0
    import pandas as pd

    return [
        EquityPoint(
            date=pd.Timestamp(when).date(),
            equity=float(equity),
            drawdown=float(dd),
        )
        for when, equity, dd in zip(series.index, values, drawdown, strict=True)
    ]
