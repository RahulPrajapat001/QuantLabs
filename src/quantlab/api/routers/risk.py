"""VaR, expected shortfall and stress replays.

Two conventions the UI depends on and should not have to know:

* **VaR and ES are positive magnitudes of loss.** A 99% VaR of ``0.031`` means
  "lose 3.1% or worse on one day in a hundred", not "-3.1%". They are formatted
  as unsigned percentages here for that reason.
* **Stress windows replay real ingested history**, not synthetic shocks. A window
  with no data returns a coverage gap rather than an invented number, because a
  stress test on data you do not have is worse than no stress test.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from quantlab.api.deps import SessionDep, load_price_panel, returns_matrix
from quantlab.api.models import DateRange, Num
from quantlab.risk.stress import STRESS_WINDOWS, stress_coverage, stress_test
from quantlab.risk.var import var_comparison

router = APIRouter(prefix="/risk", tags=["risk"])


class VaREntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    confidence: Num
    #: Positive magnitude of the one-period loss.
    var: Num
    expected_shortfall: Num
    #: ES / VaR. Above ~1.15 the tail is fatter than a normal implies.
    tail_ratio: Num
    n_observations: Num
    volatility: Num
    skew: Num
    excess_kurtosis: Num


class VaRRequest(BaseModel):
    index_name: str = "NIFTY500"
    start: date | None = None
    end: date | None = None
    lookback: int = Field(default=504, ge=30, le=5_000)
    #: Equal-weight the universe when absent.
    weights: dict[str, float] | None = None


class VaROut(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: DateRange
    n_assets: Num
    n_observations: Num
    entries: list[VaREntry]
    #: The point of the table is the disagreement between methods.
    interpretation: str


class StressWindowOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str
    name: str
    start: date
    end: date
    window_display: str
    description: str


class CoverageOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str
    name: str
    #: Any data at all for the window.
    available: bool
    #: Data covering the window end to end. Partial coverage is worse than none
    #: if it goes unnoticed — two weeks of the GFC is not a GFC stress test.
    spans_window: bool
    bars: Num
    securities_with_data: Num
    first_bar: date | None
    last_bar: date | None
    detail: str


class StressRequest(BaseModel):
    weights: dict[str, float] | None = None
    slugs: list[str] | None = Field(
        default=None, description="Window slugs. Omit for every configured window."
    )


class StressResultOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str
    name: str
    window: DateRange
    total_return: Num
    max_drawdown: Num
    worst_day: Num
    worst_day_on: date | None
    peak_on: date | None
    trough_on: date | None
    #: Null means the portfolio had NOT recovered by the window's end. It is not
    #: the window length, which would read as a fast recovery.
    recovery_days: Num
    recovered: bool
    volatility: Num
    trading_days: Num
    securities: Num
    mean_correlation: Num


class StressReportOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: list[StressResultOut]
    skipped: list[CoverageOut] = Field(
        description="Windows with no ingested data. Never replaced by an estimate."
    )
    rendered: str


@router.post("/var", response_model=VaROut, summary="VaR and ES, three estimators.")
def value_at_risk(request: VaRRequest, session: SessionDep) -> VaROut:
    import numpy as np
    import pandas as pd

    panel = load_price_panel(
        session, request.index_name, start=request.start, end=request.end
    )
    returns = returns_matrix(panel, lookback=request.lookback)
    if returns.shape[1] == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="no security has a complete return history over that window",
        )

    if request.weights:
        missing = sorted(set(request.weights) - set(returns.columns))
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"no return history for weighted securities: {missing}",
            )
        weights = pd.Series(request.weights, dtype=float)
        weights = weights / weights.abs().sum()
        portfolio = (returns[list(weights.index)] * weights).sum(axis=1)
    else:
        portfolio = returns.mean(axis=1)

    comparison = var_comparison(portfolio)
    entries = [
        VaREntry(
            method=str(r.method),
            confidence=Num.share(r.confidence),
            var=Num.share(r.var),
            expected_shortfall=Num.share(r.expected_shortfall),
            tail_ratio=Num.ratio(r.expected_shortfall / r.var if r.var else np.nan),
            n_observations=Num.count(r.n_observations),
            volatility=Num.share(r.volatility),
            skew=Num.ratio(r.skew),
            excess_kurtosis=Num.ratio(r.excess_kurtosis),
        )
        for r in comparison.results
    ]
    return VaROut(
        window=DateRange.of(
            pd.Timestamp(returns.index[0]).date(),
            pd.Timestamp(returns.index[-1]).date(),
        ),
        n_assets=Num.count(returns.shape[1]),
        n_observations=Num.count(len(portfolio)),
        entries=entries,
        interpretation=(
            "Where parametric VaR sits materially below the historical and "
            "Cornish-Fisher figures, the tail is fatter than a normal "
            "approximation admits and any sizing built on the Gaussian number is "
            "too aggressive."
        ),
    )


@router.get(
    "/stress/windows",
    response_model=list[StressWindowOut],
    summary="The configured historical stress windows.",
)
def stress_windows() -> list[StressWindowOut]:
    return [
        StressWindowOut(
            slug=w.slug,
            name=w.name,
            start=w.start,
            end=w.end,
            window_display=f"{w.start.isoformat()} → {w.end.isoformat()}",
            description=w.description,
        )
        for w in STRESS_WINDOWS
    ]


@router.get(
    "/stress/coverage",
    response_model=list[CoverageOut],
    summary="Which stress windows have ingested data.",
)
def coverage(session: SessionDep) -> list[CoverageOut]:
    return [_coverage_out(c) for c in stress_coverage(session)]


@router.post(
    "/stress",
    response_model=StressReportOut,
    summary="Replay the portfolio through each stress window.",
)
def stress(request: StressRequest, session: SessionDep) -> StressReportOut:
    from quantlab.risk.stress import get_stress_window

    windows = None
    if request.slugs:
        try:
            windows = tuple(get_stress_window(s) for s in request.slugs)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

    report = stress_test(session, weights=request.weights, windows=windows)
    replayed = {r.window.slug for r in report.results}
    return StressReportOut(
        results=[
            StressResultOut(
                slug=r.window.slug,
                name=r.window.name,
                window=DateRange.of(r.window.start, r.window.end),
                total_return=Num.percent(r.total_return),
                max_drawdown=Num.share(r.max_drawdown),
                worst_day=Num.percent(r.worst_day),
                worst_day_on=r.worst_day_date,
                peak_on=r.peak_date,
                trough_on=r.trough_date,
                recovery_days=Num.count(r.recovery_days),
                recovered=r.recovered,
                volatility=Num.share(r.annualised_volatility),
                trading_days=Num.count(r.trading_days),
                securities=Num.count(r.securities),
                mean_correlation=Num.ratio(r.mean_correlation),
            )
            for r in report.results
        ],
        skipped=[
            _coverage_out(c)
            for c in stress_coverage(session)
            if c.window.slug not in replayed
        ],
        rendered=report.render(),
    )


def _coverage_out(coverage_row: object) -> CoverageOut:
    from quantlab.risk.stress import StressCoverage

    assert isinstance(coverage_row, StressCoverage)
    return CoverageOut(
        slug=coverage_row.window.slug,
        name=coverage_row.window.name,
        available=coverage_row.available,
        spans_window=coverage_row.spans_window,
        bars=Num.count(coverage_row.bars),
        securities_with_data=Num.count(coverage_row.securities),
        first_bar=coverage_row.first_bar,
        last_bar=coverage_row.last_bar,
        detail=coverage_row.render().strip(),
    )
