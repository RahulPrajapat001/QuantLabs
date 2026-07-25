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
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pandas is imported lazily inside handlers to keep startup fast
    import pandas as pd

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


class BetaPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: date
    date_display: str
    #: Null during the leading window, where there are not yet enough
    #: observations. The chart draws a gap; it does not back-fill a flat line.
    beta: Num
    alpha: Num


class BetaRequest(BaseModel):
    index_name: str = "NIFTY500"
    start: date | None = None
    end: date | None = None
    #: Observations loaded. Distinct from ``window``, which is the rolling span
    #: the regression runs over — a 60-day beta needs far more than 60 days of
    #: history to produce a series worth plotting.
    lookback: int = Field(default=756, ge=60, le=5_000)
    window: int = Field(default=60, ge=20, le=1_000)
    #: Equal-weight the universe when absent.
    weights: dict[str, float] | None = None


class BetaOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: DateRange
    lookback: Num
    points: list[BetaPoint]
    mean_beta: Num
    min_beta: Num
    max_beta: Num
    #: Says plainly when a single full-sample beta would describe no period.
    interpretation: str


class CorrelationRequest(BaseModel):
    index_name: str = "NIFTY500"
    start: date | None = None
    end: date | None = None
    lookback: int = Field(default=252, ge=30, le=5_000)
    limit: int = Field(
        default=40,
        ge=2,
        le=120,
        description="Securities in the matrix. A 500×500 heatmap is not readable.",
    )


class CorrelationOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: DateRange
    securities: list[str]
    #: Row-major, ``securities × securities``. Raw floats: a correlation matrix
    #: is read as colour, and 1,600 formatted strings would be sent unread.
    matrix: list[list[float]]
    mean_correlation: Num
    max_correlation: Num
    n_observations: Num
    truncated: bool


class MonteCarloRequest(BaseModel):
    index_name: str = "NIFTY500"
    start: date | None = None
    end: date | None = None
    lookback: int = Field(default=504, ge=30, le=5_000)
    n_simulations: int = Field(default=5_000, ge=100, le=50_000)
    horizon_days: int = Field(default=252, ge=5, le=2_520)
    method: str = Field(default="block", description="block | iid")
    weights: dict[str, float] | None = None


class FanBand(BaseModel):
    model_config = ConfigDict(frozen=True)

    percentile: int
    #: Equity multiple of starting capital, one value per simulated day.
    values: list[float]


class MonteCarloOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: DateRange
    n_simulations: Num
    horizon_days: Num
    method: str
    block_size: Num
    source_observations: Num
    bands: list[FanBand]
    terminal_p05: Num
    terminal_p50: Num
    terminal_p95: Num
    max_drawdown_p95: Num
    #: The caveat that belongs next to every simulated result.
    interpretation: str


def _portfolio_returns(
    session: SessionDep,
    index_name: str,
    *,
    start: date | None,
    end: date | None,
    lookback: int,
    weights: dict[str, float] | None,
) -> tuple[pd.DataFrame, pd.Series[float]]:
    """Shared setup: the return matrix and the portfolio series built from it."""

    import pandas as pd

    panel = load_price_panel(session, index_name, start=start, end=end)
    returns = returns_matrix(panel, lookback=lookback)
    if returns.shape[1] == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="no security has a complete return history over that window",
        )
    if weights:
        missing = sorted(set(weights) - set(returns.columns))
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"no return history for weighted securities: {missing}",
            )
        series = pd.Series(weights, dtype=float)
        series = series / series.abs().sum()
        portfolio = (returns[list(series.index)] * series).sum(axis=1)
    else:
        portfolio = returns.mean(axis=1)
    return returns, portfolio


@router.post(
    "/beta", response_model=BetaOut, summary="Rolling beta and alpha vs the index."
)
def beta(request: BetaRequest, session: SessionDep) -> BetaOut:
    """The benchmark is the equal-weighted index over the same window.

    A rolling beta is reported rather than a single number because a full-sample
    beta averages over regimes the portfolio was not the same thing in.
    """

    import numpy as np
    import pandas as pd

    from quantlab.risk.exposure import rolling_beta_alpha
    from quantlab.risk.var import RiskError

    returns, portfolio = _portfolio_returns(
        session,
        request.index_name,
        start=request.start,
        end=request.end,
        lookback=request.lookback,
        weights=request.weights,
    )
    benchmark = returns.mean(axis=1)

    try:
        exposure = rolling_beta_alpha(portfolio, benchmark, window=request.window)
    except RiskError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    points = [
        BetaPoint(
            date=pd.Timestamp(when).date(),
            date_display=pd.Timestamp(when).date().isoformat(),
            beta=Num.ratio(None if pd.isna(b) else float(b)),
            alpha=Num.percent(None if pd.isna(a) else float(a)),
        )
        for when, b, a in zip(
            exposure.beta.index, exposure.beta, exposure.alpha, strict=True
        )
    ]
    finite = exposure.beta.dropna()
    return BetaOut(
        window=DateRange.of(
            pd.Timestamp(exposure.beta.index[0]).date(),
            pd.Timestamp(exposure.beta.index[-1]).date(),
        ),
        lookback=Num.count(request.window),
        points=points,
        mean_beta=Num.ratio(float(finite.mean()) if len(finite) else np.nan),
        min_beta=Num.ratio(float(finite.min()) if len(finite) else np.nan),
        max_beta=Num.ratio(float(finite.max()) if len(finite) else np.nan),
        interpretation=exposure.render().strip(),
    )


@router.post(
    "/correlation",
    response_model=CorrelationOut,
    summary="Pairwise return correlation matrix.",
)
def correlation(request: CorrelationRequest, session: SessionDep) -> CorrelationOut:
    import numpy as np
    import pandas as pd

    panel = load_price_panel(
        session, request.index_name, start=request.start, end=request.end
    )
    returns = returns_matrix(panel, lookback=request.lookback)
    if returns.shape[1] < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="need at least two securities with complete history to correlate",
        )

    all_columns = sorted(returns.columns)
    columns = all_columns[: request.limit]
    matrix = returns[columns].corr()

    # The diagonal is 1 by construction and would drag the mean upward.
    off_diagonal = matrix.to_numpy()[~np.eye(len(columns), dtype=bool)]

    return CorrelationOut(
        window=DateRange.of(
            pd.Timestamp(returns.index[0]).date(),
            pd.Timestamp(returns.index[-1]).date(),
        ),
        securities=columns,
        matrix=[[float(v) for v in row] for row in matrix.to_numpy()],
        mean_correlation=Num.ratio(float(off_diagonal.mean())),
        max_correlation=Num.ratio(float(off_diagonal.max())),
        n_observations=Num.count(len(returns)),
        truncated=len(all_columns) > len(columns),
    )


@router.post(
    "/monte-carlo",
    response_model=MonteCarloOut,
    summary="Bootstrap forward paths from the sampled return distribution.",
)
def monte_carlo_paths(request: MonteCarloRequest, session: SessionDep) -> MonteCarloOut:
    """A resampling of history, not a forecast.

    The bands describe what the *observed* return distribution would produce over
    the horizon if it kept behaving as it did in the sample. It contains no
    regime the sample does not contain, which is exactly why the interpretation
    string says so on every response.
    """

    import numpy as np
    import pandas as pd

    from quantlab.risk.monte_carlo import monte_carlo
    from quantlab.risk.var import RiskError

    _, portfolio = _portfolio_returns(
        session,
        request.index_name,
        start=request.start,
        end=request.end,
        lookback=request.lookback,
        weights=request.weights,
    )

    if request.method not in {"block", "iid"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"unknown method {request.method!r}; use block | iid",
        )

    percentiles = (0.05, 0.25, 0.50, 0.75, 0.95)
    try:
        result = monte_carlo(
            portfolio,
            n_simulations=request.n_simulations,
            horizon_days=request.horizon_days,
            method=request.method,  # type: ignore[arg-type]
            path_percentiles=percentiles,
        )
    except RiskError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    bands = result.path_bands
    assert bands is not None  # path_percentiles was passed
    index = portfolio.index

    return MonteCarloOut(
        window=DateRange.of(
            pd.Timestamp(index[0]).date(), pd.Timestamp(index[-1]).date()
        ),
        n_simulations=Num.count(result.n_simulations),
        horizon_days=Num.count(result.horizon_days),
        method=str(result.method),
        block_size=Num.count(result.block_size),
        source_observations=Num.count(result.source_observations),
        bands=[
            FanBand(
                percentile=round(p * 100),
                values=[float(v) for v in bands[f"p{round(p * 100):02d}"]],
            )
            for p in percentiles
        ],
        terminal_p05=Num.ratio(float(np.quantile(result.terminal_values, 0.05))),
        terminal_p50=Num.ratio(float(np.quantile(result.terminal_values, 0.50))),
        terminal_p95=Num.ratio(float(np.quantile(result.terminal_values, 0.95))),
        max_drawdown_p95=Num.share(float(np.quantile(result.max_drawdowns, 0.95))),
        interpretation=(
            "Bootstrapped from the sampled window, so the simulation contains no "
            "regime the sample does not. A band is a per-day quantile across "
            "paths, not a path any single simulation traced."
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
