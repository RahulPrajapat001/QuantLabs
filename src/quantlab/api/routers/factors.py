"""Factor evaluation — IC, decay, quantiles, turnover.

Runs *before and separately from* strategy backtesting (``CLAUDE.md`` §2). A
factor that fails the IC gate never reaches the strategy builder, so this
endpoint reports the gate verdict alongside the statistics rather than leaving
the caller to apply the threshold itself.

Nothing here writes to ``research_log``: a factor evaluation is not a backtest
and does not consume a trial. It also never returns a Sharpe, which is why no
:class:`~quantlab.api.models.SharpeObject` appears in this module.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from quantlab.api.deps import SessionDep, load_price_panel
from quantlab.api.models import DateRange, Num
from quantlab.factors.baselines import baseline_factors
from quantlab.factors.evaluation import FactorEvaluation, evaluate_factor
from quantlab.factors.gate import eligibility_reason, is_eligible

router = APIRouter(prefix="/factors", tags=["factors"])


class FactorSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    feature: str
    direction: int = Field(description="+1 scores high values high, -1 inverts.")


class ICStatsOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon: int
    method: str
    mean_ic: Num
    std_ic: Num
    ic_ir: Num
    t_stat: Num = Field(description="Newey-West corrected for overlapping returns.")
    p_value: Num
    n_periods: Num
    interpretation: str


class EvaluationRequest(BaseModel):
    factor: str = Field(description="A registered factor name.")
    index_name: str = "NIFTY500"
    start: date | None = None
    end: date | None = None
    n_quantiles: int = Field(default=5, ge=2, le=20)


class EvaluationOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    factor_name: str
    window: DateRange
    primary_horizon: int
    horizons: list[int]
    pearson_ic: list[ICStatsOut]
    spearman_ic: list[ICStatsOut]
    ic_decay: dict[int, Num]
    quantile_returns: list[Num]
    long_short_spread: Num
    turnover: Num
    autocorrelation: Num
    fitness: Num
    #: Whether the factor clears the IC gate and may reach the strategy builder.
    eligible: bool
    #: Why it does not, when it does not. Null when eligible.
    ineligibility_reason: str | None


@router.get("", response_model=list[FactorSummary], summary="Registered factors.")
def list_factors() -> list[FactorSummary]:
    return [
        FactorSummary(name=f.name, feature=f.feature, direction=f.direction)
        for f in baseline_factors()
    ]


@router.post(
    "/evaluate",
    response_model=EvaluationOut,
    summary="Evaluate one factor over a window.",
)
def evaluate(request: EvaluationRequest, session: SessionDep) -> EvaluationOut:
    factors = {f.name: f for f in baseline_factors()}
    if request.factor not in factors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"unknown factor {request.factor!r}; available: {sorted(factors)}"),
        )

    import pandas as pd

    from quantlab.feature_engineering.build import read_feature_panel, universe_by_date

    prices = load_price_panel(
        session, request.index_name, start=request.start, end=request.end
    )
    try:
        panel = read_feature_panel(start=request.start, end=request.end)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "the feature dataset has not been built. Run: "
                f"quantlab features build --index {request.index_name}"
            ),
        ) from exc
    if panel.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no feature rows in the requested window",
        )

    dates = sorted({d.date() for d in pd.to_datetime(panel["date"])})
    universe = universe_by_date(session, request.index_name, dates)

    # allow_holdout is deliberately absent: a factor decision informed by holdout
    # data would poison every strategy built on it, and the API offers no switch.
    evaluation = evaluate_factor(
        factors[request.factor],
        panel,
        prices,
        universe=universe,
        n_quantiles=request.n_quantiles,
    )
    return _to_out(evaluation, request)


def _ic_out(stats: object) -> ICStatsOut:
    from quantlab.factors.evaluation import ICStats

    assert isinstance(stats, ICStats)
    return ICStatsOut(
        horizon=stats.horizon,
        method=stats.method,
        mean_ic=Num.ic(stats.mean_ic),
        std_ic=Num.ic(stats.std_ic),
        ic_ir=Num.ratio(stats.ic_ir),
        t_stat=Num.t_stat(stats.t_stat),
        p_value=Num.p_value(stats.p_value),
        n_periods=Num.count(stats.n_periods),
        interpretation=stats.interpretation,
    )


def _to_out(evaluation: FactorEvaluation, request: EvaluationRequest) -> EvaluationOut:
    return EvaluationOut(
        factor_name=evaluation.factor_name,
        window=DateRange.of(request.start, request.end),
        primary_horizon=evaluation.primary_horizon,
        horizons=list(evaluation.horizons),
        pearson_ic=[_ic_out(s) for s in evaluation.pearson_ic.values()],
        spearman_ic=[_ic_out(s) for s in evaluation.spearman_ic.values()],
        ic_decay={h: Num.ic(v) for h, v in evaluation.ic_decay.items()},
        quantile_returns=[Num.percent(v) for v in evaluation.quantile_returns],
        long_short_spread=Num.percent(evaluation.long_short_spread),
        turnover=Num.share(evaluation.turnover),
        autocorrelation=Num.ratio(evaluation.factor_autocorrelation),
        fitness=Num.ratio(evaluation.fitness),
        eligible=is_eligible(evaluation),
        ineligibility_reason=eligibility_reason(evaluation),
    )
