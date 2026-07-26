"""Portfolio optimisation over the point-in-time universe.

Note what the response does *not* contain: a ``SharpeObject``. An optimiser's
Sharpe is an **ex-ante forecast** implied by the covariance estimate, not a
realised track record — there is no return series behind it and nothing to deflate
against. Forcing it into the deflated-Sharpe shape with an invented trials count
would be worse than not deflating it, so it returns
:class:`~quantlab.api.models.ReferenceRatio`, which requires the reason in words.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from quantlab.api.deps import SessionDep, load_price_panel, returns_matrix
from quantlab.api.models import DateRange, Num, ReferenceRatio
from quantlab.portfolio.covariance import estimate_covariance
from quantlab.portfolio.optimisers import (
    OptimisationResult,
    PortfolioConstraints,
    equal_weight,
    inverse_volatility,
    maximum_sharpe,
    minimum_variance,
    risk_parity,
)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

METHODS = (
    "equal_weight",
    "inverse_volatility",
    "risk_parity",
    "minimum_variance",
    "maximum_sharpe",
)


class OptimiseRequest(BaseModel):
    index_name: str = "NIFTY500"
    method: str = Field(default="minimum_variance", description=f"One of {METHODS}.")
    as_of: date | None = Field(
        default=None,
        description="Point-in-time date. Membership is resolved as of this date.",
    )
    lookback: int = Field(default=252, ge=30, le=2_520)
    covariance: str = Field(
        default="ledoit_wolf",
        description=(
            "sample | ledoit_wolf | constant_correlation | single_index. Shrunk "
            "by default: sample covariance is unstable when assets outnumber "
            "observations."
        ),
    )
    max_weight: float = Field(default=0.10, gt=0.0, le=1.0)
    long_only: bool = True
    previous_weights: dict[str, float] | None = Field(
        default=None, description="For turnover measurement against a live book."
    )


class Holding(BaseModel):
    model_config = ConfigDict(frozen=True)

    security_id: str
    weight: Num


class CovarianceOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    n_assets: Num
    n_observations: Num
    shrinkage: Num
    condition_number: Num
    detail: str


class OptimiseOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    status: str = Field(description="cvxpy solver status, or 'analytic'.")
    window: DateRange
    n_assets: Num
    holdings: list[Holding]
    volatility: Num
    expected_return: Num
    #: Ex-ante, not realised. See the class docstring.
    ex_ante_sharpe: ReferenceRatio
    turnover: Num
    #: ``1/Σw²`` — 50 names with an effective count of 4 is a concentrated bet.
    effective_positions: Num
    covariance: CovarianceOut


@router.get("/methods", response_model=list[str], summary="Available optimisers.")
def methods() -> list[str]:
    return list(METHODS)


@router.post("/optimise", response_model=OptimiseOut, summary="Optimise weights.")
def optimise(request: OptimiseRequest, session: SessionDep) -> OptimiseOut:
    from quantlab.universe.membership import universe_as_of

    if request.method not in METHODS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"unknown method {request.method!r}; available: {list(METHODS)}",
        )

    as_of = request.as_of or date.today()
    # Point-in-time membership: the optimiser never sees a security that was not
    # in the index on `as_of`, even if it is in the index today.
    members = universe_as_of(session, request.index_name, as_of)
    if not members:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"{request.index_name} had no constituents on {as_of.isoformat()}"),
        )

    panel = load_price_panel(
        session, request.index_name, end=as_of, security_ids=members
    )
    returns = returns_matrix(panel, lookback=request.lookback)
    if returns.shape[1] < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"only {returns.shape[1]} securities have a complete history over "
                f"the last {request.lookback} sessions; an optimiser needs at "
                "least two"
            ),
        )

    try:
        estimate = estimate_covariance(returns, method=request.covariance)  # type: ignore[arg-type]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    constraints = PortfolioConstraints(
        max_weight=request.max_weight, long_only=request.long_only
    )
    previous = request.previous_weights

    if request.method == "equal_weight":
        result = equal_weight(
            returns.columns, covariance=estimate, previous_weights=previous
        )
    elif request.method == "inverse_volatility":
        result = inverse_volatility(
            estimate, constraints=constraints, previous_weights=previous
        )
    elif request.method == "risk_parity":
        result = risk_parity(
            estimate, max_weight=request.max_weight, previous_weights=previous
        )
    elif request.method == "minimum_variance":
        result = minimum_variance(
            estimate, constraints=constraints, previous_weights=previous
        )
    else:
        result = maximum_sharpe(
            returns.mean(),
            estimate,
            constraints=constraints,
            previous_weights=previous,
        )

    return _to_out(result, estimate, returns, request)


def _to_out(
    result: OptimisationResult,
    estimate: object,
    returns: object,
    request: OptimiseRequest,
) -> OptimiseOut:
    import pandas as pd

    from quantlab.portfolio.covariance import CovarianceEstimate

    assert isinstance(estimate, CovarianceEstimate)
    assert isinstance(returns, pd.DataFrame)

    holdings = [
        Holding(security_id=str(security), weight=Num.share(float(weight)))
        for security, weight in result.weights.sort_values(ascending=False).items()
        if abs(float(weight)) > 1e-9
    ]
    return OptimiseOut(
        method=result.method,
        status=result.status,
        window=DateRange.of(
            pd.Timestamp(returns.index[0]).date(),
            pd.Timestamp(returns.index[-1]).date(),
        ),
        n_assets=Num.count(result.n_assets),
        holdings=holdings,
        # Unsigned, like /risk and /backtests: volatility is a magnitude, and
        # `Num.percent` renders it "+6.43%" — a sign on something that cannot be
        # negative.
        volatility=Num.share(result.volatility),
        expected_return=Num.percent(result.expected_return),
        ex_ante_sharpe=ReferenceRatio.ex_ante_sharpe(result.sharpe),
        turnover=Num.share(result.turnover),
        effective_positions=Num.ratio(result.effective_positions, decimals=1),
        covariance=CovarianceOut(
            method=estimate.method,
            n_assets=Num.count(estimate.n_assets),
            n_observations=Num.count(estimate.n_observations),
            shrinkage=Num.ratio(estimate.shrinkage, decimals=3),
            condition_number=Num.ratio(estimate.condition_number, decimals=1),
            detail=estimate.render(),
        ),
    )
