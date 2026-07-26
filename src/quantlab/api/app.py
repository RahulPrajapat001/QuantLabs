"""The FastAPI application — the sole data source for both frontends.

PRD Module 12 makes one architectural claim that everything else follows from:
**no computation client-side.** The Next.js UI and the Streamlit dashboard render
what this API sends and nothing more. That is not a style preference — a Sharpe
ratio recomputed in TypeScript from a returns array is a second implementation
that will disagree with the Python one eventually, and when it does, no reader can
tell which number is the finding.

So the API sends finished work: equity curves already computed, drawdown series
already derived, split badges already decomposed into segments, cost-model
warnings already decided, and every number already formatted
(:mod:`quantlab.api.formatting`).

Domain errors map to status codes here rather than in each route, so a holdout
violation reads the same whichever endpoint provoked it.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from quantlab.config import HoldoutViolationError, get_settings
from quantlab.utils.logging import (
    REQUEST_ID_HEADER,
    configure_logging,
    request_context,
)

log = structlog.get_logger(__name__)

__all__ = ["create_app"]

DESCRIPTION = """
The QuantLab research API. Sole data source for the Next.js product UI and the
Streamlit researcher dashboard.

**Two contracts constrain every response.**

*Numbers travel with their rendering.* Each numeric field is an object with a raw
`value` for charts and sorting and a `formatted` string to print. Formatting is
implemented once, server-side, so the two frontends and the PDF tearsheet cannot
drift apart. A missing value is `null` with an em dash, never `0` or `NaN`.

*A Sharpe ratio is never a bare float.* Any endpoint returning one returns
`SharpeObject` — `{value, trials, deflated_sharpe, pbo, split, cost_model, …}` —
where `trials` is read from `research_log` and can never be supplied by hand. A
risk-adjusted ratio that deflation genuinely does not apply to (a benchmark, an
optimiser's ex-ante forecast) returns `ReferenceRatio`, which requires a written
reason. There is no shape in this schema that carries a Sharpe without either.
"""

TAGS: list[dict[str, Any]] = [
    {"name": "securities", "description": "Securities and their ticker history."},
    {
        "name": "universe",
        "description": "Point-in-time index membership. Never current constituents.",
    },
    {"name": "market", "description": "OHLCV bars and the engineered feature panel."},
    {"name": "quality", "description": "Data quality findings, computed on demand."},
    {"name": "factors", "description": "Factor evaluation: IC, decay, quantiles."},
    {
        "name": "backtests",
        "description": "Submit a backtest, poll it by job id, read its result.",
    },
    {
        "name": "research",
        "description": "The research log, trials counts and deflated Sharpe ratios.",
    },
    {"name": "risk", "description": "VaR, expected shortfall and stress replays."},
    {"name": "portfolio", "description": "Portfolio optimisation over the universe."},
    {"name": "reports", "description": "PDF tearsheets."},
]


def create_app() -> FastAPI:
    """Build the application. A factory so tests get an isolated instance."""

    configure_logging()
    settings = get_settings()
    app = FastAPI(
        title="QuantLab API",
        description=DESCRIPTION,
        version="0.1.0",
        openapi_tags=TAGS,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    @app.middleware("http")
    async def _request_id(request: Request, call_next: Any) -> Any:
        """Bind a request id for the whole handler and echo it to the caller.

        An inbound `X-Request-ID` is honoured so a trace can span the frontend
        and the API; anything absent or implausibly long is replaced rather than
        trusted, since it lands in every log line.
        """

        inbound = request.headers.get(REQUEST_ID_HEADER, "")
        supplied = inbound if 0 < len(inbound) <= 64 else None
        with request_context(supplied) as request_id:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    _register_error_handlers(app)

    from quantlab.api.routers import (
        backtests,
        factors,
        features,
        market,
        portfolio,
        quality,
        reports,
        research,
        risk,
        securities,
        universe,
    )

    for module in (
        securities,
        universe,
        market,
        quality,
        features,
        factors,
        backtests,
        research,
        risk,
        portfolio,
        reports,
    ):
        app.include_router(module.router)

    from quantlab.api import health as ops

    app.include_router(ops.router)

    return app


def _register_error_handlers(app: FastAPI) -> None:
    """Map domain errors to status codes once, not per route."""

    from quantlab.portfolio.covariance import CovarianceError
    from quantlab.portfolio.optimisers import OptimisationError
    from quantlab.research.deflated import DeflationError
    from quantlab.research.log import ResearchLogError
    from quantlab.research.preregister import PreRegistrationError
    from quantlab.research.reporting import ReportingRuleError

    @app.exception_handler(HoldoutViolationError)
    def _holdout(request: Request, exc: HoldoutViolationError) -> JSONResponse:
        # 403, not 400: the request is well-formed and the server understood it.
        # It is refusing on policy grounds, and the policy is the point.
        log.warning("api.holdout_refused", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": str(exc), "error": "holdout_locked"},
        )

    @app.exception_handler(DeflationError)
    def _deflation(request: Request, exc: DeflationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(exc), "error": "cannot_deflate"},
        )

    @app.exception_handler(ReportingRuleError)
    def _reporting(request: Request, exc: ReportingRuleError) -> JSONResponse:
        # 500 deliberately: a bare Sharpe reaching the serialiser is our bug, not
        # the caller's, and it must be loud rather than degraded.
        log.error("api.reporting_rule_violated", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "error": "reporting_rule"},
        )

    @app.exception_handler(OptimisationError)
    def _optimisation(request: Request, exc: OptimisationError) -> JSONResponse:
        # An infeasible constraint set is the caller's parameters, not a fault.
        # The library's message names the fix, so it is passed through intact.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(exc), "error": "infeasible_optimisation"},
        )

    @app.exception_handler(CovarianceError)
    def _covariance(request: Request, exc: CovarianceError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(exc), "error": "covariance"},
        )

    @app.exception_handler(ResearchLogError)
    def _research_log(request: Request, exc: ResearchLogError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc), "error": "research_log"},
        )

    @app.exception_handler(PreRegistrationError)
    def _prereg(request: Request, exc: PreRegistrationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(exc), "error": "pre_registration"},
        )


app = create_app()
