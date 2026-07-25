"""Portfolio optimisers (PRD Module 10), built on ``cvxpy``.

Six methods, from the ones that ignore the covariance matrix to the ones that invert
it. That ordering is roughly the ordering of robustness, and it is not a coincidence:

* :func:`equal_weight` — ignores every estimate. Beats mean-variance out of sample
  more often than the literature finds comfortable, because it has no parameters to
  get wrong.
* :func:`inverse_volatility` — uses only the diagonal. Variances are estimated far
  more reliably than covariances, so this captures most of the available benefit for
  a fraction of the estimation risk.
* :func:`risk_parity` — equal risk *contribution*, which needs the full matrix but
  not expected returns.
* :func:`minimum_variance` — inverts the covariance matrix. No return forecast
  required, which is why it is the workhorse: expected returns are the hardest
  input to estimate and the one optimisers are most sensitive to.
* :func:`maximum_sharpe` — needs expected returns too, and is therefore the most
  fragile. Estimation error in ``μ`` translates into extreme weights.
* :func:`efficient_frontier` — the whole risk/return locus.

**Every result reports its solver status.** A cvxpy problem that ends
``infeasible`` or ``unbounded`` has no weights, and this module raises rather than
returning whatever was in the variable when the solver gave up. Silent optimiser
failure producing plausible-looking weights is a genuine hazard.

Constraints available on all of them: weight bounds, long-only or long-short, and an
L1 turnover penalty against a previous portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog

from quantlab.portfolio.covariance import (
    ILL_CONDITIONED,
    TRADING_DAYS,
    CovarianceError,
    CovarianceEstimate,
)

log = structlog.get_logger(__name__)

#: Solver statuses cvxpy considers successful.
_OK_STATUSES = frozenset({"optimal", "optimal_inaccurate"})


class OptimisationError(ValueError):
    """Raised when an optimisation cannot be solved, or its result is unusable."""


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    """Constraints applied to every optimiser.

    ``long_only=True`` with ``min_weight=0.0`` is the realistic default for Indian
    cash-segment equities, where a delivery strategy cannot short.
    """

    long_only: bool = True
    #: Minimum weight per asset. Ignored when ``long_only`` forces it to 0.
    min_weight: float = 0.0
    #: Maximum weight per asset — the concentration limit that stops an optimiser
    #: putting 90% into whichever asset has the most flattering estimate.
    max_weight: float = 0.10
    #: Weights sum to this. 1.0 is fully invested.
    budget: float = 1.0
    #: Coefficient on the L1 turnover penalty ``λ‖w - w_prev‖₁``. Real rebalancing
    #: costs money, and an optimiser with no turnover term will happily churn the
    #: book for a basis point of theoretical variance reduction.
    turnover_penalty: float = 0.0
    #: Hard cap on total turnover, if any.
    max_turnover: float | None = None

    def __post_init__(self) -> None:
        if self.max_weight <= 0:
            raise OptimisationError(
                f"max_weight must be positive, got {self.max_weight}"
            )
        if self.long_only and self.min_weight < 0:
            raise OptimisationError(
                f"min_weight={self.min_weight} contradicts long_only=True"
            )
        if not self.long_only and self.min_weight >= 0:
            log.debug(
                "constraints.long_short_with_nonnegative_floor",
                detail="long_short is enabled but min_weight >= 0 forbids shorts",
            )
        if self.turnover_penalty < 0:
            raise OptimisationError("turnover_penalty must be non-negative")
        if self.max_turnover is not None and self.max_turnover < 0:
            raise OptimisationError("max_turnover must be non-negative")

    def effective_min(self) -> float:
        return max(0.0, self.min_weight) if self.long_only else self.min_weight

    def feasible_for(self, n_assets: int) -> None:
        """Raise when the bounds cannot possibly sum to the budget."""

        lower = self.effective_min() * n_assets
        upper = self.max_weight * n_assets
        if upper < self.budget - 1e-9:
            raise OptimisationError(
                f"max_weight={self.max_weight} across {n_assets} assets caps total "
                f"weight at {upper:.4f}, below the budget of {self.budget}. Raise "
                f"max_weight above {self.budget / n_assets:.4f} or add assets."
            )
        if lower > self.budget + 1e-9:
            raise OptimisationError(
                f"min_weight={self.min_weight} across {n_assets} assets forces at "
                f"least {lower:.4f}, above the budget of {self.budget}"
            )


@dataclass(frozen=True, slots=True)
class OptimisationResult:
    """Weights, plus everything needed to judge whether to trust them."""

    weights: pd.Series
    method: str
    #: cvxpy solver status, or ``"analytic"`` for closed-form methods.
    status: str
    expected_return: float | None
    volatility: float
    sharpe: float | None
    #: One-way turnover against the previous portfolio, if one was supplied.
    turnover: float | None
    n_assets: int
    #: Condition number of the covariance matrix used, so an ill-conditioned input
    #: is visible in the output rather than only in the input's diagnostics.
    condition_number: float | None = None
    shrinkage: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_positions(self) -> float:
        """``1/Σw²`` — how many positions this portfolio really holds.

        Equal weight across ``n`` assets gives ``n``. A portfolio with 50 names and
        an effective count of 4 is a concentrated bet wearing a diversified label.
        """

        squared = float((self.weights**2).sum())
        return 1.0 / squared if squared > 0 else float("nan")

    @property
    def gross_exposure(self) -> float:
        return float(self.weights.abs().sum())

    @property
    def net_exposure(self) -> float:
        return float(self.weights.sum())

    def top_holdings(self, n: int = 5) -> pd.Series:
        return self.weights.reindex(
            self.weights.abs().sort_values(ascending=False).index
        ).head(n)

    def render(self) -> str:
        lines = [
            f"{self.method} ({self.status}, {self.n_assets} assets)",
            f"  volatility:            {self.volatility:8.2%} annualised",
        ]
        if self.expected_return is not None:
            lines.append(f"  expected return:       {self.expected_return:8.2%}")
        if self.sharpe is not None:
            lines.append(f"  expected Sharpe:       {self.sharpe:8.3f}  (ex ante)")
        lines += [
            f"  gross / net exposure:  {self.gross_exposure:8.2f} / "
            f"{self.net_exposure:.2f}",
            f"  effective positions:   {self.effective_positions:8.2f}",
        ]
        if self.turnover is not None:
            lines.append(f"  turnover:              {self.turnover:8.2%}")
        if self.shrinkage is not None:
            lines.append(f"  covariance shrinkage:  {self.shrinkage:8.4f}")
        if self.condition_number is not None:
            lines.append(f"  condition number:      {self.condition_number:8.2e}")
            if self.condition_number > ILL_CONDITIONED:
                lines.append(
                    "  WARNING: the covariance matrix is ill-conditioned. These "
                    "weights are dominated by estimation error."
                )
        lines.append("")
        lines.append("  Top holdings:")
        for asset, weight in self.top_holdings().items():
            lines.append(f"    {asset:<24s} {weight:7.2%}")
        if self.sharpe is not None:
            lines += [
                "",
                "  The Sharpe above is an EX ANTE estimate from the covariance and "
                "return inputs, not a realised result. It is not deflatable and must "
                "not be reported as performance.",
            ]
        return "\n".join(lines)


# --- Helpers -----------------------------------------------------------------


def _as_matrix(
    covariance: pd.DataFrame | CovarianceEstimate,
) -> tuple[pd.DataFrame, float | None, float | None]:
    """Return ``(matrix, condition number, shrinkage)`` from either input type."""

    if isinstance(covariance, CovarianceEstimate):
        return covariance.matrix, covariance.condition_number, covariance.shrinkage
    matrix = covariance
    eigenvalues = np.linalg.eigvalsh(matrix.to_numpy(dtype=float))
    smallest = float(eigenvalues.min())
    condition = float(eigenvalues.max()) / smallest if smallest > 0 else float("inf")
    return matrix, condition, None


def _validate(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise OptimisationError(f"covariance must be square, got shape {values.shape}")
    # Finiteness before symmetry: NaN != NaN, so a matrix containing NaN would
    # otherwise be reported as asymmetric, which sends the reader after the wrong bug.
    if not np.isfinite(values).all():
        raise OptimisationError("covariance matrix contains non-finite values")
    if not np.allclose(values, values.T, atol=1e-10):
        raise OptimisationError("covariance matrix is not symmetric")
    # Symmetrise exactly: cvxpy's PSD handling is sensitive to 1e-17 asymmetry.
    symmetric: npt.NDArray[np.float64] = (values + values.T) / 2.0
    return symmetric


def _project_to_bounds(
    weights: np.ndarray, constraints: PortfolioConstraints
) -> np.ndarray:
    """Clip to the weight bounds and restore the budget.

    Interior-point solvers satisfy constraints to a tolerance, not exactly, so a
    ``max_weight`` of 0.15 can come back as 0.1500184. That is numerically fine and
    contractually wrong: a hard bound that the returned weights violate is not a
    bound. The excess is clipped and redistributed across the assets that still have
    headroom, which moves the objective by the same negligible amount the tolerance
    already represents.
    """

    lower = (
        constraints.effective_min() if constraints.long_only else constraints.min_weight
    )
    upper = constraints.max_weight
    projected = np.clip(weights, lower, upper)

    for _ in range(8):
        gap = constraints.budget - projected.sum()
        if abs(gap) < 1e-12:
            break
        # Only assets away from the binding side can absorb the adjustment.
        room = (upper - projected) if gap > 0 else (projected - lower)
        capacity = float(room.sum())
        if capacity <= 1e-15:
            break
        projected = np.clip(projected + gap * room / capacity, lower, upper)
    result: npt.NDArray[np.float64] = projected
    return result


def _psd(values: np.ndarray) -> np.ndarray:
    """Project onto the nearest PSD matrix if rounding has made it indefinite."""

    eigenvalues, eigenvectors = np.linalg.eigh(values)
    if eigenvalues.min() >= 0:
        return values
    clipped = np.clip(eigenvalues, 0.0, None)
    projected: npt.NDArray[np.float64] = (
        eigenvectors @ np.diag(clipped) @ eigenvectors.T
    )
    return projected


def _build_constraints(
    w: cp.Variable,
    constraints: PortfolioConstraints,
    previous: np.ndarray | None,
) -> list[cp.Constraint]:
    built: list[cp.Constraint] = [cp.sum(w) == constraints.budget]
    if constraints.long_only:
        built.append(w >= max(0.0, constraints.min_weight))
    else:
        built.append(w >= constraints.min_weight)
    built.append(w <= constraints.max_weight)
    if constraints.max_turnover is not None and previous is not None:
        built.append(cp.norm1(w - previous) <= constraints.max_turnover)
    return built


def _solve(problem: cp.Problem, method: str) -> str:
    """Solve, and raise on any status that is not a usable solution."""

    try:
        problem.solve()
    except cp.error.SolverError as exc:
        raise OptimisationError(
            f"{method}: the solver failed ({exc}). Check the covariance matrix is "
            "positive semi-definite and the constraints are mutually satisfiable."
        ) from exc

    status = str(problem.status)
    if status not in _OK_STATUSES:
        raise OptimisationError(
            f"{method}: optimisation ended '{status}'. There are no weights to "
            "return — an infeasible problem does not have an approximate answer. "
            "Check that max_weight × n_assets ≥ budget and that any turnover cap is "
            "reachable."
        )
    if status == "optimal_inaccurate":
        log.warning("optimiser.inaccurate", method=method)
    return status


def _finalise(
    weights: np.ndarray,
    assets: pd.Index,
    matrix: np.ndarray,
    *,
    method: str,
    status: str,
    expected_returns: np.ndarray | None,
    previous: np.ndarray | None,
    condition_number: float | None,
    shrinkage: float | None,
    periods_per_year: int,
    diagnostics: dict[str, Any] | None = None,
) -> OptimisationResult:
    # Clean numerical dust so weights sum to the budget and tiny negatives from the
    # solver do not read as short positions.
    cleaned = np.where(np.abs(weights) < 1e-9, 0.0, weights)
    variance = float(cleaned @ matrix @ cleaned)
    volatility = float(np.sqrt(max(0.0, variance) * periods_per_year))

    expected = None
    sharpe = None
    if expected_returns is not None:
        expected = float(cleaned @ expected_returns) * periods_per_year
        sharpe = expected / volatility if volatility > 0 else None

    turnover = None
    if previous is not None:
        turnover = float(np.abs(cleaned - previous).sum())

    return OptimisationResult(
        weights=pd.Series(cleaned, index=assets, name="weight"),
        method=method,
        status=status,
        expected_return=expected,
        volatility=volatility,
        sharpe=sharpe,
        turnover=turnover,
        n_assets=len(assets),
        condition_number=condition_number,
        shrinkage=shrinkage,
        diagnostics=diagnostics or {},
    )


def _previous_vector(
    previous_weights: dict[str, float] | pd.Series | None, assets: pd.Index
) -> np.ndarray | None:
    if previous_weights is None:
        return None
    series = pd.Series(previous_weights, dtype=float).reindex(assets).fillna(0.0)
    return series.to_numpy(dtype=float)


# --- The six methods ---------------------------------------------------------


def equal_weight(
    assets: list[str] | pd.Index,
    *,
    covariance: pd.DataFrame | CovarianceEstimate | None = None,
    previous_weights: dict[str, float] | pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> OptimisationResult:
    """Equal weight — the benchmark every optimiser has to beat.

    It uses no estimates at all, which is exactly why it is hard to beat: DeMiguel,
    Garlappi & Uppal (2009) found that mean-variance optimisation failed to beat it
    out of sample across a range of datasets, because estimation error costs more than
    optimisation gains. Any optimiser here that cannot beat this one is not earning
    its complexity.
    """

    index = pd.Index(assets)
    if len(index) == 0:
        raise OptimisationError("no assets supplied")
    weights = np.full(len(index), 1.0 / len(index))

    matrix = np.zeros((len(index), len(index)))
    condition = shrinkage = None
    if covariance is not None:
        frame, condition, shrinkage = _as_matrix(covariance)
        frame = frame.reindex(index=index, columns=index)
        matrix = _validate(frame)

    return _finalise(
        weights,
        index,
        matrix,
        method="equal_weight",
        status="analytic",
        expected_returns=None,
        previous=_previous_vector(previous_weights, index),
        condition_number=condition,
        shrinkage=shrinkage,
        periods_per_year=periods_per_year,
    )


def inverse_volatility(
    covariance: pd.DataFrame | CovarianceEstimate,
    *,
    constraints: PortfolioConstraints | None = None,
    previous_weights: dict[str, float] | pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> OptimisationResult:
    """Weights proportional to ``1/σ``, using only the covariance diagonal.

    Ignores correlations entirely. That sounds like a weakness and is often a
    strength: variances are estimated far more accurately than covariances, so this
    captures much of the risk-balancing benefit without inheriting the estimation
    error that makes full optimisation fragile.

    Bounds are applied by clipping and renormalising rather than by solving, so a
    ``max_weight`` cap can leave the budget slightly under-allocated when many assets
    bind. The diagnostics record it when that happens.
    """

    limits = constraints or PortfolioConstraints()
    frame, condition, shrinkage = _as_matrix(covariance)
    matrix = _validate(frame)
    variances = np.diag(matrix)
    if np.any(variances <= 0):
        raise OptimisationError(
            "an asset has zero or negative variance; inverse-volatility weighting "
            "is undefined for it"
        )

    raw = 1.0 / np.sqrt(variances)
    weights = raw / raw.sum() * limits.budget
    clipped = np.clip(weights, limits.effective_min(), limits.max_weight)
    total = clipped.sum()
    diagnostics: dict[str, Any] = {}
    if total > 0:
        rescaled = clipped / total * limits.budget
        # Re-clip once: rescaling can push a previously-compliant weight over the cap.
        weights = np.clip(rescaled, limits.effective_min(), limits.max_weight)
        if abs(weights.sum() - limits.budget) > 1e-6:
            diagnostics["unallocated"] = float(limits.budget - weights.sum())

    return _finalise(
        weights,
        frame.index,
        matrix,
        method="inverse_volatility",
        status="analytic",
        expected_returns=None,
        previous=_previous_vector(previous_weights, frame.index),
        condition_number=condition,
        shrinkage=shrinkage,
        periods_per_year=periods_per_year,
        diagnostics=diagnostics,
    )


def minimum_variance(
    covariance: pd.DataFrame | CovarianceEstimate,
    *,
    constraints: PortfolioConstraints | None = None,
    previous_weights: dict[str, float] | pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> OptimisationResult:
    """Minimise ``w'Σw`` subject to the constraints.

    Needs no return forecast, which is its great advantage: expected returns are the
    least reliable input to any optimiser and the one whose errors do the most damage.

    With no binding bounds the solution is the closed form
    ``w = Σ⁻¹1 / (1'Σ⁻¹1)``, which the tests verify against directly for the
    two-asset case.
    """

    limits = constraints or PortfolioConstraints()
    frame, condition, shrinkage = _as_matrix(covariance)
    matrix = _psd(_validate(frame))
    n = matrix.shape[0]
    limits.feasible_for(n)

    previous = _previous_vector(previous_weights, frame.index)
    w = cp.Variable(n)
    objective = cp.quad_form(w, cp.psd_wrap(matrix))
    if limits.turnover_penalty > 0 and previous is not None:
        objective = objective + limits.turnover_penalty * cp.norm1(w - previous)

    problem = cp.Problem(
        cp.Minimize(objective), _build_constraints(w, limits, previous)
    )
    status = _solve(problem, "minimum_variance")

    return _finalise(
        _project_to_bounds(np.asarray(w.value, dtype=float).ravel(), limits),
        frame.index,
        matrix,
        method="minimum_variance",
        status=status,
        expected_returns=None,
        previous=previous,
        condition_number=condition,
        shrinkage=shrinkage,
        periods_per_year=periods_per_year,
    )


def risk_parity(
    covariance: pd.DataFrame | CovarianceEstimate,
    *,
    risk_budget: dict[str, float] | pd.Series | None = None,
    previous_weights: dict[str, float] | pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
    max_weight: float | None = None,
) -> OptimisationResult:
    """Equal risk contribution: each asset contributes the same share of variance.

    Solved through Spinu's convex formulation — minimise
    ``½w'Σw - Σbᵢ·log(wᵢ)`` over ``w > 0``, then rescale to the budget. The log
    barrier makes the problem convex and forces strictly positive weights, so risk
    parity is **long-only by construction**; a short position has no meaningful risk
    contribution to equalise.

    ``risk_budget`` allows unequal targets (a risk-budgeted rather than risk-parity
    portfolio); it defaults to equal.
    """

    frame, condition, shrinkage = _as_matrix(covariance)
    matrix = _psd(_validate(frame))
    n = matrix.shape[0]

    if risk_budget is None:
        budget = np.full(n, 1.0 / n)
    else:
        series = pd.Series(risk_budget, dtype=float).reindex(frame.index)
        if series.isna().any():
            missing = series[series.isna()].index.tolist()
            raise OptimisationError(f"no risk budget supplied for {missing}")
        if (series <= 0).any():
            raise OptimisationError("risk budgets must be strictly positive")
        budget = (series / series.sum()).to_numpy(dtype=float)

    w = cp.Variable(n, pos=True)
    objective = 0.5 * cp.quad_form(w, cp.psd_wrap(matrix)) - budget @ cp.log(w)
    problem = cp.Problem(cp.Minimize(objective))
    status = _solve(problem, "risk_parity")

    raw = np.asarray(w.value, dtype=float).ravel()
    if raw.sum() <= 0:
        raise OptimisationError("risk parity produced non-positive weights")
    weights = raw / raw.sum()
    if max_weight is not None:
        weights = np.clip(weights, 0.0, max_weight)
        weights = weights / weights.sum()

    contributions = weights * (matrix @ weights)
    total_variance = float(contributions.sum())
    diagnostics = {
        "risk_contributions": (
            (contributions / total_variance).tolist() if total_variance > 0 else []
        ),
        "max_contribution_gap": (
            float(
                np.max(contributions / total_variance)
                - np.min(contributions / total_variance)
            )
            if total_variance > 0
            else float("nan")
        ),
    }
    return _finalise(
        weights,
        frame.index,
        matrix,
        method="risk_parity",
        status=status,
        expected_returns=None,
        previous=_previous_vector(previous_weights, frame.index),
        condition_number=condition,
        shrinkage=shrinkage,
        periods_per_year=periods_per_year,
        diagnostics=diagnostics,
    )


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame | CovarianceEstimate,
    *,
    constraints: PortfolioConstraints | None = None,
    n_points: int = 40,
    previous_weights: dict[str, float] | pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """The risk/return locus: minimum variance at each of ``n_points`` target returns.

    ``expected_returns`` is per period (daily), consistent with the covariance matrix.
    Targets that cannot be reached under the constraints are omitted from the frame
    rather than reported with a fallback portfolio.
    """

    limits = constraints or PortfolioConstraints()
    frame, condition, shrinkage = _as_matrix(covariance)
    aligned_index = frame.index
    mu = expected_returns.reindex(aligned_index)
    if mu.isna().any():
        missing = mu[mu.isna()].index.tolist()
        raise OptimisationError(f"no expected return supplied for {missing}")

    matrix = _psd(_validate(frame))
    n = matrix.shape[0]
    limits.feasible_for(n)
    mu_values = mu.to_numpy(dtype=float)
    previous = _previous_vector(previous_weights, aligned_index)

    # The reachable return range under the bounds, not the raw min/max of mu — a
    # max_weight cap means the highest-return asset cannot be held outright.
    low = minimum_variance(frame, constraints=limits, periods_per_year=periods_per_year)
    lowest = float(low.weights.to_numpy() @ mu_values)

    w = cp.Variable(n)
    target = cp.Parameter()
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(matrix))),
        [*_build_constraints(w, limits, previous), mu_values @ w >= target],
    )

    # Upper end: maximise return under the same constraints.
    max_problem = cp.Problem(
        cp.Maximize(mu_values @ w), _build_constraints(w, limits, previous)
    )
    _solve(max_problem, "efficient_frontier (max return)")
    highest = float(max_problem.value)

    rows: list[dict[str, Any]] = []
    for level in np.linspace(lowest, highest, n_points):
        target.value = float(level)
        try:
            status = _solve(problem, "efficient_frontier")
        except OptimisationError:
            continue
        weights = _project_to_bounds(np.asarray(w.value, dtype=float).ravel(), limits)
        result = _finalise(
            weights,
            aligned_index,
            matrix,
            method="efficient_frontier",
            status=status,
            expected_returns=mu_values,
            previous=previous,
            condition_number=condition,
            shrinkage=shrinkage,
            periods_per_year=periods_per_year,
        )
        rows.append(
            {
                "target_return": float(level) * periods_per_year,
                "expected_return": result.expected_return,
                "volatility": result.volatility,
                "sharpe": result.sharpe,
                "effective_positions": result.effective_positions,
                "weights": result.weights,
            }
        )

    if not rows:
        raise OptimisationError(
            "no point on the efficient frontier was solvable under these constraints"
        )
    return pd.DataFrame(rows)


def maximum_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame | CovarianceEstimate,
    *,
    constraints: PortfolioConstraints | None = None,
    risk_free_rate: float = 0.0,
    n_points: int = 80,
    previous_weights: dict[str, float] | pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> OptimisationResult:
    """The highest-Sharpe portfolio reachable under the constraints.

    Found by scanning the efficient frontier rather than by the usual
    ``min w'Σw s.t. μ'w = 1`` rotation. That rotation relies on the problem being
    scale-invariant, which **weight bounds break**: rescaling the solution back to
    sum 1 can violate ``max_weight``, so the trick silently returns a portfolio that
    does not satisfy the constraints it was given. Scanning the frontier is slower and
    correct under any convex constraint set.

    The reported Sharpe is **ex ante** — computed from the inputs, not realised. It
    is not a backtest result and must never be presented as one.
    """

    frontier = efficient_frontier(
        expected_returns,
        covariance,
        constraints=constraints,
        n_points=n_points,
        previous_weights=previous_weights,
        periods_per_year=periods_per_year,
    )
    annual_rf = risk_free_rate
    frontier = frontier.assign(
        excess_sharpe=lambda f: (f["expected_return"] - annual_rf) / f["volatility"]
    )
    best = frontier.loc[frontier["excess_sharpe"].idxmax()]

    frame, condition, shrinkage = _as_matrix(covariance)
    matrix = _psd(_validate(frame))
    weights: pd.Series = best["weights"]
    return _finalise(
        weights.to_numpy(dtype=float),
        frame.index,
        matrix,
        method="maximum_sharpe",
        status="optimal",
        expected_returns=expected_returns.reindex(frame.index).to_numpy(dtype=float),
        previous=_previous_vector(previous_weights, frame.index),
        condition_number=condition,
        shrinkage=shrinkage,
        periods_per_year=periods_per_year,
        diagnostics={
            "frontier_points": len(frontier),
            "risk_free_rate": risk_free_rate,
        },
    )


def analytic_minimum_variance(covariance: pd.DataFrame) -> pd.Series:
    """Unconstrained minimum-variance weights in closed form: ``Σ⁻¹1 / 1'Σ⁻¹1``.

    Provided for verification, not for use — it permits short positions and
    arbitrarily large weights. The tests check that the constrained solver reproduces
    this when no bound binds, which is how we know the solver is set up correctly.
    """

    matrix = _validate(covariance)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise CovarianceError(
            "covariance matrix is singular and cannot be inverted — this is exactly "
            "the case Ledoit-Wolf shrinkage exists to fix"
        ) from exc
    ones = np.ones(matrix.shape[0])
    weights = inverse @ ones
    return pd.Series(weights / (ones @ weights), index=covariance.index, name="weight")
