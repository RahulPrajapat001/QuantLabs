"""Shared response models — and the type that makes the Sharpe rule enforceable.

Two contracts live here.

**Every number travels with its rendering.** :class:`Num` carries the raw value
for charting and sorting and the display string for printing, produced by
:mod:`quantlab.api.formatting`. A frontend that formats a number itself has
introduced a second source of truth; there is nothing in the schema that requires
it to, and nothing that stops it, so the discipline is that the API always sends
both and the UI always prints ``formatted``.

**A Sharpe ratio is not a float.** PRD Module 12 requires every endpoint that
returns one to return ``{value, trials, deflated_sharpe, pbo, split, cost_model}``
instead. :class:`SharpeObject` is that shape, it has no constructor that takes a
bare number, and its only factory reads a
:class:`~quantlab.research.reporting.SharpeReport` — which itself cannot be built
without a trials count from ``research_log``. The chain from a rendered Sharpe
back to the log is therefore unbroken, and the generated TypeScript client
inherits it: ``SharpeDisplay({ value: 1.8 })`` does not typecheck because the
schema has no such shape.

Not every risk-adjusted ratio in the platform is a *selected* result, though. A
benchmark's Sharpe was not chosen from a search, and an optimiser's ex-ante ratio
is a forecast from a covariance matrix rather than a track record. Deflating
either is meaningless. Forcing them into :class:`SharpeObject` with an invented
trials count would be worse than not deflating them, so they use
:class:`ReferenceRatio` — which requires an explicit ``why_not_deflated``. There
is no way to emit a Sharpe-shaped number without either deflating it or saying in
words why deflation does not apply.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from quantlab.api import formatting as fmt
from quantlab.research.reporting import SharpeReport

__all__ = [
    "CostModelStatus",
    "Num",
    "ReferenceRatio",
    "SharpeObject",
    "SplitBadge",
]

#: The only cost model that prices an Indian equity trade in full. Anything else
#: carries a visible warning (DESIGN_SYSTEM.md §7.3), decided here rather than in
#: each frontend.
REALISTIC_COST_MODEL = "IndianEquityCostModel"

_COST_WARNINGS: dict[str, str] = {
    "ZeroCostModel": "GROSS — not a tradeable result",
    "GenericBpsModel": "flat bps — not Indian statutory costs",
    "IndianEquityCostModel(turnover-approx)": "turnover approximation",
}


class Num(BaseModel):
    """A number and the exact string to print for it.

    ``value`` is ``None`` for a genuinely missing quantity, and ``formatted`` is
    then an em dash. A frontend never has to decide what absence looks like.
    """

    model_config = ConfigDict(frozen=True)

    value: float | None = Field(description="Raw value, for charts and sorting.")
    formatted: str = Field(description="Display string. Print this, never `value`.")

    # Factories, one per formatting rule in DESIGN_SYSTEM.md §5.

    @classmethod
    def percent(cls, value: float | None, *, decimals: int = 1) -> Num:
        """A return or drawdown: signed percentage of a fraction."""

        return cls(
            value=_finite(value),
            formatted=fmt.format_percent(value, decimals=decimals),
        )

    @classmethod
    def share(cls, value: float | None, *, decimals: int = 1) -> Num:
        """A proportion of a whole: unsigned percentage."""

        return cls(
            value=_finite(value), formatted=fmt.format_share(value, decimals=decimals)
        )

    @classmethod
    def ratio(cls, value: float | None, *, decimals: int = 2) -> Num:
        return cls(
            value=_finite(value), formatted=fmt.format_ratio(value, decimals=decimals)
        )

    @classmethod
    def ic(cls, value: float | None) -> Num:
        return cls(value=_finite(value), formatted=fmt.format_ic(value))

    @classmethod
    def t_stat(cls, value: float | None) -> Num:
        return cls(value=_finite(value), formatted=fmt.format_t_stat(value))

    @classmethod
    def p_value(cls, value: float | None) -> Num:
        return cls(value=_finite(value), formatted=fmt.format_p_value(value))

    @classmethod
    def currency(cls, value: float | None, *, abbreviate: bool | None = None) -> Num:
        return cls(
            value=_finite(value),
            formatted=fmt.format_currency(value, abbreviate=abbreviate),
        )

    @classmethod
    def count(cls, value: int | float | None) -> Num:
        return cls(value=_finite(value), formatted=fmt.format_count(value))

    @classmethod
    def basis_points(cls, value: float | None) -> Num:
        return cls(value=_finite(value), formatted=fmt.format_basis_points(value))


def _finite(value: Any) -> float | None:
    """Coerce to a JSON-safe float, mapping NaN/inf to ``None``.

    JSON has no NaN. Emitting one produces invalid JSON in strict parsers and a
    literal ``NaN`` on screen in lenient ones; both are worse than ``null`` paired
    with an em dash.
    """

    import math

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class SplitBadge(BaseModel):
    """The TRAIN / VALIDATION / HOLDOUT marker, decomposed server-side.

    ``quantlab.config.resolve_split`` emits compound labels when a window spans a
    boundary (``train+validation``). The UI must render those as separate
    segments and must never collapse them to the cleaner of the two, so the
    splitting is done here — the frontend does no computation, including this.
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(description="The raw split label, e.g. 'train+validation'.")
    parts: list[str] = Field(description="Decomposed segments, in order.")
    holdout_touched: bool = Field(
        description="Whether this result reaches into the locked holdout."
    )

    @classmethod
    def from_label(cls, label: str, *, holdout_touched: bool = False) -> SplitBadge:
        parts = [part for part in label.split("+") if part]
        return cls(
            label=label,
            parts=parts,
            holdout_touched=holdout_touched or "holdout" in parts,
        )


class CostModelStatus(BaseModel):
    """Whether a result's costs are realistic, and the marker if they are not."""

    model_config = ConfigDict(frozen=True)

    name: str
    realistic: bool = Field(
        description=f"True only for exactly {REALISTIC_COST_MODEL!r}."
    )
    warning: str | None = Field(
        default=None,
        description="Marker text to display beside the result. Null when realistic.",
    )

    @classmethod
    def from_name(cls, name: str) -> CostModelStatus:
        realistic = name == REALISTIC_COST_MODEL
        warning = None
        if not realistic:
            warning = _COST_WARNINGS.get(name, f"{name} — not Indian statutory costs")
        return cls(name=name, realistic=realistic, warning=warning)


class SharpeObject(BaseModel):
    """A Sharpe ratio bound to the evidence needed to interpret it.

    The shape PRD Module 12 mandates, plus the rest of
    :class:`~quantlab.research.reporting.SharpeReport` so the UI never needs a
    second request to render the component. There is deliberately no
    ``SharpeObject(value=1.8)`` path: :meth:`from_report` is the only factory,
    and a report cannot exist without a trials count read from the log.
    """

    model_config = ConfigDict(frozen=True)

    # --- The mandated shape ---
    value: float = Field(description="Annualised Sharpe as observed.")
    trials: int = Field(description="Trials preceding it, counted from research_log.")
    deflated_sharpe: float = Field(
        description="P[true Sharpe > 0] after correcting for `trials`."
    )
    pbo: float | None = Field(
        default=None, description="Probability of backtest overfitting, when computed."
    )
    split: str = Field(description="Which split produced `value`.")
    cost_model: str = Field(description="Costs the return series was net of.")

    # --- Everything needed to render it honestly ---
    expected_max_sharpe: float = Field(
        description="Best-of-`trials` Sharpe expected from an edgeless search."
    )
    run_id: str
    n_periods: int
    significant: bool = Field(
        description="Whether the deflated Sharpe clears the 95% threshold."
    )
    provenance: str = Field(description="'exploratory' or 'pre-registered (H1)'.")
    holdout_touched: bool
    badge: SplitBadge
    costs: CostModelStatus

    # --- Display-ready ---
    value_display: Num
    deflated_display: Num
    trials_display: Num
    pbo_display: Num
    summary: str = Field(
        description="One-line rendering with trials and deflation inseparable."
    )

    @classmethod
    def from_report(cls, report: SharpeReport) -> SharpeObject:
        """The only way to build one. Mirrors ``SharpeReport`` field for field."""

        return cls(
            value=report.value,
            trials=report.trials,
            deflated_sharpe=report.deflated_sharpe,
            pbo=report.pbo,
            split=report.split,
            cost_model=report.cost_model,
            expected_max_sharpe=report.expected_max_sharpe,
            run_id=report.run_id,
            n_periods=report.n_periods,
            significant=report.significant,
            provenance=report.provenance,
            holdout_touched=report.holdout_touched,
            badge=SplitBadge.from_label(
                report.split, holdout_touched=report.holdout_touched
            ),
            costs=CostModelStatus.from_name(report.cost_model),
            value_display=Num.ratio(report.value),
            deflated_display=Num.ratio(report.deflated_sharpe, decimals=3),
            trials_display=Num.count(report.trials),
            pbo_display=Num.ratio(report.pbo),
            summary=report.render(),
        )


class ReferenceRatio(BaseModel):
    """A risk-adjusted ratio that deflation does not apply to, and why.

    Used for the benchmark's Sharpe (nobody selected the index from a search) and
    an optimiser's ex-ante ratio (a forecast from a covariance matrix, not a
    track record). ``why_not_deflated`` is required, so a Sharpe-shaped number
    cannot leave the API without either a trials count or a stated reason it has
    none.
    """

    model_config = ConfigDict(frozen=True)

    kind: str = Field(description="e.g. 'benchmark_sharpe', 'ex_ante_sharpe'.")
    basis: str = Field(description="'realised' or 'ex_ante'.")
    display: Num
    why_not_deflated: str = Field(
        min_length=1,
        description="Required. Why a trials count and deflation do not apply here.",
    )

    @classmethod
    def benchmark_sharpe(cls, value: float | None) -> ReferenceRatio:
        return cls(
            kind="benchmark_sharpe",
            basis="realised",
            display=Num.ratio(value),
            why_not_deflated=(
                "the benchmark is the index, not a configuration selected from a "
                "search, so there are no trials to deflate against"
            ),
        )

    @classmethod
    def ex_ante_sharpe(cls, value: float | None) -> ReferenceRatio:
        return cls(
            kind="ex_ante_sharpe",
            basis="ex_ante",
            display=Num.ratio(value),
            why_not_deflated=(
                "this is a forecast implied by the covariance estimate, not a "
                "realised track record; there is no return series to deflate"
            ),
        )


class DateRange(BaseModel):
    """An inclusive window, with both dates display-ready."""

    model_config = ConfigDict(frozen=True)

    start: date | None
    end: date | None
    start_display: str
    end_display: str

    @classmethod
    def of(cls, start: date | None, end: date | None) -> DateRange:
        return cls(
            start=start,
            end=end,
            start_display=fmt.format_date(start),
            end_display=fmt.format_date(end),
        )
