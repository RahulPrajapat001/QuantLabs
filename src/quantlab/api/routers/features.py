"""The feature registry, as a browsable catalog.

The Feature Explorer page needs what the registry knows: every declared feature,
its category, its lookback, its dependencies, and a line describing what it
measures.

The description is a **registry field**, not the compute function's docstring.
Reading docstrings was the obvious first choice and it does not work here: most
features are closures generated in a loop (``sma_5 … sma_200``, ``momentum_5d …
momentum_200d``), so there is no docstring to read and nowhere natural to put 75
of them. The registration site is where the window is in scope, so that is where
the sentence lives — see the ``description=`` arguments in
:mod:`quantlab.feature_engineering.definitions`.

This endpoint reads the registry, not the built parquet panel: it lists what
*can* be computed. ``GET /market/features`` serves the values.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from quantlab.api.models import Num
from quantlab.feature_engineering.registry import REGISTRY, FeatureSpec

router = APIRouter(prefix="/features", tags=["features"])


class FeatureOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    category: str
    #: What the feature measures, from the registry. Never empty — a feature
    #: without one reports that, rather than rendering a blank cell.
    description: str
    #: Bars of history the feature needs before its first valid value. The
    #: Feature Explorer uses this to explain the leading gap in the chart.
    lookback: Num
    dependencies: list[str]
    #: ``time_series`` (per security) or ``cross_sectional`` (across the universe
    #: on each date). The distinction matters: a cross-sectional feature cannot be
    #: computed from one security's history alone.
    kind: str


class FeatureCatalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: Num
    returned: Num
    categories: dict[str, int] = Field(
        description="Feature count per category, over the whole registry."
    )
    items: list[FeatureOut]


def _describe(spec: FeatureSpec) -> str:
    """Never blank: an undescribed feature says so rather than rendering empty."""

    if spec.description:
        return " ".join(spec.description.split())
    return f"No description registered for {spec.name!r}."


def _to_out(spec: FeatureSpec) -> FeatureOut:
    return FeatureOut(
        name=spec.name,
        category=spec.category,
        description=_describe(spec),
        lookback=Num.count(spec.lookback),
        dependencies=list(spec.dependencies),
        kind=str(getattr(spec.kind, "value", spec.kind)),
    )


@router.get("", response_model=FeatureCatalog, summary="The registered features.")
def catalog(
    search: str | None = Query(
        default=None, description="Case-insensitive match on name or description."
    ),
    category: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2_000),
) -> FeatureCatalog:
    specs = sorted(REGISTRY.all(), key=lambda s: (s.category, s.name))

    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.category] = counts.get(spec.category, 0) + 1

    if category is not None:
        specs = [s for s in specs if s.category == category]
    if search:
        needle = search.lower()
        specs = [
            s
            for s in specs
            if needle in s.name.lower() or needle in _describe(s).lower()
        ]

    return FeatureCatalog(
        total=Num.count(len(REGISTRY.all())),
        returned=Num.count(len(specs[:limit])),
        categories=counts,
        items=[_to_out(s) for s in specs[:limit]],
    )


@router.get(
    "/{name}", response_model=FeatureOut, summary="One feature's registry entry."
)
def feature(name: str) -> FeatureOut:
    try:
        spec = REGISTRY.get(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no registered feature named {name!r}",
        ) from exc
    return _to_out(spec)
