"""Point-in-time universe construction (PRD Module 2).

The difference between a credible backtest and a worthless one. Public surface:

* :func:`universe_as_of` — the constituents on a date, never today's.
* :class:`UniverseMode`, and the resolvers
  (:class:`PointInTimeUniverse`, :class:`FixedUniverse`, :class:`CurrentUniverse`).
* :func:`ingest_reconstitution` and the S&P 500 / NIFTY parsers.
* :func:`measure_survivorship_bias` and :func:`write_survivorship_report`.
* :func:`liquidate_position` for delisting handling.
"""

from __future__ import annotations

from .delisting import (
    LiquidationResult,
    is_bankruptcy,
    liquidate_position,
    liquidation_proceeds,
)
from .membership import (
    CurrentUniverse,
    FixedUniverse,
    MembershipInterval,
    PointInTimeUniverse,
    UniverseMode,
    UniverseResolver,
    build_intervals_from_events,
    current_constituents,
    freeze_fixed_universe,
    insert_membership_intervals,
    membership_intervals,
    universe_as_of,
)
from .reconstitution import (
    ReconstitutionEvent,
    ResolutionReport,
    collect_nifty_events,
    ingest_reconstitution,
    parse_nifty_reconstitution,
    parse_sp500_changes,
)
from .survivorship import (
    PerformanceMetrics,
    StrategyFn,
    SurvivorshipComparison,
    compute_metrics,
    measure_survivorship_bias,
    render_survivorship_report,
    write_survivorship_report,
)

__all__ = [
    "CurrentUniverse",
    "FixedUniverse",
    "LiquidationResult",
    "MembershipInterval",
    "PerformanceMetrics",
    "PointInTimeUniverse",
    "ReconstitutionEvent",
    "ResolutionReport",
    "StrategyFn",
    "SurvivorshipComparison",
    "UniverseMode",
    "UniverseResolver",
    "build_intervals_from_events",
    "collect_nifty_events",
    "compute_metrics",
    "current_constituents",
    "freeze_fixed_universe",
    "ingest_reconstitution",
    "insert_membership_intervals",
    "is_bankruptcy",
    "liquidate_position",
    "liquidation_proceeds",
    "measure_survivorship_bias",
    "membership_intervals",
    "parse_nifty_reconstitution",
    "parse_sp500_changes",
    "render_survivorship_report",
    "universe_as_of",
    "write_survivorship_report",
]
