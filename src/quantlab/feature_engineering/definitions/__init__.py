"""Feature definitions, grouped by category.

Importing this package registers every feature into the global
:data:`quantlab.feature_engineering.registry.REGISTRY` as a side effect (each
submodule runs its ``@register_feature`` / ``add_feature`` calls at import time).
Import order follows the DAG dependency direction — roots (returns, moving
averages) before the features that build on them — though the registry resolves
the true order itself.
"""

from __future__ import annotations

from . import (
    cross_sectional,
    microstructure,
    momentum,
    moving_averages,
    returns,
    risk_adjusted,
    technicals,
    volatility,
)

__all__ = [
    "cross_sectional",
    "microstructure",
    "momentum",
    "moving_averages",
    "returns",
    "risk_adjusted",
    "technicals",
    "volatility",
]
