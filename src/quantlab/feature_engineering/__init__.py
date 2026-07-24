"""Feature engineering: a DAG registry of 70+ declared features (PRD Module 3).

Importing this package populates the global :data:`REGISTRY` with every feature
(via :mod:`quantlab.feature_engineering.definitions`) and exposes the compute
engine and the universe build orchestration.
"""

from __future__ import annotations

from . import definitions as definitions
from .engine import (
    add_market_return,
    compute_cross_sectional_features,
    compute_time_series_features,
    required_base_columns,
)
from .registry import (
    BASE_COLUMNS,
    REGISTRY,
    FeatureKind,
    FeatureRegistry,
    FeatureSpec,
    add_feature,
    register_feature,
)

__all__ = [
    "BASE_COLUMNS",
    "REGISTRY",
    "FeatureKind",
    "FeatureRegistry",
    "FeatureSpec",
    "add_feature",
    "add_market_return",
    "compute_cross_sectional_features",
    "compute_time_series_features",
    "register_feature",
    "required_base_columns",
]
