"""Registry: registration, the 70+ target, DAG order and cycle detection."""

from __future__ import annotations

import pytest

from quantlab.feature_engineering import (
    REGISTRY,
    FeatureKind,
    FeatureRegistry,
    FeatureSpec,
    register_feature,
)
from quantlab.feature_engineering.registry import BASE_COLUMNS


def test_registry_meets_70_feature_target() -> None:
    assert len(REGISTRY) >= 70


def test_all_categories_present() -> None:
    categories = {s.category for s in REGISTRY.all()}
    expected = {
        "returns",
        "momentum",
        "moving_average",
        "volatility",
        "technical",
        "risk_adjusted",
        "cross_sectional",
        "microstructure",
    }
    assert expected <= categories


def test_registry_validates() -> None:
    # Every dependency is a base column or a registered feature; no cycles.
    REGISTRY.validate()


def test_every_dependency_resolves() -> None:
    for spec in REGISTRY.all():
        for dep in spec.dependencies:
            assert dep in BASE_COLUMNS or dep in REGISTRY, (
                f"{spec.name} depends on unknown {dep!r}"
            )


def test_topological_order_respects_dependencies() -> None:
    order = REGISTRY.topological_order(["macd_hist"])
    # A dependency must appear before any feature that needs it.
    assert order.index("ema_12") < order.index("macd_line")
    assert order.index("ema_26") < order.index("macd_line")
    assert order.index("macd_line") < order.index("macd_signal")
    assert order.index("macd_signal") < order.index("macd_hist")


def test_dependency_closure_excludes_base_columns() -> None:
    closure = REGISTRY.dependency_closure(["rsi_14"])
    assert "rsi_14" in closure
    assert "close" not in closure  # base columns are leaves, not computed


def test_cross_sectional_features_registered() -> None:
    xs = REGISTRY.by_kind(FeatureKind.CROSS_SECTIONAL)
    names = {s.name for s in xs}
    assert "xs_zscore_momentum_20d" in names
    assert "xs_rank_momentum_20d" in names


def test_duplicate_registration_raises() -> None:
    reg = FeatureRegistry()
    reg.add(FeatureSpec("f", lambda df: df["close"], ("close",), 1, "returns"))
    with pytest.raises(ValueError, match="already registered"):
        reg.add(FeatureSpec("f", lambda df: df["close"], ("close",), 1, "returns"))


def test_cycle_is_detected() -> None:
    reg = FeatureRegistry()
    reg.add(FeatureSpec("a", lambda df: df["b"], ("b",), 1, "x"))
    reg.add(FeatureSpec("b", lambda df: df["a"], ("a",), 1, "x"))
    with pytest.raises(ValueError, match="cycle"):
        reg.topological_order(["a"])


def test_register_feature_decorator_returns_callable() -> None:
    reg = FeatureRegistry()

    @register_feature(
        name="doubled", dependencies=["close"], lookback=1, category="x", registry=reg
    )
    def doubled(df):  # type: ignore[no-untyped-def]
        return df["close"] * 2

    assert "doubled" in reg
    # The function is returned untouched, so it stays directly callable.
    import pandas as pd

    out = doubled(pd.DataFrame({"close": [1.0, 2.0]}))
    assert list(out) == [2.0, 4.0]
