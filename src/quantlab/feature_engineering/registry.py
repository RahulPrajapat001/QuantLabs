"""The feature registry and its dependency DAG (PRD Module 3 / Module 4).

Features are **declared objects**, not a script of pandas calls. Each carries a
name, its dependencies, a lookback, a category and a compute function; the
registry resolves the dependencies as a directed acyclic graph and hands the
compute engine a topological order so every intermediate is computed exactly
once (the caching that makes the full build fast — see ``docs/performance.md``).

Two kinds of feature share one registry:

* ``time_series`` — computed from a single security's date-sorted OHLCV history.
  Its function takes that security's working frame (raw columns plus any
  already-computed dependency features) and returns a ``Series`` aligned to it.
* ``cross_sectional`` — computed **across securities on each date, within the
  point-in-time universe** (PRD Module 3). Its function receives the panel's rows
  for one date and returns a per-security ``Series``.

A dependency is either a **base column** (raw OHLCV, or the provided
``market_return``) or another registered feature. The distinction is what lets
the DAG stop at the leaves.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

#: Columns present before any feature is computed. Raw OHLCV plus ``market_return``
#: (the cross-sectional mean daily return, supplied by the build layer) — the
#: leaves at which dependency resolution stops.
BASE_COLUMNS: frozenset[str] = frozenset(
    {"open", "high", "low", "close", "adj_close", "volume", "market_return"}
)


class FeatureKind(StrEnum):
    """Whether a feature is a per-security series or a per-date cross-section."""

    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"


#: A time-series feature function: ``(working_frame) -> Series``.
TimeSeriesFn = Callable[[pd.DataFrame], "pd.Series[float]"]
#: A cross-sectional feature function: ``(one_date_rows) -> Series``.
CrossSectionalFn = Callable[[pd.DataFrame], "pd.Series[float]"]
FeatureFn = TimeSeriesFn | CrossSectionalFn


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A single declared feature."""

    name: str
    func: FeatureFn
    dependencies: tuple[str, ...]
    lookback: int
    category: str
    kind: FeatureKind = FeatureKind.TIME_SERIES
    #: One line describing what the feature measures, for the data dictionary
    #: and the Feature Explorer. Stored rather than read from the function's
    #: docstring because most features are closures generated in a loop
    #: (``sma_5 … sma_200``), which have no docstring to read.
    description: str = ""

    def __post_init__(self) -> None:
        if self.lookback < 0:
            raise ValueError(f"feature {self.name!r} has negative lookback")


class FeatureRegistry:
    """A collection of :class:`FeatureSpec` with DAG resolution.

    The global default registry (:data:`REGISTRY`) is populated by the
    ``@register_feature`` decorators in :mod:`quantlab.feature_engineering.definitions`.
    Tests may construct isolated registries to avoid global state.
    """

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def add(self, spec: FeatureSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"feature {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def all(self) -> list[FeatureSpec]:
        return [self._specs[n] for n in self.names()]

    def by_category(self, category: str) -> list[FeatureSpec]:
        return [s for s in self.all() if s.category == category]

    def by_kind(self, kind: FeatureKind) -> list[FeatureSpec]:
        return [s for s in self.all() if s.kind == kind]

    # -- DAG resolution ------------------------------------------------------

    def validate(self) -> None:
        """Assert every dependency is a base column or a registered feature.

        Raised eagerly at build time so a typo in a dependency name is a loud
        failure, not a silently missing column.
        """

        for spec in self._specs.values():
            for dep in spec.dependencies:
                if dep not in BASE_COLUMNS and dep not in self._specs:
                    raise ValueError(
                        f"feature {spec.name!r} depends on {dep!r}, which is "
                        "neither a base column nor a registered feature"
                    )
        # A full topological sort over everything surfaces any cycle.
        self.topological_order(self.names())

    def dependency_closure(self, names: Iterable[str]) -> set[str]:
        """All registered features needed to compute ``names`` (incl. themselves).

        Base columns are leaves and are not included — they are provided, not
        computed.
        """

        wanted: set[str] = set()
        stack = list(names)
        while stack:
            name = stack.pop()
            if name in BASE_COLUMNS or name in wanted:
                continue
            if name not in self._specs:
                raise KeyError(f"unknown feature {name!r}")
            wanted.add(name)
            stack.extend(self._specs[name].dependencies)
        return wanted

    def topological_order(self, names: Iterable[str]) -> list[str]:
        """Return the features in ``names`` (and their deps) in compute order.

        Kahn's algorithm over the dependency DAG restricted to the closure of
        ``names``. Raises on a cycle. Deterministic: ready nodes are emitted in
        sorted order so a build is reproducible.
        """

        closure = self.dependency_closure(names)
        # Edges: feature -> feature deps (ignore base-column leaves).
        deps: dict[str, set[str]] = {
            n: {d for d in self._specs[n].dependencies if d in closure} for n in closure
        }
        dependents: dict[str, set[str]] = {n: set() for n in closure}
        for node, node_deps in deps.items():
            for d in node_deps:
                dependents[d].add(node)

        ready = sorted(n for n, d in deps.items() if not d)
        order: list[str] = []
        remaining = {n: len(d) for n, d in deps.items()}
        while ready:
            node = ready.pop(0)
            order.append(node)
            for dependent in sorted(dependents[node]):
                remaining[dependent] -= 1
                if remaining[dependent] == 0:
                    # keep the ready queue sorted for determinism
                    ready.append(dependent)
                    ready.sort()
        if len(order) != len(closure):
            cyclic = sorted(set(closure) - set(order))
            raise ValueError(f"dependency cycle among features: {cyclic}")
        return order


#: Process-wide default registry populated by the definition modules.
REGISTRY = FeatureRegistry()


def _first_line(doc: str | None) -> str:
    """A decorated feature's docstring summary, if it has one."""

    return (doc or "").strip().split("\n")[0].strip()


def register_feature(
    *,
    name: str,
    dependencies: Sequence[str],
    lookback: int,
    category: str,
    description: str = "",
    kind: FeatureKind | str = FeatureKind.TIME_SERIES,
    registry: FeatureRegistry | None = None,
) -> Callable[[FeatureFn], FeatureFn]:
    """Declare a feature and register it, returning the function unchanged.

    Usage mirrors the PRD::

        @register_feature(name="rsi_14", dependencies=["close"], lookback=14,
                          category="momentum")
        def rsi_14(df: pd.DataFrame) -> pd.Series: ...

    The function is returned untouched so it stays directly callable in unit
    tests. ``registry`` defaults to the global :data:`REGISTRY`.
    """

    target = registry if registry is not None else REGISTRY
    resolved_kind = FeatureKind(kind)

    def decorator(func: FeatureFn) -> FeatureFn:
        target.add(
            FeatureSpec(
                name=name,
                func=func,
                dependencies=tuple(dependencies),
                lookback=lookback,
                category=category,
                kind=resolved_kind,
                description=description or _first_line(func.__doc__),
            )
        )
        return func

    return decorator


def add_feature(
    name: str,
    func: FeatureFn,
    *,
    dependencies: Sequence[str],
    lookback: int,
    category: str,
    description: str = "",
    kind: FeatureKind | str = FeatureKind.TIME_SERIES,
    registry: FeatureRegistry | None = None,
) -> None:
    """Imperatively register a feature (for programmatically generated families).

    The decorator form reads best for one-off features; this form reads best when
    a category is a parametrised family (``sma_5 … sma_200``) built in a loop.
    """

    (registry if registry is not None else REGISTRY).add(
        FeatureSpec(
            name=name,
            func=func,
            dependencies=tuple(dependencies),
            lookback=lookback,
            category=category,
            kind=FeatureKind(kind),
            description=description,
        )
    )


# Convenience for definition modules that want to group specs by category.
CATEGORIES: tuple[str, ...] = (
    "returns",
    "momentum",
    "moving_average",
    "volatility",
    "technical",
    "risk_adjusted",
    "cross_sectional",
    "microstructure",
)

__all__ = [
    "BASE_COLUMNS",
    "CATEGORIES",
    "REGISTRY",
    "CrossSectionalFn",
    "FeatureFn",
    "FeatureKind",
    "FeatureRegistry",
    "FeatureSpec",
    "TimeSeriesFn",
    "add_feature",
    "register_feature",
]
