"""Performance benchmark for the feature build (PRD Module 3 deliverable).

Measures a full build across a synthetic ``500 securities x 10 years`` panel,
profiles it, and quantifies the three optimisations the DAG design enables:

1. **Shared-intermediate caching.** The naive baseline computes every feature
   from scratch — an EMA needed by five features is computed five times. The DAG
   build computes the closure once and reuses columns. Compared single-threaded.
2. **Vectorisation.** Rolling operations use pandas ``ewm``/``rolling`` (C-level)
   rather than Python loops. A micro-benchmark contrasts a hand-rolled EMA loop
   with the vectorised ``ewm``.
3. **Per-ticker parallelism.** Securities are independent, so ``joblib`` fans the
   per-security compute across cores. Compared serial vs ``n_jobs=-1``.

``run_benchmark`` returns the measured timings; ``write_performance_doc`` renders
them plus the profile into ``docs/performance.md``.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .build import build_feature_panel
from .engine import add_market_return, compute_time_series_features
from .registry import REGISTRY, FeatureKind, FeatureRegistry


def make_synthetic_panel(
    n_securities: int, n_days: int, *, seed: int = 7
) -> pd.DataFrame:
    """A long OHLCV panel of ``n_securities`` geometric-random-walk price series."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    frames = []
    for i in range(n_securities):
        rets = rng.normal(0.0003, 0.02, n_days)
        close = 100.0 * np.cumprod(1.0 + rets)
        intraday = np.abs(rng.normal(0.0, 0.01, n_days))
        frames.append(
            pd.DataFrame(
                {
                    "security_id": f"SEC-{i:04d}",
                    "date": dates,
                    "open": close * (1.0 + rng.normal(0.0, 0.005, n_days)),
                    "high": close * (1.0 + intraday),
                    "low": close * (1.0 - intraday),
                    "close": close,
                    "adj_close": close,
                    "volume": rng.integers(10_000, 1_000_000, n_days).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    n_securities: int
    n_days: int
    n_features: int
    naive_seconds: float
    cached_seconds: float
    serial_seconds: float
    parallel_seconds: float
    naive_scale_securities: int
    ema_loop_seconds: float
    ema_vectorised_seconds: float

    @property
    def caching_speedup(self) -> float:
        return self.naive_seconds / self.cached_seconds if self.cached_seconds else 0.0

    @property
    def parallel_speedup(self) -> float:
        return (
            self.serial_seconds / self.parallel_seconds
            if self.parallel_seconds
            else 0.0
        )

    @property
    def vectorisation_speedup(self) -> float:
        return (
            self.ema_loop_seconds / self.ema_vectorised_seconds
            if self.ema_vectorised_seconds
            else 0.0
        )


def _compute_naive(
    panel: pd.DataFrame, ts_names: list[str], registry: FeatureRegistry
) -> None:
    """Baseline: compute each feature in isolation, per security (no reuse)."""

    for _sid, frame in panel.groupby("security_id", sort=False):
        ordered = frame.sort_values("date").reset_index(drop=True)
        for name in ts_names:
            # One engine call per feature recomputes its whole dependency closure,
            # so shared intermediates (EMAs, returns) are recomputed repeatedly.
            compute_time_series_features(ordered, [name], registry=registry)


def _compute_cached(
    panel: pd.DataFrame, ts_names: list[str], registry: FeatureRegistry
) -> None:
    """DAG build: compute the same features in one topological pass per security.

    Apples-to-apples with :func:`_compute_naive` — same features, same securities,
    the only difference being that shared intermediates are computed once.
    """

    for _sid, frame in panel.groupby("security_id", sort=False):
        ordered = frame.sort_values("date").reset_index(drop=True)
        compute_time_series_features(ordered, ts_names, registry=registry)


def _ema_loop(series: np.ndarray, span: int) -> np.ndarray:
    """A hand-rolled Python EMA loop — the un-vectorised baseline."""

    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def run_benchmark(
    *,
    n_securities: int = 500,
    n_days: int = 2520,
    naive_scale_securities: int = 25,
    n_jobs: int = -1,
    registry: FeatureRegistry = REGISTRY,
) -> BenchmarkResult:
    """Run the three timed comparisons and return the measured result.

    The caching comparison runs on ``naive_scale_securities`` securities (the
    naive baseline is quadratic in shared-dependency count and would dominate the
    wall clock at full scale); the parallelism comparison runs at the full
    ``n_securities``.
    """

    registry.validate()
    ts_names = [s.name for s in registry.by_kind(FeatureKind.TIME_SERIES)]
    all_names = registry.names()

    full_panel = make_synthetic_panel(n_securities, n_days)
    small_panel = full_panel[
        full_panel["security_id"] < f"SEC-{naive_scale_securities:04d}"
    ].copy()

    # 1) shared-intermediate caching (single-threaded, small scale). The naive
    # path skips build_feature_panel, so attach market_return here (beta/alpha
    # depend on it); the cached builds add it themselves inside the build.
    small_with_market = add_market_return(small_panel)
    t0 = time.perf_counter()
    _compute_naive(small_with_market, ts_names, registry)
    naive_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    _compute_cached(small_with_market, ts_names, registry)
    cached_seconds = time.perf_counter() - t0

    # 2) serial vs parallel (full scale)
    t0 = time.perf_counter()
    build_feature_panel(full_panel, all_names, registry=registry, n_jobs=1)
    serial_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    build_feature_panel(full_panel, all_names, registry=registry, n_jobs=n_jobs)
    parallel_seconds = time.perf_counter() - t0

    # 3) vectorisation micro-benchmark
    series = full_panel["close"].to_numpy(dtype=float)[: n_days * 4]
    t0 = time.perf_counter()
    _ema_loop(series, 26)
    ema_loop_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    pd.Series(series).ewm(span=26, adjust=False).mean()
    ema_vectorised_seconds = time.perf_counter() - t0

    return BenchmarkResult(
        n_securities=n_securities,
        n_days=n_days,
        n_features=len(all_names),
        naive_seconds=naive_seconds,
        cached_seconds=cached_seconds,
        serial_seconds=serial_seconds,
        parallel_seconds=parallel_seconds,
        naive_scale_securities=naive_scale_securities,
        ema_loop_seconds=ema_loop_seconds,
        ema_vectorised_seconds=ema_vectorised_seconds,
    )


def profile_build(*, n_securities: int = 100, n_days: int = 2520, top: int = 15) -> str:
    """Profile a single cached build and return the top-``top`` cumulative funcs."""

    panel = make_synthetic_panel(n_securities, n_days)
    names = REGISTRY.names()
    profiler = cProfile.Profile()
    profiler.enable()
    build_feature_panel(panel, names, n_jobs=1)
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
    stats.print_stats(top)
    return buffer.getvalue()


def render_performance_doc(result: BenchmarkResult, profile_text: str) -> str:
    """Render the benchmark result + profile as ``docs/performance.md``."""

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    workload = (
        f"{result.n_securities} securities x {result.n_days} trading days "
        f"({result.n_features} features)"
    )
    return f"""# Feature build performance

_Generated {generated} by `quantlab.feature_engineering.benchmark`._

Target workload: **{workload}**.

## Optimisation 1 — cache shared intermediates in the DAG

Measured single-threaded on {result.naive_scale_securities} securities. The naive
baseline computes each feature (and its full dependency closure) in isolation, so
shared intermediates such as `ema_12`, `ema_26`, `return_1d` and `log_return_1d`
are recomputed once per dependent feature. The DAG build computes each node once.

| Build | Seconds |
|---|---:|
| Naive (per-feature recompute) | {result.naive_seconds:.2f} |
| DAG-cached | {result.cached_seconds:.2f} |
| **Speed-up** | **{result.caching_speedup:.1f}x** |

## Optimisation 2 — vectorise rolling operations

Indicators use pandas `ewm`/`rolling` (C-level) instead of Python loops. Measured
on a single long series (EMA span 26).

| EMA implementation | Seconds |
|---|---:|
| Hand-rolled Python loop | {result.ema_loop_seconds:.4f} |
| Vectorised `ewm` | {result.ema_vectorised_seconds:.4f} |
| **Speed-up** | **{result.vectorisation_speedup:.1f}x** |

## Optimisation 3 — parallelise per-ticker work with joblib

Securities are independent, so the per-security time-series compute fans out
across cores. Measured at the full {result.n_securities}-security scale.

| Build | Seconds |
|---|---:|
| Serial (`n_jobs=1`) | {result.serial_seconds:.2f} |
| Parallel (`n_jobs=-1`) | {result.parallel_seconds:.2f} |
| **Speed-up** | **{result.parallel_speedup:.1f}x** |

## Profile (top cumulative time)

Single-threaded cached build; `cProfile`, sorted by cumulative time:

```
{profile_text.strip()}
```
"""


def write_performance_doc(
    result: BenchmarkResult,
    profile_text: str,
    *,
    path: Path = Path("docs/performance.md"),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_performance_doc(result, profile_text), encoding="utf-8")
    return path
