"""Smoke tests for the performance benchmark (tiny scale, fast)."""

from __future__ import annotations

from pathlib import Path

from quantlab.feature_engineering.benchmark import (
    make_synthetic_panel,
    profile_build,
    render_performance_doc,
    run_benchmark,
    write_performance_doc,
)


def test_synthetic_panel_shape() -> None:
    panel = make_synthetic_panel(5, 100)
    assert set(panel["security_id"].unique()) == {f"SEC-{i:04d}" for i in range(5)}
    assert len(panel) == 5 * 100


def test_run_benchmark_returns_positive_timings() -> None:
    result = run_benchmark(n_securities=6, n_days=300, naive_scale_securities=3)
    assert result.n_features >= 70
    for seconds in (
        result.naive_seconds,
        result.cached_seconds,
        result.serial_seconds,
        result.parallel_seconds,
        result.ema_loop_seconds,
        result.ema_vectorised_seconds,
    ):
        assert seconds > 0.0
    # The DAG-cached pass never does *more* work than the naive per-feature pass.
    assert result.cached_seconds <= result.naive_seconds


def test_render_and_write_performance_doc(tmp_path: Path) -> None:
    result = run_benchmark(n_securities=4, n_days=300, naive_scale_securities=2)
    profile_text = profile_build(n_securities=4, n_days=300, top=5)
    doc = render_performance_doc(result, profile_text)
    assert "Feature build performance" in doc
    assert "shared intermediates" in doc

    out = write_performance_doc(result, profile_text, path=tmp_path / "performance.md")
    assert out.exists()
    assert "Profile" in out.read_text(encoding="utf-8")
