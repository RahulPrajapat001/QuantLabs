"""Multiple-testing corrections, and the trials counts they read from the log."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantlab.research.deflated import (
    DeflationError,
    deflate_run,
    deflate_strategy,
    deflated_sharpe_ratio,
    expected_max_sharpe_analytic,
    expected_max_sharpe_null,
    null_max_sharpe_for_run,
    pbo_from_log,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from quantlab.research.log import ResearchLog, ResearchLogError, RunResult

from .conftest import make_spec, zero_edge_returns

# --- Building blocks ---------------------------------------------------------


def test_probabilistic_sharpe_rises_with_track_record_length() -> None:
    short = probabilistic_sharpe_ratio(observed_sharpe=1.0, n_periods=100)
    long = probabilistic_sharpe_ratio(observed_sharpe=1.0, n_periods=2520)
    assert 0.5 < short < long < 1.0


def test_probabilistic_sharpe_punishes_negative_skew_and_fat_tails() -> None:
    normal = probabilistic_sharpe_ratio(
        observed_sharpe=1.2, n_periods=1260, skew=0.0, kurtosis=3.0
    )
    nasty = probabilistic_sharpe_ratio(
        observed_sharpe=1.2, n_periods=1260, skew=-1.5, kurtosis=9.0
    )
    assert nasty < normal


def test_expected_max_sharpe_grows_with_trials() -> None:
    one = expected_max_sharpe_analytic(trials=1, sharpe_variance=0.25)
    ten = expected_max_sharpe_analytic(trials=10, sharpe_variance=0.25)
    thousand = expected_max_sharpe_analytic(trials=1000, sharpe_variance=0.25)
    # No selection, no benchmark to clear.
    assert one == 0.0
    assert 0.0 < ten < thousand


def test_expected_max_sharpe_scales_with_the_spread_of_trials() -> None:
    narrow = expected_max_sharpe_analytic(trials=200, sharpe_variance=0.04)
    wide = expected_max_sharpe_analytic(trials=200, sharpe_variance=0.36)
    assert wide == pytest.approx(narrow * math.sqrt(0.36 / 0.04), rel=1e-9)


def test_deflated_sharpe_falls_as_trials_rise() -> None:
    kwargs = {
        "observed_sharpe": 1.2,
        "sharpe_variance": 0.2,
        "n_periods": 1260,
        "skew": -0.2,
        "kurtosis": 4.0,
    }
    single = deflated_sharpe_ratio(trials=1, **kwargs)  # type: ignore[arg-type]
    many = deflated_sharpe_ratio(trials=300, **kwargs)  # type: ignore[arg-type]
    assert single.deflated_sharpe > many.deflated_sharpe
    assert single.significant
    assert not many.significant
    # With one trial there is nothing to deflate, so DSR equals the plain PSR.
    assert single.deflated_sharpe == pytest.approx(single.probabilistic_sharpe or 0.0)


def test_degenerate_return_shape_is_refused() -> None:
    with pytest.raises(DeflationError, match="variance term is non-positive"):
        deflated_sharpe_ratio(
            observed_sharpe=40.0,
            trials=10,
            sharpe_variance=0.1,
            n_periods=100,
            skew=8.0,
            kurtosis=1.0,
            periods_per_year=1,
        )


# --- The headline test: best of 200 edgeless strategies ----------------------


def _log_zero_edge_strategies(
    research_log: ResearchLog, *, count: int = 200, n_periods: int = 1260
) -> None:
    rng = np.random.default_rng(20240101)
    for i in range(count):
        with research_log.run(make_spec(f"noise_{i:03d}", variant=i)) as run:
            run.record(
                RunResult.from_returns(
                    zero_edge_returns(rng, n_periods=n_periods),
                    metrics={"volatility": 0.012},
                )
            )


def test_best_of_200_zero_edge_strategies_deflates_to_insignificance(
    research_log: ResearchLog,
) -> None:
    """The core integrity claim of the whole layer.

    Two hundred strategies with no edge are logged. The best of them has a Sharpe
    that would look publishable in isolation. Deflated against the trials count the
    log itself holds, it must not be significant — and it must not even clear the
    expected maximum of an edgeless search of that size.
    """

    _log_zero_edge_strategies(research_log)
    best = research_log.best_run()
    assert best is not None

    # Undeflated, this looks like a result: a Sharpe near 1 with five years of
    # daily data has a probabilistic Sharpe close to certainty.
    naive = probabilistic_sharpe_ratio(
        observed_sharpe=float(best.sharpe or 0.0),
        n_periods=int(best.n_periods or 0),
        skew=float(best.returns_skew or 0.0),
        kurtosis=float(best.returns_kurtosis or 3.0),
    )
    assert float(best.sharpe or 0.0) > 0.8
    assert naive > 0.95

    result = deflate_run(research_log, best.run_id)

    assert result.trials == 200  # read from research_log, not passed in
    assert result.deflated_sharpe < 0.5
    assert not result.significant
    assert not result.clears_selection_bias()
    assert result.expected_max_sharpe > result.observed_sharpe
    # The correction, not the input, is what changed the conclusion.
    assert result.deflated_sharpe < naive


def test_a_genuine_edge_survives_the_same_deflation(research_log: ResearchLog) -> None:
    """The correction must not simply reject everything.

    Same 200-trial log, plus one strategy with a real edge. A Sharpe far above what
    the search could produce by luck still clears.
    """

    _log_zero_edge_strategies(research_log)
    rng = np.random.default_rng(99)
    # ~2.2 annualised Sharpe: mean/vol per day of 0.14.
    real = rng.normal(0.0, 0.012, 1260) + 0.012 * 0.14
    with research_log.run(make_spec("real_edge")) as run:
        run.record(RunResult.from_returns(real, metrics={"volatility": 0.012}))

    result = deflate_strategy(research_log, "real_edge")
    assert result.trials == 201
    assert result.observed_sharpe > result.expected_max_sharpe
    assert result.significant


def test_scoping_trials_to_one_strategy_is_less_conservative(
    research_log: ResearchLog,
) -> None:
    _log_zero_edge_strategies(research_log, count=40, n_periods=500)
    best = research_log.best_run()
    assert best is not None
    whole_log = deflate_run(research_log, best.run_id)
    scoped = deflate_run(research_log, best.run_id, scope_strategy=best.strategy_name)
    assert whole_log.trials == 40
    assert scoped.trials == 1
    assert scoped.deflated_sharpe > whole_log.deflated_sharpe


def test_deflating_a_run_without_metrics_raises(research_log: ResearchLog) -> None:
    with research_log.run(make_spec("no_moments")) as run:
        run.record(RunResult(metrics={"total_return": 0.1}))
    with pytest.raises(DeflationError, match="no recorded Sharpe"):
        deflate_run(research_log, run.run_id)


def test_deflating_an_unrun_strategy_raises(research_log: ResearchLog) -> None:
    with pytest.raises(ResearchLogError, match="nothing to deflate"):
        deflate_strategy(research_log, "never_run")


# --- Expected maximum Sharpe under the null (simulated) ---------------------


def test_simulated_null_max_grows_with_trials() -> None:
    few = expected_max_sharpe_null(trials=5, n_periods=1260, n_simulations=200, seed=1)
    many = expected_max_sharpe_null(
        trials=500, n_periods=1260, n_simulations=200, seed=1
    )
    assert few.mean < many.mean
    assert many.max_sharpes.size == 200


def test_simulated_null_agrees_with_the_analytic_benchmark() -> None:
    """The Monte Carlo and the closed form must describe the same null.

    ``E[max]`` from the Gumbel approximation, evaluated at the variance the
    simulated Sharpes actually have, should land close to the simulated mean.
    """

    trials, n_periods = 100, 1260
    null = expected_max_sharpe_null(
        trials=trials, n_periods=n_periods, n_simulations=400, seed=42
    )
    # Variance of a single zero-edge annualised Sharpe over n_periods days.
    single_sharpe_variance = 252.0 / n_periods
    analytic = expected_max_sharpe_analytic(
        trials=trials, sharpe_variance=single_sharpe_variance
    )
    assert null.mean == pytest.approx(analytic, rel=0.15)


def test_null_p_value_flags_a_noise_strategy(research_log: ResearchLog) -> None:
    _log_zero_edge_strategies(research_log, count=50, n_periods=756)
    best = research_log.best_run()
    assert best is not None
    null = null_max_sharpe_for_run(research_log, best.run_id, n_simulations=400, seed=5)
    assert null.trials == 50
    assert null.n_periods == 756
    observed = float(best.sharpe or 0.0)
    # The best of 50 noise strategies is, by construction, an ordinary draw from
    # the distribution of "best of 50 noise strategies".
    assert null.p_value(observed) > 0.05
    assert not null.clears(observed)
    assert "IS NOISE" in null.render(observed)


def test_bootstrap_null_uses_the_strategys_own_distribution() -> None:
    rng = np.random.default_rng(4)
    fat_tailed = rng.standard_t(df=3, size=2000) * 0.01
    null = expected_max_sharpe_null(
        trials=50, n_periods=1260, n_simulations=200, returns=fat_tailed, seed=3
    )
    assert null.method == "bootstrap"
    # The pool is demeaned, so the null has no edge regardless of the input's mean.
    assert null.mean > 0.0
    assert null.volatility == pytest.approx(float(fat_tailed.std(ddof=1)), rel=1e-9)


def test_null_rejects_impossible_arguments() -> None:
    with pytest.raises(DeflationError, match="trials must be at least 1"):
        expected_max_sharpe_null(trials=0, n_periods=100)
    with pytest.raises(DeflationError, match="n_periods must be at least 2"):
        expected_max_sharpe_null(trials=5, n_periods=1)
    with pytest.raises(DeflationError, match="volatility must be positive"):
        expected_max_sharpe_null(trials=5, n_periods=100, volatility=0.0)


# --- Probability of backtest overfitting ------------------------------------


def test_pbo_is_high_when_every_configuration_is_noise() -> None:
    """Selecting the best of 30 noise configurations should overfit outright.

    With no signal to find, the in-sample winner is chosen on sampling noise and
    lands on either side of the out-of-sample median with equal probability, so PBO
    sits near 0.5 — the definition of a procedure that has learned nothing.
    """

    rng = np.random.default_rng(0)
    matrix = rng.normal(0.0, 0.01, size=(1260, 30))
    result = probability_of_backtest_overfitting(matrix, n_splits=10)
    assert result.n_configurations == 30
    assert result.n_combinations == math.comb(10, 5)
    assert 0.3 < result.pbo < 0.7


def test_pbo_is_low_when_one_configuration_genuinely_dominates() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(0.0, 0.01, size=(1260, 12))
    # One configuration has a real, stable edge in every period.
    matrix[:, 3] += 0.01 * 0.15
    result = probability_of_backtest_overfitting(matrix, n_splits=10)
    assert result.pbo < 0.1
    assert result.probability_of_loss < 0.2


def test_pbo_default_splits_are_tractable() -> None:
    rng = np.random.default_rng(2)
    matrix = rng.normal(0.0, 0.01, size=(1008, 20))
    result = probability_of_backtest_overfitting(matrix)
    assert result.n_splits == 16
    assert result.n_combinations == 12_870
    assert 0.0 <= result.pbo <= 1.0
    assert result.logits.size == 12_870


def test_pbo_validates_its_inputs() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(DeflationError, match="2-D"):
        probability_of_backtest_overfitting(rng.normal(size=100))
    with pytest.raises(DeflationError, match="compares configurations"):
        probability_of_backtest_overfitting(rng.normal(size=(500, 1)))
    with pytest.raises(DeflationError, match="even and at least 2"):
        probability_of_backtest_overfitting(rng.normal(size=(500, 5)), n_splits=7)
    with pytest.raises(DeflationError, match="blocks"):
        probability_of_backtest_overfitting(rng.normal(size=(8, 5)), n_splits=16)
    with pytest.raises(DeflationError, match="exceeds max_combinations"):
        probability_of_backtest_overfitting(
            rng.normal(size=(500, 5)), n_splits=16, max_combinations=100
        )


def test_pbo_from_log_requires_every_logged_trial(research_log: ResearchLog) -> None:
    """PBO over a hand-picked subset would measure a search that never happened."""

    rng = np.random.default_rng(7)
    returns: dict[str, np.ndarray] = {}
    for i in range(12):
        series = zero_edge_returns(rng, n_periods=504)
        with research_log.run(make_spec("gridsearch", variant=i)) as run:
            run.record(RunResult.from_returns(series))
        returns[run.run_id] = series

    dropped = dict(returns)
    dropped.pop(next(iter(dropped)))
    with pytest.raises(DeflationError, match="have no return series"):
        pbo_from_log(research_log, dropped, strategy_name="gridsearch")

    result = pbo_from_log(research_log, returns, strategy_name="gridsearch", n_splits=8)
    assert result.n_configurations == 12
    assert 0.0 <= result.pbo <= 1.0


def test_pbo_from_log_rejects_misaligned_series(research_log: ResearchLog) -> None:
    rng = np.random.default_rng(8)
    returns: dict[str, np.ndarray] = {}
    for i, length in enumerate((504, 400)):
        series = zero_edge_returns(rng, n_periods=length)
        with research_log.run(make_spec("ragged", variant=i)) as run:
            run.record(RunResult.from_returns(series))
        returns[run.run_id] = series
    with pytest.raises(DeflationError, match="same periods"):
        pbo_from_log(research_log, returns, strategy_name="ragged")
