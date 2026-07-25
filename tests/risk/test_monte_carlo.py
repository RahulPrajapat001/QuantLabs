"""Monte Carlo from bootstrapped returns, and why the block variant is the default."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.risk.monte_carlo import monte_carlo
from quantlab.risk.var import RiskError

from .conftest import fat_tailed_returns, normal_returns

# --- The distribution is recovered -------------------------------------------


def test_median_terminal_value_tracks_the_sampled_drift() -> None:
    """A 0.0004/day drift over 252 days compounds to roughly 1.105."""

    result = monte_carlo(
        normal_returns(n=2000, mean=0.0004, sd=0.010),
        n_simulations=4000,
        horizon_days=252,
        method="iid",
        seed=1,
    )
    expected = (1.0 + 0.0004) ** 252
    assert result.median_terminal == pytest.approx(expected, rel=0.05)


def test_a_negative_drift_produces_mostly_losing_paths() -> None:
    result = monte_carlo(
        normal_returns(n=2000, mean=-0.0005, sd=0.010),
        n_simulations=3000,
        horizon_days=252,
        seed=2,
    )
    assert result.probability_of_loss > 0.7


def test_a_longer_horizon_widens_the_distribution() -> None:
    short = monte_carlo(
        normal_returns(n=1500), horizon_days=60, n_simulations=3000, seed=3
    )
    long = monte_carlo(
        normal_returns(n=1500), horizon_days=504, n_simulations=3000, seed=3
    )
    short_spread = short.percentiles((0.05, 0.95))
    long_spread = long.percentiles((0.05, 0.95))
    short_width = float(
        short_spread["terminal_value"].iloc[1] - short_spread["terminal_value"].iloc[0]
    )
    long_width = float(
        long_spread["terminal_value"].iloc[1] - long_spread["terminal_value"].iloc[0]
    )
    assert long_width > short_width


def test_bootstrapping_preserves_fat_tails_a_gaussian_fit_would_lose() -> None:
    """The reason this resamples rather than fitting a normal.

    A t(3) source produces a materially worse 5th-percentile outcome than a Gaussian
    source with the same standard deviation.
    """

    fat = monte_carlo(
        fat_tailed_returns(n=2000, df=3), n_simulations=4000, horizon_days=252, seed=4
    )
    thin = monte_carlo(
        normal_returns(
            n=2000, mean=0.0, sd=float(fat_tailed_returns(n=2000, df=3).std())
        ),
        n_simulations=4000,
        horizon_days=252,
        seed=4,
    )
    assert fat.probability_of_drawdown(0.30) > thin.probability_of_drawdown(0.30)


# --- IID versus block --------------------------------------------------------


def test_the_block_bootstrap_produces_deeper_drawdowns_than_iid() -> None:
    """The documented reason ``block`` is the default.

    Shuffling days independently destroys volatility clustering, and real losses arrive
    in runs. An IID bootstrap therefore reports drawdowns that are systematically too
    shallow — optimistic in exactly the statistic risk work depends on.
    """

    # A series with genuine volatility clustering: alternating calm and stormy runs.
    rng = np.random.default_rng(5)
    segments = []
    for i in range(20):
        vol = 0.005 if i % 2 == 0 else 0.030
        segments.append(rng.normal(-0.0002 if vol > 0.01 else 0.0004, vol, 60))
    clustered = pd.Series(np.concatenate(segments))

    iid = monte_carlo(
        clustered, method="iid", n_simulations=6000, horizon_days=252, seed=6
    )
    block = monte_carlo(
        clustered,
        method="block",
        block_size=30,
        n_simulations=6000,
        horizon_days=252,
        seed=6,
    )

    # The typical path is deeper under block resampling.
    assert np.median(block.max_drawdowns) > np.median(iid.max_drawdowns)
    # And the effect is concentrated in the tail, which is what it is for. A 20%
    # threshold is saturated here — nearly every path breaches it either way — so the
    # discriminating comparison is further out.
    for threshold in (0.30, 0.40, 0.50):
        assert block.probability_of_drawdown(threshold) > iid.probability_of_drawdown(
            threshold
        ), f"block should breach {threshold:.0%} more often than iid"
    assert np.quantile(block.max_drawdowns, 0.95) > np.quantile(iid.max_drawdowns, 0.95)


def test_the_iid_variant_warns_about_its_own_limitation() -> None:
    result = monte_carlo(normal_returns(), method="iid", n_simulations=500, seed=7)
    text = result.render()
    assert "IID bootstrap shuffles days independently" in text
    assert "optimistic" in text


def test_block_size_defaults_to_the_cube_root_rule() -> None:
    returns = normal_returns(n=1000)
    result = monte_carlo(returns, method="block", n_simulations=200, seed=8)
    assert result.block_size == pytest.approx(int(np.ceil(1000 ** (1 / 3))))
    assert "block≈" in result.render()


def test_iid_reports_no_block_size() -> None:
    result = monte_carlo(normal_returns(), method="iid", n_simulations=200, seed=9)
    assert result.block_size is None


def test_block_sampling_wraps_rather_than_truncating() -> None:
    """Wrapping is what makes the stationary bootstrap stationary.

    Every observation must be equally likely to appear; truncating at the sample end
    would under-sample the most recent history.
    """

    # A series where each value is unique and identifiable.
    returns = pd.Series(np.arange(100) / 10_000.0)
    result = monte_carlo(
        returns,
        method="block",
        block_size=10,
        n_simulations=200,
        horizon_days=200,
        seed=10,
    )
    # With wrapping and a 200-day horizon over a 100-day sample, paths must revisit
    # the start, which shows up as a valid finite result rather than an error.
    assert np.isfinite(result.terminal_values).all()
    assert result.source_observations == 100


# --- Reported statistics -----------------------------------------------------


def test_percentiles_are_ordered() -> None:
    result = monte_carlo(normal_returns(n=1500), n_simulations=3000, seed=11)
    table = result.percentiles((0.05, 0.25, 0.5, 0.75, 0.95))
    assert table["terminal_value"].is_monotonic_increasing
    assert table["annual_return"].is_monotonic_increasing
    assert table["max_drawdown"].is_monotonic_increasing


def test_drawdowns_are_positive_fractions_below_one() -> None:
    result = monte_carlo(normal_returns(n=1500), n_simulations=2000, seed=12)
    assert (result.max_drawdowns >= 0).all()
    assert (result.max_drawdowns <= 1.0).all()


def test_a_falling_path_still_records_a_drawdown() -> None:
    """The starting value is included in the peak, so a monotonic decline shows up."""

    always_down = pd.Series([-0.002] * 300)
    result = monte_carlo(
        always_down, n_simulations=100, horizon_days=100, method="iid", seed=13
    )
    assert result.max_drawdowns.min() > 0.15
    assert result.probability_of_loss == 1.0


def test_drawdown_probabilities_are_monotonic_in_the_threshold() -> None:
    result = monte_carlo(normal_returns(n=1500), n_simulations=3000, seed=14)
    assert result.probability_of_drawdown(0.10) >= result.probability_of_drawdown(0.20)
    assert result.probability_of_drawdown(0.20) >= result.probability_of_ruin(0.50)


def test_terminal_value_at_risk_is_a_positive_loss() -> None:
    result = monte_carlo(normal_returns(n=1500, mean=0.0), n_simulations=3000, seed=15)
    var = result.value_at_risk(0.95)
    assert 0.0 <= var <= 1.0
    assert result.value_at_risk(0.99) >= var


def test_render_states_what_the_simulation_cannot_do() -> None:
    """It cannot produce a crisis the sample never contained, and says so."""

    text = monte_carlo(normal_returns(), n_simulations=500, seed=16).render()
    assert "cannot produce a crisis the sample does not contain" in text
    assert "stress module" in text


# --- Determinism and validation ----------------------------------------------


def test_the_same_seed_gives_the_same_answer() -> None:
    returns = normal_returns(n=1000)
    first = monte_carlo(returns, n_simulations=500, seed=17)
    second = monte_carlo(returns, n_simulations=500, seed=17)
    assert np.array_equal(first.terminal_values, second.terminal_values)


def test_different_seeds_give_different_answers() -> None:
    returns = normal_returns(n=1000)
    first = monte_carlo(returns, n_simulations=500, seed=18)
    second = monte_carlo(returns, n_simulations=500, seed=19)
    assert not np.array_equal(first.terminal_values, second.terminal_values)


def test_too_short_a_history_is_refused() -> None:
    with pytest.raises(RiskError, match="at least 60 finite observations"):
        monte_carlo(pd.Series([0.01] * 30))


def test_invalid_arguments_are_refused() -> None:
    returns = normal_returns()
    with pytest.raises(RiskError, match="n_simulations"):
        monte_carlo(returns, n_simulations=0)
    with pytest.raises(RiskError, match="horizon_days"):
        monte_carlo(returns, horizon_days=0)
    with pytest.raises(RiskError, match="unknown bootstrap method"):
        monte_carlo(returns, method="magic")  # type: ignore[arg-type]


def test_drawdown_threshold_is_validated() -> None:
    result = monte_carlo(normal_returns(), n_simulations=200, seed=20)
    with pytest.raises(RiskError, match="threshold must lie"):
        result.probability_of_drawdown(1.5)


def test_non_finite_inputs_are_dropped() -> None:
    returns = normal_returns(n=300).copy()
    returns.iloc[5] = np.nan
    result = monte_carlo(returns, n_simulations=200, seed=21)
    assert result.source_observations == 299
