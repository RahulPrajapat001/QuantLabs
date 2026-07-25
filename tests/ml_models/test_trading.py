"""The paper long-short overlay that the baseline comparison runs on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.ml_models.trading import (
    TradingRule,
    default_one_way_cost,
    trade_predictions,
)

GROSS = TradingRule(cost_per_unit_traded=0.0)


def frame(*, n_dates: int = 20, n_securities: int = 10, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_dates)
    rows = []
    for day in dates:
        prediction = rng.normal(size=n_securities)
        rows.append(
            pd.DataFrame(
                {
                    "date": day,
                    "security_id": [f"SEC{i:03d}" for i in range(n_securities)],
                    "prediction": prediction,
                    # The realised return follows the prediction, so the rule has
                    # something to find.
                    "forward_return": 0.01 * prediction
                    + rng.normal(scale=0.002, size=n_securities),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --- Weights -----------------------------------------------------------------


def test_buy_and_hold_earns_the_equal_weighted_cross_sectional_mean():
    data = frame()
    result = trade_predictions(data, label="bh", rule=GROSS, equal_weight_all=True)
    expected = data.groupby("date")["forward_return"].mean().to_numpy()
    assert np.allclose(result.gross_returns, expected)


def test_long_short_is_market_neutral_and_unit_gross():
    """Both legs at half the book: net zero exposure, gross one."""

    data = frame(n_dates=1, n_securities=10)
    result = trade_predictions(
        data, label="ls", rule=TradingRule(quantile=0.2, cost_per_unit_traded=0.0)
    )
    # Entering from cash trades the whole gross book.
    assert result.traded_notional[0] == pytest.approx(1.0)


def test_long_only_holds_the_top_slice_only():
    data = frame(n_dates=1, n_securities=10)
    rule = TradingRule(mode="long_only", quantile=0.3, cost_per_unit_traded=0.0)
    result = trade_predictions(data, label="lo", rule=rule)

    top = data.nlargest(3, "prediction")["forward_return"].mean()
    assert result.gross_returns[0] == pytest.approx(top)


def test_a_perfect_predictor_beats_holding_everything():
    data = frame()
    smart = trade_predictions(data, label="model", rule=GROSS)
    hold = trade_predictions(data, label="bh", rule=GROSS, equal_weight_all=True)
    assert smart.gross_returns.mean() > hold.gross_returns.mean()


# --- Costs -------------------------------------------------------------------


def test_zero_cost_leaves_net_equal_to_gross():
    result = trade_predictions(frame(), label="x", rule=GROSS)
    assert np.allclose(result.net_returns, result.gross_returns)
    assert result.cost_model == "ZeroCostModel"
    assert result.total_cost == pytest.approx(0.0)


def test_costs_are_charged_on_the_notional_actually_traded():
    data = frame()
    cost = 0.001
    priced = trade_predictions(
        data, label="x", rule=TradingRule(cost_per_unit_traded=cost)
    )
    assert np.allclose(
        priced.gross_returns - priced.net_returns, priced.traded_notional * cost
    )
    assert priced.total_cost > 0.0


def test_a_costed_run_names_a_cost_model_that_admits_what_it_is():
    """The log must not be able to pass this off as the itemised ledger."""

    result = trade_predictions(frame(), label="x", rule=TradingRule())
    assert result.cost_model == "IndianEquityCostModel(turnover-approx)"


def test_the_default_one_way_cost_is_half_the_indian_round_trip():
    from quantlab.backtesting.costs import IndianEquityCostModel

    assert default_one_way_cost() == pytest.approx(
        IndianEquityCostModel().round_trip_cost_fraction() / 2.0
    )
    # Sanity: an Indian delivery round trip is around 20 bps.
    assert 0.0005 < default_one_way_cost() < 0.003


def test_a_static_book_pays_nothing_after_it_is_established():
    """Turnover, not holdings, is what costs money."""

    dates = pd.bdate_range("2016-01-04", periods=5)
    rows = []
    for day in dates:
        rows.append(
            pd.DataFrame(
                {
                    "date": day,
                    "security_id": [f"SEC{i:03d}" for i in range(10)],
                    "prediction": np.arange(10.0),  # identical ranking every day
                    "forward_return": np.full(10, 0.001),
                }
            )
        )
    result = trade_predictions(
        pd.concat(rows, ignore_index=True),
        label="static",
        rule=TradingRule(cost_per_unit_traded=0.001),
    )
    assert result.traded_notional[0] > 0.0
    assert np.allclose(result.traded_notional[1:], 0.0)


# --- Overlapping observations ------------------------------------------------


def test_a_multi_day_horizon_rebalances_every_horizon_dates():
    """Daily rebalancing on a 5-day label would overlap 4 days of price movement.

    Overlapping observations understate the standard deviation and inflate the
    Sharpe, so the series is sampled at the horizon instead.
    """

    data = frame(n_dates=20)
    result = trade_predictions(data, label="x", rule=GROSS, horizon=5)
    assert len(result) == 4
    assert result.periods_per_year == 252 // 5


def test_a_daily_horizon_uses_every_date():
    data = frame(n_dates=20)
    result = trade_predictions(data, label="x", rule=GROSS, horizon=1)
    assert len(result) == 20
    assert result.periods_per_year == 252


# --- The result object -------------------------------------------------------


def test_the_sharpe_matches_the_definition_used_by_the_research_log():
    from quantlab.research.log import RunResult

    result = trade_predictions(frame(), label="x", rule=GROSS)
    logged = RunResult.from_returns(
        result.net_returns, periods_per_year=result.periods_per_year
    )
    assert logged.sharpe == pytest.approx(result.annualised_sharpe)


def test_traded_metrics_are_nested_so_they_cannot_pose_as_the_run_s_own_sharpe():
    from quantlab.research.reporting import find_bare_sharpe_keys

    result = trade_predictions(frame(), label="x", rule=GROSS)
    metrics = {"traded": dict(result.as_dict())}
    assert find_bare_sharpe_keys(metrics) == []


def test_max_drawdown_is_non_positive_and_hit_rate_is_a_fraction():
    result = trade_predictions(frame(), label="x", rule=GROSS)
    assert result.max_drawdown <= 0.0
    assert 0.0 <= result.hit_rate <= 1.0


def test_render_states_both_gross_and_net():
    text = trade_predictions(frame(), label="model", rule=TradingRule()).render()
    assert "Sharpe (net)" in text
    assert "Sharpe (gross)" in text
    assert "total cost paid" in text


# --- Configuration -----------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "market_neutral"}, "long_short.*long_only"),
        ({"quantile": 0.8}, "quantile must lie"),
        ({"quantile": 0.0}, "quantile must lie"),
    ],
)
def test_the_rule_validates_itself(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TradingRule(**kwargs)


def test_a_missing_column_is_named():
    with pytest.raises(KeyError, match="forward_return"):
        trade_predictions(
            pd.DataFrame({"date": [], "security_id": [], "prediction": []}),
            label="x",
        )
