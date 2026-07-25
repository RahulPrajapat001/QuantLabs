"""Baseline honesty: the comparison, and the verdict when the model loses."""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.ml_models.baselines import (
    BuyAndHoldBaseline,
    NaiveMomentumBaseline,
    UncalibratedBaselineError,
    baseline_outcome,
    compare_to_baselines,
    default_baselines,
    evaluate_baseline,
    prediction_frame,
    score_predictions,
)
from quantlab.ml_models.dataset import TargetKind, TargetSpec, build_dataset
from quantlab.ml_models.metrics import Sample
from quantlab.ml_models.trading import TradingRule

from .conftest import FEATURES

GROSS = TradingRule(cost_per_unit_traded=0.0)


def split_indices(dataset):
    """Train on the first two thirds, score the last third."""

    cut = int(len(dataset) * 2 / 3)
    return np.arange(cut), np.arange(cut, len(dataset))


# --- Which baselines are required --------------------------------------------


def test_the_two_required_baselines_are_resolved_from_the_panel(dataset):
    baselines = default_baselines(dataset)
    assert [b.name for b in baselines] == ["buy_and_hold", "naive_momentum"]


def test_a_panel_with_no_momentum_feature_is_a_build_error(panel, universe):
    """Module 11 requires this comparison, so it is not silently skipped."""

    dataset = build_dataset(
        panel,
        features=["signal", "noise_a"],
        universe=universe,
        index_name="TESTIDX",
    )
    with pytest.raises(KeyError, match="requires every model to be reported"):
        default_baselines(dataset)


def test_a_comparison_missing_a_required_baseline_is_refused(dataset):
    train_idx, test_idx = split_indices(dataset)
    scores = dataset.forward_return[test_idx]
    metrics = score_predictions(dataset, test_idx, scores, sample=Sample.OUT_OF_SAMPLE)
    from quantlab.ml_models.trading import trade_predictions

    traded = trade_predictions(
        prediction_frame(dataset, test_idx, scores), label="m", rule=GROSS
    )
    only_one = [
        evaluate_baseline(
            BuyAndHoldBaseline(), dataset, train_idx, test_idx, rule=GROSS, horizon=1
        )
    ]
    with pytest.raises(ValueError, match="naive_momentum"):
        compare_to_baselines(
            model_label="m",
            model_metrics=metrics,
            model_traded=traded,
            outcomes=only_one,
        )


# --- Calibration is training-data only ---------------------------------------


def test_buy_and_hold_refuses_to_score_before_it_is_calibrated(dataset):
    with pytest.raises(UncalibratedBaselineError, match="would be a leak"):
        BuyAndHoldBaseline().score(dataset, np.arange(10))


def test_buy_and_hold_predicts_the_training_mean_and_nothing_else(dataset):
    train_idx, test_idx = split_indices(dataset)
    calibrated = BuyAndHoldBaseline().calibrate(dataset, train_idx)
    scores = calibrated.score(dataset, test_idx)
    assert scores.std() == 0.0
    assert scores[0] == pytest.approx(dataset.y.to_numpy()[train_idx].mean())


def test_buy_and_holds_directional_accuracy_is_the_base_rate(panel, universe):
    """The honest floor: the rate at which the market went up anyway.

    Calibrated on training data, so the constant it predicts is the *training*
    base rate. Whether that beats 0.5 on the test set is the question the model
    has to answer, not something the baseline is allowed to peek at.
    """

    dataset = build_dataset(
        panel,
        features=FEATURES,
        universe=universe,
        index_name="T",
        target=TargetSpec(kind=TargetKind.DIRECTION),
    )
    train_idx, test_idx = split_indices(dataset)
    outcome = evaluate_baseline(
        BuyAndHoldBaseline(), dataset, train_idx, test_idx, rule=GROSS, horizon=1
    )
    labels = dataset.y.to_numpy()
    train_rate = labels[train_idx].mean()
    test_rate = labels[test_idx].mean()
    expected = test_rate if train_rate >= 0.5 else 1.0 - test_rate
    assert outcome.metrics.directional_accuracy == pytest.approx(expected)
    assert outcome.metrics.base_rate == pytest.approx(test_rate)


def test_naive_momentum_rescaling_does_not_change_the_ranking(dataset):
    train_idx, test_idx = split_indices(dataset)
    baseline = NaiveMomentumBaseline(feature="momentum_12_1")
    raw = dataset.X["momentum_12_1"].to_numpy()[test_idx]
    scaled = baseline.calibrate(dataset, train_idx).score(dataset, test_idx)

    assert baseline.calibrate(dataset, train_idx).scale > 0.0
    assert np.array_equal(np.argsort(raw), np.argsort(scaled))
    assert np.array_equal(np.sign(raw), np.sign(scaled))


def test_naive_momentum_is_not_rescaled_for_the_direction_target(panel, universe):
    dataset = build_dataset(
        panel,
        features=FEATURES,
        universe=universe,
        index_name="T",
        target=TargetSpec(kind=TargetKind.DIRECTION),
    )
    baseline = NaiveMomentumBaseline(feature="momentum_12_1")
    assert baseline.calibrate(dataset).scale == 1.0
    # …and it decides on the sign of momentum, not on a 0.5 probability threshold.
    assert baseline.decision_threshold == 0.0


def test_a_missing_momentum_column_names_the_available_features(dataset):
    with pytest.raises(KeyError, match="Available"):
        NaiveMomentumBaseline(feature="not_a_feature").score(dataset, np.arange(5))


# --- The verdict -------------------------------------------------------------


def build_comparison(dataset, model_scores):
    train_idx, test_idx = split_indices(dataset)
    from quantlab.ml_models.trading import trade_predictions

    metrics = score_predictions(
        dataset, test_idx, model_scores, sample=Sample.OUT_OF_SAMPLE
    )
    traded = trade_predictions(
        prediction_frame(dataset, test_idx, model_scores),
        label="model",
        rule=GROSS,
    )
    outcomes = [
        baseline_outcome(
            calibrated := baseline.calibrate(dataset, train_idx),
            dataset,
            test_idx,
            calibrated.score(dataset, test_idx),
            rule=GROSS,
            horizon=1,
        )
        for baseline in default_baselines(dataset)
    ]
    return compare_to_baselines(
        model_label="model",
        model_metrics=metrics,
        model_traded=traded,
        outcomes=outcomes,
    )


def test_a_useless_model_is_reported_as_not_beating_the_baselines(dataset):
    """Pure noise as a prediction. The verdict must say so in plain language."""

    _, test_idx = split_indices(dataset)
    noise = np.random.default_rng(0).normal(size=test_idx.size)
    comparison = build_comparison(dataset, noise)

    assert not comparison.beats_all_baselines
    verdict = comparison.verdict()
    assert "did NOT beat" in verdict
    assert "does not earn its complexity" in verdict
    assert comparison.losses_on_prediction or comparison.losses_on_trading


def test_an_oracle_model_is_reported_as_beating_them_but_still_flagged(dataset):
    """Beating the baselines is necessary, not sufficient — the text says so."""

    _, test_idx = split_indices(dataset)
    oracle = dataset.forward_return[test_idx]
    comparison = build_comparison(dataset, oracle)

    assert comparison.beats_all_baselines
    verdict = comparison.verdict()
    assert "beat both required baselines" in verdict
    assert "necessary condition, not a" in verdict
    assert "deflated Sharpe" in verdict


def test_the_verdict_names_the_baseline_that_won(dataset):
    _, test_idx = split_indices(dataset)
    # Deliberately inverted: worse than the momentum rule on every axis.
    inverted = -dataset.X["momentum_12_1"].to_numpy()[test_idx]
    comparison = build_comparison(dataset, inverted)
    assert "naive_momentum" in comparison.verdict()


def test_render_puts_the_model_and_both_baselines_in_one_table(dataset):
    _, test_idx = split_indices(dataset)
    comparison = build_comparison(dataset, dataset.forward_return[test_idx])
    text = comparison.render()
    for name in ("model", "buy_and_hold", "naive_momentum", "VERDICT"):
        assert name in text


def test_the_comparison_serialises_with_the_verdict_included(dataset):
    from quantlab.research.reporting import find_bare_sharpe_keys

    _, test_idx = split_indices(dataset)
    comparison = build_comparison(dataset, dataset.forward_return[test_idx])
    payload = comparison.as_dict()
    assert "verdict" in payload
    assert payload["beats_all_baselines"] is True
    assert "buy_and_hold" in payload and "naive_momentum" in payload
    # Nested, so a baseline's Sharpe can never be paired with the model's
    # deflation by a metrics dict flowing into a report.
    assert find_bare_sharpe_keys({"baselines": payload}) == []


def test_beating_only_one_axis_is_not_beating_the_baselines(dataset):
    """The test is deliberately strict: better on prediction *and* on trading."""

    _, test_idx = split_indices(dataset)
    comparison = build_comparison(dataset, dataset.forward_return[test_idx])
    weakened = type(comparison)(
        model_label=comparison.model_label,
        model_metrics=comparison.baselines[0].metrics,  # ties the buy-and-hold
        model_traded=comparison.model_traded,
        baselines=comparison.baselines,
    )
    assert not weakened.beats_all_baselines
    assert "buy_and_hold" in weakened.losses_on_prediction
