"""Metrics, and the label that travels with them."""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.ml_models.metrics import (
    Sample,
    compute_classification_metrics,
    compute_regression_metrics,
    metrics_as_dict,
)

# --- Regression --------------------------------------------------------------


def test_mae_and_rmse_match_the_hand_computation():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    predicted = np.array([0.0, 2.0, 0.0, 3.0])
    # errors: 0, +1, -2, 0  ->  MAE = 3/4, RMSE = sqrt(5/4)
    metrics = compute_regression_metrics(truth, predicted, sample=Sample.OUT_OF_SAMPLE)
    assert metrics.mae == pytest.approx(0.75)
    assert metrics.rmse == pytest.approx(np.sqrt(1.25))
    assert metrics.n == 4


def test_r2_is_one_for_a_perfect_fit_and_zero_for_the_mean():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_regression_metrics(
        truth, truth, sample=Sample.IN_SAMPLE
    ).r2 == pytest.approx(1.0)
    mean_only = np.full_like(truth, truth.mean())
    assert compute_regression_metrics(
        truth, mean_only, sample=Sample.OUT_OF_SAMPLE
    ).r2 == pytest.approx(0.0)


def test_directional_accuracy_ignores_zero_calls():
    """Predicting 0.0 everywhere is not a 100% correct directional call."""

    truth = np.array([0.01, -0.01, 0.02, -0.02])
    metrics = compute_regression_metrics(
        truth, np.zeros(4), sample=Sample.OUT_OF_SAMPLE
    )
    assert np.isnan(metrics.directional_accuracy)

    half_right = np.array([0.01, 0.01, 0.01, -0.01])
    assert compute_regression_metrics(
        truth, half_right, sample=Sample.OUT_OF_SAMPLE
    ).directional_accuracy == pytest.approx(0.75)


def test_rank_ic_is_one_for_a_monotone_prediction():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaled = truth * 3.0 + 10.0
    metrics = compute_regression_metrics(truth, scaled, sample=Sample.OUT_OF_SAMPLE)
    assert metrics.rank_ic == pytest.approx(1.0)
    assert metrics.headline == metrics.rank_ic


def test_non_finite_pairs_are_dropped():
    truth = np.array([1.0, np.nan, 3.0])
    predicted = np.array([1.0, 2.0, np.inf])
    assert (
        compute_regression_metrics(truth, predicted, sample=Sample.OUT_OF_SAMPLE).n == 1
    )


def test_misaligned_arrays_raise():
    with pytest.raises(ValueError, match="misaligned"):
        compute_regression_metrics(
            np.zeros(3), np.zeros(4), sample=Sample.OUT_OF_SAMPLE
        )


# --- Classification ----------------------------------------------------------


def test_confusion_matrix_and_derived_rates():
    truth = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.4, 0.7, 0.2, 0.1, 0.3, 0.6])
    # threshold 0.5 -> predicted 1,1,0,1,0,0,0,1
    #   tp = 3 (0.9, 0.8, 0.6), fn = 1 (0.4)
    #   fp = 1 (0.7), tn = 3 (0.2, 0.1, 0.3)
    metrics = compute_classification_metrics(truth, score, sample=Sample.OUT_OF_SAMPLE)
    assert metrics.confusion == ((3, 1), (1, 3))
    assert metrics.accuracy == pytest.approx(6 / 8)
    assert metrics.precision == pytest.approx(3 / 4)
    assert metrics.recall == pytest.approx(3 / 4)
    assert metrics.f1 == pytest.approx(0.75)
    assert metrics.base_rate == pytest.approx(0.5)


def test_roc_auc_is_one_for_a_perfectly_separating_score():
    truth = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_classification_metrics(truth, score, sample=Sample.OUT_OF_SAMPLE)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.headline == metrics.roc_auc


def test_roc_auc_is_nan_when_only_one_class_is_present():
    """An AUC is undefined on one class. Returning 0.5 would be inventing a number."""

    metrics = compute_classification_metrics(
        np.ones(5), np.linspace(0.1, 0.9, 5), sample=Sample.OUT_OF_SAMPLE
    )
    assert np.isnan(metrics.roc_auc)
    assert metrics.headline == 0.5
    assert "n/a" in metrics.render()


def test_lift_over_base_rate_is_negative_for_a_worse_than_constant_model():
    """Equity direction is not a coin flip, so the majority class is the real floor."""

    truth = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])  # 70% up
    score = np.full(10, 0.49)  # always predicts "down"
    metrics = compute_classification_metrics(truth, score, sample=Sample.OUT_OF_SAMPLE)
    assert metrics.accuracy == pytest.approx(0.3)
    assert metrics.lift_over_base_rate == pytest.approx(0.3 - 0.7)


def test_a_custom_threshold_moves_the_decision_boundary():
    truth = np.array([1, 1, 0, 0])
    score = np.array([0.2, 0.3, -0.1, -0.4])
    metrics = compute_classification_metrics(
        truth, score, sample=Sample.OUT_OF_SAMPLE, threshold=0.0
    )
    assert metrics.accuracy == pytest.approx(1.0)


# --- The sample label --------------------------------------------------------


def test_metric_keys_are_prefixed_by_their_sample():
    truth = np.array([1.0, 2.0, 3.0])
    oos = compute_regression_metrics(truth, truth, sample=Sample.OUT_OF_SAMPLE)
    ins = compute_regression_metrics(truth, truth, sample=Sample.IN_SAMPLE)
    assert "oos_mae" in oos.as_dict()
    assert "is_mae" in ins.as_dict()
    assert not set(oos.as_dict()) & set(ins.as_dict())


def test_in_sample_metrics_say_so_when_rendered():
    truth = np.array([1.0, 2.0, 3.0])
    rendered = compute_regression_metrics(
        truth, truth, sample=Sample.IN_SAMPLE
    ).render()
    assert "IN-SAMPLE" in rendered
    assert "not a result" in rendered


def test_out_of_sample_metrics_say_so_too():
    truth = np.array([1.0, 2.0, 3.0])
    rendered = compute_regression_metrics(
        truth, truth, sample=Sample.OUT_OF_SAMPLE
    ).render()
    assert "OUT-OF-SAMPLE" in rendered


def test_metrics_as_dict_namespaces_several_objects():
    truth = np.array([1.0, 2.0, 3.0])
    merged = metrics_as_dict(
        {
            "cv": compute_regression_metrics(truth, truth, sample=Sample.OUT_OF_SAMPLE),
            "final": compute_regression_metrics(truth, truth, sample=Sample.IN_SAMPLE),
        }
    )
    assert "cv_oos_mae" in merged
    assert "final_is_mae" in merged


def test_scoring_an_empty_sample_raises():
    with pytest.raises(ValueError, match="no finite"):
        compute_regression_metrics([], [], sample=Sample.OUT_OF_SAMPLE)
