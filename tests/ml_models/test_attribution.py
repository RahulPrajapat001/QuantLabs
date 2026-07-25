"""SHAP attribution — which features the model actually used."""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.ml_models.attribution import shap_attribution
from quantlab.ml_models.models import ModelSpec, Task

pytest.importorskip("shap")

FAMILIES = ["ridge", "lasso", "random_forest", "xgboost", "lightgbm"]


def fit(dataset, family: str, task: Task = Task.REGRESSION):
    params = {"n_estimators": 40} if family in {"xgboost", "lightgbm"} else {}
    if family == "random_forest":
        params = {"n_estimators": 30}
    spec = ModelSpec(family=family, task=task, params=params)
    estimator = spec.build()
    target = dataset.y if task is Task.REGRESSION else (dataset.y > 0).astype(int)
    estimator.fit(dataset.X, target)
    return estimator, spec


@pytest.mark.parametrize("family", FAMILIES)
def test_attribution_ranks_the_planted_signal_first(dataset, family):
    """The fixture's forward return is driven by ``signal`` and nothing else."""

    estimator, spec = fit(dataset, family)
    attribution = shap_attribution(estimator, dataset.X, spec=spec, max_samples=400)
    assert attribution.ranked[0][0] == "signal"
    assert set(attribution.feature_names) == set(dataset.feature_names)
    assert attribution.n_samples == 400


@pytest.mark.parametrize("family", ["ridge", "lightgbm"])
def test_the_explainer_is_chosen_by_model_family(dataset, family):
    estimator, spec = fit(dataset, family)
    attribution = shap_attribution(estimator, dataset.X, spec=spec, max_samples=200)
    expected = "TreeExplainer" if spec.is_tree else "LinearExplainer"
    assert attribution.explainer == expected


def test_classification_attribution_reduces_to_the_up_class(dataset):
    estimator, spec = fit(dataset, "lightgbm", Task.CLASSIFICATION)
    attribution = shap_attribution(estimator, dataset.X, spec=spec, max_samples=200)
    assert attribution.mean_abs_shap.shape == (len(dataset.feature_names),)
    assert np.all(attribution.mean_abs_shap >= 0.0)


def test_concentration_reports_how_few_features_the_model_really_uses(dataset):
    estimator, spec = fit(dataset, "lightgbm")
    attribution = shap_attribution(estimator, dataset.X, spec=spec, max_samples=200)
    assert 0.0 <= attribution.concentration <= 1.0
    # Four features, so the top five carry everything.
    assert attribution.concentration == pytest.approx(1.0)


def test_the_sample_is_capped_and_reported(dataset):
    estimator, spec = fit(dataset, "ridge")
    attribution = shap_attribution(estimator, dataset.X, spec=spec, max_samples=50)
    assert attribution.n_samples == 50
    assert "50 samples" in attribution.render()


def test_a_small_frame_is_used_whole(dataset):
    estimator, spec = fit(dataset, "ridge")
    small = dataset.X.head(25)
    attribution = shap_attribution(estimator, small, spec=spec, max_samples=1000)
    assert attribution.n_samples == 25


def test_render_lists_features_strongest_first(dataset):
    estimator, spec = fit(dataset, "random_forest")
    lines = (
        shap_attribution(estimator, dataset.X, spec=spec, max_samples=200)
        .render()
        .splitlines()
    )
    named = [line.split()[0] for line in lines[2:]]
    assert named[0] == "signal"


def test_attributing_an_empty_frame_raises(dataset):
    estimator, spec = fit(dataset, "ridge")
    with pytest.raises(ValueError, match="empty feature frame"):
        shap_attribution(estimator, dataset.X.head(0), spec=spec)


def test_the_serialised_form_carries_the_explainer_and_the_ranking(dataset):
    estimator, spec = fit(dataset, "xgboost")
    payload = shap_attribution(
        estimator, dataset.X, spec=spec, max_samples=200
    ).as_dict()
    assert payload["explainer"] == "TreeExplainer"
    assert set(payload["mean_abs_shap"]) == set(dataset.feature_names)
    assert payload["n_samples"] == 200
