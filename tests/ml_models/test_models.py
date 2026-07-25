"""The model registry: what can be built, and what deliberately cannot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.ml_models.models import (
    DEFAULT_SEARCH_SPACE,
    MODEL_FAMILIES,
    ModelSpec,
    Task,
    UnknownModelError,
    build_model,
    supports_probability,
)

FAMILIES = sorted(MODEL_FAMILIES)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(200, 4)), columns=["a", "b", "c", "d"])
    target = features["a"] * 0.5 + rng.normal(scale=0.1, size=200)
    return features, target


# --- What the registry permits -----------------------------------------------


def test_the_permitted_families_are_exactly_the_five_the_prd_lists():
    assert FAMILIES == ["lasso", "lightgbm", "random_forest", "ridge", "xgboost"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("task", list(Task))
def test_every_family_builds_and_fits_for_both_targets(family, task, toy):
    features, target = toy
    labels = target if task is Task.REGRESSION else (target > 0).astype(int)
    estimator = build_model(family, task)
    estimator.fit(features, labels)
    predictions = estimator.predict(features)
    assert len(predictions) == len(features)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_classifier_can_produce_probabilities(family, toy):
    """ROC-AUC and the ranking the trading rule needs both require a score."""

    features, target = toy
    estimator = build_model(family, Task.CLASSIFICATION)
    estimator.fit(features, (target > 0).astype(int))
    assert supports_probability(estimator)
    proba = estimator.predict_proba(features)
    assert proba.shape == (len(features), 2)


@pytest.mark.parametrize(
    "name", ["lstm", "gru", "transformer", "neural_network", "arima"]
)
def test_sequence_models_cannot_be_built(name):
    """The scope cut in CLAUDE.md §4 is enforced, not merely documented."""

    with pytest.raises(UnknownModelError, match="deliberate scope cut"):
        build_model(name, Task.REGRESSION)


# --- Leak-proof preprocessing ------------------------------------------------


@pytest.mark.parametrize("family", ["ridge", "lasso"])
def test_linear_models_carry_their_scaler_inside_the_estimator(family, toy):
    """Scaling the panel before splitting leaks the test fold's moments into train."""

    estimator = build_model(family, Task.REGRESSION)
    assert hasattr(estimator, "named_steps")
    assert "scale" in estimator.named_steps

    features, target = toy
    estimator.fit(features, target)
    # The scaler saw only the data it was fitted on.
    assert estimator.named_steps["scale"].n_samples_seen_ == len(features)


def test_tree_models_are_not_wrapped_in_a_scaler():
    estimator = build_model("random_forest", Task.REGRESSION)
    assert not hasattr(estimator, "named_steps")


# --- Reproducibility ---------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_the_same_spec_produces_the_same_predictions(family, toy):
    """A logged configuration has to be rebuildable from its logged parameters."""

    features, target = toy
    spec = ModelSpec(family=family, task=Task.REGRESSION, random_state=3)
    first = spec.build()
    first.fit(features, target)
    rebuilt = ModelSpec(
        family=spec.parameters()["family"],
        task=Task(spec.parameters()["task"]),
        random_state=spec.parameters()["random_state"],
    )
    second = rebuilt.build()
    second.fit(features, target)
    assert np.allclose(first.predict(features), second.predict(features))


def test_build_returns_a_fresh_estimator_each_time():
    spec = ModelSpec(family="ridge", task=Task.REGRESSION)
    assert spec.build() is not spec.build()


# --- The spec ----------------------------------------------------------------


def test_the_logged_name_identifies_family_and_task():
    spec = ModelSpec(family="xgboost", task=Task.CLASSIFICATION)
    assert spec.name == "ml:xgboost:classification"


def test_parameters_round_trip_through_the_log_shape():
    spec = ModelSpec(family="lightgbm", task=Task.REGRESSION, params={"num_leaves": 7})
    parameters = spec.parameters()
    assert parameters["family"] == "lightgbm"
    assert parameters["task"] == "regression"
    assert parameters["num_leaves"] == 7
    assert "random_state" in parameters


def test_with_params_merges_rather_than_replaces():
    spec = ModelSpec(family="xgboost", task=Task.REGRESSION, params={"max_depth": 3})
    updated = spec.with_params(learning_rate=0.1)
    assert updated.params == {"max_depth": 3, "learning_rate": 0.1}
    assert spec.params == {"max_depth": 3}, "the original must be untouched"


def test_only_tree_families_are_marked_as_trees():
    assert ModelSpec(family="xgboost", task=Task.REGRESSION).is_tree
    assert not ModelSpec(family="ridge", task=Task.REGRESSION).is_tree


def test_every_family_declares_a_search_space():
    """The space has to be data, because the PRD requires it to be reported."""

    assert set(DEFAULT_SEARCH_SPACE) == set(MODEL_FAMILIES)
    for family, grid in DEFAULT_SEARCH_SPACE.items():
        assert grid, f"{family} declares an empty grid"
        for axis, values in grid.items():
            assert len(values) > 1, f"{family}.{axis} has nothing to search over"
