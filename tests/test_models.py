"""Model interface, registry and scikit-learn wrapper tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_stock.models import (
    MeanReversionBaseline,
    Model,
    MomentumBaseline,
    ZeroSignal,
    available_models,
    build_sklearn_model,
    create_model,
    register_model,
)


@pytest.fixture
def training_data(dataset):
    return dataset.features.iloc[:400], dataset.forward_return.iloc[:400]


@pytest.mark.parametrize("name", available_models())
def test_every_registered_model_fits_and_predicts(name: str, dataset) -> None:
    model = create_model(name)
    target = dataset.target(classification=model.is_classifier).iloc[:400]
    features = dataset.features.iloc[:400]
    usable = target.notna().to_numpy()

    model.fit(features[usable], target[usable])
    predictions = model.predict(dataset.features.iloc[400:500])

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (100,)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("name", available_models())
def test_signal_units_are_declared_consistently(name: str, dataset) -> None:
    model = create_model(name)
    assert model.signal_units in {"return", "edge"}
    assert model.signal_units == ("edge" if model.is_classifier else "return")


def test_classifier_edges_stay_within_bounds(dataset) -> None:
    model = create_model("logistic")
    target = dataset.direction.iloc[:400]
    usable = target.notna().to_numpy()
    model.fit(dataset.features.iloc[:400][usable], target[usable])

    edges = model.predict(dataset.features.iloc[400:500])
    assert np.all(np.abs(edges) <= 1.0)
    probabilities = model.predict_proba_up(dataset.features.iloc[400:500])
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_classifier_survives_a_single_class_fold(dataset) -> None:
    """A relentless bull run leaves one class; predict a constant, do not crash."""
    features = dataset.features.iloc[:200]
    target = pd.Series(1.0, index=features.index)

    model = create_model("logistic")
    model.fit(features, target)
    predictions = model.predict(dataset.features.iloc[200:250])

    assert np.allclose(predictions, 1.0)


def test_zero_signal_predicts_nothing(dataset, training_data) -> None:
    model = ZeroSignal().fit(*training_data)
    assert np.allclose(model.predict(dataset.features.iloc[:50]), 0.0)


def test_momentum_and_reversion_are_mirror_images(dataset, training_data) -> None:
    momentum = MomentumBaseline().fit(*training_data)
    reversion = MeanReversionBaseline().fit(*training_data)
    features = dataset.features.iloc[400:450]

    assert np.allclose(momentum.predict(features), -reversion.predict(features))
    assert np.allclose(momentum.predict(features), features["ret_1d"].to_numpy())


def test_train_mean_baseline_predicts_the_training_mean(training_data) -> None:
    features, target = training_data
    model = create_model("train_mean").fit(features, target)
    assert np.allclose(model.predict(features), target.mean())


def test_baseline_reports_a_missing_feature_clearly(training_data) -> None:
    features, target = training_data
    model = MomentumBaseline(feature="not_a_feature")
    with pytest.raises(KeyError, match="not_a_feature"):
        model.fit(features, target)


def test_predicting_before_fitting_raises(dataset) -> None:
    model = build_sklearn_model("ridge")
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(dataset.features.iloc[:10])


def test_fit_rejects_bad_input(dataset) -> None:
    features = dataset.features.iloc[:100]
    model = build_sklearn_model("ridge")

    with pytest.raises(ValueError, match="lengths differ"):
        model.fit(features, dataset.forward_return.iloc[:50])
    with pytest.raises(ValueError, match="empty"):
        model.fit(features.iloc[:0], dataset.forward_return.iloc[:0])

    with_nan = dataset.forward_return.iloc[:100].copy()
    with_nan.iloc[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.fit(features, with_nan)


def test_feature_importance_aligns_with_feature_names(dataset, training_data) -> None:
    for name in ("ridge", "random_forest"):
        model = create_model(name).fit(*training_data)
        importance = model.feature_importance()
        assert importance is not None
        assert list(importance.index) == dataset.feature_names


def test_target_scaling_keeps_predictions_in_return_units(dataset, training_data) -> None:
    """Standardising y must not change the units of the prediction."""
    features, target = training_data
    model = create_model("ridge").fit(features, target)
    predictions = model.predict(features)

    # Predictions of daily log returns should live on the same tiny scale.
    assert np.abs(predictions).max() < 20 * target.abs().max()
    assert np.abs(np.mean(predictions) - target.mean()) < 5 * target.std()


def test_models_are_deterministic(dataset, training_data) -> None:
    features = dataset.features.iloc[400:450]
    first = create_model("random_forest").fit(*training_data).predict(features)
    second = create_model("random_forest").fit(*training_data).predict(features)
    assert np.array_equal(first, second)


def test_unknown_model_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        create_model("transformer_9000")
    with pytest.raises(ValueError, match="unknown scikit-learn model kind"):
        build_sklearn_model("xgboost")


def test_custom_models_can_be_registered(dataset, training_data) -> None:
    class Constant(Model):
        name = "constant_seven"

        def fit(self, features, target):
            return self

        def predict(self, features):
            return np.full(len(features), 7.0)

    register_model("constant_seven", Constant)
    try:
        model = create_model("constant_seven").fit(*training_data)
        assert np.allclose(model.predict(dataset.features.iloc[:5]), 7.0)
        with pytest.raises(ValueError, match="already registered"):
            register_model("constant_seven", Constant)
        register_model("constant_seven", Constant, overwrite=True)
    finally:
        from ai_stock.models.registry import _REGISTRY

        _REGISTRY.pop("constant_seven", None)


def test_model_kwargs_reach_the_estimator(training_data) -> None:
    features, target = training_data
    strong = create_model("ridge", alpha=1e6).fit(features, target)
    weak = create_model("ridge", alpha=1e-6).fit(features, target)

    # Heavier shrinkage must produce a flatter signal.
    assert np.std(strong.predict(features)) < np.std(weak.predict(features))
