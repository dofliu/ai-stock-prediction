"""Thin, uniform wrappers around scikit-learn estimators.

Each wrapper keeps two habits that matter for time-series work:

* the scaler is fitted **inside** each walk-forward fold via a pipeline, so
  test-fold statistics never leak into training;
* regression targets are standardised the same way, which makes penalties and
  learning rates scale-free - daily log returns are ~1e-2, small enough that
  an unscaled neural net or ridge silently mis-tunes;
* ``random_state`` is always pinned, so a reported Sharpe ratio is a property
  of the method rather than of the seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_stock.models.base import Model

__all__ = ["SklearnClassifier", "SklearnRegressor", "build_sklearn_model"]


def _maybe_scale(estimator: Any, scale: bool) -> Any:
    """Wrap ``estimator`` in a scaling pipeline when ``scale`` is set."""
    if not scale:
        return estimator
    return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])


def _final_estimator(fitted: Any) -> Any:
    """Peel off target/feature-transform wrappers to reach the real estimator."""
    if isinstance(fitted, TransformedTargetRegressor):
        fitted = fitted.regressor_
    return fitted.named_steps["estimator"] if isinstance(fitted, Pipeline) else fitted


class _SklearnModel(Model):
    """Common fit/importance behaviour for the scikit-learn wrappers."""

    _scale_target = False

    def __init__(self, name: str, estimator: Any, *, scale: bool = True) -> None:
        self.name = name
        self._template = estimator
        self._scale = scale
        self._fitted: Any | None = None
        self._feature_names: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> _SklearnModel:
        self._validate_fit_input(features, target)
        from sklearn.base import clone  # noqa: PLC0415 - keeps import cost off module load

        self._feature_names = list(features.columns)
        estimator = _maybe_scale(clone(self._template), self._scale)
        if self._scale_target:
            estimator = TransformedTargetRegressor(
                regressor=estimator, transformer=StandardScaler()
            )
        self._fitted = estimator
        self._fitted.fit(features, self._prepare_target(target))
        return self

    def _prepare_target(self, target: pd.Series) -> np.ndarray:
        return target.to_numpy(dtype=float)

    def _require_fitted(self, features: pd.DataFrame) -> Any:
        if self._fitted is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        missing = [c for c in self._feature_names if c not in features.columns]
        if missing:
            raise ValueError(f"{self.name}: features missing at predict time: {missing}")
        return self._fitted

    def feature_importance(self) -> pd.Series | None:
        if self._fitted is None:
            return None
        estimator = _final_estimator(self._fitted)
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float)
            values = coef.ravel() if coef.ndim > 1 else coef
        else:
            return None
        if len(values) != len(self._feature_names):
            return None
        return pd.Series(values, index=self._feature_names, name=f"{self.name}_importance")


class SklearnRegressor(_SklearnModel):
    """Regress the forward log return; the prediction *is* the signal.

    The target is standardised per fold and predictions are transformed back,
    so the output stays in log-return units while hyper-parameters do not have
    to be re-tuned for each price series.
    """

    is_classifier = False
    signal_units = "return"
    _scale_target = True

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        fitted = self._require_fitted(features)
        return np.asarray(fitted.predict(features[self._feature_names]), dtype=float).ravel()


class SklearnClassifier(_SklearnModel):
    """Classify direction; the signal is the probability edge ``2p - 1``.

    A fold can legitimately contain a single class (a relentless bull run, say).
    Rather than crashing, the wrapper falls back to a constant signal of the
    observed class - which is the honest answer for that fold.
    """

    is_classifier = True
    signal_units = "edge"

    def __init__(self, name: str, estimator: Any, *, scale: bool = True) -> None:
        super().__init__(name, estimator, scale=scale)
        self._single_class: float | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> SklearnClassifier:
        self._validate_fit_input(features, target)
        classes = np.unique(target.to_numpy())
        if len(classes) < 2:
            self._feature_names = list(features.columns)
            self._single_class = float(classes[0])
            self._fitted = None
            return self
        self._single_class = None
        super().fit(features, target)
        return self

    def _prepare_target(self, target: pd.Series) -> np.ndarray:
        return target.to_numpy(dtype=int)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._single_class is not None:
            edge = 1.0 if self._single_class > 0 else -1.0
            return np.full(len(features), edge, dtype=float)
        fitted = self._require_fitted(features)
        matrix = features[self._feature_names]
        if hasattr(fitted, "predict_proba"):
            probabilities = np.asarray(fitted.predict_proba(matrix), dtype=float)
            up_index = int(np.argmax(_final_estimator(fitted).classes_))
            return 2.0 * probabilities[:, up_index] - 1.0
        scores = np.asarray(fitted.decision_function(matrix), dtype=float).ravel()
        return np.tanh(scores)

    def predict_proba_up(self, features: pd.DataFrame) -> np.ndarray:
        """Probability of an up move, in ``[0, 1]``."""
        return (self.predict(features) + 1.0) / 2.0


def build_sklearn_model(kind: str, random_state: int = 0, **kwargs: Any) -> Model:
    """Construct one of the supported scikit-learn models by short name.

    Supported kinds: ``ridge``, ``elastic_net``, ``logistic``,
    ``random_forest``, ``random_forest_classifier``, ``gradient_boosting``,
    ``gradient_boosting_classifier``, ``mlp``. Extra keyword arguments are
    forwarded to the underlying estimator.
    """
    if kind == "ridge":
        return SklearnRegressor("ridge", Ridge(alpha=kwargs.pop("alpha", 50.0), **kwargs))
    if kind == "elastic_net":
        return SklearnRegressor(
            "elastic_net",
            ElasticNet(
                alpha=kwargs.pop("alpha", 0.05),
                l1_ratio=kwargs.pop("l1_ratio", 0.5),
                random_state=random_state,
                **kwargs,
            ),
        )
    if kind == "logistic":
        return SklearnClassifier(
            "logistic",
            LogisticRegression(
                C=kwargs.pop("C", 0.1),
                max_iter=kwargs.pop("max_iter", 1000),
                random_state=random_state,
                **kwargs,
            ),
        )
    if kind == "random_forest":
        return SklearnRegressor(
            "random_forest",
            RandomForestRegressor(
                n_estimators=kwargs.pop("n_estimators", 300),
                max_depth=kwargs.pop("max_depth", 5),
                min_samples_leaf=kwargs.pop("min_samples_leaf", 20),
                max_features=kwargs.pop("max_features", "sqrt"),
                n_jobs=kwargs.pop("n_jobs", 1),
                random_state=random_state,
                **kwargs,
            ),
            scale=False,
        )
    if kind == "random_forest_classifier":
        return SklearnClassifier(
            "random_forest_classifier",
            RandomForestClassifier(
                n_estimators=kwargs.pop("n_estimators", 300),
                max_depth=kwargs.pop("max_depth", 5),
                min_samples_leaf=kwargs.pop("min_samples_leaf", 20),
                max_features=kwargs.pop("max_features", "sqrt"),
                n_jobs=kwargs.pop("n_jobs", 1),
                random_state=random_state,
                **kwargs,
            ),
            scale=False,
        )
    if kind == "gradient_boosting":
        return SklearnRegressor(
            "gradient_boosting",
            HistGradientBoostingRegressor(
                max_depth=kwargs.pop("max_depth", 3),
                learning_rate=kwargs.pop("learning_rate", 0.03),
                max_iter=kwargs.pop("max_iter", 200),
                min_samples_leaf=kwargs.pop("min_samples_leaf", 40),
                l2_regularization=kwargs.pop("l2_regularization", 1.0),
                early_stopping=kwargs.pop("early_stopping", False),
                random_state=random_state,
                **kwargs,
            ),
            scale=False,
        )
    if kind == "gradient_boosting_classifier":
        return SklearnClassifier(
            "gradient_boosting_classifier",
            HistGradientBoostingClassifier(
                max_depth=kwargs.pop("max_depth", 3),
                learning_rate=kwargs.pop("learning_rate", 0.03),
                max_iter=kwargs.pop("max_iter", 200),
                min_samples_leaf=kwargs.pop("min_samples_leaf", 40),
                l2_regularization=kwargs.pop("l2_regularization", 1.0),
                early_stopping=kwargs.pop("early_stopping", False),
                random_state=random_state,
                **kwargs,
            ),
            scale=False,
        )
    if kind == "mlp":
        return SklearnRegressor(
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=kwargs.pop("hidden_layer_sizes", (32, 16)),
                alpha=kwargs.pop("alpha", 0.1),
                learning_rate_init=kwargs.pop("learning_rate_init", 1e-3),
                max_iter=kwargs.pop("max_iter", 1000),
                early_stopping=kwargs.pop("early_stopping", False),
                random_state=random_state,
                **kwargs,
            ),
        )
    raise ValueError(f"unknown scikit-learn model kind: {kind!r}")
