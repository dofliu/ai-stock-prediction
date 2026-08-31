"""Models: one small interface, baselines and scikit-learn estimators."""

from ai_stock.models.base import Model
from ai_stock.models.baselines import (
    MeanReversionBaseline,
    MomentumBaseline,
    TrainMeanBaseline,
    ZeroSignal,
)
from ai_stock.models.registry import (
    MODEL_NAMES,
    available_models,
    create_model,
    register_model,
)
from ai_stock.models.sklearn_models import (
    SklearnClassifier,
    SklearnRegressor,
    build_sklearn_model,
)

__all__ = [
    "MODEL_NAMES",
    "MeanReversionBaseline",
    "Model",
    "MomentumBaseline",
    "SklearnClassifier",
    "SklearnRegressor",
    "TrainMeanBaseline",
    "ZeroSignal",
    "available_models",
    "build_sklearn_model",
    "create_model",
    "register_model",
]
