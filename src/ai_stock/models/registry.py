"""Name -> model factory registry, used by the CLI and the comparison runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_stock.models.base import Model
from ai_stock.models.baselines import (
    MeanReversionBaseline,
    MomentumBaseline,
    TrainMeanBaseline,
    ZeroSignal,
)
from ai_stock.models.sklearn_models import build_sklearn_model

__all__ = ["MODEL_NAMES", "available_models", "create_model", "register_model"]

ModelFactory = Callable[..., Model]

_SKLEARN_KINDS = (
    "ridge",
    "elastic_net",
    "logistic",
    "random_forest",
    "random_forest_classifier",
    "gradient_boosting",
    "gradient_boosting_classifier",
    "mlp",
)

_REGISTRY: dict[str, ModelFactory] = {
    "zero": ZeroSignal,
    "train_mean": TrainMeanBaseline,
    "momentum": MomentumBaseline,
    "reversion": MeanReversionBaseline,
    **{kind: (lambda _k=kind, **kw: build_sklearn_model(_k, **kw)) for kind in _SKLEARN_KINDS},
}

MODEL_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def available_models() -> tuple[str, ...]:
    """Names accepted by :func:`create_model`, in registration order."""
    return tuple(_REGISTRY)


def create_model(name: str, **kwargs: Any) -> Model:
    """Instantiate a model by registry name.

    >>> create_model("zero").name
    'zero'
    >>> create_model("ridge", alpha=1.0).name
    'ridge'
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; available: {', '.join(available_models())}"
        ) from None
    return factory(**kwargs)


def register_model(name: str, factory: ModelFactory, *, overwrite: bool = False) -> None:
    """Add a custom model factory to the registry."""
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"model {name!r} is already registered; pass overwrite=True to replace")
    _REGISTRY[name] = factory
