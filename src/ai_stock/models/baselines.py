"""Baselines. A model that cannot beat these has not earned its complexity.

Deliberately trivial and mostly parameter-free, so that a walk-forward
comparison has an honest reference point rather than only other ML models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_stock.models.base import Model

__all__ = ["MeanReversionBaseline", "MomentumBaseline", "TrainMeanBaseline", "ZeroSignal"]


class ZeroSignal(Model):
    """Predict "no move". Stays flat, and is therefore the cost-free null."""

    name = "zero"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ZeroSignal:
        self._validate_fit_input(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


class TrainMeanBaseline(Model):
    """Predict the training-set mean target: a constant long (or short) tilt.

    Its backtest is buy-and-hold whenever the training mean is positive, which
    is the benchmark most "AI beats the market" claims quietly fail against.
    """

    name = "train_mean"

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> TrainMeanBaseline:
        self._validate_fit_input(features, target)
        self._mean = float(target.mean())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self._mean, dtype=float)


class _TrailingReturnBaseline(Model):
    """Shared machinery for the momentum / mean-reversion baselines."""

    _direction: float = 1.0

    def __init__(self, feature: str = "ret_1d", scale: float = 1.0) -> None:
        self.feature = feature
        self.scale = float(scale)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> _TrailingReturnBaseline:
        self._validate_fit_input(features, target)
        self._require_feature(features)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self._require_feature(features)
        values = features[self.feature].to_numpy(dtype=float)
        return self._direction * self.scale * values

    def _require_feature(self, features: pd.DataFrame) -> None:
        if self.feature not in features.columns:
            raise KeyError(
                f"{self.name} baseline needs the {self.feature!r} feature; "
                f"available columns start with {list(features.columns)[:5]}"
            )


class MomentumBaseline(_TrailingReturnBaseline):
    """Extrapolate the last move: signal = trailing return."""

    name = "momentum"
    _direction = 1.0


class MeanReversionBaseline(_TrailingReturnBaseline):
    """Bet against the last move: signal = -trailing return."""

    name = "reversion"
    _direction = -1.0
