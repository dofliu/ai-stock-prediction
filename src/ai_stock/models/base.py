"""The single model interface the rest of the pipeline depends on."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

__all__ = ["Model", "SignalUnits"]

SignalUnits = str
"""Either ``"return"`` (expected log return) or ``"edge"`` (probability edge)."""


class Model(ABC):
    """A forecaster that maps a feature matrix to a trading signal.

    Implementations only need :meth:`fit` and :meth:`predict`. Two attributes
    tell the rest of the pipeline how to treat the model:

    ``is_classifier``
        When ``True`` the walk-forward runner feeds the ``direction`` target
        (``{0, 1}``) instead of the forward return.
    ``signal_units``
        ``"return"`` means :meth:`predict` outputs an expected log return;
        ``"edge"`` means it outputs ``2 * P(up) - 1`` in ``[-1, 1]``. Sign-based
        position sizing is unaffected, but thresholds are read in these units.
    """

    name: str = "model"
    is_classifier: bool = False
    signal_units: SignalUnits = "return"

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.Series) -> Model:
        """Fit on a training slice and return ``self``."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict a signal for every row of ``features``."""

    def feature_importance(self) -> pd.Series | None:
        """Per-feature importance if the model exposes one, else ``None``."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"

    @staticmethod
    def _validate_fit_input(features: pd.DataFrame, target: pd.Series) -> None:
        if len(features) != len(target):
            raise ValueError(
                f"features and target lengths differ: {len(features)} vs {len(target)}"
            )
        if len(features) == 0:
            raise ValueError("cannot fit on an empty training set")
        if target.isna().any():
            raise ValueError("target contains NaN; drop those rows before fitting")
