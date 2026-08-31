"""Walk-forward (rolling-origin) evaluation.

A single train/test split on financial data is close to worthless: it measures
one arbitrary market period. Walk-forward instead retrains as time advances and
concatenates the out-of-sample predictions, which is both a better estimate of
live performance and the only way to see a strategy decay.

Two details matter more than the loop itself:

**Embargo.** With a horizon of ``h`` days, the label of the last training bar
depends on prices ``h`` days later - which fall inside the test window. The
splitter therefore skips ``h`` bars between train and test (Lopez de Prado's
purging/embargo), so nothing in training has seen a price the test set uses.

**A fresh model per fold.** Each fold instantiates the model from a factory, so
no state, scaler or fitted parameter leaks backwards in time.

**Pooled vs. per-fold metrics.** Concatenating every fold's predictions and
correlating once is tempting but misleading: folds sit in different volatility
regimes, so the *between-fold* variation in means can dominate and even flip
the sign of the correlation while most individual folds are positive - a
textbook Simpson's paradox. :meth:`WalkForwardResult.metrics` therefore reports
the pooled IC *and* the per-fold IC mean, its dispersion and a t-statistic.
When the two disagree, believe the per-fold numbers.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_stock.config import WalkForwardConfig
from ai_stock.evaluation.metrics import classification_metrics, regression_metrics
from ai_stock.features.builder import Dataset
from ai_stock.models.base import Model

__all__ = [
    "Fold",
    "FoldResult",
    "WalkForwardResult",
    "WalkForwardSplitter",
    "run_walk_forward",
]

ModelFactory = Callable[[], Model]


@dataclass(frozen=True)
class Fold:
    """Positional train/test indices for one step of the walk."""

    number: int
    train: np.ndarray
    test: np.ndarray

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)


@dataclass(frozen=True)
class FoldResult:
    """Per-fold predictions and their predictive metrics."""

    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    predictions: pd.Series
    metrics: dict[str, float]


@dataclass
class WalkForwardResult:
    """Concatenated out-of-sample predictions plus per-fold diagnostics."""

    model_name: str
    is_classifier: bool
    signal_units: str
    predictions: pd.Series
    forward_return: pd.Series
    direction: pd.Series
    close: pd.Series
    folds: list[FoldResult] = field(default_factory=list)
    feature_importance: pd.Series | None = None

    def __len__(self) -> int:
        return len(self.predictions)

    def metrics(self) -> dict[str, float]:
        """Predictive metrics, pooled across folds and aggregated per fold.

        ``ic_pearson`` pools every out-of-sample prediction, while
        ``ic_fold_mean``/``ic_fold_t`` average the per-fold information
        coefficients. The per-fold view is the trustworthy one: it is immune to
        the regime-scale effect described in this module's docstring.
        ``ic_fold_t`` is the classic information ratio of the IC,
        ``mean / std * sqrt(n_folds)``; roughly ``|t| > 2`` is the usual bar
        for taking a measured edge seriously.
        """
        summary = regression_metrics(self.forward_return, self.predictions)
        summary.update(classification_metrics(self.direction, self.predictions))
        summary["n_folds"] = float(len(self.folds))
        summary.update(self.fold_summary())
        return summary

    def fold_summary(self) -> dict[str, float]:
        """Aggregate the per-fold information coefficients."""
        empty = {
            "ic_fold_mean": float("nan"),
            "ic_fold_std": float("nan"),
            "ic_fold_t": float("nan"),
            "ic_fold_positive_rate": float("nan"),
        }
        frame = self.fold_metrics()
        if frame.empty or "ic_pearson" not in frame:
            return empty
        values = frame["ic_pearson"].dropna()
        if values.empty:
            return empty

        mean = float(values.mean())
        deviation = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        t_stat = (
            mean / deviation * np.sqrt(len(values))
            if deviation and np.isfinite(deviation) and deviation > 1e-12
            else float("nan")
        )
        return {
            "ic_fold_mean": mean,
            "ic_fold_std": deviation,
            "ic_fold_t": float(t_stat),
            "ic_fold_positive_rate": float((values > 0).mean()),
        }

    def fold_metrics(self) -> pd.DataFrame:
        """One row per fold: the stability check the pooled numbers hide."""
        if not self.folds:
            return pd.DataFrame()
        rows = [
            {
                "fold": fold.number,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                **fold.metrics,
            }
            for fold in self.folds
        ]
        return pd.DataFrame(rows).set_index("fold")


class WalkForwardSplitter:
    """Generate expanding or rolling train/test folds with an embargo gap."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def split(self, n_samples: int, horizon: int = 1) -> Iterator[Fold]:
        """Yield folds over ``n_samples`` bars for a ``horizon``-day target.

        Raises
        ------
        ValueError
            If the sample is too short to produce even one fold, with the
            minimum length spelled out.
        """
        cfg = self.config
        embargo = cfg.resolved_embargo(horizon)
        step = cfg.resolved_step()

        minimum = cfg.train_size + embargo + 1
        if n_samples < minimum:
            raise ValueError(
                f"need at least {minimum} bars for a walk-forward split "
                f"(train_size={cfg.train_size} + embargo={embargo} + 1), got {n_samples}"
            )

        number = 0
        train_end = cfg.train_size
        while True:
            test_start = train_end + embargo
            if test_start >= n_samples:
                break
            test_end = min(test_start + cfg.test_size, n_samples)
            train_start = 0 if cfg.expanding else max(0, train_end - cfg.train_size)

            if train_end - train_start >= cfg.resolved_min_train_size():
                yield Fold(
                    number=number,
                    train=np.arange(train_start, train_end),
                    test=np.arange(test_start, test_end),
                )
                number += 1

            if test_end >= n_samples:
                break
            train_end += step

    def n_splits(self, n_samples: int, horizon: int = 1) -> int:
        """Number of folds :meth:`split` would produce."""
        return sum(1 for _ in self.split(n_samples, horizon))


def _fit_one_fold(
    model: Model, dataset: Dataset, fold: Fold
) -> tuple[np.ndarray, pd.Series | None]:
    """Fit ``model`` on the fold's training slice and predict its test slice."""
    train = dataset.slice(fold.train)
    test = dataset.slice(fold.test)

    target = train.target(classification=model.is_classifier)
    usable = target.notna().to_numpy()
    if usable.sum() < 2:
        raise ValueError(
            f"fold {fold.number} has {int(usable.sum())} usable training labels; "
            "widen train_size or reduce the neutral band"
        )

    model.fit(train.features.loc[usable], target[usable])
    predictions = np.asarray(model.predict(test.features), dtype=float)
    if len(predictions) != len(test.features):
        raise ValueError(
            f"{model.name} returned {len(predictions)} predictions "
            f"for {len(test.features)} test rows"
        )
    return predictions, model.feature_importance()


def run_walk_forward(
    dataset: Dataset,
    model_factory: ModelFactory | str,
    config: WalkForwardConfig | None = None,
    *,
    model_kwargs: dict | None = None,
) -> WalkForwardResult:
    """Run a full walk-forward evaluation and pool the out-of-sample forecasts.

    Parameters
    ----------
    dataset:
        Aligned features/targets from :func:`ai_stock.features.build_dataset`.
    model_factory:
        A zero-argument callable returning a fresh :class:`Model`, or a name
        from the model registry.
    config:
        Fold schedule; defaults to :class:`WalkForwardConfig`.
    model_kwargs:
        Extra arguments when ``model_factory`` is given as a registry name.

    Notes
    -----
    If ``step < test_size`` the test windows overlap; predictions for a shared
    bar are averaged and a warning is emitted, because overlapping folds make
    the pooled sample size look larger than the independent information in it.
    """
    if isinstance(model_factory, str):
        from ai_stock.models.registry import create_model  # noqa: PLC0415 - avoids a cycle

        name = model_factory
        kwargs = model_kwargs or {}

        def factory() -> Model:
            return create_model(name, **kwargs)
    else:
        factory = model_factory

    splitter = WalkForwardSplitter(config)
    folds = list(splitter.split(len(dataset), dataset.horizon))
    if not folds:
        raise ValueError("the walk-forward schedule produced no folds")

    probe = factory()
    fold_results: list[FoldResult] = []
    prediction_chunks: list[pd.Series] = []
    importances: list[pd.Series] = []

    for fold in folds:
        model = factory()
        values, importance = _fit_one_fold(model, dataset, fold)

        test = dataset.slice(fold.test)
        predictions = pd.Series(values, index=test.index, name=f"{model.name}_signal")
        prediction_chunks.append(predictions)
        if importance is not None:
            importances.append(importance)

        metrics = regression_metrics(test.forward_return, predictions)
        metrics.update(classification_metrics(test.direction, predictions))

        train_index = dataset.index[fold.train]
        fold_results.append(
            FoldResult(
                number=fold.number,
                train_start=train_index[0],
                train_end=train_index[-1],
                test_start=test.index[0],
                test_end=test.index[-1],
                n_train=fold.train_size,
                n_test=fold.test_size,
                predictions=predictions,
                metrics=metrics,
            )
        )

    pooled = pd.concat(prediction_chunks)
    if pooled.index.has_duplicates:
        warnings.warn(
            "overlapping walk-forward test windows: predictions for shared bars were "
            "averaged; set step >= test_size for independent folds",
            UserWarning,
            stacklevel=2,
        )
        pooled = pooled.groupby(level=0).mean()
    pooled = pooled.sort_index()
    pooled.name = f"{probe.name}_signal"

    mean_importance = (
        pd.concat(importances, axis=1).mean(axis=1).rename(f"{probe.name}_importance")
        if importances
        else None
    )

    return WalkForwardResult(
        model_name=probe.name,
        is_classifier=probe.is_classifier,
        signal_units=probe.signal_units,
        predictions=pooled,
        forward_return=dataset.forward_return.reindex(pooled.index),
        direction=dataset.direction.reindex(pooled.index),
        close=dataset.close.reindex(pooled.index),
        folds=fold_results,
        feature_importance=mean_importance,
    )
