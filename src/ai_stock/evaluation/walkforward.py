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

**How many folds is that, really.** The per-fold t-statistic divides by the
square root of a sample size, and the number of folds is only that sample size
when the folds do not share market. Two things in the schedule make them share
it: ``step`` below ``test_size`` overlaps the test windows outright, and a
``horizon``-day target makes even abutting windows share ``horizon - 1`` bars
of outcome. :meth:`WalkForwardResult.independent_folds` counts the fold-widths
of distinct market the schedule actually reaches, and ``ic_fold_t`` divides by
that; ``ic_fold_t_naive`` keeps the old per-fold count beside it.

**Hyper-parameters chosen inside the fold.** Passing a
:class:`~ai_stock.evaluation.tuning.TuningConfig` makes each fold select its
own hyper-parameters from an inner walk-forward over its *training* bars
alone, so the reported out-of-sample number describes a procedure a live run
could actually have followed. See that module for what this does and does not
remove; :meth:`WalkForwardResult.selection_stability` is the table to read
first.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ai_stock.config import WalkForwardConfig
from ai_stock.evaluation.metrics import classification_metrics, regression_metrics
from ai_stock.evaluation.tuning import TuningConfig, describe_params, inner_schedule
from ai_stock.features import indicators as ind
from ai_stock.features.builder import Dataset
from ai_stock.models.base import Model

__all__ = [
    "Fold",
    "FoldResult",
    "WalkForwardResult",
    "WalkForwardSplitter",
    "run_walk_forward",
]

ModelFactory = Callable[..., Model]
"""Builds a fresh model.

Called with no arguments when no tuning is configured, and with the winning
grid entry as keyword arguments when there is one - so a factory used with
:class:`~ai_stock.evaluation.tuning.TuningConfig` must accept the grid's keys.
"""


def _covered_window_ratio(starts: np.ndarray, ends: np.ndarray) -> float:
    """How many average-width windows the union of ``[start, end)`` spans covers.

    The generalisation of :func:`ai_stock.journal.independent_blocks` to windows
    that are wider than one recording step: there, ``n`` forecasts recorded one
    day apart against an ``h``-day horizon cover ``n / h`` blocks; here, folds
    recurring every ``step`` bars and each scoring ``width`` bars of outcome
    cover ``n * step / width`` of them. Stated as covered span over mean width
    so that uneven folds need no special case.

    The result is bounded by construction: the union can never exceed the sum
    of the widths, so the ratio never exceeds ``len(starts)``, and it can never
    fall below one, because the union is at least as wide as the widest window.

    Disjoint windows count once each:

    >>> _covered_window_ratio(np.array([0, 129, 258]), np.array([129, 258, 387]))
    3.0

    Windows that almost entirely coincide count as barely more than one:

    >>> round(_covered_window_ratio(np.array([0, 1, 2]), np.array([129, 130, 131])), 4)
    1.0155
    """
    if len(starts) == 0:
        return 0.0
    order = np.argsort(starts, kind="stable")
    covered = 0
    merged_start, merged_end = int(starts[order[0]]), int(ends[order[0]])
    for index in order[1:]:
        start, end = int(starts[index]), int(ends[index])
        if start > merged_end:
            covered += merged_end - merged_start
            merged_start, merged_end = start, end
        else:
            merged_end = max(merged_end, end)
    covered += merged_end - merged_start
    mean_width = float(np.mean(ends.astype(float) - starts.astype(float)))
    if mean_width <= 0.0:
        return float(len(starts))
    return float(covered) / mean_width


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
    params: dict[str, Any] = field(default_factory=dict)
    """Hyper-parameters this fold was fitted with; empty when none were selected."""
    selection: list[dict[str, Any]] = field(default_factory=list)
    """One record per candidate scored on this fold's inner validation folds."""


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
    feature_importance_std: pd.Series | None = None
    realised_volatility: pd.Series | None = None
    """Trailing annualised volatility at each prediction, for :meth:`regime_metrics`."""
    horizon: int = 1
    """Bars between a prediction and the price it is scored against.

    Kept on the result because :meth:`independent_folds` needs it: it is what
    makes two abutting test windows share outcome bars.
    """

    def __len__(self) -> int:
        return len(self.predictions)

    def feature_importance_stability(self) -> pd.DataFrame:
        """Per-feature importance mean, dispersion and coefficient of variation.

        ``feature_importance`` alone treats a feature that scores the same in
        every fold the same as one that swings from irrelevant to dominant and
        happens to average out the same - the two are different claims about
        how much to trust the feature. ``cv`` (std / |mean|) makes that
        difference visible; it is undefined (``NaN``) with fewer than two
        folds or a mean of zero.
        """
        if self.feature_importance is None:
            return pd.DataFrame(columns=["mean", "std", "cv"])
        mean = self.feature_importance
        std = (
            self.feature_importance_std
            if self.feature_importance_std is not None
            else pd.Series(float("nan"), index=mean.index)
        )
        cv = std / mean.abs().replace(0.0, np.nan)
        frame = pd.DataFrame({"mean": mean, "std": std, "cv": cv})
        return frame.reindex(mean.abs().sort_values(ascending=False).index)

    def selected_params(self) -> pd.DataFrame:
        """One row per fold, one column per hyper-parameter that was selected.

        Empty when the run used fixed hyper-parameters, which is the default -
        report code branches on ``.empty`` the same way it does for the
        feature-importance and regime tables.
        """
        rows = {fold.number: dict(fold.params) for fold in self.folds if fold.params}
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "fold"
        return frame

    def selection_stability(self) -> pd.DataFrame:
        """Per hyper-parameter: how often the inner selection changed its mind.

        The table to read before any performance number from a tuned run.
        ``n_distinct`` is how many different values won across the folds and
        ``modal_share`` how often the most common one did. A parameter that
        wins with a different value almost every fold is being selected on
        noise: the grid entries are indistinguishable at this sample size, and
        the honest claim becomes "this procedure, instability included, earns
        this much" rather than "these hyper-parameters work".

        Read it beside :meth:`selection_spread`, which asks the same question
        from the other direction and is kept separate because it is a property
        of the whole grid rather than of any one parameter.
        """
        params = self.selected_params()
        if params.empty:
            return pd.DataFrame()
        rows = []
        for name in params.columns:
            values = params[name]
            counts = values.astype(str).value_counts()
            rows.append(
                {
                    "parameter": name,
                    "n_distinct": int(counts.size),
                    "modal_value": str(counts.index[0]),
                    "modal_share": float(counts.iloc[0] / len(values)),
                }
            )
        return pd.DataFrame(rows).set_index("parameter")

    def selection_spread(self) -> float:
        """Best minus worst candidate mean inner score, averaged over folds.

        The size of the difference the selection is choosing between, and
        therefore how much the choice could possibly be worth. Near zero means
        the grid is flat: a stable winner over a flat grid is arbitrary rather
        than robust, which :meth:`selection_stability` on its own cannot say.

        It is one number for the whole grid, not one per parameter - the
        candidates are scored as combinations, so there is no per-parameter
        spread to report without attributing a joint difference to one axis.
        ``NaN`` when no fold scored two candidates finitely.
        """
        spreads = []
        for fold in self.folds:
            finite = [record["score"] for record in fold.selection if np.isfinite(record["score"])]
            if len(finite) > 1:
                spreads.append(max(finite) - min(finite))
        return float(np.mean(spreads)) if spreads else float("nan")

    def selection_scores(self) -> pd.DataFrame:
        """Every candidate's mean inner-validation score, one row per fold-candidate."""
        records = [record for fold in self.folds for record in fold.selection]
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def independent_folds(self) -> float:
        """How many fold-widths of distinct market the schedule actually reaches.

        ``ic_fold_t`` divides by the square root of a sample size, and the fold
        count is that sample size only when the folds do not share market.
        Two things in the schedule make them share it, and both are properties
        of the schedule rather than of the returns:

        **Overlapping test windows.** ``step`` defaults to ``test_size``, which
        makes consecutive test windows abut. Set it smaller and they overlap:
        at ``step=1`` every fold re-scores almost the same window, so twenty
        folds are one observation reported twenty times.
        :func:`run_walk_forward` already warns that this inflates the *pooled*
        sample size; the t-statistic across folds had the same problem and did
        not say so.

        **The label tail.** With a ``horizon``-day target the last bar of a
        test window is scored against a price ``horizon`` bars later, which
        falls inside the next window. Even abutting folds therefore share
        ``horizon - 1`` bars of outcome - the overlapping-signal limitation, at
        the fold boundary instead of at every bar.

        Each fold contributes the half-open span ``[test_start, test_end +
        horizon)``, and the count is the union of those spans over their mean
        width. Like :func:`ai_stock.journal.independent_blocks`, it estimates
        no correlation, so there is nothing here to tune or to get wrong; and
        the same way, it errs low - fold positions are taken from the pooled
        prediction calendar, which omits the bars between test windows when
        ``step > test_size``, so well-separated folds read as merely adjacent.

        On the default schedule the answer is close to the fold count: abutting
        windows of 125 bars sharing a 5-day tail lose about a third of a fold
        across ten of them. It is a small correction where ``step`` is left
        alone and a large one where it is not.
        """
        if not self.folds:
            return 0.0
        calendar = self.predictions.index
        starts = calendar.get_indexer([fold.test_start for fold in self.folds])
        ends = calendar.get_indexer([fold.test_end for fold in self.folds])
        if (starts < 0).any() or (ends < 0).any():
            return float(len(self.folds))
        return _covered_window_ratio(starts, ends + max(int(self.horizon), 1))

    def metrics(self) -> dict[str, float]:
        """Predictive metrics, pooled across folds and aggregated per fold.

        ``ic_pearson`` pools every out-of-sample prediction, while
        ``ic_fold_mean``/``ic_fold_t`` average the per-fold information
        coefficients. The per-fold view is the trustworthy one: it is immune to
        the regime-scale effect described in this module's docstring.
        ``ic_fold_t`` is the information ratio of the IC,
        ``mean / std * sqrt(ic_fold_n_eff)``, where the sample size is
        :meth:`independent_folds` rather than the raw fold count; roughly
        ``|t| > 2`` is the usual bar for taking a measured edge seriously.
        """
        summary = regression_metrics(self.forward_return, self.predictions)
        summary.update(classification_metrics(self.direction, self.predictions))
        summary["n_folds"] = float(len(self.folds))
        summary.update(self.fold_summary())
        return summary

    def fold_summary(self) -> dict[str, float]:
        """Aggregate the per-fold information coefficients.

        ``ic_fold_t`` is measured at :meth:`independent_folds`, and
        ``ic_fold_t_naive`` at the raw fold count - the figure this summary
        used to report on its own. The pair brackets the honest answer the
        same way ``hit_rate_z`` and ``hit_rate_z_naive`` do in the forecast
        journal, and while they disagree, believe the smaller.
        """
        empty = {
            "ic_fold_mean": float("nan"),
            "ic_fold_std": float("nan"),
            "ic_fold_n_eff": float("nan"),
            "ic_fold_t": float("nan"),
            "ic_fold_t_naive": float("nan"),
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
        # Folds whose IC could not be computed are not folds this t-statistic
        # rests on, so the effective count is scaled down by the same fraction
        # the dropna() removed rather than counting windows nothing scored.
        n_eff = self.independent_folds() * len(values) / len(frame)
        usable = deviation and np.isfinite(deviation) and deviation > 1e-12
        return {
            "ic_fold_mean": mean,
            "ic_fold_std": deviation,
            "ic_fold_n_eff": float(n_eff),
            "ic_fold_t": float(mean / deviation * np.sqrt(n_eff)) if usable else float("nan"),
            "ic_fold_t_naive": (
                float(mean / deviation * np.sqrt(len(values))) if usable else float("nan")
            ),
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

    def regime_metrics(self, n_bins: int = 3) -> pd.DataFrame:
        """Predictive metrics split by realised-volatility bin (terciles by default).

        A model whose information coefficient only shows up in the calm bin is
        a materially weaker claim than one that holds across every regime -
        pooling every bar together, as :meth:`metrics` does, cannot tell the
        two apart. Bins are quantiles of :attr:`realised_volatility` at each
        prediction, so each one holds roughly the same number of bars.
        """
        if self.realised_volatility is None:
            return pd.DataFrame()
        volatility = self.realised_volatility.dropna()
        if len(volatility.unique()) < 2:
            return pd.DataFrame()

        bins = pd.qcut(volatility, min(n_bins, len(volatility.unique())), duplicates="drop")
        labels = _regime_labels(len(bins.cat.categories))
        bins = bins.cat.rename_categories(labels)

        rows = []
        for label in labels:
            index = bins[bins == label].index
            forward_return = self.forward_return.reindex(index)
            predictions = self.predictions.reindex(index)
            row = regression_metrics(forward_return, predictions)
            row.update(classification_metrics(self.direction.reindex(index), predictions))
            row["realised_vol_mean"] = float(volatility.reindex(index).mean())
            row["regime"] = label
            rows.append(row)
        return pd.DataFrame(rows).set_index("regime")


def _regime_labels(n_bins: int) -> list[str]:
    if n_bins == 3:
        return ["low_vol", "mid_vol", "high_vol"]
    return [f"vol_q{i + 1}_of_{n_bins}" for i in range(n_bins)]


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


def _score_candidate(
    train: Dataset,
    inner_folds: list[Fold],
    factory: ModelFactory,
    params: dict[str, Any],
    tuning: TuningConfig,
) -> float:
    """Mean inner-validation score for one candidate, or ``NaN`` if it never fitted.

    A candidate that raises is scored ``NaN`` rather than aborting the run:
    a grid written by hand will contain combinations some estimator rejects,
    and :meth:`TuningConfig.better` already refuses to let a non-finite score
    win. ``ValueError`` is the only exception caught, because that is what
    :func:`_fit_one_fold` and scikit-learn raise for a rejected parameter -
    anything else is a bug and should surface.
    """
    scores: list[float] = []
    for inner in inner_folds:
        try:
            values, _ = _fit_one_fold(factory(**params), train, inner)
        except ValueError:
            return float("nan")
        test = train.slice(inner.test)
        predictions = pd.Series(values, index=test.index)
        metrics = regression_metrics(test.forward_return, predictions)
        metrics.update(classification_metrics(test.direction, predictions))
        scores.append(float(metrics.get(tuning.metric, float("nan"))))
    finite = [value for value in scores if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _select_params(
    dataset: Dataset,
    fold: Fold,
    factory: ModelFactory,
    tuning: TuningConfig,
    embargo: int,
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose hyper-parameters from this fold's training bars alone.

    The inner schedule is an ordinary :class:`WalkForwardSplitter` run over
    the training slice, so the embargo between inner train and inner
    validation is the same code path the outer loop uses. Nothing outside
    ``fold.train`` is read, which is what makes the outer test window still
    out-of-sample after selection.

    Returns the winning parameters and one record per candidate. When the
    training window is too short for the requested inner schedule, or no
    candidate scored finitely, the winner is the empty dict - the model's own
    defaults - and the caller reports that no selection happened rather than
    inventing one.
    """
    train = dataset.slice(fold.train)
    schedule = inner_schedule(len(train), embargo, tuning)
    if schedule is None:
        return {}, []

    train_size, test_size = schedule
    inner_config = WalkForwardConfig(
        train_size=train_size,
        test_size=test_size,
        step=test_size,
        expanding=True,
        embargo=embargo,
    )
    inner_folds = list(WalkForwardSplitter(inner_config).split(len(train), horizon))
    if not inner_folds:
        return {}, []

    best_params: dict[str, Any] = {}
    best_score = float("nan")
    best_index = -1
    records: list[dict[str, Any]] = []
    for index, params in enumerate(tuning.candidates()):
        score = _score_candidate(train, inner_folds, factory, params, tuning)
        records.append(
            {
                "fold": fold.number,
                "params": describe_params(params),
                "score": score,
                "chosen": False,
            }
        )
        if tuning.better(score, best_score):
            best_score, best_params, best_index = score, params, index

    if best_index < 0:
        return {}, []
    # Marked by position rather than by label: a grid that repeats a value
    # gives two candidates the same label, and exactly one of them won.
    records[best_index]["chosen"] = True
    return best_params, records


def run_walk_forward(
    dataset: Dataset,
    model_factory: ModelFactory | str,
    config: WalkForwardConfig | None = None,
    *,
    model_kwargs: dict | None = None,
    regime_vol_window: int = 20,
    tuning: TuningConfig | None = None,
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
    regime_vol_window:
        Trailing window (bars) for the realised volatility that
        :meth:`WalkForwardResult.regime_metrics` bins predictions by. Computed
        on ``dataset.close`` before folding, so it stays causal and free of the
        gaps between test windows.
    tuning:
        When set and non-trivial, each fold selects its hyper-parameters from
        an inner walk-forward over its own training bars instead of using the
        fixed ones. Multiplies the fit count by
        ``len(grid) * n_inner_folds + 1``; see
        :mod:`ai_stock.evaluation.tuning` for what it buys.

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

        def factory(**params: Any) -> Model:
            return create_model(name, **{**kwargs, **params})
    else:
        given = model_factory

        def factory(**params: Any) -> Model:
            return given(**params) if params else given()

    splitter = WalkForwardSplitter(config)
    folds = list(splitter.split(len(dataset), dataset.horizon))
    if not folds:
        raise ValueError("the walk-forward schedule produced no folds")

    select = tuning is not None and tuning.enabled
    embargo = (config or WalkForwardConfig()).resolved_embargo(dataset.horizon)

    probe = factory()
    fold_results: list[FoldResult] = []
    prediction_chunks: list[pd.Series] = []
    importances: list[pd.Series] = []

    for fold in folds:
        params, selection = (
            _select_params(dataset, fold, factory, tuning, embargo, dataset.horizon)
            if select and tuning is not None
            else ({}, [])
        )
        model = factory(**params)
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
                params=params,
                selection=selection,
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

    if importances:
        importance_frame = pd.concat(importances, axis=1)
        mean_importance = importance_frame.mean(axis=1).rename(f"{probe.name}_importance")
        std_importance = importance_frame.std(axis=1, ddof=1).rename(f"{probe.name}_importance_std")
    else:
        mean_importance = None
        std_importance = None

    volatility = ind.realised_volatility(dataset.close, regime_vol_window).reindex(pooled.index)

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
        feature_importance_std=std_importance,
        realised_volatility=volatility,
        horizon=int(dataset.horizon),
    )
