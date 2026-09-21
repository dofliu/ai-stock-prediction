"""Walk-forward splitter and runner tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ai_stock.config import FeatureConfig, WalkForwardConfig
from ai_stock.evaluation.walkforward import (
    WalkForwardResult,
    WalkForwardSplitter,
    run_walk_forward,
)
from ai_stock.features.builder import build_dataset
from ai_stock.models.base import Model


class CountingModel(Model):
    """Records how often it was fitted, to prove folds get fresh instances."""

    name = "counting"
    instances: list[CountingModel] = []

    def __init__(self) -> None:
        self.fit_calls = 0
        self.train_sizes: list[int] = []
        CountingModel.instances.append(self)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> CountingModel:
        self.fit_calls += 1
        self.train_sizes.append(len(features))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features))


def test_train_and_test_never_overlap() -> None:
    splitter = WalkForwardSplitter(WalkForwardConfig(train_size=100, test_size=25))
    for fold in splitter.split(400, horizon=1):
        assert set(fold.train).isdisjoint(set(fold.test))
        assert fold.train.max() < fold.test.min()


@pytest.mark.parametrize("horizon", [1, 3, 10])
def test_embargo_defaults_to_the_forecast_horizon(horizon: int) -> None:
    splitter = WalkForwardSplitter(WalkForwardConfig(train_size=100, test_size=25))
    for fold in splitter.split(500, horizon=horizon):
        gap = int(fold.test.min() - fold.train.max())
        # A gap of `horizon` bars means `horizon + 1` positions of separation.
        assert gap == horizon + 1


def test_explicit_embargo_overrides_the_horizon() -> None:
    splitter = WalkForwardSplitter(WalkForwardConfig(train_size=100, test_size=25, embargo=7))
    fold = next(iter(splitter.split(400, horizon=1)))
    assert int(fold.test.min() - fold.train.max()) == 8


def test_expanding_window_grows_and_rolling_window_does_not() -> None:
    expanding = list(
        WalkForwardSplitter(WalkForwardConfig(train_size=100, test_size=25)).split(400)
    )
    rolling = list(
        WalkForwardSplitter(WalkForwardConfig(train_size=100, test_size=25, expanding=False)).split(
            400
        )
    )

    assert expanding[0].train_size < expanding[-1].train_size
    assert all(fold.train[0] == 0 for fold in expanding)
    assert {fold.train_size for fold in rolling} == {100}


def test_step_controls_the_advance() -> None:
    folds = list(
        WalkForwardSplitter(WalkForwardConfig(train_size=100, test_size=50, step=25)).split(400)
    )
    starts = [int(fold.test.min()) for fold in folds]
    assert np.all(np.diff(starts) == 25)


def test_folds_cover_the_sample_without_running_past_the_end() -> None:
    n = 383
    folds = list(WalkForwardSplitter(WalkForwardConfig(train_size=100, test_size=50)).split(n))
    assert folds[-1].test.max() == n - 1
    assert all(fold.test.max() < n for fold in folds)


def test_too_short_a_sample_is_rejected_with_a_useful_message() -> None:
    splitter = WalkForwardSplitter(WalkForwardConfig(train_size=100, test_size=25))
    with pytest.raises(ValueError, match="at least 102 bars"):
        list(splitter.split(50, horizon=1))


def test_n_splits_matches_the_generator() -> None:
    splitter = WalkForwardSplitter(WalkForwardConfig(train_size=100, test_size=25))
    assert splitter.n_splits(400) == len(list(splitter.split(400)))


def test_min_train_size_skips_undersized_folds() -> None:
    config = WalkForwardConfig(train_size=50, test_size=25, expanding=False, min_train_size=200)
    assert list(WalkForwardSplitter(config).split(400)) == []


def test_each_fold_gets_a_fresh_model(dataset) -> None:
    CountingModel.instances = []
    config = WalkForwardConfig(train_size=300, test_size=100, min_train_size=100)

    result = run_walk_forward(dataset, CountingModel, config)

    fitted = [model for model in CountingModel.instances if model.fit_calls > 0]
    assert len(fitted) == len(result.folds)
    assert all(model.fit_calls == 1 for model in fitted)
    # Expanding window: each fold trains on strictly more data than the last.
    sizes = [model.train_sizes[0] for model in fitted]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_predictions_are_out_of_sample_and_ordered(dataset, walk_forward_config) -> None:
    result = run_walk_forward(dataset, "ridge", walk_forward_config)

    assert result.predictions.index.is_monotonic_increasing
    assert not result.predictions.index.has_duplicates
    assert result.predictions.index.isin(dataset.index).all()
    assert len(result.predictions) == sum(fold.n_test for fold in result.folds)
    assert result.forward_return.index.equals(result.predictions.index)


def test_overlapping_test_windows_warn_and_average(dataset) -> None:
    config = WalkForwardConfig(train_size=300, test_size=100, step=50, min_train_size=100)
    with pytest.warns(UserWarning, match="overlapping"):
        result = run_walk_forward(dataset, "ridge", config)
    assert not result.predictions.index.has_duplicates


def test_fold_summary_statistics_are_consistent(dataset, walk_forward_config) -> None:
    result = run_walk_forward(dataset, "ridge", walk_forward_config)
    frame = result.fold_metrics()
    summary = result.fold_summary()

    assert len(frame) == len(result.folds)
    assert summary["ic_fold_mean"] == pytest.approx(frame["ic_pearson"].mean())
    ratio = frame["ic_pearson"].mean() / frame["ic_pearson"].std(ddof=1)
    assert summary["ic_fold_t_naive"] == pytest.approx(ratio * np.sqrt(len(frame)))
    assert summary["ic_fold_t"] == pytest.approx(ratio * np.sqrt(summary["ic_fold_n_eff"]))
    assert 0.0 <= summary["ic_fold_positive_rate"] <= 1.0


def test_a_one_day_horizon_costs_the_fold_count_nothing(dataset, walk_forward_config) -> None:
    """With ``horizon=1`` abutting test windows really are independent.

    Pinned because the correction is as easy to over-apply as it is to
    forget. Nothing is shared here - each bar's label is the next bar, which
    is still inside the same window - so a count below the fold count would
    be inventing redundancy, and that understates an edge exactly as
    dishonestly as the raw count overstates one.
    """
    assert dataset.horizon == 1
    result = run_walk_forward(dataset, "ridge", walk_forward_config)

    assert result.independent_folds() == float(len(result.folds))
    summary = result.fold_summary()
    assert summary["ic_fold_t"] == pytest.approx(summary["ic_fold_t_naive"])


def test_abutting_folds_share_the_horizon_length_label_tail(ohlcv, walk_forward_config) -> None:
    """A 5-day target makes each window reach 4 bars into the next one."""
    dataset = build_dataset(ohlcv, FeatureConfig(horizon=5))
    result = run_walk_forward(dataset, "ridge", walk_forward_config)

    n_folds = len(result.folds)
    n_eff = result.independent_folds()
    assert n_folds - 1 < n_eff < n_folds

    # Hand-checkable: five folds of 100, 100, 100, 100 and 41 test bars, each
    # reaching horizon - 1 = 4 bars further, so widths 104, 104, 104, 104, 45
    # over a union spanning 445 bars. 445 / (461 / 5) = 4.8265.
    assert [fold.n_test for fold in result.folds] == [100, 100, 100, 100, 41]
    assert n_eff == pytest.approx(445 / (461 / 5))

    summary = result.fold_summary()
    assert abs(summary["ic_fold_t"]) < abs(summary["ic_fold_t_naive"])


def test_overlapping_folds_do_not_count_as_independent_reads(dataset) -> None:
    """A tiny step re-scores the same window; twenty folds are not twenty bets."""
    config = WalkForwardConfig(train_size=300, test_size=100, step=5, min_train_size=100)
    with pytest.warns(UserWarning, match="overlapping"):
        result = run_walk_forward(dataset, "ridge", config)

    n_eff = result.independent_folds()
    assert len(result.folds) > 10
    assert n_eff < len(result.folds) / 4
    assert n_eff >= 1.0

    summary = result.fold_summary()
    assert abs(summary["ic_fold_t"]) < abs(summary["ic_fold_t_naive"])


def test_independent_folds_never_exceeds_the_fold_count(dataset) -> None:
    """Whatever the schedule, the correction can only ever shrink the sample."""
    for step in (10, 50, 100, 200):
        config = WalkForwardConfig(train_size=300, test_size=100, step=step, min_train_size=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_walk_forward(dataset, "ridge", config)
        assert 1.0 <= result.independent_folds() <= len(result.folds)


def test_independent_folds_of_a_result_without_folds_is_zero() -> None:
    empty = pd.Series(dtype=float)
    result = WalkForwardResult(
        model_name="none",
        is_classifier=False,
        signal_units="return",
        predictions=empty,
        forward_return=empty,
        direction=empty,
        close=empty,
    )
    assert result.independent_folds() == 0.0
    assert np.isnan(result.fold_summary()["ic_fold_n_eff"])


def test_metrics_include_both_pooled_and_per_fold_views(dataset, walk_forward_config) -> None:
    metrics = run_walk_forward(dataset, "ridge", walk_forward_config).metrics()
    for key in (
        "ic_pearson",
        "ic_fold_mean",
        "ic_fold_n_eff",
        "ic_fold_t",
        "ic_fold_t_naive",
        "n_folds",
        "roc_auc",
    ):
        assert key in metrics


def test_feature_importance_is_averaged_across_folds(dataset, walk_forward_config) -> None:
    result = run_walk_forward(dataset, "ridge", walk_forward_config)
    assert result.feature_importance is not None
    assert list(result.feature_importance.index) == dataset.feature_names


def test_feature_importance_stability_reports_dispersion_across_folds(
    dataset, walk_forward_config
) -> None:
    result = run_walk_forward(dataset, "ridge", walk_forward_config)
    assert len(result.folds) > 1  # otherwise std is trivially NaN and proves nothing

    assert result.feature_importance_std is not None
    assert list(result.feature_importance_std.index) == dataset.feature_names
    assert (result.feature_importance_std.dropna() >= 0).all()

    stability = result.feature_importance_stability()
    assert list(stability.columns) == ["mean", "std", "cv"]
    assert set(stability.index) == set(dataset.feature_names)
    # Ranked by |mean| descending, same order the report displays.
    assert list(stability["mean"].abs()) == sorted(stability["mean"].abs(), reverse=True)


def test_feature_importance_stability_is_empty_without_importances(
    dataset, walk_forward_config
) -> None:
    result = run_walk_forward(dataset, CountingModel, walk_forward_config)
    assert result.feature_importance is None
    assert result.feature_importance_stability().empty


def test_unknown_model_name_is_rejected(dataset, walk_forward_config) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        run_walk_forward(dataset, "does_not_exist", walk_forward_config)


def test_regime_metrics_splits_predictions_into_volatility_terciles(
    dataset, walk_forward_config
) -> None:
    result = run_walk_forward(dataset, "ridge", walk_forward_config)
    regime = result.regime_metrics()

    assert list(regime.index) == ["low_vol", "mid_vol", "high_vol"]
    assert regime["n"].sum() == pytest.approx(len(result))
    # Terciles of realised volatility must themselves be ordered low to high.
    assert regime["realised_vol_mean"].is_monotonic_increasing


def test_regime_metrics_is_empty_without_enough_volatility_variation() -> None:
    index = pd.bdate_range("2024-01-01", periods=10, name="date")
    constant = pd.Series(1.0, index=index)
    result = WalkForwardResult(
        model_name="stub",
        is_classifier=False,
        signal_units="return",
        predictions=constant,
        forward_return=constant,
        direction=constant,
        close=constant,
        realised_volatility=pd.Series(0.2, index=index),
    )
    assert result.regime_metrics().empty


def test_regime_metrics_is_empty_without_volatility_recorded() -> None:
    index = pd.bdate_range("2024-01-01", periods=10, name="date")
    constant = pd.Series(1.0, index=index)
    result = WalkForwardResult(
        model_name="stub",
        is_classifier=False,
        signal_units="return",
        predictions=constant,
        forward_return=constant,
        direction=constant,
        close=constant,
    )
    assert result.regime_metrics().empty
