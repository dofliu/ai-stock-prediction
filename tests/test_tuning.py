"""Per-fold hyper-parameter selection.

The test that carries the claim is
``test_selection_reads_nothing_outside_the_training_window``: everything else
here checks that the machinery behaves, but that one checks the only thing
this feature asserts - that choosing hyper-parameters inside a fold leaves the
fold's test bars out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_stock.config import FeatureConfig, WalkForwardConfig
from ai_stock.evaluation.tuning import (
    TuningConfig,
    describe_params,
    expand_grid,
    inner_schedule,
)
from ai_stock.evaluation.walkforward import WalkForwardSplitter, run_walk_forward
from ai_stock.features.builder import build_dataset
from ai_stock.models.base import Model


class AlphaModel(Model):
    """Predicts a constant proportional to ``alpha``, so selection is decidable.

    Deliberately not a real forecaster: the inner score becomes a pure
    function of the parameter and the validation bars, which lets a test
    assert *which* value must win rather than only that something did.
    """

    name = "alpha_model"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._scale = 0.0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> AlphaModel:
        self._validate_fit_input(features, target)
        self._scale = float(target.mean())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        signal = features.iloc[:, 0].to_numpy(dtype=float)
        return self.alpha * signal + self._scale


class TwoParamModel(AlphaModel):
    """Two axes, so the stability table has two rows and the spread still has one."""

    name = "two_param"

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(alpha)
        self.beta = beta

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.beta * super().predict(features)


class PickyModel(AlphaModel):
    """Rejects one grid entry, the way a real estimator rejects a bad argument."""

    name = "picky"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> PickyModel:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        super().fit(features, target)
        return self


@pytest.fixture(scope="module")
def tuned_dataset(ohlcv: pd.DataFrame):
    return build_dataset(ohlcv, FeatureConfig(horizon=5))


@pytest.fixture(scope="module")
def tuned_config() -> WalkForwardConfig:
    return WalkForwardConfig(train_size=400, test_size=100, min_train_size=200)


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #


def test_expand_grid_is_deterministic_so_ties_resolve_the_same_way() -> None:
    grid = {"alpha": [1.0, 2.0], "beta": ["a", "b"]}
    assert expand_grid(grid) == expand_grid(grid)
    assert expand_grid(grid)[0] == {"alpha": 1.0, "beta": "a"}
    assert len(expand_grid(grid)) == 4


def test_empty_grid_is_one_candidate_not_zero() -> None:
    assert expand_grid({}) == [{}]
    assert not TuningConfig().enabled
    assert not TuningConfig(grid={"alpha": [1.0]}).enabled, "one candidate is not a choice"
    assert TuningConfig(grid={"alpha": [1.0, 2.0]}).enabled


def test_an_empty_value_list_is_an_error_not_an_empty_grid() -> None:
    with pytest.raises(ValueError, match="empty list of values"):
        expand_grid({"alpha": []})


def test_describe_params_labels_the_default_candidate() -> None:
    assert describe_params({}) == "defaults"
    assert describe_params({"alpha": 1.0}) == "alpha=1.0"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_inner_folds": 0}, "n_inner_folds"),
        ({"inner_fraction": 0.0}, "inner_fraction"),
        ({"inner_fraction": 1.0}, "inner_fraction"),
        ({"metric": ""}, "metric"),
        ({"min_inner_train": 0}, "min_inner_train"),
    ],
)
def test_tuning_config_rejects_impossible_settings(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TuningConfig(**kwargs)


def test_a_non_finite_score_never_wins() -> None:
    config = TuningConfig()
    assert not config.better(float("nan"), -10.0)
    assert config.better(-10.0, float("nan")), "anything finite beats nothing"
    assert not TuningConfig(metric="rmse", higher_is_better=False).better(0.2, 0.1)


# --------------------------------------------------------------------------- #
# The inner schedule
# --------------------------------------------------------------------------- #


def test_inner_schedule_fits_exactly_the_requested_number_of_folds() -> None:
    config = TuningConfig(grid={"alpha": [1.0, 2.0]}, n_inner_folds=3)
    train_size, test_size = inner_schedule(750, 5, config)
    folds = list(
        WalkForwardSplitter(
            WalkForwardConfig(train_size=train_size, test_size=test_size, step=test_size, embargo=5)
        ).split(750, horizon=5)
    )
    assert len(folds) == config.n_inner_folds


def test_inner_validation_stays_inside_the_training_window() -> None:
    """The whole point: no inner fold may reach a bar the outer fold trains past."""
    n_train, embargo = 750, 5
    config = TuningConfig(grid={"alpha": [1.0, 2.0]}, n_inner_folds=3)
    train_size, test_size = inner_schedule(n_train, embargo, config)
    folds = list(
        WalkForwardSplitter(
            WalkForwardConfig(
                train_size=train_size, test_size=test_size, step=test_size, embargo=embargo
            )
        ).split(n_train, horizon=embargo)
    )
    for fold in folds:
        assert fold.test.max() < n_train
        assert fold.train.max() + embargo <= fold.test.min()


def test_a_training_window_too_short_to_split_selects_nothing() -> None:
    grid = {"alpha": [1.0, 2.0]}
    assert inner_schedule(10, 5, TuningConfig(grid=grid, n_inner_folds=8)) is None
    assert inner_schedule(40, 5, TuningConfig(grid=grid)) is None, "23 bars is not a fit"


def test_the_floor_on_inner_training_bars_is_what_rejects_a_short_window() -> None:
    """The arithmetic alone always 'fits'; only ``min_inner_train`` says no.

    Without the floor a 40-bar window splits into a 23-bar training block and
    reports a winner, which is a coin flip dressed as a choice. Pinned because
    the floor looks like a redundant guard next to ``train_size < test_size``
    and is the one that actually fires.
    """
    grid = {"alpha": [1.0, 2.0]}
    assert inner_schedule(40, 5, TuningConfig(grid=grid, min_inner_train=1)) == (23, 4)
    assert inner_schedule(40, 5, TuningConfig(grid=grid, min_inner_train=24)) is None


# --------------------------------------------------------------------------- #
# The claim
# --------------------------------------------------------------------------- #


def test_selection_reads_nothing_outside_the_training_window(
    tuned_dataset, tuned_config: WalkForwardConfig
) -> None:
    """Corrupting the test bars must not change a single fold's chosen value.

    This is the property that makes a tuned run's out-of-sample number mean
    anything. If selection ever peeked past ``fold.train`` - a splitter off by
    one, an inner fold measured on the pooled calendar, an embargo applied in
    the wrong direction - the winners would move when the future moves. They
    must not.
    """
    tuning = TuningConfig(grid={"alpha": [0.5, 1.0, 2.0, 4.0]}, n_inner_folds=2)
    baseline = run_walk_forward(tuned_dataset, AlphaModel, tuned_config, tuning=tuning)

    first_fold = list(WalkForwardSplitter(tuned_config).split(len(tuned_dataset), 5))[0]
    boundary = int(first_fold.train.max()) + 1

    scrambled = tuned_dataset.features.copy()
    rng = np.random.default_rng(0)
    tail = scrambled.iloc[boundary:]
    scrambled.iloc[boundary:] = rng.normal(0.0, 10.0, size=tail.shape)
    corrupted = type(tuned_dataset)(
        features=scrambled,
        forward_return=tuned_dataset.forward_return,
        direction=tuned_dataset.direction,
        close=tuned_dataset.close,
        config=tuned_dataset.config,
    )
    after = run_walk_forward(corrupted, AlphaModel, tuned_config, tuning=tuning)

    assert baseline.folds[0].params == after.folds[0].params
    assert baseline.folds[0].params, "the test is vacuous if nothing was selected"


def test_the_chosen_value_is_the_best_scoring_candidate(
    tuned_dataset, tuned_config: WalkForwardConfig
) -> None:
    tuning = TuningConfig(grid={"alpha": [0.5, 1.0, 2.0, 4.0]}, n_inner_folds=2)
    result = run_walk_forward(tuned_dataset, AlphaModel, tuned_config, tuning=tuning)

    scores = result.selection_scores()
    for number, group in scores.groupby("fold"):
        chosen = group[group["chosen"]]
        assert len(chosen) == 1, f"fold {number} recorded {len(chosen)} winners"
        assert chosen["score"].iloc[0] == pytest.approx(group["score"].max())


def test_a_repeated_grid_value_still_yields_exactly_one_winner(
    tuned_dataset, tuned_config: WalkForwardConfig
) -> None:
    """Two candidates can share a label; only one of them won.

    Marking the winner by label rather than by position is the obvious-looking
    implementation and it double-marks here.
    """
    tuning = TuningConfig(grid={"alpha": [2.0, 2.0, 4.0]}, n_inner_folds=2)
    result = run_walk_forward(tuned_dataset, AlphaModel, tuned_config, tuning=tuning)

    scores = result.selection_scores()
    assert not scores.empty
    for number, group in scores.groupby("fold"):
        assert group["chosen"].sum() == 1, f"fold {number} marked {group['chosen'].sum()} winners"


def test_a_candidate_that_raises_is_scored_rather_than_crashing_the_run(
    tuned_dataset, tuned_config: WalkForwardConfig
) -> None:
    """A hand-written grid contains combinations some estimator rejects."""
    tuning = TuningConfig(grid={"alpha": [-1.0, 1.0, 2.0]}, n_inner_folds=2)
    result = run_walk_forward(tuned_dataset, PickyModel, tuned_config, tuning=tuning)

    scores = result.selection_scores()
    rejected = scores[scores["params"] == "alpha=-1.0"]
    assert not rejected.empty
    assert rejected["score"].isna().all()
    assert not rejected["chosen"].any(), "a candidate that never fitted cannot win"
    assert (result.selected_params()["alpha"] > 0).all()


# --------------------------------------------------------------------------- #
# Reporting the selection
# --------------------------------------------------------------------------- #


def test_a_run_without_tuning_reports_no_selection(
    tuned_dataset, tuned_config: WalkForwardConfig
) -> None:
    """The default path must be untouched, tables included."""
    result = run_walk_forward(tuned_dataset, AlphaModel, tuned_config)
    assert result.selected_params().empty
    assert result.selection_stability().empty
    assert result.selection_scores().empty
    assert all(fold.params == {} for fold in result.folds)


def test_a_single_candidate_grid_is_not_tuning(
    tuned_dataset, tuned_config: WalkForwardConfig
) -> None:
    """Nothing was chosen, so nothing may be reported as chosen."""
    result = run_walk_forward(
        tuned_dataset, AlphaModel, tuned_config, tuning=TuningConfig(grid={"alpha": [2.0]})
    )
    assert result.selected_params().empty


def test_stability_counts_distinct_winners_and_their_share(
    tuned_dataset, tuned_config: WalkForwardConfig
) -> None:
    tuning = TuningConfig(grid={"alpha": [0.5, 1.0, 2.0, 4.0]}, n_inner_folds=2)
    result = run_walk_forward(tuned_dataset, AlphaModel, tuned_config, tuning=tuning)

    stability = result.selection_stability()
    chosen = result.selected_params()["alpha"]
    row = stability.loc["alpha"]
    assert row["n_distinct"] == chosen.nunique()
    assert row["modal_share"] == pytest.approx(chosen.value_counts().iloc[0] / len(chosen))
    assert 0.0 < row["modal_share"] <= 1.0
    assert "selection_spread" not in stability, "a grid-wide quantity is not per-parameter"


def test_selection_spread_is_one_number_for_the_whole_grid(
    tuned_dataset, tuned_config: WalkForwardConfig
) -> None:
    """Two parameters, one spread: candidates are scored as combinations."""
    tuning = TuningConfig(grid={"alpha": [0.5, 2.0], "beta": [1.0, 3.0]}, n_inner_folds=2)
    result = run_walk_forward(tuned_dataset, TwoParamModel, tuned_config, tuning=tuning)

    assert list(result.selection_stability().index) == ["alpha", "beta"]
    spread = result.selection_spread()
    assert np.isfinite(spread) and spread >= 0.0

    scores = result.selection_scores()
    by_fold = scores.groupby("fold")["score"]
    expected = float((by_fold.max() - by_fold.min()).mean())
    assert spread == pytest.approx(expected)


def test_selection_spread_is_nan_without_a_selection(
    tuned_dataset, tuned_config: WalkForwardConfig
) -> None:
    result = run_walk_forward(tuned_dataset, AlphaModel, tuned_config)
    assert np.isnan(result.selection_spread())


def test_the_report_states_what_was_selected(
    ohlcv: pd.DataFrame, tuned_config: WalkForwardConfig
) -> None:
    from dataclasses import replace

    from ai_stock.config import ExperimentConfig
    from ai_stock.pipeline import run_model
    from ai_stock.reporting.studies import render_backtest_report

    config = replace(
        ExperimentConfig(),
        features=FeatureConfig(horizon=5),
        walk_forward=tuned_config,
    )
    tuning = TuningConfig(grid={"alpha": [0.1, 1.0, 10.0]}, n_inner_folds=2)
    run = run_model(ohlcv, "ridge", config, tuning=tuning)
    report = render_backtest_report(run, config)

    assert "Hyper-parameters, selected inside each fold" in report
    assert "modal_share" in report
    assert "still out-of-sample after selection" in report

    untuned = render_backtest_report(run_model(ohlcv, "ridge", config), config)
    assert "Hyper-parameters, selected inside each fold" not in untuned
