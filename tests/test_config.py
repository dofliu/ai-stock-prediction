"""Configuration validation and derived values."""

from __future__ import annotations

import pytest

from ai_stock.config import (
    BacktestConfig,
    ExperimentConfig,
    FeatureConfig,
    SimulationConfig,
    WalkForwardConfig,
)


def test_step_defaults_to_non_overlapping_tests() -> None:
    assert WalkForwardConfig(test_size=60).resolved_step() == 60
    assert WalkForwardConfig(test_size=60, step=10).resolved_step() == 10


def test_embargo_defaults_to_the_horizon() -> None:
    config = WalkForwardConfig()
    assert config.resolved_embargo(5) == 5
    assert WalkForwardConfig(embargo=0).resolved_embargo(5) == 0


def test_min_train_size_defaults_to_train_size() -> None:
    assert WalkForwardConfig(train_size=123).resolved_min_train_size() == 123
    assert WalkForwardConfig(train_size=123, min_train_size=50).resolved_min_train_size() == 50


def test_total_cost_sums_commission_and_slippage() -> None:
    assert BacktestConfig(cost_bps=1.5, slippage_bps=2.5).total_cost_bps == 4.0


def test_experiment_config_horizon_helper_is_non_mutating() -> None:
    config = ExperimentConfig()
    updated = config.with_horizon(7)

    assert updated.features.horizon == 7
    assert config.features.horizon == 1
    assert updated.walk_forward is config.walk_forward


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: FeatureConfig(horizon=0), "horizon"),
        (lambda: FeatureConfig(neutral_band=-0.1), "neutral_band"),
        (lambda: FeatureConfig(sma_windows=(0,)), "windows"),
        (lambda: FeatureConfig(bollinger=(20, 0.0)), "bollinger"),
        (lambda: WalkForwardConfig(train_size=0), "train_size"),
        (lambda: WalkForwardConfig(step=0), "step"),
        (lambda: WalkForwardConfig(embargo=-1), "embargo"),
        (lambda: WalkForwardConfig(min_train_size=0), "min_train_size"),
        (lambda: BacktestConfig(sizing="martingale"), "sizing"),
        (lambda: BacktestConfig(max_leverage=0), "max_leverage"),
        (lambda: BacktestConfig(threshold=-1), "threshold"),
        (lambda: BacktestConfig(cost_bps=-1), "costs"),
        (lambda: BacktestConfig(vol_target=0), "vol_target"),
        (lambda: BacktestConfig(vol_lookback=1), "vol_lookback"),
        (lambda: BacktestConfig(initial_equity=0), "initial_equity"),
        (lambda: SimulationConfig(method="mcmc"), "method"),
        (lambda: SimulationConfig(n_paths=0), "n_paths"),
        (lambda: SimulationConfig(horizon_days=0), "horizon_days"),
        (lambda: SimulationConfig(block_size=0), "block_size"),
        (lambda: SimulationConfig(n_permutations=-1), "n_permutations"),
        (lambda: SimulationConfig(quantiles=(0.0, 0.5)), "quantiles"),
    ],
)
def test_invalid_configuration_is_rejected(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_configs_are_frozen() -> None:
    config = BacktestConfig()
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        config.cost_bps = 5.0  # type: ignore[misc]
