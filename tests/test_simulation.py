"""Monte-Carlo, bootstrap and significance-test behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_stock.config import BacktestConfig, SimulationConfig
from ai_stock.simulation.monte_carlo import (
    bootstrap_metric,
    drawdown_distribution,
    sample_returns,
    significance_test,
    simulate_price_paths,
)


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(3)
    return pd.Series(rng.normal(0.0005, 0.012, 500))


@pytest.mark.parametrize("method", ["iid_bootstrap", "block_bootstrap", "gaussian"])
def test_sample_returns_has_the_requested_shape(returns: pd.Series, method: str) -> None:
    config = SimulationConfig(n_paths=25, horizon_days=63, method=method, seed=0)
    drawn = sample_returns(returns, config)
    assert drawn.shape == (25, 63)
    assert np.isfinite(drawn).all()


def test_bootstrap_only_reuses_observed_values(returns: pd.Series) -> None:
    config = SimulationConfig(n_paths=10, horizon_days=40, method="iid_bootstrap", seed=0)
    drawn = sample_returns(returns, config)
    assert np.isin(drawn, returns.to_numpy()).all()


def test_block_bootstrap_preserves_contiguous_runs() -> None:
    """Consecutive draws should often be consecutive in the source series."""
    source = pd.Series(np.arange(100, dtype=float))
    config = SimulationConfig(
        n_paths=50, horizon_days=40, method="block_bootstrap", block_size=10, seed=1
    )
    drawn = sample_returns(source, config)

    steps = np.diff(drawn, axis=1)
    # Within a block, values advance by exactly 1 (modulo the circular wrap).
    contiguous = np.mean((steps == 1.0) | (steps == -99.0))
    assert contiguous > 0.8


def test_sampling_is_reproducible(returns: pd.Series) -> None:
    config = SimulationConfig(n_paths=10, horizon_days=20, seed=42)
    assert np.array_equal(sample_returns(returns, config), sample_returns(returns, config))


def test_simulated_quantiles_are_ordered(returns: pd.Series) -> None:
    result = simulate_price_paths(returns, SimulationConfig(n_paths=500, horizon_days=126, seed=5))
    values = [result.quantiles[q] for q in sorted(result.quantiles)]
    assert values == sorted(values)
    assert 0.0 <= result.probability_of_loss <= 1.0
    assert result.conditional_value_at_risk <= result.value_at_risk
    assert (result.max_drawdowns <= 0).all()


def test_paths_are_stored_only_on_request(returns: pd.Series) -> None:
    config = SimulationConfig(n_paths=20, horizon_days=30, seed=0)
    assert simulate_price_paths(returns, config).paths is None
    stored = simulate_price_paths(returns, config, store_paths=True).paths
    assert stored is not None and stored.shape == (20, 30)


def test_initial_value_scales_the_distribution(returns: pd.Series) -> None:
    config = SimulationConfig(n_paths=100, horizon_days=30, seed=9)
    one = simulate_price_paths(returns, config, initial_value=1.0)
    hundred = simulate_price_paths(returns, config, initial_value=100.0)
    assert np.allclose(hundred.terminal_values, one.terminal_values * 100.0)
    assert hundred.summary()["median_terminal_return"] == pytest.approx(
        one.summary()["median_terminal_return"]
    )


def test_simulation_rejects_bad_input(returns: pd.Series) -> None:
    with pytest.raises(ValueError, match="initial_value"):
        simulate_price_paths(returns, initial_value=0.0)
    with pytest.raises(ValueError, match="at least two"):
        simulate_price_paths(pd.Series([0.01]))


def test_drawdown_distribution_matches_manual_calculation() -> None:
    paths = np.array([[1.0, 1.2, 0.9], [1.0, 1.0, 1.0]])
    assert drawdown_distribution(paths) == pytest.approx([-0.25, 0.0])
    with pytest.raises(ValueError, match="2-D"):
        drawdown_distribution(np.array([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# Significance testing
# --------------------------------------------------------------------------- #
@pytest.fixture
def prices_and_signals():
    """A market plus a signal that genuinely predicts it, and one that cannot."""
    rng = np.random.default_rng(11)
    n = 800
    index = pd.bdate_range("2015-01-01", periods=n)
    daily = rng.normal(0.0, 0.01, n)
    close = pd.Series(100.0 * np.exp(np.cumsum(daily)), index=index)

    # An informed signal knows tomorrow's return, blurred with noise.
    future = pd.Series(daily, index=index).shift(-1).fillna(0.0)
    informed = future + rng.normal(0.0, 0.02, n)
    noise = pd.Series(rng.normal(0.0, 0.01, n), index=index)
    return close, pd.Series(informed, index=index), noise


@pytest.mark.parametrize("method", ["rotation", "shuffle"])
def test_informed_signal_beats_its_null(prices_and_signals, method: str) -> None:
    close, informed, _ = prices_and_signals
    result = significance_test(
        close,
        informed,
        simulation_config=SimulationConfig(n_permutations=200, seed=0),
        method=method,
    )
    assert result.p_value < 0.05
    assert result.observed > result.null_mean
    assert result.percentile_of_observed() > 95.0


def test_noise_signal_is_indistinguishable_from_luck(prices_and_signals) -> None:
    close, _, noise = prices_and_signals
    result = significance_test(
        close, noise, simulation_config=SimulationConfig(n_permutations=200, seed=0)
    )
    assert result.p_value > 0.05


def test_rotation_null_is_a_harder_bar_than_shuffle(prices_and_signals) -> None:
    """Shuffling destroys signal autocorrelation, inflating the null's costs.

    That makes the null look worse than it should and flatters the strategy,
    which is exactly why rotation is the default.
    """
    close, informed, _ = prices_and_signals
    # A persistent signal has autocorrelation for shuffling to destroy.
    persistent = informed.rolling(10, min_periods=1).mean()
    config = SimulationConfig(n_permutations=200, seed=0)
    costly = BacktestConfig(cost_bps=15.0, slippage_bps=15.0)

    rotation = significance_test(
        close, persistent, backtest_config=costly, simulation_config=config, method="rotation"
    )
    shuffle = significance_test(
        close, persistent, backtest_config=costly, simulation_config=config, method="shuffle"
    )
    assert rotation.null_mean > shuffle.null_mean


def test_p_value_stays_strictly_positive(prices_and_signals) -> None:
    close, informed, _ = prices_and_signals
    result = significance_test(
        close, informed, simulation_config=SimulationConfig(n_permutations=50, seed=0)
    )
    assert result.p_value >= 1.0 / 51.0


@pytest.mark.parametrize("statistic", ["sharpe", "total_return", "annualised_return"])
def test_alternative_statistics_are_supported(prices_and_signals, statistic: str) -> None:
    close, informed, _ = prices_and_signals
    result = significance_test(
        close,
        informed,
        simulation_config=SimulationConfig(n_permutations=50, seed=0),
        statistic=statistic,
    )
    assert result.statistic == statistic
    assert np.isfinite(result.observed)


def test_significance_test_validates_its_arguments(prices_and_signals) -> None:
    close, informed, _ = prices_and_signals
    with pytest.raises(ValueError, match="rotation"):
        significance_test(close, informed, method="jumble")
    with pytest.raises(ValueError, match="statistic must be"):
        significance_test(
            close,
            informed,
            statistic="alpha",
            simulation_config=SimulationConfig(n_permutations=1),
        )
    with pytest.raises(ValueError, match="ten overlapping"):
        significance_test(close.iloc[:5], informed.iloc[:5])


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_interval_brackets_the_point_estimate(returns: pd.Series) -> None:
    result = bootstrap_metric(returns, SimulationConfig(n_paths=300, seed=4))
    low, high = result.confidence_interval

    assert low < result.point_estimate < high
    assert result.point_estimate == pytest.approx(
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    )
    assert result.summary()["confidence"] == 0.9


def test_tighter_confidence_gives_a_narrower_interval(returns: pd.Series) -> None:
    config = SimulationConfig(n_paths=300, seed=4)
    narrow = bootstrap_metric(returns, config, confidence=0.5).confidence_interval
    wide = bootstrap_metric(returns, config, confidence=0.99).confidence_interval
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_bootstrap_accepts_a_custom_metric(returns: pd.Series) -> None:
    result = bootstrap_metric(returns, SimulationConfig(n_paths=100, seed=0), metric=np.mean)
    assert result.point_estimate == pytest.approx(returns.mean())


def test_bootstrap_rejects_an_impossible_confidence(returns: pd.Series) -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_metric(returns, confidence=1.5)
