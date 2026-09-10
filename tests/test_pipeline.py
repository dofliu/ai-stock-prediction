"""End-to-end pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai_stock.config import (
    BacktestConfig,
    ExperimentConfig,
    SimulationConfig,
    SyntheticConfig,
    WalkForwardConfig,
)
from ai_stock.data.loaders import save_csv
from ai_stock.evaluation.metrics import probabilistic_sharpe_ratio
from ai_stock.pipeline import (
    compare_models,
    deflated_sharpe_ratios,
    load_prices,
    run_model,
    run_simulation,
)


@pytest.fixture(scope="module")
def config() -> ExperimentConfig:
    return ExperimentConfig(
        synthetic=SyntheticConfig(n_days=900, seed=1234),
        walk_forward=WalkForwardConfig(train_size=400, test_size=100),
        backtest=BacktestConfig(cost_bps=1.0, slippage_bps=1.0),
        simulation=SimulationConfig(n_paths=100, n_permutations=50, seed=0),
    )


def test_load_prices_prefers_the_file(ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    path = save_csv(ohlcv, tmp_path / "prices.csv")
    pd.testing.assert_frame_equal(load_prices(path), ohlcv)


def test_load_prices_falls_back_to_synthetic() -> None:
    frame = load_prices(None, SyntheticConfig(n_days=100, seed=0))
    assert len(frame) == 100


def test_run_model_links_forecasts_to_the_backtest(ohlcv, config) -> None:
    run = run_model(ohlcv, "ridge", config)

    assert run.name == "ridge"
    assert run.walk_forward.predictions.index.equals(run.backtest.positions.index)
    assert len(run.backtest.equity) == len(run.walk_forward.predictions)

    metrics = run.metrics()
    for key in ("ic_fold_mean", "sharpe", "benchmark_sharpe", "excess_sharpe"):
        assert key in metrics
    assert metrics["excess_sharpe"] == pytest.approx(
        metrics["sharpe"] - metrics["benchmark_sharpe"]
    )
    assert run.sharpe == pytest.approx(metrics["sharpe"])


def test_compare_models_uses_identical_folds_and_ranks(ohlcv, config) -> None:
    names = ["zero", "momentum", "ridge"]
    runs = compare_models(ohlcv, names, config)

    assert {run.name for run in runs} == set(names)
    indices = [run.walk_forward.predictions.index for run in runs]
    assert all(index.equals(indices[0]) for index in indices)

    sharpes = [run.metrics()["sharpe"] for run in runs]
    ranked = [value for value in sharpes if pd.notna(value)]
    assert ranked == sorted(ranked, reverse=True)


def test_compare_models_can_rank_by_another_metric(ohlcv, config) -> None:
    runs = compare_models(ohlcv, ["momentum", "reversion", "ridge"], config, sort_by="ic_fold_mean")
    values = [run.metrics()["ic_fold_mean"] for run in runs]
    assert values == sorted(values, reverse=True)


def test_compare_models_rejects_an_empty_list(ohlcv, config) -> None:
    with pytest.raises(ValueError, match="empty"):
        compare_models(ohlcv, [], config)


def test_deflated_sharpe_ratios_covers_every_run(ohlcv, config) -> None:
    runs = compare_models(ohlcv, ["zero", "momentum", "ridge"], config)
    deflated = deflated_sharpe_ratios(runs)

    assert set(deflated) == {run.name for run in runs}
    assert all(0.0 <= value <= 1.0 or pd.isna(value) for value in deflated.values())


def test_deflated_sharpe_ratios_needs_no_peers_for_a_single_run(ohlcv, config) -> None:
    run = run_model(ohlcv, "ridge", config)
    deflated = deflated_sharpe_ratios([run])

    assert deflated["ridge"] == pytest.approx(
        probabilistic_sharpe_ratio(run.backtest.returns), nan_ok=True
    )


def test_deflated_sharpe_ratios_of_an_empty_list_is_empty() -> None:
    assert deflated_sharpe_ratios([]) == {}


def test_run_simulation_returns_a_complete_bundle(ohlcv, config) -> None:
    bundle = run_simulation(ohlcv, "ridge", config)

    assert bundle.run.name == "ridge"
    assert len(bundle.significance.null_distribution) <= config.simulation.n_permutations
    assert len(bundle.forward_paths.terminal_values) == config.simulation.n_paths
    assert bundle.sharpe_bootstrap.point_estimate == pytest.approx(
        bundle.run.backtest.metrics["sharpe"]
    )
    assert 0.0 <= bundle.significance.p_value <= 1.0


def test_reusing_a_dataset_gives_identical_results(ohlcv, config) -> None:
    from ai_stock.features.builder import build_dataset

    dataset = build_dataset(ohlcv, config.features)
    with_dataset = run_model(ohlcv, "ridge", config, dataset=dataset)
    without = run_model(ohlcv, "ridge", config)

    pd.testing.assert_series_equal(
        with_dataset.walk_forward.predictions, without.walk_forward.predictions
    )


def test_model_kwargs_are_forwarded(ohlcv, config) -> None:
    strong = run_model(ohlcv, "ridge", config, model_kwargs={"alpha": 1e6})
    weak = run_model(ohlcv, "ridge", config, model_kwargs={"alpha": 1e-6})
    assert strong.walk_forward.predictions.std() < weak.walk_forward.predictions.std()


def test_planted_edge_is_recovered_but_an_efficient_market_is_not() -> None:
    """The framework's own credibility check, run as a test.

    On a market with a known edge the pipeline should find it; on a market with
    none it must not manufacture one.
    """
    config = ExperimentConfig(
        walk_forward=WalkForwardConfig(train_size=750, test_size=250),
        simulation=SimulationConfig(n_paths=50, n_permutations=150, seed=0),
    )

    with_edge = run_simulation(
        load_prices(None, SyntheticConfig(n_days=4000, seed=11)), "ridge", config
    )
    efficient = run_simulation(
        load_prices(None, SyntheticConfig(n_days=4000, seed=11, ar1=0.0, reversion=0.0)),
        "ridge",
        config,
    )

    assert with_edge.run.metrics()["ic_fold_mean"] > 0.02
    assert with_edge.significance.p_value < 0.10
    assert abs(efficient.run.metrics()["ic_fold_mean"]) < 0.05
    assert efficient.significance.p_value > 0.10
