"""End-to-end orchestration: data -> features -> walk-forward -> backtest.

The CLI is a thin shell over these functions, so anything the command line can
do is equally available from a notebook or a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ai_stock.backtest.engine import BacktestResult, run_backtest
from ai_stock.config import ExperimentConfig, SimulationConfig, SyntheticConfig
from ai_stock.data.loaders import load_csv
from ai_stock.data.synthetic import generate_ohlcv
from ai_stock.evaluation.walkforward import WalkForwardResult, run_walk_forward
from ai_stock.features.builder import Dataset, build_dataset
from ai_stock.simulation.monte_carlo import (
    BootstrapResult,
    PathSimulationResult,
    SignificanceResult,
    bootstrap_metric,
    significance_test,
    simulate_price_paths,
)

__all__ = [
    "ModelRun",
    "SimulationBundle",
    "compare_models",
    "load_prices",
    "run_model",
    "run_simulation",
]


@dataclass(frozen=True)
class ModelRun:
    """One model's out-of-sample forecasts and the backtest built from them."""

    name: str
    walk_forward: WalkForwardResult
    backtest: BacktestResult

    @property
    def sharpe(self) -> float:
        return float(self.backtest.metrics["sharpe"])

    def metrics(self) -> dict[str, float]:
        """Predictive and financial metrics merged into one flat mapping."""
        merged: dict[str, float] = dict(self.walk_forward.metrics())
        merged.update(self.backtest.metrics)
        merged["benchmark_sharpe"] = float(self.backtest.benchmark_metrics["sharpe"])
        merged["excess_sharpe"] = self.backtest.excess_sharpe()
        return merged


@dataclass(frozen=True)
class SimulationBundle:
    """A model run plus the Monte-Carlo work built on top of it."""

    run: ModelRun
    forward_paths: PathSimulationResult
    significance: SignificanceResult
    sharpe_bootstrap: BootstrapResult


def load_prices(
    data_path: str | Path | None = None,
    synthetic: SyntheticConfig | None = None,
) -> pd.DataFrame:
    """Load OHLCV from ``data_path``, or generate a synthetic market instead."""
    if data_path is not None:
        return load_csv(data_path)
    return generate_ohlcv(synthetic or SyntheticConfig())


def run_model(
    ohlcv: pd.DataFrame,
    model_name: str,
    config: ExperimentConfig | None = None,
    *,
    dataset: Dataset | None = None,
    model_kwargs: dict | None = None,
) -> ModelRun:
    """Walk-forward evaluate one model and backtest its pooled forecasts.

    Passing a pre-built ``dataset`` avoids recomputing features when several
    models share the same feature configuration.
    """
    config = config or ExperimentConfig()
    dataset = dataset if dataset is not None else build_dataset(ohlcv, config.features)

    walk_forward = run_walk_forward(
        dataset, model_name, config.walk_forward, model_kwargs=model_kwargs
    )
    backtest = run_backtest(walk_forward.close, walk_forward.predictions, config.backtest)
    return ModelRun(name=model_name, walk_forward=walk_forward, backtest=backtest)


def compare_models(
    ohlcv: pd.DataFrame,
    model_names: list[str],
    config: ExperimentConfig | None = None,
    *,
    sort_by: str = "sharpe",
) -> list[ModelRun]:
    """Run several models over identical folds and rank them.

    Every model sees the same dataset and the same fold schedule, which is the
    only way a comparison table means anything.
    """
    if not model_names:
        raise ValueError("model_names is empty")
    config = config or ExperimentConfig()
    dataset = build_dataset(ohlcv, config.features)

    runs = [run_model(ohlcv, name, config, dataset=dataset) for name in model_names]

    def key(run: ModelRun) -> float:
        value = run.metrics().get(sort_by, float("nan"))
        return float("-inf") if pd.isna(value) else float(value)

    return sorted(runs, key=key, reverse=True)


def run_simulation(
    ohlcv: pd.DataFrame,
    model_name: str,
    config: ExperimentConfig | None = None,
    *,
    simulation: SimulationConfig | None = None,
    significance_method: str = "rotation",
) -> SimulationBundle:
    """Full study for one model: walk-forward, backtest, Monte Carlo, null test."""
    config = config or ExperimentConfig()
    simulation = simulation or config.simulation

    run = run_model(ohlcv, model_name, config)
    forward_paths = simulate_price_paths(run.backtest.returns, simulation, initial_value=1.0)
    significance = significance_test(
        run.walk_forward.close,
        run.walk_forward.predictions,
        backtest_config=config.backtest,
        simulation_config=simulation,
        method=significance_method,
    )
    sharpe_bootstrap = bootstrap_metric(run.backtest.returns, simulation, metric="sharpe")
    return SimulationBundle(
        run=run,
        forward_paths=forward_paths,
        significance=significance,
        sharpe_bootstrap=sharpe_bootstrap,
    )
