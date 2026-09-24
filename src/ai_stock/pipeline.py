"""End-to-end orchestration: data -> features -> walk-forward -> backtest.

The CLI is a thin shell over these functions, so anything the command line can
do is equally available from a notebook or a test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ai_stock.backtest.engine import BacktestResult, run_backtest
from ai_stock.config import ExperimentConfig, SimulationConfig, SyntheticConfig
from ai_stock.data.loaders import load_csv, load_yfinance, save_csv
from ai_stock.data.synthetic import generate_ohlcv
from ai_stock.evaluation.metrics import deflated_sharpe_ratio, sharpe_ratio
from ai_stock.evaluation.multiple_testing import (
    benjamini_hochberg,
    bonferroni_threshold,
    expected_false_positives,
)
from ai_stock.evaluation.tuning import TuningConfig
from ai_stock.evaluation.walkforward import WalkForwardResult, run_walk_forward
from ai_stock.features.builder import Dataset, build_dataset
from ai_stock.portfolio import PortfolioResult, build_portfolio
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
    "ScreenEntry",
    "ScreenResult",
    "SimulationBundle",
    "compare_models",
    "deflated_sharpe_ratios",
    "load_prices",
    "load_universe",
    "run_model",
    "run_simulation",
    "screen_universe",
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
    tuning: TuningConfig | None = None,
) -> ModelRun:
    """Walk-forward evaluate one model and backtest its pooled forecasts.

    Passing a pre-built ``dataset`` avoids recomputing features when several
    models share the same feature configuration.

    ``tuning`` makes every fold select its own hyper-parameters from its own
    training bars; without it the fixed ones are used, which is the default
    and the cheaper, less honest answer.
    """
    config = config or ExperimentConfig()
    dataset = dataset if dataset is not None else build_dataset(ohlcv, config.features)

    walk_forward = run_walk_forward(
        dataset, model_name, config.walk_forward, model_kwargs=model_kwargs, tuning=tuning
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


def deflated_sharpe_ratios(runs: list[ModelRun]) -> dict[str, float]:
    """Deflated Sharpe ratio for each of several models compared on identical folds.

    Comparing models and reporting the best one's Sharpe has the same
    multiple-testing problem `screen_universe` corrects for across symbols:
    the models tried are the trials, so the benchmark each one's Sharpe must
    clear is the best a batch of ``len(runs)`` skill-less strategies would be
    expected to show by chance, given how much their Sharpes actually
    disagree. With a single run there is nothing to correct for.
    """
    if not runs:
        return {}
    trial_sharpes = np.array(
        [sharpe_ratio(run.backtest.returns, periods_per_year=1) for run in runs]
    )
    finite = trial_sharpes[np.isfinite(trial_sharpes)]
    trial_std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return {
        run.name: deflated_sharpe_ratio(
            run.backtest.returns, n_trials=len(runs), trial_sharpe_std=trial_std
        )
        for run in runs
    }


@dataclass(frozen=True)
class ScreenEntry:
    """One symbol's result inside a screen, or the reason it has none."""

    symbol: str
    n_bars: int
    run: ModelRun | None = None
    significance: SignificanceResult | None = None
    q_value: float = float("nan")
    """Benjamini-Hochberg adjusted p-value across the whole screen."""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.run is not None

    @property
    def p_value(self) -> float:
        return self.significance.p_value if self.significance else float("nan")

    def metrics(self) -> dict[str, float]:
        """Flat metrics for the ranking table; empty when the symbol failed."""
        if self.run is None:
            return {}
        merged = dict(self.run.metrics())
        merged["p_value"] = self.p_value
        merged["q_value"] = self.q_value
        merged["n_bars"] = float(self.n_bars)
        return merged


@dataclass(frozen=True)
class ScreenResult:
    """A universe screened with one model, ranked by one metric."""

    entries: list[ScreenEntry]
    model_name: str
    rank_by: str
    alpha: float = 0.05
    config: ExperimentConfig = field(default_factory=ExperimentConfig)

    @property
    def ranked(self) -> list[ScreenEntry]:
        """Symbols that produced a result, best first."""
        return [entry for entry in self.entries if entry.ok]

    @property
    def failures(self) -> list[ScreenEntry]:
        return [entry for entry in self.entries if not entry.ok]

    @property
    def n_tested(self) -> int:
        return len(self.ranked)

    def bonferroni(self) -> float:
        """Per-symbol p-value needed for family-wise significance."""
        return bonferroni_threshold(max(1, self.n_tested), self.alpha)

    def expected_false_positives(self) -> float:
        """How many symbols pure noise would flag at ``alpha``."""
        return expected_false_positives(max(1, self.n_tested), self.alpha)

    def survivors(self) -> list[ScreenEntry]:
        """Symbols that beat buy-and-hold *and* survive the FDR correction.

        Both conditions matter: an edge that loses to holding the stock is not
        worth trading, and one that only looks real because it was the best of
        many is not an edge at all.
        """
        return [
            entry
            for entry in self.ranked
            if entry.metrics().get("excess_sharpe", float("-inf")) > 0
            and np.isfinite(entry.q_value)
            and entry.q_value <= self.alpha
        ]

    def portfolio(self, *, scheme: str = "equal") -> PortfolioResult:
        """Hold every evaluated symbol at once, and measure what that diversifies.

        The ranking table scores each symbol as a standalone decision, which is
        the one thing a portfolio is not. Sleeves that share a driver share
        their drawdowns, so the combined risk is not the sum of the parts - see
        :mod:`ai_stock.portfolio`.

        Failed symbols are left out (they have no return stream), and so is the
        survivor filter: a screen usually has no survivors, and the correlation
        structure of the universe is worth knowing either way.
        """
        if not self.ranked:
            raise ValueError("no evaluated symbols to combine into a portfolio")
        entries = [entry for entry in self.ranked if entry.run]
        return build_portfolio(
            {entry.symbol: entry.run.backtest.returns for entry in entries},
            asset_returns_by_symbol={
                entry.symbol: entry.run.backtest.asset_returns for entry in entries
            },
            scheme=scheme,
        )

    def table(self) -> pd.DataFrame:
        """Ranking table, one row per symbol that produced a result."""
        rows = []
        for entry in self.ranked:
            metrics = entry.metrics()
            rows.append(
                {
                    "symbol": entry.symbol,
                    "n_bars": entry.n_bars,
                    "ic_fold_mean": metrics.get("ic_fold_mean"),
                    "ic_fold_t": metrics.get("ic_fold_t"),
                    "directional_accuracy": metrics.get("directional_accuracy"),
                    "sharpe": metrics.get("sharpe"),
                    "benchmark_sharpe": metrics.get("benchmark_sharpe"),
                    "excess_sharpe": metrics.get("excess_sharpe"),
                    "annualised_return": metrics.get("annualised_return"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "annual_turnover": metrics.get("annual_turnover"),
                    "p_value": entry.p_value,
                    "q_value": entry.q_value,
                }
            )
        frame = pd.DataFrame(rows)
        return frame.set_index("symbol") if not frame.empty else frame


def _expand_data_paths(paths: Sequence[Path | str]) -> list[Path]:
    """Expand directories to the CSV files inside them, keeping order."""
    expanded: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.csv")))
        else:
            expanded.append(path)
    return expanded


def load_universe(
    *,
    data_paths: Sequence[Path | str] | None = None,
    tickers: Sequence[str] | None = None,
    period: str = "12y",
    cache_dir: Path | str | None = None,
    loader: Callable[[str], pd.DataFrame] | None = None,
    skip_errors: bool = False,
) -> dict[str, pd.DataFrame]:
    """Collect a symbol -> OHLCV mapping from CSV files and/or live tickers.

    Parameters
    ----------
    data_paths:
        CSV files, or directories whose ``*.csv`` files are all loaded. The
        symbol is the file stem.
    tickers:
        Symbols to download through :func:`ai_stock.data.load_yfinance`, which
        needs network access and the optional ``yfinance`` package.
    period:
        History length requested per ticker.
    cache_dir:
        When set, a downloaded ticker is written here and reused on the next
        run, so a screen can be repeated offline.
    loader:
        Overrides the download function; used by the tests, and handy for
        plugging in another data vendor.
    skip_errors:
        Omit a symbol that fails to load instead of raising. A delisted or
        mistyped ticker should not stop a scheduled job collecting the rest;
        recover the omissions as ``set(tickers) - set(universe)``.

    Raises
    ------
    ValueError
        If neither source is given, or two sources claim the same symbol.
        Duplicate symbols always raise - that is a configuration error, not a
        data one, and silently dropping one would hide it.
    """
    if not data_paths and not tickers:
        raise ValueError("provide data_paths, tickers, or both")

    universe: dict[str, pd.DataFrame] = {}

    for path in _expand_data_paths(data_paths or []):
        symbol = path.stem
        if symbol in universe:
            raise ValueError(f"duplicate symbol {symbol!r} in the universe")
        universe[symbol] = load_csv(path)

    if tickers:
        download = loader or (lambda t: load_yfinance(t, period=period))
        cache = Path(cache_dir) if cache_dir else None
        for ticker in tickers:
            if ticker in universe:
                raise ValueError(f"duplicate symbol {ticker!r} in the universe")
            cached = cache / f"{ticker.replace('/', '_')}.csv" if cache else None
            try:
                if cached is not None and cached.exists():
                    universe[ticker] = load_csv(cached)
                    continue
                frame = download(ticker)
                if cached is not None:
                    save_csv(frame, cached)
                universe[ticker] = frame
            except Exception:
                if not skip_errors:
                    raise

    return universe


def screen_universe(
    universe: Mapping[str, pd.DataFrame],
    model_name: str = "random_forest",
    config: ExperimentConfig | None = None,
    *,
    rank_by: str = "excess_sharpe",
    permutations: int = 200,
    alpha: float = 0.05,
    significance_method: str = "rotation",
) -> ScreenResult:
    """Run the same study on every symbol and rank them.

    Every symbol gets its own walk-forward, backtest and null test, so the
    comparison is like for like. A symbol that cannot be evaluated - too few
    bars, bad data - records the reason and does not abort the screen.

    Because ranking many symbols by the same statistic is exactly how false
    positives are manufactured, p-values are adjusted across the screen with
    Benjamini-Hochberg and exposed as ``q_value``.

    Parameters
    ----------
    universe:
        Symbol -> OHLCV frames, e.g. from :func:`load_universe`.
    permutations:
        Null draws per symbol; ``0`` skips the significance test, which also
        skips the correction.
    """
    if not universe:
        raise ValueError("the universe is empty")
    config = config or ExperimentConfig()

    raw: list[ScreenEntry] = []
    for symbol, ohlcv in universe.items():
        n_bars = len(ohlcv)
        try:
            run = run_model(ohlcv, model_name, config)
            significance = None
            if permutations > 0:
                significance = significance_test(
                    run.walk_forward.close,
                    run.walk_forward.predictions,
                    backtest_config=config.backtest,
                    simulation_config=replace(config.simulation, n_permutations=permutations),
                    method=significance_method,
                )
            raw.append(
                ScreenEntry(symbol=symbol, n_bars=n_bars, run=run, significance=significance)
            )
        except (ValueError, KeyError, RuntimeError) as error:
            raw.append(ScreenEntry(symbol=symbol, n_bars=n_bars, error=str(error)))

    q_values = benjamini_hochberg([entry.p_value for entry in raw])
    scored = [replace(entry, q_value=float(q)) for entry, q in zip(raw, q_values, strict=True)]

    def sort_key(entry: ScreenEntry) -> float:
        value = entry.metrics().get(rank_by, float("nan"))
        return float("-inf") if value is None or pd.isna(value) else float(value)

    ok = sorted([e for e in scored if e.ok], key=sort_key, reverse=True)
    failed = [e for e in scored if not e.ok]
    return ScreenResult(
        entries=[*ok, *failed],
        model_name=model_name,
        rank_by=rank_by,
        alpha=alpha,
        config=config,
    )


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
