"""Monte-Carlo simulation and significance testing.

A single backtest produces one number. These tools ask how much of that number
survives contact with randomness:

* :func:`simulate_price_paths` resamples the return distribution forward to get
  a *distribution* of outcomes rather than a single projection - the honest way
  to state "what could happen next year".
* :func:`significance_test` builds a null distribution of the strategy's own
  Sharpe ratio by breaking the alignment between signal and future returns
  while leaving both series otherwise intact. If the observed Sharpe sits
  inside that cloud, the strategy is indistinguishable from luck.
* :func:`bootstrap_metric` puts a confidence interval on a realised metric.

The default null uses a *circular rotation* rather than a shuffle. Shuffling
destroys the signal's autocorrelation, which slashes turnover and therefore
costs, quietly biasing the null in the strategy's favour. Rotation preserves
turnover and only destroys the timing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_stock.config import TRADING_DAYS_PER_YEAR, BacktestConfig, SimulationConfig
from ai_stock.evaluation.metrics import max_drawdown, sharpe_ratio

__all__ = [
    "BootstrapResult",
    "PathSimulationResult",
    "SignificanceResult",
    "bootstrap_metric",
    "sample_returns",
    "significance_test",
    "simulate_price_paths",
]

_BPS = 1e-4


@dataclass(frozen=True)
class PathSimulationResult:
    """Distribution of simulated outcomes over the simulation horizon."""

    terminal_values: np.ndarray
    quantiles: dict[float, float]
    probability_of_loss: float
    value_at_risk: float
    """5% worst-case *loss* on the terminal value, as a negative return."""
    conditional_value_at_risk: float
    """Mean return in the worst 5% of paths (expected shortfall)."""
    max_drawdowns: np.ndarray
    initial_value: float
    config: SimulationConfig
    paths: np.ndarray | None = None

    def summary(self) -> dict[str, float]:
        """Flat, report-friendly summary of the simulated distribution."""
        terminal_return = self.terminal_values / self.initial_value - 1.0
        summary = {
            "n_paths": float(len(self.terminal_values)),
            "horizon_days": float(self.config.horizon_days),
            "median_terminal_value": float(np.median(self.terminal_values)),
            "mean_terminal_return": float(np.mean(terminal_return)),
            "median_terminal_return": float(np.median(terminal_return)),
            "probability_of_loss": self.probability_of_loss,
            "value_at_risk_95": self.value_at_risk,
            "conditional_value_at_risk_95": self.conditional_value_at_risk,
            "median_max_drawdown": float(np.median(self.max_drawdowns)),
            "worst_max_drawdown": float(np.min(self.max_drawdowns)),
        }
        for q, value in self.quantiles.items():
            summary[f"terminal_return_q{int(round(q * 100)):02d}"] = (
                value / self.initial_value - 1.0
            )
        return summary


@dataclass(frozen=True)
class SignificanceResult:
    """Observed statistic against a null distribution built by resampling."""

    statistic: str
    observed: float
    null_distribution: np.ndarray
    p_value: float
    method: str

    @property
    def null_mean(self) -> float:
        return (
            float(np.mean(self.null_distribution)) if len(self.null_distribution) else float("nan")
        )

    @property
    def null_std(self) -> float:
        return (
            float(np.std(self.null_distribution, ddof=1))
            if len(self.null_distribution) > 1
            else float("nan")
        )

    def percentile_of_observed(self) -> float:
        """Where the observed statistic falls in the null, in ``[0, 100]``."""
        if not len(self.null_distribution):
            return float("nan")
        return float(100.0 * np.mean(self.null_distribution <= self.observed))

    def summary(self) -> dict[str, float | str]:
        return {
            "statistic": self.statistic,
            "method": self.method,
            "observed": self.observed,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "null_q95": float(np.quantile(self.null_distribution, 0.95))
            if len(self.null_distribution)
            else float("nan"),
            "percentile_of_observed": self.percentile_of_observed(),
            "p_value": self.p_value,
            "n_permutations": float(len(self.null_distribution)),
        }


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap distribution of a metric computed on strategy returns."""

    metric: str
    point_estimate: float
    samples: np.ndarray
    confidence: float
    confidence_interval: tuple[float, float] = field(default=(float("nan"), float("nan")))

    def summary(self) -> dict[str, float]:
        low, high = self.confidence_interval
        return {
            "point_estimate": self.point_estimate,
            "bootstrap_mean": float(np.mean(self.samples)) if len(self.samples) else float("nan"),
            "ci_low": low,
            "ci_high": high,
            "confidence": self.confidence,
            "fraction_above_zero": float(np.mean(self.samples > 0))
            if len(self.samples)
            else float("nan"),
        }


def _clean_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float).ravel()
    values = values[~np.isnan(values)]
    if len(values) < 2:
        raise ValueError("need at least two non-missing returns to simulate")
    return values


def sample_returns(
    returns: pd.Series | np.ndarray,
    config: SimulationConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw ``(n_paths, horizon_days)`` simulated returns.

    Methods
    -------
    ``iid_bootstrap``
        Resample individual returns. Keeps the marginal distribution (fat tails
        included) but destroys volatility clustering.
    ``block_bootstrap``
        Resample contiguous blocks with circular wrap-around, preserving
        short-range dependence such as volatility clusters. The default.
    ``gaussian``
        Fit a normal distribution. Included mostly to show how badly it
        understates tail risk relative to the bootstrap.
    """
    config = config or SimulationConfig()
    values = _clean_returns(returns)
    rng = rng or np.random.default_rng(config.seed)
    shape = (config.n_paths, config.horizon_days)

    if config.method == "gaussian":
        return rng.normal(values.mean(), values.std(ddof=1), size=shape)

    if config.method == "iid_bootstrap":
        return values[rng.integers(0, len(values), size=shape)]

    # Block bootstrap: lay down ceil(horizon / block) blocks, then trim.
    block = min(config.block_size, len(values))
    n_blocks = int(np.ceil(config.horizon_days / block))
    starts = rng.integers(0, len(values), size=(config.n_paths, n_blocks))
    offsets = np.arange(block)
    # (n_paths, n_blocks, block) circular indices into the historical sample.
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(values)
    drawn = values[indices].reshape(config.n_paths, n_blocks * block)
    return drawn[:, : config.horizon_days]


def simulate_price_paths(
    returns: pd.Series | np.ndarray,
    config: SimulationConfig | None = None,
    *,
    initial_value: float = 1.0,
    store_paths: bool = False,
) -> PathSimulationResult:
    """Project an equity or price series forward by resampling its returns.

    Parameters
    ----------
    returns:
        Historical *simple* returns to resample.
    initial_value:
        Starting value of every simulated path.
    store_paths:
        Keep the full ``(n_paths, horizon_days)`` array. Off by default because
        it is the memory-hungry part and the summary rarely needs it.
    """
    config = config or SimulationConfig()
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")

    drawn = sample_returns(returns, config)
    growth = np.cumprod(1.0 + drawn, axis=1)
    # A path that loses everything stays dead rather than recovering.
    growth = np.maximum(growth, 0.0)
    paths = initial_value * growth

    terminal = paths[:, -1]
    running_peak = np.maximum.accumulate(paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(running_peak > 0, paths / running_peak - 1.0, -1.0)
    max_drawdowns = drawdowns.min(axis=1)

    terminal_return = terminal / initial_value - 1.0
    var_level = float(np.quantile(terminal_return, 0.05))
    tail = terminal_return[terminal_return <= var_level]

    return PathSimulationResult(
        terminal_values=terminal,
        quantiles={q: float(np.quantile(terminal, q)) for q in config.quantiles},
        probability_of_loss=float(np.mean(terminal < initial_value)),
        value_at_risk=var_level,
        conditional_value_at_risk=float(tail.mean()) if len(tail) else var_level,
        max_drawdowns=max_drawdowns,
        initial_value=initial_value,
        config=config,
        paths=paths if store_paths else None,
    )


def _net_returns(asset_returns: np.ndarray, positions: np.ndarray, cost_rate: float) -> np.ndarray:
    """Net strategy returns for already-lagged positions. Pure numpy, no pandas."""
    trades = np.empty_like(positions)
    trades[0] = positions[0]
    trades[1:] = np.diff(positions)
    return positions * asset_returns - np.abs(trades) * cost_rate


def _statistic_value(returns: np.ndarray, statistic: str) -> float:
    if statistic == "sharpe":
        return sharpe_ratio(returns)
    if statistic == "total_return":
        return float(np.prod(1.0 + returns) - 1.0)
    if statistic == "annualised_return":
        growth = float(np.prod(1.0 + returns))
        if growth <= 0:
            return -1.0
        return float(growth ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1.0)
    raise ValueError(
        f"statistic must be 'sharpe', 'total_return' or 'annualised_return', got {statistic!r}"
    )


def significance_test(
    close: pd.Series,
    signal: pd.Series,
    *,
    backtest_config: BacktestConfig | None = None,
    simulation_config: SimulationConfig | None = None,
    method: str = "rotation",
    statistic: str = "sharpe",
) -> SignificanceResult:
    """Test a strategy statistic against a resampled null.

    Methods
    -------
    ``rotation``
        Circularly shift the signal relative to the prices. Preserves the
        signal's autocorrelation - and therefore its turnover and costs - while
        destroying its alignment with future returns. This is the default and
        the fairer test.
    ``shuffle``
        Randomly permute the signal. Simpler, but the resulting null trades far
        more often than the real strategy, so it flatters the strategy.

    Returns
    -------
    SignificanceResult
        ``p_value`` is one-sided: the share of null draws at least as good as
        the observed statistic, with the usual ``(1 + k) / (1 + n)`` correction
        that keeps it strictly positive.
    """
    from ai_stock.backtest.engine import signal_to_positions, simple_returns  # noqa: PLC0415

    backtest_config = backtest_config or BacktestConfig()
    simulation_config = simulation_config or SimulationConfig()
    if method not in ("rotation", "shuffle"):
        raise ValueError(f"method must be 'rotation' or 'shuffle', got {method!r}")

    index = signal.index.intersection(close.index).sort_values()
    if len(index) < 10:
        raise ValueError("need at least ten overlapping bars for a significance test")

    prices = close.reindex(index).astype(float)
    aligned_signal = signal.reindex(index).astype(float).fillna(0.0)
    asset_returns = simple_returns(prices)
    cost_rate = backtest_config.total_cost_bps * _BPS

    def statistic_for(values: pd.Series) -> float:
        target = signal_to_positions(values, backtest_config, asset_returns=asset_returns)
        positions = target.shift(1).fillna(0.0).to_numpy()
        return _statistic_value(
            _net_returns(asset_returns.to_numpy(), positions, cost_rate), statistic
        )

    observed = statistic_for(aligned_signal)

    n = len(index)
    rng = np.random.default_rng(simulation_config.seed)
    raw_values = aligned_signal.to_numpy()
    null = np.empty(simulation_config.n_permutations, dtype=float)

    for i in range(simulation_config.n_permutations):
        if method == "rotation":
            # Avoid the identity rotation, which would just re-measure `observed`.
            shift = int(rng.integers(1, n)) if n > 1 else 0
            resampled = np.roll(raw_values, shift)
        else:
            resampled = rng.permutation(raw_values)
        null[i] = statistic_for(pd.Series(resampled, index=index))

    finite = null[np.isfinite(null)]
    if len(finite) == 0 or not np.isfinite(observed):
        p_value = float("nan")
    else:
        p_value = float((1.0 + np.sum(finite >= observed)) / (1.0 + len(finite)))

    return SignificanceResult(
        statistic=statistic,
        observed=observed,
        null_distribution=finite,
        p_value=p_value,
        method=method,
    )


def bootstrap_metric(
    returns: pd.Series | np.ndarray,
    config: SimulationConfig | None = None,
    *,
    metric: str | Callable[[np.ndarray], float] = "sharpe",
    confidence: float = 0.9,
) -> BootstrapResult:
    """Bootstrap a confidence interval for a metric of strategy returns.

    Uses the same resampling scheme as :func:`sample_returns` (block bootstrap
    by default) over the *realised* length of the input, so the interval
    reflects the sample size actually available.
    """
    config = config or SimulationConfig()
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    values = _clean_returns(returns)
    metric_name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
    evaluate: Callable[[np.ndarray], float] = (
        (lambda sample: _statistic_value(sample, metric)) if isinstance(metric, str) else metric
    )

    resample_config = SimulationConfig(
        n_paths=config.n_paths,
        horizon_days=len(values),
        method=config.method,
        block_size=config.block_size,
        n_permutations=config.n_permutations,
        seed=config.seed,
        quantiles=config.quantiles,
    )
    drawn = sample_returns(values, resample_config)
    samples = np.array([evaluate(row) for row in drawn], dtype=float)
    samples = samples[np.isfinite(samples)]

    if len(samples) == 0:
        interval = (float("nan"), float("nan"))
    else:
        alpha = (1.0 - confidence) / 2.0
        interval = (
            float(np.quantile(samples, alpha)),
            float(np.quantile(samples, 1.0 - alpha)),
        )

    return BootstrapResult(
        metric=metric_name,
        point_estimate=evaluate(values),
        samples=samples,
        confidence=confidence,
        confidence_interval=interval,
    )


def drawdown_distribution(paths: np.ndarray) -> np.ndarray:
    """Maximum drawdown of every simulated path in ``(n_paths, horizon)``."""
    if paths.ndim != 2:
        raise ValueError("paths must be a 2-D array of shape (n_paths, horizon)")
    return np.array([max_drawdown(path) for path in paths], dtype=float)
