"""Turn a forecast into a position, and a position into an equity curve.

**Timing convention** - the single most important thing in this file, and the
place most backtests quietly cheat:

``signal_t`` is derived from information available at the *close* of bar ``t``.
The resulting position is therefore only held over bar ``t + 1``::

    position_held_during(t+1) = size(signal_t)
    strategy_return(t+1)      = position_held_during(t+1) * asset_return(t+1)
                                - |position(t+1) - position(t)| * cost_rate

In code this is one ``shift(1)``. Without it, the backtest trades on a bar
using that same bar's outcome and every model looks brilliant.

Costs are charged on traded notional, so a signal that flips daily pays for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_stock.config import TRADING_DAYS_PER_YEAR, BacktestConfig
from ai_stock.evaluation.metrics import financial_metrics

__all__ = ["BacktestResult", "run_backtest", "signal_to_positions", "simple_returns"]

_BPS = 1e-4


@dataclass(frozen=True)
class BacktestResult:
    """Everything the simulation produced, plus the benchmark it must beat."""

    positions: pd.Series
    """Position actually held during each bar (already lagged by one bar)."""
    asset_returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    returns: pd.Series
    """Net strategy returns after costs."""
    equity: pd.Series
    benchmark_returns: pd.Series
    benchmark_equity: pd.Series
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    config: BacktestConfig

    @property
    def total_costs(self) -> float:
        """Cumulative cost drag as a fraction of starting equity."""
        return float(self.costs.sum())

    def summary(self) -> pd.DataFrame:
        """Strategy vs. buy-and-hold as a two-column table."""
        keys = sorted(set(self.metrics) | set(self.benchmark_metrics))
        return pd.DataFrame(
            {
                "strategy": [self.metrics.get(k, float("nan")) for k in keys],
                "buy_and_hold": [self.benchmark_metrics.get(k, float("nan")) for k in keys],
            },
            index=pd.Index(keys, name="metric"),
        )

    def excess_sharpe(self) -> float:
        """Strategy Sharpe minus buy-and-hold Sharpe."""
        return float(self.metrics["sharpe"] - self.benchmark_metrics["sharpe"])


def simple_returns(close: pd.Series) -> pd.Series:
    """Simple (arithmetic) period returns; the first bar is 0, not NaN.

    Simple returns are used rather than log returns because portfolio equity
    compounds arithmetically: ``equity_t = equity_{t-1} * (1 + r_t)``.
    """
    returns = close / close.shift(1) - 1.0
    return returns.fillna(0.0).rename("asset_return")


def _trailing_volatility(returns: pd.Series, lookback: int) -> pd.Series:
    """Annualised trailing volatility, causal by construction."""
    return returns.rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )


def signal_to_positions(
    signal: pd.Series,
    config: BacktestConfig | None = None,
    *,
    asset_returns: pd.Series | None = None,
) -> pd.Series:
    """Map a signal to a *target* position, before the one-bar execution lag.

    The returned series is indexed like ``signal``: entry ``t`` is the position
    decided at the close of ``t``. :func:`run_backtest` applies the lag.

    Sizing modes
    ------------
    ``sign``
        Full size long or short once the signal clears ``threshold``.
    ``long_only``
        Full size long, otherwise flat - the realistic mode for accounts that
        cannot short.
    ``proportional``
        ``signal * scale`` clipped to ``max_leverage``: conviction-weighted.
    """
    config = config or BacktestConfig()
    values = signal.astype(float)

    inside_band = values.abs() <= config.threshold

    if config.sizing == "sign":
        target = np.sign(values) * config.max_leverage
    elif config.sizing == "long_only":
        target = (values > config.threshold).astype(float) * config.max_leverage
    else:  # proportional
        target = (values * config.scale).clip(-config.max_leverage, config.max_leverage)

    target = pd.Series(target, index=signal.index, dtype=float).where(~inside_band, 0.0)

    if not config.allow_short:
        target = target.clip(lower=0.0)

    if config.vol_target is not None:
        if asset_returns is None:
            raise ValueError("vol_target requires asset_returns to size positions")
        realised = _trailing_volatility(asset_returns.reindex(signal.index), config.vol_lookback)
        # Unknown volatility (warm-up) means no informed size: stay flat.
        scaler = (config.vol_target / realised.where(realised > 0)).fillna(0.0)
        target = (target * scaler).clip(-config.max_leverage, config.max_leverage)

    return target.fillna(0.0).rename("target_position")


def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Simulate trading ``signal`` on ``close`` and compare with buy-and-hold.

    Parameters
    ----------
    close:
        Close prices covering at least the signal's index.
    signal:
        Model output aligned to bar closes. Only its overlap with ``close`` is
        traded; bars without a signal are treated as flat.
    config:
        Sizing and cost settings.

    Notes
    -----
    With a multi-day forecast horizon the position is still refreshed every
    bar from the latest forecast, which is the usual overlapping-signal
    convention. Buy-and-hold pays the same entry cost, so the comparison is
    like for like.
    """
    config = config or BacktestConfig()
    if signal.empty:
        raise ValueError("signal is empty")

    index = signal.index.intersection(close.index).sort_values()
    if len(index) < 2:
        raise ValueError("need at least two overlapping bars of price and signal")

    prices = close.reindex(index).astype(float)
    aligned_signal = signal.reindex(index).astype(float).fillna(0.0)

    asset_returns = simple_returns(prices)
    target = signal_to_positions(aligned_signal, config, asset_returns=asset_returns)

    # The one-bar lag: a position decided at the close of t is held over t+1.
    positions = target.shift(1).fillna(0.0).rename("position")

    trades = positions.diff()
    trades.iloc[0] = positions.iloc[0]  # opening the very first position is a trade
    cost_rate = config.total_cost_bps * _BPS
    costs = (trades.abs() * cost_rate).rename("cost")

    gross_returns = (positions * asset_returns).rename("gross_return")
    returns = (gross_returns - costs).rename("strategy_return")
    equity = (config.initial_equity * (1.0 + returns).cumprod()).rename("equity")

    benchmark_returns = asset_returns.copy()
    benchmark_returns.iloc[0] = benchmark_returns.iloc[0] - cost_rate  # same entry cost
    benchmark_returns = benchmark_returns.rename("benchmark_return")
    benchmark_equity = (config.initial_equity * (1.0 + benchmark_returns).cumprod()).rename(
        "benchmark_equity"
    )

    return BacktestResult(
        positions=positions,
        asset_returns=asset_returns,
        gross_returns=gross_returns,
        costs=costs,
        returns=returns,
        equity=equity,
        benchmark_returns=benchmark_returns,
        benchmark_equity=benchmark_equity,
        metrics=financial_metrics(returns, equity=equity, positions=positions),
        benchmark_metrics=financial_metrics(
            benchmark_returns,
            equity=benchmark_equity,
            positions=pd.Series(1.0, index=index),
        ),
        config=config,
    )
