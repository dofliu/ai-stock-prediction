"""Backtesting: signal to position to equity curve, with trading costs."""

from ai_stock.backtest.engine import (
    BacktestResult,
    run_backtest,
    signal_to_positions,
    simple_returns,
)

__all__ = ["BacktestResult", "run_backtest", "signal_to_positions", "simple_returns"]
