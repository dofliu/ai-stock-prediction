"""Backtest tests: execution timing, costs and position sizing.

The timing test is the important one - a backtest that trades on the same bar
it predicts will show a huge, entirely fictional edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_stock.backtest.engine import run_backtest, signal_to_positions, simple_returns
from ai_stock.config import BacktestConfig

FREE = BacktestConfig(cost_bps=0.0, slippage_bps=0.0)


def test_simple_returns_start_at_zero(toy_prices: pd.Series) -> None:
    result = simple_returns(toy_prices)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == pytest.approx(0.10)
    assert result.iloc[2] == pytest.approx(-0.10)


def test_position_is_lagged_by_exactly_one_bar(toy_prices: pd.Series) -> None:
    """A signal at t must be traded over t+1, never over t itself."""
    signal = pd.Series([1.0, -1.0, 1.0, 0.0], index=toy_prices.index)
    result = run_backtest(toy_prices, signal, FREE)

    assert result.positions.tolist() == [0.0, 1.0, -1.0, 1.0]
    # +10%, then short into a -10% bar, then long into a +10% bar.
    assert result.returns.round(10).tolist() == [0.0, 0.1, 0.1, 0.1]
    assert result.equity.iloc[-1] == pytest.approx(100_000 * 1.1**3)


def test_a_signal_that_knows_the_future_cannot_be_traded_today(
    toy_prices: pd.Series,
) -> None:
    """Perfect foresight applied to the *current* bar earns nothing extra.

    Shifting the perfect signal one bar later (so it "predicts" the bar it is
    already in) must not raise the equity curve, because the engine still
    executes with a one-bar delay.
    """
    perfect = np.sign(toy_prices.pct_change().shift(-1)).fillna(0.0)
    honest = run_backtest(toy_prices, perfect, FREE)
    too_late = run_backtest(toy_prices, perfect.shift(1).fillna(0.0), FREE)

    assert honest.equity.iloc[-1] > too_late.equity.iloc[-1]


def test_costs_are_charged_on_traded_notional(toy_prices: pd.Series) -> None:
    signal = pd.Series([1.0, -1.0, 1.0, 0.0], index=toy_prices.index)
    result = run_backtest(toy_prices, signal, BacktestConfig(cost_bps=10.0, slippage_bps=0.0))

    # Trades are 0 -> 1 (1 unit), 1 -> -1 (2 units), -1 -> 1 (2 units).
    assert (result.costs * 1e4).round(6).tolist() == [0.0, 10.0, 20.0, 20.0]
    assert result.total_costs == pytest.approx(50e-4)


def test_flat_signal_never_trades_and_never_pays(toy_prices: pd.Series) -> None:
    signal = pd.Series(0.0, index=toy_prices.index)
    result = run_backtest(toy_prices, signal, BacktestConfig(cost_bps=25.0))

    assert (result.positions == 0.0).all()
    assert result.total_costs == 0.0
    assert result.equity.iloc[-1] == pytest.approx(result.config.initial_equity)


def test_benchmark_pays_the_same_entry_cost(toy_prices: pd.Series) -> None:
    signal = pd.Series(0.0, index=toy_prices.index)
    config = BacktestConfig(cost_bps=10.0, slippage_bps=0.0)
    result = run_backtest(toy_prices, signal, config)

    expected_first = toy_prices.iloc[0] * 0 + (0.0 - 10e-4)
    assert result.benchmark_returns.iloc[0] == pytest.approx(expected_first)
    assert result.benchmark_metrics["avg_exposure"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("sizing", "signal", "expected"),
    [
        ("sign", [0.01, -0.01, 0.0], [1.0, -1.0, 0.0]),
        ("long_only", [0.01, -0.01, 0.0], [1.0, 0.0, 0.0]),
        ("proportional", [0.005, -0.02, 0.0], [0.5, -1.0, 0.0]),
    ],
)
def test_sizing_modes(sizing: str, signal: list[float], expected: list[float]) -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    config = BacktestConfig(sizing=sizing, scale=100.0, max_leverage=1.0)
    positions = signal_to_positions(pd.Series(signal, index=index), config)
    assert positions.tolist() == expected


def test_threshold_creates_a_dead_band() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    signal = pd.Series([0.001, 0.02, -0.02], index=index)
    positions = signal_to_positions(signal, BacktestConfig(threshold=0.01))
    assert positions.tolist() == [0.0, 1.0, -1.0]


def test_shorting_can_be_disabled() -> None:
    index = pd.bdate_range("2024-01-01", periods=2)
    signal = pd.Series([0.01, -0.01], index=index)
    positions = signal_to_positions(signal, BacktestConfig(allow_short=False))
    assert positions.tolist() == [1.0, 0.0]


def test_volatility_targeting_scales_positions_and_stays_causal() -> None:
    index = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.02, 60), index=index)
    signal = pd.Series(1.0, index=index)

    config = BacktestConfig(vol_target=0.10, vol_lookback=20, max_leverage=2.0)
    positions = signal_to_positions(signal, config, asset_returns=returns)

    # The warm-up window has no volatility estimate, so no position is taken.
    assert (positions.iloc[:19] == 0.0).all()
    assert (positions.iloc[19:] > 0).all()
    assert positions.max() <= 2.0


def test_volatility_targeting_requires_returns() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    with pytest.raises(ValueError, match="asset_returns"):
        signal_to_positions(pd.Series(1.0, index=index), BacktestConfig(vol_target=0.1))


def test_signal_and_price_indices_are_intersected(ohlcv: pd.DataFrame) -> None:
    close = ohlcv["close"]
    signal = pd.Series(1.0, index=close.index[100:200])
    result = run_backtest(close, signal, FREE)

    assert len(result.returns) == 100
    assert result.returns.index.equals(close.index[100:200])


def test_empty_or_misaligned_signal_is_rejected(ohlcv: pd.DataFrame) -> None:
    close = ohlcv["close"]
    with pytest.raises(ValueError, match="empty"):
        run_backtest(close, pd.Series(dtype=float))
    stray = pd.Series(1.0, index=pd.bdate_range("1990-01-01", periods=5))
    with pytest.raises(ValueError, match="overlapping"):
        run_backtest(close, stray)


def test_summary_reports_both_sides(ohlcv: pd.DataFrame) -> None:
    close = ohlcv["close"]
    signal = pd.Series(1.0, index=close.index)
    result = run_backtest(close, signal, FREE)
    summary = result.summary()

    assert set(summary.columns) == {"strategy", "buy_and_hold"}
    assert "sharpe" in summary.index
    # Always-long with no costs is buy-and-hold, one bar late.
    assert result.metrics["sharpe"] == pytest.approx(result.benchmark_metrics["sharpe"], abs=0.15)
