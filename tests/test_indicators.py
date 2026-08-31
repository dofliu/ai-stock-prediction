"""Indicator tests against hand-computed values and degenerate inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_stock.features import indicators as ind


@pytest.fixture
def prices() -> pd.Series:
    return pd.Series(
        [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.5, 15.0],
        index=pd.bdate_range("2024-01-01", periods=8),
        name="close",
    )


def test_sma_matches_manual_mean(prices: pd.Series) -> None:
    result = ind.sma(prices, 3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx((10.0 + 11.0 + 12.0) / 3)
    assert result.iloc[-1] == pytest.approx((14.0 + 13.5 + 15.0) / 3)


def test_ema_recursion_matches_manual_calculation(prices: pd.Series) -> None:
    span = 3
    alpha = 2.0 / (span + 1)
    result = ind.ema(prices, span)

    # adjust=False seeds the recursion with the first observation and only
    # masks the first span-1 outputs, so the running value starts at prices[0].
    expected = prices.iloc[0]
    for value in prices.iloc[1:]:
        expected = alpha * value + (1 - alpha) * expected
    assert result.iloc[-1] == pytest.approx(expected)
    assert result.iloc[: span - 1].isna().all()


def test_log_returns_are_differences_of_logs(prices: pd.Series) -> None:
    result = ind.log_returns(prices, 2)
    assert np.isnan(result.iloc[0])
    assert result.iloc[2] == pytest.approx(np.log(12.0 / 10.0))


def test_roc_is_a_simple_percentage_change(prices: pd.Series) -> None:
    result = ind.roc(prices, 3)
    assert result.iloc[3] == pytest.approx(11.0 / 10.0 - 1.0)


def test_rsi_is_bounded_and_directional() -> None:
    rising = pd.Series(np.arange(1.0, 40.0), index=pd.bdate_range("2024-01-01", periods=39))
    falling = rising.iloc[::-1].reset_index(drop=True)
    falling.index = rising.index

    assert ind.rsi(rising, 14).dropna().max() == pytest.approx(100.0)
    assert ind.rsi(falling, 14).dropna().min() == pytest.approx(0.0)

    values = ind.rsi(rising, 14).dropna()
    assert values.between(0.0, 100.0).all()


def test_rsi_on_a_flat_series_is_fifty(flat_ohlcv: pd.DataFrame) -> None:
    result = ind.rsi(flat_ohlcv["close"], 14).dropna()
    assert (result == 50.0).all()


def test_macd_histogram_is_line_minus_signal(prices: pd.Series) -> None:
    longer = pd.concat([prices] * 8, ignore_index=True)
    longer.index = pd.bdate_range("2024-01-01", periods=len(longer))
    frame = ind.macd(longer, 3, 6, 4)

    residual = (frame["macd"] - frame["macd_signal"] - frame["macd_hist"]).dropna()
    assert np.allclose(residual.to_numpy(), 0.0)


def test_bollinger_percent_b_is_zero_at_the_lower_band() -> None:
    series = pd.Series(
        np.concatenate([np.full(19, 100.0), [90.0]]),
        index=pd.bdate_range("2024-01-01", periods=20),
    )
    frame = ind.bollinger(series, 20, 2.0)
    last = frame.iloc[-1]

    assert last["bb_middle"] == pytest.approx(series.mean())
    assert last["bb_percent_b"] < 0.5
    assert last["bb_upper"] > last["bb_middle"] > last["bb_lower"]


def test_bollinger_on_a_flat_series_has_zero_width(flat_ohlcv: pd.DataFrame) -> None:
    frame = ind.bollinger(flat_ohlcv["close"], 20, 2.0).dropna(subset=["bb_middle"])
    assert np.allclose(frame["bb_bandwidth"].to_numpy(), 0.0)
    # Zero width means %B is undefined rather than infinite.
    assert frame["bb_percent_b"].isna().all()


def test_true_range_includes_the_overnight_gap() -> None:
    index = pd.bdate_range("2024-01-01", periods=2)
    high = pd.Series([10.0, 12.0], index=index)
    low = pd.Series([9.0, 11.5], index=index)
    close = pd.Series([9.5, 11.8], index=index)

    result = ind.true_range(high, low, close)
    # Second bar: range 0.5, gap up from 9.5 to 12.0 = 2.5 -> true range 2.5.
    assert result.iloc[1] == pytest.approx(2.5)


def test_atr_is_positive_and_warms_up(ohlcv: pd.DataFrame) -> None:
    result = ind.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
    assert result.iloc[:13].isna().all()
    assert (result.dropna() > 0).all()


def test_stochastic_hits_the_extremes() -> None:
    index = pd.bdate_range("2024-01-01", periods=20)
    close = pd.Series(np.linspace(10.0, 20.0, 20), index=index)
    frame = ind.stochastic(close, close, close, 14, 3)
    # A monotonically rising close is always at the top of its range.
    assert frame["stoch_k"].dropna().iloc[-1] == pytest.approx(100.0)


def test_stochastic_on_a_flat_window_is_the_midpoint(flat_ohlcv: pd.DataFrame) -> None:
    frame = ind.stochastic(
        flat_ohlcv["high"], flat_ohlcv["low"], flat_ohlcv["close"], 14, 3
    ).dropna()
    assert (frame["stoch_k"] == 50.0).all()


def test_donchian_position_is_bounded(ohlcv: pd.DataFrame) -> None:
    result = ind.donchian_position(ohlcv["high"], ohlcv["low"], ohlcv["close"], 20).dropna()
    assert result.between(0.0, 1.0).all()


def test_obv_accumulates_signed_volume() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series([10.0, 11.0, 10.5, 10.5], index=index)
    volume = pd.Series([100.0, 200.0, 300.0, 400.0], index=index)

    result = ind.obv(close, volume)
    # First bar has no direction (0), then +200, then -300, then flat (0).
    assert result.tolist() == [0.0, 200.0, -100.0, -100.0]


def test_realised_volatility_annualises() -> None:
    index = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(0)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))), index=index)

    daily = ind.realised_volatility(close, 20, annualise=False).dropna()
    annual = ind.realised_volatility(close, 20, annualise=True).dropna()
    assert np.allclose(annual.to_numpy(), daily.to_numpy() * np.sqrt(252))


@pytest.mark.parametrize(
    "call",
    [
        lambda s: ind.sma(s, 0),
        lambda s: ind.ema(s, 0),
        lambda s: ind.rsi(s, 0),
        lambda s: ind.roc(s, 0),
        lambda s: ind.log_returns(s, 0),
        lambda s: ind.realised_volatility(s, 1),
        lambda s: ind.bollinger(s, 1),
        lambda s: ind.macd(s, 26, 12),
        lambda s: ind.wilder_smooth(s, 0),
    ],
)
def test_invalid_windows_are_rejected(prices: pd.Series, call) -> None:
    with pytest.raises(ValueError):
        call(prices)
