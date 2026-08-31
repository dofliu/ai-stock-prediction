"""Technical indicators.

Every function here is **causal**: the value at bar ``t`` depends only on bars
``<= t``. Warm-up periods are returned as ``NaN`` rather than back-filled,
because silently filling them is one of the easiest ways to leak the future
into a backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_stock.config import TRADING_DAYS_PER_YEAR

__all__ = [
    "atr",
    "bollinger",
    "donchian_position",
    "ema",
    "log_returns",
    "macd",
    "obv",
    "realised_volatility",
    "roc",
    "rsi",
    "sma",
    "stochastic",
    "true_range",
    "wilder_smooth",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields ``NaN`` instead of ``inf`` on zeros."""
    return numerator / denominator.where(denominator != 0)


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over ``periods`` bars.

    >>> s = pd.Series([100.0, 110.0, 121.0])
    >>> log_returns(s).round(6).tolist()
    [nan, 0.09531, 0.09531]
    """
    if periods < 1:
        raise ValueError("periods must be >= 1")
    return np.log(close).diff(periods).rename(f"log_return_{periods}")


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` bars."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.rolling(window, min_periods=window).mean().rename(f"sma_{window}")


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average (``adjust=False``, the trading convention)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.ewm(span=window, adjust=False, min_periods=window).mean().rename(f"ema_{window}")


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing, i.e. an EMA with ``alpha = 1 / window``."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def roc(close: pd.Series, window: int) -> pd.Series:
    """Rate of change: ``close_t / close_{t-window} - 1``."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return (close / close.shift(window) - 1.0).rename(f"roc_{window}")


def realised_volatility(close: pd.Series, window: int, *, annualise: bool = True) -> pd.Series:
    """Rolling standard deviation of daily log returns."""
    if window < 2:
        raise ValueError("window must be >= 2")
    returns = np.log(close).diff()
    vol = returns.rolling(window, min_periods=window).std(ddof=1)
    if annualise:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol.rename(f"volatility_{window}")


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, bounded to ``[0, 100]``.

    A flat series has no losses, which would divide by zero; those bars are
    reported as the conventional 100 (all gains) or 50 (no movement at all).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = wilder_smooth(gains, window)
    avg_loss = wilder_smooth(losses, window)

    rs = _safe_divide(avg_gain, avg_loss)
    result = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0: pure gains -> 100, or a completely flat window -> 50.
    flat = (avg_loss == 0) & (avg_gain == 0)
    only_gains = (avg_loss == 0) & (avg_gain > 0)
    result = result.mask(only_gains, 100.0).mask(flat, 50.0)
    return result.rename(f"rsi_{window}")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, its signal line and the histogram (line minus signal)."""
    if not 0 < fast < slow:
        raise ValueError("require 0 < fast < slow")
    if signal < 1:
        raise ValueError("signal must be >= 1")
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger bands plus %B and bandwidth.

    Uses the population standard deviation (``ddof=0``), matching the original
    formulation. ``percent_b`` is 0 at the lower band and 1 at the upper one.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if k <= 0:
        raise ValueError("k must be positive")
    middle = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = middle + k * std
    lower = middle - k * std
    width = upper - lower
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_percent_b": _safe_divide(close - lower, width),
            "bb_bandwidth": _safe_divide(width, middle),
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: the greatest of today's range and the two gap ranges."""
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1).rename("true_range")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    return wilder_smooth(true_range(high, low, close), window).rename(f"atr_{window}")


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator %K and its %D moving average, in ``[0, 100]``."""
    if k_window < 1 or d_window < 1:
        raise ValueError("k_window and d_window must be >= 1")
    lowest = low.rolling(k_window, min_periods=k_window).min()
    highest = high.rolling(k_window, min_periods=k_window).max()
    percent_k = 100.0 * _safe_divide(close - lowest, highest - lowest)
    # A perfectly flat window has no range; treat it as the midpoint.
    percent_k = percent_k.mask((highest == lowest) & highest.notna(), 50.0)
    percent_d = percent_k.rolling(d_window, min_periods=d_window).mean()
    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_d})


def donchian_position(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Where the close sits inside the Donchian channel, in ``[0, 1]``.

    The channel is built from the ``window`` bars *ending at* ``t``, so the
    value is available at the close of ``t``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    upper = high.rolling(window, min_periods=window).max()
    lower = low.rolling(window, min_periods=window).min()
    position = _safe_divide(close - lower, upper - lower)
    position = position.mask((upper == lower) & upper.notna(), 0.5)
    return position.rename(f"donchian_position_{window}")


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: volume signed by the direction of the close."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum().rename("obv")
