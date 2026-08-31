"""Turn raw OHLCV bars into a leakage-free supervised-learning dataset.

Two rules are enforced throughout, and are the whole point of this module:

1. **Features at bar t use only bars <= t.** Every indicator is causal and
   nothing is back-filled.
2. **Targets at bar t look strictly forward.** ``forward_return`` at ``t`` is
   the log return from the close of ``t`` to the close of ``t + horizon``,
   which is unknown at ``t`` - exactly what a forecaster must predict.

Features are also expressed as ratios, z-scores or bounded oscillators rather
than raw prices, so that a model trained on one price level transfers to
another instead of memorising the level itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_stock.config import FeatureConfig
from ai_stock.data.loaders import validate_ohlcv
from ai_stock.features import indicators as ind

__all__ = ["Dataset", "build_dataset", "build_features", "build_targets"]


@dataclass(frozen=True)
class Dataset:
    """Aligned features, targets and prices ready for walk-forward evaluation.

    All members share one index: every row is a bar at which the features are
    fully warmed up *and* the forward target exists.
    """

    features: pd.DataFrame
    forward_return: pd.Series
    direction: pd.Series
    close: pd.Series
    config: FeatureConfig

    def __len__(self) -> int:
        return len(self.features)

    @property
    def index(self) -> pd.Index:
        return self.features.index

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    @property
    def horizon(self) -> int:
        return self.config.horizon

    def target(self, *, classification: bool) -> pd.Series:
        """Return the classification or regression target."""
        return self.direction if classification else self.forward_return

    def slice(self, positions: np.ndarray | slice) -> Dataset:
        """Positional subset of the dataset, keeping every member aligned."""
        return Dataset(
            features=self.features.iloc[positions],
            forward_return=self.forward_return.iloc[positions],
            direction=self.direction.iloc[positions],
            close=self.close.iloc[positions],
            config=self.config,
        )


def _feature_frame(ohlcv: pd.DataFrame, config: FeatureConfig) -> dict[str, pd.Series]:
    """Compute the raw feature columns as a name -> series mapping."""
    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    volume = ohlcv["volume"].astype(float)

    columns: dict[str, pd.Series] = {}

    # --- Trailing returns (the cumulative return over the last k bars). ---
    for lag in config.return_lags:
        columns[f"ret_{lag}d"] = ind.log_returns(close, lag)

    # --- Trend: where price sits relative to its own moving averages. ---
    for window in config.sma_windows:
        columns[f"close_over_sma_{window}"] = close / ind.sma(close, window) - 1.0
    for window in config.ema_windows:
        columns[f"close_over_ema_{window}"] = close / ind.ema(close, window) - 1.0

    # --- Momentum. ---
    for window in config.momentum_windows:
        columns[f"roc_{window}"] = ind.roc(close, window)

    # --- Volatility (annualised) and its short/long ratio. ---
    vol_windows = sorted(config.volatility_windows)
    for window in vol_windows:
        columns[f"volatility_{window}"] = ind.realised_volatility(close, window)
    if len(vol_windows) >= 2:
        short, long = vol_windows[0], vol_windows[-1]
        columns["volatility_ratio"] = columns[f"volatility_{short}"] / columns[
            f"volatility_{long}"
        ].replace(0.0, np.nan)

    # --- Oscillators. ---
    columns[f"rsi_{config.rsi_window}"] = ind.rsi(close, config.rsi_window)

    fast, slow, signal = config.macd
    macd_frame = ind.macd(close, fast, slow, signal)
    columns["macd_norm"] = macd_frame["macd"] / close
    columns["macd_hist_norm"] = macd_frame["macd_hist"] / close

    bb_window, bb_k = config.bollinger
    bb_frame = ind.bollinger(close, bb_window, bb_k)
    columns["bb_percent_b"] = bb_frame["bb_percent_b"]
    columns["bb_bandwidth"] = bb_frame["bb_bandwidth"]

    k_window, d_window = config.stochastic
    stoch_frame = ind.stochastic(high, low, close, k_window, d_window)
    columns["stoch_k"] = stoch_frame["stoch_k"]
    columns["stoch_d"] = stoch_frame["stoch_d"]

    # --- Range / channel position. ---
    columns["atr_norm"] = ind.atr(high, low, close, config.atr_window) / close
    columns["donchian_position"] = ind.donchian_position(high, low, close, bb_window)

    # --- Volume. ---
    if config.include_volume_features:
        log_volume = np.log(volume.where(volume > 0))
        for window in config.volume_windows:
            mean = log_volume.rolling(window, min_periods=window).mean()
            std = log_volume.rolling(window, min_periods=window).std(ddof=1)
            columns[f"volume_z_{window}"] = (log_volume - mean) / std.where(std > 0)
        obv_series = ind.obv(close, volume)
        obv_window = max(config.volume_windows)
        turnover = volume.rolling(obv_window, min_periods=obv_window).mean() * obv_window
        columns[f"obv_slope_{obv_window}"] = obv_series.diff(obv_window) / turnover.where(
            turnover > 0
        )

    return columns


def build_features(ohlcv: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Build the causal feature matrix for ``ohlcv``.

    Rows retain their warm-up ``NaN`` values; :func:`build_dataset` is what
    drops them. Keeping them here makes the causality property easy to test.
    """
    config = config or FeatureConfig()
    validate_ohlcv(ohlcv, name="ohlcv")
    frame = pd.DataFrame(_feature_frame(ohlcv, config), index=ohlcv.index)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.index.name = ohlcv.index.name
    return frame


def build_targets(ohlcv: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Build forward-looking targets: ``forward_return`` and ``direction``.

    ``forward_return_t = log(close_{t+h} / close_t)`` and ``direction`` is its
    sign as ``{0, 1}``. When ``config.neutral_band > 0``, bars whose move is
    smaller than the band get ``NaN`` direction: they are genuinely ambiguous
    and training on them mostly teaches the model to fit noise.
    """
    config = config or FeatureConfig()
    close = ohlcv["close"].astype(float)
    horizon = config.horizon

    forward = (np.log(close).shift(-horizon) - np.log(close)).rename("forward_return")

    direction = pd.Series(np.nan, index=close.index, name="direction", dtype=float)
    is_known = forward.notna()
    direction[is_known] = (forward[is_known] > 0).astype(float)
    if config.neutral_band > 0:
        direction = direction.mask(forward.abs() <= config.neutral_band)

    return pd.DataFrame({"forward_return": forward, "direction": direction})


def build_dataset(ohlcv: pd.DataFrame, config: FeatureConfig | None = None) -> Dataset:
    """Assemble an aligned :class:`Dataset` from raw bars.

    Rows are kept only where every feature is warmed up and the forward return
    exists, so the last ``horizon`` bars are dropped by construction.

    Raises
    ------
    ValueError
        If no usable rows remain - almost always a series shorter than the
        longest indicator window.
    """
    config = config or FeatureConfig()
    features = build_features(ohlcv, config)
    targets = build_targets(ohlcv, config)

    usable = features.notna().all(axis=1) & targets["forward_return"].notna()
    if not usable.any():
        longest = max(
            (
                *config.return_lags,
                *config.sma_windows,
                *config.ema_windows,
                *config.momentum_windows,
                *config.volatility_windows,
                config.macd[1] + config.macd[2],
                config.bollinger[0],
                *config.volume_windows,
            )
        )
        raise ValueError(
            f"no usable rows: {len(ohlcv)} bars is too short for the configured windows "
            f"(need roughly {longest + config.horizon + 1} bars or more)"
        )

    kept = features.index[usable]
    return Dataset(
        features=features.loc[kept],
        forward_return=targets["forward_return"].loc[kept],
        direction=targets["direction"].loc[kept],
        close=ohlcv["close"].astype(float).loc[kept],
        config=config,
    )
