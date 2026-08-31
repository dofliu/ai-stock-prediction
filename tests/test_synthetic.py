"""Tests for the synthetic market generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_stock.config import TRADING_DAYS_PER_YEAR, SyntheticConfig
from ai_stock.data.synthetic import generate_market, generate_ohlcv


def test_bars_are_internally_consistent(ohlcv: pd.DataFrame) -> None:
    body_high = ohlcv[["open", "close"]].max(axis=1)
    body_low = ohlcv[["open", "close"]].min(axis=1)

    assert (ohlcv["high"] >= body_high).all()
    assert (ohlcv["low"] <= body_low).all()
    assert (ohlcv["high"] >= ohlcv["low"]).all()
    assert (ohlcv[["open", "high", "low", "close"]] > 0).all().all()
    assert (ohlcv["volume"] > 0).all()


def test_index_is_sorted_unique_business_days(ohlcv: pd.DataFrame) -> None:
    assert isinstance(ohlcv.index, pd.DatetimeIndex)
    assert ohlcv.index.is_monotonic_increasing
    assert not ohlcv.index.has_duplicates
    assert ohlcv.index.name == "date"
    assert (ohlcv.index.dayofweek < 5).all()


def test_same_seed_reproduces_identical_data(synthetic_config: SyntheticConfig) -> None:
    first = generate_ohlcv(synthetic_config)
    second = generate_ohlcv(synthetic_config)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_data(synthetic_config: SyntheticConfig) -> None:
    from dataclasses import replace

    other = generate_ohlcv(replace(synthetic_config, seed=999))
    assert not np.allclose(other["close"].to_numpy(), generate_ohlcv(synthetic_config)["close"])


def test_overrides_are_applied() -> None:
    frame = generate_ohlcv(n_days=64, seed=3, initial_price=42.0)
    assert len(frame) == 64
    # The first close reflects one day of return from the initial price.
    assert 20.0 < frame["close"].iloc[0] < 90.0


def test_planted_edge_creates_measurable_autocorrelation() -> None:
    market = generate_market(SyntheticConfig(n_days=8000, seed=5))
    returns = market.log_returns.dropna()
    # The default configuration is net mean-reverting at lags 2..5.
    assert float(returns.autocorr(3)) < -0.01
    assert market.theoretical_information_coefficient() > 0.05


def test_efficient_market_has_no_predictable_component() -> None:
    market = generate_market(SyntheticConfig(n_days=8000, seed=5, ar1=0.0, reversion=0.0))
    returns = market.log_returns.dropna()

    assert abs(float(returns.autocorr(1))) < 0.05
    # With no AR terms the conditional mean only moves when the regime switches,
    # so it explains almost nothing of the realised return.
    assert abs(market.theoretical_information_coefficient()) < 0.05


def test_volatility_clustering_is_present() -> None:
    market = generate_market(SyntheticConfig(n_days=8000, seed=8))
    absolute = market.log_returns.dropna().abs()
    assert float(absolute.autocorr(1)) > 0.05


def test_realised_volatility_matches_the_configured_level() -> None:
    config = SyntheticConfig(
        n_days=8000,
        seed=2,
        regime_drifts=(0.0,),
        regime_vols=(0.20,),
        regime_persistence=(1.0,),
        ar1=0.0,
        reversion=0.0,
    )
    market = generate_market(config)
    realised = market.log_returns.dropna().std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert realised == pytest.approx(0.20, rel=0.12)


def test_conditional_mean_and_volatility_align_with_prices(market) -> None:
    assert len(market.conditional_mean) == len(market.ohlcv)
    assert market.conditional_mean.index.equals(market.ohlcv.index)
    assert (market.volatility > 0).all()
    assert market.regime.between(0, len(market.config.regime_drifts) - 1).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_days": 1}, "n_days"),
        ({"regime_vols": (0.1, 0.2)}, "align"),
        ({"garch_alpha": 0.5, "garch_beta": 0.6}, "stationarity"),
        ({"fat_tail_df": 1.5}, "fat_tail_df"),
        ({"initial_price": 0.0}, "initial_price"),
        ({"regime_persistence": (1.5, 0.9, 0.9)}, "regime_persistence"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SyntheticConfig(**kwargs)


def test_gaussian_shocks_are_supported() -> None:
    market = generate_market(SyntheticConfig(n_days=500, seed=1, fat_tail_df=None))
    assert market.log_returns.dropna().notna().all()
