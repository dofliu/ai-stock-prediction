"""Shared fixtures. Datasets are kept small so the suite stays fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_stock.config import FeatureConfig, SyntheticConfig, WalkForwardConfig
from ai_stock.data.synthetic import generate_market, generate_ohlcv
from ai_stock.features.builder import build_dataset


@pytest.fixture(scope="session")
def synthetic_config() -> SyntheticConfig:
    return SyntheticConfig(n_days=900, seed=1234)


@pytest.fixture(scope="session")
def ohlcv(synthetic_config: SyntheticConfig) -> pd.DataFrame:
    return generate_ohlcv(synthetic_config)


@pytest.fixture(scope="session")
def market(synthetic_config: SyntheticConfig):
    return generate_market(synthetic_config)


@pytest.fixture(scope="session")
def feature_config() -> FeatureConfig:
    return FeatureConfig()


@pytest.fixture(scope="session")
def dataset(ohlcv: pd.DataFrame, feature_config: FeatureConfig):
    return build_dataset(ohlcv, feature_config)


@pytest.fixture(scope="session")
def walk_forward_config() -> WalkForwardConfig:
    return WalkForwardConfig(train_size=400, test_size=100, min_train_size=200)


@pytest.fixture
def toy_prices() -> pd.Series:
    """Four bars with exactly +/-10% moves, for hand-checkable backtests."""
    index = pd.bdate_range("2024-01-01", periods=4, name="date")
    return pd.Series([100.0, 110.0, 99.0, 108.9], index=index, name="close")


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """A perfectly flat market: the degenerate case indicators must survive."""
    index = pd.bdate_range("2024-01-01", periods=120, name="date")
    return pd.DataFrame(
        {
            "open": np.full(120, 50.0),
            "high": np.full(120, 50.0),
            "low": np.full(120, 50.0),
            "close": np.full(120, 50.0),
            "volume": np.full(120, 1000.0),
        },
        index=index,
    )
