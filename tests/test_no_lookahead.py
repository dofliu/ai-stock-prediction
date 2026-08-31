"""The tests that matter most: no future information may reach a feature.

If any indicator peeked ahead, ``test_features_are_truncation_invariant`` would
fail, because recomputing on a truncated series would change earlier values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_stock.config import FeatureConfig, WalkForwardConfig
from ai_stock.evaluation.walkforward import run_walk_forward
from ai_stock.features.builder import build_dataset, build_features, build_targets


@pytest.mark.parametrize("cut", [150, 300, 500, 899])
def test_features_are_truncation_invariant(ohlcv: pd.DataFrame, cut: int) -> None:
    """Features computed on bars 0..cut must equal the full-series values there.

    This is the operational definition of causality: what the model sees at
    bar t cannot depend on anything after t.
    """
    config = FeatureConfig()
    full = build_features(ohlcv, config)
    truncated = build_features(ohlcv.iloc[:cut], config)

    pd.testing.assert_frame_equal(truncated, full.iloc[:cut])


def test_every_feature_column_is_individually_causal(ohlcv: pd.DataFrame) -> None:
    """Name any column that changes when the future is removed."""
    config = FeatureConfig()
    full = build_features(ohlcv, config)
    truncated = build_features(ohlcv.iloc[:400], config)

    offenders = [
        column
        for column in full.columns
        if not np.allclose(
            truncated[column].to_numpy(dtype=float),
            full[column].iloc[:400].to_numpy(dtype=float),
            equal_nan=True,
        )
    ]
    assert offenders == [], f"these features used future data: {offenders}"


@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_forward_return_looks_strictly_forward(ohlcv: pd.DataFrame, horizon: int) -> None:
    targets = build_targets(ohlcv, FeatureConfig(horizon=horizon))
    close = ohlcv["close"].astype(float)
    expected = np.log(close.shift(-horizon)) - np.log(close)

    pd.testing.assert_series_equal(targets["forward_return"], expected.rename("forward_return"))
    # The last `horizon` bars cannot have a target yet.
    assert targets["forward_return"].tail(horizon).isna().all()


def test_direction_matches_the_sign_of_the_forward_return(ohlcv: pd.DataFrame) -> None:
    targets = build_targets(ohlcv, FeatureConfig(horizon=2))
    known = targets["forward_return"].notna()
    expected = (targets.loc[known, "forward_return"] > 0).astype(float)
    pd.testing.assert_series_equal(targets.loc[known, "direction"], expected.rename("direction"))


def test_neutral_band_drops_only_ambiguous_labels(ohlcv: pd.DataFrame) -> None:
    band = 0.01
    targets = build_targets(ohlcv, FeatureConfig(horizon=1, neutral_band=band))
    forward = targets["forward_return"]

    small = forward.abs() <= band
    assert targets.loc[small & forward.notna(), "direction"].isna().all()
    assert targets.loc[~small & forward.notna(), "direction"].notna().all()


def test_dataset_drops_warmup_and_unlabelled_tail(ohlcv: pd.DataFrame) -> None:
    horizon = 3
    dataset = build_dataset(ohlcv, FeatureConfig(horizon=horizon))

    assert dataset.features.notna().all().all()
    assert dataset.forward_return.notna().all()
    # Nothing from the final `horizon` bars may survive.
    assert dataset.index.max() <= ohlcv.index[-(horizon + 1)]
    assert len(dataset) < len(ohlcv)


def test_shuffled_targets_destroy_predictive_power(dataset) -> None:
    """A sanity check on the harness itself: no signal in, no signal out.

    If the walk-forward machinery leaked, a model trained on randomised labels
    would still score, because it could read the answer from the test window.
    """
    shuffled = dataset.forward_return.sample(frac=1.0, random_state=42)
    shuffled.index = dataset.index
    scrambled = type(dataset)(
        features=dataset.features,
        forward_return=shuffled,
        direction=(shuffled > 0).astype(float),
        close=dataset.close,
        config=dataset.config,
    )

    result = run_walk_forward(
        scrambled, "ridge", WalkForwardConfig(train_size=400, test_size=100, min_train_size=200)
    )
    assert abs(result.metrics()["ic_fold_mean"]) < 0.1


def test_measured_ic_does_not_exceed_the_theoretical_ceiling(market) -> None:
    """A leak would show up as an IC above what the true model can achieve."""
    ceiling = market.theoretical_information_coefficient()
    dataset = build_dataset(market.ohlcv)
    result = run_walk_forward(
        dataset, "ridge", WalkForwardConfig(train_size=400, test_size=100, min_train_size=200)
    )
    measured = result.metrics()["ic_fold_mean"]

    assert measured < ceiling + 0.10, (
        f"measured IC {measured:.4f} is implausibly close to or above the "
        f"theoretical ceiling {ceiling:.4f}; suspect leakage"
    )
