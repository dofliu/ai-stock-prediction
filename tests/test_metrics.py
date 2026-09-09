"""Metric tests against closed-form values and degenerate inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_stock.config import TRADING_DAYS_PER_YEAR
from ai_stock.evaluation.metrics import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    classification_metrics,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    financial_metrics,
    information_coefficient,
    max_drawdown,
    probabilistic_sharpe_ratio,
    regression_metrics,
    sharpe_ratio,
    sortino_ratio,
)


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0006, 0.011, 1000))


def test_sharpe_matches_its_definition(returns: pd.Series) -> None:
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_subtracts_the_risk_free_rate(returns: pd.Series) -> None:
    with_rf = sharpe_ratio(returns, risk_free_rate=0.05)
    assert with_rf < sharpe_ratio(returns)


def test_constant_returns_have_no_defined_sharpe() -> None:
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 50)))
    assert np.isnan(sharpe_ratio(pd.Series([0.01])))


def test_max_drawdown_is_the_worst_peak_to_trough() -> None:
    assert max_drawdown(pd.Series([1.0, 1.2, 0.9, 1.1])) == pytest.approx(-0.25)
    assert max_drawdown(pd.Series([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_annualised_return_compounds_geometrically() -> None:
    daily = pd.Series([0.001] * TRADING_DAYS_PER_YEAR)
    assert annualised_return(daily) == pytest.approx(1.001**252 - 1)


def test_total_loss_is_reported_as_minus_one() -> None:
    assert annualised_return(pd.Series([-1.0, 0.5])) == pytest.approx(-1.0)


def test_annualised_volatility_scales_with_the_square_root_of_time(
    returns: pd.Series,
) -> None:
    assert annualised_volatility(returns) == pytest.approx(
        returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def test_sortino_only_penalises_downside() -> None:
    upside_only = pd.Series([0.01, 0.02, 0.03, 0.01])
    assert np.isnan(sortino_ratio(upside_only))

    mixed = pd.Series([0.02, -0.01, 0.03, -0.005])
    assert sortino_ratio(mixed) > sharpe_ratio(mixed)


def test_calmar_divides_return_by_drawdown() -> None:
    daily = pd.Series([0.01, -0.02, 0.03, -0.01])
    equity = (1 + daily).cumprod()
    expected = annualised_return(daily) / abs(max_drawdown(equity))
    assert calmar_ratio(daily, equity) == pytest.approx(expected)


def test_calmar_is_undefined_without_a_drawdown() -> None:
    rising = pd.Series([0.01, 0.01, 0.01])
    assert np.isnan(calmar_ratio(rising, (1 + rising).cumprod()))


def test_information_coefficient_detects_perfect_and_inverted_forecasts() -> None:
    truth = pd.Series([0.01, -0.02, 0.03, -0.01, 0.005])
    assert information_coefficient(truth, truth) == pytest.approx(1.0)
    assert information_coefficient(truth, -truth) == pytest.approx(-1.0)
    assert np.isnan(information_coefficient(truth, pd.Series([0.0] * 5)))


def test_spearman_ignores_monotone_rescaling() -> None:
    truth = pd.Series([1.0, 2.0, 3.0, 4.0])
    forecast = pd.Series([10.0, 20.0, 300.0, 4000.0])
    assert information_coefficient(truth, forecast, method="spearman") == pytest.approx(1.0)


def test_information_coefficient_rejects_unknown_methods() -> None:
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="pearson"):
        information_coefficient(series, series, method="kendall")


def test_regression_metrics_on_a_perfect_forecast() -> None:
    truth = pd.Series([0.01, -0.02, 0.03])
    metrics = regression_metrics(truth, truth)

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["directional_accuracy"] == pytest.approx(1.0)
    assert metrics["n"] == 3


def test_regression_metrics_align_series_by_index() -> None:
    truth = pd.Series([0.01, np.nan, 0.03])
    forecast = pd.Series([0.02, 0.01, 0.02])
    assert regression_metrics(truth, forecast)["n"] == 2


def test_regression_metrics_on_empty_input() -> None:
    metrics = regression_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
    assert metrics["n"] == 0
    assert np.isnan(metrics["rmse"])


def test_directional_accuracy_ignores_flat_forecasts() -> None:
    truth = pd.Series([0.01, -0.02, 0.03])
    forecast = pd.Series([0.01, 0.0, 0.02])
    # Only the two non-zero forecasts count, and both have the right sign.
    assert regression_metrics(truth, forecast)["directional_accuracy"] == pytest.approx(1.0)


def test_classification_metrics_report_the_base_rate() -> None:
    truth = pd.Series([1.0, 1.0, 1.0, 0.0])
    metrics = classification_metrics(truth, pd.Series([1.0, 1.0, 1.0, 1.0]))

    assert metrics["base_rate"] == pytest.approx(0.75)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(1.0)


def test_roc_auc_is_undefined_for_a_single_class() -> None:
    truth = pd.Series([1.0, 1.0, 1.0])
    assert np.isnan(classification_metrics(truth, pd.Series([0.1, 0.2, 0.3]))["roc_auc"])


def test_probabilistic_sharpe_rises_with_track_record_length() -> None:
    rng = np.random.default_rng(1)
    short = pd.Series(rng.normal(0.001, 0.01, 60))
    long = pd.Series(np.tile(short.to_numpy(), 20))

    assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)
    assert 0.0 <= probabilistic_sharpe_ratio(long) <= 1.0


def test_probabilistic_sharpe_needs_a_minimum_sample() -> None:
    assert np.isnan(probabilistic_sharpe_ratio(pd.Series([0.01, 0.02])))


def test_expected_max_sharpe_is_zero_with_a_single_trial() -> None:
    assert expected_max_sharpe(1, 0.1) == 0.0
    assert expected_max_sharpe(50, 0.0) == 0.0


def test_expected_max_sharpe_grows_with_trials_and_dispersion() -> None:
    assert expected_max_sharpe(100, 0.1) > expected_max_sharpe(10, 0.1)
    assert expected_max_sharpe(100, 0.2) > expected_max_sharpe(100, 0.1)


def test_expected_max_sharpe_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 0.1)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -0.1)


def test_deflated_sharpe_matches_psr_with_one_trial(returns: pd.Series) -> None:
    assert deflated_sharpe_ratio(
        returns, n_trials=1, trial_sharpe_std=0.0
    ) == probabilistic_sharpe_ratio(returns)


def test_deflated_sharpe_is_never_kinder_than_psr(returns: pd.Series) -> None:
    # More trials (or more disagreement between them) raises the bar, so the
    # deflated verdict can only be equal to or harsher than the unconditional one.
    plain = probabilistic_sharpe_ratio(returns)
    deflated = deflated_sharpe_ratio(returns, n_trials=20, trial_sharpe_std=0.05)
    assert deflated <= plain


def test_financial_metrics_cover_returns_positions_and_risk(returns: pd.Series) -> None:
    positions = pd.Series(1.0, index=returns.index)
    metrics = financial_metrics(returns, positions=positions)

    assert metrics["n_periods"] == len(returns)
    assert metrics["total_return"] == pytest.approx(np.prod(1 + returns) - 1)
    assert metrics["avg_exposure"] == pytest.approx(1.0)
    assert metrics["time_in_market"] == pytest.approx(1.0)
    # A position held from day one and never changed barely turns over.
    assert metrics["annual_turnover"] == pytest.approx(TRADING_DAYS_PER_YEAR / len(returns))


def test_turnover_reflects_daily_flipping() -> None:
    index = pd.RangeIndex(100)
    returns = pd.Series(0.001, index=index)
    flipping = pd.Series([1.0 if i % 2 == 0 else -1.0 for i in index], index=index)

    metrics = financial_metrics(returns, positions=flipping)
    # 99 flips of 2 units each across 100 bars, annualised.
    assert metrics["annual_turnover"] == pytest.approx((1.0 + 99 * 2) / 100 * TRADING_DAYS_PER_YEAR)


def test_profit_factor_is_undefined_without_losses() -> None:
    assert np.isnan(financial_metrics(pd.Series([0.01, 0.02]))["profit_factor"])
