"""Multi-asset portfolio construction: alignment, weighting and diversification."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ai_stock.config import TRADING_DAYS_PER_YEAR
from ai_stock.portfolio import (
    align_sleeves,
    build_portfolio,
    diversification_ratio,
    effective_number_of_bets,
    sleeve_weights,
)


def _sleeves(
    n_sleeves: int = 4,
    *,
    n_days: int = 750,
    correlation: float = 0.0,
    volatilities: list[float] | None = None,
    seed: int = 0,
) -> dict[str, pd.Series]:
    """Sleeve returns with a known equicorrelation structure.

    Built from one shared factor plus independent noise, mixed so that every
    pair correlates at ``correlation`` in expectation.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n_days, name="date")
    common = rng.standard_normal(n_days)
    scales = volatilities or [0.01] * n_sleeves
    out = {}
    for i, scale in enumerate(scales):
        idiosyncratic = rng.standard_normal(n_days)
        mixed = math.sqrt(correlation) * common + math.sqrt(1.0 - correlation) * idiosyncratic
        out[f"S{i}"] = pd.Series(mixed * scale, index=index, name="returns")
    return out


def _equicorrelation(n: int, rho: float) -> np.ndarray:
    """Unit-variance covariance with a common off-diagonal correlation."""
    return np.full((n, n), rho) + np.diag(np.full(n, 1.0 - rho))


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def test_align_sleeves_keeps_only_the_shared_calendar() -> None:
    index = pd.bdate_range("2024-01-01", periods=10, name="date")
    sleeves = {
        "A": pd.Series(np.arange(10, dtype=float), index=index),
        "B": pd.Series(np.arange(8, dtype=float), index=index[2:]),
    }
    aligned = align_sleeves(sleeves)
    assert list(aligned.index) == list(index[2:])
    assert list(aligned.columns) == ["A", "B"]


def test_align_sleeves_rejects_calendars_that_barely_overlap() -> None:
    index = pd.bdate_range("2024-01-01", periods=10, name="date")
    sleeves = {
        "A": pd.Series(np.arange(5, dtype=float), index=index[:5]),
        "B": pd.Series(np.arange(5, dtype=float), index=index[5:]),
    }
    with pytest.raises(ValueError, match="fewer than two common dates"):
        align_sleeves(sleeves)


def test_align_sleeves_rejects_an_empty_universe() -> None:
    with pytest.raises(ValueError, match="no sleeves"):
        align_sleeves({})


def test_common_fraction_reports_what_the_intersection_cost() -> None:
    index = pd.bdate_range("2024-01-01", periods=100, name="date")
    rng = np.random.default_rng(1)
    sleeves = {
        "A": pd.Series(rng.normal(0, 0.01, 100), index=index),
        "B": pd.Series(rng.normal(0, 0.01, 90), index=index[10:]),
    }
    portfolio = build_portfolio(sleeves)
    assert portfolio.common_fraction == pytest.approx(0.9)
    assert len(portfolio.returns) == 90


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #
def test_equal_weights_are_equal_and_sum_to_one() -> None:
    frame = pd.DataFrame(_sleeves(3))
    weights = sleeve_weights(frame, "equal")
    assert weights.sum() == pytest.approx(1.0)
    assert weights.to_numpy() == pytest.approx(np.full(3, 1 / 3))


def test_inverse_vol_gives_the_quieter_sleeve_more_capital() -> None:
    frame = pd.DataFrame(_sleeves(2, volatilities=[0.01, 0.02], seed=3))
    weights = sleeve_weights(frame, "inverse_vol")
    assert weights.sum() == pytest.approx(1.0)
    assert weights["S0"] > weights["S1"]
    # Weights are inversely proportional to the realised volatilities.
    vols = frame.std(ddof=1)
    assert weights["S0"] / weights["S1"] == pytest.approx(vols["S1"] / vols["S0"])


def test_inverse_vol_refuses_a_sleeve_that_never_moves() -> None:
    index = pd.bdate_range("2024-01-01", periods=50, name="date")
    frame = pd.DataFrame(
        {"A": np.random.default_rng(0).normal(0, 0.01, 50), "B": np.zeros(50)}, index=index
    )
    with pytest.raises(ValueError, match="positive volatility"):
        sleeve_weights(frame, "inverse_vol")


def test_unknown_weight_scheme_raises() -> None:
    frame = pd.DataFrame(_sleeves(2))
    with pytest.raises(ValueError, match="unknown weight scheme"):
        sleeve_weights(frame, "max_sharpe")


# --------------------------------------------------------------------------- #
# The diversification arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho", [0.0, 0.25, 0.6, 0.95])
def test_effective_bets_matches_the_closed_form_for_equicorrelation(rho: float) -> None:
    n = 4
    weights = np.full(n, 1 / n)
    expected = n / (1 + (n - 1) * rho)
    assert effective_number_of_bets(_equicorrelation(n, rho), weights) == pytest.approx(expected)


def test_perfectly_correlated_sleeves_are_one_bet() -> None:
    weights = np.full(3, 1 / 3)
    assert effective_number_of_bets(_equicorrelation(3, 1.0), weights) == pytest.approx(1.0)
    assert diversification_ratio(_equicorrelation(3, 1.0), weights) == pytest.approx(1.0)


def test_uncorrelated_sleeves_give_a_root_n_diversification_ratio() -> None:
    for n in (2, 4, 9):
        weights = np.full(n, 1 / n)
        assert diversification_ratio(np.eye(n), weights) == pytest.approx(math.sqrt(n))


def test_negatively_correlated_sleeves_beat_independence() -> None:
    """A hedge is worth more than an independent bet, and the count says so."""
    hedged = np.array([[1.0, -0.5], [-0.5, 1.0]])
    weights = np.full(2, 0.5)
    assert effective_number_of_bets(hedged, weights) == pytest.approx(4.0)
    assert effective_number_of_bets(hedged, weights) > effective_number_of_bets(np.eye(2), weights)


def test_diversification_ratio_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        diversification_ratio(np.ones((2, 3)), np.full(2, 0.5))
    with pytest.raises(ValueError, match="weight"):
        diversification_ratio(np.eye(3), np.full(2, 0.5))
    with pytest.raises(ValueError, match="non-negative"):
        diversification_ratio(np.eye(2), np.array([1.5, -0.5]))


def test_a_zero_variance_portfolio_reports_nan_rather_than_dividing_by_zero() -> None:
    assert math.isnan(diversification_ratio(np.zeros((2, 2)), np.full(2, 0.5)))


# --------------------------------------------------------------------------- #
# build_portfolio
# --------------------------------------------------------------------------- #
def test_portfolio_returns_are_the_weighted_sum_of_the_sleeves() -> None:
    sleeves = _sleeves(3, seed=5)
    portfolio = build_portfolio(sleeves)
    expected = sum(series / 3 for series in sleeves.values())
    assert portfolio.returns.to_numpy() == pytest.approx(expected.to_numpy())
    assert portfolio.equity.iloc[-1] == pytest.approx(
        float(np.prod(1.0 + portfolio.returns.to_numpy()))
    )


def test_correlated_sleeves_are_fewer_bets_than_independent_ones() -> None:
    independent = build_portfolio(_sleeves(4, correlation=0.0, seed=7)).metrics()
    correlated = build_portfolio(_sleeves(4, correlation=0.7, seed=7)).metrics()

    assert independent["effective_bets"] == pytest.approx(4.0, abs=0.3)
    assert correlated["effective_bets"] == pytest.approx(4 / (1 + 3 * 0.7), abs=0.25)
    assert correlated["effective_bets"] < independent["effective_bets"]
    assert correlated["mean_correlation"] == pytest.approx(0.7, abs=0.05)


def test_the_independence_sharpe_is_the_diversification_the_correlation_withheld() -> None:
    """Correlation cannot move the mean, only the volatility - so only the Sharpe moves."""
    metrics = build_portfolio(_sleeves(4, correlation=0.7, seed=11)).metrics()
    assert metrics["sharpe_if_independent"] > metrics["sharpe"]
    assert metrics["sharpe_diversification_gap"] == pytest.approx(
        metrics["sharpe_if_independent"] - metrics["sharpe"]
    )

    # With no correlation left to remove, the two readings coincide.
    diagonal = build_portfolio(_sleeves(4, correlation=0.0, seed=11)).metrics()
    assert diagonal["sharpe_if_independent"] == pytest.approx(diagonal["sharpe"], rel=0.05)


def test_sharpe_if_independent_uses_the_realised_mean() -> None:
    portfolio = build_portfolio(_sleeves(3, correlation=0.5, seed=13))
    volatilities = portfolio.sleeves.std(ddof=1).to_numpy()
    weights = portfolio.weights.to_numpy()
    expected = (
        portfolio.returns.mean()
        / math.sqrt(float(np.sum((weights * volatilities) ** 2)))
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    assert portfolio.sharpe_if_independent() == pytest.approx(expected)


def test_risk_contributions_sum_to_one_and_diverge_from_the_weights() -> None:
    portfolio = build_portfolio(
        _sleeves(3, correlation=0.4, volatilities=[0.005, 0.01, 0.03], seed=17)
    )
    contributions = portfolio.risk_contributions()
    assert contributions.sum() == pytest.approx(1.0)
    # Equal capital is not equal risk: the loudest sleeve carries most of it.
    assert contributions["S2"] > portfolio.weights["S2"]
    assert contributions["S0"] < portfolio.weights["S0"]


def test_inverse_vol_evens_out_the_risk_shares() -> None:
    sleeves = _sleeves(3, correlation=0.3, volatilities=[0.005, 0.01, 0.03], seed=19)
    equal = build_portfolio(sleeves, scheme="equal").risk_contributions()
    parity = build_portfolio(sleeves, scheme="inverse_vol").risk_contributions()
    assert parity.std(ddof=0) < equal.std(ddof=0)


def test_sleeve_table_reports_every_sleeve_once() -> None:
    portfolio = build_portfolio(_sleeves(4, seed=23))
    table = portfolio.sleeve_table()
    assert list(table.index) == portfolio.symbols
    assert table.index.name == "symbol"
    assert table["weight"].sum() == pytest.approx(1.0)
    assert set(table.columns) == {
        "weight",
        "annualised_volatility",
        "annualised_return",
        "sharpe",
        "risk_contribution",
    }


def test_a_lone_sleeve_is_exactly_one_bet() -> None:
    metrics = build_portfolio(_sleeves(1, seed=29)).metrics()
    assert metrics["n_sleeves"] == 1.0
    assert metrics["effective_bets"] == pytest.approx(1.0)
    assert math.isnan(metrics["mean_correlation"])


# --------------------------------------------------------------------------- #
# Sleeves vs the shares they trade
# --------------------------------------------------------------------------- #
def test_asset_correlation_is_absent_unless_supplied() -> None:
    portfolio = build_portfolio(_sleeves(3, seed=31))
    assert portfolio.assets is None
    assert portfolio.asset_correlation() is None
    metrics = portfolio.metrics()
    assert math.isnan(metrics["mean_asset_correlation"])
    assert math.isnan(metrics["asset_effective_bets"])


def test_sleeves_can_decorrelate_from_the_shares_they_trade() -> None:
    """The case that matters: near-independent sleeves on tightly correlated names."""
    metrics = build_portfolio(
        _sleeves(4, correlation=0.02, seed=37),
        asset_returns_by_symbol=_sleeves(4, correlation=0.8, seed=41),
    ).metrics()

    assert metrics["mean_asset_correlation"] == pytest.approx(0.8, abs=0.05)
    assert metrics["mean_correlation"] < metrics["mean_asset_correlation"]
    assert metrics["asset_effective_bets"] == pytest.approx(4 / (1 + 3 * 0.8), abs=0.2)
    assert metrics["effective_bets"] > metrics["asset_effective_bets"]


def test_asset_returns_are_cut_to_the_sleeve_calendar() -> None:
    index = pd.bdate_range("2024-01-01", periods=60, name="date")
    rng = np.random.default_rng(43)
    sleeves = {
        "A": pd.Series(rng.normal(0, 0.01, 60), index=index),
        "B": pd.Series(rng.normal(0, 0.01, 50), index=index[10:]),
    }
    assets = {
        "A": pd.Series(rng.normal(0, 0.02, 60), index=index),
        "B": pd.Series(rng.normal(0, 0.02, 60), index=index),
    }
    portfolio = build_portfolio(sleeves, asset_returns_by_symbol=assets)
    assert portfolio.assets is not None
    assert list(portfolio.assets.index) == list(portfolio.sleeves.index)
    assert list(portfolio.assets.columns) == list(portfolio.sleeves.columns)


def test_asset_returns_must_cover_the_same_symbols() -> None:
    with pytest.raises(ValueError, match="exactly the same symbols"):
        build_portfolio(_sleeves(3, seed=47), asset_returns_by_symbol=_sleeves(2, seed=47))


def test_asset_returns_must_cover_every_date_the_sleeves_trade() -> None:
    """A gap would leave the asset covariance estimated pair by pair on different samples."""
    index = pd.bdate_range("2024-01-01", periods=40, name="date")
    rng = np.random.default_rng(53)
    sleeves = {name: pd.Series(rng.normal(0, 0.01, 40), index=index) for name in ("A", "B")}
    assets = {
        "A": pd.Series(rng.normal(0, 0.02, 40), index=index),
        "B": pd.Series(rng.normal(0, 0.02, 35), index=index[:35]),
    }
    with pytest.raises(ValueError, match="5 date\\(s\\) are missing"):
        build_portfolio(sleeves, asset_returns_by_symbol=assets)


def test_build_portfolio_rejects_an_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="unknown weight scheme"):
        build_portfolio(_sleeves(2), scheme="mean_variance")
