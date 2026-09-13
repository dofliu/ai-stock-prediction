"""Universe screening: ranking, failure isolation and multiple-testing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_stock.config import (
    ExperimentConfig,
    SimulationConfig,
    SyntheticConfig,
    WalkForwardConfig,
)
from ai_stock.data.loaders import save_csv
from ai_stock.data.synthetic import generate_ohlcv
from ai_stock.evaluation.multiple_testing import (
    benjamini_hochberg,
    bonferroni_threshold,
    expected_false_positives,
)
from ai_stock.pipeline import load_universe, screen_universe
from ai_stock.reporting.studies import render_screen_report


@pytest.fixture(scope="module")
def screen_config() -> ExperimentConfig:
    return ExperimentConfig(
        walk_forward=WalkForwardConfig(train_size=400, test_size=100),
        simulation=SimulationConfig(seed=0),
    )


@pytest.fixture(scope="module")
def universe() -> dict[str, pd.DataFrame]:
    base = SyntheticConfig(n_days=900, seed=7)
    return {
        "AAA": generate_ohlcv(replace(base, seed=11)),
        "BBB": generate_ohlcv(replace(base, seed=22)),
        "CCC": generate_ohlcv(replace(base, seed=33, ar1=0.0, reversion=0.0)),
    }


# --------------------------------------------------------------------------- #
# Multiple-testing corrections
# --------------------------------------------------------------------------- #
def test_benjamini_hochberg_matches_the_definition() -> None:
    p = np.array([0.01, 0.02, 0.03, 0.9])
    q = benjamini_hochberg(p)
    m = len(p)
    expected = np.minimum.accumulate((np.sort(p) * m / np.arange(1, m + 1))[::-1])[::-1]
    assert np.allclose(np.sort(q), expected)


def test_q_values_never_fall_below_their_p_values() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(50)
    assert np.all(benjamini_hochberg(p) >= p - 1e-12)


def test_q_values_preserve_the_p_value_ordering() -> None:
    p = np.array([0.2, 0.001, 0.05, 0.4])
    q = benjamini_hochberg(p)
    assert list(np.argsort(q)) == list(np.argsort(p))


def test_a_single_test_needs_no_correction() -> None:
    assert benjamini_hochberg([0.031]) == pytest.approx([0.031])


def test_missing_p_values_are_excluded_from_the_correction() -> None:
    with_gap = benjamini_hochberg([0.01, np.nan, 0.5])
    without_gap = benjamini_hochberg([0.01, 0.5])

    assert np.isnan(with_gap[1])
    assert with_gap[[0, 2]] == pytest.approx(without_gap)


def test_benjamini_hochberg_handles_degenerate_input() -> None:
    assert np.all(np.isnan(benjamini_hochberg([np.nan, np.nan])))
    assert len(benjamini_hochberg([])) == 0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg([0.5, 1.5])


def test_bonferroni_and_expected_false_positives() -> None:
    assert bonferroni_threshold(10) == pytest.approx(0.005)
    assert expected_false_positives(20) == pytest.approx(1.0)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="n_tests"):
            bonferroni_threshold(bad)
    with pytest.raises(ValueError, match="alpha"):
        expected_false_positives(5, alpha=1.5)


# --------------------------------------------------------------------------- #
# Loading a universe
# --------------------------------------------------------------------------- #
def test_load_universe_reads_files_and_directories(tmp_path: Path, universe) -> None:
    folder = tmp_path / "prices"
    for symbol, frame in universe.items():
        save_csv(frame, folder / f"{symbol}.csv")

    from_dir = load_universe(data_paths=[folder])
    assert set(from_dir) == set(universe)
    pd.testing.assert_frame_equal(from_dir["AAA"], universe["AAA"])

    from_files = load_universe(data_paths=[folder / "AAA.csv", folder / "BBB.csv"])
    assert list(from_files) == ["AAA", "BBB"]


def test_load_universe_uses_the_injected_loader(universe) -> None:
    calls: list[str] = []

    def loader(ticker: str) -> pd.DataFrame:
        calls.append(ticker)
        return universe["AAA"]

    loaded = load_universe(tickers=["MU", "2408.TW"], loader=loader)
    assert calls == ["MU", "2408.TW"]
    assert set(loaded) == {"MU", "2408.TW"}


def test_downloads_are_cached_and_reused(tmp_path: Path, universe) -> None:
    calls: list[str] = []

    def loader(ticker: str) -> pd.DataFrame:
        calls.append(ticker)
        return universe["AAA"]

    cache = tmp_path / "cache"
    load_universe(tickers=["MU"], cache_dir=cache, loader=loader)
    assert (cache / "MU.csv").exists()

    load_universe(tickers=["MU"], cache_dir=cache, loader=loader)
    assert calls == ["MU"], "the second run should read the cache, not download again"


def test_load_universe_rejects_empty_and_duplicate_sources(tmp_path: Path, universe) -> None:
    with pytest.raises(ValueError, match="data_paths, tickers"):
        load_universe()

    save_csv(universe["AAA"], tmp_path / "MU.csv")
    with pytest.raises(ValueError, match="duplicate symbol"):
        load_universe(
            data_paths=[tmp_path / "MU.csv"],
            tickers=["MU"],
            loader=lambda _t: universe["AAA"],
        )


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
def test_screen_ranks_every_symbol_on_identical_folds(universe, screen_config) -> None:
    result = screen_universe(universe, "ridge", screen_config, permutations=30)

    assert result.n_tested == 3
    assert not result.failures
    scores = [entry.metrics()["excess_sharpe"] for entry in result.ranked]
    assert scores == sorted(scores, reverse=True)

    lengths = {len(entry.run.walk_forward.predictions) for entry in result.ranked}
    folds = {len(entry.run.walk_forward.folds) for entry in result.ranked}
    assert len(lengths) == 1 and len(folds) == 1


def test_screen_can_rank_by_another_metric(universe, screen_config) -> None:
    result = screen_universe(
        universe, "ridge", screen_config, rank_by="ic_fold_mean", permutations=0
    )
    values = [entry.metrics()["ic_fold_mean"] for entry in result.ranked]
    assert values == sorted(values, reverse=True)


def test_one_bad_symbol_does_not_abort_the_screen(universe, screen_config) -> None:
    """A short series must be reported, not raised - screens run unattended."""
    mixed = {**universe, "SHORT": generate_ohlcv(n_days=120, seed=5)}
    result = screen_universe(mixed, "ridge", screen_config, permutations=0)

    assert result.n_tested == 3
    assert [entry.symbol for entry in result.failures] == ["SHORT"]
    failure = result.failures[0]
    assert failure.run is None
    assert "bars" in (failure.error or "")
    assert failure.metrics() == {}


def test_q_values_are_computed_across_the_screen(universe, screen_config) -> None:
    result = screen_universe(universe, "ridge", screen_config, permutations=40)

    for entry in result.ranked:
        assert 0.0 <= entry.p_value <= 1.0
        assert entry.q_value >= entry.p_value - 1e-12
    assert result.bonferroni() == pytest.approx(0.05 / 3)
    assert result.expected_false_positives() == pytest.approx(0.15)


def test_skipping_permutations_leaves_significance_undefined(universe, screen_config) -> None:
    result = screen_universe(universe, "ridge", screen_config, permutations=0)

    assert all(entry.significance is None for entry in result.ranked)
    assert all(np.isnan(entry.q_value) for entry in result.ranked)
    assert result.survivors() == []


def test_survivors_need_both_an_edge_and_significance(universe, screen_config) -> None:
    """A symbol that beats its benchmark but fails the FDR test is not a survivor."""
    result = screen_universe(universe, "ridge", screen_config, permutations=40)

    for entry in result.survivors():
        assert entry.metrics()["excess_sharpe"] > 0
        assert entry.q_value <= result.alpha
    beat_benchmark = [e for e in result.ranked if e.metrics()["excess_sharpe"] > 0]
    assert len(result.survivors()) <= len(beat_benchmark)


def test_screen_table_has_one_row_per_evaluated_symbol(universe, screen_config) -> None:
    result = screen_universe(universe, "ridge", screen_config, permutations=20)
    table = result.table()

    assert list(table.index) == [entry.symbol for entry in result.ranked]
    for column in ("excess_sharpe", "p_value", "q_value", "ic_fold_t"):
        assert column in table.columns


def test_empty_universe_is_rejected(screen_config) -> None:
    with pytest.raises(ValueError, match="universe is empty"):
        screen_universe({}, "ridge", screen_config)


def test_screen_report_leads_with_the_correction(universe, screen_config) -> None:
    result = screen_universe(universe, "ridge", screen_config, permutations=30)
    rendered = render_screen_report(result, screen_config)

    assert "# Universe screen" in rendered
    assert "## The multiple-comparison correction" in rendered
    assert "Bonferroni threshold" in rendered
    assert "false positives expected from noise alone" in rendered
    for symbol in universe:
        assert symbol in rendered
    # Bar counts are counts, not measurements.
    assert "900.0" not in rendered


def test_screen_report_lists_failures(universe, screen_config) -> None:
    mixed = {**universe, "SHORT": generate_ohlcv(n_days=120, seed=5)}
    result = screen_universe(mixed, "ridge", screen_config, permutations=0)
    rendered = render_screen_report(result, screen_config)

    assert "## Symbols that could not be evaluated" in rendered
    assert "SHORT" in rendered


# --------------------------------------------------------------------------- #
# Holding the screened symbols together
# --------------------------------------------------------------------------- #
def test_screen_portfolio_holds_every_evaluated_symbol(universe, screen_config) -> None:
    result = screen_universe(universe, "ridge", screen_config, permutations=0)
    portfolio = result.portfolio()

    assert portfolio.symbols == [entry.symbol for entry in result.ranked]
    assert portfolio.weights.sum() == pytest.approx(1.0)

    metrics = portfolio.metrics()
    assert metrics["n_sleeves"] == 3.0
    assert metrics["effective_bets"] >= 1.0
    assert metrics["n_periods"] == len(portfolio.sleeves)
    assert np.isfinite(metrics["mean_correlation"])


def test_screen_portfolio_excludes_symbols_that_failed(universe, screen_config) -> None:
    mixed = {**universe, "SHORT": generate_ohlcv(n_days=120, seed=5)}
    result = screen_universe(mixed, "ridge", screen_config, permutations=0)
    assert "SHORT" not in result.portfolio().symbols


def test_screen_portfolio_needs_something_to_combine(screen_config) -> None:
    result = screen_universe(
        {"SHORT": generate_ohlcv(n_days=120, seed=5)}, "ridge", screen_config, permutations=0
    )
    with pytest.raises(ValueError, match="no evaluated symbols"):
        result.portfolio()


def test_screen_report_says_how_many_bets_the_universe_really_is(universe, screen_config) -> None:
    result = screen_universe(universe, "ridge", screen_config, permutations=0)
    rendered = render_screen_report(result, screen_config)

    assert "## Held together, not one at a time" in rendered
    assert "independent bets" in rendered
    assert "effective number of bets" in rendered
    assert "risk_contribution" in rendered
    assert "Sharpe if the sleeves were independent" in rendered
    # The sleeve correlation is never shown without the shares to compare it to.
    assert "The shares themselves correlate" in rendered
    assert "Correlation of the shares themselves" in rendered


def test_screen_report_survives_a_single_evaluated_symbol(universe, screen_config) -> None:
    one = {"AAA": universe["AAA"], "SHORT": generate_ohlcv(n_days=120, seed=5)}
    result = screen_universe(one, "ridge", screen_config, permutations=0)
    rendered = render_screen_report(result, screen_config)

    assert "## Held together, not one at a time" in rendered
    assert "nothing here to diversify" in rendered
