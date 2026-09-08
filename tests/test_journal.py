"""The forecast journal: recording, maturation and live-versus-backtest."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_stock.backtest.engine import signal_to_positions, simple_returns
from ai_stock.config import TRADING_DAYS_PER_YEAR, BacktestConfig, ExperimentConfig, FeatureConfig
from ai_stock.data.synthetic import generate_ohlcv
from ai_stock.journal import (
    JOURNAL_COLUMNS,
    Forecast,
    ScoreResult,
    append_forecasts,
    compare_with_backtest,
    load_journal,
    record_forecasts,
    rolling_compare_with_backtest,
    score_journal,
)


@pytest.fixture(scope="module")
def journal_config() -> ExperimentConfig:
    return ExperimentConfig(
        features=FeatureConfig(horizon=5),
        backtest=BacktestConfig(cost_bps=2.0, slippage_bps=3.0),
    )


@pytest.fixture(scope="module")
def prices() -> dict[str, pd.DataFrame]:
    return {"AAA": generate_ohlcv(n_days=700, seed=11), "BBB": generate_ohlcv(n_days=700, seed=22)}


@pytest.fixture
def daily_journal(tmp_path: Path, prices, journal_config) -> Path:
    """Ten days of the live loop, each day seeing exactly one more bar."""
    path = tmp_path / "forecasts.csv"
    for day in range(10):
        visible = {s: d.iloc[: len(d) - 10 + day] for s, d in prices.items()}
        append_forecasts(path, record_forecasts(visible, "ridge", journal_config))
    return path


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def test_a_forecast_cannot_see_past_its_own_bar(prices, journal_config) -> None:
    """Today's forecast must not change when tomorrow's bars are appended.

    This is the live-path version of the truncation-invariance test: what the
    model says on day D is a function of bars <= D and nothing else.
    """
    frame = prices["AAA"]
    cut = len(frame) - 20

    today = record_forecasts({"AAA": frame.iloc[:cut]}, "ridge", journal_config)[0]
    with_future = record_forecasts({"AAA": frame.iloc[: cut + 15]}, "ridge", journal_config)
    replayed = record_forecasts({"AAA": frame.iloc[:cut]}, "ridge", journal_config)[0]

    assert today.signal == pytest.approx(replayed.signal)
    assert today.asof_date == frame.index[cut - 1]
    # The later run forecasts a later bar - it must not be the same row.
    assert with_future[0].asof_date > today.asof_date


def test_forecast_is_anchored_to_the_latest_bar(prices, journal_config) -> None:
    frame = prices["AAA"]
    forecast = record_forecasts({"AAA": frame}, "ridge", journal_config)[0]

    assert forecast.asof_date == frame.index[-1]
    assert forecast.close == pytest.approx(float(frame["close"].iloc[-1]))
    assert forecast.horizon == journal_config.features.horizon
    assert forecast.position in (-1.0, 0.0, 1.0)
    assert np.isfinite(forecast.signal)


def test_a_symbol_too_short_to_fit_is_skipped_not_raised(prices, journal_config) -> None:
    """A series shorter than the indicator warm-up cannot produce a dataset."""
    universe = {"AAA": prices["AAA"], "TINY": generate_ohlcv(n_days=40, seed=1)}
    recorded = record_forecasts(universe, "ridge", journal_config)

    assert [f.symbol for f in recorded] == ["AAA"]
    assert set(universe) - {f.symbol for f in recorded} == {"TINY"}


def test_a_symbol_with_too_little_history_is_skipped(prices, journal_config) -> None:
    """Fittable is not the same as trustworthy: 6 training rows is not a forecast."""
    thin = generate_ohlcv(n_days=60, seed=1)
    assert record_forecasts({"THIN": thin}, "ridge", journal_config, min_train_rows=2)
    assert record_forecasts({"THIN": thin}, "ridge", journal_config) == []


def test_every_registered_model_can_be_recorded(prices, journal_config) -> None:
    for name in ("ridge", "random_forest", "logistic", "momentum"):
        recorded = record_forecasts({"AAA": prices["AAA"]}, name, journal_config)
        assert len(recorded) == 1 and recorded[0].model == name


def test_vol_target_is_honoured_not_silently_skipped(prices, journal_config) -> None:
    """Before this fix, setting `vol_target` here dropped every symbol silently.

    `signal_to_positions` raises when `vol_target` is set without
    `asset_returns`, and `record_forecasts` swallows that raise in its
    per-symbol ``except (ValueError, ...)`` - so a vol-targeted journal
    config looked identical to a healthy universe with nothing to record.
    """
    config = ExperimentConfig(
        features=journal_config.features,
        backtest=BacktestConfig(
            cost_bps=2.0, slippage_bps=3.0, vol_target=0.10, vol_lookback=20, max_leverage=3.0
        ),
    )

    forecasts = record_forecasts({"AAA": prices["AAA"]}, "momentum", config)

    assert len(forecasts) == 1
    forecast = forecasts[0]

    # The position must match sizing the signal by the symbol's own trailing
    # volatility over its full history, exactly as the backtest would.
    returns = simple_returns(prices["AAA"]["close"].astype(float))
    expected_signal = pd.Series(0.0, index=returns.index)
    expected_signal.loc[forecast.asof_date] = forecast.signal
    expected_position = signal_to_positions(
        expected_signal, config.backtest, asset_returns=returns
    ).loc[forecast.asof_date]

    assert forecast.position == pytest.approx(expected_position)
    assert 0.0 < abs(forecast.position) <= config.backtest.max_leverage


def test_vol_target_sizes_two_symbols_by_their_own_volatility(journal_config) -> None:
    """Per-symbol sizing, not one flat size for the whole universe.

    A quiet symbol and a symbol scaled up to be much noisier should not end
    up trading the same size once `vol_target` is honoured, even with an
    identical model and signal-generating process.
    """
    calm = generate_ohlcv(n_days=700, seed=11)
    # Scale log returns (not price levels) so the resulting series is exactly
    # 5x as volatile, then flatten OHLC to that close - the bar shape carries
    # no volatility information here, only the day-to-day return does.
    log_returns = np.log(calm["close"].astype(float)).diff().fillna(0.0)
    scaled_close = float(calm["close"].iloc[0]) * np.exp((log_returns * 5.0).cumsum())
    loud = calm.copy()
    for column in ("open", "high", "low", "close"):
        loud[column] = scaled_close

    config = ExperimentConfig(
        features=journal_config.features,
        backtest=BacktestConfig(
            cost_bps=2.0, slippage_bps=3.0, vol_target=0.10, vol_lookback=20, max_leverage=10.0
        ),
    )
    forecasts = record_forecasts({"CALM": calm, "LOUD": loud}, "momentum", config)
    by_symbol = {f.symbol: f for f in forecasts}

    assert set(by_symbol) == {"CALM", "LOUD"}
    assert abs(by_symbol["LOUD"].position) < abs(by_symbol["CALM"].position)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_journal_round_trips_through_csv(daily_journal: Path) -> None:
    frame = load_journal(daily_journal)

    assert list(frame.columns) == list(JOURNAL_COLUMNS)
    assert len(frame) == 20  # 10 days x 2 symbols
    assert frame["asof_date"].is_monotonic_increasing
    assert set(frame["symbol"]) == {"AAA", "BBB"}


def test_appending_the_same_day_twice_is_a_no_op(tmp_path: Path, prices, journal_config) -> None:
    path = tmp_path / "f.csv"
    forecasts = record_forecasts(prices, "ridge", journal_config)

    first = append_forecasts(path, forecasts)
    second = append_forecasts(path, forecasts)

    assert len(first) == len(second) == 2


def test_the_first_write_to_a_new_journal_warns_about_nothing(
    tmp_path: Path, prices, journal_config
) -> None:
    """Concatenating onto an empty frame is a FutureWarning on pandas 2.

    The first write to a fresh journal is exactly that case, so it broke only
    on the Python version whose resolver picked pandas 2 - the kind of failure
    that reaches CI rather than a local run.
    """
    import warnings

    forecasts = record_forecasts(prices, "ridge", journal_config)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        written = append_forecasts(tmp_path / "fresh.csv", forecasts)

    assert len(written) == len(forecasts)


def test_a_missing_journal_reads_as_empty(tmp_path: Path) -> None:
    frame = load_journal(tmp_path / "absent.csv")
    assert frame.empty
    assert list(frame.columns) == list(JOURNAL_COLUMNS)


def test_a_malformed_journal_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("asof_date,symbol\n2024-01-02,AAA\n")
    with pytest.raises(ValueError, match="missing journal column"):
        load_journal(path)


def test_appending_nothing_leaves_the_journal_alone(tmp_path: Path, prices, journal_config) -> None:
    path = tmp_path / "f.csv"
    append_forecasts(path, record_forecasts(prices, "ridge", journal_config))
    before = load_journal(path)
    after = append_forecasts(path, [])
    pd.testing.assert_frame_equal(before, after)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_only_elapsed_horizons_are_scored(daily_journal, prices, journal_config) -> None:
    """A row is scored exactly when the bar it forecast has actually arrived."""
    journal = load_journal(daily_journal)
    result = score_journal(journal, prices, journal_config)

    assert len(result.scored) + len(result.pending) == len(journal)
    assert result.scored["realised_return"].notna().all()
    assert "realised_return" not in result.pending.columns

    def has_arrived(row) -> bool:
        index = prices[row.symbol].index
        return index.get_loc(row.asof_date) + int(row.horizon) < len(index)

    assert all(has_arrived(row) for row in result.scored.itertuples(index=False))
    assert not any(has_arrived(row) for row in result.pending.itertuples(index=False))


def test_realised_return_matches_the_price_series(daily_journal, prices, journal_config) -> None:
    result = score_journal(load_journal(daily_journal), prices, journal_config)
    row = result.scored.iloc[0]

    closes = prices[row["symbol"]]["close"].astype(float)
    start = closes.index.get_loc(row["asof_date"])
    expected = math.log(closes.iloc[start + int(row["horizon"])] / closes.iloc[start])
    assert row["realised_return"] == pytest.approx(expected)


def test_costs_are_charged_on_the_change_in_position(tmp_path: Path, prices) -> None:
    config = ExperimentConfig(
        features=FeatureConfig(horizon=1), backtest=BacktestConfig(cost_bps=10.0, slippage_bps=0.0)
    )
    dates = prices["AAA"].index[-4:]
    flipping = [
        Forecast(d, "AAA", "manual", 1, s, p, 100.0)
        for d, s, p in zip(dates, [1.0, -1.0, 1.0, 1.0], [1.0, -1.0, 1.0, 1.0], strict=True)
    ]
    path = tmp_path / "f.csv"
    append_forecasts(path, flipping)

    result = score_journal(load_journal(path), prices, config)
    costs = pd.concat([result.scored["cost"], result.pending["cost"]]).round(8).tolist()
    # 0 -> +1 (1 unit), +1 -> -1 (2), -1 -> +1 (2), then no change (0).
    assert costs == [10e-4, 20e-4, 20e-4, 0.0]

    turnover = pd.concat([result.scored["turnover"], result.pending["turnover"]]).tolist()
    assert turnover == [1.0, 2.0, 2.0, 0.0]


def test_pnl_is_position_times_return_less_cost(daily_journal, prices, journal_config) -> None:
    result = score_journal(load_journal(daily_journal), prices, journal_config)
    frame = result.scored
    expected = frame["position"] * frame["realised_return"] - frame["cost"]
    assert np.allclose(frame["pnl"], expected)


def test_a_symbol_missing_from_the_universe_stays_pending(
    daily_journal, prices, journal_config
) -> None:
    result = score_journal(load_journal(daily_journal), {"AAA": prices["AAA"]}, journal_config)

    assert set(result.pending["symbol"]) >= {"BBB"}
    assert set(result.scored["symbol"]) == {"AAA"}


def test_scoring_an_empty_journal_is_safe(prices, journal_config) -> None:
    result = score_journal(load_journal(Path("does-not-exist.csv")), prices, journal_config)

    assert len(result) == 0
    assert result.metrics()["n_scored"] == 0
    assert result.by_symbol().empty


def test_metrics_and_per_symbol_breakdown(daily_journal, prices, journal_config) -> None:
    result = score_journal(load_journal(daily_journal), prices, journal_config)
    metrics = result.metrics()

    assert metrics["n_scored"] == len(result.scored)
    assert metrics["n_symbols"] == 2
    assert 0.0 <= metrics["hit_rate"] <= 1.0
    assert metrics["total_pnl"] == pytest.approx(result.scored["pnl"].sum())
    expected_turnover = result.scored["turnover"].mean() * TRADING_DAYS_PER_YEAR
    assert metrics["annual_turnover"] == pytest.approx(expected_turnover)

    per_symbol = result.by_symbol()
    assert set(per_symbol.index) == {"AAA", "BBB"}
    assert per_symbol["n"].sum() == (result.scored["position"] != 0).sum()


# --------------------------------------------------------------------------- #
# Live vs backtest
# --------------------------------------------------------------------------- #
def test_comparison_reports_the_gap_and_its_z_score(daily_journal, prices, journal_config) -> None:
    result = score_journal(load_journal(daily_journal), prices, journal_config)
    claimed = {"directional_accuracy": 0.52, "ic_fold_mean": 0.08}
    comparison = compare_with_backtest(result, claimed)

    live = result.metrics()["hit_rate"]
    n = result.metrics()["n_scored"]
    standard_error = math.sqrt(0.52 * 0.48 / n)

    assert comparison["hit_rate_gap"] == pytest.approx(live - 0.52)
    assert comparison["hit_rate_z"] == pytest.approx((live - 0.52) / standard_error)
    assert comparison["backtest_ic"] == pytest.approx(0.08)


def test_the_same_gap_grows_more_significant_with_more_forecasts() -> None:
    """A shortfall means little at n=10 and a lot at n=1000."""

    class _Fake:
        def __init__(self, n: float, hit: float) -> None:
            self._n, self._hit = n, hit

        def metrics(self) -> dict[str, float]:
            return {"n_scored": self._n, "hit_rate": self._hit, "live_ic": float("nan")}

    claimed = {"directional_accuracy": 0.55}
    small = compare_with_backtest(_Fake(10, 0.40), claimed)
    large = compare_with_backtest(_Fake(1000, 0.40), claimed)

    assert small["hit_rate_gap"] == pytest.approx(large["hit_rate_gap"])
    assert abs(large["hit_rate_z"]) > abs(small["hit_rate_z"]) * 5


def test_comparison_without_scored_forecasts_yields_no_z(prices, journal_config) -> None:
    empty = score_journal(load_journal(Path("nope.csv")), prices, journal_config)
    comparison = compare_with_backtest(empty, {"directional_accuracy": 0.52})

    assert comparison["n_scored"] == 0
    assert np.isnan(comparison["hit_rate_z"])


def _fake_scored(n: int, hits: list[bool]) -> pd.DataFrame:
    """A minimal scored frame: always long, right or wrong on cue."""
    position = np.ones(n)
    realised = np.where(hits, 1.0, -1.0) * 0.01
    return pd.DataFrame(
        {
            "asof_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "symbol": "AAA",
            "model": "manual",
            "horizon": 1,
            "signal": position,
            "position": position,
            "close": 100.0,
            "turnover": 0.0,
            "cost": 0.0,
            "realised_return": realised,
            "pnl": position * realised,
        }
    )


def test_rolling_comparison_is_empty_below_the_window() -> None:
    scored = _fake_scored(10, [True] * 10)
    result = ScoreResult(scored=scored, pending=scored.iloc[:0], cost_bps=5.0)

    rolling = rolling_compare_with_backtest(result, {"directional_accuracy": 0.5}, window=30)

    assert rolling.empty
    assert list(rolling.columns) == [
        "asof_date",
        "n_scored",
        "backtest_directional_accuracy",
        "live_hit_rate",
        "hit_rate_gap",
        "backtest_ic",
        "live_ic",
        "hit_rate_z",
    ]


def test_rolling_comparison_finds_when_a_pooled_z_hides_it() -> None:
    """30 clean hits then 30 clean misses average out to a so-so pooled z.

    The rolling window is the point of this feature: it should show the
    strong start and the collapse that the single pooled number cannot.
    """
    scored = pd.concat(
        [_fake_scored(30, [True] * 30), _fake_scored(30, [False] * 30)], ignore_index=True
    )
    scored["asof_date"] = pd.date_range("2024-01-01", periods=60, freq="D")
    result = ScoreResult(scored=scored, pending=scored.iloc[:0], cost_bps=5.0)
    claim = {"directional_accuracy": 0.55}

    rolling = rolling_compare_with_backtest(result, claim, window=30)
    pooled = compare_with_backtest(result, claim)

    assert len(rolling) == 60 - 30 + 1
    assert (rolling["n_scored"] == 30.0).all()
    assert rolling["asof_date"].iloc[0] == scored["asof_date"].iloc[29]
    assert rolling["asof_date"].iloc[-1] == scored["asof_date"].iloc[-1]
    assert rolling["live_hit_rate"].iloc[0] == pytest.approx(1.0)
    assert rolling["live_hit_rate"].iloc[-1] == pytest.approx(0.0)
    assert rolling["hit_rate_z"].iloc[0] > 2
    assert rolling["hit_rate_z"].iloc[-1] < -2
    # The pooled view averages the collapse away; the rolling view does not.
    assert pooled["hit_rate_z"] > rolling["hit_rate_z"].iloc[-1]
