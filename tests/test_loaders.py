"""CSV loading, column normalisation and OHLCV validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_stock.data.loaders import (
    clamp_bar_extremes,
    drop_unclosed_session,
    drop_untraded_rows,
    load_csv,
    save_csv,
    validate_ohlcv,
)


def test_csv_roundtrip_preserves_the_frame(ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    path = save_csv(ohlcv, tmp_path / "nested" / "prices.csv")
    assert path.exists()

    loaded = load_csv(path)
    pd.testing.assert_frame_equal(loaded, ohlcv)


@pytest.mark.parametrize("date_column", ["Date", "timestamp", "DateTime"])
def test_date_column_is_auto_detected(tmp_path: Path, date_column: str) -> None:
    text = (
        f"{date_column},Open,High,Low,Close,Volume\n"
        "2024-01-02,10,11,9,10.5,1000\n"
        "2024-01-03,10.5,12,10,11.5,1200\n"
    )
    path = tmp_path / "prices.csv"
    path.write_text(text)

    frame = load_csv(path)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.name == "date"
    assert len(frame) == 2


def test_adjusted_close_alias_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        "date,open,high,low,close,Adj Close,volume\n"
        "2024-01-02,10,11,9,10.5,10.4,1000\n"
        "2024-01-03,10.5,12,10,11.5,11.4,1200\n"
    )
    frame = load_csv(path)
    assert "adj_close" in frame.columns


def test_rows_are_sorted_by_date(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        "date,open,high,low,close,volume\n"
        "2024-01-03,10.5,12,10,11.5,1200\n"
        "2024-01-02,10,11,9,10.5,1000\n"
    )
    frame = load_csv(path)
    assert frame.index.is_monotonic_increasing


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no such data file"):
        load_csv(tmp_path / "absent.csv")


def test_missing_date_column_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text("open,high,low,close,volume\n10,11,9,10.5,1000\n")
    with pytest.raises(ValueError, match="no recognisable date column"):
        load_csv(path)


def test_explicit_date_column_must_exist(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text("date,open,high,low,close,volume\n2024-01-02,10,11,9,10.5,1000\n")
    with pytest.raises(ValueError, match="no column named"):
        load_csv(path, date_column="when")


def test_validation_requires_all_price_columns(ohlcv: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing required column"):
        validate_ohlcv(ohlcv.drop(columns=["volume"]))


def test_validation_requires_a_datetime_index(ohlcv: pd.DataFrame) -> None:
    reset = ohlcv.reset_index(drop=True)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        validate_ohlcv(reset)


def test_validation_rejects_duplicate_timestamps(ohlcv: pd.DataFrame) -> None:
    duplicated = pd.concat([ohlcv.iloc[:5], ohlcv.iloc[:5]])
    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_ohlcv(duplicated)


def test_validation_rejects_unsorted_data(ohlcv: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="sorted by time"):
        validate_ohlcv(ohlcv.iloc[::-1])


def test_validation_rejects_empty_data(ohlcv: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_ohlcv(ohlcv.iloc[:0])


def test_validation_rejects_non_positive_prices(ohlcv: pd.DataFrame) -> None:
    broken = ohlcv.copy()
    broken.iloc[3, broken.columns.get_loc("low")] = 0.0
    with pytest.raises(ValueError, match="non-positive prices"):
        validate_ohlcv(broken)


def test_validation_rejects_missing_prices(ohlcv: pd.DataFrame) -> None:
    broken = ohlcv.copy()
    broken.iloc[3, broken.columns.get_loc("close")] = float("nan")
    with pytest.raises(ValueError, match="missing prices"):
        validate_ohlcv(broken)


def test_validation_rejects_negative_volume(ohlcv: pd.DataFrame) -> None:
    broken = ohlcv.copy()
    broken.iloc[3, broken.columns.get_loc("volume")] = -1
    with pytest.raises(ValueError, match="negative volume"):
        validate_ohlcv(broken)


def test_validation_rejects_inconsistent_bars(ohlcv: pd.DataFrame) -> None:
    broken = ohlcv.copy()
    broken.iloc[3, broken.columns.get_loc("high")] = broken["low"].iloc[3] / 2
    with pytest.raises(ValueError, match="inconsistent bars"):
        validate_ohlcv(broken)


# --------------------------------------------------------------------------- #
# Vendor padding
# --------------------------------------------------------------------------- #
def test_untraded_rows_are_dropped_not_accepted(ohlcv: pd.DataFrame) -> None:
    """Yahoo pads Taiwan listings with blank rows for holidays and halts.

    Those are absences, not data. They are cleaned in the loader so that
    validate_ohlcv can stay strict for everyone else.
    """
    padded = ohlcv.copy().astype(float)
    padded.iloc[[3, 17]] = np.nan

    with pytest.raises(ValueError, match="missing prices"):
        validate_ohlcv(padded, name="padded")

    with pytest.warns(UserWarning, match="dropped 2 row"):
        cleaned = drop_untraded_rows(padded, name="yfinance:2408.TW")

    assert len(cleaned) == len(ohlcv) - 2
    assert validate_ohlcv(cleaned, name="cleaned") is cleaned


def test_a_clean_frame_is_returned_untouched(ohlcv: pd.DataFrame) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert drop_untraded_rows(ohlcv, name="clean") is ohlcv


def test_a_mostly_blank_download_is_an_error_not_a_holiday(ohlcv: pd.DataFrame) -> None:
    broken = ohlcv.copy().astype(float)
    broken.iloc[: int(len(broken) * 0.8)] = np.nan

    with pytest.raises(ValueError, match="broken download, not holidays"):
        drop_untraded_rows(broken, name="yfinance:BROKEN")


def test_dropping_is_a_no_op_without_price_columns() -> None:
    frame = pd.DataFrame({"volume": [1.0, 2.0]})
    assert drop_untraded_rows(frame) is frame


def test_adjustment_rounding_is_repaired_not_rejected(ohlcv: pd.DataFrame) -> None:
    """Adjusted prices are rounded, and rounding breaks low <= body <= high.

    Taiwan listings adjust often enough that a long history reliably contains
    a few such bars. The repair is the one the synthetic generator applies to
    itself: the extremes must at least contain the body.
    """
    nudged = ohlcv.copy().astype(float)
    row = nudged.index[5]
    nudged.loc[row, "high"] = nudged.loc[row, ["open", "close"]].max() * 0.9999

    with pytest.raises(ValueError, match="inconsistent bars"):
        validate_ohlcv(nudged, name="raw")

    with pytest.warns(UserWarning, match="clamped 1 bar"):
        repaired = clamp_bar_extremes(nudged, name="yfinance:2408.TW")

    assert validate_ohlcv(repaired, name="repaired") is repaired
    # Only the extremes move; the body is data and is left alone.
    pd.testing.assert_frame_equal(repaired[["open", "close"]], nudged[["open", "close"]])


def test_a_large_inconsistency_is_corruption_and_raises(ohlcv: pd.DataFrame) -> None:
    broken = ohlcv.copy().astype(float)
    broken.loc[broken.index[5], "high"] = broken.loc[broken.index[5], "close"] * 0.8

    with pytest.raises(ValueError, match="beyond the .* attributable to adjustment rounding"):
        clamp_bar_extremes(broken, name="yfinance:BROKEN")


def test_consistent_bars_are_returned_untouched(ohlcv: pd.DataFrame) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert clamp_bar_extremes(ohlcv, name="clean") is ohlcv


def _bars(dates: list[str], volumes: list[int]) -> pd.DataFrame:
    """A minimal valid OHLCV frame; only the index and volume matter here."""
    return pd.DataFrame(
        {
            "open": [10.0] * len(dates),
            "high": [11.0] * len(dates),
            "low": [9.0] * len(dates),
            "close": [10.5] * len(dates),
            "volume": volumes,
        },
        index=pd.to_datetime(dates),
    )


class TestDropUnclosedSession:
    """The bar for a session still trading is not a bar yet."""

    def test_taiwan_bar_is_dropped_while_the_session_is_open(self) -> None:
        # The real failure: a 22:00 UTC cron that landed at 01:25 UTC, 25
        # minutes into the next Taipei session, and committed its first
        # 1.3M shares as if they were the day's 15M.
        frame = _bars(["2026-10-01", "2026-10-02"], [16_249_153, 1_346_000])

        with pytest.warns(UserWarning, match="still trading"):
            kept = drop_unclosed_session(
                frame, "2337.TW", now=pd.Timestamp("2026-10-02 01:25", tz="UTC")
            )

        assert list(kept.index.strftime("%Y-%m-%d")) == ["2026-10-01"]

    def test_the_same_bar_is_kept_once_taiwan_has_closed(self) -> None:
        frame = _bars(["2026-10-01", "2026-10-02"], [16_249_153, 15_900_000])

        # 05:30 UTC is 13:30 in Taipei: the close itself, not a minute after.
        kept = drop_unclosed_session(
            frame, "2337.TW", now=pd.Timestamp("2026-10-02 05:30", tz="UTC")
        )

        assert kept is frame

    def test_us_bar_follows_new_york_hours_and_its_daylight_saving(self) -> None:
        frame = _bars(["2026-06-30", "2026-07-01"], [31_103_200, 2_000_000])

        # 19:59 UTC is 15:59 in New York on a summer date - still trading.
        with pytest.warns(UserWarning, match="still trading"):
            kept = drop_unclosed_session(
                frame, "MU", now=pd.Timestamp("2026-07-01 19:59", tz="UTC")
            )
        assert list(kept.index.strftime("%Y-%m-%d")) == ["2026-06-30"]

        # A minute later the bell has rung.
        assert (
            drop_unclosed_session(frame, "MU", now=pd.Timestamp("2026-07-01 20:00", tz="UTC"))
            is frame
        )

    def test_a_winter_bar_needs_the_extra_hour_est_costs(self) -> None:
        frame = _bars(["2026-01-05", "2026-01-06"], [31_103_200, 2_000_000])

        # 20:00 UTC is 15:00 in New York in January: the summer rule would
        # have let this one through.
        with pytest.warns(UserWarning, match="still trading"):
            kept = drop_unclosed_session(
                frame, "MU", now=pd.Timestamp("2026-01-06 20:00", tz="UTC")
            )
        assert len(kept) == 1

        assert (
            drop_unclosed_session(frame, "MU", now=pd.Timestamp("2026-01-06 21:00", tz="UTC"))
            is frame
        )

    def test_an_unknown_suffix_warns_and_keeps_the_data(self) -> None:
        frame = _bars(["2026-10-01", "2026-10-02"], [100, 100])

        with pytest.warns(UserWarning, match="EXCHANGE_SESSIONS"):
            kept = drop_unclosed_session(
                frame, "0700.HK", now=pd.Timestamp("2026-10-02 01:25", tz="UTC")
            )

        assert kept is frame

    def test_a_naive_clock_is_read_as_utc(self) -> None:
        frame = _bars(["2026-10-01", "2026-10-02"], [100, 100])

        with pytest.warns(UserWarning, match="still trading"):
            kept = drop_unclosed_session(frame, "2337.TW", now=pd.Timestamp("2026-10-02 01:25"))

        assert len(kept) == 1

    def test_an_empty_frame_is_returned_unchanged(self) -> None:
        frame = _bars([], [])
        assert drop_unclosed_session(frame, "2337.TW") is frame

    def test_every_bar_in_the_future_is_a_clock_problem_and_raises(self) -> None:
        frame = _bars(["2026-10-02"], [100])

        with pytest.raises(ValueError, match="clock or a timezone problem"):
            drop_unclosed_session(frame, "2337.TW", now=pd.Timestamp("2026-10-01 00:00", tz="UTC"))

    def test_only_trailing_bars_are_dropped_not_a_hole_in_the_middle(self) -> None:
        frame = _bars(["2026-09-30", "2026-10-01", "2026-10-02"], [100, 100, 100])

        with pytest.warns(UserWarning, match="still trading"):
            kept = drop_unclosed_session(
                frame, "2337.TW", now=pd.Timestamp("2026-10-02 01:25", tz="UTC")
            )

        assert list(kept.index.strftime("%Y-%m-%d")) == ["2026-09-30", "2026-10-01"]
