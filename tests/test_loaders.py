"""CSV loading, column normalisation and OHLCV validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_stock.data.loaders import drop_untraded_rows, load_csv, save_csv, validate_ohlcv


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
