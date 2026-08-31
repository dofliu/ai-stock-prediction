"""Loading, validating and persisting OHLCV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
"""Canonical column names used everywhere downstream."""

_COLUMN_ALIASES = {
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "adjusted close": "adj_close",
    "vol": "volume",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
}
_DATE_ALIASES = ("date", "datetime", "timestamp", "time", "index")


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace("-", " ")
        renamed[column] = _COLUMN_ALIASES.get(key, key.replace(" ", "_"))
    return frame.rename(columns=renamed)


def validate_ohlcv(frame: pd.DataFrame, *, name: str = "data") -> pd.DataFrame:
    """Return ``frame`` if it is a well-formed OHLCV table, else raise.

    Checks performed: required columns present, a sorted and unique
    ``DatetimeIndex``, strictly positive prices, non-negative volume and
    internally consistent bars (``low <= open, close <= high``).
    """
    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must be indexed by a DatetimeIndex, got {type(frame.index).__name__}")
    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique()[:3]
        raise ValueError(f"{name} has duplicate timestamps, e.g. {list(duplicates)}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted by time in ascending order")
    if frame.empty:
        raise ValueError(f"{name} is empty")

    prices = frame[["open", "high", "low", "close"]]
    if not prices.notna().all().all():
        raise ValueError(f"{name} contains missing prices")
    if (prices <= 0).any().any():
        raise ValueError(f"{name} contains non-positive prices")
    if (frame["volume"] < 0).any():
        raise ValueError(f"{name} contains negative volume")

    body_high = frame[["open", "close"]].max(axis=1)
    body_low = frame[["open", "close"]].min(axis=1)
    if (frame["high"] < body_high).any() or (frame["low"] > body_low).any():
        bad = frame.index[(frame["high"] < body_high) | (frame["low"] > body_low)][:3]
        raise ValueError(f"{name} has inconsistent bars (high/low outside open/close), e.g. {list(bad)}")
    return frame


def load_csv(path: str | Path, *, date_column: str | None = None) -> pd.DataFrame:
    """Read an OHLCV CSV file into a validated, time-indexed frame.

    The date column is auto-detected among ``date``/``datetime``/``timestamp``/
    ``time``/``index`` (case-insensitive) unless ``date_column`` is given.
    Extra columns such as ``adj_close`` are preserved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such data file: {path}")

    frame = _normalise_columns(pd.read_csv(path))

    if date_column is None:
        candidates = [c for c in frame.columns if c in _DATE_ALIASES]
        if not candidates:
            raise ValueError(
                f"{path} has no recognisable date column "
                f"(looked for {', '.join(_DATE_ALIASES)}); pass date_column explicitly"
            )
        date_column = candidates[0]
    elif date_column not in frame.columns:
        raise ValueError(f"{path} has no column named {date_column!r}")

    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.set_index(date_column).sort_index()
    frame.index.name = "date"
    return validate_ohlcv(frame, name=str(path))


def save_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write an OHLCV frame to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label=frame.index.name or "date")
    return path


def load_yfinance(
    ticker: str,
    *,
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLCV data via :mod:`yfinance` (an optional dependency).

    Kept deliberately thin: it normalises column names, validates the result
    and otherwise stays out of the way. Requires network access and
    ``pip install yfinance``.
    """
    try:
        import yfinance  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "load_yfinance requires the optional 'yfinance' package: pip install yfinance"
        ) from exc

    raw = yfinance.download(
        ticker,
        start=start,
        end=end,
        period=None if start or end else period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if raw is None or raw.empty:  # pragma: no cover - depends on network
        raise ValueError(f"yfinance returned no rows for {ticker!r}")

    if isinstance(raw.columns, pd.MultiIndex):  # single-ticker download still nests columns
        raw.columns = raw.columns.get_level_values(0)

    frame = _normalise_columns(raw)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    frame = frame.sort_index()
    keep = [c for c in (*OHLCV_COLUMNS, "adj_close") if c in frame.columns]
    return validate_ohlcv(frame[keep], name=f"yfinance:{ticker}")
