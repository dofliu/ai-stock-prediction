"""Loading, validating and persisting OHLCV data."""

from __future__ import annotations

import warnings
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

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


MAX_UNTRADED_FRACTION = 0.5
"""Above this share of blank rows, a download is broken rather than padded."""

MAX_BAR_REPAIR = 0.005
"""Largest high/low inconsistency, as a fraction of price, treated as rounding."""


def drop_untraded_rows(frame: pd.DataFrame, *, name: str = "data") -> pd.DataFrame:
    """Remove calendar rows the venue did not trade on.

    Yahoo pads some markets - Taiwan listings notably - with rows carrying
    ``NaN`` prices for exchange holidays and trading suspensions. Those are
    absences, not data, and :func:`validate_ohlcv` rightly refuses them, so
    vendor padding is cleaned here rather than by loosening the contract every
    other part of the package relies on.

    A file that is mostly blank is a different problem, and raises.
    """
    price_columns = [c for c in ("open", "high", "low", "close") if c in frame.columns]
    if not price_columns:
        return frame

    cleaned = frame.dropna(subset=price_columns)
    dropped = len(frame) - len(cleaned)
    if not dropped:
        return frame

    if len(frame) and dropped / len(frame) > MAX_UNTRADED_FRACTION:
        raise ValueError(
            f"{name}: {dropped} of {len(frame)} rows have no prices "
            f"({dropped / len(frame):.0%}) - that is a broken download, not holidays"
        )
    warnings.warn(
        f"{name}: dropped {dropped} row(s) with no prices (exchange holidays or halts)",
        UserWarning,
        stacklevel=3,
    )
    return cleaned


def clamp_bar_extremes(
    frame: pd.DataFrame, *, tolerance: float = MAX_BAR_REPAIR, name: str = "data"
) -> pd.DataFrame:
    """Repair bars whose high/low sit just inside their own open/close.

    Split-and-dividend adjusted prices are the vendor's raw prices multiplied
    by a factor and rounded, and the rounding does not always preserve
    ``low <= open, close <= high``. Taiwan listings adjust often enough that a
    fifteen-year history reliably contains a few such bars.

    A discrepancy of a fraction of a percent is that rounding, and the honest
    repair is the one the synthetic generator already applies to itself: the
    extremes must at least contain the body. A larger discrepancy is not
    rounding, and raises.
    """
    columns = ("open", "high", "low", "close")
    if any(column not in frame.columns for column in columns):
        return frame

    body_high = frame[["open", "close"]].max(axis=1)
    body_low = frame[["open", "close"]].min(axis=1)
    over = (body_high - frame["high"]).clip(lower=0.0)
    under = (frame["low"] - body_low).clip(lower=0.0)

    broken = (over > 0) | (under > 0)
    if not broken.any():
        return frame

    scale = frame["close"].abs().replace(0.0, float("nan"))
    worst = float(((over + under) / scale).max())
    if worst > tolerance:
        raise ValueError(
            f"{name}: high/low inconsistent with open/close by up to {worst:.2%} of price "
            f"- beyond the {tolerance:.1%} attributable to adjustment rounding"
        )

    repaired = frame.copy()
    repaired["high"] = frame[["open", "high", "close"]].max(axis=1)
    repaired["low"] = frame[["open", "low", "close"]].min(axis=1)
    warnings.warn(
        f"{name}: clamped {int(broken.sum())} bar(s) whose high/low sat inside their "
        f"open/close by up to {worst:.3%} of price (adjustment rounding)",
        UserWarning,
        stacklevel=3,
    )
    return repaired


EXCHANGE_SESSIONS: dict[str, tuple[str, time]] = {
    "": ("America/New_York", time(16, 0)),
    "TW": ("Asia/Taipei", time(13, 30)),
    "TWO": ("Asia/Taipei", time(13, 30)),
}
"""Regular close, by the Yahoo ticker suffix (``""`` is the unsuffixed US tape).

Only the markets this project's universe actually lists. Adding a market means
adding its hours here; an unknown suffix warns rather than guesses, because the
guess that matters is the one in :func:`drop_unclosed_session`.
"""


def drop_unclosed_session(
    frame: pd.DataFrame, ticker: str, *, now: pd.Timestamp | None = None, name: str = "data"
) -> pd.DataFrame:
    """Remove trailing bars whose trading session has not finished yet.

    Asked for a daily history while a market is open, Yahoo answers with a bar
    for the session in progress: the day's open, the high and low so far, the
    *last trade* in the close column and a few minutes of volume. It is shaped
    exactly like a finished bar and nothing in :func:`validate_ohlcv` can tell
    them apart - the prices are positive and the body sits inside the extremes,
    because it is a real bar, just not a whole one.

    That matters here more than it would in a notebook. The daily job records a
    forecast against the newest bar and writes its close into the forecast
    journal, which is append-only by design: the next day's download silently
    replaces the partial bar with the finished one, but the journal keeps the
    mid-session price for ever and scores against it. On 2026-09-29 that cost
    ``2337.TW`` a recorded entry of 116.50 against a true close of 118.50 -
    1.7% of the price, on a five-day horizon whose whole edge is a tenth of
    that.

    The job is supposed to run after every close in the universe, and the
    schedule is written to. What cannot be relied on is *being* run then:
    GitHub's scheduler drifts under load, and a 22:00 UTC cron that lands at
    01:25 UTC is a quarter of an hour into the next Taiwan session. So the
    guarantee is taken here, where it can be checked, rather than assumed from
    a cron line.

    A bar is kept once the regular close of its own exchange has passed in real
    time. Early closes are not modelled, which only ever delays a bar that was
    already complete - the safe direction - and an unknown suffix warns and
    keeps the data rather than inventing a calendar for it.

    >>> frame = pd.DataFrame(
    ...     {"open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
    ...      "close": [1.0, 1.0], "volume": [10, 1]},
    ...     index=pd.to_datetime(["2026-10-01", "2026-10-02"]),
    ... )
    >>> kept = drop_unclosed_session(
    ...     frame, "2337.TW", now=pd.Timestamp("2026-10-02 01:25", tz="UTC")
    ... )
    >>> list(kept.index.strftime("%Y-%m-%d"))
    ['2026-10-01']
    """
    if frame.empty:
        return frame

    suffix = ticker.rsplit(".", 1)[1].upper() if "." in ticker else ""
    session = EXCHANGE_SESSIONS.get(suffix)
    if session is None:
        warnings.warn(
            f"{name}: no session hours known for the {suffix!r} suffix of {ticker!r}, so a "
            f"bar for a market still trading cannot be recognised - add it to "
            f"EXCHANGE_SESSIONS",
            UserWarning,
            stacklevel=2,
        )
        return frame

    zone, close_time = session
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")

    closes = pd.DatetimeIndex(
        [
            pd.Timestamp.combine(day.date(), close_time)
            .tz_localize(ZoneInfo(zone))
            .tz_convert("UTC")
            for day in frame.index
        ]
    )
    kept = frame[closes <= now]

    dropped = len(frame) - len(kept)
    if not dropped:
        return frame
    if kept.empty:
        raise ValueError(
            f"{name}: every one of {len(frame)} bar(s) is dated at or after the next "
            f"{zone} close - that is a clock or a timezone problem, not a live session"
        )
    warnings.warn(
        f"{name}: dropped {dropped} bar(s) for a session still trading, "
        f"latest {frame.index[-1].date()} (closes {close_time} {zone})",
        UserWarning,
        stacklevel=2,
    )
    return kept


def next_session_close(ticker: str, bar_date: pd.Timestamp | str) -> pd.Timestamp | None:
    """When the session *after* ``bar_date`` closes on ``ticker``'s own exchange, in UTC.

    The companion to :func:`drop_unclosed_session`. That one asks whether a bar
    is finished; this one asks whether the *next* bar is, which is the question
    a forecast journal has to answer before it writes a row down. A forecast
    made from the ``bar_date`` close is scored over the sessions that follow
    it, so once this instant has passed the first day of that outcome already
    exists - and a row written then is no longer a prediction.

    Weekdays only, and no holiday calendar, for the reason
    :data:`EXCHANGE_SESSIONS` gives: a closure makes this return a moment that
    has already passed when the market was in fact shut, which costs a
    forecast. The error in the other direction costs the journal's one
    guarantee, so this is the side to be wrong on.

    ``None`` means the suffix is not in :data:`EXCHANGE_SESSIONS` and the
    question cannot be answered; the caller decides what to do with that.
    :func:`drop_unclosed_session` already warns about such a ticker on the
    download path, so this stays quiet rather than warning twice per run.

    >>> next_session_close("2337.TW", "2026-10-01")  # Taipei closes 13:30
    Timestamp('2026-10-02 05:30:00+0000', tz='UTC')
    >>> next_session_close("MU", "2026-10-02")  # Friday bar -> Monday's close
    Timestamp('2026-10-05 20:00:00+0000', tz='UTC')
    >>> next_session_close("X.XX", "2026-10-02") is None
    True
    """
    suffix = ticker.rsplit(".", 1)[1].upper() if "." in ticker else ""
    session = EXCHANGE_SESSIONS.get(suffix)
    if session is None:
        return None

    zone, close_time = session
    day = pd.Timestamp(bar_date).normalize() + pd.offsets.BDay(1)
    return (
        pd.Timestamp.combine(day.date(), close_time).tz_localize(ZoneInfo(zone)).tz_convert("UTC")
    )


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
        raise TypeError(
            f"{name} must be indexed by a DatetimeIndex, got {type(frame.index).__name__}"
        )
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
        raise ValueError(
            f"{name} has inconsistent bars (high/low outside open/close), e.g. {list(bad)}"
        )
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
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Download OHLCV data via :mod:`yfinance` (an optional dependency).

    Kept deliberately thin: it normalises column names, drops any bar for a
    session still trading (see :func:`drop_unclosed_session`), validates the
    result and otherwise stays out of the way. ``now`` overrides the clock that
    check reads, and exists so it can be tested. Requires network access and
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
    frame = drop_unclosed_session(frame[keep], ticker, now=now, name=f"yfinance:{ticker}")
    frame = drop_untraded_rows(frame, name=f"yfinance:{ticker}")
    frame = clamp_bar_extremes(frame, name=f"yfinance:{ticker}")
    return validate_ohlcv(frame, name=f"yfinance:{ticker}")
