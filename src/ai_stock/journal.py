"""A forecast journal: record predictions now, score them when the future arrives.

A backtest scores a model against history it was fitted near. This module does
the harder thing: it writes down what the model predicts *today*, with no
knowledge of the outcome, and scores that row only once the horizon has
actually elapsed. Nothing here can be tuned after the fact, because the
prediction is already on disk before the answer exists.

That makes it the only honest check on the number a backtest reports. A
strategy whose live hit rate sits far below its backtested one has decayed, or
was overfitted to begin with; :func:`compare_with_backtest` states the gap in
units of its own sampling error so the difference can be told from noise.

The journal is a plain append-only CSV, so it diffs cleanly in git and can be
inspected without this package.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ai_stock.backtest.engine import signal_to_positions, simple_returns, trailing_volatility
from ai_stock.config import TRADING_DAYS_PER_YEAR, BacktestConfig, ExperimentConfig
from ai_stock.data.loaders import validate_ohlcv
from ai_stock.evaluation.metrics import sharpe_ratio
from ai_stock.features.builder import build_dataset, build_features
from ai_stock.models.registry import create_model

__all__ = [
    "Forecast",
    "ScoreResult",
    "append_forecasts",
    "compare_with_backtest",
    "data_freshness",
    "independent_blocks",
    "load_journal",
    "record_forecasts",
    "rolling_compare_with_backtest",
    "score_journal",
    "stale_symbols",
]

JOURNAL_COLUMNS = (
    "asof_date",
    "symbol",
    "model",
    "horizon",
    "signal",
    "position",
    "close",
)
"""Columns written when a forecast is recorded; scoring adds more."""

_BPS = 1e-4


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, or ``NaN`` when it is undefined."""
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _decided(scored: pd.DataFrame) -> pd.DataFrame:
    """The rows that actually took a side.

    A zero position is not a directional call, so it is not a trial the hit
    rate can be right or wrong about. ``metrics()`` already excludes these
    from ``hit_rate``; anything measuring that rate's precision has to
    exclude them too, or it counts trials that were never made.
    """
    if scored.empty or "position" not in scored.columns:
        return scored
    return scored[scored["position"].to_numpy(float) != 0.0]


def independent_blocks(scored: pd.DataFrame, calendar: pd.Index | None = None) -> int:
    """How many non-overlapping horizon windows a set of forecasts covers.

    A journal records every symbol every trading day, but a forecast with a
    5-day horizon shares four of its five outcome days with the forecast made
    the day before. Ten such forecasts are not ten independent bets on the
    market; they are two, observed five times each. On top of that, forecasts
    made on the same day across a correlated universe move together - the four
    memory names in ``config/universe.txt`` are one cycle seen four ways.

    This counts the conservative end of that range: forecasts are grouped into
    consecutive blocks of ``horizon`` trading days, and every forecast in a
    block - across all symbols - is treated as one observation. The count is
    a property of the recording schedule alone, not of the returns, so there
    is no correlation estimate here to be wrong about or to tune.

    ``calendar`` is the trading sessions the forecasts were recorded against -
    :func:`score_journal` supplies the union of the universe's price indices,
    and blocks are cut on it starting from the first date in ``scored``. A
    universe spanning two calendars (Taipei and New York) contributes the
    union of its trading days, which can only make the block count larger and
    the resulting standard error smaller, so it is the generous direction.

    **Without a ``calendar`` the journal's own dates stand in for it**, which
    assumes the journal holds a row for every session. That is what a journal
    is meant to be and not what one with holes in it is: a day the job never
    ran, or an outage, is then not a gap at all but two adjacent dates, and
    two forecasts sharing no outcome day whatever are counted as one
    observation. The error is always in the same direction - fewer
    independent windows than were really placed, so a larger standard error
    and a ``hit_rate_z`` pulled toward zero - which is the expensive
    direction, because it is the decay warning that goes quiet.

    >>> frame = pd.DataFrame(
    ...     {
    ...         "asof_date": pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"] * 2),
    ...         "horizon": 5,
    ...     }
    ... )
    >>> independent_blocks(frame)
    1

    A gap the journal cannot see, next to the sessions that fill it:

    >>> gapped = pd.DataFrame(
    ...     {"asof_date": pd.to_datetime(["2024-01-01", "2024-01-11"]), "horizon": 5}
    ... )
    >>> independent_blocks(gapped)
    1
    >>> independent_blocks(gapped, pd.bdate_range("2024-01-01", "2024-01-11"))
    2
    """
    if scored.empty:
        return 0
    horizon = max(int(pd.to_numeric(scored["horizon"]).max()), 1)
    dates = pd.to_datetime(scored["asof_date"])
    sessions = pd.DatetimeIndex(np.sort(dates.unique()))
    if calendar is not None:
        # A journal date missing from the calendar was still a session - a bar
        # was recorded on it - so it is added rather than dropped, and blocks
        # are cut from the first forecast so the origin does not depend on how
        # much price history happens to sit in front of the journal.
        sessions = pd.DatetimeIndex(calendar).union(sessions)
        sessions = sessions[sessions >= dates.min()]
    position = sessions.get_indexer(pd.Index(dates))
    return int(np.unique(position // horizon).size)


@dataclass(frozen=True)
class Forecast:
    """One prediction, made with no knowledge of its outcome."""

    asof_date: pd.Timestamp
    """Close date of the last bar the model was allowed to see.

    Named ``asof_date`` rather than ``asof`` because ``DataFrame.asof`` is a
    pandas method: a column called ``asof`` is unreachable as an attribute.
    """
    symbol: str
    model: str
    horizon: int
    signal: float
    position: float
    close: float

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["asof_date"] = pd.Timestamp(self.asof_date).date().isoformat()
        return row


@dataclass(frozen=True)
class ScoreResult:
    """Forecasts whose horizon has elapsed, with their realised outcomes."""

    scored: pd.DataFrame
    """One row per matured forecast, with ``realised_return`` and ``pnl``."""
    pending: pd.DataFrame
    """Forecasts still waiting for the future to arrive."""
    cost_bps: float
    calendar: pd.Index | None = None
    """The trading sessions these forecasts were recorded against.

    The union of the universe's price indices, as :func:`score_journal` saw
    them. It is what :func:`independent_blocks` cuts its windows on, so that a
    day the daily job missed stays a gap instead of becoming two adjacent
    journal rows. ``None`` when the result was not built by
    :func:`score_journal` - the block count then falls back to the journal's
    own dates, with the consequence :func:`independent_blocks` describes.
    """

    def __len__(self) -> int:
        return len(self.scored)

    def _annual_turnover(self, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
        """Average annualised position turnover, mirroring the backtest's ``annual_turnover``.

        Uses every recorded forecast, matured or not: a position that flips
        every day pays for it the moment it flips, not once its horizon
        elapses, so this should not wait on scoring the way `hit_rate` and
        `live_ic` do. Averaged per symbol first, same convention as `live_ic`.

        Rate per *session held*, not per journal row - see
        :func:`_annualised_turnover`.
        """
        columns = ["asof_date", "symbol", "turnover", "sessions_held"]
        # `ScoreResult` is public and can be built from frames that never went
        # through `score_journal`, so the columns are not guaranteed. Without
        # them there is nothing to quote a rate over - see
        # :func:`_annualised_turnover`.
        if any(c not in self.scored.columns or c not in self.pending.columns for c in columns):
            return float("nan")
        scored, pending = self.scored[columns], self.pending[columns]
        # Never hand an empty frame to concat: on pandas 2 that raises a
        # FutureWarning about all-NA columns (see `append_forecasts`), and a
        # journal with nothing scored yet - the common case right after it
        # starts - is exactly that.
        if scored.empty and pending.empty:
            return float("nan")
        if scored.empty:
            combined = pending
        elif pending.empty:
            combined = scored
        else:
            combined = pd.concat([scored, pending], ignore_index=True)
        turnovers = []
        for _, group in combined.groupby("symbol"):
            # One forecast is one trade, not a rate: there is no stretch of
            # holding to divide it over, so a lone row says nothing about how
            # often this symbol trades and is left out rather than annualised.
            if len(group) < 2:
                continue
            rate = _annualised_turnover(group, periods_per_year=periods_per_year)
            if np.isfinite(rate):
                turnovers.append(rate)
        return float(np.mean(turnovers)) if turnovers else float("nan")

    def metrics(self) -> dict[str, float]:
        """Live performance across every matured forecast."""
        empty = dict.fromkeys(
            (
                "n_scored",
                "n_pending",
                "hit_rate",
                "live_ic",
                "live_ic_pooled",
                "mean_pnl",
                "total_pnl",
                "live_sharpe",
                "annual_turnover",
                "n_symbols",
                "span_days",
                "live_annual_turnover",
            ),
            float("nan"),
        )
        empty["n_scored"] = 0.0
        empty["n_pending"] = float(len(self.pending))
        empty["live_annual_turnover"] = self._annual_turnover()
        if self.scored.empty:
            return empty

        frame = self.scored
        realised = frame["realised_return"].to_numpy(float)
        signal = frame["signal"].to_numpy(float)
        pnl = frame["pnl"].to_numpy(float)

        decided = frame["position"].to_numpy(float) != 0.0
        hit_rate = (
            float(np.mean(np.sign(frame.loc[decided, "position"]) == np.sign(realised[decided])))
            if decided.any()
            else float("nan")
        )
        # Pooling the IC across symbols has the same defect as pooling it
        # across walk-forward folds: symbols sit at different volatilities, so
        # the between-symbol variation can dominate and even flip the sign.
        # `live_ic` therefore averages per-symbol ICs, mirroring `ic_fold_mean`;
        # the pooled figure is kept alongside it for reference only.
        pooled_ic = _correlation(signal, realised)
        per_symbol_ic = [
            _correlation(group["signal"].to_numpy(float), group["realised_return"].to_numpy(float))
            for _, group in frame.groupby("symbol")
        ]
        finite_ic = [value for value in per_symbol_ic if np.isfinite(value)]
        ic = float(np.mean(finite_ic)) if finite_ic else float("nan")
        # Per-forecast returns overlap when horizon > 1, so this Sharpe is a
        # rough health check, not a tradable statistic.
        sharpe = _live_sharpe(frame)
        span = pd.to_datetime(frame["asof_date"])
        # Mirrors the backtest's `annual_turnover`: traded notional per trading
        # session, annualised by the number of sessions in a year. The backtest
        # averages over a dense daily index, where one row *is* one session;
        # the journal's rows are whatever the daily job managed to record, so
        # the sessions are counted from the bars rather than from the rows.
        annual_turnover = _annualised_turnover(frame)
        return {
            "n_scored": float(len(frame)),
            "n_pending": float(len(self.pending)),
            "hit_rate": hit_rate,
            "live_ic": ic,
            "live_ic_pooled": pooled_ic,
            "mean_pnl": float(np.mean(pnl)),
            "total_pnl": float(np.sum(pnl)),
            "live_sharpe": sharpe,
            "annual_turnover": annual_turnover,
            "n_symbols": float(frame["symbol"].nunique()),
            "span_days": float((span.max() - span.min()).days),
            "live_annual_turnover": empty["live_annual_turnover"],
        }

    def by_symbol(self) -> pd.DataFrame:
        """Live hit rate and P&L per symbol."""
        if self.scored.empty:
            return pd.DataFrame()
        frame = self.scored.copy()
        frame["hit"] = np.sign(frame["position"]) == np.sign(frame["realised_return"])
        grouped = frame[frame["position"] != 0].groupby("symbol")
        summary = grouped.agg(
            n=("pnl", "size"),
            hit_rate=("hit", "mean"),
            mean_pnl=("pnl", "mean"),
            total_pnl=("pnl", "sum"),
        )
        return summary.sort_values("total_pnl", ascending=False)


def load_journal(path: str | Path) -> pd.DataFrame:
    """Read the journal, or return an empty frame with the right columns.

    Read with ``float_precision="round_trip"``. pandas' default CSV float
    parser is fast rather than correctly rounded, and lands up to an ulp away
    from the value the text denotes. That is far too small to move any
    statistic here - but :func:`append_forecasts` rewrites the whole file each
    day, so the slightly-wrong float is what gets written back, and a row
    recorded before its outcome existed silently stops being the row that was
    recorded. The journal's entire claim is that it is append-only; a parser
    that perturbs old rows on every read quietly makes that false.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(JOURNAL_COLUMNS))
    frame = pd.read_csv(path, float_precision="round_trip")
    missing = [column for column in JOURNAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing journal column(s): {', '.join(missing)}")
    frame["asof_date"] = pd.to_datetime(frame["asof_date"])
    return frame


def append_forecasts(path: str | Path, forecasts: list[Forecast]) -> pd.DataFrame:
    """Append forecasts to the journal, ignoring ones already recorded.

    Re-running on a day whose bars have not moved is a no-op rather than a
    duplicate row, so the daily job is safe to retry.
    """
    path = Path(path)
    existing = load_journal(path)
    incoming = pd.DataFrame([f.as_row() for f in forecasts], columns=list(JOURNAL_COLUMNS))
    if incoming.empty:
        return existing
    incoming["asof_date"] = pd.to_datetime(incoming["asof_date"])

    key = ["asof_date", "symbol", "model", "horizon"]
    if not existing.empty:
        already = existing.set_index(key).index
        incoming = incoming[~incoming.set_index(key).index.isin(already)]

    # Never hand an empty frame to concat: on pandas 2 that raises a
    # FutureWarning about all-NA columns, and the first write to a new journal
    # is exactly that case.
    if incoming.empty:
        combined = existing
    elif existing.empty:
        combined = incoming
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.sort_values(["asof_date", "symbol", "model"]).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    output = combined.copy()
    output["asof_date"] = pd.to_datetime(output["asof_date"]).dt.date.astype(str)
    output.to_csv(path, index=False)
    return combined


MIN_TRAIN_ROWS = 250
"""Labelled bars a symbol needs before its forecast is worth recording."""

STALE_AFTER_DAYS = 4
"""Calendar days after which :func:`stale_symbols` is asked to report a feed.

Only the default for the ``journal --fail-if-stale`` bare flag, which counts
calendar days because it backs an exit code and has to clear a market holiday
without crying wolf - see :func:`stale_symbols`. The *report's* verdict is not
this: it counts missed weekday sessions, for the reason
:data:`MISSED_SESSIONS_ALLOWED` gives.
"""

MISSED_SESSIONS_ALLOWED = 1
"""Weekday sessions that may be missing before a feed is reported as behind.

One is ordinary. The download runs shortly after a close and a provider
publishes a day's bar with its own lag, so a symbol routinely arrives one
session behind its peers - on 2026-09-25 `MU` did exactly that while the three
Taiwan symbols were current. Two is not that: two closes have come and gone
with nothing written, which is a feed that has stopped rather than one that is
slow.
"""


def _last_bar_date(frame: pd.DataFrame | None) -> pd.Timestamp | None:
    """Date of the most recent bar, or ``None`` if the frame cannot supply one."""
    if frame is None or len(frame) == 0:
        return None
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        try:
            index = pd.to_datetime(index)
        except (TypeError, ValueError):
            return None
    index = index.dropna()
    if len(index) == 0:
        return None
    return pd.Timestamp(index.max()).normalize()


def _missed_sessions(last_bar: pd.Timestamp, reference: pd.Timestamp) -> float:
    """Weekdays strictly between ``last_bar`` and ``reference``.

    Both ends are excluded on purpose. ``last_bar`` has arrived, and
    ``reference`` is today, whose own session has not closed yet at the hour
    this runs - counting either would make a healthy feed look behind every
    single day.

    Mon-Fri, with no holiday calendar: both markets in the current universe
    trade weekdays, and a closure therefore reads as a missed session. That is
    the same direction :func:`data_freshness` already errs in.
    """
    start = last_bar + pd.Timedelta(days=1)
    end = reference - pd.Timedelta(days=1)
    if end < start:
        return 0.0
    return float(len(pd.bdate_range(start, end)))


def data_freshness(
    universe: dict[str, pd.DataFrame],
    *,
    asof: pd.Timestamp | None = None,
    allow_missed_sessions: int = MISSED_SESSIONS_ALLOWED,
) -> pd.DataFrame:
    """How far behind each symbol's most recent bar is, relative to ``asof``.

    Every other number in this module describes a model. This one describes the
    data underneath it, and the two fail in opposite directions. When a price
    feed stops, the journal does not go quiet - it goes stale *confidently*.
    :func:`score_journal` keeps maturing the same forecasts against the same
    bars, :func:`record_forecasts` adds nothing because the last bar is already
    journalled, and the report reads exactly as it did the day the feed was
    healthy. A hit rate that has not moved for a week looks identical whether
    the strategy is quiet or the downloader is dead.

    Two measures, and only one of them decides. ``age_days`` is calendar days,
    reported because it is what a reader wants to see. ``missed_sessions`` is
    the number of weekdays that have closed since the last bar, and it is what
    ``stale`` reads, because a calendar-day age means a different thing on
    every weekday: three days old is a healthy Monday reading a Friday bar and
    a symptom on a Wednesday. A single threshold over a quantity whose healthy
    value moves with the day of week has to be loose enough for the loosest day
    and is therefore blind on the rest - it will call a Thursday bar read on a
    Sunday "current" while Friday's session is missing from every symbol, and
    let a feed that stopped on a Tuesday run three more trading days before it
    says anything. ``missed_sessions`` has the same healthy value, zero, every
    day of the week.

    No holiday calendar is consulted, so a market closure reads as a missed
    session. That is the deliberate direction to err in: a spurious "check the
    feed" costs a glance, and a silently stale hit rate costs the only number
    in this project that cannot be tuned after the fact. Reading the difference
    requires knowing each market's calendar, which this frame does not claim
    to - it reports the gap and names it, rather than deciding.

    Symbols whose frame is empty or carries no usable dates are reported with a
    missing (``NaN``) ``last_bar`` and infinite counts, because an unreadable
    feed is not a fresh one.

    >>> import pandas as pd
    >>> bars = pd.DataFrame(
    ...     {"close": [1.0, 2.0]},
    ...     index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
    ... )
    >>> frame = data_freshness({"X": bars}, asof=pd.Timestamp("2026-01-13"))
    >>> frame.loc["X", "last_bar"]
    '2026-01-06'
    >>> float(frame.loc["X", "age_days"])
    7.0
    >>> float(frame.loc["X", "missed_sessions"])  # 7th to 12th, weekdays only
    4.0
    >>> bool(frame.loc["X", "stale"])
    True

    A Friday bar read on the following Monday is not a feed that stopped, and
    the weekend does not count against it:

    >>> friday = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2026-01-09"]))
    >>> monday = data_freshness({"X": friday}, asof=pd.Timestamp("2026-01-12"))
    >>> float(monday.loc["X", "age_days"]), float(monday.loc["X", "missed_sessions"])
    (3.0, 0.0)
    """
    columns = ["symbol", "last_bar", "age_days", "missed_sessions", "stale"]
    if not universe:
        return pd.DataFrame(columns=columns).set_index("symbol")

    reference = (pd.Timestamp(asof) if asof is not None else pd.Timestamp.today()).normalize()

    rows = []
    for symbol in sorted(universe):
        last_bar = _last_bar_date(universe[symbol])
        if last_bar is None:
            rows.append(
                {
                    "symbol": symbol,
                    "last_bar": None,
                    "age_days": float("inf"),
                    "missed_sessions": float("inf"),
                    "stale": True,
                }
            )
            continue
        missed = _missed_sessions(last_bar, reference)
        rows.append(
            {
                "symbol": symbol,
                "last_bar": last_bar.date().isoformat(),
                "age_days": float((reference - last_bar).days),
                "missed_sessions": missed,
                "stale": missed > float(allow_missed_sessions),
            }
        )
    frame = pd.DataFrame(rows, columns=columns).set_index("symbol")
    return frame


def stale_symbols(freshness: pd.DataFrame, *, older_than_days: float) -> list[str]:
    """Symbols in ``freshness`` whose most recent bar is older than ``older_than_days``.

    Deliberately re-derives the verdict from ``age_days`` rather than reading the
    ``stale`` column, so the caller can ask a *different* question from the one
    :func:`data_freshness` answered. The two have different costs of being
    wrong. The ``stale`` column is read by a person glancing at a report, where
    a needless "check the feed" costs a glance, so it fires early - after one
    missed weekday session, which on an ordinary week is the day after next.
    This function backs an exit code, which fires a build failure that someone
    has to triage, and an alarm that cries wolf every Lunar New Year - when the
    Taiwan market is legitimately shut for up to nine calendar days - is an
    alarm that gets muted. It counts calendar days for that reason: a holiday
    is not a missed session to anyone triaging a red build, and a threshold in
    calendar days is the one a reader can check against a holiday calendar
    directly. Callers wiring up an alarm should pass a threshold wide enough to
    clear the longest holiday in their universe's calendars.

    Symbols whose feed could not be read carry an infinite ``age_days`` and so
    are reported at every threshold, which is the intended reading: an
    unreadable feed is not a fresh one.

    >>> import pandas as pd
    >>> bars = pd.DataFrame(
    ...     {"close": [1.0, 2.0]},
    ...     index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
    ... )
    >>> frame = data_freshness({"X": bars}, asof=pd.Timestamp("2026-01-13"))
    >>> bool(frame.loc["X", "stale"])  # the report says behind, at four sessions
    True
    >>> stale_symbols(frame, older_than_days=4)
    ['X']
    >>> stale_symbols(frame, older_than_days=10)  # a holiday this long is normal
    []
    """
    if freshness.empty:
        return []
    ages = pd.to_numeric(freshness["age_days"], errors="coerce")
    behind = freshness.index[ages.isna() | (ages > float(older_than_days))]
    return sorted(str(symbol) for symbol in behind)


def _size_position(
    signal: float, asof: pd.Timestamp, ohlcv: pd.DataFrame, config: BacktestConfig
) -> float:
    """Size one forecast exactly as the backtest would size its last bar.

    ``signal_to_positions`` needs a rolling window of asset returns to apply
    ``vol_target``, which the single-row series a live forecast produces
    cannot supply. This instead computes trailing volatility from the
    symbol's full price history up to ``asof`` - causal by construction,
    since every input predates the forecast - and applies the same scaling
    :func:`signal_to_positions` uses on its final bar. Without this, a
    non-default ``vol_target`` made every recorded position silently 0: the
    single-row call raised ``ValueError`` for lack of ``asset_returns``, and
    ``record_forecasts`` treats that as "skip this symbol".
    """
    unscaled = config if config.vol_target is None else replace(config, vol_target=None)
    base = float(signal_to_positions(pd.Series([signal], index=[asof]), unscaled).iloc[0])
    if config.vol_target is None:
        return base

    asset_returns = simple_returns(ohlcv["close"].astype(float))
    realised = trailing_volatility(asset_returns, config.vol_lookback)
    latest_vol = realised.get(asof, float("nan"))
    if not (latest_vol > 0):  # warm-up, or a flat/degenerate price series
        return 0.0
    scaler = config.vol_target / latest_vol
    return float(np.clip(base * scaler, -config.max_leverage, config.max_leverage))


def record_forecasts(
    universe: dict[str, pd.DataFrame],
    model_name: str = "random_forest",
    config: ExperimentConfig | None = None,
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> list[Forecast]:
    """Predict the next ``horizon`` bars for every symbol, from its latest bar.

    The model is fitted on every labelled bar available - which necessarily
    ends ``horizon`` bars before the last one, since later bars have no target
    yet - and then asked about the most recent close. No part of the answer
    exists in the data at the time of the call.

    A symbol is skipped - rather than raising - when it cannot be fitted or has
    fewer than ``min_train_rows`` labelled bars, so one newly added ticker does
    not stop the rest of the universe being recorded. Skipping matters as much
    as recording: a forecast fitted on a handful of rows is worthless, and once
    written to the journal it is indistinguishable from a well-founded one.
    Callers can recover the skipped names as
    ``set(universe) - {f.symbol for f in forecasts}``.
    """
    config = config or ExperimentConfig()
    forecasts: list[Forecast] = []

    for symbol, ohlcv in universe.items():
        try:
            validate_ohlcv(ohlcv, name=symbol)
            dataset = build_dataset(ohlcv, config.features)
            model = create_model(model_name)
            target = dataset.target(classification=model.is_classifier)
            usable = target.notna().to_numpy()
            if usable.sum() < max(2, min_train_rows):
                continue
            model.fit(dataset.features.loc[usable], target[usable])

            # The final warmed-up feature row is the one the model has never
            # been trained on: it is the bar we are forecasting from.
            features = build_features(ohlcv, config.features)
            live = features.dropna()
            if live.empty:
                continue
            latest = live.iloc[[-1]]
            asof = latest.index[-1]
            signal = float(np.asarray(model.predict(latest), dtype=float).ravel()[0])
            position = _size_position(signal, asof, ohlcv, config.backtest)

            forecasts.append(
                Forecast(
                    asof_date=asof,
                    symbol=symbol,
                    model=model_name,
                    horizon=config.features.horizon,
                    signal=signal,
                    position=position,
                    close=float(ohlcv.loc[asof, "close"]),
                )
            )
        except (ValueError, KeyError, RuntimeError):
            continue

    return forecasts


def _realised_return(prices: pd.Series, asof: pd.Timestamp, horizon: int) -> float | None:
    """Log return over ``horizon`` bars after ``asof``, or ``None`` if unknown."""
    index = prices.index
    if asof not in index:
        return None
    start = int(index.get_loc(asof))
    end = start + horizon
    if end >= len(index):
        return None
    return float(np.log(prices.iloc[end] / prices.iloc[start]))


def _sessions_held(bars: pd.Index, asof: pd.Series) -> pd.Series:
    """Trading sessions each journal row's position was held over.

    The position taken on ``asof[i]`` stands until the next entry for that
    symbol, so the sessions it covers are the bars in ``(asof[i-1], asof[i]]``.
    The first entry opens the book on its own bar and covers one session, which
    is what the backtest charges its opening trade over.

    Counted from ``bars`` - the symbol's own price index - so a market holiday
    is not a missing session, and a symbol that did not trade on a day its
    neighbours did is not penalised for it.

    >>> bars = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"])
    >>> _sessions_held(bars, pd.Series(pd.to_datetime(["2026-09-01", "2026-09-03"]))).tolist()
    [1.0, 2.0]
    """
    if asof.empty:
        return pd.Series(dtype=float, index=asof.index)
    # `searchsorted(side="right")` counts the bars up to and including a date,
    # so the difference between two of those counts is the bars strictly after
    # the earlier one and up to the later.
    seen = bars.searchsorted(asof.to_numpy(), side="right").astype(float)
    held = np.diff(seen, prepend=np.nan)
    held[0] = 1.0
    return pd.Series(held, index=asof.index)


def _annualised_turnover(
    frame: pd.DataFrame, *, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Traded notional per session held, annualised.

    Divides by the sessions the positions actually covered rather than by the
    number of journal rows. The two agree only when the journal has a row for
    every bar; a journal with holes in it - the daily job missing a run, a
    symbol whose market was shut - has fewer rows than sessions, and a
    per-row mean would then report the rate of a position that traded on every
    one of them.
    """
    if "sessions_held" not in frame.columns or "turnover" not in frame.columns:
        return float("nan")
    usable = frame[np.isfinite(frame["sessions_held"])]
    sessions = float(usable["sessions_held"].sum())
    if not usable.empty and sessions > 0:
        return float(usable["turnover"].sum() / sessions * periods_per_year)
    return float("nan")


def _live_sharpe(scored: pd.DataFrame, *, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe of the journal's per-forecast P&L.

    One row of ``pnl`` is what a position made over ``horizon`` sessions, not
    over one. Annualising it at ``periods_per_year`` - the session count the
    backtest uses, because a backtest's rows *are* sessions - scales the ratio
    by ``sqrt(252)`` when the journal's own period is five days and only
    ``sqrt(252 / 5)`` of them fit in a year. The error is a flat factor of
    ``sqrt(horizon)``, it is always upward, and it lands on the one number in
    this report that reads as risk-adjusted return: at the five-day horizon
    this project runs, a Sharpe of 0.32 prints as 0.71.

    So the rate is quoted per *holding period*: ``periods_per_year / horizon``,
    read from the journal's own ``horizon`` column rather than from a config,
    because the column is what the rows were actually recorded at.

    A journal mixing horizons has no single period length and therefore no
    single annualisation, so it returns ``nan`` rather than pick one - the same
    bargain :func:`_annualised_turnover` takes when it cannot see the columns
    it needs. ``ScoreResult`` is public and can be built from a frame that
    never went through :func:`score_journal`, so a missing column is a real
    case and not a bug.

    Still a health check and not a tradable number, for a reason the scaling
    does not fix: forecasts recorded daily against a multi-day horizon overlap,
    so these rows are not independent draws and the ratio's standard error is
    understated whatever it is divided by. This makes the point estimate mean
    what it says; it does not make it precise.

    >>> frame = pd.DataFrame({"pnl": [0.02, -0.01, 0.03, -0.02], "horizon": [5] * 4})
    >>> round(_live_sharpe(frame), 4)
    1.4912
    >>> round(_live_sharpe(frame.assign(horizon=1)), 4)  # the same P&L read daily
    3.3343
    >>> frame["horizon"] = [5, 5, 10, 10]  # no single period to annualise over
    >>> _live_sharpe(frame)
    nan
    """
    if "pnl" not in scored.columns or "horizon" not in scored.columns:
        return float("nan")
    horizons = pd.to_numeric(scored["horizon"], errors="coerce").dropna().unique()
    if len(horizons) != 1 or horizons[0] <= 0:
        return float("nan")
    return sharpe_ratio(
        scored["pnl"].to_numpy(float), periods_per_year=periods_per_year / float(horizons[0])
    )


def _trading_calendar(universe: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Every session any symbol in the universe traded, in order.

    The union rather than the intersection, for the reason
    :func:`independent_blocks` gives: a universe spanning Taipei and New York
    has days only one of them traded, and counting them is the generous
    direction for the block count that divides the headline standard error.
    """
    sessions = pd.DatetimeIndex([])
    for ohlcv in universe.values():
        sessions = sessions.union(pd.DatetimeIndex(ohlcv.index))
    return sessions


def score_journal(
    journal: pd.DataFrame,
    universe: dict[str, pd.DataFrame],
    config: ExperimentConfig | None = None,
) -> ScoreResult:
    """Split the journal into matured forecasts and ones still in flight.

    A forecast matures only when the bar ``horizon`` positions after its
    ``asof_date`` exists in the price series. Costs are charged on the change in
    position from that symbol's previous journal entry, so a signal that flips
    every day pays for it here exactly as it would in the backtest.

    The result carries the universe's trading calendar, because two of the
    numbers read off it - ``sessions_held`` and the block count behind
    ``hit_rate_z`` - are rates over sessions and the journal's rows are not
    sessions wherever a day's run went missing.
    """
    config = config or ExperimentConfig()
    cost_rate = config.backtest.total_cost_bps * _BPS
    calendar = _trading_calendar(universe)

    if journal.empty:
        empty = pd.DataFrame(
            columns=[
                *JOURNAL_COLUMNS,
                "turnover",
                "cost",
                "sessions_held",
                "realised_return",
                "pnl",
            ]
        )
        return ScoreResult(
            scored=empty,
            pending=empty.drop(columns=["realised_return", "pnl"]),
            cost_bps=config.backtest.total_cost_bps,
            calendar=calendar,
        )

    frame = journal.copy()
    frame["asof_date"] = pd.to_datetime(frame["asof_date"])
    frame = frame.sort_values(["symbol", "model", "asof_date"]).reset_index(drop=True)

    # Cost is charged on the change from the position previously held in that
    # symbol, which is what the journal's own history says it was.
    previous = frame.groupby(["symbol", "model"])["position"].shift(1).fillna(0.0)
    frame["turnover"] = (frame["position"] - previous).abs()
    frame["cost"] = frame["turnover"] * cost_rate
    # How long each of those positions stood, in that symbol's own sessions.
    # `turnover` is what was traded; this is what it was traded over, and
    # without it a rate can only be quoted per journal row.
    frame["sessions_held"] = float("nan")
    for (symbol, _model), group in frame.groupby(["symbol", "model"], sort=False):
        ohlcv = universe.get(symbol)
        if ohlcv is None:
            continue
        frame.loc[group.index, "sessions_held"] = _sessions_held(ohlcv.index, group["asof_date"])

    realised: list[float | None] = []
    for row in frame.itertuples(index=False):
        ohlcv = universe.get(row.symbol)
        if ohlcv is None:
            realised.append(None)
            continue
        realised.append(
            _realised_return(ohlcv["close"].astype(float), row.asof_date, int(row.horizon))
        )
    frame["realised_return"] = realised

    matured = frame["realised_return"].notna()
    frame["pnl"] = frame["position"] * frame["realised_return"] - frame["cost"]

    scored = frame[matured].reset_index(drop=True)
    pending = frame[~matured].drop(columns=["realised_return", "pnl"]).reset_index(drop=True)
    return ScoreResult(
        scored=scored,
        pending=pending,
        cost_bps=config.backtest.total_cost_bps,
        calendar=calendar,
    )


def compare_with_backtest(
    live: ScoreResult, backtest_metrics: dict[str, float]
) -> dict[str, float]:
    """Measure the live-versus-backtest gap in units of its own sampling error.

    Both z-scores divide the same shortfall in directional accuracy by the
    standard error of a proportion; they disagree only on how many independent
    trials the journal has actually seen, and that disagreement is the point.

    - ``hit_rate_z_naive`` counts every matured forecast that took a side.
      That is the number the journal would deserve if each forecast were an
      independent bet, which it is not: overlapping horizons and a correlated
      universe both mean the same market move is counted several times.
    - ``hit_rate_z`` counts :func:`independent_blocks` instead - non-overlapping
      windows of ``horizon`` trading days, with everything inside a window
      treated as one observation. Cut on ``live.calendar``, the sessions the
      market actually held, so a day the journal is missing stays a gap rather
      than closing up and merging two windows that share no outcome day.

    The first assumes zero redundancy and the second assumes total redundancy
    within a window, so the honest significance sits between them and the
    journal cannot yet say where. ``hit_rate_z`` is the headline because
    overstating a decay warning is as damaging here as missing one: a z below
    -2 is meant to mean something.

    With few scored forecasts both are close to zero *whatever* happens: read
    ``n_independent`` before either z.
    """
    metrics = live.metrics()
    decided = _decided(live.scored)
    n_decided = float(len(decided))
    n_independent = float(independent_blocks(decided, live.calendar))
    claimed = backtest_metrics.get("directional_accuracy", float("nan"))
    observed = metrics["hit_rate"]

    comparison = {
        "n_scored": metrics["n_scored"],
        "n_decided": n_decided,
        "n_independent": n_independent,
        "backtest_directional_accuracy": float(claimed),
        "live_hit_rate": float(observed),
        "hit_rate_gap": float(observed - claimed),
        "backtest_ic": float(backtest_metrics.get("ic_fold_mean", float("nan"))),
        "live_ic": metrics["live_ic"],
    }

    comparable = np.isfinite(claimed) and np.isfinite(observed) and 0.0 < claimed < 1.0
    spread = claimed * (1.0 - claimed) if comparable else float("nan")
    for key, trials in (("hit_rate_z_naive", n_decided), ("hit_rate_z", n_independent)):
        if comparable and trials >= 1:
            comparison[key] = float((observed - claimed) / math.sqrt(spread / trials))
        else:
            comparison[key] = float("nan")
    return comparison


ROLLING_COMPARISON_COLUMNS = (
    "asof_date",
    "n_scored",
    "n_decided",
    "n_independent",
    "backtest_directional_accuracy",
    "live_hit_rate",
    "hit_rate_gap",
    "backtest_ic",
    "live_ic",
    "hit_rate_z",
    "hit_rate_z_naive",
)


def rolling_compare_with_backtest(
    live: ScoreResult, backtest_metrics: dict[str, float], window: int = 30
) -> pd.DataFrame:
    """Slide :func:`compare_with_backtest` over a trailing window of forecasts.

    A single ``hit_rate_z`` over the whole journal answers only whether the
    live record has drifted from the backtest, not when: a bad early stretch
    and a good later one can average out and read as zero. This recomputes the
    same comparison over a trailing window of matured forecasts, ending at each
    date in turn, so the point where the gap opened is visible rather than only
    its current size.

    **The window moves one date at a time and keeps whole dates**, so both ends
    hold a complete cross-section of whatever the universe was that day. The
    journal records every symbol under one ``asof_date`` and carries no
    ordering within it - the rows come out in universe order, not in time
    order - so a window ending part-way through a date would keep an arbitrary
    subset of that day's symbols. That is not a small effect at this size: with
    per-symbol live hit rates spanning 0.50 to 0.86, which symbols a cut
    happened to keep moved the window's hit rate by several points for reasons
    that had nothing to do with when anything happened, and printed one row per
    *forecast* under a column headed ``asof_date``, so a four-symbol day became
    four points on a date axis.

    ``window`` is therefore a floor on matured forecasts, not an exact count:
    each window is the shortest run of trailing dates holding at least
    ``window`` of them, and ``n_scored`` reports what each one actually held.
    Returns one row per qualifying date with the columns of
    :func:`compare_with_backtest` plus ``asof_date``; empty (but correctly
    columned) while fewer than ``window`` forecasts have matured.

    Note that 30 forecasts over a four-symbol universe span only about eight
    trading days, which at a 5-day horizon is two independent blocks. The
    per-window ``hit_rate_z`` is correspondingly wide and jumpy; it is a
    picture of *when* the gap moved, not a per-date significance test.
    """
    frame = live.scored.sort_values("asof_date").reset_index(drop=True)
    if len(frame) < window:
        return pd.DataFrame(columns=list(ROLLING_COMPARISON_COLUMNS))

    # `frame` is sorted by date, so a run of whole dates is a positional slice
    # and the cumulative counts give its bounds without searching.
    per_date = frame["asof_date"].value_counts().sort_index()
    dates = per_date.index
    cumulative = per_date.to_numpy().cumsum()

    empty_pending = frame.iloc[:0].drop(columns=["realised_return", "pnl"])
    rows = []
    start = 0
    for end in range(len(dates)):
        # Drop dates off the front while the rest would still clear the floor.
        while start < end and cumulative[end] - cumulative[start] >= window:
            start += 1
        low = int(cumulative[start - 1]) if start else 0
        high = int(cumulative[end])
        if high - low < window:
            continue
        chunk = frame.iloc[low:high]
        window_result = ScoreResult(
            scored=chunk,
            pending=empty_pending,
            cost_bps=live.cost_bps,
            calendar=live.calendar,
        )
        comparison = compare_with_backtest(window_result, backtest_metrics)
        comparison["asof_date"] = dates[end]
        rows.append(comparison)
    return pd.DataFrame(rows, columns=list(ROLLING_COMPARISON_COLUMNS))
