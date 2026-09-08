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
from ai_stock.backtest.engine import signal_to_positions, simple_returns
from ai_stock.config import TRADING_DAYS_PER_YEAR, ExperimentConfig
from ai_stock.data.loaders import validate_ohlcv
from ai_stock.features.builder import build_dataset, build_features
from ai_stock.models.registry import create_model

__all__ = [
    "Forecast",
    "ScoreResult",
    "append_forecasts",
    "compare_with_backtest",
    "load_journal",
    "record_forecasts",
    "rolling_compare_with_backtest",
    "score_journal",
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

    def __len__(self) -> int:
        return len(self.scored)

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
            ),
            float("nan"),
        )
        empty["n_scored"] = 0.0
        empty["n_pending"] = float(len(self.pending))
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
        sharpe = (
            float(np.mean(pnl) / np.std(pnl, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
            if len(frame) >= 2 and np.std(pnl, ddof=1) > 1e-12
            else float("nan")
        )
        span = pd.to_datetime(frame["asof_date"])
        # Mirrors the backtest's `annual_turnover`: mean traded notional per
        # forecast, annualised by the number of trading days in a year. Each
        # symbol is recorded at most once per trading day, so this is on the
        # same footing as the backtest's mean-per-bar figure.
        annual_turnover = float(frame["turnover"].mean() * TRADING_DAYS_PER_YEAR)
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
    """Read the journal, or return an empty frame with the right columns."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(JOURNAL_COLUMNS))
    frame = pd.read_csv(path)
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
            signal = float(np.asarray(model.predict(latest), dtype=float).ravel()[0])
            position = _size_position(signal, latest.index[-1], ohlcv, config.backtest)

            # Sized over the symbol's own trailing volatility, honouring
            # `BacktestConfig.vol_target` exactly as the backtest does - a
            # single flat size for every symbol makes the live P&L
            # incomparable to the backtested one whenever volatilities
            # differ. The rolling estimate needs history *before* the asof
            # date, so the signal is placed on the full return series rather
            # than a lone point; only the asof row is kept.
            asof = latest.index[-1]
            asset_returns = simple_returns(ohlcv["close"].astype(float))
            sized_signal = pd.Series(0.0, index=asset_returns.index)
            sized_signal.loc[asof] = signal
            position = float(
                signal_to_positions(sized_signal, config.backtest, asset_returns=asset_returns).loc[
                    asof
                ]
            )
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
    """
    config = config or ExperimentConfig()
    cost_rate = config.backtest.total_cost_bps * _BPS

    if journal.empty:
        empty = pd.DataFrame(
            columns=[*JOURNAL_COLUMNS, "turnover", "cost", "realised_return", "pnl"]
        )
        return ScoreResult(
            scored=empty,
            pending=empty.drop(columns=["realised_return", "pnl"]),
            cost_bps=config.backtest.total_cost_bps,
        )

    frame = journal.copy()
    frame["asof_date"] = pd.to_datetime(frame["asof_date"])
    frame = frame.sort_values(["symbol", "model", "asof_date"]).reset_index(drop=True)

    # Cost is charged on the change from the position previously held in that
    # symbol, which is what the journal's own history says it was.
    previous = frame.groupby(["symbol", "model"])["position"].shift(1).fillna(0.0)
    frame["turnover"] = (frame["position"] - previous).abs()
    frame["cost"] = frame["turnover"] * cost_rate

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
    return ScoreResult(scored=scored, pending=pending, cost_bps=config.backtest.total_cost_bps)


def compare_with_backtest(
    live: ScoreResult, backtest_metrics: dict[str, float]
) -> dict[str, float]:
    """Measure the live-versus-backtest gap in units of its own sampling error.

    ``hit_rate_z`` is the shortfall in live directional accuracy divided by the
    standard error of a proportion at the live sample size. Values around zero
    mean the live record is consistent with the backtest; a large negative
    value is decay, or an overfitted backtest finally showing itself.

    With few scored forecasts the standard error is large and the z-score is
    close to zero *whatever* happens: read ``n_scored`` before the z.
    """
    metrics = live.metrics()
    n = metrics["n_scored"]
    claimed = backtest_metrics.get("directional_accuracy", float("nan"))
    observed = metrics["hit_rate"]

    comparison = {
        "n_scored": n,
        "backtest_directional_accuracy": float(claimed),
        "live_hit_rate": float(observed),
        "hit_rate_gap": float(observed - claimed),
        "backtest_ic": float(backtest_metrics.get("ic_fold_mean", float("nan"))),
        "live_ic": metrics["live_ic"],
    }

    if n >= 1 and np.isfinite(claimed) and np.isfinite(observed) and 0.0 < claimed < 1.0:
        standard_error = math.sqrt(claimed * (1.0 - claimed) / n)
        comparison["hit_rate_z"] = float((observed - claimed) / standard_error)
    else:
        comparison["hit_rate_z"] = float("nan")
    return comparison


ROLLING_COMPARISON_COLUMNS = (
    "asof_date",
    "n_scored",
    "backtest_directional_accuracy",
    "live_hit_rate",
    "hit_rate_gap",
    "backtest_ic",
    "live_ic",
    "hit_rate_z",
)


def rolling_compare_with_backtest(
    live: ScoreResult, backtest_metrics: dict[str, float], window: int = 30
) -> pd.DataFrame:
    """Slide :func:`compare_with_backtest` over a trailing window of forecasts.

    A single ``hit_rate_z`` over the whole journal answers only whether the
    live record has drifted from the backtest, not when: a bad early stretch
    and a good later one can average out and read as zero. This recomputes the
    same comparison over the most recent ``window`` matured forecasts, ending
    at each date in turn, so the point where the gap opened is visible rather
    than only its current size.

    The window counts *matured forecasts*, not calendar days, since a
    multi-symbol universe records several per day. Returns one row per window
    end-date with the columns of :func:`compare_with_backtest` plus
    ``asof_date``; empty (but correctly columned) once fewer than ``window``
    forecasts have matured.
    """
    frame = live.scored.sort_values("asof_date").reset_index(drop=True)
    if len(frame) < window:
        return pd.DataFrame(columns=list(ROLLING_COMPARISON_COLUMNS))

    empty_pending = frame.iloc[:0].drop(columns=["realised_return", "pnl"])
    rows = []
    for end in range(window, len(frame) + 1):
        chunk = frame.iloc[end - window : end]
        window_result = ScoreResult(scored=chunk, pending=empty_pending, cost_bps=live.cost_bps)
        comparison = compare_with_backtest(window_result, backtest_metrics)
        comparison["asof_date"] = chunk["asof_date"].iloc[-1]
        rows.append(comparison)
    return pd.DataFrame(rows, columns=list(ROLLING_COMPARISON_COLUMNS))
