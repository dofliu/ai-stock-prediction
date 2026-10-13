"""A daily market-watch report: what happened, measured against each stock's own past.

The rest of this package asks whether a price can be forecast. This module asks
nothing of the future. It describes the latest session - returns over several
windows, where the price sits in its 52-week range, how unusual today's move
and volume were - and states each figure against the same stock's own history,
because "down 4%" means one thing for a utility and another for a memory maker.

It also reads an industry-notes log kept by hand (``data/notes/industry.csv``).
A note is dated when it is written, so the prices that follow it are outcomes
the note could not have seen - the same discipline as the forecast journal.
Months of notes beside their follow-up returns are the raw material for any
method built later; nothing here claims a note predicts anything.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ai_stock.config import TRADING_DAYS_PER_YEAR
from ai_stock.reporting.report import Report, markdown_table

__all__ = [
    "NOTE_COLUMNS",
    "OvernightLink",
    "Snapshot",
    "load_notes",
    "market_of",
    "note_followups",
    "overnight_links",
    "read_symbol_names",
    "render_watch_report",
    "snapshot",
]

NOTE_COLUMNS = ("date", "symbols", "category", "note")
"""Columns of the industry-notes log. ``symbols`` is ``;``-separated."""

RETURN_WINDOWS = (1, 5, 20, 60)
"""Trailing windows, in sessions, that every snapshot reports."""

FOLLOWUP_WINDOWS = (5, 20)
"""Sessions after a note over which the related symbols' returns are shown."""

YEAR = TRADING_DAYS_PER_YEAR
UNUSUAL_Z = 2.0
"""A daily move this many of its own standard deviations is flagged."""


def market_of(symbol: str) -> str:
    """The exchange a Yahoo-style symbol trades on, as ``"TW"`` or ``"US"``.

    >>> market_of("2408.TW"), market_of("6488.TWO"), market_of("MU")
    ('TW', 'TW', 'US')
    """
    return "TW" if symbol.upper().endswith((".TW", ".TWO")) else "US"


def read_symbol_names(path: str | Path) -> dict[str, str]:
    """Display names from ``config/universe.txt``: the first word of each comment.

    ``2408.TW  # 南亞科 Nanya Technology`` gives ``{"2408.TW": "南亞科"}``. A
    symbol with no comment is left out and is shown by its ticker alone.
    """
    names: dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        ticker, _, comment = line.partition("#")
        ticker, words = ticker.strip(), comment.split()
        if ticker and words:
            names[ticker] = words[0]
    return names


@dataclass(frozen=True)
class Snapshot:
    """One symbol's latest session, each figure set against its own history."""

    symbol: str
    date: pd.Timestamp
    close: float
    returns: dict[int, float]
    """Trailing simple return over each of :data:`RETURN_WINDOWS` sessions."""
    ytd: float
    high_52w: float
    low_52w: float
    range_position: float
    """Where the close sits in its 52-week range: 0 at the low, 1 at the high."""
    ma_gap: dict[int, float]
    """Close relative to its 20/60/200-session moving average, minus one."""
    volatility: float
    """Annualised 20-session volatility of daily log returns."""
    volatility_percentile: float
    """Share of the past three years' 20-session volatilities at or below today's."""
    move_z: float
    """Today's return over the standard deviation of the 60 returns before it."""
    similar_moves: int
    """Sessions in the prior year whose move was at least as large as today's."""
    volume_ratio: float
    """Today's volume over the average of the 20 sessions before it."""

    def flags(self) -> list[str]:
        """Plain-language notes on whatever about this session was out of the ordinary."""
        out: list[str] = []
        if math.isfinite(self.move_z) and abs(self.move_z) >= UNUSUAL_Z:
            direction = "大漲" if self.returns[1] > 0 else "大跌"
            out.append(
                f"{direction} {self.returns[1]:+.2%}，約為平常波動的 {abs(self.move_z):.1f} 倍"
                f"（過去一年同等幅度出現 {self.similar_moves} 次）"
            )
        if self.close >= self.high_52w:
            out.append("創 52 週新高")
        elif self.close <= self.low_52w:
            out.append("創 52 週新低")
        if math.isfinite(self.volume_ratio) and self.volume_ratio >= 2.0:
            out.append(f"成交量為 20 日均量的 {self.volume_ratio:.1f} 倍")
        if math.isfinite(self.volatility_percentile) and self.volatility_percentile >= 0.9:
            out.append(f"近 20 日波動高於近三年 {self.volatility_percentile:.0%} 的時候")
        return out


def _trailing_return(close: pd.Series, sessions: int) -> float:
    if len(close) <= sessions:
        return math.nan
    return float(close.iloc[-1] / close.iloc[-1 - sessions] - 1.0)


def snapshot(ohlcv: pd.DataFrame, symbol: str) -> Snapshot:
    """Describe the last bar of ``ohlcv`` against the bars before it.

    Every statistic is computed from data up to and including the last bar;
    the baselines a move is judged against (its standard deviation, the
    average volume) stop the bar *before* it, so today never inflates the yard
    stick it is measured with.
    """
    if len(ohlcv) < 2:
        raise ValueError(f"{symbol}: need at least two bars, got {len(ohlcv)}")
    close = ohlcv["close"].astype(float)
    last = float(close.iloc[-1])
    date = pd.Timestamp(close.index[-1])

    prior_year = close[close.index.year < date.year]
    ytd = last / float(prior_year.iloc[-1]) - 1.0 if len(prior_year) else math.nan

    window = close.iloc[-YEAR:]
    high, low = float(window.max()), float(window.min())
    range_position = (last - low) / (high - low) if high > low else math.nan

    ma_gap = {
        n: (last / float(close.iloc[-n:].mean()) - 1.0) if len(close) >= n else math.nan
        for n in (20, 60, 200)
    }

    log_returns = np.log(close).diff().dropna()
    rolling_vol = log_returns.rolling(20).std() * math.sqrt(YEAR)
    volatility = float(rolling_vol.iloc[-1]) if len(rolling_vol) else math.nan
    history = rolling_vol.iloc[-3 * YEAR :].dropna()
    vol_pct = (
        float((history <= volatility).mean())
        if len(history) and np.isfinite(volatility)
        else math.nan
    )

    simple = close.pct_change().dropna()
    today = float(simple.iloc[-1])
    baseline = simple.iloc[-61:-1]
    spread = float(baseline.std()) if len(baseline) >= 20 else math.nan
    move_z = today / spread if spread and math.isfinite(spread) else math.nan
    similar = int((simple.iloc[-YEAR - 1 : -1].abs() >= abs(today)).sum())

    volume_ratio = math.nan
    if "volume" in ohlcv:
        volume = ohlcv["volume"].astype(float)
        average = float(volume.iloc[-21:-1].mean()) if len(volume) > 20 else math.nan
        if average and math.isfinite(average):
            volume_ratio = float(volume.iloc[-1]) / average

    return Snapshot(
        symbol=symbol,
        date=date,
        close=last,
        returns={n: _trailing_return(close, n) for n in RETURN_WINDOWS},
        ytd=ytd,
        high_52w=high,
        low_52w=low,
        range_position=range_position,
        ma_gap=ma_gap,
        volatility=volatility,
        volatility_percentile=vol_pct,
        move_z=move_z,
        similar_moves=similar,
        volume_ratio=volume_ratio,
    )


@dataclass(frozen=True)
class OvernightLink:
    """How a Taiwan stock's opening gap has followed a US stock's previous session.

    The US session dated ``d - 1`` closes before Taipei opens on ``d``, so its
    return is public information at the Taiwan open. The gap is where that
    information shows up; ``slope`` and ``correlation`` say how much of it has,
    over the ``n`` sessions before ``date``.
    """

    leader: str
    follower: str
    date: pd.Timestamp
    leader_return: float
    """The leader's return over its last session before ``date``."""
    gap: float
    """The follower's open on ``date`` against its previous close."""
    intraday: float
    """The follower's close on ``date`` against its own open."""
    slope: float
    correlation: float
    intraday_correlation: float
    """Correlation of the leader's return with the follower's open-to-close move."""
    n: int

    @property
    def expected_gap(self) -> float:
        return self.slope * self.leader_return


def _corr(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation, ``NaN`` (not a warning) when either side is constant."""
    if a.std() == 0 or b.std() == 0:
        return math.nan
    return float(a.corr(b))


def overnight_links(
    leader: pd.DataFrame,
    follower: pd.DataFrame,
    *,
    leader_name: str,
    follower_name: str,
    window: int = YEAR,
) -> OvernightLink | None:
    """Pair each follower session with the leader session that closed before it opened.

    Returns ``None`` when fewer than 30 paired sessions exist.
    """
    leader_ret = leader["close"].astype(float).pct_change().dropna()
    f_close = follower["close"].astype(float)
    gap = (follower["open"].astype(float) / f_close.shift(1) - 1.0).dropna()
    intraday = (f_close / follower["open"].astype(float) - 1.0).reindex(gap.index)

    # For each follower date, the last leader date strictly before it.
    position = leader_ret.index.searchsorted(gap.index, side="left") - 1
    valid = position >= 0
    paired = pd.DataFrame(
        {
            "leader": leader_ret.to_numpy()[position[valid]],
            "gap": gap.to_numpy()[valid],
            "intraday": intraday.to_numpy()[valid],
        },
        index=gap.index[valid],
    ).dropna()
    # The latest session is the one being explained, so it is kept out of the
    # history it is explained with.
    recent = paired.iloc[-window - 1 : -1]
    if len(recent) < 30:
        return None
    variance = float(recent["leader"].var())
    slope = float(recent["leader"].cov(recent["gap"]) / variance) if variance > 0 else math.nan
    last = paired.iloc[-1]
    return OvernightLink(
        leader=leader_name,
        follower=follower_name,
        date=pd.Timestamp(paired.index[-1]),
        leader_return=float(last["leader"]),
        gap=float(last["gap"]),
        intraday=float(last["intraday"]),
        slope=slope,
        correlation=_corr(recent["leader"], recent["gap"]),
        intraday_correlation=_corr(recent["leader"], recent["intraday"]),
        n=len(recent),
    )


def load_notes(path: str | Path) -> pd.DataFrame:
    """Read the industry-notes log; a missing file is an empty log, not an error."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(NOTE_COLUMNS))
    notes = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in NOTE_COLUMNS if c not in notes.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(missing)}")
    notes["date"] = pd.to_datetime(notes["date"])
    return notes.sort_values("date", kind="stable").reset_index(drop=True)


def note_followups(notes: pd.DataFrame, universe: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Each note's related symbols, with their returns over the sessions that followed.

    The base is the last close on or before the note's date - the price the
    writer could already see. A window that has not fully elapsed is ``NaN``:
    a partial outcome is not shown as if it were the answer.
    """
    rows = []
    for _, note in notes.iterrows():
        for symbol in [s.strip() for s in str(note["symbols"]).split(";") if s.strip()]:
            row = {"date": note["date"], "symbol": symbol, "category": note["category"]}
            row["note"] = note["note"]
            frame = universe.get(symbol)
            for n in FOLLOWUP_WINDOWS:
                row[f"ret_{n}"] = math.nan
            if frame is not None:
                close = frame["close"].astype(float)
                base = close.index.searchsorted(note["date"], side="right") - 1
                if base >= 0:
                    for n in FOLLOWUP_WINDOWS:
                        if base + n < len(close):
                            row[f"ret_{n}"] = float(close.iloc[base + n] / close.iloc[base] - 1.0)
            rows.append(row)
    columns = ["date", "symbol", "category", "note", *(f"ret_{n}" for n in FOLLOWUP_WINDOWS)]
    return pd.DataFrame(rows, columns=columns)


def _pct(value: float, digits: int = 1, *, signed: bool = True) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:+.{digits}%}" if signed else f"{value:.{digits}%}"


def _label(symbol: str, names: Mapping[str, str]) -> str:
    name = names.get(symbol)
    return f"{name} {symbol}" if name else symbol


def _range_bar(position: float, width: int = 10) -> str:
    """``low ├───●──┤ high`` as text, so the table reads without a chart."""
    if not math.isfinite(position):
        return "—"
    filled = min(width - 1, max(0, int(round(position * (width - 1)))))
    return "".join("●" if i == filled else "─" for i in range(width))


def render_watch_report(
    snapshots: list[Snapshot],
    *,
    names: Mapping[str, str] | None = None,
    links: list[OvernightLink] | None = None,
    followups: pd.DataFrame | None = None,
    recent_notes: int = 10,
) -> str:
    """The daily watch report as Markdown, in Traditional Chinese."""
    names = names or {}
    if not snapshots:
        raise ValueError("no symbols to report on")
    as_of = max(s.date for s in snapshots)
    report = Report(
        f"每日看盤報告 {as_of:%Y-%m-%d}",
        subtitle="只描述已發生的事，並以每檔股票自己的歷史為比較基準。不是預測，也不構成投資建議。",
    )

    report.heading("今日重點")
    highlights = [f"**{_label(s.symbol, names)}**：{flag}" for s in snapshots for flag in s.flags()]
    for s in snapshots:
        if s.date < as_of:
            highlights.append(
                f"**{_label(s.symbol, names)}**：資料只到 {s.date:%Y-%m-%d}"
                "（休市、停牌或下載延遲），以下數字都是那天的"
            )
    report.bullets(highlights or ["沒有異常：所有標的的漲跌、成交量與波動都在各自的正常範圍內。"])

    report.heading("漲跌幅")
    report.table(
        ["標的", "日期", "收盤", "1 日", "5 日", "20 日", "60 日", "今年以來"],
        [
            [
                _label(s.symbol, names),
                f"{s.date:%m-%d}",
                f"{s.close:,.2f}",
                *(_pct(s.returns[n]) for n in RETURN_WINDOWS),
                _pct(s.ytd),
            ]
            for s in snapshots
        ],
    )

    report.heading("位置與趨勢")
    report.table(
        [
            "標的",
            "52 週低",
            "區間位置",
            "52 週高",
            "距高點",
            "vs 20 日線",
            "vs 60 日線",
            "vs 200 日線",
        ],
        [
            [
                _label(s.symbol, names),
                f"{s.low_52w:,.2f}",
                _range_bar(s.range_position),
                f"{s.high_52w:,.2f}",
                _pct(s.close / s.high_52w - 1.0),
                *(_pct(s.ma_gap[n]) for n in (20, 60, 200)),
            ]
            for s in snapshots
        ],
    )
    report.text("「vs 均線」是收盤價高於（+）或低於（−）該均線多少。")

    report.heading("今天有多不尋常")
    report.table(
        [
            "標的",
            "今日漲跌",
            "為平常的幾倍",
            "過去一年同等幅度次數",
            "量 / 20 日均量",
            "20 日年化波動",
            "波動在近三年的百分位",
        ],
        [
            [
                _label(s.symbol, names),
                _pct(s.returns[1], 2),
                f"{abs(s.move_z):.1f}" if math.isfinite(s.move_z) else "—",
                str(s.similar_moves),
                f"{s.volume_ratio:.2f}" if math.isfinite(s.volume_ratio) else "—",
                _pct(s.volatility, 0, signed=False),
                _pct(s.volatility_percentile, 0, signed=False),
            ]
            for s in snapshots
        ],
    )
    report.bullets(
        [
            "「為平常的幾倍」是今日漲跌除以前 60 個交易日的日報酬標準差；2 倍以上會列入今日重點。",
            "「過去一年同等幅度次數」數的是前 252 個交易日裡漲或跌至少這麼多的天數。"
            "0 次代表一年來最大的一天。",
            "波動百分位 90% 代表近三年只有一成的時間比現在更動盪。",
        ]
    )

    if links:
        report.heading("美股收盤後，台股隔天開盤怎麼反應")
        report.table(
            [
                "台股",
                "參考美股",
                "美股前一晚",
                "依過去一年推估的跳空",
                "實際開盤跳空",
                "開盤後到收盤",
                "跳空相關",
                "盤中相關",
            ],
            [
                [
                    _label(link.follower, names),
                    _label(link.leader, names),
                    _pct(link.leader_return, 2),
                    _pct(link.expected_gap, 2),
                    _pct(link.gap, 2),
                    _pct(link.intraday, 2),
                    f"{link.correlation:.2f}",
                    f"{link.intraday_correlation:.2f}",
                ]
                for link in links
            ],
        )
        report.text(
            "美股前一晚的漲跌在台股開盤前就已公開，所以它幾乎全反映在開盤跳空上（跳空相關高），"
            "開盤之後就幾乎沒有關係（盤中相關接近 0）。推估跳空與實際差距大的日子，"
            "代表台股當天有美股以外的消息。相關係數取過去一年的交易日。"
        )

    report.heading("產業觀察紀錄")
    if followups is None or followups.empty:
        report.text(
            "還沒有紀錄。在 `data/notes/industry.csv` 加一列即可，欄位為 "
            "`date,symbols,category,note`，例如：\n\n"
            "```\n2026-10-09,2408.TW;2344.TW,報價,DRAM 合約價第四季預估調漲 10-15%\n```\n\n"
            "寫下的日期之後的股價，就是這則觀察無法事先看到的結果。累積幾個月後，"
            "可以回頭看哪一類消息之後股價真的有反應。"
        )
    else:
        latest_dates = sorted(followups["date"].unique())[-recent_notes:]
        shown = followups[followups["date"].isin(latest_dates)].iloc[::-1]
        rows = [
            [
                f"{row.date:%Y-%m-%d}",
                _label(row.symbol, names),
                str(row.category),
                str(row.note),
                *(_pct(getattr(row, f"ret_{n}")) for n in FOLLOWUP_WINDOWS),
            ]
            for row in shown.itertuples(index=False)
        ]
        report.raw_table(
            markdown_table(["日期", "標的", "類別", "觀察", "之後 5 日", "之後 20 日"], rows)
        )
        report.text(
            "「之後 N 日」從紀錄當天（或之前最後一個交易日）的收盤算起；還沒走完的期間顯示「—」。"
            "這裡只列出發生了什麼，一則紀錄之後股價上漲，不代表兩者有因果關係。"
        )

    return report.render()
