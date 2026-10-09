"""The daily watch report: snapshots, the overnight link, and the notes log."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_stock.cli import main
from ai_stock.data.loaders import save_csv
from ai_stock.watch import (
    load_notes,
    note_followups,
    overnight_links,
    read_symbol_names,
    render_watch_report,
    snapshot,
)


def _bars(close: np.ndarray, *, start: str = "2024-01-01", volume: float = 1000.0) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(close), name="date")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), volume),
        },
        index=index,
    )


@pytest.fixture
def noisy() -> np.ndarray:
    rng = np.random.default_rng(7)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 400)))


def test_trailing_returns_and_range(noisy: np.ndarray) -> None:
    snap = snapshot(_bars(noisy), "X")
    assert snap.returns[1] == pytest.approx(noisy[-1] / noisy[-2] - 1)
    assert snap.returns[20] == pytest.approx(noisy[-1] / noisy[-21] - 1)
    assert snap.high_52w == pytest.approx(noisy[-252:].max())
    assert snap.low_52w == pytest.approx(noisy[-252:].min())
    assert 0.0 <= snap.range_position <= 1.0


def test_ytd_is_measured_from_the_last_close_of_the_prior_year(noisy: np.ndarray) -> None:
    bars = _bars(noisy)
    snap = snapshot(bars, "X")
    prior = bars["close"][bars.index.year < bars.index[-1].year].iloc[-1]
    assert snap.ytd == pytest.approx(noisy[-1] / prior - 1)


def test_a_jump_is_judged_against_the_moves_before_it(noisy: np.ndarray) -> None:
    jumped = np.append(noisy, noisy[-252:].max() * 1.08)
    bars = _bars(jumped)
    bars.iloc[-1, bars.columns.get_loc("volume")] = 3000.0
    snap = snapshot(bars, "X")
    move = jumped[-1] / jumped[-2] - 1
    baseline = pd.Series(jumped).pct_change().iloc[-61:-1].std()
    assert snap.move_z == pytest.approx(move / baseline)
    assert snap.similar_moves == 0
    assert snap.volume_ratio == pytest.approx(3.0)
    flags = " ".join(snap.flags())
    assert "大漲" in flags and "52 週新高" in flags and "3.0 倍" in flags


def test_a_quiet_session_raises_no_flags() -> None:
    rng = np.random.default_rng(11)
    # Choppy early, calm late: today's volatility sits low in its own history,
    # and the 52-week extremes were set months ago.
    steps = np.concatenate([rng.normal(0, 0.03, 300), rng.normal(0, 0.005, 199), [0.0001]])
    close = 100.0 * np.exp(np.cumsum(steps))
    close[-120] = close[-200:].max() * 1.2
    close[-150] = close[-200:].min() * 0.8
    assert snapshot(_bars(close), "X").flags() == []


def test_short_history_is_reported_as_missing_not_invented() -> None:
    snap = snapshot(_bars(np.linspace(100, 110, 30)), "X")
    assert math.isnan(snap.returns[60])
    assert math.isnan(snap.ma_gap[200])
    assert math.isnan(snap.ytd)


def test_snapshot_needs_two_bars() -> None:
    with pytest.raises(ValueError, match="two bars"):
        snapshot(_bars(np.array([100.0])), "X")


def test_overnight_link_pairs_each_open_with_the_session_before_it() -> None:
    rng = np.random.default_rng(3)
    n = 300
    leader_close = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    leader = _bars(leader_close)
    leader_ret = np.nan_to_num(pd.Series(leader_close).pct_change().to_numpy())

    # The follower gaps by exactly half of the leader's previous session.
    close = np.empty(n)
    open_ = np.empty(n)
    close[0] = open_[0] = 100.0
    for t in range(1, n):
        open_[t] = close[t - 1] * (1 + 0.5 * leader_ret[t - 1])
        close[t] = open_[t] * (1 + rng.normal(0, 0.01))
    follower = _bars(close)
    follower["open"] = open_
    follower["high"] = np.maximum(open_, close) * 1.01
    follower["low"] = np.minimum(open_, close) * 0.99

    link = overnight_links(leader, follower, leader_name="L", follower_name="F")
    assert link is not None
    assert link.slope == pytest.approx(0.5, abs=1e-9)
    assert link.correlation == pytest.approx(1.0, abs=1e-9)
    assert abs(link.intraday_correlation) < 0.2
    assert link.leader_return == pytest.approx(leader_ret[-2])
    assert link.expected_gap == pytest.approx(link.gap, abs=1e-9)


def test_overnight_link_needs_enough_history() -> None:
    bars = _bars(np.linspace(100, 101, 20))
    assert overnight_links(bars, bars, leader_name="L", follower_name="F") is None


def test_symbol_names_come_from_the_universe_comments(tmp_path: Path) -> None:
    path = tmp_path / "universe.txt"
    path.write_text("# header\nMU   # 美光 Micron\n2408.TW # 南亞科\nAAPL\n", encoding="utf-8")
    assert read_symbol_names(path) == {"MU": "美光", "2408.TW": "南亞科"}
    assert read_symbol_names(tmp_path / "missing.txt") == {}


def test_a_missing_notes_log_is_empty(tmp_path: Path) -> None:
    assert load_notes(tmp_path / "none.csv").empty


def test_a_malformed_notes_log_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.csv"
    path.write_text("date,note\n2024-01-02,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="symbols"):
        load_notes(path)


def test_followups_start_from_the_close_the_writer_could_see(tmp_path: Path) -> None:
    close = np.arange(100.0, 130.0)
    universe = {"A": _bars(close, start="2024-01-01")}
    path = tmp_path / "notes.csv"
    # 2024-01-06 is a Saturday: the base is Friday 2024-01-05, the fifth bar.
    path.write_text(
        "date,symbols,category,note\n"
        "2024-01-06,A;ZZZ,報價,weekend note\n"
        "2024-02-08,A,需求,too recent\n",
        encoding="utf-8",
    )
    out = note_followups(load_notes(path), universe)
    first = out[(out["symbol"] == "A") & (out["note"] == "weekend note")].iloc[0]
    assert first["ret_5"] == pytest.approx(close[9] / close[4] - 1)
    assert first["ret_20"] == pytest.approx(close[24] / close[4] - 1)
    unknown = out[out["symbol"] == "ZZZ"].iloc[0]
    assert math.isnan(unknown["ret_5"])
    recent = out[out["note"] == "too recent"].iloc[0]
    assert math.isnan(recent["ret_20"])


def test_report_names_a_lagging_symbol_and_explains_an_empty_log(noisy: np.ndarray) -> None:
    fresh = snapshot(_bars(noisy), "A")
    stale = snapshot(_bars(noisy[:-3]), "B")
    text = render_watch_report([fresh, stale], names={"A": "甲"})
    assert "每日看盤報告" in text
    assert "甲 A" in text
    assert f"**B**：資料只到 {stale.date:%Y-%m-%d}" in text
    assert "還沒有紀錄" in text


def test_watch_command_writes_dated_and_latest_reports(tmp_path: Path, noisy: np.ndarray) -> None:
    prices = tmp_path / "prices"
    save_csv(_bars(noisy), prices / "MU.csv")
    save_csv(_bars(noisy * 3), prices / "2408.TW.csv")
    universe = tmp_path / "universe.txt"
    universe.write_text("MU # 美光\n2408.TW # 南亞科\n", encoding="utf-8")
    notes = tmp_path / "notes.csv"
    notes.write_text("date,symbols,category,note\n2024-06-03,2408.TW,報價,合約價上調\n")
    out = tmp_path / "out"

    args = ["watch", "--data", str(prices), "--universe", str(universe)]
    args += ["--notes", str(notes), "--out", str(out), "--quiet"]
    assert main(args) == 0

    latest = (out / "latest.md").read_text(encoding="utf-8")
    dated = list(out.glob("watch_*.md"))
    assert len(dated) == 1 and dated[0].read_text(encoding="utf-8") == latest
    assert "美光 MU" in latest and "南亞科 2408.TW" in latest
    assert "台股隔天開盤" in latest
    assert "合約價上調" in latest
