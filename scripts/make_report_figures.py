"""Render the figures for a docs/reports/<date>/ test-run report.

The package itself draws its charts in ASCII and depends on nothing beyond
numpy, pandas and scikit-learn. This script is the one place matplotlib is
used, and only to turn numbers the CLI already produced into PNGs for a
human-readable report - it computes no statistic the CLI does not.

    pip install matplotlib
    python3 scripts/make_report_figures.py --run-dir RUN --out docs/reports/DATE/img

``RUN`` is the directory the report's CLI commands wrote into
(``compare/comparison.csv``, ``screen/screen_random_forest.csv``,
``journal/journal_scored_random_forest.csv``). The one figure that needs more
than a CSV - the rotation null distributions - is recomputed through
:func:`ai_stock.pipeline.run_simulation` with the CLI's defaults, and prints its
p-values so they can be checked against the CLI's own output.

It never reads or writes ``data/journal/forecasts.csv`` - only the scored
export the ``journal`` command already wrote.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Reference categorical palette, light mode, slots 1-4 in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b3"
POS, NEG = "#2a78d6", "#e34948"  # diverging poles (blue <-> red)
UNIVERSE = ["MU", "2408.TW", "2344.TW", "2337.TW"]  # colour follows the symbol


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK,
            "lines.linewidth": 2,
            "figure.dpi": 110,
        }
    )


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    print(f"wrote {out / name}")


def fig_validation(plt, out: Path) -> None:
    """Rotation null vs realised Sharpe, planted edge and efficient market side by side."""
    from ai_stock.config import ExperimentConfig, SimulationConfig
    from ai_stock.data import generate_ohlcv
    from ai_stock.pipeline import run_simulation

    markets = {
        "植入邊際（ar1=0.06, reversion=-0.05）": generate_ohlcv(n_days=3000, seed=42),
        "效率市場（ar1=0, reversion=0）": generate_ohlcv(
            n_days=3000, seed=42, ar1=0.0, reversion=0.0
        ),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    tallest = 0.0
    for ax, (label, ohlcv) in zip(axes, markets.items(), strict=True):
        bundle = run_simulation(
            ohlcv, "ridge", ExperimentConfig(), simulation=SimulationConfig(n_paths=1000)
        )
        sig = bundle.significance
        null = np.asarray(sig.null_distribution, dtype=float)
        print(f"{label}: Sharpe {sig.observed:.4f}  p {sig.p_value:.4f}")
        counts, _, _ = ax.hist(null, bins=30, color=NEUTRAL, edgecolor=SURFACE, linewidth=1)
        tallest = max(tallest, float(counts.max()))
        # Labels sit in axes-fraction height, in the headroom left above the bars.
        band = ax.get_xaxis_transform()
        q95 = float(np.quantile(null, 0.95))
        ax.axvline(q95, color=INK_2, linestyle="--", linewidth=1.2)
        ax.text(q95, 0.94, "虛無 95 百分位 ", color=INK_2, fontsize=9, ha="right", transform=band)
        ax.axvline(sig.observed, color=SERIES[0], linewidth=2.5)
        # Label on whichever side of the line is clear of the 95th-percentile mark.
        left = sig.observed < q95
        ax.text(
            sig.observed,
            0.80,
            f"實現 Sharpe {sig.observed:.2f} \np = {sig.p_value:.3f} "
            if left
            else f" 實現 Sharpe {sig.observed:.2f}\n p = {sig.p_value:.3f}",
            color=INK,
            fontsize=10,
            ha="right" if left else "left",
            va="top",
            transform=band,
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("年化 Sharpe（循環位移虛無分布）")
    axes[0].set_ylim(0, tallest * 1.45)
    axes[0].set_ylabel("排列次數")
    fig.suptitle("圖 1  框架驗證：有邊際時抓得到，沒有邊際時不無中生有", x=0.01, ha="left")
    _save(fig, out, "fig1_validation.png")


def fig_models(plt, run_dir: Path, out: Path) -> None:
    table = pd.read_csv(run_dir / "compare" / "comparison.csv")
    name_col = "model" if "model" in table.columns else table.columns[0]
    table = table.dropna(subset=["sharpe"]).sort_values("sharpe")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(table))
    for ax, col, title in (
        (axes[0], "excess_sharpe", "超額 Sharpe（相對買進持有）"),
        (axes[1], "ic_fold_t", "跨 fold IC t 值"),
    ):
        values = table[col].to_numpy(float)
        colors = [POS if v >= 0 else NEG for v in np.nan_to_num(values)]
        ax.barh(y, np.nan_to_num(values), color=colors, height=0.6)
        for yi, v in zip(y, values, strict=True):
            if np.isfinite(v):
                ax.text(
                    v,
                    yi,
                    f" {v:.2f} " if v >= 0 else f"{v:.2f} ",
                    va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=9,
                    color=INK,
                )
            else:
                ax.text(0, yi, " n/a（不產生訊號變化）", va="center", fontsize=9, color=INK_2)
        finite = values[np.isfinite(values)]
        span = finite.max() - min(finite.min(), 0.0)
        ax.set_xlim(min(finite.min(), 0.0) - 0.3 * span, finite.max() + 0.15 * span)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[1].axvline(2, color=INK_2, linestyle="--", linewidth=1)
    axes[0].set_yticks(y, table[name_col])
    fig.suptitle("圖 2  合成市場上的模型比較（相同 18 個 fold）", x=0.01, ha="left")
    _save(fig, out, "fig2_models.png")


def fig_prices(plt, out: Path, since: str = "2025-01-01") -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ends = {}
    for symbol, color in zip(UNIVERSE, SERIES, strict=True):
        frame = pd.read_csv(f"data/prices/{symbol}.csv", parse_dates=["date"]).set_index("date")
        close = frame["close"].loc[since:]
        indexed = 100 * close / close.iloc[0]
        ax.plot(indexed.index, indexed, color=color, label=symbol)
        ends[symbol] = (indexed.index[-1], float(indexed.iloc[-1]))
    # End labels, nudged apart so two series finishing close together stay legible.
    gap = 0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    placed: list[float] = []
    for symbol, (x, value) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        y = value if not placed else max(value, placed[-1] + gap)
        placed.append(y)
        ax.text(x, y, f" {symbol} {value:.0f}", color=INK, fontsize=9, va="center")
    ax.axhline(100, color=INK_2, linewidth=1)
    ax.set_ylabel(f"收盤價指數（{since[:7]} = 100）")
    ax.legend(loc="upper left")
    ax.set_title("圖 3  研究標的：記憶體族群 2025 年以來的走勢")
    _save(fig, out, "fig3_prices.png")


def fig_screen(plt, run_dir: Path, out: Path) -> None:
    table = pd.read_csv(run_dir / "screen" / "screen_random_forest.csv")
    table = table.sort_values("excess_sharpe")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    y = np.arange(len(table))
    values = table["excess_sharpe"].to_numpy(float)
    ax.barh(y, values, color=[POS if v >= 0 else NEG for v in values], height=0.55)
    for yi, row in zip(y, table.itertuples(), strict=True):
        ax.text(
            row.excess_sharpe,
            yi,
            f"{row.excess_sharpe:.2f}   p={row.p_value:.3f}  q={row.q_value:.3f} ",
            va="center",
            ha="right",
            fontsize=9,
            color=INK,
        )
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(y, table["symbol"])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(values.min() * 1.9, -1.6), 0.25)
    ax.set_xlabel("超額 Sharpe = 策略 Sharpe − 買進持有 Sharpe")
    ax.set_title("圖 4  真實資料：random_forest（h=5）四檔全數輸給買進持有")
    _save(fig, out, "fig4_screen.png")


def fig_journal(plt, run_dir: Path, out: Path) -> None:
    scored = pd.read_csv(
        run_dir / "journal" / "journal_scored_random_forest.csv", parse_dates=["asof_date"]
    )
    daily = scored.groupby("asof_date")[["pnl", "realised_return"]].sum().cumsum()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    for col, color, label in (
        ("pnl", SERIES[0], "模型（扣成本）"),
        ("realised_return", SERIES[1], "一直做多（不扣成本）"),
    ):
        ax.plot(daily.index, 100 * daily[col], color=color, marker="o", markersize=4, label=label)
        ax.text(
            daily.index[-1],
            100 * daily[col].iloc[-1],
            f" {100 * daily[col].iloc[-1]:+.1f}%",
            color=INK,
            fontsize=9,
            va="center",
        )
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_ylabel("累積 P&L（%，逐列加總）")
    ax.legend(loc="lower left")
    ax.set_title("實盤日誌：累積 P&L", fontsize=11)
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    decided = scored[scored["position"] != 0]
    per = (
        decided.assign(
            model_hit=np.sign(decided["position"]) == np.sign(decided["realised_return"]),
            long_hit=decided["realised_return"] > 0,
        )
        .groupby("symbol")[["model_hit", "long_hit"]]
        .mean()
    )
    per = per.reindex([s for s in UNIVERSE if s in per.index])
    y = np.arange(len(per))
    ax.barh(y + 0.18, 100 * per["model_hit"], height=0.34, color=SERIES[0], label="模型")
    ax.barh(y - 0.18, 100 * per["long_hit"], height=0.34, color=SERIES[1], label="一直做多")
    for yi, (m, a) in zip(y, per.to_numpy(), strict=True):
        ax.text(100 * m, yi + 0.18, f" {100 * m:.0f}%", va="center", fontsize=8, color=INK)
        ax.text(100 * a, yi - 0.18, f" {100 * a:.0f}%", va="center", fontsize=8, color=INK)
    ax.axvline(50, color=INK_2, linestyle="--", linewidth=1)
    ax.set_yticks(y, per.index)
    ax.set_xlim(0, 110)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("命中率（%）")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("各標的命中率", fontsize=11)
    fig.suptitle(f"圖 5  預測日誌（{len(scored)} 筆到期）：模型 vs 一直做多", x=0.01, ha="left")
    _save(fig, out, "fig5_journal.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-validation", action="store_true", help="skip the slow figure 1")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_validation:
        fig_validation(plt, args.out)
    fig_models(plt, args.run_dir, args.out)
    fig_prices(plt, args.out)
    fig_screen(plt, args.run_dir, args.out)
    fig_journal(plt, args.run_dir, args.out)


if __name__ == "__main__":
    main()
