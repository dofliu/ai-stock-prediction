# recorder.py — 預測記錄與績效追蹤模組
import json
import csv
import os
from datetime import datetime, timedelta
from config import PREDICTIONS_FILE, PERFORMANCE_FILE, REPORTS_DIR, BACKTEST_DAYS
from data_fetcher import fetch_latest_price, get_stock_name


# ── 儲存預測 ─────────────────────────────────────────────────

def save_prediction(date_str: str, recommendations: list[dict]):
    """將今日推薦寫入 predictions.json（追加）"""
    records = _load_json(PREDICTIONS_FILE)

    records[date_str] = [
        {
            "rank":   r["rank"],
            "ticker": r["ticker"],
            "name":   r["name"],
            "score":  r["score"],
            "close":  r["close"],
            "signals": r["signals"],
        }
        for r in recommendations
    ]

    _save_json(PREDICTIONS_FILE, records)


# ── 儲存報告文字檔 ────────────────────────────────────────────

def save_report(date_str: str, report_text: str):
    """將報告存為 reports/YYYY-MM-DD.txt"""
    path = os.path.join(REPORTS_DIR, f"{date_str}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return path


# ── 績效追蹤 ──────────────────────────────────────────────────

def update_performance(today_str: str):
    """
    對 BACKTEST_DAYS 天前的預測進行績效評估，
    計算推薦股票的實際漲跌幅並寫入 performance.json。
    """
    eval_date = (
        datetime.strptime(today_str, "%Y-%m-%d") - timedelta(days=BACKTEST_DAYS)
    ).strftime("%Y-%m-%d")

    predictions = _load_json(PREDICTIONS_FILE)
    if eval_date not in predictions:
        return None  # 無可評估的預測

    performance = _load_json(PERFORMANCE_FILE)
    if eval_date in performance:
        return performance[eval_date]  # 已評估過

    past_recs = predictions[eval_date]
    results = []
    for rec in past_recs:
        ticker     = rec["ticker"]
        entry_price = rec["close"]
        exit_price  = fetch_latest_price(ticker)

        if exit_price and entry_price and entry_price > 0:
            change_pct = round((exit_price - entry_price) / entry_price * 100, 2)
        else:
            change_pct = None

        results.append({
            "ticker":       ticker,
            "name":         rec["name"],
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "change_pct":   change_pct,
        })

    performance[eval_date] = {
        "eval_date":    today_str,
        "predict_date": eval_date,
        "stocks":       results,
        "avg_return":   _avg_return(results),
    }

    _save_json(PERFORMANCE_FILE, performance)
    return performance[eval_date]


def _avg_return(results: list) -> float | None:
    valid = [r["change_pct"] for r in results if r["change_pct"] is not None]
    return round(sum(valid) / len(valid), 2) if valid else None


# ── 歷史記錄摘要 ─────────────────────────────────────────────

def load_history_summary() -> str:
    """讀取過去所有預測與績效，輸出摘要字串"""
    predictions  = _load_json(PREDICTIONS_FILE)
    performance  = _load_json(PERFORMANCE_FILE)

    if not predictions:
        return "  （尚無歷史紀錄）"

    lines = ["  ── 歷史預測摘要 ──"]
    for date in sorted(predictions.keys(), reverse=True)[:10]:  # 最近 10 筆
        recs = predictions[date]
        tickers = ", ".join(f"{r['ticker']}({r['name']})" for r in recs)
        lines.append(f"  {date}：{tickers}")

        if date in performance:
            p = performance[date]
            avg = p.get("avg_return")
            avg_str = f"{avg:+.2f}%" if avg is not None else "N/A"
            lines.append(f"          └ {BACKTEST_DAYS}日後平均報酬：{avg_str}")

    return "\n".join(lines)


def export_csv(output_path: str = None):
    """將全部預測記錄匯出為 CSV"""
    predictions = _load_json(PREDICTIONS_FILE)
    performance = _load_json(PERFORMANCE_FILE)

    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, "all_predictions.csv")

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["日期", "排名", "代碼", "名稱", "推薦分數", "推薦收盤價",
                         "5日後收盤", "5日報酬率(%)"])
        for date in sorted(predictions.keys()):
            perf_stocks = {s["ticker"]: s for s in performance.get(date, {}).get("stocks", [])}
            for rec in predictions[date]:
                perf = perf_stocks.get(rec["ticker"], {})
                writer.writerow([
                    date,
                    rec["rank"],
                    rec["ticker"],
                    rec["name"],
                    rec["score"],
                    rec["close"],
                    perf.get("exit_price", ""),
                    perf.get("change_pct", ""),
                ])

    return output_path


# ── 工具函式 ──────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
