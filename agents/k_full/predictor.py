# predictor.py — 選股與推薦模組
from config import TOP_N, SCORE_WEIGHTS
from data_fetcher import get_stock_name


# 評分維度中文說明
SCORE_LABELS = {
    "trend_score":    "趨勢",
    "momentum_score": "動能",
    "rsi_score":      "RSI",
    "volume_score":   "量能",
    "bb_score":       "布林",
    "ma_cross_score": "均線交叉",
}


def select_top(analysis_results: dict, top_n: int = TOP_N) -> list[dict]:
    """從分析結果中選出評分最高的 top_n 檔，回傳推薦清單"""
    ranked = sorted(
        analysis_results.values(),
        key=lambda x: x.get("total_score", 0),
        reverse=True,
    )

    recommendations = []
    for rank, item in enumerate(ranked[:top_n], 1):
        ticker = item["ticker"]
        name   = get_stock_name(ticker)
        score  = item["total_score"]
        close  = item.get("close", 0)

        # 產生訊號說明
        signals = _build_signals(item)
        
        # 產生建議理由
        reason = _build_recommendation_reason(item, score)

        recommendations.append({
            "rank":    rank,
            "ticker":  ticker,
            "name":    name,
            "score":   score,
            "close":   close,
            "signals": signals,
            "recommendation_reason": reason,
            "detail":  {k: round(item.get(k, 0), 3) for k in SCORE_WEIGHTS},
        })

    return recommendations


def _build_recommendation_reason(item: dict, total_score: float) -> str:
    """根據評分產生建議理由"""
    reasons = []
    
    if total_score >= 0.6:
        reasons.append("技術面強勢，多項指標同時偏多")
    elif total_score >= 0.4:
        reasons.append("技術面中性偏多，有一定上漲潛力")
    else:
        reasons.append("技術面觀望為主，建議小部位操作")
    
    # 根據趨勢
    if item.get("trend_score", 0) >= 0.8:
        reasons.append("均線呈現多頭排列，中長線趨勢明確")
    
    # 根據動能
    if item.get("momentum_score", 0) >= 0.8:
        reasons.append("MACD 動能強勁，短期漲勢可期")
    
    # 根據 RSI
    rsi = item.get("rsi_score", 0)
    if rsi >= 0.8:
        reasons.append("RSI 處於最佳進場區")
    elif rsi <= 0.3:
        reasons.append("RSI 超賣，可能有反彈機會")
    
    return "；".join(reasons)


def _build_signals(item: dict) -> list[str]:
    """根據各指標分數產生人類可讀的訊號說明"""
    signals = []

    trend = item.get("trend_score", 0)
    if trend >= 0.8:
        signals.append("均線完美多頭排列")
    elif trend >= 0.5:
        signals.append("均線偏多")

    momentum = item.get("momentum_score", 0)
    if momentum >= 0.9:
        signals.append("MACD 正值且持續上揚")
    elif momentum >= 0.5:
        signals.append("MACD 動能轉正")

    rsi = item.get("rsi_score", 0)
    if rsi >= 0.9:
        signals.append("RSI 超賣反彈（最佳進場區）")
    elif rsi >= 0.6:
        signals.append("RSI 中性偏多")

    vol = item.get("volume_score", 0)
    if vol >= 0.9:
        signals.append("成交量爆量（2x 均量）")
    elif vol >= 0.6:
        signals.append("量能放大")

    bb = item.get("bb_score", 0)
    if bb >= 0.9:
        signals.append("股價接近布林下軌（逢低布局）")

    cross = item.get("ma_cross_score", 0)
    if cross >= 0.9:
        signals.append("5MA 上穿 20MA 黃金交叉")
    elif cross >= 0.4:
        signals.append("5MA 站上 20MA")

    return signals if signals else ["無明顯技術訊號"]


def format_report(recommendations: list[dict], date_str: str) -> str:
    """格式化每日推薦報告為可讀字串"""
    lines = [
        "=" * 55,
        f"  台灣股市每日推薦  【{date_str}】",
        "=" * 55,
    ]

    for rec in recommendations:
        lines.append(
            f"\n  #{rec['rank']}  {rec['ticker']}  {rec['name']}"
            f"   收盤價：{rec['close']:.2f}  評分：{rec['score']:.3f}"
        )
        
        # 分析的理由
        if rec.get('signals'):
            lines.append("  【分析理由】")
            for sig in rec['signals']:
                lines.append(f"    • {sig}")
        
        # 建議的理由
        if rec.get('recommendation_reason'):
            lines.append(f"  【建議理由】{rec['recommendation_reason']}")
        
        # 各維度分數
        lines.append("  【技術指標】")
        detail_parts = []
        for key, label in SCORE_LABELS.items():
            val = rec["detail"].get(key, 0)
            bar = "█" * int(val * 5)
            detail_parts.append(f"    {label:<6} {bar:<5} {val:.2f}")
        lines.extend(detail_parts)

    lines.append("\n" + "=" * 55)
    lines.append("  ⚠ 本推薦僅供技術分析參考，投資需自行判斷風險。")
    lines.append("=" * 55)
    return "\n".join(lines)
