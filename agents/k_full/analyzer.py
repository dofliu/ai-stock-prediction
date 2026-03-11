# analyzer.py — 技術指標計算與評分模組
import pandas as pd
import numpy as np
from config import TA_PARAMS, SCORE_WEIGHTS


# ── 技術指標計算 ─────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def calc_ma(close: pd.Series, periods: list) -> dict:
    return {f"ma{p}": close.rolling(p).mean() for p in periods}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在 DataFrame 上計算所有技術指標，回傳附加指標欄位的 DataFrame"""
    p = TA_PARAMS
    close = df["Close"]
    volume = df["Volume"]

    # RSI
    df["rsi"] = calc_rsi(close, p["rsi_period"])

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(
        close, p["macd_fast"], p["macd_slow"], p["macd_signal"]
    )

    # Bollinger Bands
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = calc_bollinger(
        close, p["bb_period"], p["bb_std"]
    )

    # 移動平均線
    for name, ma in calc_ma(close, [p["ma_short"], p["ma_mid"], p["ma_long"]]).items():
        df[name] = ma

    # 成交量均線
    df["vol_ma"] = volume.rolling(p["volume_ma"]).mean()

    return df


# ── 評分計算 ─────────────────────────────────────────────────

def score_stock(df: pd.DataFrame) -> dict:
    """對一檔股票計算各維度評分（0~1），回傳評分字典"""
    p = TA_PARAMS
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    scores = {}

    # 1. 趨勢評分：短中長均線多頭排列
    ma5  = latest.get(f"ma{p['ma_short']}", np.nan)
    ma20 = latest.get(f"ma{p['ma_mid']}", np.nan)
    ma60 = latest.get(f"ma{p['ma_long']}", np.nan)
    close = latest["Close"]
    trend = 0
    if not any(pd.isna([ma5, ma20, ma60])):
        if close > ma5:   trend += 0.25
        if ma5   > ma20:  trend += 0.35
        if ma20  > ma60:  trend += 0.40
    scores["trend_score"] = trend

    # 2. 動能評分：MACD histogram 正值且上升
    macd_hist = latest.get("macd_hist", np.nan)
    prev_hist  = prev.get("macd_hist", np.nan)
    momentum = 0
    if not pd.isna(macd_hist):
        if macd_hist > 0:           momentum += 0.5
        if not pd.isna(prev_hist) and macd_hist > prev_hist:
            momentum += 0.5
    scores["momentum_score"] = momentum

    # 3. RSI 評分：超賣反彈區（30~60 最佳）
    rsi = latest.get("rsi", np.nan)
    rsi_score = 0
    if not pd.isna(rsi):
        if rsi < 30:      rsi_score = 0.3   # 超賣但可能持續下跌
        elif rsi <= 45:   rsi_score = 1.0   # 剛脫離超賣，最佳
        elif rsi <= 60:   rsi_score = 0.7   # 中性偏多
        elif rsi <= 70:   rsi_score = 0.4   # 偏高
        else:             rsi_score = 0.1   # 超買
    scores["rsi_score"] = rsi_score

    # 4. 量能評分：今日成交量 vs 均量
    vol = latest.get("Volume", np.nan)
    vol_ma = latest.get("vol_ma", np.nan)
    vol_score = 0
    if not pd.isna(vol) and not pd.isna(vol_ma) and vol_ma > 0:
        ratio = vol / vol_ma
        if ratio >= 2.0:   vol_score = 1.0
        elif ratio >= 1.5: vol_score = 0.8
        elif ratio >= 1.2: vol_score = 0.6
        elif ratio >= 1.0: vol_score = 0.4
        else:              vol_score = 0.2
    scores["volume_score"] = vol_score

    # 5. 布林通道評分：接近下軌（逢低布局）
    bb_upper = latest.get("bb_upper", np.nan)
    bb_lower = latest.get("bb_lower", np.nan)
    bb_score = 0
    if not any(pd.isna([bb_upper, bb_lower, close])):
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            position = (close - bb_lower) / bb_range  # 0=下軌, 1=上軌
            if position <= 0.2:    bb_score = 1.0
            elif position <= 0.4:  bb_score = 0.7
            elif position <= 0.6:  bb_score = 0.5
            elif position <= 0.8:  bb_score = 0.3
            else:                  bb_score = 0.1
    scores["bb_score"] = bb_score

    # 6. 均線交叉評分：5MA 上穿 20MA（黃金交叉）
    ma5_prev  = prev.get(f"ma{p['ma_short']}", np.nan)
    ma20_prev = prev.get(f"ma{p['ma_mid']}", np.nan)
    cross_score = 0
    if not any(pd.isna([ma5, ma20, ma5_prev, ma20_prev])):
        if ma5_prev < ma20_prev and ma5 > ma20:
            cross_score = 1.0    # 黃金交叉
        elif ma5 > ma20:
            cross_score = 0.5    # 多頭排列但無交叉
    scores["ma_cross_score"] = cross_score

    # 加權總分
    total = sum(scores[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
    scores["total_score"] = round(total, 4)

    # 附加最新收盤價
    scores["close"] = round(float(close), 2)

    return scores


def analyze_all(stock_data: dict) -> dict:
    """分析所有股票，回傳 {ticker: {scores..., indicators...}}"""
    results = {}
    for ticker, df in stock_data.items():
        try:
            df = add_indicators(df.copy())
            scores = score_stock(df)
            scores["ticker"] = ticker
            results[ticker] = scores
        except Exception as e:
            pass  # 跳過分析失敗的股票
    return results
