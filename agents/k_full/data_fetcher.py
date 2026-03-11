# data_fetcher.py — 台灣股市資料擷取模組
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import STOCK_POOL, HISTORY_DAYS
import logging

logger = logging.getLogger(__name__)

# 股票中文名稱對照表
STOCK_NAMES = {
    "2330.TW": "台積電",
    "2303.TW": "聯電",
    "2308.TW": "台達電",
    "2454.TW": "聯發科",
    "2379.TW": "瑞昱",
    "3711.TW": "日月光投控",
    "2344.TW": "華邦電",
    "2408.TW": "南亞科",
    "3034.TW": "聯詠",
    "2388.TW": "威盛",
    "2317.TW": "鴻海",
    "2382.TW": "廣達",
    "2357.TW": "華碩",
    "2353.TW": "宏碁",
    "2324.TW": "仁寶",
    "2356.TW": "英業達",
    "3231.TW": "緯創",
    "2327.TW": "國巨",
    "2395.TW": "研華",
    "3037.TW": "欣興",
    "2498.TW": "宏達電",
    "3481.TW": "群創",
    "2409.TW": "友達",
    "2881.TW": "富邦金",
    "2882.TW": "國泰金",
    "2886.TW": "兆豐金",
    "2891.TW": "中信金",
    "2892.TW": "第一金",
    "2884.TW": "玉山金",
    "2885.TW": "元大金",
    "1301.TW": "台塑",
    "1303.TW": "南亞",
    "1326.TW": "台化",
    "2002.TW": "中鋼",
    "1216.TW": "統一",
    "1101.TW": "台泥",
    "2412.TW": "中華電",
    "3045.TW": "台灣大",
    "4904.TW": "遠傳",
    "2376.TW": "技嘉",
    "2377.TW": "微星",
    "6669.TW": "緯穎",
}


def get_stock_name(ticker: str) -> str:
    return STOCK_NAMES.get(ticker, ticker.replace(".TW", ""))


def fetch_stock_data(ticker: str, days: int = HISTORY_DAYS) -> pd.DataFrame | None:
    """擷取單一股票的 OHLCV 歷史資料"""
    end = datetime.today()
    start = end - timedelta(days=days + 30)  # 多抓 30 天備用
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df.empty or len(df) < 30:
            logger.warning(f"  {ticker}: 資料不足（{len(df)} 筆）")
            return None
        # 攤平多層欄位（yfinance v0.2+ 有時會產生 MultiIndex）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df.tail(days)
    except Exception as e:
        logger.error(f"  {ticker}: 擷取失敗 — {e}")
        return None


def fetch_all_stocks(pool: list[str] = STOCK_POOL) -> dict[str, pd.DataFrame]:
    """批次擷取股票池中的所有股票資料"""
    result = {}
    total = len(pool)
    for i, ticker in enumerate(pool, 1):
        name = get_stock_name(ticker)
        print(f"  [{i:02d}/{total}] 擷取 {ticker} {name} ...", end="\r")
        df = fetch_stock_data(ticker)
        if df is not None:
            result[ticker] = df
    print(f"\n  共成功擷取 {len(result)}/{total} 檔資料")
    return result


def fetch_latest_price(ticker: str) -> float | None:
    """取得股票最新收盤價（用於績效評估）"""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None
