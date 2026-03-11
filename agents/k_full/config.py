# config.py — 台灣股市預測系統設定
# K (Claude Code) 建立

# ── 台灣股票池（上市/上櫃主要個股）──────────────────────────
STOCK_POOL = [
    # 半導體/IC設計
    "2330.TW",  # 台積電
    "2303.TW",  # 聯電
    "2308.TW",  # 台達電
    "2454.TW",  # 聯發科
    "2379.TW",  # 瑞昱
    "3711.TW",  # 日月光投控
    "2344.TW",  # 華邦電
    "2408.TW",  # 南亞科
    "3034.TW",  # 聯詠
    "2388.TW",  # 威盛
    # 電子/科技
    "2317.TW",  # 鴻海
    "2382.TW",  # 廣達
    "2357.TW",  # 華碩
    "2353.TW",  # 宏碁
    "2324.TW",  # 仁寶
    "2356.TW",  # 英業達
    "3231.TW",  # 緯創
    "2327.TW",  # 國巨
    "2395.TW",  # 研華
    "3037.TW",  # 欣興
    # 面板/光電
    "2498.TW",  # 宏達電
    "3481.TW",  # 群創
    "2409.TW",  # 友達
    # 金融
    "2881.TW",  # 富邦金
    "2882.TW",  # 國泰金
    "2886.TW",  # 兆豐金
    "2891.TW",  # 中信金
    "2892.TW",  # 第一金
    "2884.TW",  # 玉山金
    "2885.TW",  # 元大金
    # 傳產/石化
    "1301.TW",  # 台塑
    "1303.TW",  # 南亞
    "1326.TW",  # 台化
    "2002.TW",  # 中鋼
    "1216.TW",  # 統一
    "1101.TW",  # 台泥
    # 電信/網路
    "2412.TW",  # 中華電
    "3045.TW",  # 台灣大
    "4904.TW",  # 遠傳
    # AI/伺服器概念
    "2376.TW",  # 技嘉
    "2377.TW",  # 微星
    "6669.TW",  # 緯穎
    "3考.TW",   # placeholder removed
]

# 清除無效項目
STOCK_POOL = [s for s in STOCK_POOL if ".TW" in s and len(s.split(".")[0]) == 4]

# ── 技術指標參數 ────────────────────────────────────────────
TA_PARAMS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "ma_short": 5,
    "ma_mid": 20,
    "ma_long": 60,
    "volume_ma": 20,
}

# ── 評分權重（總和需為 1.0） ─────────────────────────────────
SCORE_WEIGHTS = {
    "trend_score": 0.25,      # 趨勢（均線多頭）
    "momentum_score": 0.20,   # 動能（MACD）
    "rsi_score": 0.15,        # RSI 超賣反彈
    "volume_score": 0.20,     # 量能放大
    "bb_score": 0.10,         # 布林通道位置
    "ma_cross_score": 0.10,   # 均線交叉訊號
}

# ── 每日推薦數量 ────────────────────────────────────────────
TOP_N = 5

# ── 資料區間 ────────────────────────────────────────────────
HISTORY_DAYS = 90       # 抓取近 90 天歷史資料
BACKTEST_DAYS = 5       # 預測後 5 日評估績效

# ── 路徑設定 ────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for _d in [DATA_DIR, LOGS_DIR, REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)

PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")
PERFORMANCE_FILE = os.path.join(DATA_DIR, "performance.json")
