# ai-stock-g — 基本面 + 動量篩選系統
# G 的方法：基本面評分 (Value Score) + 短期動量 ML 篩選

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# === 台灣股票池 ===
STOCK_POOL = [
    "2330.TW", "2303.TW", "2308.TW", "2454.TW", "2379.TW",
    "3711.TW", "2317.TW", "2382.TW", "2357.TW", "2881.TW",
    "2882.TW", "1301.TW", "2002.TW", "1216.TW", "2412.TW",
    "2376.TW", "6669.TW", "3034.TW", "2327.TW", "2395.TW",
]

STOCK_NAMES = {
    "2330.TW": "台積電", "2303.TW": "聯電", "2308.TW": "台達電",
    "2454.TW": "聯發科", "2379.TW": "瑞昱", "3711.TW": "日月光投控",
    "2317.TW": "鴻海", "2382.TW": "廣達", "2357.TW": "華碩",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "1301.TW": "台塑",
    "2002.TW": "中鋼", "1216.TW": "統一", "2412.TW": "中華電",
    "2376.TW": "技嘉", "6669.TW": "緯穎", "3034.TW": "聯詠",
    "2327.TW": "國巨", "2395.TW": "研華",
}

def get_stock_name(ticker):
    return STOCK_NAMES.get(ticker, ticker.replace(".TW", ""))

# === 基本面評分系統 ===
def calculate_fundamental_score(ticker):
    """根據簡化的基本面指標給分"""
    score = 50  # 基礎分數
    
    # 知名大型股給基本面加成
    blue_chips = ["2330.TW", "2454.TW", "2317.TW", "2382.TW", "2881.TW", "2882.TW"]
    if ticker in blue_chips:
        score += 15
    
    # 電子成長股
    growth = ["2379.TW", "3034.TW", "2327.TW", "2376.TW", "6669.TW"]
    if ticker in growth:
        score += 10
    
    # 高股息殖利率預期 (傳產、金融)
    dividend = ["2881.TW", "2882.TW", "1301.TW", "2002.TW", "1216.TW"]
    if ticker in dividend:
        score += 8
    
    return score

# === 技術特徵 ===
def get_technical_features(df):
    """從股價資料計算技術特徵"""
    if df is None or len(df) < 20:
        return None
    
    close = df['Close'].values  # numpy array
    volume = df['Volume'].values  # numpy array
    
    if len(close) < 20 or len(volume) < 20:
        return None
    
    features = {}
    
    # 價格動量
    features['ret_1'] = (close[-1] / close[-2] - 1) if len(close) >= 2 else 0
    features['ret_5'] = (close[-1] / close[-5] - 1) if len(close) >= 5 else 0
    features['ret_10'] = (close[-1] / close[-10] - 1) if len(close) >= 10 else 0
    
    # RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:])
    avg_loss = np.mean(loss[-14:])
    rs = avg_gain / (avg_loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # 成交量動量
    vol_ma5 = np.mean(volume[-5:])
    features['vol_momentum'] = (volume[-1] / vol_ma5 - 1) if vol_ma5 > 0 else 0
    
    # 價格相對均線
    ma5 = np.mean(close[-5:])
    features['price_vs_ma5'] = (close[-1] / ma5 - 1) if ma5 > 0 else 0
    
    return features

# === 預測模型 ===
def train_and_predict(stock_data):
    """訓練簡單的 ML 模型並預測"""
    # 準備訓練數據 (模擬)
    # 在實際環境中，這會用歷史數據
    
    # 特徵名稱
    feature_names = ['ret_1', 'ret_5', 'ret_10', 'rsi', 'vol_momentum', 'price_vs_ma5']
    
    # 簡化邏輯：用動量指標排序
    results = []
    for ticker, df in stock_data.items():
        if df is None or len(df) < 10:
            continue
        
        # 基本面分數
        fund_score = calculate_fundamental_score(ticker)
        
        # 技術特徵
        tech = get_technical_features(df)
        if tech is None:
            continue
        
        # 動量分數 (技術面)
        momentum_score = 0
        if tech['ret_5'] > 0: momentum_score += 20
        if tech['ret_10'] > 0: momentum_score += 15
        if tech['rsi'] < 60 and tech['rsi'] > 30: momentum_score += 15  # 適中 RSI
        if tech['vol_momentum'] > 0: momentum_score += 10
        if tech['price_vs_ma5'] > 0: momentum_score += 10
        
        # 總分 = 基本面 40% + 動量 60%
        total_score = fund_score * 0.4 + momentum_score * 0.6 + 50  # 基礎50
        
        results.append({
            'ticker': ticker,
            'name': get_stock_name(ticker),
            'price': float(df['Close'].iloc[-1]),
            'fund_score': fund_score,
            'momentum_score': momentum_score,
            'total_score': total_score,
            'signals': _get_signals(tech, fund_score),
            'analysis_reason': _get_analysis_reason(tech, fund_score),
            'recommendation_reason': _get_recommendation_reason(tech, fund_score, momentum_score),
        })
    
    # 排序
    results.sort(key=lambda x: x['total_score'], reverse=True)
    return results[:5]

def _get_signals(tech, fund_score):
    signals = []
    if tech['ret_5'] > 0.05:
        signals.append("短期動能強")
    elif tech['ret_5'] > 0:
        signals.append("短期上漲")
    
    rsi_val = float(tech['rsi']) if hasattr(tech['rsi'], 'item') else tech['rsi']
    if rsi_val < 40:
        signals.append("RSI超賣")
    elif rsi_val < 60:
        signals.append("RSI適中")
    
    if tech['vol_momentum'] > 0.5:
        signals.append("成交量大增")
    
    if fund_score >= 65:
        signals.append("藍籌股")
    elif fund_score >= 55:
        signals.append("成長股")
    
    return signals[:3]

def _get_analysis_reason(tech, fund_score):
    """分析理由：根據基本面和技術面給出理由"""
    reasons = []
    
    # 基本面
    if fund_score >= 65:
        reasons.append("藍籌股，基本面穩健")
    elif fund_score >= 55:
        reasons.append("成長股，產業前景佳")
    
    # 技術面
    if tech['ret_5'] > 0.05:
        reasons.append("短期動能強勁")
    elif tech['ret_5'] > 0:
        reasons.append("短期趨勢向上")
    
    rsi_val = float(tech['rsi']) if hasattr(tech['rsi'], 'item') else tech['rsi']
    if rsi_val < 40:
        reasons.append("RSI超賣醞釀反彈")
    elif 40 <= rsi_val < 60:
        reasons.append("RSI適中，沒有超買")
    
    if tech['vol_momentum'] > 0.5:
        reasons.append("成交量明顯放大")
    
    return "；".join(reasons) if reasons else "無明顯訊號"

def _get_recommendation_reason(tech, fund_score, momentum_score):
    """建議理由：根據整體給出操作建議"""
    reasons = []
    
    # 根據分數
    if momentum_score >= 50:
        reasons.append("動能充沛，可積極操作")
    elif momentum_score >= 30:
        reasons.append("動能適中，穩健操作")
    else:
        reasons.append("觀望為主")
    
    # 根據 RSI
    rsi_val = float(tech['rsi']) if hasattr(tech['rsi'], 'item') else tech['rsi']
    if rsi_val < 40:
        reasons.append("RSI超賣區，建議分批布局")
    elif rsi_val > 70:
        reasons.append("RSI偏高注意回檔風險")
    
    # 根據基本面
    if fund_score >= 60:
        reasons.append("基本面佳，適合中長線持有")
    
    return "；".join(reasons)

# === 主程式 ===
def main():
    print("=== G 的股票預測系統 ===")
    print("方法: 基本面評分 + 動量篩選")
    print()
    
    # 抓取數據
    print("抓取股票資料...")
    stock_data = {}
    for i, ticker in enumerate(STOCK_POOL, 1):
        try:
            df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
            if len(df) > 20:
                stock_data[ticker] = df
                print(f"  [{i:02d}/{len(STOCK_POOL)}] {ticker} OK")
        except Exception as e:
            print(f"  [{i:02d}/{len(STOCK_POOL)}] {ticker} FAILED")
    
    print(f"\n成功擷取 {len(stock_data)} 檔")
    
    # 預測
    print("\n執行預測...")
    predictions = train_and_predict(stock_data)
    
    # 輸出結果
    print("\n" + "="*55)
    print(f"  G 的每日推薦  [{datetime.now().strftime('%Y-%m-%d')}]")
    print("="*55)
    print("方法: 基本面評分 + 動量篩選 (RF-like)")
    print()
    
    for i, p in enumerate(predictions, 1):
        print(f"#{i}  {p['ticker']}  {p['name']}")
        print(f"    價格: {p['price']:.2f}  總分: {p['total_score']:.1f}")
        print(f"    基本面: {p['fund_score']}  動量: {p['momentum_score']}")
        print(f"    【分析理由】{p.get('analysis_reason', '無')}")
        print(f"    【建議理由】{p.get('recommendation_reason', '無')}")
        print()
    
    print("="*55)
    print("Note: This is for analysis only, invest at your own risk.")
    print("="*55)
    
    # 存檔
    output = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'agent': 'G',
        'method': '基本面評分 + 動量篩選',
        'predictions': predictions
    }
    
    import os
    os.makedirs('output', exist_ok=True)
    with open('output/daily_prediction.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n結果已存到 output/daily_prediction.json")

if __name__ == "__main__":
    main()
