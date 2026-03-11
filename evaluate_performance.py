# -*- coding: utf-8 -*-
"""
股票預測系統 - 自我評估與優化腳本
每天檢視前一天的預測表現，決定是否要調整演算法
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "daily_results"

def get_yesterday_performance(predictions):
    """取得昨日預測的實際漲跌幅"""
    import yfinance as yf
    
    results = []
    for pred in predictions[:5]:
        ticker = pred.get('ticker', '')
        if not ticker:
            continue
        
        # 確保 ticker 格式正確
        if not ticker.endswith('.TW'):
            ticker = ticker + '.TW'
        
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d")
            if len(hist) >= 2:
                yesterday = hist['Close'].iloc[-2]
                today = hist['Close'].iloc[-1]
                change = ((today - yesterday) / yesterday) * 100
                results.append({
                    'ticker': ticker,
                    'name': pred.get('name', ''),
                    'predicted': pred.get('price', 0),
                    'actual_change': round(float(change), 2),
                    'correct': bool(change > 0)
                })
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
    
    return results

def evaluate_performance(agent_name, predictions):
    """評估單一代理人的表現"""
    if not predictions:
        return None
    
    # 取得實際漲跌幅
    actuals = get_yesterday_performance(predictions)
    
    if not actuals:
        return None
    
    # 計算指標
    winners = sum(1 for a in actuals if a['correct'])
    total = len(actuals)
    avg_change = sum(a['actual_change'] for a in actuals) / total
    
    # 最佳/最差
    best = max(actuals, key=lambda x: x['actual_change'])
    worst = min(actuals, key=lambda x: x['actual_change'])
    
    return {
        'winners': int(winners),
        'total': int(total),
        'win_rate': float(round(winners / total * 100, 1)),
        'avg_change': float(round(avg_change, 2)),
        'best': best,
        'worst': worst,
        'actuals': actuals
    }

def should_update_algorithm(performance):
    """根據表現決定是否要更新演算法"""
    if not performance:
        return {'should_update': False, 'reason': 'No performance data'}
    
    win_rate = performance['win_rate']
    avg_change = performance['avg_change']
    
    # 條件：
    # 1. 勝率低於 40% -> 應該檢視
    # 2. 平均漲跌幅為負 -> 應該檢視
    # 3. 連續兩天表現差 -> 強制更新
    # 4. 勝率超過 60% -> 可以微調
    
    if win_rate < 40:
        return {
            'should_update': True,
            'reason': f'Win rate {win_rate}% is below 40%',
            'priority': 'HIGH'
        }
    elif avg_change < 0:
        return {
            'should_update': True,
            'reason': f'Average change {avg_change}% is negative',
            'priority': 'HIGH'
        }
    elif win_rate >= 60:
        return {
            'should_update': False,
            'reason': f'Win rate {win_rate}% is good, keep current algorithm',
            'priority': 'LOW'
        }
    else:
        return {
            'should_update': False,
            'reason': f'Win rate {win_rate}% is acceptable',
            'priority': 'MEDIUM'
        }

def suggest_improvements(agent_name, performance):
    """根據表現給出改進建議"""
    if not performance:
        return ["[No data]"]
    
    suggestions = []
    
    win_rate = performance['win_rate']
    avg_change = performance['avg_change']
    worst = performance.get('worst', {})
    
    if win_rate < 40:
        suggestions.append("[WARN] Win rate too low, review scoring weights")
    
    if avg_change < 0:
        suggestions.append("[WARN] Average return is negative, add defensive indicators")
    
    # 根據最差的股票給建議
    if worst.get('actual_change', 0) < -5:
        suggestions.append(f"[NOTE] {worst.get('ticker')} dropped too much, consider excluding volatile stocks")
    
    if win_rate >= 60:
        suggestions.append("[OK] Good performance, can fine-tune parameters")
    
    return suggestions if suggestions else ["[OK] Stable performance, maintain current strategy"]

def main():
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\n{'#'*60}")
    print(f"# 股票預測系統 - 自我評估 ({today})")
    print(f"# 檢視昨日: {yesterday}")
    print(f"{'#'*60}\n")
    
    # 讀取昨日結果
    yesterday_file = RESULTS_DIR / f"predictions_{yesterday}.json"
    
    if not yesterday_file.exists():
        print(f"找不到昨日結果: {yesterday_file}")
        print("請先執行 run_all_agents.py")
        return
    
    with open(yesterday_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    evaluation = {
        'date': today,
        'evaluated_date': yesterday,
        'agents': {}
    }
    
    # 評估每位代理人
    for agent_name, agent_data in data.get('agents', {}).items():
        print(f"\n{'='*50}")
        print(f"評估 {agent_name}...")
        print('='*50)
        
        predictions = agent_data.get('predictions', [])
        
        # 評估表現
        performance = evaluate_performance(agent_name, predictions)
        
        if not performance:
            print(f"  無法評估 {agent_name}")
            continue
        
        print(f"  勝率: {performance['win_rate']}% ({performance['winners']}/{performance['total']})")
        print(f"  平均漲跌幅: {performance['avg_change']:+}%")
        print(f"  最佳: {performance['best']['ticker']} ({performance['best']['actual_change']:+}%)")
        print(f"  最差: {performance['worst']['ticker']} ({performance['worst']['actual_change']:+}%)")
        
        # 決定是否要更新
        update_decision = should_update_algorithm(performance)
        print(f"\n  更新建議: {update_decision['reason']}")
        print(f"  優先級: {update_decision.get('priority', 'N/A')}")
        
        # 改進建議
        suggestions = suggest_improvements(agent_name, performance)
        print(f"\n  改進建議:")
        for s in suggestions:
            print(f"    {s}")
        
        evaluation['agents'][agent_name] = {
            'performance': performance,
            'update_decision': update_decision,
            'suggestions': suggestions
        }
    
    # 儲存評估結果
    eval_file = RESULTS_DIR / f"evaluation_{yesterday}.json"
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"評估結果已儲存: {eval_file}")
    print('='*60)
    
    # 總結
    print(f"\n[SUMMARY]")
    for agent_name, eval_data in evaluation['agents'].items():
        decision = eval_data.get('update_decision', {})
        if decision.get('should_update'):
            print(f"  {agent_name}: NEEDS UPDATE ({decision.get('reason')})")
        else:
            print(f"  {agent_name}: KEEP CURRENT")

if __name__ == "__main__":
    main()
