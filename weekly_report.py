# -*- coding: utf-8 -*-
"""
股票預測系統 - 長期觀測與共識報告
每週產生一次綜合分析，檢視三位代理人的長期表現與共識
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "daily_results"

def load_all_results(days=30):
    """載入過去 N 天的所有結果"""
    results = []
    for f in RESULTS_DIR.glob("predictions_*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                results.append(data)
        except:
            pass
    return sorted(results, key=lambda x: x.get('date', ''))

def calculate_long_term_stats(results):
    """計算長期統計"""
    stats = {}
    
    for agent in ['K', 'C', 'G']:
        wins = 0
        total = 0
        all_changes = []
        
        for r in results:
            agent_data = r.get('agents', {}).get(agent, {})
            predictions = agent_data.get('predictions', [])
            
            for pred in predictions[:5]:
                ticker = pred.get('ticker', '')
                actual_change = pred.get('actual_change')
                
                if actual_change is not None:
                    total += 1
                    all_changes.append(actual_change)
                    if actual_change > 0:
                        wins += 1
        
        if total > 0:
            stats[agent] = {
                'wins': wins,
                'total': total,
                'win_rate': round(wins / total * 100, 1),
                'avg_change': round(sum(all_changes) / total, 2),
                'best': max(all_changes) if all_changes else 0,
                'worst': min(all_changes) if all_changes else 0,
            }
    
    return stats

def find_consensus(results, lookback_days=7):
    """找出共識股票"""
    ticker_counts = Counter()
    ticker_scores = {}
    
    # 只看最近幾天
    recent = [r for r in results[-lookback_days:]]
    
    for r in recent:
        for agent, agent_data in r.get('agents', {}).items():
            for pred in agent_data.get('predictions', [])[:3]:  # 只看前三名
                ticker = pred.get('ticker', '')
                if ticker:
                    ticker_counts[ticker] += 1
                    score = pred.get('score', pred.get('total_score', 0))
                    if ticker not in ticker_scores:
                        ticker_scores[ticker] = []
                    ticker_scores[ticker].append(score)
    
    # 取最多共識的
    consensus = []
    for ticker, count in ticker_counts.most_common(5):
        avg_score = sum(ticker_scores[ticker]) / len(ticker_scores[ticker])
        consensus.append({
            'ticker': ticker,
            'count': count,
            'avg_score': round(avg_score, 2)
        })
    
    return consensus

def generate_weekly_report():
    """產生每週報告"""
    results = load_all_results(30)
    
    if not results:
        print("No data available")
        return
    
    # 長期統計
    stats = calculate_long_term_stats(results)
    
    # 共識
    consensus = find_consensus(results)
    
    # 報告
    report = {
        'report_date': datetime.now().strftime('%Y-%m-%d'),
        'data_period': f"{results[0].get('date', 'N/A')} to {results[-1].get('date', 'N/A')}",
        'total_days': len(results),
        'agent_stats': stats,
        'consensus': consensus,
        'generated_at': datetime.now().isoformat()
    }
    
    # 存檔
    report_file = RESULTS_DIR / f"weekly_report_{datetime.now().strftime('%Y-%m-%d')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report

def print_report(report):
    """輸出報告"""
    print("\n" + "="*60)
    print("  股票預測系統 - 長期觀測報告")
    print("="*60)
    print(f"資料期間: {report['data_period']}")
    print(f"總天數: {report['total_days']} 天")
    
    print("\n--- 各代理人長期表現 ---")
    for agent, stat in report['agent_stats'].items():
        print(f"\n{agent}:")
        print(f"  勝率: {stat['win_rate']}% ({stat['wins']}/{stat['total']})")
        print(f"  平均漲跌幅: {stat['avg_change']:+}%")
        print(f"  最佳: {stat['best']:+}%")
        print(f"  最差: {stat['worst']:+}%")
    
    print("\n--- 共識股票 (過去7天) ---")
    for c in report['consensus']:
        print(f"  {c['ticker']}: 出現 {c['count']} 次, 平均評分 {c['avg_score']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    report = generate_weekly_report()
    if report:
        print_report(report)
