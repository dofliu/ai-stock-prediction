# -*- coding: utf-8 -*-
"""
統一股票預測系統 - 每日執行腳本
每天自動執行三位代理人的預測程式，並統一記錄結果
"""

import subprocess
import json
import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "daily_results"
RESULTS_DIR.mkdir(exist_ok=True)

def run_prediction(agent_name, script_path, cwd):
    """執行單一代理人的預測腳本"""
    print(f"\n{'='*50}")
    print(f"執行 {agent_name} 預測...")
    print('='*50)
    
    try:
        result = subprocess.run(
            ["python", script_path],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=180
        )
        
        # 嘗試讀取輸出 JSON
        output_file = Path(cwd) / "output" / "daily_prediction.json"
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'agent': agent_name,
                    'status': 'success',
                    'predictions': data.get('predictions', []),
                    'method': data.get('method', 'Unknown'),
                    'output': data
                }
        else:
            return {
                'agent': agent_name,
                'status': 'no_output',
                'error': 'No output file found'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'agent': agent_name,
            'status': 'timeout',
            'error': 'Execution timeout'
        }
    except Exception as e:
        return {
            'agent': agent_name,
            'status': 'error',
            'error': str(e)
        }

def main():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'#'*60}")
    print(f"# 統一股票預測系統 - {today}")
    print(f"# 三位代理人同時執行")
    print(f"{'#'*60}\n")
    
    results = {
        'date': today,
        'execution_time': datetime.now().isoformat(),
        'agents': {}
    }
    
    # 執行 K
    k_result = run_prediction(
        'K',
        'run_predict.py',
        BASE_DIR / 'ai-stock-k'
    )
    results['agents']['K'] = k_result
    print(f"K 完成: {k_result['status']}")
    
    # 執行 C
    c_result = run_prediction(
        'C',
        'src/daily_runner.py',
        BASE_DIR / 'ai-stock-c'
    )
    results['agents']['C'] = c_result
    print(f"C 完成: {c_result['status']}")
    
    # 執行 G
    g_result = run_prediction(
        'G',
        'stock_predictor.py',
        BASE_DIR / 'ai-stock-g'
    )
    results['agents']['G'] = g_result
    print(f"G 完成: {g_result['status']}")
    
    # 儲存結果
    result_file = RESULTS_DIR / f"predictions_{today}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果已儲存到: {result_file}")
    
    # 產生摘要
    print(f"\n{'='*60}")
    print("今日預測摘要")
    print('='*60)
    
    for agent_name, agent_data in results['agents'].items():
        if agent_data.get('status') == 'success':
            preds = agent_data.get('predictions', [])
            print(f"\n{agent_name} ({agent_data.get('method', 'N/A')}):")
            for p in preds[:5]:
                print(f"  {p.get('ticker', p.get('rank', '?'))} {p.get('name', '')}")
        else:
            print(f"\n{agent_name}: {agent_data.get('status')} - {agent_data.get('error', 'N/A')}")
    
    print(f"\n{'='*60}")
    print("Note: This is for analysis only, invest at your own risk.")
    print('='*60)

if __name__ == "__main__":
    main()
