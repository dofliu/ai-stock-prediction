# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from data_fetcher import fetch_all_stocks
from analyzer import analyze_all
from predictor import select_top
from datetime import datetime

print('=== Fetching stock data...')
data = fetch_all_stocks()
print(f'Success: {len(data)} stocks')

print('\n=== Analyzing...')
results = analyze_all(data)
print(f'Analyzed: {len(results)} stocks')

print('\n=== Selecting top 5...')
recs = select_top(results, 5)

print('\n' + '='*55)
print(f'  Taiwan Stock Picks  [{datetime.now().strftime("%Y-%m-%d")}]')
print('='*55)

for rec in recs:
    print(f"\n#{rec['rank']}  {rec['ticker']}  {rec['name']}")
    print(f"    Price: {rec['close']:.2f}  Score: {rec['score']:.3f}")
    signals_str = ', '.join(rec['signals'])
    print(f"    Signals: {signals_str}")

print('\n' + '='*55)
print('  Note: This is for analysis only, invest at your own risk.')
print('='*55)
