.PHONY: help install lint format test demo clean

PY ?= python3

help:
	@echo "install  安裝套件與開發依賴"
	@echo "lint     ruff 靜態檢查"
	@echo "format   ruff 自動修正 + 格式化"
	@echo "test     執行 pytest"
	@echo "demo     產生合成資料並跑完整模型比較報告"
	@echo "clean    清除快取與輸出"

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m ai_stock data --days 2500 --seed 42 --out data/synthetic.csv
	$(PY) -m ai_stock compare --data data/synthetic.csv --models zero,momentum,reversion,ridge,logistic,random_forest,gradient_boosting --out reports
	$(PY) -m ai_stock simulate --data data/synthetic.csv --model ridge --paths 500 --out reports

clean:
	rm -rf .pytest_cache .ruff_cache reports data
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
