# The local gate. `make check` runs exactly what .github/workflows/ci.yml runs,
# in the same order, so a clean `check` means a clean CI - and, while CI cannot
# run, it is the only thing standing between a change and `main`. If you add a
# step to ci.yml, add it here too; the two drifting apart is the failure mode
# this file is written to avoid.
.PHONY: help install lint format test doctest smoke check demo clean

PY ?= python3

# Linted and formatted as one set, matching ci.yml. `scripts` belongs here:
# fetch_prices.py is the one path CI could never test, so its lint is all the
# checking it gets.
SOURCES = src tests scripts

help:
	@echo "install  安裝套件與開發依賴"
	@echo "check    完整本地把關（等同 CI）：lint + test + doctest + smoke"
	@echo "lint     ruff 靜態檢查"
	@echo "format   ruff 自動修正 + 格式化"
	@echo "test     執行 pytest"
	@echo "doctest  執行 src/ai_stock 的 doctest"
	@echo "smoke    CLI 端到端冒煙測試（寫入暫存目錄，不動到 repo）"
	@echo "demo     產生合成資料並跑完整模型比較報告"
	@echo "clean    清除快取與輸出（不會碰 data/prices 與 data/journal）"

install:
	$(PY) -m pip install -e ".[dev]"

# Invoked as `$(PY) -m ruff` rather than a bare `ruff`, so the gate always runs
# the ruff that `make install` put in this interpreter - a stale `ruff` earlier
# on PATH would otherwise silently check with a different formatter version.
lint:
	$(PY) -m ruff check $(SOURCES)
	$(PY) -m ruff format --check $(SOURCES)

format:
	$(PY) -m ruff check --fix $(SOURCES)
	$(PY) -m ruff format $(SOURCES)

test:
	$(PY) -m pytest -q

doctest:
	$(PY) -m pytest --doctest-modules src/ai_stock -q

# CI runs this against a throwaway runner and writes into the repo; locally that
# would leave synthetic CSVs next to the real ones, so everything goes to a
# temporary directory that is removed however the recipe exits.
smoke:
	@tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT; \
	set -e; \
	$(PY) -m ai_stock data --days 800 --seed 1 --out "$$tmp/ci.csv" --quiet; \
	$(PY) -m ai_stock compare --data "$$tmp/ci.csv" --models zero,momentum,ridge \
		--train-size 400 --test-size 100 --out "$$tmp/reports" --quiet; \
	$(PY) -m ai_stock simulate --data "$$tmp/ci.csv" --model ridge --paths 100 \
		--permutations 50 --train-size 400 --test-size 100 --out "$$tmp/reports" --quiet; \
	$(PY) -m ai_stock data --days 800 --seed 2 --out "$$tmp/universe/BBB.csv" --quiet; \
	cp "$$tmp/ci.csv" "$$tmp/universe/AAA.csv"; \
	$(PY) -m ai_stock screen --data "$$tmp/universe" --model ridge --permutations 20 \
		--train-size 400 --test-size 100 --out "$$tmp/reports" --quiet; \
	test -s "$$tmp/reports/comparison.md"; \
	test -s "$$tmp/reports/simulation_ridge.md"; \
	test -s "$$tmp/reports/screen_ridge.md"; \
	echo "smoke OK"

# Ordered cheapest-first so an obvious failure comes back in seconds.
check: lint test doctest smoke
	@echo "check OK - 等同 CI 的全部關卡都過了"

demo:
	$(PY) -m ai_stock data --days 2500 --seed 42 --out data/synthetic.csv
	$(PY) -m ai_stock compare --data data/synthetic.csv --models zero,momentum,reversion,ridge,logistic,random_forest,gradient_boosting --out reports
	$(PY) -m ai_stock simulate --data data/synthetic.csv --model ridge --paths 500 --out reports

# `rm -rf data` used to be here. data/ holds data/journal/forecasts.csv, which
# is the one record in this project that cannot be regenerated: every row was
# written before its outcome existed, so re-recording a lost day would use a
# model fitted on more data than the original ever saw. The paths kept below
# are exactly the two .gitignore un-ignores for the same reason.
clean:
	rm -rf .pytest_cache .ruff_cache reports
	find data -mindepth 1 -maxdepth 1 ! -name prices ! -name journal -exec rm -rf {} + 2>/dev/null || true
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
