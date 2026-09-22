# AI Stock Prediction — 股價預測與策略模擬框架

一套用來**誠實地**做股價預測研究的 Python 框架：從合成市場、特徵工程、walk-forward
驗證、含成本的回測，一路到蒙地卡羅模擬與「運氣 vs 技巧」的統計檢定。

大多數股價預測專案的問題不在模型不夠新，而在於**量測方式讓自己看起來很厲害**：
用到未來資訊的特徵、單一次切分的驗證、沒有交易成本的回測、以及沒有對照組的績效數字。
本專案把這些陷阱一個一個拆開處理，並且提供一個**已知答案的合成市場**來驗證整條管線
——因為在真實股價上，你永遠分不清模型是找到了訊號，還是只是過度擬合了雜訊。

> ⚠️ 本專案為教學與研究工具，**不構成任何投資建議**。

---

## 核心設計

| 常見陷阱 | 本專案的處理方式 |
|---|---|
| 特徵用到未來資料 | 所有指標皆為**因果**計算，暖機期保留 `NaN` 不回填；並有測試比對「截斷序列」與「完整序列」的特徵值必須逐點相同 |
| 標籤與訓練期重疊 | walk-forward 在訓練與測試之間插入 **embargo**（預設等於預測期距 `horizon`） |
| 訊號當天就成交 | 回測嚴格套用 **一根 bar 的延遲**：`t` 收盤產生的訊號，只能吃 `t+1` 的報酬 |
| 沒有交易成本 | 手續費與滑價依**成交名目金額**計費，報告同時列出年換手率與成本拖累 |
| 沒有對照組 | 每次比較都包含 `zero`、`train_mean`、`momentum`、`reversion` 等基準與 buy & hold |
| 單一數字當結論 | 提供逐 fold 指標、bootstrap 信賴區間，以及排列檢定的 p 值 |
| 分不清運氣與技巧 | **循環位移（rotation）虛無假設**：保留訊號自相關與換手率，只破壞時序對齊 |
| 篩一整個族群後只報最好的那檔 | **Benjamini–Hochberg FDR 校正**：`screen` 同時輸出原始 p 與跨標的校正後的 q 值 |

### 兩個容易被忽略的細節

**1. Pooled IC 會說謊。** 把各 fold 的預測串接起來算一次相關係數，會混入不同波動
regime 之間的均值差異；實測中曾出現「六成 fold 的 IC 為正，串接後的 IC 卻是負的」
——典型的辛普森悖論。因此本框架同時輸出 `ic_fold_mean`、`ic_fold_t`
（跨 fold 的 t 統計量），並以**逐 fold 指標為準**。

**2. 篩越多檔，越容易撞到假訊號。** 同時測 10 檔、每檔用 5% 門檻，就算沒有任何一檔真的有邊際，
平均也會有 0.5 檔「顯著」。`screen` 因此把原始 p 值與 BH 校正後的 q 值並列，並直接寫出
「純雜訊預期會標記幾檔」。實測中曾出現某檔原始 p = 0.0498（剛好低於 0.05），
校正後 q = 0.0997——同時篩四檔就是這個代價。

**3. 洗牌（shuffle）的虛無假設對策略太仁慈。** 隨機打亂訊號會摧毀它的自相關，
使虛無策略的換手率暴增、成本大增，於是門檻被壓低。改用**循環位移**保留換手率，
只破壞「訊號與未來報酬的對齊」，才是公平的對照。實測中 rotation 虛無分布的
95 百分位是 0.40，而 shuffle 只有 0.13——差距就是這個偏誤的大小。

---

## 安裝

```bash
git clone https://github.com/dofliu/ai-stock-prediction.git
cd ai-stock-prediction
pip install -e ".[dev]"
```

只需要 `numpy`、`pandas`、`scikit-learn`；圖表以 ASCII 繪製，不需要 matplotlib。

### 送出修改前：`make check`

```bash
make check    # lint + test + doctest + smoke，約 40 秒
```

這一條跑的是 `.github/workflows/ci.yml` 的完整內容、同樣順序。**CI 目前因為
Actions 額度用盡而無法執行**（最後一次綠燈 2026-09-13），所以在額度恢復之前，
這是一個修改與 `main` 之間唯一的關卡——請在 push 前跑過，並把結果寫進 PR。
同一個額度問題也讓每日行情 workflow 停擺，日誌自 2026-09-11 起沒有新資料；
狀態與代價記在 [roadmap 的 Blocked 一節](docs/roadmap.md)。

| 關卡 | 內容 |
|---|---|
| `make lint` | `ruff check` + `ruff format --check`，涵蓋 `src tests scripts` |
| `make test` | `pytest -q` |
| `make doctest` | `pytest --doctest-modules src/ai_stock`（`make test` **不會**跑到這些） |
| `make smoke` | CLI 端到端：`data → compare → simulate → screen`，寫入暫存目錄 |

> `make clean` 不會刪除 `data/prices/` 與 `data/journal/`。
> 日誌的每一列都在結果出現前就寫下了，重跑一次會用到更多資料配適的模型——
> 那正是本專案拒絕的事後調整。被刪掉的當日預測救不回來。

---

## 快速開始

```bash
# 1. 產生一段有「已知邊際」的合成市場
ai-stock data --days 3000 --seed 42 --out data/synthetic.csv

# 2. 對單一模型做 walk-forward 驗證與回測
ai-stock backtest --data data/synthetic.csv --model ridge --out reports

# 3. 在相同 fold 上比較所有模型
ai-stock compare --data data/synthetic.csv --out reports

# 4. 蒙地卡羅模擬 + 顯著性檢定
ai-stock simulate --data data/synthetic.csv --model ridge --paths 1000 --out reports

# 5. 一次篩選一整個族群，並校正多重檢定
ai-stock screen --tickers MU,2408.TW,2344.TW,2337.TW --model random_forest \
    --horizon 5 --permutations 300 --cache-dir data/cache --out reports

# 6. 記錄今天的預測，並把已到期的計分
ai-stock journal --data data/prices --model random_forest --horizon 5 \
    --journal data/journal/forecasts.csv --out reports
```

或直接用 `make demo` 一次跑完。若未安裝套件，`python -m ai_stock ...` 等價於 `ai-stock ...`。

### 族群篩選輸出範例

```
| metric               | DRAM_PUREPLAY | DRAM_MAJOR | NOR_FLASH | NICHE_MEM |
|----------------------|---------------|------------|-----------|-----------|
| ic_fold_t            | 5.5899        | 1.8553     | 1.4483    | 0.8279    |
| sharpe               | 0.8748        | 0.7547     | 0.2568    | 0.5671    |
| benchmark_sharpe     | 0.4998        | 0.6619     | 0.4242    | 0.9853    |
| excess_sharpe        | 0.3750        | 0.0929     | -0.1674   | -0.4182   |
| p_value              | 0.0100        | 0.0498     | 0.2525    | 0.4020    |
| q_value              | 0.0399        | 0.0997     | 0.3367    | 0.4020    |

held together   4 sleeves, equally weighted, correlated 0.05 -> 3.51 effective bet(s)
the shares      correlated 0.36 -> 1.94 if you just held them
portfolio       Sharpe 0.8030 vs 0.8566 if the sleeves were independent

4 symbol(s) tested · noise alone would flag 0.2000 · survivors: DRAM_PUREPLAY
```

判讀順序：**先看 `excess_sharpe`**（負的就不用往下看了，你贏不過買進持有），
**再看 `q_value`**（不是 `p_value`），最後才是 IC 與準確率。

> 上面那張表是**合成**族群，用來示範輸出格式與「有存活者」時的樣子。
> 真實族群的答案在下一節，而且不一樣。

### 真實族群上的答案：沒有

同一條指令，換成 `config/universe.txt` 裡四檔真實的記憶體股，
各 3,664 根日線（2011 起至 2026-09-11），`random_forest`、`horizon=5`、
成本 2 bps + 滑價 3 bps、200 次 rotation 排列：

```
| metric               | 2408.TW  | 2337.TW | MU      | 2344.TW |
|----------------------|----------|---------|---------|---------|
| ic_fold_mean         | -0.0390  | 0.0739  | 0.0462  | -0.0320 |
| ic_fold_t            | -1.0102  | 2.0598  | 1.4919  | -1.0918 |
| directional_accuracy | 50.79%   | 52.80%  | 51.91%  | 47.04%  |
| sharpe               | 0.4446   | 0.3040  | 0.3694  | -0.0690 |
| benchmark_sharpe     | 0.7376   | 0.6850  | 0.8166  | 0.8538  |
| excess_sharpe        | -0.2930  | -0.3810 | -0.4473 | -0.9228 |
| annual_turnover      | 104.6344 | 64.3830 | 71.5262 | 90.3236 |
| p_value              | 0.1343   | 0.1194  | 0.4378  | 0.8010  |
| q_value              | 0.2687   | 0.2687  | 0.5837  | 0.8010  |

held together   4 sleeves, equally weighted, correlated 0.08 -> 3.21 effective bet(s)
the shares      correlated 0.36 -> 1.94 if you just held them
portfolio       Sharpe 0.4168 vs 0.4645 if the sleeves were independent

4 symbol(s) tested · noise alone would flag 0.2000 · no symbol both beat buy & hold and survived q <= 0.05
```

**四檔的 `excess_sharpe` 全是負的**，照判讀順序第一步就該停下來：四檔都有正的
Sharpe（除了 2344.TW），但四檔都輸給買進持有。依照判讀順序，後面的 `q` 值
其實不必看了——看了也沒有一檔低於 0.05。

這一節存在的理由，不是因為結果好看，而是因為它是本專案唯一有意義的**整體驗證**：
同一條管線，在植入邊際的合成市場上還原出 Sharpe 0.56 / p = 0.020，
在效率市場上交出 −0.05 / p = 0.478，在真實的四檔記憶體股上交出「沒有」。
一個在真實日線上找得到邊際的框架，比較可能是壞掉了，而不是聰明。

兩個不該過度解讀的地方。`2337.TW` 的 `ic_fold_t` = 2.06 看起來像個訊號，但它的
`excess_sharpe` 是 −0.38：**預測方向的能力，和扣掉成本後贏過買進持有，是兩件事**，
中間隔著 64 倍的年換手率。另外，腿與腿的相關 0.08 遠低於股票本身的 0.36，
有效賭注數因此是 3.21——但在沒有任何一腿證明有邊際的前提下，這只是四個互相亂猜的
模型剛好站在不同邊，不是分散。

### 四檔一起做，等於幾個賭注？

排名表的每一列都假設「只做這一檔」。一起持有是另一個問題，報告的
「Held together」一節會回答：相關矩陣、分散比、**有效賭注數**
（$\mathrm{DR}^2$，等權等波動下即 $n/(1+(n-1)\rho)$）、各腿的風險貢獻，
以及「如果四腿彼此獨立，Sharpe 會是多少」。

兩張表都出現同一件事：**策略腿之間的相關（合成 0.05、真實 0.08）遠低於股票本身
的 0.36**。這不自動是好消息：腿與腿會因為模型剛好站在不同邊而去相關，而**互相亂猜的模型
看起來就長這樣**。要先確認每一腿真的有邊際（看 `vs B&H` 與 `q`），
去相關才算得上分散。詳見 [方法論 8 節](docs/methodology.md)。

### 輸出範例

```
model            ridge
out-of-sample    2199 bars over 18 folds
fold IC          0.0832 (t = 2.9685)
Sharpe           0.5646 vs buy & hold 0.0612
annual return    10.46%
max drawdown     -36.80%
annual turnover  103.9400x
```

---

## 每日預測日誌：唯一無法事後調整的分數

回測是拿模型去對「它被配適在附近」的歷史打分。預測日誌做的是更難的事：
**在結果還不存在時就把預測寫進 CSV，等期距真的走完才計分。**
這一列在答案出現前就已經落地，任何事後的參數調整都改不了它。

```bash
ai-stock journal --data data/prices --journal data/journal/forecasts.csv --out reports
```

每天跑一次，它會：

1. 用目前所有已標記的 bar 配適模型（必然止於最後一根的前 `horizon` 根），對最新收盤發出預測並附加到日誌；
2. 掃描日誌，只有「`asof_date + horizon` 那根 bar 已經存在」的列才計分；
3. 比對 live 命中率與回測宣稱值，並用 **z 值**（差距 ÷ 自身標準誤）說明兩者是否一致。

這是本 repo 的日誌**實際**在 2026-09-22 跑出來的樣子，不是示意：

```
data as of         2026-09-11  (STALE: 2337.TW, 2344.TW, 2408.TW, MU - check the downloader)
scored / pending   15 / 16
live hit rate      80.00%
live IC            0.6639
total P&L          45.44%
vs backtest        claim 50.64% -> live 80.00%  (z = 0.5873 over 1 independent horizon(s); naive z = 2.2747)
```

判讀：**先看 `n_scored`**。少於 30 筆時 z 值不管發生什麼都接近 0，報告會直接說
「太少，什麼都不能講」。z ≤ −2 才是衰減訊號——回測承諾了 live 交不出來的東西。

上面這份輸出正好示範了為什麼要先看 `n_scored`：80% 的命中率、0.66 的 IC、45% 的
P&L，三個數字都很漂亮，而它們全部來自 **1 個獨立期距**。naive z = 2.27 會說
「顯著」，把重疊的期距與相關的標的各算一次；改用不重疊區塊之後是 0.59，也就是
**什麼都還沒證明**。兩者之間的差距不會隨時間自己消失，只會隨著新的交易日一天一天
被填上——這正是 `data as of` 停在 2026-09-11 的代價，見 [roadmap 的 Blocked 一節](docs/roadmap.md)。

> **再看 `data as of`。** 行情源停掉時，這份報告不會變安靜，它會有自信地重複：
> 同一批預測對同一批 bar 到期，命中率一字不差地再報一次。任何標的落後超過
> 四個日曆天，這一行會標成 `STALE` 並點名，報告也會在所有績效數字之前說明。
> 判斷標準是日曆天而非交易日，所以長假會被誤報為落後——這是刻意選的方向：
> 多看一眼下載器幾乎沒有成本，一個悄悄停止更新的命中率則毀掉唯一無法事後
> 調整的數字。細節見 `docs/methodology.md` 7d 節。

> `live_ic` 取各標的 IC 的平均，而非把所有標的丟進同一個相關係數
> （後者列為 `live_ic_pooled` 僅供對照）。理由與 `ic_fold_mean` 相同：
> 跨群體匯總會讓 between-group 的均值差異蓋過訊號，甚至翻轉符號。

### 自動化：GitHub Actions 每天抓真實行情

`.github/workflows/daily-prices.yml` 每個交易日 22:00 UTC（美股收盤後，
同日台股也已收盤）用 runner 的網路抓 `config/universe.txt` 列出的標的，
記錄預測、計分到期項，然後把 `data/prices/` 與 `data/journal/` commit 回 repo。
單一 ticker 失敗不影響其他標的；全部失敗才視為異常並讓 job 失敗。

抓價是唯一無法離線測試的路徑，所以只要動到相關檔案，同一個 workflow
就會在 PR 上以 dry run 執行（真的抓、但不 commit）。

### Actions 跑不動時：在本機跑同一個迴圈

```bash
python3 scripts/daily_update.py             # 抓 → 記錄計分 → commit → push
python3 scripts/daily_update.py --dry-run   # 前兩步照跑，不 commit 也不 push
```

workflow 停擺時日誌就**停止累積**，而這不只是閒置：`hit_rate_z` 需要的是獨立
期距，而獨立期距只能一個交易日一個交易日地到。這支腳本就是那個 workflow，
少了 runner，並保留它那個重要的順序——

```
抓價 → 記錄與計分 → commit → 然後才回報下載失敗
```

局部斷線不該賠掉已經抓到的 bar，所以資料先進 commit，這一輪才允許變紅。

它另外加了 workflow 不需要、但筆電需要的三道保險。runner 從乾淨的 `main`
checkout 開始，工作目錄不是：

| 風險 | 處理 |
|---|---|
| 在錯的分支上 commit 當日資料 | 不在 `--branch`（預設 `main`）上就直接拒絕執行 |
| 把暫存中的半成品一起推上 main | 用 `git commit --only`，其餘 staged 內容原封不動留著 |
| 日誌被改寫 | commit 前比對 HEAD 版本，只要有任何一列變動就中止 |

第三道是重點。日誌的每一列都在結果出現前就寫下了，**變動過的列就是用後見之明
重算過的列**，而且事後從檔案上看不出來。這道檢查在第一次對真實資料執行時就
攔下了一個真的缺陷（見 `load_journal` 的 `float_precision`）。

---

## 框架驗證：它會不會「無中生有」？

這是任何預測框架都該回答的問題。合成市場的可預測成分由 `--ar1` 與 `--reversion`
控制，把兩者設為 0 就得到一個**效率市場**——理論上任何模型都不該賺到錢。

```bash
ai-stock simulate --model ridge --days 3000                # 有植入邊際
ai-stock simulate --model ridge --days 3000 --efficient    # 無邊際（純雜訊）
```

| 市場 | 實現 Sharpe | 排列檢定 p 值 | 結論 |
|---|---|---|---|
| 植入邊際（`ar1=0.06, reversion=-0.05`） | **0.56** | **0.020** | 成功還原已知訊號 |
| 效率市場（`--efficient`） | −0.05 | 0.478 | 未產生偽陽性 |

此外，用 OLS 對合成資料回歸可還原出資料生成過程的真實係數
（隱含落後係數 `r_{t-3..t-5} ≈ −0.05`，與設定值一致），
且樣本內 IC 0.11 ≈ 理論上限 0.12——特徵與標籤的對齊沒有偏移。

---

## 專案結構

```
src/ai_stock/
├── config.py            # 所有階段共用的 frozen dataclass 設定
├── pipeline.py          # 端到端流程（CLI 只是它的薄殼）
├── cli.py               # data / backtest / compare / simulate / screen / journal / models
├── journal.py           # 預測日誌：事前記錄、到期計分、live vs 回測
├── portfolio.py         # 合併多檔：相關性、分散比、有效賭注數、風險貢獻
├── data/
│   ├── synthetic.py     # regime 切換 + GARCH 波動叢聚 + 厚尾 + 已知邊際
│   └── loaders.py       # CSV 載入、OHLCV 驗證、yfinance（選用）
├── features/
│   ├── indicators.py    # SMA/EMA/RSI/MACD/BB/ATR/Stochastic/OBV/Donchian
│   └── builder.py       # 平穩化特徵矩陣 + 前瞻標籤
├── models/
│   ├── baselines.py     # zero / train_mean / momentum / reversion
│   ├── sklearn_models.py# ridge / elastic_net / logistic / RF / GBM / MLP
│   └── registry.py      # 以名稱建立模型，可註冊自訂模型
├── evaluation/
│   ├── walkforward.py   # 帶 embargo 的滾動/擴張視窗
│   ├── multiple_testing.py # BH FDR、Bonferroni、預期偽陽性數
│   └── metrics.py       # 迴歸/分類/財務指標、Probabilistic Sharpe
├── backtest/engine.py   # 訊號→部位→權益曲線（含成本、波動目標）
├── simulation/          # 路徑模擬、bootstrap 區間、排列檢定
└── reporting/           # Markdown 報告與 ASCII 圖表
```

---

## Python API

```python
from ai_stock import ExperimentConfig, WalkForwardConfig, BacktestConfig
from ai_stock.data import generate_ohlcv
from ai_stock.pipeline import run_model, compare_models

ohlcv = generate_ohlcv(n_days=3000, seed=42)

config = ExperimentConfig(
    walk_forward=WalkForwardConfig(train_size=750, test_size=125, expanding=True),
    backtest=BacktestConfig(sizing="sign", cost_bps=1.0, slippage_bps=1.0),
)

run = run_model(ohlcv, "ridge", config)
print(run.walk_forward.metrics()["ic_fold_t"])   # 跨 fold 的 IC t 值
print(run.backtest.summary())                    # 策略 vs buy & hold

runs = compare_models(ohlcv, ["zero", "momentum", "ridge", "random_forest"], config)
```

### 註冊自訂模型

```python
from ai_stock.models import Model, register_model

class MyModel(Model):
    name = "my_model"

    def fit(self, features, target):
        ...
        return self

    def predict(self, features):
        ...  # 回傳每列的訊號（預期報酬）

register_model("my_model", MyModel)
```

---

## 使用真實資料

CSV 只需含 `date, open, high, low, close, volume`（欄位名稱不分大小寫，
`Adj Close` 等別名會自動對應）：

```bash
ai-stock compare --data path/to/2330.csv --out reports
```

亦可透過選用的 `yfinance` 下載：

```python
from ai_stock.data import load_yfinance
ohlcv = load_yfinance("2330.TW", period="10y")   # pip install yfinance
```

**請預期真實資料上的結果遠比合成資料難看。** 日頻價格的可預測成分極小，
扣掉成本後往往所剩無幾——這正是框架要如實呈現的事。

---

## 限制

- 訊號逐檔獨立產生；`screen` 會報出投組層級的相關性與有效賭注數，但權重全樣本固定、
  腿間再平衡不計成本，也沒有資金限制與融券成本；不模擬市場衝擊與流動性。
- 日頻；跨時區標的以日期對齊，同日相關因此被低估。
- 特徵集、模型清單與超參數是看著資料選的，這種選擇偏誤無法被回測消除。
- 合成市場的邊際是**刻意植入**的；能還原它只證明管線正確，不代表真實市場存在邊際。
- 報告中的所有數字皆為研究輸出，**不是投資建議**。

## 授權

MIT License，詳見 [LICENSE](LICENSE)。
