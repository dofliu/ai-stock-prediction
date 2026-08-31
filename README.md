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

4 symbol(s) tested · noise alone would flag 0.2000 · survivors: DRAM_PUREPLAY
```

判讀順序：**先看 `excess_sharpe`**（負的就不用往下看了，你贏不過買進持有），
**再看 `q_value`**（不是 `p_value`），最後才是 IC 與準確率。

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
├── cli.py               # data / backtest / compare / simulate / screen / models
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

- 單一標的、日頻、無資金限制與融券成本；不模擬市場衝擊與流動性。
- 特徵集、模型清單與超參數是看著資料選的，這種選擇偏誤無法被回測消除。
- 合成市場的邊際是**刻意植入**的；能還原它只證明管線正確，不代表真實市場存在邊際。
- 報告中的所有數字皆為研究輸出，**不是投資建議**。

## 授權

MIT License，詳見 [LICENSE](LICENSE)。
