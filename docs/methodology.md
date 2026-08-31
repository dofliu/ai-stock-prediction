# 方法論

本文說明框架中每個關鍵設計的數學形式與理由。程式碼與本文一一對應，
行為若有變更，兩邊應同時更新。

---

## 1. 合成市場的資料生成過程

`ai_stock/data/synthetic.py`

隱含 regime 由持續性 Markov 鏈驅動，狀態 $s_t \in \{\text{bull}, \text{sideways}, \text{bear}\}$，
每日維持原狀態的機率由 `regime_persistence` 給定。

日對數報酬：

$$
r_t = \mu_t + \sigma_t \varepsilon_t,\qquad
\mu_t = \frac{d_{s_t}}{252} + \phi\, r_{t-1} + \theta \sum_{k=1}^{5} r_{t-k}
$$

其中 $\phi$ = `ar1`、$\theta$ = `reversion`。波動採 GARCH(1,1) 形式的乘子：

$$
h_t = \omega + \alpha z_{t-1}^2 h_{t-1} + \beta h_{t-1},\qquad
\sigma_t = \frac{v_{s_t}}{\sqrt{252}}\sqrt{h_t},\qquad
\omega = 1 - \alpha - \beta
$$

$\omega$ 這樣取可使 $\mathbb{E}[h_t] = 1$，因此 `regime_vols` 仍代表各 regime 的平均年化波動。
衝擊 $\varepsilon_t$ 為標準化 Student-t（`fat_tail_df`，除以 $\sqrt{\nu/(\nu-2)}$ 使變異數為 1）。

**可預測成分的大小。** 以預設值 $\phi = 0.06$、$\theta = -0.05$ 展開：

$$
\mu_{t+1} = \text{drift} + 0.06\,r_t - 0.05\sum_{k=0}^{4} r_{t-k}
= \text{drift} + 0.01\,r_t - 0.05\,(r_{t-1} + \cdots + r_{t-4})
$$

係數絕對值總和為 0.21 < 1，過程為定態。理論資訊係數
$\mathrm{corr}(\mu_t, r_t) \approx 0.12$，對應 $R^2 \approx 1.4\%$
——刻意設定在真實市場的量級，大到可偵測、小到扣完成本後不保證賺錢。

`SyntheticMarket.theoretical_information_coefficient()` 會回傳這個上限。
**若 walk-forward 量到的 IC 明顯高於它，代表管線有洩漏。**

### OHLCV 的一致性

開盤價由前收加上隔夜跳空，上下影線以半常態分布產生：

$$
H_t = \max(O_t, C_t)\,e^{|u_1|\,m\sigma_t},\qquad
L_t = \min(O_t, C_t)\,e^{-|u_2|\,m\sigma_t}
$$

如此可**保證** $L_t \le \min(O_t,C_t) \le \max(O_t,C_t) \le H_t$；四捨五入後再夾一次極值。

---

## 2. 特徵的因果性

`ai_stock/features/indicators.py`

所有指標在 bar $t$ 的值僅依賴 $\{$bar $\le t\}$，暖機期一律為 `NaN` 且**不回填**。
回填是最常見的洩漏來源之一：`fillna(method="bfill")` 會把未來的價格搬到過去。

特徵一律表示為**比值、z 分數或有界振盪量**（如 `close/sma_20 - 1`、`atr/close`），
而非原始價格。原因是原始價格非定態：模型會記住價格水準，換一檔股票就失效。

**測試方式**（`tests/test_no_lookahead.py`）：以完整序列建構特徵，再以截斷至 $t$ 的
序列重新建構，兩者在重疊區間必須逐點相同。任何用到未來資訊的實作都會使此測試失敗。

---

## 3. 標籤

$$
y_t = \log \frac{C_{t+h}}{C_t},\qquad
\text{direction}_t = \mathbb{1}[y_t > 0]
$$

$y_t$ 在時點 $t$ **不可知**，這正是預測任務的定義。最後 $h$ 根 bar 因無標籤而被丟棄。

`neutral_band` 可將 $|y_t| \le \delta$ 的樣本標記為 `NaN`（僅影響分類目標）：
這些樣本方向本質上是雜訊，訓練它們等於教模型擬合雜訊。

---

## 4. Walk-forward 與 embargo

`ai_stock/evaluation/walkforward.py`

設訓練集最後一根 bar 為 $T$。其標籤 $y_T$ 用到 $C_{T+h}$，也就是**未來 $h$ 天的價格**。
若測試集自 $T+1$ 開始，訓練標籤便已「看過」測試期的價格。因此測試集自
$T + h + 1$ 開始（`embargo` 預設等於 `horizon`）。

每個 fold 都由 factory **重新建立模型**，標準化器也在 fold 內重新配適，
確保沒有任何狀態或統計量往回洩漏。

### Pooled IC 的陷阱

把各 fold 的樣本外預測串接後計算一次相關係數，混合了兩個成分：

$$
\mathrm{Cov}(\hat{y}, y) = \underbrace{\mathbb{E}_f[\mathrm{Cov}_f(\hat{y}, y)]}_{\text{within-fold：真正的預測力}}
+ \underbrace{\mathrm{Cov}_f(\bar{\hat{y}}_f, \bar{y}_f)}_{\text{between-fold：regime 均值差異}}
$$

在 regime 切換的市場中，第二項可能為大幅負值（模型從多頭訓練期學到正漂移，
接著遇上空頭），足以蓋過第一項並翻轉符號。實測中曾出現 61% 的 fold IC 為正、
中位數 +0.023，但 pooled IC 為 −0.06。

因此 `metrics()` 同時回報：

- `ic_pearson`：pooled（保留但不作為判準）
- `ic_fold_mean`、`ic_fold_std`
- `ic_fold_t` $= \dfrac{\overline{\mathrm{IC}}}{s_{\mathrm{IC}}}\sqrt{n_{\text{folds}}}$
- `ic_fold_positive_rate`

判讀以逐 fold 指標為準，慣例門檻約 $|t| > 2$。

---

## 5. 回測的時序約定

`ai_stock/backtest/engine.py`

$$
p_{t+1} = \text{size}(\hat{y}_t),\qquad
R_{t+1} = p_{t+1} r^{\text{simple}}_{t+1} - |p_{t+1} - p_t| \cdot c
$$

$\hat{y}_t$ 由 $t$ 收盤的資訊產生，因此只能持有 $t+1$ 的部位——程式碼中就是一個
`shift(1)`。少了它，回測會用當根 bar 的結果去交易同一根 bar，任何模型都會顯得神準。

成本 $c$ = (`cost_bps` + `slippage_bps`) × 1e-4，按**成交名目金額**計費，
所以每天翻轉方向的訊號會被如實懲罰。權益以簡單報酬複利：
$E_t = E_{t-1}(1 + R_t)$。buy & hold 基準同樣支付一次進場成本，比較才對等。

### 部位規則

| `sizing` | 規則 |
|---|---|
| `sign` | $\text{sign}(\hat{y})\cdot L$，且 $\lvert\hat{y}\rvert \le$ `threshold` 時為 0 |
| `long_only` | $\hat{y} >$ `threshold` 時為 $L$，否則 0 |
| `proportional` | $\mathrm{clip}(\hat{y}\cdot\text{scale},\ -L,\ L)$ |

`vol_target` 啟用時，部位再乘上 $\sigma_{\text{target}} / \hat\sigma_t$，
其中 $\hat\sigma_t$ 為**過去** `vol_lookback` 天的年化波動（因果），暖機期視為未知而空手。

---

## 6. 顯著性檢定：運氣還是技巧

`ai_stock/simulation/monte_carlo.py`

虛無假設：訊號與未來報酬之間**沒有時序關聯**。作法是重排訊號後重跑同一套回測，
得到統計量的虛無分布。

- **`rotation`（預設）**：循環位移 $\tilde{s} = \text{roll}(s, k)$，$k \sim U\{1, n-1\}$。
  保留訊號的自相關，因而保留換手率與成本，只破壞對齊。
- **`shuffle`**：隨機排列。摧毀自相關 → 虛無策略每天亂翻部位 → 成本暴增 → 門檻被壓低。

實測差異（同一策略，Sharpe 0.565）：

| 虛無方法 | 虛無均值 | 虛無 95 百分位 | p 值 |
|---|---|---|---|
| `rotation` | −0.10 | **0.40** | 0.020 |
| `shuffle` | −0.21 | 0.13 | 0.003 |

`shuffle` 讓 p 值好看了近一個數量級，純粹是因為它的對照組付了更多成本。

p 值採單尾並加上標準修正，確保恆為正：

$$
p = \frac{1 + \#\{b : \text{null}_b \ge \text{observed}\}}{1 + B}
$$

---

## 7. Bootstrap 與 Probabilistic Sharpe

**區塊 bootstrap**（預設 `block_size=20`）以環狀方式抽取連續區塊，
保留波動叢聚等短程相依；iid bootstrap 會摧毀它，因而低估風險。

**Probabilistic Sharpe Ratio**（Bailey & López de Prado, 2012）修正樣本長度、
偏態與峰態：

$$
\widehat{\mathrm{PSR}} = \Phi\!\left(
\frac{(\hat{SR} - SR^*)\sqrt{n-1}}
{\sqrt{1 - \gamma_3 \hat{SR} + \frac{\gamma_4 - 1}{4}\hat{SR}^2}}
\right)
$$

負偏態、厚尾、樣本短的績效，即使 Sharpe 好看，PSR 也上不去——這正是它的用途。

---

## 8. 已知限制

1. **選擇偏誤**：特徵集與超參數是看著這份資料挑的。walk-forward 能防止單一 fold 的
   洩漏，但擋不住「研究者在整份資料上反覆試」造成的偏誤。
2. **成本模型過於簡化**：固定價差，未含市場衝擊、借券成本、跳空穿價。
3. **單一標的**：無投組配置、無相關性結構、無資金上限。
4. **重疊訊號**：$h > 1$ 時每天以最新預測更新部位，屬慣例作法，但會使有效樣本數
   小於 bar 數；`ic_fold_t` 的自由度因此偏樂觀。
5. **合成 ≠ 真實**：能還原植入的邊際只證明管線正確，不代表真實市場存在該邊際。

---

## 參考

- Bailey, D. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.*
- López de Prado, M. (2018). *Advances in Financial Machine Learning.*（purging / embargo）
- Politis, D. & Romano, J. (1994). *The Stationary Bootstrap.*
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity.*
