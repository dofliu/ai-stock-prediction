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

### Regime-conditional 評估

`WalkForwardResult.regime_metrics()`

pooled 與逐 fold 指標都把所有時期混在一起：一個只在盤整期有效、遇到高波動就失靈的
模型，跟一個全程都穩定的模型，兩者的 `ic_fold_mean` 可能長得一模一樣，但這是完全
不同等級的宣稱。`regime_metrics()` 依**盤前已知**的落後已實現波動度（在完整價格序列
上算好、再對齊到 pooled 預測，因此不會被 fold 之間的 embargo 缺口污染）把樣本切成
三等分（terciles），逐一回報 IC 與方向準確率。判讀順序與逐 fold 指標相同：三個
regime 都為正、量級相近，才算是穩定的訊號；只有高波動 regime 撐起平均值的模型，
遇到盤整期很可能交不出東西。

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

## 6b. 篩選族群時的多重檢定

`ai_stock/evaluation/multiple_testing.py`

對 $m$ 檔各做一次 5% 門檻的檢定，即使沒有任何一檔真的有邊際，
預期仍會有 $0.05m$ 檔被標記為「顯著」。把族群排序後只報最好的那檔的原始 p 值，
是製造假邊際最常見的算術。

`screen` 因此用 **Benjamini–Hochberg** 把 p 值轉成 q 值（控制 FDR）：
將 $p_{(1)} \le \cdots \le p_{(m)}$ 排序後

$$
q_{(i)} = \min_{j \ge i} \left( \frac{m}{j} p_{(j)} \right),\qquad q_{(i)} \le 1
$$

$q = 0.10$ 的意思是：若接受所有強度到此為止的結果，其中約一成是雜訊。
選 BH 而非 Bonferroni，是因為族群篩選的情境下通常確實存在少數真實邊際，
Bonferroni 過於保守會把它們一起濾掉；報告仍同時列出 Bonferroni 門檻 $\alpha/m$ 作為對照。

檢定失敗（資料太短等）的標的其 p 值為 `NaN`，會被排除在校正之外，
不會膨脹其他標的的 q 值。

**判讀順序**：先看 `excess_sharpe`（贏不過買進持有就到此為止），再看 `q`，最後才看 IC。

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

## 7b. Deflated Sharpe：修正「試過幾個模型」

`ai_stock/evaluation/metrics.py`

PSR 把觀測到的 Sharpe 拿去對照一個**固定**門檻（預設 0）。但 `compare` 一次測好幾個
模型、`screen` 一次測好幾檔標的，選出「看起來最好」的那個，本質上與 6b 節的族群篩選
是同一個問題，只是試的是模型或視窗，而非標的。`screen` 已經用 BH 校正了跨標的的部分；
`compare` 呈報的 `deflated_sharpe`（Bailey & López de Prado, 2014）補上跨模型的部分：
把 PSR 的固定門檻換成「$N$ 次無技巧的嘗試中，最好的一次期望會有多高」：

$$
\mathbb{E}[\max\{SR_n\}] \approx \hat\sigma_{SR}\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\frac1N\right)
+ \gamma\,\Phi^{-1}\!\left(1-\frac1{Ne}\right)\right]
$$

其中 $N$ = 試過的模型數（`compare` 裡就是模型清單的長度）、
$\hat\sigma_{SR}$ = 這些模型各自（單期、未年化）Sharpe 的標準差、
$\gamma \approx 0.5772$ 為 Euler–Mascheroni 常數。把這個期望值年化後代入
`probabilistic_sharpe_ratio` 的 `benchmark_sharpe`，就得到 `deflated_sharpe_ratio`：
$N=1$（只試一個模型）時期望值為 0，DSR 退化為 PSR，因為沒有什麼好修正的。

**修正力道有多大。** 對合成市場跑 `compare --models zero,momentum,ridge,random_forest`：
`random_forest` 單獨看 PSR 高達 0.97，但把「同一次跑試了 4 個模型」算進去後
DSR 降到 0.56；`ridge` 的 PSR 0.68 更是被壓到 0.10。試的模型越多、彼此 Sharpe
差異越大，門檻就墊得越高。

**這個修正只看得到當次 `compare`。** 換一批模型、換一個視窗再跑一次、
挑看起來最好的那次結果，選擇偏誤就再乘一次——`deflated_sharpe` 不知道你跑過幾次
`compare`，如同 `q` 值不知道你篩過幾次族群（見 6b 節）。

---

## 7c. 預測日誌的有效樣本數：`hit_rate_z` 該除以幾

`ai_stock/journal.py`

`hit_rate_z` 是本專案唯一無法事後調整的數字：它把「實盤方向準確率」與「回測宣稱的
準確率」之差，除以該比例的標準誤。分子沒有爭議，**分母才是問題**——標準誤
$\sqrt{p(1-p)/n}$ 裡的 $n$，要算成幾？

日誌每個交易日、每檔標的各記一列。表面上跑三個月就累積數百列，但這些列遠不是
數百次獨立下注：

1. **視窗重疊。** $h = 5$ 時，今天的預測與昨天的預測共用五天結果中的四天。連續
   十天的預測不是十次獨立的市場押注，而是兩次押注被觀測了五遍（見第 8 節限制 4，
   `ic_fold_t` 有同樣的毛病）。
2. **橫斷面相關。** 同一天記錄的各檔標的會一起動。`config/universe.txt` 是四檔
   記憶體類股，實質上是同一個景氣循環的四種看法，不是四個獨立訊息。

`independent_blocks()` 取這個區間的**保守端**：把交易日切成連續、互不重疊的 $h$ 天
區塊，區塊內的所有預測（跨所有標的、跨所有日期）算作**一個**觀測。於是：

- `hit_rate_z_naive`：以 `n_decided`（所有有實際方向的已到期預測）為 $n$，假設
  **完全沒有**冗餘。這是本報告從前印的數字。
- `hit_rate_z`：以 `n_independent`（區塊數）為 $n$，假設區塊內**完全**冗餘。

真實的顯著性落在兩者之間，日誌本身無法斷定落在哪裡。**以保守的那個為主**：把衰退
警訊誇大，和漏掉衰退一樣有害——`z <= -2` 要能代表什麼，就不能靠重複計算同一次
行情湊出來。日誌拉長後兩者會收斂。

刻意不做的事：這裡**不估任何相關係數**。區塊數只由記錄排程決定，不由報酬決定，
所以沒有東西可以調、也沒有東西會估錯。代價是四檔真正不相關的標的會被低估成一檔，
這個代價是明說的（`screen` 的投組區段另外量測實際相關性）。

另外，`n_decided` 排除了部位為 0 的列：沒有方向的預測談不上對錯，不算一次試驗。
`hit_rate` 本來就只在有方向的列上計算，標準誤先前卻用了含 0 部位的 `n_scored`，
這使 z 值比其樣本所能支持的更大——一併修正。

**多久才夠。** 報告在 `n_independent < 10` 時拒絕給結論，只說還不夠。十個區塊
不是統計檢定力的計算結果，而是「再少下去，結論描述的就只是自己的雜訊」的界線。

---

## 8. 已知限制
## 8. 投組層級：四檔相關的記憶體股不是四個獨立賭注

`ai_stock/portfolio.py`

前面每一節都在評估**單一標的**。`screen` 把多個標的並排排名，但排名的每一列
仍然是「只做這一檔」的假設。把它們一起持有是另一個問題：把四份回測加起來，
等於默認這四檔互相獨立——而 `config/universe.txt` 裡的四檔全是記憶體族群，
由同一個循環驅動。

### 共同交易日，而不是聯集

`MU` 在紐約掛牌、`2337.TW` 等三檔在台北，兩地假日不同。相關係數只有在
**同一天的觀測值之間**才有意義，因此 `align_sleeves` 取交集而非聯集。
把休市日補 0 會憑空造出一個「安靜且不相關」的交易日，把所有相關係數往 0 拉。
交集捨棄了多少，以 `common_fraction` 明白報出來，而不是藏起來。

即使取了交集，**跨時區的同日相關仍然被低估**：同一個日期在台北與紐約並不是同一段
時間，共同的衝擊有一部分會落到對方的下一個日期上。因此 `MU` 與三檔台股之間的
相關係數（實測約 0.13–0.19，遠低於三檔台股彼此的 0.46–0.65）不能直接解讀為
「MU 與台灣記憶體股關聯不深」，有效賭注數也因此偏高。

### 分散比與有效賭注數

設各腿權重 $w_i \ge 0$（$\sum_i w_i = 1$）、年化波動 $\sigma_i$、共變異數矩陣 $\Sigma$：

$$
\mathrm{DR} = \frac{\sum_i w_i \sigma_i}{\sqrt{w^\top \Sigma\, w}},
\qquad
N_{\text{eff}} = \mathrm{DR}^2
$$

（Choueifaty & Coignard, 2008）。分子是「各腿單獨的風險加權平均」，分母是
「合起來之後真正的風險」，兩者的比值就是相關性替你省下多少。

$n$ 檔等權、等波動、兩兩相關皆為 $\rho$ 時，可直接展開：

$$
\sigma_p^2 = \frac{\sigma^2}{n^2}\bigl[n + n(n-1)\rho\bigr]
\;\Longrightarrow\;
N_{\text{eff}} = \frac{n}{1 + (n-1)\rho}
$$

$\rho = 0$ 時 $N_{\text{eff}} = n$（四檔就是四個賭注）；$\rho = 1$ 時退化為 1
（四個名字、一個賭注）。$\rho = 0.6$ 的四檔只剩 **1.43** 個賭注。
$\mathrm{DR}^2$ 是這個古典公式對「權重不等、相關矩陣任意」的推廣。

$N_{\text{eff}} > n$ 也可能發生，且不是錯誤：$\rho < 0$ 代表兩腿互相對沖，
比兩個獨立賭注更值錢。但相關性為負若只出現在樣本內，通常是小樣本雜訊，
不是下一次回檔時還在的避險。

### 「如果彼此獨立」的 Sharpe

相關性改變不了平均報酬——它對權重是線性的——只改變波動。因此把實際波動
$\sqrt{w^\top \Sigma w}$ 換成獨立情形下的 $\sqrt{\sum_i w_i^2 \sigma_i^2}$，
其餘不動，就得到 `sharpe_if_independent`。它與實際投組 Sharpe 的差距，
正是「四份回測相加」會宣稱、而這個族群並未提供的那部分分散效果。

### 不做權重最佳化

只提供**等權**與**反波動**兩種配置，兩者都不看相關矩陣。
用同一份樣本估共變異數、解出權重、再用同一份樣本評分，是製造漂亮回測最可靠的
方法之一，也正是本專案存在的目的所要避免的。風險貢獻
$\mathrm{RC}_i = w_i (\Sigma w)_i / (w^\top \Sigma w)$ 一併報出，
因為等額資金不等於等額風險。

**這一節的兩個保留。** 權重在整個樣本期間固定，腿與腿之間的再平衡不計成本；
而且相關性本身並不穩定——它會在回檔時上升，也就是分散效果最該發揮的時候失效。
報表上的有效賭注數是全樣本平均值，不是承諾。

---

## 9. 已知限制

1. **選擇偏誤**：特徵集與超參數是看著這份資料挑的。walk-forward 能防止單一 fold 的
   洩漏，但擋不住「研究者在整份資料上反覆試」造成的偏誤。
2. **成本模型過於簡化**：固定價差，未含市場衝擊、借券成本、跳空穿價。
3. **投組層級仍然很薄**：`screen` 會報出相關性、有效賭注數與風險貢獻（見 8 節），
   但權重全樣本固定、腿間再平衡不計成本、也沒有資金上限或部位上限；
   訊號本身仍是逐檔獨立產生的，沒有任何跨標的的建模。
4. **重疊訊號**：$h > 1$ 時每天以最新預測更新部位，屬慣例作法，但會使有效樣本數
   小於 bar 數；`ic_fold_t` 的自由度因此偏樂觀。預測日誌的 `hit_rate_z` 已依
   7c 節改用不重疊區塊計數，`ic_fold_t` 尚未比照辦理。
5. **合成 ≠ 真實**：能還原植入的邊際只證明管線正確，不代表真實市場存在該邊際。
6. **FDR 與 deflated Sharpe 都只涵蓋單次跑法**：`screen` 的 `q` 值涵蓋跨標的的選擇，
   `compare` 的 `deflated_sharpe` 涵蓋跨模型的選擇，但兩者都只看得到當次那一輪。
   反覆用不同模型、不同視窗、不同族群再跑一次，選擇偏誤會再乘一次，沒有任何一個
   指標會知道你總共跑了幾輪。

---

## 參考

- Bailey, D. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.*
- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality.*
- López de Prado, M. (2018). *Advances in Financial Machine Learning.*（purging / embargo）
- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate.*
- Choueifaty, Y. & Coignard, Y. (2008). *Toward Maximum Diversification.*（分散比）
- Politis, D. & Romano, J. (1994). *The Stationary Bootstrap.*
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity.*
