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
- `ic_fold_n_eff`、`ic_fold_t` $= \dfrac{\overline{\mathrm{IC}}}{s_{\mathrm{IC}}}\sqrt{n_{\text{eff}}}$
- `ic_fold_t_naive` $= \dfrac{\overline{\mathrm{IC}}}{s_{\mathrm{IC}}}\sqrt{n_{\text{folds}}}$（舊值，保留作對照）
- `ic_fold_positive_rate`

判讀以逐 fold 指標為準，慣例門檻約 $|t| > 2$。分母的 $n_{\text{eff}}$ 見下一節。

### `ic_fold_t` 該除以幾：`independent_folds()`

`WalkForwardResult.independent_folds()`

分母若用 fold 數，等於宣稱「每個 fold 都是一次獨立的市場觀測」。排程本身有兩個地方
不成立，而且兩者都只由**排程**決定，與報酬無關：

1. **測試視窗重疊。** `step` 預設等於 `test_size`，此時視窗恰好首尾相接；一旦把
   `step` 調小，視窗就真的重疊——`step=1` 時二十個 fold 幾乎在評分同一段行情，是
   一次觀測被報告了二十遍。`run_walk_forward()` 早就針對 pooled 樣本數發出警告，
   但逐 fold 的 t 統計量有同樣的問題，而它先前沒說。
2. **標籤尾巴。** $h$ 日目標讓測試視窗最後一根 bar 的結果落在 $h$ 根之後，也就是
   落進下一個 fold 的視窗裡。即使首尾相接的 fold，也共用 $h-1$ 根結果 bar——這正是
   限制 4 的重疊訊號問題，只是發生在 fold 邊界而非每一根 bar。

每個 fold 貢獻半開區間 $[\text{test\_start},\ \text{test\_end} + h)$，
$n_{\text{eff}}$ 取這些區間**聯集的長度**除以**平均寬度**：這些 fold 之間究竟走過了
幾個 fold 寬度的不同行情。這是 7c 節 `independent_blocks()` 在「視窗比記錄間隔寬」
情況下的推廣——那裡每天記一列、每列覆蓋 $h$ 天，得 $n/h$ 個區塊；這裡每 `step` 根
出一個 fold、每個 fold 覆蓋 `width` 根，得 $n \cdot \text{step} / \text{width}$ 個。

界限由構造保證：聯集不可能大於各區間寬度之和，所以 $n_{\text{eff}} \le n_{\text{folds}}$；
聯集至少和最寬的區間一樣寬，所以 $n_{\text{eff}} \ge 1$。

**預設排程下這個修正很小，而且本來就該很小。** `horizon=1` 時首尾相接的 fold 真的
互相獨立，$n_{\text{eff}}$ 恰等於 fold 數——沒有冗餘就不該憑空捏造冗餘，低估邊際和
高估邊際一樣不誠實。$h=5$、`test_size=100` 的五個 fold 則是 4.83 而非 5，t 從 2.53
降到 2.49。真正咬人的是 `step` 被調小的時候。

一個明說的保守偏誤：fold 位置取自 pooled 預測日曆，其中不含測試視窗之間的空隙，
所以 `step > test_size` 時彼此分得很開的 fold 會被讀成僅僅相鄰。方向與 7c 節一致——
寧可低估。

### Fold 內選超參數：`TuningConfig`

`ai_stock/evaluation/tuning.py`

超參數若「先在整份資料上挑一次，再拿去跑 walk-forward」，報出來的數字是真實運作
中不可能產生的：真的要上線，只能用**過去**的資料把超參數挑出來。`run_walk_forward`
接受 `tuning=TuningConfig(grid=...)` 後，每個外層 fold 會**只用自己的訓練 bar**
再跑一次內層 walk-forward，把贏家挑出來，再用它配適整個訓練窗、預測測試窗。

內層排程刻意複用同一個 `WalkForwardSplitter`：訓練窗末端的
`inner_fraction`（預設 0.3）切成 `n_inner_folds`（預設 3）個驗證窗，內層 train 與
內層 validation 之間套用與外層相同的 embargo。因此「不看未來」這件事走的是同一段
已測試的程式碼，而不是另寫一份。`tests/test_tuning.py::
test_selection_reads_nothing_outside_the_training_window` 直接釘住這個性質：把外層
訓練窗**之後**的特徵全部打亂，每個 fold 選出的超參數必須一字不變。

三個值得記下來的選擇：

- **選擇指標不是 Sharpe。** Sharpe 同時吃成本模型、部位規則與槓桿上限，拿它來選
  超參數，等於允許某個候選靠「剛好配合這組交易設定」獲勝，而交易設定正是下游使用者
  最常改的東西。預設用 `ic_pearson`，只問預測力。
- **內層是 walk-forward，不是單一 hold-out。** 單一 hold-out 等於在某一段市況上挑
  超參數，正是本專案在別處花力氣量測的那種脆弱性。
- **窗太短就不選。** `inner_schedule` 的算術會隨窗長等比縮放，永遠不會自己撞到界線，
  所以 `min_inner_train`（預設 60 根）是明白寫死的判斷：低於它，怎麼配適都是雜訊，
  此時退回模型預設值，並在報告中說「沒有選」，而不是假裝選過。

**先看穩定性，再看績效。** `selection_stability()` 回報每個超參數被選中的相異值個數、
眾數及其佔比，以及 `selection_spread`（每個 fold 中最佳與最差候選的內層分數差，
再跨 fold 平均）。若贏家幾乎每個 fold 都在換，代表這些候選在現有樣本下根本分不出
高下，誠實的說法不是「這組超參數有效」，而是「這套**流程**（含它的不穩定）賺這麼多」；
若 `selection_spread` 趨近於零，則是從另一個方向講同一件事——網格太平，穩定的贏家
也只是任意的贏家。

實測（合成資料、`ridge`、`alpha ∈ {0.1, 1, 10, 100}`、horizon 5）：`ic_fold_mean`
由固定超參數的 0.166 降到 0.146。**這個下降就是被移除的偏誤本身，不是退步**——
這也是做這件事唯一的理由。

它**沒有**移除的部分：網格本身、特徵集與模型清單，仍然是看著這份資料做的人為選擇。
限制 1 依舊成立，本節只拆掉其中一層。

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
   十天的預測不是十次獨立的市場押注，而是兩次押注被觀測了五遍（見第 8 節限制 4；
   `ic_fold_t` 曾有同樣的毛病，現已依第 4 節〈`ic_fold_t` 該除以幾〉改用 `independent_folds()`）。
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

### 7d. 資料新鮮度：停掉的行情源不會讓報告安靜下來

`journal.data_freshness()`

前面每個數字描述的都是模型，這一個描述的是模型底下的資料——兩者的失效方向
相反。行情源停止更新時，預測日誌不會變安靜，它會**有自信地過期**：
`score_journal` 繼續拿同一批 bar 讓同一批預測到期，`record_forecasts` 因為
最後一根 bar 早已入帳而什麼都不寫，報告讀起來和行情正常那天一模一樣。一個
一週沒動過的命中率，看起來就像策略在觀望，也像下載器已經死了——兩者無法從
數字上分辨。

這一節存在的理由來自一次真實事故：2026-09-14 起 GitHub Actions 因配額問題
完全無法配置 runner，價格停在 2026-09-11，而日誌照樣回報 80% 命中率，整整
三天沒有任何一處指出資料已經過期。

**為什麼用日曆天而不是交易日。** `age_days` 是日曆天，所以較長的休市（例如
農曆年）會被標成落後。這是刻意選擇的犯錯方向：多看一眼下載器幾乎沒有成本，
而一個悄悄停止更新的命中率，毀掉的是本專案唯一無法事後調整的數字。要正確
區分「休市」和「壞掉」需要知道各市場的行事曆，這個模組不宣稱自己知道——
它只報告落差並指名，不代替讀者下判斷。

讀不出日期的標的（空的 frame、壞掉的索引）一律以無限大的 age 回報為過期：
讀不到的行情源不是新鮮的行情源。

---

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

這個低估有多大，可以用**週頻**量出來：把同一批報酬以日曆週複合後重算相關，一個
跨越換日線的共同波動就會落進同一個觀測值裡。實測中四檔買進持有的平均兩兩相關
由日頻的 **0.34** 升到週頻的 **0.47**——而且上升幾乎全在跨市場的配對上
（`MU` 對三檔台股由 0.13–0.17 升到 0.33–0.45），三檔台股彼此則幾乎不動
（0.56/0.42/0.60 → 0.58/0.40/0.66）。「只有跨市場的配對在動、同市場的不動」正是
時區假象的指紋，而不是相關性本身變了。`weekly_returns` 與
`PortfolioResult.weekly_correlation()` 產生這組數字，報表的「Same day, or same week?」
一節把日頻與週頻並列。有效賭注數本身仍以日頻計算，只是把這個差距明白報成保留，
而非改用週頻重算——挑一個讓相關看起來最高的頻率，就是本節其餘部分拒絕的那種事後選擇。

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

### 滾動有效賭注數：分散效果在該發揮的時候還在嗎

全樣本的 $N_{\text{eff}}$ 是一個平均值，而平均值正是這個量最不該用的總結方式：
相關性會在回檔時上升，也就是說分散效果最薄的時刻，恰好就是它被買來要發揮作用的
時刻。一個把平靜期與壓力期混在一起平均的數字，描述的是兩邊都不存在的狀態。

`rolling_effective_bets` 以 **63 個交易日**（一季）的滾動視窗重算相關矩陣與
$N_{\text{eff}}$，`PortfolioResult.rolling_bets()` 再把同一天的**回檔幅度**併進去。
兩者都在該日期上是因果的——視窗以該日為終點，回檔以「到該日為止的最高權益」為基準——
所以兩者可以並排解讀而不引入前視。視窗長度寫死在程式碼裡而不是在報表裡挑：
挑一個讓曲線看起來最平穩（或最劇烈）的視窗，就是本節其餘部分拒絕的那種事後選擇。

「該發揮的時候」以**三分位**定義，不用門檻。「回檔超過 x%」這種寫法，等於邀請 x 在
答案已經看得見之後才被決定；三分位沒有東西可以挑。`bets_by_drawdown()` 依視窗終點
的回檔深度分成三組（最深的在前），報出各組的平均相關、平均有效賭注數。

**差距要先定大小，才能被描述。** 兩組平均差個十分之一個賭注不是發現，把它寫成發現
正是本專案存在的目的所要避免的。因此 `effective_bets_stress_z` 把「最深三分位減最淺
三分位」除以它自己的標準誤，而分母的樣本數**依視窗長度折減**：相鄰視窗共用
`window - 1` 天，一段視窗長度的期間只值一次對市場的觀測。這與預測日誌的
`hit_rate_z` 用不重疊區塊、`ic_fold_t` 用 `independent_folds()` 是同一個處理。
不折減的版本以 `effective_bets_stress_z_naive` 並列報出：前者假設視窗內完全冗餘、
後者假設完全獨立，誠實的答案在兩者之間，兩者不一致時信較小的那個。報表只在
$|z| \ge 2$ 時才下判斷，否則明說「這份樣本答不出來」——而「答不出來」不等於「沒問題」。

**這一節的三個保留。** 權重在整個樣本期間固定，腿與腿之間的再平衡不計成本；
滾動視窗彼此重疊，所以那條線上的低點是「許多相關抽樣的最小值」，偏低，不能當成
「最壞的一季」來讀；而這一節要找的失效本來就是尾部事件——一份平靜的樣本裡不會有，
所以 $z$ 落在雜訊區間只代表這份樣本沒抓到，不代表下一次回檔時相關性不會併攏。

---

## 9. 已知限制

1. **選擇偏誤**：特徵集與模型清單是看著這份資料挑的。walk-forward 能防止單一 fold 的
   洩漏，但擋不住「研究者在整份資料上反覆試」造成的偏誤。
   超參數這一層可以拆掉：`tuning=TuningConfig(grid=...)` 會在每個 fold 內、只用該
   fold 的訓練 bar 選超參數（見第 4 節〈Fold 內選超參數〉），代價是配適次數乘上
   `len(grid) * n_inner_folds + 1`，而且報出來的績效通常會**變低**。預設不開，
   所以預設跑法仍帶著這層偏誤——這是速度與誠實之間的取捨，這裡把它寫明而不是藏起來。
   網格本身、特徵集與模型清單則沒有任何 walk-forward 救得了。
2. **成本模型過於簡化**：固定價差，未含市場衝擊、借券成本、跳空穿價。
3. **投組層級仍然很薄**：`screen` 會報出相關性、有效賭注數、風險貢獻，以及有效賭注數
   在回檔期間是否縮水（見 8 節），但權重全樣本固定、腿間再平衡不計成本、也沒有資金
   上限或部位上限；訊號本身仍是逐檔獨立產生的，沒有任何跨標的的建模。
4. **重疊訊號**：$h > 1$ 時每天以最新預測更新部位，屬慣例作法，但會使有效樣本數
   小於 bar 數。兩個吃到這個問題的顯著性數字都已改掉分母：預測日誌的 `hit_rate_z`
   依 7c 節改用不重疊區塊，`ic_fold_t` 依第 4 節〈`ic_fold_t` 該除以幾〉改用 `independent_folds()`。剩下的
   限制是**其餘指標沒改**——`ic_pearson`、`directional_accuracy`、`sharpe` 等仍以
   bar 為單位計數，它們的隱含精度依舊偏樂觀；這些指標本身不附帶顯著性宣稱，所以
   優先序較低，但讀的時候要知道。
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
