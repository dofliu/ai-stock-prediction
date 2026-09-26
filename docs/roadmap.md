# Roadmap

The daily routine reads this file to decide what to work on, and edits it when
something is finished or a new gap appears. Items are ordered by how much they
change what the project can honestly claim - not by how interesting they are to
build.

Keep each item small enough to land in one reviewed pull request.

## Now

Nothing queued. The daily routine adds an item here when it finds a gap.

## Next

Nothing queued. The daily routine adds an item here when it finds a gap; an
empty queue is a real answer, and a day that pushes nothing is a fine outcome.

## Later

- [ ] **Intraday or weekly bars.** The whole framework assumes daily.
- [ ] **Borrow costs and short availability.** Shorting is currently free and
  always possible, which it is not - especially for Taiwan small caps.

## Done

- [x] **Show the whole open book in the journal's "In flight" table**
  (`in_flight_table`, 2026-09-26). A defect found while reading the day's
  journal report, not from the queue - the queue was empty.

  The section printed `live.pending.tail(12)`. `score_journal` sorts pending
  by `symbol` first, so a twelve-row slice does not keep the twelve most
  recent forecasts - it keeps the last twelve in *symbol* order, and drops the
  alphabetically first symbol's open positions. At the steady state of the
  current universe, four symbols at a five-day horizon, the open book holds
  sixteen forecasts and the table silently dropped one symbol entirely: a
  reader would have seen a four-symbol strategy with three symbols in flight.

  The 2026-09-26 report showed it in its mildest form and still showed it. It
  said "Still in flight (horizon not elapsed): 13" and printed twelve rows,
  omitting `2026-09-22 2337.TW` - the oldest open forecast, the next one due
  to be scored - with no note that anything had been left out.

  This is the same defect the rolling window carried a day earlier, so it
  takes the same answer: cut between whole dates, never through a date's
  cross-section, and say what was left out. The table is now ordered oldest
  first, which is the order these mature in, and `IN_FLIGHT_ROW_BUDGET` is a
  readability limit rather than a correctness one - a single date is printed
  whole even when it alone overruns the budget, because a date that looks
  complete and is not is worse than a long table. Nothing truncates at the
  current universe size; the budget binds only on a universe wide enough that
  the table would stop being readable anyway.

  No existing test touched this section at all, which is why it went
  unnoticed - every journal report test used a single symbol, where the row
  order within a date cannot matter.
- [x] **Step the live-vs-backtest rolling window by date, not by forecast row**
  (`rolling_compare_with_backtest`, 2026-09-25). A defect found while reading
  the day's journal report, not from the queue - the queue was empty.

  The window advanced one matured forecast at a time while the table and the
  chart were keyed on `asof_date`, so a four-symbol day became four rows under
  one date and three of them ended part-way through that day's cross-section.
  The journal records every symbol under one `asof_date` and carries no
  ordering within it - rows come out in universe order, not in time order - so
  a mid-day cut kept an arbitrary subset of the symbols. With per-symbol live
  hit rates currently spanning 0.50 to 0.86, that moved a window's hit rate by
  several points for reasons with nothing to do with when anything happened.

  The 2026-09-25 report showed it in its mildest form and still showed it: two
  rows both dated 2026-09-11, both reading 0.5667, when the pooled hit rate
  over the same matured forecasts was 0.5806. A series meant to say *when* a
  gap opened was disagreeing with the number it was meant to decompose. It now
  prints one row per date, 0.5806, which is the pooled figure because the whole
  matured journal is currently one window.

  `window` is now a floor rather than an exact count: each window is the
  shortest run of trailing whole dates holding at least `window` matured
  forecasts, so `n_scored` varies and the report prints its range instead of
  asserting a fixed one. The carrying test makes one symbol always right and
  three always wrong, so every whole-day window hits at exactly 0.25 and no
  mid-day one can; the other two pin that dates are unique and that the window
  shrinks from the front rather than quietly becoming an expanding window.
  Every existing test used a single symbol, where the row order within a date
  cannot matter - which is why this went unnoticed.
- [x] **Walk-forward hyper-parameter selection** (`ai_stock.evaluation.tuning`,
  `run_walk_forward(..., tuning=...)`). Hyper-parameters chosen once on the
  full sample and then evaluated out-of-sample report a number no live run
  could have produced: the live run would have had to pick them from the past.
  Each fold now selects its own from an inner walk-forward over its own
  training bars, reusing the same `WalkForwardSplitter` so the inner embargo
  is the same tested code path as the outer one.

  The test that carries the claim scrambles every feature *after* the outer
  training window and asserts that not one fold's chosen value moves. That is
  the only thing this feature asserts, and a splitter off by one, an inner
  fold measured on the pooled calendar, or an embargo applied in the wrong
  direction would each break it.

  Three choices worth recording. **The selection metric is `ic_pearson`, not
  Sharpe** - Sharpe reads the cost model, the sizing rule and the leverage cap
  as well as the forecast, so tuning on it lets a candidate win by suiting the
  trading configuration, which is the one thing downstream of this a user
  changes freely. **The inner split is a walk-forward, not a single hold-out**,
  because a hold-out picks hyper-parameters on one stretch of market - the
  fragility this project spends its effort measuring elsewhere. **`min_inner_train`
  is an admitted judgement call**: the schedule arithmetic scales with the
  window and never reaches a natural break point, so without an explicit floor
  a 40-bar window splits into a 23-bar training block and reports a winner.
  Below the floor the defaults are kept and the report says nothing was
  selected, rather than dressing a coin flip as a choice.

  Read `selection_stability()` before any performance number from a tuned run:
  a winner that changes every fold, or a `selection_spread` near zero, both
  mean the candidates are indistinguishable at this sample size, and neither
  shows up in the performance table. On the synthetic screen `ic_fold_mean`
  falls from 0.166 fixed to 0.146 tuned - **that drop is the bias being
  removed, not a regression**, and it is the only reason to do this.

  Off by default, which is itself a limitation and is now stated in
  limitation 1 rather than left implicit: the fit count multiplies by
  `len(grid) * n_inner_folds + 1`, so the default run still carries the bias.
  What no walk-forward can remove, and limitation 1 still says: the grid, the
  feature set and the model list are human choices made with this data in view.
  This closes one layer of limitation 1, not the whole.
- [x] **GitHub Actions minutes restored** (2026-09-23, by the repository
  owner - it was never a code item). The outage ran from 2026-09-13 to
  2026-09-22: every workflow, on push, `pull_request` and `schedule` alike,
  died in three to five seconds with `runner_id: 0` and no logs, because no
  runner was ever assigned. Run #25 of `daily-prices.yml` completed
  successfully on 2026-09-23 and committed the first new bars in ten days.

  What it cost, recorded here rather than left in someone's memory. The
  journal sat frozen at 15 matured forecasts covering **one** independent
  horizon for nine calendar days, and the ten-day hole between the
  2026-09-11 and 2026-09-21 rows is permanent: independent horizons arrive
  one trading day at a time and cannot be backfilled, because a forecast
  written today about a day that has already happened is not a forecast. The
  journal now stands at 31 matured forecasts over **two** independent
  horizons - the count that governs every significance claim here, and the
  one the outage was spending.

  Two things built during the outage are worth keeping now that it is over,
  because neither was only a workaround: `make check` as the local gate, and
  `scripts/daily_update.py` as a feed that does not need a runner. Keep both
  green - the next quota reset is also a future outage.
- [x] Rolling correlation in the portfolio section
  (`rolling_effective_bets`, `PortfolioResult.rolling_bets()`,
  `bets_by_drawdown()`). `effective_bets` was a full-sample average of a
  quantity bought as insurance, and correlations rise in exactly the
  drawdowns the diversification was supposed to cushion, so the average
  describes a state that held in neither half. The bet count is re-measured
  over a trailing 63-day window and conditioned on the drawdown each window
  ended in, both causal at their date, split by tercile rather than by a
  threshold that could be chosen once the answer was visible.

  The part worth flagging, because the first draft got it wrong: the report
  originally called a gap of 0.08 bets a collapse, which is the exact failure
  this project exists to avoid. `effective_bets_stress_z` now divides the
  tercile gap by its own standard error at a window count deflated by the
  window length - consecutive windows share `window - 1` days, so a tercile
  holding 129 windows does not hold 129 reads of the market - with
  `effective_bets_stress_z_naive` beside it, the same bracketing
  `hit_rate_z` / `hit_rate_z_naive` uses. The report only draws a conclusion
  at `|z| >= 2` and otherwise says the sample cannot tell, which is not the
  same as saying there is no problem. On the synthetic screen the deflated z
  is -0.18 against a naive -1.46: nothing to report, correctly.

  What it does not do: the failure being tested for is a tail event, so a
  quiet sample not containing it is weak evidence either way, and the minimum
  of the rolling line is a minimum over heavily overlapping draws and is
  biased low. Both are stated in the section rather than left for the reader.
- [x] Time-zone-aware alignment for cross-market correlation, sized by weekly
  returns. Matching a Taipei bar to a New York bar by calendar date splits a
  shared move that crosses midnight across two dates, so the same-day
  correlation understates it and the effective bet count is flattered.
  `weekly_returns` compounds the sleeves into calendar weeks (dropping weeks
  nobody traded, the same reason `align_sleeves` intersects rather than
  unions), and `PortfolioResult.weekly_correlation` /
  `weekly_asset_correlation` re-measure the correlation there.
  `metrics()` gains `mean_correlation_weekly`, `mean_asset_correlation_weekly`,
  `n_weeks` and the two `*_alignment_gap`s; the screen report's `Same day, or
  same week?` section prints both frequencies side by side.

  Chose weekly returns over lagging one market by a bar, because the lag needs
  a hardcoded map of which symbol trades in which market - coupling a
  symbol-agnostic module to `config/universe.txt` - and taking the best
  correlation over a set of candidate lags is a max-over-noise the rest of
  this project spends its effort removing. Weekly is one pre-committed
  alternative frequency with no selection in it. The effective bet count is
  left on daily returns with the gap reported as a caveat, not recomputed
  weekly: switching to whichever frequency shows the highest correlation would
  be the same after-the-fact choice the section already refuses for weights.
  On the four real symbols the mean buy-and-hold correlation rises from 0.34
  (daily) to 0.47 (weekly), and the rise sits entirely on the cross-market
  `MU` pairs (0.13-0.17 -> 0.33-0.45) while the same-market Taiwan pairs barely
  move - the fingerprint of a time-zone artefact rather than a real change in
  co-movement. The residual limitation is that it is measured post-intersection
  and on one alternative frequency; it sizes the effect, it does not correct it.
- [x] Fail the daily workflow on a stale feed. `ai-stock journal
  --fail-if-stale [DAYS]` exits 3 when no symbol has a bar newer than DAYS,
  turning `data_freshness`'s observation into an alarm a scheduler can act on.
  Wired into `daily-prices.yml` after the commit step, behind
  `continue-on-error`, so a stopped feed can never cost the bars that did
  arrive. The alarm carries its own threshold (10 days in the workflow) rather
  than reusing the report's four: the report is read by a person, where a
  needless glance is cheap, while this one fails a build, and the Taiwan
  market shuts for up to nine calendar days over Lunar New Year. Exit code 3
  is distinct from the 2 used for a crash, because "this command is broken"
  and "the data underneath it stopped moving" need opposite responses. Note
  what it does *not* cover, which is the outage in progress: this fires from
  inside a job, and a quota failure means no job ever starts.
- [x] Effective sample size for `ic_fold_t`
  (`WalkForwardResult.independent_folds()`). The t-statistic across folds
  divided by `sqrt(n_folds)`, which is the right sample size only when the
  folds do not share market. Two things in the schedule make them share it:
  `step` below `test_size` overlaps the test windows outright - a case
  `run_walk_forward` already warned about for the *pooled* sample while the
  t-statistic said nothing - and a `horizon`-day target makes even abutting
  windows share `horizon - 1` bars of outcome. `ic_fold_n_eff` counts the
  union of each fold's `[test_start, test_end + horizon)` span over their mean
  width, `ic_fold_t` divides by that, and `ic_fold_t_naive` keeps the old
  figure beside it, the same way `hit_rate_z_naive` sits beside `hit_rate_z`.

  Two things worth recording, because the roadmap entry this closes got both
  wrong. **The journal's block logic does not transfer directly** - applying
  `independent_blocks()` to the pooled predictions would count ~250 blocks
  against 10 folds and make the t-statistic *larger*. What generalises is the
  ratio behind it, not the grouping. **And on the default schedule the
  correction is small, which is the correct answer, not a disappointment.**
  At `horizon=1` abutting folds are genuinely independent and `ic_fold_n_eff`
  equals the fold count exactly; at `horizon=5` five folds read as 4.83 and t
  moves 2.53 -> 2.49. Limitation 4 in `docs/methodology.md` claimed the
  degrees of freedom were optimistic without saying through which channel, and
  now names it. The honest residue is that every *other* bar-counted metric
  still carries the optimistic precision, which limitation 4 now says.
- [x] Core pipeline: synthetic market, causal features, walk-forward with
  embargo, cost-aware backtest, Monte Carlo, reports, CLI. (#1)
- [x] Horizon-aware labelling of the simulate summary. (#2)
- [x] Universe screen with Benjamini-Hochberg correction across symbols. (#3)
- [x] Forecast journal, live-vs-backtest gap, and the daily price workflow. (#4)
- [x] Rolling hit_rate_z over the journal's timeline, so a decay shows *when*
  it started rather than only that it happened in aggregate.
- [x] Per-symbol position sizing in the journal, honouring
  `BacktestConfig.vol_target` the same way the backtest does. Also fixed a
  latent bug this exposed: a configured `vol_target` made every recorded
  forecast silently vanish, because sizing a lone-row signal series raised
  `ValueError` for lack of a returns window, and `record_forecasts` treats any
  `ValueError` as "skip this symbol".
- [x] Turnover in the journal report: `score_journal` records each row's raw
  position change as `turnover`, and `ScoreResult.metrics()` reports
  `annual_turnover` (matured forecasts only) alongside `live_annual_turnover`
  (every recorded forecast, matured or not).
- [x] Deflated Sharpe ratio (`ai_stock.evaluation.metrics.deflated_sharpe_ratio`),
  covering the same selection bias `screen`'s BH correction covers across
  symbols, but across the models compared in one `compare` run. Reported as
  `deflated_sharpe` in the comparison report, benchmarked against the Sharpe
  the best of that many skill-less trials would show by chance rather than
  against zero.
- [x] Feature importance stability: `WalkForwardResult.feature_importance_std`
  and `feature_importance_stability()` report each feature's std and
  coefficient of variation across folds alongside its mean, so a feature that
  swings from irrelevant to dominant between folds no longer looks the same
  as one that is consistently useful. Surfaced in the backtest report as a
  mean/std/cv table.
- [x] Fixed the standard error behind `hit_rate_z`, the one number in this
  project that cannot be tuned after the fact. It divided the live-versus-
  backtest gap by the standard error at every matured forecast, which counts
  the same market move once per overlapping horizon and once per correlated
  symbol; `independent_blocks()` now counts non-overlapping horizon windows
  instead, and the old figure is kept beside it as `hit_rate_z_naive` so the
  two bracket the honest answer. Also fixed a smaller defect the same code
  path hid: `hit_rate` was computed over forecasts that took a side, but its
  standard error used `n_scored`, which includes flat ones.
- [x] Multi-asset portfolio construction (`ai_stock.portfolio`): the screened
  symbols aligned onto their common trading calendar and held together, with
  the diversification the realised correlation actually delivers reported as a
  diversification ratio, an effective number of bets (`DR^2`, which reduces to
  `n / (1 + (n-1) * rho)` for the equicorrelated case), per-sleeve risk
  contributions, and the Sharpe the same sleeves would show if independent.
  Equal-weight and inverse-volatility only - no scheme reads the correlation
  matrix, so none can overfit it. The sleeve correlation is always reported
  next to the buy-and-hold correlation of the same names, because sleeves that
  decorrelate only because two models disagree look identical to sleeves that
  decorrelate because the names do.
- [x] Regime-conditional evaluation: `WalkForwardResult.regime_metrics()`
  splits pooled out-of-sample predictions into terciles of trailing realised
  volatility (computed causally on the full price series before folding, so
  fold-boundary gaps cannot contaminate it) and reports IC and directional
  accuracy within each. Surfaced in the backtest report so a model whose edge
  only shows up in the calm tercile no longer looks the same as one that
  holds throughout.
- [x] Fixed a silent rewrite of the forecast journal. `load_journal` read the
  CSV with pandas' default float parser, which is fast rather than correctly
  rounded and lands up to an ulp from the value the text denotes;
  `append_forecasts` then rewrites the whole file, so every run wrote the
  slightly-wrong float back and a row recorded before its outcome existed
  stopped being the row that was recorded. Too small to move any statistic,
  and that is exactly the problem: the journal's whole claim is that it is
  append-only, and it was not. Reading with `float_precision="round_trip"`
  makes the file byte-stable - the committed record now survives any number
  of no-op days unchanged. Found by a guard in the local daily runner that
  refused to commit a journal that was not an extension of the committed one.
- [x] A local path for the daily update, for as long as Actions cannot run
  (`scripts/daily_update.py`). The same fetch-record-score-commit loop the
  workflow runs, keeping its ordering - the data is committed before a
  download failure is allowed to go red - plus the guards a runner never
  needed and a working tree does: it refuses to run off `main`, commits with
  `--only` so staged work in progress cannot ride along to `main`, and
  compares the journal against the committed version before committing,
  aborting if any existing row moved. That last guard found the
  `float_precision` defect on its first run against real data.
- [x] The Makefile as the local gate, now that CI cannot run. `make check`
  runs exactly what `.github/workflows/ci.yml` runs, in the same order
  (lint, test, doctest, CLI smoke), in about 40 seconds. Closes two gaps that
  were harmless while CI was the real gate and are not any more: `make lint`
  covered `src tests` but not `scripts`, and `make test` ran none of the 14
  doctests, because `testpaths` is `tests` and they live under `src`. Also
  fixes a `make clean` that ran `rm -rf data` - which is `data/journal/`,
  the record that cannot be regenerated, and which `.gitignore` goes out of
  its way to preserve. `tests/test_makefile.py` pins all of it.
- [x] Data freshness in the journal (`ai_stock.journal.data_freshness`): the
  age of each symbol's most recent bar, stated in the report and on the CLI
  above the performance it qualifies. Found while diagnosing a three-day
  outage in the daily workflow - the journal had gone on reporting an 80% hit
  rate from bars that stopped arriving on 2026-09-11, because a stopped feed
  does not make this report go quiet, it makes it repeat. Calendar days, not
  trading days, so a long market holiday reads as behind; that is the cheap
  direction to be wrong in.

## Rejected, and why

- **Deep learning models (LSTM/Transformer).** The measured edge on daily bars
  is a fraction of a percent of variance. Model capacity is not the binding
  constraint; data and cost are. Adding one would raise the framework's
  apparent sophistication without changing a single honest conclusion.
- **Sentiment or news features.** Would require a data source this project has
  no licence for, and the leakage risk (headline timestamps that precede the
  event they describe) is severe enough to need its own audit.
