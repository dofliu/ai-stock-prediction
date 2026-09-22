# Roadmap

An automated session reads this file to decide what to work on, and edits it
when something is finished or a new gap appears. Items are ordered by how much
they change what the project can honestly claim - not by how interesting they
are to build.

Keep each item small enough to land in one reviewed pull request.

**How a session picks work.** In order, stopping at the first that applies:

1. An open pull request that is not mergeable - red gate, merge conflict, or an
   unanswered review comment. Drive it to mergeable before starting anything
   new. Two sessions opening two pull requests for the same item is the waste
   this rule exists to prevent.
2. The first unchecked item under **Now**.
3. If **Now** is empty, promote the top item of **Next**, or of **Later** if
   **Next** is empty, and do that.

**Blocked** is not a work list. Nothing in it can be fixed by writing code
here, so nothing in it is ever picked; it is there so the cost stays on the
record rather than in someone's memory.

`make check` is the gate, and while Actions cannot run it is the only thing
standing between a change and `main`. Push nothing that has not passed it, and
say so in the pull request.

## Now

- [ ] **Rolling correlation in the portfolio section.** `effective_bets` is a
  full-sample average, and correlations rise in exactly the drawdowns the
  diversification was supposed to cushion. A rolling window would show whether
  the bet count collapses when it matters, the same way `hit_rate_z` over the
  journal's timeline shows *when* a decay started. Section 8 of
  `docs/methodology.md` already ends by admitting this ("報表上的有效賭注數是
  全樣本平均值，不是承諾"), so the doc change is to replace an admission with a
  number.
- [ ] **Time-zone-aware alignment for cross-market correlation.** Matching a
  Taipei bar to a New York bar by calendar date understates their correlation:
  part of a shared move lands on the next date for one of them. Lagging one
  market by a bar, or comparing weekly returns, would size the effect. It is
  visible in the current output - `MU` correlates 0.13-0.19 with the Taiwan
  names against 0.46-0.65 among themselves - so the effective bet count is
  currently flattered by an unknown amount.

Both run entirely on the 3,664 bars already committed under `data/prices/`.
Neither needs a network, a runner, or a fresh bar, which is why they are here
while the feed is down.

## Next

Empty on purpose. Both items that were here moved up when the Actions blocker
moved out of **Now**, and the queue behind them is **Later**. An item added
here needs a reason it outranks what is already in **Later**; without one, a
roadmap becomes a wish list.

## Later

- [ ] **Intraday or weekly bars.** The whole framework assumes daily.
- [ ] **Borrow costs and short availability.** Shorting is currently free and
  always possible, which it is not - especially for Taiwan small caps.
- [ ] **Walk-forward hyper-parameter selection.** Hyper-parameters are fixed
  and were chosen while looking at the data; selecting them inside each fold
  would remove one layer of selection bias.

## Blocked

Not a work list. Nothing here can be fixed by writing code in this repository,
so a session never picks from it. It is here so the cost stays on the record.

- [ ] **GitHub Actions minutes are exhausted.** This repository is private, so
  Actions minutes are metered and the included quota is spent. The signature is
  unmistakable once you know it: *every* workflow - `ci.yml` as much as
  `daily-prices.yml`, on push, `pull_request` and `schedule` alike - fails
  after three to five seconds with `runner_id: 0` and no downloadable logs,
  because no runner was ever assigned. No step ran, so no step can be debugged.
  Last green run of anything: 2026-09-13. The scheduled feed has failed every
  weekday since, most recently run 24 on 2026-09-22.

  Three ways out, all of them the repository owner's to take: raise the
  spending limit under <https://github.com/settings/billing>, make the
  repository public (standard runners are unmetered for public repositories),
  or wait for the monthly reset.

  **A cloud session is not a substitute for the runner, and this is the part
  worth recording so nobody spends another hour rediscovering it.** Claude
  Code's remote environment reaches PyPI and GitHub but not the price feed: a
  CONNECT to `query1.finance.yahoo.com:443` is refused by the environment's
  network policy with `403 Forbidden`, before any request is sent, so no
  retry, user-agent, or alternative client changes the outcome. `make check`
  runs there cleanly in about 40 seconds (456 tests, 17 doctests, smoke), so
  *code* work is unblocked; `scripts/daily_update.py` is not. Until that host
  is allowed by the environment's policy, the only machine that can run the
  daily loop is one the owner controls.

  What it is costing, measured rather than asserted: the journal has been
  frozen since 2026-09-11 - eleven calendar days and seven trading days - at
  31 recorded forecasts of which 15 have matured, covering **one** independent
  horizon. Scored on 2026-09-22 it still reads a live hit rate of 80.00%
  against the backtest's 50.64% claim, z = 0.5873 over that one horizon beside
  a naive z of 2.2747. The distance between those two z values is the entire
  reason the feed has to keep running: the naive one looks like a finding, and
  there is not yet enough independent market underneath it to say. Independent
  horizons arrive one trading day at a time and cannot be backfilled, because a
  forecast written today about a day that has already happened is not a
  forecast.

## Done

- [x] Record what the framework actually finds on the real universe, and make
  the roadmap safe for an automated session to read. Two separate defects in
  the documentation, both of the same kind - a number that looks like a result
  from this repository and is not.

  **The README's screen example was synthetic and unlabelled**, showing a
  survivor at `q = 0.0399`, directly above a paragraph about `config/
  universe.txt`. The real four-symbol run is now beside it: every
  `excess_sharpe` negative, nothing surviving `q <= 0.05`, `2337.TW` reaching
  `ic_fold_t = 2.06` and still losing to buy & hold by 0.38 of a Sharpe across
  64x annual turnover. That null is the project's only end-to-end validation
  that means anything - planted edge recovered, efficient market refused, real
  daily bars returning nothing - and it was the one result not written down.

  **The journal example showed `108 / 12` scored against a real `15 / 16`**,
  under a real `data as of 2026-09-11`. Replaced with the actual output, which
  makes the point the prose was making anyway: 80% hit rate, 0.66 IC and 45%
  P&L, all from one independent horizon, naive z 2.27 against an honest 0.59.

  The roadmap itself had the structural version of the same problem. **Its
  only `Now` item was one no session can do** - restoring Actions minutes is
  the owner's to fix - so every firing read `Now`, found nothing actionable,
  and chose for itself. The blocker moved to a new `Blocked` section that is
  explicitly never picked from, the two unblocked items moved up, and the
  preamble now states the pick order, including "drive an open pull request to
  mergeable before opening another". Also recorded there, because it costs an
  hour to rediscover: Claude Code's cloud environment cannot stand in for the
  runner either - `query1.finance.yahoo.com:443` is refused at CONNECT by the
  environment's network policy, while `make check` runs there in 40 seconds.

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
