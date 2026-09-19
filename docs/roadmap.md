# Roadmap

The daily routine reads this file to decide what to work on, and edits it when
something is finished or a new gap appears. Items are ordered by how much they
change what the project can honestly claim - not by how interesting they are to
build.

Keep each item small enough to land in one reviewed pull request.

## Now

- [ ] **Restore GitHub Actions minutes. Not a code item - nobody here can fix
  it by writing anything.** This repository is private, so Actions minutes are
  metered, and the included quota is spent. The signature is unmistakable once
  you know it: *every* workflow - `ci.yml` as much as `daily-prices.yml`, on
  push, `pull_request` and `schedule` alike - fails after three to five
  seconds with `runner_id: 0` and no downloadable logs, because no runner was
  ever assigned. No step ran, so no step can be debugged. Last green run of
  anything: 2026-09-13.

  Three ways out, all of them the repository owner's to take: raise the
  spending limit under <https://github.com/settings/billing>, make the
  repository public (standard runners are unmetered for public repositories),
  or wait for the monthly reset. Until one of them happens, `make check` is
  the gate (see below) and `scripts/daily_update.py` is the feed.

  What it is costing, so that the cost is on the record rather than in
  someone's memory: the journal has been frozen since 2026-09-11 at 15 matured
  forecasts covering **one** independent horizon. Independent horizons arrive
  one trading day at a time and cannot be backfilled, because a forecast
  written today about a day that has already happened is not a forecast. Every
  day the feed stays down is a day the only untunable number here does not
  earn.

## Next

- [ ] **Effective sample size for `ic_fold_t`.** The journal's `hit_rate_z` now
  divides by non-overlapping horizon blocks rather than by row count, but
  `ic_fold_t` still carries the optimistic degrees of freedom that limitation 4
  in `docs/methodology.md` describes. The same block logic applies.
- [ ] **Rolling correlation in the portfolio section.** `effective_bets` is a
  full-sample average, and correlations rise in exactly the drawdowns the
  diversification was supposed to cushion. A rolling window would show whether
  the bet count collapses when it matters, the same way `hit_rate_z` over the
  journal's timeline shows *when* a decay started.
- [ ] **Time-zone-aware alignment for cross-market correlation.** Matching a
  Taipei bar to a New York bar by calendar date understates their correlation:
  part of a shared move lands on the next date for one of them. Lagging one
  market by a bar, or comparing weekly returns, would size the effect. It is
  visible in the current output - `MU` correlates 0.13-0.19 with the Taiwan
  names against 0.46-0.65 among themselves - so the effective bet count is
  currently flattered by an unknown amount.

## Later

- [ ] **Intraday or weekly bars.** The whole framework assumes daily.
- [ ] **Borrow costs and short availability.** Shorting is currently free and
  always possible, which it is not - especially for Taiwan small caps.
- [ ] **Walk-forward hyper-parameter selection.** Hyper-parameters are fixed
  and were chosen while looking at the data; selecting them inside each fold
  would remove one layer of selection bias.

## Done

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
