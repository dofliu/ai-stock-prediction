# Roadmap

The daily routine reads this file to decide what to work on, and edits it when
something is finished or a new gap appears. Items are ordered by how much they
change what the project can honestly claim - not by how interesting they are to
build.

Keep each item small enough to land in one reviewed pull request.

## Now

- [ ] **Multi-asset portfolio construction.** Everything is single-asset today.
  Correlated memory names traded together are not four independent bets, and
  nothing in the framework says so.

## Next

- [ ] **Effective sample size for `ic_fold_t`.** The journal's `hit_rate_z` now
  divides by non-overlapping horizon blocks rather than by row count, but
  `ic_fold_t` still carries the optimistic degrees of freedom that limitation 4
  in `docs/methodology.md` describes. The same block logic applies.

## Later

- [ ] **Intraday or weekly bars.** The whole framework assumes daily.
- [ ] **Borrow costs and short availability.** Shorting is currently free and
  always possible, which it is not - especially for Taiwan small caps.
- [ ] **Walk-forward hyper-parameter selection.** Hyper-parameters are fixed
  and were chosen while looking at the data; selecting them inside each fold
  would remove one layer of selection bias.

## Done

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
- [x] Regime-conditional evaluation: `WalkForwardResult.regime_metrics()`
  splits pooled out-of-sample predictions into terciles of trailing realised
  volatility (computed causally on the full price series before folding, so
  fold-boundary gaps cannot contaminate it) and reports IC and directional
  accuracy within each. Surfaced in the backtest report so a model whose edge
  only shows up in the calm tercile no longer looks the same as one that
  holds throughout.

## Rejected, and why

- **Deep learning models (LSTM/Transformer).** The measured edge on daily bars
  is a fraction of a percent of variance. Model capacity is not the binding
  constraint; data and cost are. Adding one would raise the framework's
  apparent sophistication without changing a single honest conclusion.
- **Sentiment or news features.** Would require a data source this project has
  no licence for, and the leakage risk (headline timestamps that precede the
  event they describe) is severe enough to need its own audit.
