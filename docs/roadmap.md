# Roadmap

The daily routine reads this file to decide what to work on, and edits it when
something is finished or a new gap appears. Items are ordered by how much they
change what the project can honestly claim - not by how interesting they are to
build.

Keep each item small enough to land in one reviewed pull request.

## Now

- [x] **Per-symbol position sizing in the journal.** `record_forecasts` now
  sizes each forecast by the symbol's own trailing volatility when
  `BacktestConfig.vol_target` is set, the same way the backtest does. It
  previously called `signal_to_positions` without `asset_returns`, which
  raises when `vol_target` is set - a raise the per-symbol `except
  (ValueError, ...)` swallowed, so a vol-targeted config silently recorded
  nothing for any symbol instead of erroring or scaling.
- [x] **Turnover in the journal report.** `ScoreResult.metrics()` now reports
  `live_annual_turnover`, averaged per symbol the same way `live_ic` is. It
  counts every recorded forecast, matured or not, since a position pays for
  flipping the day it flips rather than once its horizon elapses - waiting
  on `scored` would report nothing for weeks after the journal starts.

## Next

- [ ] **Multi-asset portfolio construction.** Everything is single-asset today.
  Correlated memory names traded together are not four independent bets, and
  nothing in the framework says so.
- [ ] **Regime-conditional evaluation.** Report metrics split by realised
  volatility tercile. A model that only works in calm markets is a different
  proposition from one that works throughout.
- [ ] **Deflated Sharpe ratio.** `probabilistic_sharpe_ratio` corrects for
  sample length and higher moments but not for the number of configurations
  tried. `screen` already corrects across symbols; the same problem exists
  across models and windows.
- [ ] **Feature importance stability.** Importances are averaged across folds
  but their variance is never reported, so a feature that matters in one fold
  and not the next looks the same as a consistent one.

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

## Rejected, and why

- **Deep learning models (LSTM/Transformer).** The measured edge on daily bars
  is a fraction of a percent of variance. Model capacity is not the binding
  constraint; data and cost are. Adding one would raise the framework's
  apparent sophistication without changing a single honest conclusion.
- **Sentiment or news features.** Would require a data source this project has
  no licence for, and the leakage risk (headline timestamps that precede the
  event they describe) is severe enough to need its own audit.
