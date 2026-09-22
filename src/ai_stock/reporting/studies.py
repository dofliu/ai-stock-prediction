"""Composed reports for the three study types the CLI exposes.

Each renderer also writes down how to *read* the numbers, because a table of
Sharpe ratios with no context is how backtests get oversold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_stock.config import ExperimentConfig
from ai_stock.journal import ScoreResult
from ai_stock.pipeline import ModelRun, ScreenResult, SimulationBundle, deflated_sharpe_ratios
from ai_stock.portfolio import PortfolioResult
from ai_stock.reporting.report import (
    Report,
    ascii_bars,
    ascii_histogram,
    ascii_line_chart,
    format_number,
    markdown_table,
    metrics_table,
)

__all__ = [
    "render_backtest_report",
    "render_journal_report",
    "render_comparison_report",
    "render_screen_report",
    "render_simulation_report",
    "summarise_run_row",
]

_RANKING_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    # (metric key, header, render as percent)
    ("ic_fold_mean", "fold IC", False),
    ("ic_fold_t", "IC t-stat", False),
    ("directional_accuracy", "dir. acc.", True),
    ("sharpe", "Sharpe", False),
    ("annualised_return", "ann. return", True),
    ("max_drawdown", "max DD", True),
    ("annual_turnover", "turnover", False),
    ("excess_sharpe", "vs B&H", False),
    ("deflated_sharpe", "deflated Sharpe", False),
)

_SUMMARY_KEYS = (
    "n",
    "n_folds",
    "ic_pearson",
    "ic_fold_mean",
    "ic_fold_std",
    "ic_fold_n_eff",
    "ic_fold_t",
    "ic_fold_t_naive",
    "ic_fold_positive_rate",
    "ic_spearman",
    "r2",
    "rmse",
    "directional_accuracy",
    "accuracy",
    "roc_auc",
    "base_rate",
)

_FINANCIAL_KEYS = (
    "total_return",
    "annualised_return",
    "annualised_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "hit_rate",
    "profit_factor",
    "probabilistic_sharpe",
    "avg_exposure",
    "annual_turnover",
    "time_in_market",
)


def _setup_bullets(config: ExperimentConfig, ohlcv_span: tuple[str, str], n_bars: int) -> list[str]:
    walk_forward = config.walk_forward
    backtest = config.backtest
    return [
        f"Data: {n_bars} bars, {ohlcv_span[0]} to {ohlcv_span[1]}",
        f"Forecast horizon: {config.features.horizon} trading day(s)",
        f"Walk-forward: train={walk_forward.train_size}, test={walk_forward.test_size}, "
        f"step={walk_forward.resolved_step()}, "
        f"{'expanding' if walk_forward.expanding else 'rolling'} window, "
        f"embargo={walk_forward.resolved_embargo(config.features.horizon)} bar(s)",
        f"Position sizing: {backtest.sizing}, max leverage {backtest.max_leverage:g}, "
        f"threshold {backtest.threshold:g}, shorting {'on' if backtest.allow_short else 'off'}",
        f"Costs: {backtest.cost_bps:g} bps commission + {backtest.slippage_bps:g} bps slippage "
        f"per unit traded ({backtest.total_cost_bps:g} bps total)",
    ]


def summarise_run_row(run: ModelRun, extra: dict[str, float] | None = None) -> list[str]:
    """One ranking-table row for ``run``.

    ``extra`` supplies metrics that depend on the other runs it is being
    compared against (e.g. `deflated_sharpe`), so they cannot live on
    ``run.metrics()`` itself.
    """
    metrics = {**run.metrics(), **(extra or {})}
    return [
        run.name,
        *(format_number(metrics.get(key), percent=percent) for key, _, percent in _RANKING_COLUMNS),
    ]


def _ranking_table(
    runs: list[ModelRun], deflated: dict[str, float]
) -> tuple[list[str], list[list[str]]]:
    headers = ["model", *(header for _, header, _ in _RANKING_COLUMNS)]
    rows = [
        summarise_run_row(run, extra={"deflated_sharpe": deflated.get(run.name, float("nan"))})
        for run in runs
    ]
    return headers, rows


def _equity_chart(runs: list[ModelRun], *, limit: int = 3) -> str:
    series: dict[str, pd.Series] = {}
    for run in runs[:limit]:
        series[run.name] = run.backtest.equity
    if runs:
        series["buy & hold"] = runs[0].backtest.benchmark_equity
    return ascii_line_chart(series, width=76, height=16)


def _reading_notes() -> list[str]:
    return [
        "**fold IC** is the mean per-fold correlation between forecast and realised "
        "return; **IC t-stat** is its t-statistic across folds. Believe the per-fold "
        "numbers over the pooled `ic_pearson`, which mixes folds of different volatility.",
        "**IC t-stat** is measured at `ic_fold_n_eff`, not at the fold count: folds "
        "whose test windows overlap, or that abut and share a horizon-length label "
        "tail, are not that many independent reads of the market.",
        "**dir. acc.** above 50% is necessary but nowhere near sufficient: being right "
        "51% of the time on small moves and wrong on large ones still loses money.",
        "**turnover** is annualised traded notional. Multiply it by the cost in bps to "
        "see the yearly cost drag - that is usually what kills a live strategy.",
        "**vs B&H** is Sharpe minus buy-and-hold Sharpe. A strategy that only wins "
        "because the asset rose has not demonstrated skill.",
    ]


def _caveats() -> list[str]:
    return [
        "These results are out-of-sample within a walk-forward schedule, but the "
        "feature set, model list and hyper-parameters were still chosen by looking at "
        "this data. That is a form of selection bias no backtest can remove.",
        "Costs are modelled as a flat spread on traded notional. Real execution adds "
        "market impact, borrowing costs for shorts, and gaps that skip your stop.",
        "Synthetic data contains a deliberately planted edge. Recovering it validates "
        "the pipeline; it says nothing about whether a real market has one.",
        "Nothing here is investment advice.",
    ]


def render_backtest_report(run: ModelRun, config: ExperimentConfig) -> str:
    """Full report for a single model: forecasts, folds, backtest, caveats."""
    walk_forward = run.walk_forward
    backtest = run.backtest
    index = walk_forward.predictions.index
    span = (str(index[0].date()), str(index[-1].date()))

    report = Report(
        f"Walk-forward backtest: `{run.name}`",
        subtitle="Out-of-sample forecasts turned into a traded equity curve, net of costs.",
    )
    report.heading("Setup").bullets(_setup_bullets(config, span, len(index)))

    report.heading("Predictive performance")
    report.raw_table(
        metrics_table({run.name: walk_forward.metrics()}, keys=_SUMMARY_KEYS, label="metric")
    )
    report.bullets(
        [
            "`ic_fold_t` divides the mean fold IC by its dispersion and by the square root "
            "of `ic_fold_n_eff` - the fold-widths of distinct market the schedule reaches, "
            "not the number of folds. Test windows that overlap (`step` below `test_size`) "
            "score the same market twice, and even abutting windows share the last "
            "`horizon - 1` bars of outcome.",
            "`ic_fold_t_naive` is the same statistic at the raw fold count, which is what "
            "this report used to print. It assumes the folds are fully independent and "
            "`ic_fold_t` assumes the shared span is worth nothing, so the honest figure "
            "lies between them; while they disagree, believe the smaller.",
            "Neither estimates a correlation. `ic_fold_n_eff` follows from the fold dates "
            "and the horizon alone, so there is nothing in it to tune.",
        ]
    )

    report.heading("Trading performance")
    report.raw_table(
        metrics_table(
            {"strategy": backtest.metrics, "buy & hold": backtest.benchmark_metrics},
            keys=_FINANCIAL_KEYS,
        )
    )
    report.text(
        f"Cumulative cost drag: {format_number(backtest.total_costs, percent=True)} of starting "
        f"equity across {len(backtest.returns)} bars."
    )

    report.heading("Equity curve")
    report.code_block(
        ascii_line_chart(
            {"strategy": backtest.equity, "buy & hold": backtest.benchmark_equity},
            width=76,
            height=16,
        )
    )

    fold_metrics = walk_forward.fold_metrics()
    if not fold_metrics.empty:
        report.heading("Per-fold stability")
        columns = [
            c
            for c in ("test_start", "test_end", "n_test", "ic_pearson", "directional_accuracy")
            if c in fold_metrics
        ]
        table = fold_metrics[columns].copy()
        for column in ("test_start", "test_end"):
            if column in table:
                table[column] = table[column].dt.date.astype(str)
        report.dataframe(table)

    stability = walk_forward.feature_importance_stability()
    if not stability.empty:
        report.heading("Feature importance (mean, std and cv across folds)")
        report.dataframe(stability.head(12))
        report.text(
            "`cv` (std / |mean|) is undefined with a single fold. A feature with a "
            "high `cv` swings between folds and should be trusted less than its mean "
            "alone suggests, even if another feature with the same mean is stable."
        )

    regime = walk_forward.regime_metrics()
    if not regime.empty:
        report.heading("Regime-conditional performance")
        columns = [
            c
            for c in ("n", "realised_vol_mean", "ic_pearson", "directional_accuracy", "accuracy")
            if c in regime
        ]
        report.dataframe(regime[columns])
        report.text(
            "Split by trailing realised-volatility tercile. A model whose edge only shows "
            "up in the calm bin is a different, weaker claim than one that holds across "
            "all three - pooling every bar together, as the tables above do, cannot tell "
            "the two apart."
        )

    report.heading("How to read this").bullets(_reading_notes())
    report.heading("Caveats").bullets(_caveats())
    return report.render()


def render_comparison_report(runs: list[ModelRun], config: ExperimentConfig) -> str:
    """Ranking report across models evaluated on identical folds."""
    if not runs:
        raise ValueError("no runs to report")

    index = runs[0].walk_forward.predictions.index
    span = (str(index[0].date()), str(index[-1].date()))

    report = Report(
        "Model comparison",
        subtitle="Every model saw the same features, the same folds and the same costs.",
    )
    report.heading("Setup").bullets(_setup_bullets(config, span, len(index)))

    report.heading("Ranking")
    headers, rows = _ranking_table(runs, deflated_sharpe_ratios(runs))
    report.table(headers, rows)
    report.text(
        f"`deflated Sharpe` corrects `probabilistic_sharpe` for having tried "
        f"{len(runs)} model(s) here and kept the best: it is the probability the "
        "true Sharpe beats what the best of that many skill-less attempts would "
        "show by chance, not just zero (Bailey & Lopez de Prado, 2014)."
    )

    report.heading("Equity curves (top 3)")
    report.code_block(_equity_chart(runs))

    report.heading("Predictive metrics")
    report.raw_table(
        metrics_table(
            {run.name: run.walk_forward.metrics() for run in runs},
            keys=_SUMMARY_KEYS,
        )
    )

    report.heading("Trading metrics")
    report.raw_table(
        metrics_table(
            {
                **{run.name: run.backtest.metrics for run in runs},
                "buy & hold": runs[0].backtest.benchmark_metrics,
            },
            keys=_FINANCIAL_KEYS,
        )
    )

    report.heading("How to read this").bullets(
        [
            *_reading_notes(),
            "**deflated Sharpe** only knows about the models compared in this one run. "
            "Re-running `compare` with a different model list, feature set or window and "
            "keeping whichever run looks best reintroduces the same selection bias one "
            "level up - it has no way to see across runs.",
        ]
    )
    report.heading("Caveats").bullets(_caveats())
    return report.render()


def _significance_verdict(p_value: float, observed: float) -> str:
    if pd.isna(p_value):
        return "The null distribution was degenerate, so no verdict can be given."
    if p_value <= 0.01:
        strength = "strong evidence against the luck hypothesis"
    elif p_value <= 0.05:
        strength = "moderate evidence against the luck hypothesis"
    elif p_value <= 0.20:
        strength = "weak, inconclusive evidence"
    else:
        strength = "no evidence of skill: this is what luck looks like"
    return (
        f"Observed statistic {format_number(observed)} with p = {format_number(p_value)} - "
        f"{strength}."
    )


def render_simulation_report(bundle: SimulationBundle, config: ExperimentConfig) -> str:
    """Monte-Carlo study: forward distribution, null test and bootstrap interval."""
    run = bundle.run
    significance = bundle.significance
    paths = bundle.forward_paths
    bootstrap = bundle.sharpe_bootstrap

    index = run.walk_forward.predictions.index
    span = (str(index[0].date()), str(index[-1].date()))

    report = Report(
        f"Monte-Carlo simulation: `{run.name}`",
        subtitle="How much of the backtest survives contact with randomness.",
    )
    report.heading("Setup").bullets(
        [
            *_setup_bullets(config, span, len(index)),
            f"Simulation: {paths.config.n_paths} paths x {paths.config.horizon_days} days "
            f"via {paths.config.method}, {significance.method} null with "
            f"{len(significance.null_distribution)} draws",
        ]
    )

    report.heading("Realised backtest")
    report.raw_table(
        metrics_table(
            {"strategy": run.backtest.metrics, "buy & hold": run.backtest.benchmark_metrics},
            keys=_FINANCIAL_KEYS,
        )
    )

    report.heading("Is the edge distinguishable from luck?")
    report.text(_significance_verdict(significance.p_value, significance.observed))
    report.raw_table(
        metrics_table(
            {
                "null test": {
                    k: v for k, v in significance.summary().items() if isinstance(v, float)
                }
            },
            label="quantity",
        )
    )
    report.code_block(
        ascii_histogram(
            significance.null_distribution,
            bins=20,
            width=44,
            marker=significance.observed,
            marker_label="observed Sharpe",
        )
    )
    report.text(
        f"The null keeps the signal's own turnover ({significance.method}) and only destroys "
        "its timing, so it prices in the cost of trading as often as the strategy does."
    )

    report.heading("Confidence interval for the realised Sharpe")
    report.raw_table(metrics_table({"bootstrap": bootstrap.summary()}, label="quantity"))
    low, high = bootstrap.confidence_interval
    report.text(
        f"A {bootstrap.confidence:.0%} block-bootstrap interval of "
        f"[{format_number(low)}, {format_number(high)}] around a point estimate of "
        f"{format_number(bootstrap.point_estimate)} shows how wide the uncertainty on a single "
        "Sharpe ratio really is."
    )

    report.heading(f"Forward distribution ({paths.config.horizon_days} trading days)")
    report.raw_table(metrics_table({"simulated": paths.summary()}, label="quantity"))
    report.code_block(
        ascii_histogram(
            paths.terminal_values / paths.initial_value - 1.0,
            bins=20,
            width=44,
            marker=0.0,
            marker_label="break-even",
        )
    )

    report.heading("How to read this").bullets(
        [
            "The forward distribution resamples the *strategy's own* realised returns. It "
            "assumes the future resembles the backtest period - the assumption that fails "
            "first in practice.",
            "A p-value below 0.05 means the timing of this signal mattered on this sample. "
            "It is not a guarantee that it will keep mattering.",
            "Read the 5% quantile and the worst drawdown before the median: position sizing "
            "is decided by the bad paths, not the typical one.",
        ]
    )
    report.heading("Caveats").bullets(_caveats())
    return report.render()


_SCREEN_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("n_bars", "bars", False),
    ("ic_fold_mean", "fold IC", False),
    ("ic_fold_t", "IC t-stat", False),
    ("directional_accuracy", "dir. acc.", True),
    ("sharpe", "Sharpe", False),
    ("benchmark_sharpe", "B&H Sharpe", False),
    ("excess_sharpe", "vs B&H", False),
    ("annualised_return", "ann. return", True),
    ("max_drawdown", "max DD", True),
    ("annual_turnover", "turnover", False),
    ("p_value", "p", False),
    ("q_value", "q (FDR)", False),
)


def _sleeve_vs_asset_verdict(metrics: dict[str, float]) -> str:
    """Say where the diversification came from - the names, or the models disagreeing.

    A sleeve correlation well below the correlation of the shares it trades is
    the reading most likely to be over-sold, so it gets the longest warning.
    """
    sleeve = metrics["mean_correlation"]
    asset = metrics["mean_asset_correlation"]
    if not np.isfinite(asset):
        return (
            "Buy-and-hold returns were not supplied, so there is no way to tell whether the "
            "sleeves are less correlated than the shares they trade."
        )

    shares = (
        f"The shares themselves correlate {format_number(asset, digits=2)}, and holding them "
        f"at the same weights would be {format_number(metrics['asset_effective_bets'], digits=2)}"
        " bet(s). "
    )
    if sleeve < asset - 0.1:
        return shares + (
            "The strategies are the less correlated of the two, which is not automatically "
            "good news: sleeves decorrelate when the models are positioned differently, and "
            "models that disagree at random look exactly like this. Read it as evidence of "
            "diversification only once each sleeve has an edge worth diversifying - check the "
            "`vs B&H` and `q` columns above first."
        )
    if sleeve > asset + 0.1:
        return shares + (
            "The strategies are **more** correlated than the shares they trade. Trading the "
            "universe is concentrating risk that holding it would have spread."
        )
    return shares + (
        "The strategies inherit roughly the correlation of the shares, so trading the universe "
        "neither adds nor removes diversification relative to holding it."
    )


def _alignment_section(
    report: Report, portfolio: PortfolioResult, metrics: dict[str, float]
) -> None:
    """Size the same-day alignment penalty by re-measuring correlation weekly.

    Every correlation in this section is measured on same-day returns, and a
    date does not mean the same hours in every market. Recomputing on weekly
    returns lets a shared move that straddled a date boundary land in one
    observation; the gap between the two frequencies is how much the same-day
    match could not see, and therefore how much the bet counts are flattered.
    """
    if not np.isfinite(metrics["n_weeks"]):
        return

    report.heading("Same day, or same week?", level=3)
    report.text(
        f"Every correlation above is same-day. Recomputed on the {int(metrics['n_weeks']):,} "
        "overlapping calendar weeks, the sleeves correlate "
        f"{format_number(metrics['mean_correlation_weekly'])} against "
        f"{format_number(metrics['mean_correlation'])} by day. A shared move that crosses "
        "midnight in one market but not the other is split across two dates, so the same-day "
        "figure cannot see it - and every bet count above is flattered by however much of this "
        "difference is real co-movement rather than the noise a coarser sample adds."
    )

    weekly_assets = portfolio.weekly_asset_correlation()
    if weekly_assets is not None:
        report.text(
            "The shares move the same way: "
            f"{format_number(metrics['mean_asset_correlation'])} correlated by day, "
            f"{format_number(metrics['mean_asset_correlation_weekly'])} by week. The gap falls on "
            "the pairs whose markets keep different hours and barely touches the pairs that "
            "already share a trading calendar, which is what a time-zone artefact looks like and "
            "what a genuine change in how the names move together would not."
        )

    report.heading("Correlation of the sleeve returns, weekly", level=4)
    report.dataframe(portfolio.weekly_correlation())
    if weekly_assets is not None:
        report.heading("Correlation of the shares themselves, weekly", level=4)
        report.dataframe(weekly_assets)


def _portfolio_section(report: Report, result: ScreenResult) -> None:
    """Append what the ranking cannot say: how many bets these symbols really are.

    Every row above is a standalone study. Held together they are not
    independent, and the gap between the two readings is the whole point of
    the section.
    """
    report.heading("Held together, not one at a time")
    try:
        portfolio = result.portfolio()
    except ValueError as error:
        report.text(f"These symbols cannot be combined into a portfolio: {error}.")
        return

    metrics = portfolio.metrics()
    n_sleeves = int(metrics["n_sleeves"])
    effective = metrics["effective_bets"]
    if n_sleeves < 2:
        report.text(
            "A single evaluated symbol is a single bet; there is nothing here to diversify."
        )
        return

    report.text(
        f"Equally weighted, these **{n_sleeves} sleeves behave like "
        f"{format_number(effective, digits=2)} independent bets**, against the "
        f"{n_sleeves} that adding {n_sleeves} separate backtests together would assume. "
        f"Their returns correlate {format_number(metrics['mean_correlation'], digits=2)} "
        "on average."
    )
    report.text(_sleeve_vs_asset_verdict(metrics))

    report.raw_table(
        markdown_table(
            ["quantity", "value"],
            [
                ["sleeves", str(n_sleeves)],
                ["common trading days", f"{int(metrics['n_periods']):,}"],
                [
                    "share of the combined calendar",
                    format_number(metrics["common_fraction"], percent=True),
                ],
                ["mean pairwise correlation (sleeves)", format_number(metrics["mean_correlation"])],
                [
                    "mean pairwise correlation (buy & hold)",
                    format_number(metrics["mean_asset_correlation"]),
                ],
                ["max pairwise correlation (sleeves)", format_number(metrics["max_correlation"])],
                ["diversification ratio", format_number(metrics["diversification_ratio"])],
                ["effective number of bets", format_number(metrics["effective_bets"])],
                [
                    "effective bets from holding the shares instead",
                    format_number(metrics["asset_effective_bets"]),
                ],
                ["portfolio Sharpe", format_number(metrics["sharpe"])],
                [
                    "Sharpe if the sleeves were independent",
                    format_number(metrics["sharpe_if_independent"]),
                ],
                ["max drawdown", format_number(metrics["max_drawdown"], percent=True)],
            ],
        )
    )

    report.heading("Weights, and where the risk actually sits", level=3)
    report.dataframe(portfolio.sleeve_table())
    report.heading("Correlation of the sleeve returns", level=3)
    report.dataframe(portfolio.correlation())
    asset_correlation = portfolio.asset_correlation()
    if asset_correlation is not None:
        report.heading("Correlation of the shares themselves (buy & hold)", level=3)
        report.dataframe(asset_correlation)

    _alignment_section(report, portfolio, metrics)

    report.bullets(
        [
            "`effective number of bets` is the squared diversification ratio. For equally "
            "weighted, equally volatile sleeves correlated at `rho` it is exactly "
            "`n / (1 + (n - 1) * rho)`, so it reads as the count of independent positions "
            "the correlation leaves you actually holding.",
            "`Sharpe if the sleeves were independent` keeps the same returns and the same "
            "weights and only removes the correlation. The distance between it and the "
            "portfolio Sharpe is the diversification a four-backtest sum would have claimed "
            "and this universe does not provide.",
            "`risk_contribution` is each sleeve's share of portfolio variance. Equal capital "
            "is not equal risk: read it against `weight` before concluding the allocation is "
            "balanced.",
            "Correlations are measured only on the days every sleeve traded - "
            f"{format_number(metrics['common_fraction'], percent=True)} of the combined "
            "calendar here. A holiday in one market is not a quiet day for that sleeve, so "
            "filling it with a zero would flatter every number in this section.",
            "Symbols in different time zones are matched by calendar date, and a date does "
            "not mean the same hours in Taipei as it does in New York, so a same-day "
            "correlation across those two markets is understated and the effective bet count "
            "correspondingly flattered. The `Same day, or same week?` section sizes that: the "
            "weekly correlation is the same number with the date-boundary split removed.",
            "Weights are fixed for the whole sample and no scheme here looks at the "
            "correlation matrix. Fitting weights to the same correlations they are then "
            "scored against would make this section a backtest of itself.",
        ]
    )


def render_screen_report(result: ScreenResult, config: ExperimentConfig) -> str:
    """Rank a universe by how well one model trades each symbol.

    The report leads with the multiple-comparison correction rather than
    burying it: the whole point of a screen is that the winner was selected
    from many candidates, which is also how a fake edge is manufactured.
    """
    ranked = result.ranked
    if not ranked and not result.failures:
        raise ValueError("no symbols to report")

    report = Report(
        f"Universe screen: `{result.model_name}`",
        subtitle="Same features, same folds, same costs on every symbol - then ranked.",
    )

    survivors = result.survivors()
    if survivors:
        names = ", ".join(f"`{entry.symbol}`" for entry in survivors)
        verdict = (
            f"**{len(survivors)} of {result.n_tested} symbols** beat buy-and-hold *and* "
            f"survive the FDR correction at q <= {result.alpha:g}: {names}."
        )
    elif ranked:
        verdict = (
            f"**No symbol** both beat buy-and-hold and survived the FDR correction at "
            f"q <= {result.alpha:g}. On this universe, holding the shares was the better "
            "of the two available choices everywhere."
        )
    else:
        verdict = "**No symbol produced a result.** See the failures below."
    report.text(verdict)

    if ranked:
        span_start = min(entry.run.walk_forward.predictions.index[0] for entry in ranked)
        span_end = max(entry.run.walk_forward.predictions.index[-1] for entry in ranked)
        span = (str(span_start.date()), str(span_end.date()))
        bars = max(len(entry.run.walk_forward.predictions) for entry in ranked)
    else:
        span, bars = ("n/a", "n/a"), 0

    report.heading("Setup").bullets(
        [
            f"Universe: {len(result.entries)} symbol(s), {result.n_tested} evaluated",
            f"Model: `{result.model_name}`, ranked by `{result.rank_by}`",
            *_setup_bullets(config, span, bars)[1:],
        ]
    )

    report.heading("Ranking")
    headers = ["symbol", *(header for _, header, _ in _SCREEN_COLUMNS)]
    rows = []
    for entry in ranked:
        metrics = entry.metrics()
        cells = []
        for key, _, percent in _SCREEN_COLUMNS:
            value = metrics.get(key)
            # A bar count is a count, not a measurement: render it as one.
            cells.append(
                f"{int(value):,}"
                if key == "n_bars" and value is not None and not pd.isna(value)
                else format_number(value, percent=percent)
            )
        rows.append([entry.symbol, *cells])
    report.table(headers, rows)

    if ranked:
        report.heading("Excess Sharpe vs buy-and-hold")
        report.code_block(
            ascii_bars(
                [entry.symbol for entry in ranked],
                [entry.metrics().get("excess_sharpe", float("nan")) for entry in ranked],
                width=48,
            )
        )

    if ranked:
        _portfolio_section(report, result)

    report.heading("The multiple-comparison correction")
    report.raw_table(
        markdown_table(
            ["quantity", "value"],
            [
                ["symbols tested", str(result.n_tested)],
                ["alpha", format_number(result.alpha)],
                ["Bonferroni threshold (per symbol)", format_number(result.bonferroni())],
                [
                    "false positives expected from noise alone",
                    format_number(result.expected_false_positives()),
                ],
                [
                    "symbols with raw p <= alpha",
                    str(sum(e.p_value <= result.alpha for e in ranked)),
                ],
                ["symbols with q <= alpha", str(sum(e.q_value <= result.alpha for e in ranked))],
            ],
        )
    )
    report.bullets(
        [
            "`p` is the raw one-sided permutation p-value for that symbol on its own.",
            "`q` is the same test adjusted across the screen (Benjamini-Hochberg). "
            "A q of 0.10 means roughly a tenth of everything you accept at that level "
            "is noise.",
            "Read `q`, not `p`. Ranking N symbols and quoting the winner's raw p-value "
            "is the arithmetic that produces most published trading edges.",
        ]
    )

    if result.failures:
        report.heading("Symbols that could not be evaluated")
        report.table(
            ["symbol", "bars", "reason"],
            [
                [e.symbol, str(e.n_bars), (e.error or "").split("\n")[0][:110]]
                for e in result.failures
            ],
        )

    report.heading("How to read this").bullets(
        [
            "**Start at `vs B&H`.** A negative value means the model lost to simply "
            "holding that stock; nothing else in the row can rescue it.",
            "**Then `q`.** A symbol clearing both is a candidate for further work, not a "
            "signal to trade.",
            "**Then the portfolio section.** Two symbols that each clear the bar are still "
            "one bet if they move together.",
            *_reading_notes()[:2],
        ]
    )
    report.heading("Caveats").bullets(
        [
            "Screening the same universe repeatedly with different models or windows "
            "multiplies the selection problem again, and the FDR correction here only "
            "covers the symbols in this one run.",
            "The portfolio section holds fixed weights over the whole sample and charges "
            "nothing to rebalance between sleeves. Correlations are also not stable: they "
            "rise in exactly the drawdowns the diversification was supposed to cushion, so "
            "the effective bet count here is a full-sample average, not a promise.",
            *_caveats(),
        ]
    )
    return report.render()


MIN_INDEPENDENT_BLOCKS = 10
"""Non-overlapping horizons the journal needs before any verdict is offered.

Ten blocks is not a power calculation - detecting a plausible edge would take
far more - it is the point below which the standard error is so wide that the
verdict would be describing its own noise.
"""


def _freshness_verdict(freshness: pd.DataFrame) -> str:
    """State whether the prices under this report are current."""
    stale = freshness[freshness["stale"]]
    if stale.empty:
        newest = freshness["age_days"].min()
        return f"Prices are current: every symbol's last bar is {int(newest)} day(s) old."

    unreadable = sorted(stale.index[~np.isfinite(stale["age_days"])])
    behind = stale[np.isfinite(stale["age_days"])]
    parts = []
    if not behind.empty:
        names = ", ".join(
            f"`{symbol}` ({int(row['age_days'])}d, last bar {row['last_bar']})"
            for symbol, row in behind.iterrows()
        )
        parts.append(f"behind: {names}")
    if unreadable:
        parts.append("unreadable: " + ", ".join(f"`{s}`" for s in unreadable))

    return (
        f"**The price data is not current** - {'; '.join(parts)}. "
        "Every number below describes the market as of those bars, not today, and it "
        "will keep describing them - unchanged and without complaint - for as long as "
        "the feed stays down. Check the downloader before reading the performance as "
        "a live result."
    )


def _decay_verdict(comparison: dict[str, float]) -> str:
    """State what the live record says about the backtest's claim."""
    n = comparison.get("n_scored", 0.0)
    blocks = comparison.get("n_independent", 0.0)
    z = comparison.get("hit_rate_z", float("nan"))
    naive = comparison.get("hit_rate_z_naive", float("nan"))
    if n < 30:
        return (
            f"Only {int(n)} forecast(s) have matured. That is far too few to say anything: "
            "the standard error on a hit rate this small swamps any plausible edge. "
            "Keep recording."
        )
    if blocks < MIN_INDEPENDENT_BLOCKS:
        return (
            f"{int(n)} forecasts have matured, but they cover only {int(blocks)} non-overlapping "
            f"horizon(s). Recorded daily against a {int(n)}-row journal that looks like a sample "
            "of hundreds, but a forecast shares almost all of its outcome window with the one "
            "before it, and the whole universe moves together on any given day. There is not yet "
            "enough independent information for a verdict. Keep recording."
        )
    if not np.isfinite(z):
        return (
            f"{int(n)} forecasts scored, but the backtest claim is unavailable to compare against."
        )
    divergence = ""
    if np.isfinite(naive) and (naive <= -2.0) and (z > -2.0):
        divergence = (
            " Counting every forecast as its own bet would put this at "
            f"{format_number(naive)} and trip the alarm; that reading double-counts overlapping "
            "windows, so it is watched rather than acted on."
        )
    if z <= -2.0:
        return (
            f"Live accuracy is {format_number(abs(z))} standard errors **below** the backtested "
            f"claim across {int(blocks)} independent horizons. That is decay, or a backtest that "
            "was overfitted to begin with. Re-examine before trusting the model further."
        )
    if z >= 2.0:
        return (
            f"Live accuracy is {format_number(z)} standard errors **above** the backtested claim "
            f"across {int(blocks)} independent horizons. Pleasant, but treat a large positive gap "
            "with the same suspicion as a negative one: it usually means the live and backtest "
            "setups differ."
        )
    return (
        f"Live accuracy sits within {format_number(abs(z))} standard errors of the backtested "
        f"claim across {int(blocks)} independent horizons - consistent with the backtest, no "
        f"decay detected.{divergence}"
    )


def render_journal_report(
    live: ScoreResult,
    config: ExperimentConfig,
    *,
    model_name: str,
    comparisons: dict[str, dict[str, float]] | None = None,
    rolling: pd.DataFrame | None = None,
    recorded: int = 0,
    skipped: list[str] | None = None,
    freshness: pd.DataFrame | None = None,
) -> str:
    """Report what the live forecast journal says, against what was promised.

    The journal is the only score that cannot be tuned after the fact, so this
    report leads with the live-versus-backtest gap rather than with P&L.

    ``freshness`` is :func:`ai_stock.journal.data_freshness` over the same
    universe. It is rendered above the performance tables, because a stale feed
    makes every number below it describe a day that has already passed.
    """
    metrics = live.metrics()
    report = Report(
        f"Forecast journal: `{model_name}`",
        subtitle="Predictions recorded before the outcome existed, scored once it arrived.",
    )

    pooled = (comparisons or {}).get("__all__", {})
    report.text(
        _decay_verdict(pooled) if pooled else _decay_verdict({"n_scored": metrics["n_scored"]})
    )

    if freshness is not None and not freshness.empty:
        report.text(_freshness_verdict(freshness))

    report.heading("Today's run").bullets(
        [
            f"Forecasts recorded: {recorded}",
            "Symbols skipped (too little history to fit): "
            + (", ".join(f"`{s}`" for s in skipped) if skipped else "none"),
            f"Forecasts matured and scored: {int(metrics['n_scored'])}",
            f"Still in flight (horizon not elapsed): {int(metrics['n_pending'])}",
            f"Forecast horizon: {config.features.horizon} trading day(s)",
            f"Costs: {config.backtest.total_cost_bps:g} bps per unit traded",
        ]
    )

    if freshness is not None and not freshness.empty:
        report.heading("Data freshness")
        report.table(
            ["symbol", "last bar", "age (days)", "behind?"],
            [
                [
                    str(symbol),
                    "-" if pd.isna(row["last_bar"]) else str(row["last_bar"]),
                    "-" if not np.isfinite(row["age_days"]) else f"{row['age_days']:.0f}",
                    "yes" if row["stale"] else "no",
                ]
                for symbol, row in freshness.iterrows()
            ],
        )
        report.bullets(
            [
                "`age (days)` is calendar days from the symbol's last bar to today, so a "
                "long market holiday reads as behind. That is the cheap direction to be "
                "wrong in: a needless glance at the feed costs nothing, a hit rate that "
                "quietly stopped moving costs the only untunable number here.",
                "A stopped feed does not make this report go quiet - it makes it repeat. "
                "The same forecasts mature against the same bars and the same hit rate "
                "comes back, which is why the age is stated before the performance.",
            ]
        )

    report.heading("Live performance")
    report.raw_table(
        metrics_table(
            {"live": metrics},
            keys=(
                "n_scored",
                "n_pending",
                "hit_rate",
                "live_ic",
                "live_ic_pooled",
                "mean_pnl",
                "total_pnl",
                "live_sharpe",
                "annual_turnover",
                "n_symbols",
                "span_days",
                "live_annual_turnover",
            ),
            label="metric",
        )
    )
    report.bullets(
        [
            "`live_ic` averages the per-symbol correlations, the same way `ic_fold_mean` "
            "averages per-fold ones. `live_ic_pooled` throws every symbol into one "
            "correlation and can carry the opposite sign - it is shown only for contrast.",
            "`live_sharpe` is a health check, not a tradable number: with a multi-day "
            "horizon the per-forecast returns overlap, so its standard error is understated.",
            "`live_annual_turnover` counts every recorded forecast, matured or not - a "
            "position pays for flipping the day it flips, not once its horizon elapses.",
            "`annual_turnover` is the same annualised traded-notional measure the backtest "
            "reports. Multiply it by the cost in bps to see the yearly cost drag this "
            "journal has actually paid - `total_pnl` already nets it out, but turnover is "
            "what makes that drag visible on its own.",
        ]
    )

    if comparisons:
        report.heading("Live vs. backtest")
        named = {k: v for k, v in comparisons.items() if k != "__all__"}
        if named:
            report.raw_table(
                metrics_table(
                    named,
                    keys=(
                        "n_scored",
                        "n_decided",
                        "n_independent",
                        "backtest_directional_accuracy",
                        "live_hit_rate",
                        "hit_rate_gap",
                        "hit_rate_z",
                        "hit_rate_z_naive",
                        "backtest_ic",
                        "live_ic",
                    ),
                    label="quantity",
                )
            )
        report.bullets(
            [
                "`hit_rate_z` is the live shortfall in units of its own standard error. "
                "Around zero means consistent with the backtest; below -2 means the "
                "backtest was promising something the live record is not delivering.",
                "`n_independent` is the sample size that z is computed at: the number of "
                "non-overlapping `horizon`-day windows the journal covers, counting every "
                "forecast inside a window - all symbols, all dates - as one observation. "
                "Forecasts recorded daily against a multi-day horizon overlap, and a "
                "single-sector universe moves together, so `n_decided` badly overstates how "
                "many independent bets have been placed.",
                "`hit_rate_z_naive` is the same shortfall at `n_decided` trials, which is what "
                "this report used to print. It assumes zero redundancy and `hit_rate_z` assumes "
                "total redundancy within a window, so the honest figure lies between them. They "
                "converge as the journal lengthens; while they disagree, believe the smaller.",
                "`n_decided` excludes forecasts that took no side - a zero position cannot be "
                "right or wrong, so it is not a trial.",
                "With few independent windows the z-score is near zero whatever happens. "
                "Read `n_independent` first.",
            ]
        )

    if rolling is not None and not rolling.empty:
        report.heading("Live vs. backtest over time")
        report.text(
            f"`hit_rate_z` over the trailing {int(rolling['n_scored'].iloc[0])} matured "
            "forecasts, ending at each date shown. A single pooled z-score cannot say "
            "*when* a gap opened; this can. A window that short spans only a handful of "
            "non-overlapping horizons, so read the shape of the line rather than whether "
            "any one point crosses -2."
        )
        dates = pd.to_datetime(rolling["asof_date"]).dt.date.astype(str)
        report.code_block(
            ascii_line_chart(
                {"hit_rate_z": pd.Series(rolling["hit_rate_z"].to_numpy(), index=dates)},
                width=72,
                height=12,
            )
        )
        columns = ["asof_date", "n_scored", "n_independent", "live_hit_rate", "hit_rate_z"]
        tail = rolling[columns].tail(10).copy()
        tail["asof_date"] = pd.to_datetime(tail["asof_date"]).dt.date.astype(str)
        report.dataframe(tail.reset_index(drop=True), index=False)

    per_symbol = live.by_symbol()
    if not per_symbol.empty:
        report.heading("By symbol")
        report.dataframe(per_symbol)
        report.code_block(
            ascii_bars(
                list(per_symbol.index),
                per_symbol["total_pnl"].tolist(),
                width=44,
            )
        )

    if not live.pending.empty:
        report.heading("In flight")
        columns = [c for c in ("asof_date", "symbol", "signal", "position") if c in live.pending]
        upcoming = live.pending[columns].tail(12).copy()
        if "asof_date" in upcoming:
            upcoming["asof_date"] = pd.to_datetime(upcoming["asof_date"]).dt.date.astype(str)
        report.dataframe(upcoming.reset_index(drop=True), index=False)

    report.heading("How to read this").bullets(
        [
            "This is the only score in the project that cannot be tuned after the fact - "
            "every row was written before its outcome existed.",
            "A live record that matches the backtest is evidence the pipeline is sound. "
            "It is still not evidence that the edge will persist.",
            "Costs here are charged on the change in position from the previous journal "
            "entry, the same convention the backtest uses.",
        ]
    )
    report.heading("Caveats").bullets(_caveats())
    return report.render()
