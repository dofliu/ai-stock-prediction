"""Composed reports for the three study types the CLI exposes.

Each renderer also writes down how to *read* the numbers, because a table of
Sharpe ratios with no context is how backtests get oversold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_stock.config import ExperimentConfig
from ai_stock.journal import ScoreResult
from ai_stock.pipeline import ModelRun, ScreenResult, SimulationBundle
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
)

_SUMMARY_KEYS = (
    "n",
    "n_folds",
    "ic_pearson",
    "ic_fold_mean",
    "ic_fold_std",
    "ic_fold_t",
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


def summarise_run_row(run: ModelRun) -> list[str]:
    """One ranking-table row for ``run``."""
    metrics = run.metrics()
    return [
        run.name,
        *(format_number(metrics.get(key), percent=percent) for key, _, percent in _RANKING_COLUMNS),
    ]


def _ranking_table(runs: list[ModelRun]) -> tuple[list[str], list[list[str]]]:
    headers = ["model", *(header for _, header, _ in _RANKING_COLUMNS)]
    return headers, [summarise_run_row(run) for run in runs]


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

    importance = walk_forward.feature_importance
    if importance is not None and not importance.empty:
        report.heading("Feature importance (mean across folds)")
        top = importance.reindex(importance.abs().sort_values(ascending=False).index).head(12)
        report.dataframe(top.to_frame("importance"))

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
    headers, rows = _ranking_table(runs)
    report.table(headers, rows)

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

    report.heading("How to read this").bullets(_reading_notes())
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
            *_reading_notes()[:2],
        ]
    )
    report.heading("Caveats").bullets(
        [
            "Screening the same universe repeatedly with different models or windows "
            "multiplies the selection problem again, and the FDR correction here only "
            "covers the symbols in this one run.",
            *_caveats(),
        ]
    )
    return report.render()


def _decay_verdict(comparison: dict[str, float]) -> str:
    """State what the live record says about the backtest's claim."""
    n = comparison.get("n_scored", 0.0)
    z = comparison.get("hit_rate_z", float("nan"))
    if n < 30:
        return (
            f"Only {int(n)} forecast(s) have matured. That is far too few to say anything: "
            "the standard error on a hit rate this small swamps any plausible edge. "
            "Keep recording."
        )
    if not np.isfinite(z):
        return (
            f"{int(n)} forecasts scored, but the backtest claim is unavailable to compare against."
        )
    if z <= -2.0:
        return (
            f"Live accuracy is {format_number(abs(z))} standard errors **below** the backtested "
            f"claim across {int(n)} forecasts. That is decay, or a backtest that was overfitted "
            "to begin with. Re-examine before trusting the model further."
        )
    if z >= 2.0:
        return (
            f"Live accuracy is {format_number(z)} standard errors **above** the backtested claim "
            f"across {int(n)} forecasts. Pleasant, but treat a large positive gap with the same "
            "suspicion as a negative one: it usually means the live and backtest setups differ."
        )
    return (
        f"Live accuracy sits within {format_number(abs(z))} standard errors of the backtested "
        f"claim across {int(n)} forecasts - consistent with the backtest, no decay detected."
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
) -> str:
    """Report what the live forecast journal says, against what was promised.

    The journal is the only score that cannot be tuned after the fact, so this
    report leads with the live-versus-backtest gap rather than with P&L.
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
                        "backtest_directional_accuracy",
                        "live_hit_rate",
                        "hit_rate_gap",
                        "hit_rate_z",
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
                "With few scored forecasts the z-score is near zero whatever happens. "
                "Read `n_scored` first.",
            ]
        )

    if rolling is not None and not rolling.empty:
        report.heading("Live vs. backtest over time")
        report.text(
            f"`hit_rate_z` over the trailing {int(rolling['n_scored'].iloc[0])} matured "
            "forecasts, ending at each date shown. A single pooled z-score cannot say "
            "*when* a gap opened; this can."
        )
        dates = pd.to_datetime(rolling["asof_date"]).dt.date.astype(str)
        report.code_block(
            ascii_line_chart(
                {"hit_rate_z": pd.Series(rolling["hit_rate_z"].to_numpy(), index=dates)},
                width=72,
                height=12,
            )
        )
        tail = rolling[["asof_date", "n_scored", "live_hit_rate", "hit_rate_z"]].tail(10).copy()
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
