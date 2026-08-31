"""Composed reports for the three study types the CLI exposes.

Each renderer also writes down how to *read* the numbers, because a table of
Sharpe ratios with no context is how backtests get oversold.
"""

from __future__ import annotations

import pandas as pd

from ai_stock.config import ExperimentConfig
from ai_stock.pipeline import ModelRun, SimulationBundle
from ai_stock.reporting.report import (
    Report,
    ascii_histogram,
    ascii_line_chart,
    format_number,
    metrics_table,
)

__all__ = [
    "render_backtest_report",
    "render_comparison_report",
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
