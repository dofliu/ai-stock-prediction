"""Command-line interface.

    ai-stock data      generate a synthetic market and write it to CSV
    ai-stock backtest  walk-forward evaluate one model and trade its forecasts
    ai-stock compare   rank several models over identical folds
    ai-stock simulate  Monte-Carlo and luck-vs-skill test for one model
    ai-stock screen    rank a universe of symbols, corrected for multiple testing
    ai-stock journal   record today's forecasts and score the ones that matured
    ai-stock models    list the available model names

Every command works on synthetic data by default, so the whole pipeline is
runnable with no data files and no network access.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ai_stock.config import (
    BacktestConfig,
    ExperimentConfig,
    FeatureConfig,
    SimulationConfig,
    SyntheticConfig,
    WalkForwardConfig,
)
from ai_stock.data.loaders import save_csv
from ai_stock.journal import (
    MIN_TRAIN_ROWS,
    append_forecasts,
    compare_with_backtest,
    load_journal,
    record_forecasts,
    rolling_compare_with_backtest,
    score_journal,
)
from ai_stock.models.registry import available_models
from ai_stock.pipeline import (
    compare_models,
    load_prices,
    load_universe,
    run_model,
    run_simulation,
    screen_universe,
)
from ai_stock.reporting.report import format_number, metrics_table
from ai_stock.reporting.studies import (
    render_backtest_report,
    render_comparison_report,
    render_journal_report,
    render_screen_report,
    render_simulation_report,
)

__all__ = ["build_parser", "main"]

_DEFAULT_COMPARE_MODELS = (
    "zero,train_mean,momentum,reversion,ridge,random_forest,gradient_boosting,logistic"
)


# --------------------------------------------------------------------------- #
# Argument plumbing
# --------------------------------------------------------------------------- #
def _data_options(parser: argparse.ArgumentParser, *, multi: bool = False) -> None:
    group = parser.add_argument_group("data")
    if multi:
        group.add_argument(
            "--data",
            type=Path,
            nargs="+",
            metavar="PATH",
            help="OHLCV CSV files, or directories whose *.csv files are all loaded",
        )
        group.add_argument(
            "--tickers",
            help="comma-separated symbols to download via yfinance, e.g. MU,2408.TW",
        )
        group.add_argument("--period", default="12y", help="history length per ticker")
        group.add_argument(
            "--cache-dir",
            type=Path,
            default=None,
            help="write downloads here and reuse them, so the screen can be repeated offline",
        )
    else:
        group.add_argument("--data", type=Path, help="OHLCV CSV file; omit to use synthetic data")
    group.add_argument("--days", type=int, default=2500, help="synthetic trading days")
    group.add_argument("--seed", type=int, default=42, help="synthetic data seed")
    group.add_argument("--start", default="2010-01-04", help="synthetic start date")
    group.add_argument(
        "--initial-price", type=float, default=100.0, help="synthetic starting price"
    )
    group.add_argument(
        "--ar1", type=float, default=None, help="synthetic AR(1) coefficient on daily returns"
    )
    group.add_argument(
        "--reversion", type=float, default=None, help="synthetic 5-day mean-reversion loading"
    )
    group.add_argument(
        "--efficient",
        action="store_true",
        help="generate an unpredictable market (ar1 = reversion = 0) to test for false positives",
    )


def _feature_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("features")
    group.add_argument("--horizon", type=int, default=1, help="forecast horizon in trading days")
    group.add_argument(
        "--neutral-band",
        type=float,
        default=0.0,
        help="drop classification labels whose |forward return| is below this",
    )
    group.add_argument(
        "--no-volume-features", action="store_true", help="exclude volume-derived features"
    )


def _walk_forward_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("walk-forward")
    group.add_argument("--train-size", type=int, default=750, help="training bars per fold")
    group.add_argument("--test-size", type=int, default=125, help="test bars per fold")
    group.add_argument("--step", type=int, default=None, help="fold advance (default: test size)")
    group.add_argument(
        "--rolling",
        action="store_true",
        help="use a fixed-length rolling window instead of an expanding one",
    )
    group.add_argument(
        "--embargo",
        type=int,
        default=None,
        help="bars skipped between train and test (default: the forecast horizon)",
    )


def _backtest_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("backtest")
    group.add_argument(
        "--sizing",
        choices=("sign", "long_only", "proportional"),
        default="sign",
        help="how a signal becomes a position",
    )
    group.add_argument("--threshold", type=float, default=0.0, help="signal dead band")
    group.add_argument("--max-leverage", type=float, default=1.0, help="position cap")
    group.add_argument(
        "--scale", type=float, default=100.0, help="signal multiplier for proportional sizing"
    )
    group.add_argument("--cost-bps", type=float, default=1.0, help="commission in basis points")
    group.add_argument("--slippage-bps", type=float, default=1.0, help="slippage in basis points")
    group.add_argument("--no-short", action="store_true", help="disallow short positions")
    group.add_argument(
        "--vol-target",
        type=float,
        default=None,
        help="annualised volatility target for position scaling",
    )


def _output_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")
    group.add_argument("--out", type=Path, default=None, help="directory for reports and CSVs")
    group.add_argument("--quiet", action="store_true", help="suppress the stdout summary")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-stock",
        description="AI stock prediction and strategy simulation toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Educational research tooling. Not investment advice.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="generate synthetic OHLCV data")
    _data_options(data)
    data.add_argument(
        "--out", type=Path, default=Path("data/synthetic.csv"), help="output CSV path"
    )
    data.add_argument("--quiet", action="store_true", help="suppress the stdout summary")

    backtest = subparsers.add_parser(
        "backtest", help="walk-forward evaluate one model and trade its forecasts"
    )
    backtest.add_argument("--model", default="ridge", help="model name (see 'ai-stock models')")
    for add_options in (
        _data_options,
        _feature_options,
        _walk_forward_options,
        _backtest_options,
        _output_options,
    ):
        add_options(backtest)

    compare = subparsers.add_parser("compare", help="rank several models over identical folds")
    compare.add_argument(
        "--models", default=_DEFAULT_COMPARE_MODELS, help="comma-separated model names"
    )
    compare.add_argument("--sort-by", default="sharpe", help="metric used to rank the models")
    for add_options in (
        _data_options,
        _feature_options,
        _walk_forward_options,
        _backtest_options,
        _output_options,
    ):
        add_options(compare)

    simulate = subparsers.add_parser(
        "simulate", help="Monte-Carlo and luck-vs-skill test for one model"
    )
    simulate.add_argument("--model", default="ridge", help="model to simulate")
    simulate.add_argument("--paths", type=int, default=1000, help="Monte-Carlo paths")
    simulate.add_argument(
        "--sim-horizon", type=int, default=252, help="forward simulation horizon in days"
    )
    simulate.add_argument(
        "--sim-method",
        choices=("block_bootstrap", "iid_bootstrap", "gaussian"),
        default="block_bootstrap",
        help="resampling scheme for the forward paths",
    )
    simulate.add_argument("--block-size", type=int, default=20, help="bootstrap block length")
    simulate.add_argument(
        "--permutations", type=int, default=500, help="null draws for the significance test"
    )
    simulate.add_argument(
        "--null-method",
        choices=("rotation", "shuffle"),
        default="rotation",
        help="how the null breaks the signal/return alignment",
    )
    simulate.add_argument("--sim-seed", type=int, default=7, help="simulation seed")
    for add_options in (
        _data_options,
        _feature_options,
        _walk_forward_options,
        _backtest_options,
        _output_options,
    ):
        add_options(simulate)

    screen = subparsers.add_parser(
        "screen", help="rank a universe of symbols by how well one model trades each"
    )
    screen.add_argument("--model", default="random_forest", help="model applied to every symbol")
    screen.add_argument(
        "--rank-by", default="excess_sharpe", help="metric used to order the ranking"
    )
    screen.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="null draws per symbol for the luck test; 0 skips it",
    )
    screen.add_argument(
        "--alpha", type=float, default=0.05, help="significance level after FDR correction"
    )
    screen.add_argument(
        "--null-method",
        choices=("rotation", "shuffle"),
        default="rotation",
        help="how the null breaks the signal/return alignment",
    )
    _data_options(screen, multi=True)
    for add_options in (
        _feature_options,
        _walk_forward_options,
        _backtest_options,
        _output_options,
    ):
        add_options(screen)

    journal = subparsers.add_parser(
        "journal",
        help="record today's forecasts and score the ones whose horizon has elapsed",
    )
    journal.add_argument("--model", default="random_forest", help="model used for forecasts")
    journal.add_argument(
        "--journal",
        type=Path,
        default=Path("data/journal/forecasts.csv"),
        help="append-only CSV holding the forecast journal",
    )
    journal.add_argument(
        "--min-train-rows",
        type=int,
        default=MIN_TRAIN_ROWS,
        help="labelled bars a symbol needs before its forecast is recorded",
    )
    journal.add_argument(
        "--skip-record", action="store_true", help="score only, do not add today's forecasts"
    )
    journal.add_argument(
        "--skip-score", action="store_true", help="record only, do not score matured forecasts"
    )
    journal.add_argument(
        "--no-compare",
        action="store_true",
        help="skip the walk-forward that produces the backtest claim to compare against",
    )
    journal.add_argument(
        "--rolling-window",
        type=int,
        default=30,
        help="matured forecasts per point when tracking hit_rate_z over time",
    )
    _data_options(journal, multi=True)
    for add_options in (
        _feature_options,
        _walk_forward_options,
        _backtest_options,
        _output_options,
    ):
        add_options(journal)

    subparsers.add_parser("models", help="list the available model names")
    return parser


# --------------------------------------------------------------------------- #
# Config assembly
# --------------------------------------------------------------------------- #
def _synthetic_config(args: argparse.Namespace) -> SyntheticConfig:
    config = SyntheticConfig(
        n_days=args.days,
        seed=args.seed,
        start=args.start,
        initial_price=args.initial_price,
    )
    if args.efficient:
        config = replace(config, ar1=0.0, reversion=0.0)
    if args.ar1 is not None:
        config = replace(config, ar1=args.ar1)
    if args.reversion is not None:
        config = replace(config, reversion=args.reversion)
    return config


def _experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        synthetic=_synthetic_config(args),
        features=FeatureConfig(
            horizon=args.horizon,
            neutral_band=args.neutral_band,
            include_volume_features=not args.no_volume_features,
        ),
        walk_forward=WalkForwardConfig(
            train_size=args.train_size,
            test_size=args.test_size,
            step=args.step,
            expanding=not args.rolling,
            embargo=args.embargo,
        ),
        backtest=BacktestConfig(
            sizing=args.sizing,
            threshold=args.threshold,
            max_leverage=args.max_leverage,
            scale=args.scale,
            cost_bps=args.cost_bps,
            slippage_bps=args.slippage_bps,
            allow_short=not args.no_short,
            vol_target=args.vol_target,
        ),
    )


def _simulation_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        n_paths=args.paths,
        horizon_days=args.sim_horizon,
        method=args.sim_method,
        block_size=args.block_size,
        n_permutations=args.permutations,
        seed=args.sim_seed,
    )


def _model_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("--models did not contain any model name")
    unknown = [name for name in names if name not in available_models()]
    if unknown:
        raise ValueError(
            f"unknown model(s): {', '.join(unknown)}; available: {', '.join(available_models())}"
        )
    return names


def _horizon_label(days: int) -> str:
    """Human-readable span for a simulation horizon in trading days.

    >>> _horizon_label(5), _horizon_label(21), _horizon_label(252)
    ('1w', '1mo', '1y')
    """
    for length, label in ((252, "y"), (21, "mo"), (5, "w")):
        if days % length == 0:
            count = days // length
            return f"{count}{label}"
    return f"{days}d"


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _echo(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def _command_data(args: argparse.Namespace) -> int:
    config = _synthetic_config(args)
    ohlcv = load_prices(None, config)
    path = save_csv(ohlcv, args.out)

    _echo(
        "\n".join(
            [
                f"wrote {len(ohlcv)} bars to {path}",
                f"  span            {ohlcv.index[0].date()} .. {ohlcv.index[-1].date()}",
                f"  close range     {ohlcv['close'].min():.2f} .. {ohlcv['close'].max():.2f}",
                f"  planted edge    ar1={config.ar1:g}, reversion={config.reversion:g}"
                + (
                    "  (efficient market: no learnable edge)"
                    if config.ar1 == config.reversion == 0
                    else ""
                ),
            ]
        ),
        quiet=args.quiet,
    )
    return 0


def _command_backtest(args: argparse.Namespace) -> int:
    config = _experiment_config(args)
    ohlcv = load_prices(args.data, config.synthetic)
    run = run_model(ohlcv, args.model, config)
    metrics = run.metrics()
    trading = run.backtest.metrics

    report = render_backtest_report(run, config)
    if args.out:
        _write(args.out / f"backtest_{args.model}.md", report)
        run.walk_forward.predictions.to_frame("signal").join(run.backtest.positions).to_csv(
            args.out / f"signals_{args.model}.csv"
        )
        run.backtest.equity.to_frame("equity").join(run.backtest.benchmark_equity).to_csv(
            args.out / f"equity_{args.model}.csv"
        )
        _echo(f"wrote report to {args.out / f'backtest_{args.model}.md'}", quiet=args.quiet)

    _echo(
        "\n".join(
            [
                f"model            {run.name}",
                f"out-of-sample    {len(run.walk_forward)} bars over "
                f"{len(run.walk_forward.folds)} folds",
                f"fold IC          {format_number(metrics['ic_fold_mean'])} "
                f"(t = {format_number(metrics['ic_fold_t'])})",
                f"Sharpe           {format_number(trading['sharpe'])} "
                f"vs buy & hold {format_number(run.backtest.benchmark_metrics['sharpe'])}",
                f"annual return    {format_number(trading['annualised_return'], percent=True)}",
                f"max drawdown     {format_number(trading['max_drawdown'], percent=True)}",
                f"annual turnover  {format_number(trading['annual_turnover'])}x",
            ]
        ),
        quiet=args.quiet,
    )
    return 0


def _command_compare(args: argparse.Namespace) -> int:
    config = _experiment_config(args)
    names = _model_names(args.models)
    ohlcv = load_prices(args.data, config.synthetic)
    runs = compare_models(ohlcv, names, config, sort_by=args.sort_by)

    report = render_comparison_report(runs, config)
    if args.out:
        _write(args.out / "comparison.md", report)
        frame = pd.DataFrame({run.name: run.metrics() for run in runs}).T
        frame.index.name = "model"
        frame.to_csv(args.out / "comparison.csv")
        _echo(f"wrote report to {args.out / 'comparison.md'}", quiet=args.quiet)

    _echo(
        metrics_table(
            {run.name: run.metrics() for run in runs},
            keys=(
                "ic_fold_mean",
                "ic_fold_t",
                "directional_accuracy",
                "sharpe",
                "annualised_return",
                "max_drawdown",
                "annual_turnover",
                "excess_sharpe",
            ),
        ),
        quiet=args.quiet,
    )
    return 0


def _command_simulate(args: argparse.Namespace) -> int:
    config = _experiment_config(args)
    ohlcv = load_prices(args.data, config.synthetic)
    bundle = run_simulation(
        ohlcv,
        args.model,
        config,
        simulation=_simulation_config(args),
        significance_method=args.null_method,
    )

    report = render_simulation_report(bundle, config)
    if args.out:
        _write(args.out / f"simulation_{args.model}.md", report)
        _echo(f"wrote report to {args.out / f'simulation_{args.model}.md'}", quiet=args.quiet)

    significance = bundle.significance
    forward = bundle.forward_paths.summary()
    ci_low, ci_high = bundle.sharpe_bootstrap.confidence_interval
    # Label the projection with the horizon actually simulated: calling a
    # 5-day figure "1y" invites a reader to act on a number 50x too small.
    span = _horizon_label(bundle.forward_paths.config.horizon_days)
    _echo(
        "\n".join(
            [
                f"model              {bundle.run.name}",
                f"realised Sharpe    {format_number(bundle.run.backtest.metrics['sharpe'])}",
                f"null ({significance.method})    mean {format_number(significance.null_mean)}, "
                f"95th pct {format_number(significance.summary()['null_q95'])}",
                f"p-value            {format_number(significance.p_value)}",
                f"Sharpe 90% CI      [{format_number(ci_low)}, {format_number(ci_high)}]",
                f"{span} median return".ljust(19)
                + f"{format_number(forward['median_terminal_return'], percent=True)}",
                f"{span} 5% quantile".ljust(19)
                + f"{format_number(forward['terminal_return_q05'], percent=True)}",
                f"{span} prob. of loss".ljust(19)
                + f"{format_number(bundle.forward_paths.probability_of_loss, percent=True)}",
            ]
        ),
        quiet=args.quiet,
    )
    return 0


def _command_screen(args: argparse.Namespace) -> int:
    config = _experiment_config(args)
    tickers = [t.strip() for t in (args.tickers or "").split(",") if t.strip()]
    if not args.data and not tickers:
        raise ValueError("screen needs --data and/or --tickers")

    universe = load_universe(
        data_paths=args.data,
        tickers=tickers or None,
        period=args.period,
        cache_dir=args.cache_dir,
    )
    result = screen_universe(
        universe,
        args.model,
        config,
        rank_by=args.rank_by,
        permutations=args.permutations,
        alpha=args.alpha,
        significance_method=args.null_method,
    )

    if args.out:
        _write(args.out / f"screen_{args.model}.md", render_screen_report(result, config))
        table = result.table()
        if not table.empty:
            table.to_csv(args.out / f"screen_{args.model}.csv")
        _echo(f"wrote report to {args.out / f'screen_{args.model}.md'}", quiet=args.quiet)

    table = result.table()
    lines = []
    if not table.empty:
        lines.append(
            metrics_table(
                {entry.symbol: entry.metrics() for entry in result.ranked},
                keys=(
                    "ic_fold_mean",
                    "ic_fold_t",
                    "directional_accuracy",
                    "sharpe",
                    "benchmark_sharpe",
                    "excess_sharpe",
                    "annual_turnover",
                    "p_value",
                    "q_value",
                ),
                label="metric",
            )
        )
    if len(result.ranked) > 1:
        try:
            portfolio = result.portfolio().metrics()
        except ValueError as error:
            lines.append(f"\nno portfolio view: {error}")
        else:
            lines.append(
                f"\nheld together   {int(portfolio['n_sleeves'])} sleeves, equally weighted, "
                f"correlated {format_number(portfolio['mean_correlation'], digits=2)} "
                f"-> {format_number(portfolio['effective_bets'], digits=2)} effective bet(s)"
                f"\nthe shares      correlated "
                f"{format_number(portfolio['mean_asset_correlation'], digits=2)} "
                f"-> {format_number(portfolio['asset_effective_bets'], digits=2)} "
                "if you just held them"
                f"\nportfolio       Sharpe {format_number(portfolio['sharpe'])} "
                f"vs {format_number(portfolio['sharpe_if_independent'])} "
                "if the sleeves were independent"
            )

    survivors = result.survivors()
    lines.append(
        f"\n{result.n_tested} symbol(s) tested · "
        f"noise alone would flag {format_number(result.expected_false_positives())} · "
        + (
            "survivors (beat buy & hold and q <= "
            f"{args.alpha:g}): {', '.join(e.symbol for e in survivors)}"
            if survivors
            else f"no symbol both beat buy & hold and survived q <= {args.alpha:g}"
        )
    )
    for entry in result.failures:
        lines.append(f"skipped {entry.symbol}: {(entry.error or '').splitlines()[0]}")
    _echo("\n".join(lines), quiet=args.quiet)
    return 0


def _command_journal(args: argparse.Namespace) -> int:
    config = _experiment_config(args)
    tickers = [t.strip() for t in (args.tickers or "").split(",") if t.strip()]
    if not args.data and not tickers:
        raise ValueError("journal needs --data and/or --tickers")

    universe = load_universe(
        data_paths=args.data,
        tickers=tickers or None,
        period=args.period,
        cache_dir=args.cache_dir,
    )

    recorded: list = []
    skipped: list[str] = []
    if not args.skip_record:
        recorded = record_forecasts(
            universe, args.model, config, min_train_rows=args.min_train_rows
        )
        append_forecasts(args.journal, recorded)
        skipped = sorted(set(universe) - {f.symbol for f in recorded})

    live = score_journal(load_journal(args.journal), universe, config)

    comparisons: dict[str, dict[str, float]] = {}
    rolling = None
    if not args.skip_score and not args.no_compare and len(live) > 0:
        claims: list[dict[str, float]] = []
        for symbol in sorted(live.scored["symbol"].unique()):
            try:
                claim = run_model(universe[symbol], args.model, config).metrics()
            except (ValueError, KeyError, RuntimeError):
                continue
            claims.append(claim)
            symbol_only = replace(live, scored=live.scored[live.scored["symbol"] == symbol])
            comparisons[symbol] = compare_with_backtest(symbol_only, claim)
        if claims:
            pooled = {
                "directional_accuracy": float(
                    np.mean([c.get("directional_accuracy", np.nan) for c in claims])
                ),
                "ic_fold_mean": float(np.mean([c.get("ic_fold_mean", np.nan) for c in claims])),
            }
            comparisons["__all__"] = compare_with_backtest(live, pooled)
            rolling = rolling_compare_with_backtest(live, pooled, window=args.rolling_window)

    if args.out:
        report = render_journal_report(
            live,
            config,
            model_name=args.model,
            comparisons=comparisons or None,
            rolling=rolling,
            recorded=len(recorded),
            skipped=skipped,
        )
        _write(args.out / f"journal_{args.model}.md", report)
        if not live.scored.empty:
            live.scored.to_csv(args.out / f"journal_scored_{args.model}.csv", index=False)
        _echo(f"wrote report to {args.out / f'journal_{args.model}.md'}", quiet=args.quiet)

    metrics = live.metrics()
    lines = [
        f"journal            {args.journal}",
        f"recorded today     {len(recorded)}"
        + (f"  (skipped: {', '.join(skipped)})" if skipped else ""),
        f"scored / pending   {int(metrics['n_scored'])} / {int(metrics['n_pending'])}",
        f"live hit rate      {format_number(metrics['hit_rate'], percent=True)}",
        f"live IC            {format_number(metrics['live_ic'])}",
        f"total P&L          {format_number(metrics['total_pnl'], percent=True)}",
        f"annual turnover    {format_number(metrics['annual_turnover'])}",
    ]
    pooled = comparisons.get("__all__")
    if pooled:
        claim = format_number(pooled["backtest_directional_accuracy"], percent=True)
        actual = format_number(pooled["live_hit_rate"], percent=True)
        lines.append(
            f"vs backtest        claim {claim} -> live {actual}"
            f"  (z = {format_number(pooled['hit_rate_z'])}"
            f" over {int(pooled['n_independent'])} independent horizon(s);"
            f" naive z = {format_number(pooled['hit_rate_z_naive'])})"
        )
    _echo("\n".join(lines), quiet=args.quiet)
    return 0


def _command_models(args: argparse.Namespace) -> int:
    del args
    print("\n".join(available_models()))
    return 0


_COMMANDS = {
    "data": _command_data,
    "backtest": _command_backtest,
    "compare": _command_compare,
    "simulate": _command_simulate,
    "screen": _command_screen,
    "journal": _command_journal,
    "models": _command_models,
}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (ValueError, FileNotFoundError, TypeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
