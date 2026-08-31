"""CLI behaviour: exit codes, generated artefacts and error reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai_stock.cli import _horizon_label, build_parser, main
from ai_stock.data.loaders import load_csv
from ai_stock.models.registry import available_models

SMALL = ["--days", "700", "--seed", "5", "--train-size", "400", "--test-size", "100"]


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "prices.csv"
    assert main(["data", "--days", "700", "--seed", "5", "--out", str(path), "--quiet"]) == 0
    return path


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_models_command_lists_the_registry(capsys) -> None:
    assert main(["models"]) == 0
    listed = capsys.readouterr().out.split()
    assert listed == list(available_models())


def test_data_command_writes_a_loadable_csv(tmp_path: Path, capsys) -> None:
    path = tmp_path / "nested" / "synthetic.csv"
    assert main(["data", "--days", "300", "--seed", "1", "--out", str(path)]) == 0

    frame = load_csv(path)
    assert len(frame) == 300
    output = capsys.readouterr().out
    assert "wrote 300 bars" in output
    assert "planted edge" in output


def test_data_command_can_generate_an_efficient_market(tmp_path: Path, capsys) -> None:
    path = tmp_path / "efficient.csv"
    assert main(["data", "--days", "300", "--efficient", "--out", str(path)]) == 0
    assert "no learnable edge" in capsys.readouterr().out


def test_data_command_honours_explicit_edge_overrides(tmp_path: Path, capsys) -> None:
    path = tmp_path / "custom.csv"
    assert (
        main(["data", "--days", "300", "--ar1", "0.2", "--reversion", "0", "--out", str(path)]) == 0
    )
    assert "ar1=0.2" in capsys.readouterr().out


def test_quiet_suppresses_output(tmp_path: Path, capsys) -> None:
    main(["data", "--days", "300", "--out", str(tmp_path / "a.csv"), "--quiet"])
    assert capsys.readouterr().out == ""


def test_backtest_command_prints_a_summary(data_file: Path, capsys) -> None:
    assert main(["backtest", "--data", str(data_file), "--model", "ridge", *SMALL[4:]]) == 0
    output = capsys.readouterr().out

    for label in ("model", "fold IC", "Sharpe", "annual return", "max drawdown"):
        assert label in output


def test_backtest_command_writes_reports_and_csvs(data_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    assert (
        main(
            [
                "backtest",
                "--data",
                str(data_file),
                "--model",
                "ridge",
                "--out",
                str(out),
                "--quiet",
                *SMALL[4:],
            ]
        )
        == 0
    )

    report = out / "backtest_ridge.md"
    assert report.exists() and report.read_text().startswith("# Walk-forward backtest")
    signals = pd.read_csv(out / "signals_ridge.csv")
    assert {"signal", "position"} <= set(signals.columns)
    equity = pd.read_csv(out / "equity_ridge.csv")
    assert {"equity", "benchmark_equity"} <= set(equity.columns)


def test_compare_command_ranks_models(tmp_path: Path, capsys) -> None:
    out = tmp_path / "reports"
    assert main(["compare", "--models", "zero,momentum,ridge", "--out", str(out), *SMALL]) == 0

    assert (out / "comparison.md").exists()
    frame = pd.read_csv(out / "comparison.csv", index_col="model")
    assert set(frame.index) == {"zero", "momentum", "ridge"}
    assert "sharpe" in frame.columns
    assert "ic_fold_mean" in capsys.readouterr().out


def test_compare_command_accepts_an_alternative_sort_key(tmp_path: Path) -> None:
    assert (
        main(
            [
                "compare",
                "--models",
                "momentum,reversion",
                "--sort-by",
                "ic_fold_mean",
                "--quiet",
                *SMALL,
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    ("days", "expected"),
    [(1, "1d"), (3, "3d"), (5, "1w"), (10, "2w"), (21, "1mo"), (63, "3mo"), (252, "1y")],
)
def test_horizon_label_matches_the_span(days: int, expected: str) -> None:
    assert _horizon_label(days) == expected


def test_simulate_summary_is_labelled_with_the_simulated_horizon(capsys) -> None:
    """A 5-day projection must not be reported as a 1-year one."""
    assert (
        main(
            [
                "simulate",
                "--model",
                "ridge",
                "--sim-horizon",
                "5",
                "--paths",
                "50",
                "--permutations",
                "20",
                *SMALL,
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "1w median return" in output
    assert "1w 5% quantile" in output
    assert "1w prob. of loss" in output
    assert "1y" not in output


def test_simulate_command_reports_significance(tmp_path: Path, capsys) -> None:
    out = tmp_path / "reports"
    assert (
        main(
            [
                "simulate",
                "--model",
                "ridge",
                "--paths",
                "50",
                "--permutations",
                "40",
                "--out",
                str(out),
                *SMALL,
            ]
        )
        == 0
    )

    assert (out / "simulation_ridge.md").exists()
    output = capsys.readouterr().out
    for label in ("realised Sharpe", "p-value", "Sharpe 90% CI", "prob. of loss"):
        assert label in output
    # Default --sim-horizon is 252 trading days.
    assert "1y median return" in output


def test_backtest_options_change_the_result(tmp_path: Path, capsys) -> None:
    base = ["backtest", "--model", "ridge", *SMALL]
    assert main([*base, "--sizing", "long_only", "--no-short"]) == 0
    long_only = capsys.readouterr().out
    assert main([*base, "--cost-bps", "50", "--slippage-bps", "50"]) == 0
    expensive = capsys.readouterr().out

    assert long_only != expensive


def test_rolling_window_and_horizon_options_are_accepted() -> None:
    assert (
        main(["backtest", "--model", "ridge", "--rolling", "--horizon", "3", "--quiet", *SMALL])
        == 0
    )


def test_screen_command_ranks_a_directory_of_csvs(tmp_path: Path, capsys) -> None:
    from dataclasses import replace as _replace

    from ai_stock.config import SyntheticConfig
    from ai_stock.data.loaders import save_csv
    from ai_stock.data.synthetic import generate_ohlcv

    folder = tmp_path / "universe"
    base = SyntheticConfig(n_days=700, seed=5)
    for symbol, seed in (("AAA", 11), ("BBB", 22)):
        save_csv(generate_ohlcv(_replace(base, seed=seed)), folder / f"{symbol}.csv")

    out = tmp_path / "reports"
    assert (
        main(
            [
                "screen",
                "--data",
                str(folder),
                "--model",
                "ridge",
                "--permutations",
                "20",
                "--train-size",
                "400",
                "--test-size",
                "100",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    report = out / "screen_ridge.md"
    assert report.exists() and report.read_text().startswith("# Universe screen")
    table = pd.read_csv(out / "screen_ridge.csv", index_col="symbol")
    assert set(table.index) == {"AAA", "BBB"}
    assert {"excess_sharpe", "p_value", "q_value"} <= set(table.columns)

    output = capsys.readouterr().out
    assert "symbol(s) tested" in output
    assert "noise alone would flag" in output


def test_screen_command_accepts_tickers_via_the_loader(tmp_path: Path, monkeypatch) -> None:
    """The --tickers path is exercised without touching the network."""
    from ai_stock.data.synthetic import generate_ohlcv

    frame = generate_ohlcv(n_days=700, seed=3)
    monkeypatch.setattr("ai_stock.pipeline.load_yfinance", lambda t, period="12y": frame)

    cache = tmp_path / "cache"
    assert (
        main(
            [
                "screen",
                "--tickers",
                "MU,2408.TW",
                "--cache-dir",
                str(cache),
                "--model",
                "ridge",
                "--permutations",
                "0",
                "--train-size",
                "400",
                "--test-size",
                "100",
                "--quiet",
            ]
        )
        == 0
    )
    assert (cache / "MU.csv").exists()
    assert (cache / "2408.TW.csv").exists()


def test_screen_without_a_source_exits_with_code_two(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["screen", "--model", "ridge"])
    assert excinfo.value.code == 2
    assert "--data and/or --tickers" in capsys.readouterr().err


def test_unknown_model_exits_with_code_two(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["compare", "--models", "not_a_model", *SMALL])
    assert excinfo.value.code == 2
    assert "unknown model" in capsys.readouterr().err


def test_missing_data_file_exits_with_code_two(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["backtest", "--data", str(tmp_path / "absent.csv"), "--model", "ridge"])
    assert excinfo.value.code == 2
    assert "no such data file" in capsys.readouterr().err


def test_too_little_data_exits_with_code_two(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["backtest", "--days", "300", "--train-size", "5000", "--model", "ridge"])
    assert excinfo.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_invalid_sizing_is_caught_by_argparse() -> None:
    with pytest.raises(SystemExit):
        main(["backtest", "--sizing", "martingale"])
