"""CLI behaviour: exit codes, generated artefacts and error reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai_stock.cli import build_parser, main
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
