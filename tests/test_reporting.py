"""Report primitives and composed study reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_stock.config import (
    BacktestConfig,
    ExperimentConfig,
    SimulationConfig,
    SyntheticConfig,
    WalkForwardConfig,
)
from ai_stock.pipeline import compare_models, run_model, run_simulation
from ai_stock.reporting.report import (
    Report,
    ascii_histogram,
    ascii_line_chart,
    format_number,
    markdown_table,
    metrics_table,
    sparkline,
)
from ai_stock.reporting.studies import (
    render_backtest_report,
    render_comparison_report,
    render_simulation_report,
)


@pytest.fixture(scope="module")
def report_config() -> ExperimentConfig:
    return ExperimentConfig(
        synthetic=SyntheticConfig(n_days=900, seed=1234),
        walk_forward=WalkForwardConfig(train_size=400, test_size=100),
        backtest=BacktestConfig(),
        simulation=SimulationConfig(n_paths=50, n_permutations=40, seed=0),
    )


@pytest.mark.parametrize(
    ("value", "kwargs", "expected"),
    [
        (0.12345, {}, "0.1235"),
        (0.1234, {"percent": True}, "12.34%"),
        (float("nan"), {}, "n/a"),
        (float("inf"), {}, "+inf"),
        (float("-inf"), {}, "-inf"),
        (None, {}, "n/a"),
        (12345.6, {}, "12,345.6"),
        ("text", {}, "text"),
    ],
)
def test_format_number(value, kwargs, expected: str) -> None:
    assert format_number(value, **kwargs) == expected


def test_sparkline_spans_the_block_range() -> None:
    assert sparkline([1, 2, 3, 4], width=4) == "▁▃▆█"
    assert sparkline([5, 5, 5], width=3) == "▁▁▁"
    assert sparkline([]) == ""


def test_sparkline_downsamples_to_the_requested_width() -> None:
    assert len(sparkline(np.arange(1000.0), width=40)) == 40


def test_markdown_table_pads_columns() -> None:
    rendered = markdown_table(["a", "bb"], [["1", "2"], ["333", "4"]])
    lines = rendered.splitlines()

    assert len(lines) == 4
    assert lines[0].startswith("| a   | bb |")
    assert set(lines[1]) <= {"|", "-"}


def test_metrics_table_renders_percentages_and_missing_values() -> None:
    rendered = metrics_table(
        {"a": {"sharpe": 0.5, "max_drawdown": -0.25}, "b": {"sharpe": float("nan")}}
    )
    assert "-25.00%" in rendered
    assert "n/a" in rendered
    assert metrics_table({}) == "_no metrics_"


def test_ascii_line_chart_has_axis_labels_and_a_legend() -> None:
    index = pd.RangeIndex(50)
    rendered = ascii_line_chart(
        {"one": pd.Series(np.arange(50.0), index=index)}, width=20, height=5
    )
    lines = rendered.splitlines()

    assert len(lines) == 5 + 2  # rows plus axis and legend
    assert "one" in lines[-1]
    assert ascii_line_chart({}) == "_no data_"


def test_ascii_histogram_flags_the_observed_value() -> None:
    rng = np.random.default_rng(0)
    rendered = ascii_histogram(rng.normal(0, 1, 200), bins=5, width=10, marker=0.0)
    assert "<== observed" in rendered

    outside = ascii_histogram(rng.normal(0, 1, 200), bins=5, width=10, marker=99.0)
    assert "above the null range" in outside
    assert ascii_histogram([]) == "_no data_"


def test_report_builder_composes_sections() -> None:
    report = Report("Title", subtitle="sub")
    report.heading("Section").text("body").bullets(["one", "two"])
    report.table(["h"], [["v"]]).code_block("code", "python")
    rendered = report.render()

    assert rendered.startswith("# Title")
    assert "_sub_" in rendered
    assert "## Section" in rendered
    assert "- one" in rendered
    assert "```python" in rendered


def test_report_renders_dataframes() -> None:
    frame = pd.DataFrame({"x": [1.5, 2.5]}, index=pd.Index(["a", "b"], name="key"))
    rendered = Report("t").dataframe(frame).render()

    assert "key" in rendered and "1.5000" in rendered
    assert "_empty table_" in Report("t").dataframe(pd.DataFrame()).render()


def test_backtest_report_contains_the_key_sections(ohlcv, report_config) -> None:
    run = run_model(ohlcv, "ridge", report_config)
    rendered = render_backtest_report(run, report_config)

    for section in (
        "# Walk-forward backtest",
        "## Setup",
        "## Predictive performance",
        "## Trading performance",
        "## Equity curve",
        "## Per-fold stability",
        "## Feature importance",
        "## Caveats",
    ):
        assert section in rendered
    assert "nothing here is investment advice" in rendered.lower()


def test_comparison_report_lists_every_model(ohlcv, report_config) -> None:
    runs = compare_models(ohlcv, ["zero", "momentum", "ridge"], report_config)
    rendered = render_comparison_report(runs, report_config)

    for name in ("zero", "momentum", "ridge"):
        assert name in rendered
    assert "## Ranking" in rendered
    assert "buy & hold" in rendered


def test_comparison_report_needs_at_least_one_run(report_config) -> None:
    with pytest.raises(ValueError, match="no runs"):
        render_comparison_report([], report_config)


def test_simulation_report_states_a_verdict(ohlcv, report_config) -> None:
    bundle = run_simulation(ohlcv, "ridge", report_config)
    rendered = render_simulation_report(bundle, report_config)

    assert "## Is the edge distinguishable from luck?" in rendered
    assert "p = " in rendered
    assert any(
        phrase in rendered
        for phrase in (
            "evidence against the luck hypothesis",
            "what luck looks like",
            "inconclusive",
        )
    )
    assert "## Forward distribution" in rendered
