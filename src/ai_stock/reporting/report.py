"""Markdown and plain-text report rendering.

Deliberately dependency-free: charts are drawn with block characters so a
report is readable in a terminal, a pull request and a Markdown viewer alike,
with matplotlib remaining strictly optional.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Report",
    "ascii_histogram",
    "ascii_line_chart",
    "format_number",
    "markdown_table",
    "metrics_table",
    "sparkline",
]

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"

_PERCENT_METRICS = frozenset(
    {
        "accuracy",
        "annualised_return",
        "annualised_volatility",
        "avg_exposure",
        "base_rate",
        "conditional_value_at_risk_95",
        "directional_accuracy",
        "hit_rate",
        "ic_fold_positive_rate",
        "max_drawdown",
        "median_max_drawdown",
        "probability_of_loss",
        "time_in_market",
        "total_return",
        "value_at_risk_95",
        "worst_max_drawdown",
    }
)


def format_number(value: object, *, digits: int = 4, percent: bool = False) -> str:
    """Format a metric for display, keeping ``NaN``/``inf`` legible.

    >>> format_number(0.12345)
    '0.1235'
    >>> format_number(0.1234, percent=True)
    '12.34%'
    >>> format_number(float("nan"))
    'n/a'
    """
    if isinstance(value, str):
        return value
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    if percent:
        return f"{number * 100:.2f}%"
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    return f"{number:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Render a GitHub-flavoured Markdown table with aligned columns."""
    body = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in body:
        for i, cell in enumerate(row[: len(widths)]):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        padded = [str(cell).ljust(widths[i]) for i, cell in enumerate(cells[: len(widths)])]
        return "| " + " | ".join(padded) + " |"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(headers), separator, *(line(row) for row in body)])


def metrics_table(
    named_metrics: dict[str, dict[str, float]],
    *,
    keys: Sequence[str] | None = None,
    label: str = "metric",
    percent_keys: Iterable[str] | None = None,
) -> str:
    """Render one column per run, one row per metric.

    Metrics known to be fractions (returns, drawdowns, hit rates) are shown as
    percentages so the table can be read without mental arithmetic.
    """
    if not named_metrics:
        return "_no metrics_"
    if keys is None:
        seen: dict[str, None] = {}
        for metrics in named_metrics.values():
            for key in metrics:
                seen.setdefault(key, None)
        keys = list(seen)

    percent = _PERCENT_METRICS if percent_keys is None else frozenset(percent_keys)
    headers = [label, *named_metrics.keys()]
    rows = [
        [
            key,
            *(
                format_number(metrics.get(key), percent=key in percent)
                for metrics in named_metrics.values()
            ),
        ]
        for key in keys
    ]
    return markdown_table(headers, rows)


def _downsample(values: np.ndarray, width: int) -> np.ndarray:
    """Reduce ``values`` to ``width`` points, keeping the last of each bucket."""
    n = len(values)
    if n <= width:
        return values
    edges = np.linspace(0, n, width + 1).astype(int)
    buckets = zip(edges[:-1], edges[1:], strict=True)
    return np.array([values[max(start, end - 1)] for start, end in buckets])


def sparkline(values: Sequence[float] | pd.Series | np.ndarray, width: int = 60) -> str:
    """One-line block-character sparkline.

    >>> sparkline([1, 2, 3, 4], width=4)
    '▁▃▆█'
    """
    array = np.asarray(values, dtype=float).ravel()
    array = array[~np.isnan(array)]
    if len(array) == 0:
        return ""
    array = _downsample(array, max(1, width))
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-12:
        return _SPARK_BLOCKS[0] * len(array)
    scaled = (array - low) / (high - low) * (len(_SPARK_BLOCKS) - 1)
    return "".join(_SPARK_BLOCKS[int(round(level))] for level in scaled)


def ascii_line_chart(
    series: dict[str, pd.Series],
    *,
    width: int = 72,
    height: int = 14,
    title: str | None = None,
) -> str:
    """Overlay several series as an ASCII chart, first series drawn on top.

    Intended for equity curves: no dependency, and it survives being pasted
    into a commit message or a code review.
    """
    cleaned = {
        name: np.asarray(values, dtype=float).ravel()
        for name, values in series.items()
        if len(values) > 0
    }
    if not cleaned:
        return "_no data_"

    width = max(8, width)
    height = max(3, height)
    markers = "#*+.:"

    sampled = {name: _downsample(values, width) for name, values in cleaned.items()}
    columns = max(len(values) for values in sampled.values())
    finite = np.concatenate([values[np.isfinite(values)] for values in sampled.values()])
    if len(finite) == 0:
        return "_no data_"
    low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-12:
        high = low + 1.0

    grid = [[" "] * columns for _ in range(height)]
    for order, (_, values) in enumerate(sampled.items()):
        marker = markers[order % len(markers)]
        for column, value in enumerate(values):
            if not np.isfinite(value):
                continue
            level = int(round((value - low) / (high - low) * (height - 1)))
            row = height - 1 - min(max(level, 0), height - 1)
            if grid[row][column] == " ":
                grid[row][column] = marker

    label_width = max(len(format_number(low)), len(format_number(high)))
    lines: list[str] = []
    if title:
        lines.append(title)
    for row_index, row in enumerate(grid):
        if row_index == 0:
            label = format_number(high)
        elif row_index == height - 1:
            label = format_number(low)
        else:
            label = ""
        lines.append(f"{label.rjust(label_width)} | {''.join(row)}")
    lines.append(" " * label_width + " +" + "-" * columns)

    legend = "  ".join(
        f"{markers[order % len(markers)]} {name}" for order, name in enumerate(sampled)
    )
    lines.append(" " * label_width + f"   {legend}")
    return "\n".join(lines)


def ascii_histogram(
    values: Sequence[float] | np.ndarray,
    *,
    bins: int = 20,
    width: int = 40,
    marker: float | None = None,
    marker_label: str = "observed",
) -> str:
    """Horizontal histogram, optionally flagging where ``marker`` falls.

    Used to show an observed Sharpe ratio against its null distribution: if the
    arrow sits inside the bulk of the bars, the result is noise.
    """
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return "_no data_"

    counts, edges = np.histogram(array, bins=max(1, bins))
    peak = int(counts.max()) or 1
    marker_bin = None
    if marker is not None and np.isfinite(marker):
        marker_bin = int(
            np.clip(np.searchsorted(edges, marker, side="right") - 1, 0, len(counts) - 1)
        )

    lines = []
    for index, count in enumerate(counts):
        bar = "#" * int(round(count / peak * width))
        row = f"{edges[index]:+8.3f} | {bar:<{width}} {count:>5d}"
        if marker_bin is not None and index == marker_bin:
            row += f"  <== {marker_label} ({format_number(marker)})"
        lines.append(row)

    if marker is not None and np.isfinite(marker):
        if marker > edges[-1]:
            lines.append(f"{'':8} | {'':<{width}} {'':>5}  <== {marker_label} above the null range")
        elif marker < edges[0]:
            lines.append(f"{'':8} | {'':<{width}} {'':>5}  <== {marker_label} below the null range")
    return "\n".join(lines)


class Report:
    """Small accumulator for Markdown sections."""

    def __init__(self, title: str, *, subtitle: str | None = None) -> None:
        self._parts: list[str] = [f"# {title}"]
        if subtitle:
            self._parts.append(f"_{subtitle}_")

    def heading(self, text: str, level: int = 2) -> Report:
        self._parts.append(f"{'#' * max(1, min(level, 6))} {text}")
        return self

    def text(self, text: str) -> Report:
        self._parts.append(text)
        return self

    def bullets(self, items: Iterable[str]) -> Report:
        rendered = [f"- {item}" for item in items]
        if rendered:
            self._parts.append("\n".join(rendered))
        return self

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[str]]) -> Report:
        self._parts.append(markdown_table(headers, rows))
        return self

    def raw_table(self, markdown: str) -> Report:
        self._parts.append(markdown)
        return self

    def code_block(self, content: str, language: str = "") -> Report:
        self._parts.append(f"```{language}\n{content}\n```")
        return self

    def dataframe(self, frame: pd.DataFrame, *, digits: int = 4, index: bool = True) -> Report:
        """Render a DataFrame as a Markdown table without extra dependencies."""
        if frame.empty:
            self._parts.append("_empty table_")
            return self
        display = frame.copy()
        headers = ([display.index.name or ""] if index else []) + [str(c) for c in display.columns]
        rows = []
        for label, row in display.iterrows():
            cells = [self._format_cell(value, digits) for value in row]
            rows.append(([str(label)] if index else []) + cells)
        self._parts.append(markdown_table(headers, rows))
        return self

    @staticmethod
    def _format_cell(value: object, digits: int) -> str:
        """Integers stay integers; floats get fixed precision; the rest is text."""
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return format_number(value, digits=digits)
        return str(value)

    def render(self) -> str:
        return "\n\n".join(part for part in self._parts if part) + "\n"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()
