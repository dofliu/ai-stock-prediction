"""Report rendering: Markdown tables and dependency-free ASCII charts."""

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

__all__ = [
    "Report",
    "ascii_histogram",
    "ascii_line_chart",
    "format_number",
    "markdown_table",
    "metrics_table",
    "render_backtest_report",
    "render_comparison_report",
    "render_simulation_report",
    "sparkline",
]
