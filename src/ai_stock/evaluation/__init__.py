"""Evaluation: walk-forward validation, metrics and multiple-testing controls."""

from ai_stock.evaluation.metrics import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    classification_metrics,
    financial_metrics,
    information_coefficient,
    max_drawdown,
    probabilistic_sharpe_ratio,
    regression_metrics,
    sharpe_ratio,
    sortino_ratio,
)
from ai_stock.evaluation.multiple_testing import (
    benjamini_hochberg,
    bonferroni_threshold,
    expected_false_positives,
)
from ai_stock.evaluation.walkforward import (
    Fold,
    FoldResult,
    WalkForwardResult,
    WalkForwardSplitter,
    run_walk_forward,
)

__all__ = [
    "Fold",
    "FoldResult",
    "WalkForwardResult",
    "WalkForwardSplitter",
    "annualised_return",
    "annualised_volatility",
    "benjamini_hochberg",
    "bonferroni_threshold",
    "calmar_ratio",
    "classification_metrics",
    "expected_false_positives",
    "financial_metrics",
    "information_coefficient",
    "max_drawdown",
    "probabilistic_sharpe_ratio",
    "regression_metrics",
    "run_walk_forward",
    "sharpe_ratio",
    "sortino_ratio",
]
