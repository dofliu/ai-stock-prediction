"""AI stock prediction & strategy simulation toolkit.

The package is deliberately organised around the pieces of an *honest*
prediction study:

``ai_stock.data``
    Synthetic market generators (with known ground truth) and CSV loaders.
``ai_stock.features``
    Causal technical indicators and a leakage-free feature/target builder.
``ai_stock.models``
    A thin, uniform interface over baselines and scikit-learn estimators.
``ai_stock.evaluation``
    Walk-forward validation with embargo plus predictive/financial metrics.
``ai_stock.backtest``
    Signal to position to equity-curve simulation including trading costs.
``ai_stock.simulation``
    Monte-Carlo path simulation and permutation tests for luck vs. skill.
``ai_stock.reporting``
    Markdown/CSV report rendering.
"""

from ai_stock.config import (
    BacktestConfig,
    ExperimentConfig,
    FeatureConfig,
    SimulationConfig,
    SyntheticConfig,
    WalkForwardConfig,
)

__all__ = [
    "BacktestConfig",
    "ExperimentConfig",
    "FeatureConfig",
    "SimulationConfig",
    "SyntheticConfig",
    "WalkForwardConfig",
    "__version__",
]

__version__ = "0.1.0"
