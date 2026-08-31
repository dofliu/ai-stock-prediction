"""Typed configuration objects shared by every stage of the pipeline.

Configs are plain frozen dataclasses so an experiment can be described,
logged and reproduced without touching global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

TRADING_DAYS_PER_YEAR = 252
"""Annualisation factor used consistently across metrics and simulations."""


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the regime-switching synthetic market.

    The generator intentionally embeds a *small but real* predictable
    component (``ar1`` and ``reversion``) so that a correctly wired pipeline
    can recover a known edge. Set both to ``0.0`` to obtain an efficient
    (unpredictable) market and confirm that models then earn nothing.
    """

    n_days: int = 2500
    start: str = "2010-01-04"
    initial_price: float = 100.0
    seed: int | None = 42

    # Regime means/vols expressed in annualised terms, ordered
    # bull / sideways / bear.
    regime_drifts: tuple[float, ...] = (0.18, 0.02, -0.22)
    regime_vols: tuple[float, ...] = (0.16, 0.13, 0.34)
    # Probability of staying in the current regime on any given day.
    regime_persistence: tuple[float, ...] = (0.985, 0.975, 0.965)

    # Learnable structure.
    ar1: float = 0.06
    """AR(1) coefficient on daily returns (momentum when positive)."""
    reversion: float = -0.05
    """Loading on the 5-day cumulative return (mean reversion when negative)."""

    # Volatility clustering: sigma_t^2 = omega + alpha * eps^2 + beta * sigma^2.
    garch_alpha: float = 0.08
    garch_beta: float = 0.88

    fat_tail_df: float | None = 6.0
    """Student-t degrees of freedom for shocks; ``None`` uses Gaussian."""

    # Intraday / volume shape.
    overnight_gap_vol: float = 0.3
    """Overnight gap std as a fraction of the day's volatility."""
    intraday_range_mult: float = 1.6
    base_volume: float = 1_000_000.0
    volume_vol_beta: float = 2.5
    """How strongly volume responds to |return|."""

    def __post_init__(self) -> None:
        n = len(self.regime_drifts)
        if not (n == len(self.regime_vols) == len(self.regime_persistence)):
            raise ValueError("regime_drifts, regime_vols and regime_persistence must align")
        if n < 1:
            raise ValueError("at least one regime is required")
        if self.n_days < 2:
            raise ValueError("n_days must be >= 2")
        if any(v <= 0 for v in self.regime_vols):
            raise ValueError("regime volatilities must be positive")
        if any(not 0.0 < p <= 1.0 for p in self.regime_persistence):
            raise ValueError("regime_persistence entries must lie in (0, 1]")
        if self.garch_alpha < 0 or self.garch_beta < 0:
            raise ValueError("GARCH parameters must be non-negative")
        if self.garch_alpha + self.garch_beta >= 1:
            raise ValueError("garch_alpha + garch_beta must be < 1 for stationarity")
        if self.initial_price <= 0:
            raise ValueError("initial_price must be positive")
        if self.fat_tail_df is not None and self.fat_tail_df <= 2:
            raise ValueError("fat_tail_df must exceed 2 for a finite variance")


@dataclass(frozen=True)
class FeatureConfig:
    """Which features to build and what to predict."""

    horizon: int = 1
    """Prediction horizon in trading days (target is the forward log return)."""

    return_lags: tuple[int, ...] = (1, 2, 3, 5, 10)
    sma_windows: tuple[int, ...] = (5, 10, 20, 50)
    ema_windows: tuple[int, ...] = (12, 26)
    momentum_windows: tuple[int, ...] = (5, 10, 20)
    volatility_windows: tuple[int, ...] = (5, 20)
    rsi_window: int = 14
    macd: tuple[int, int, int] = (12, 26, 9)
    bollinger: tuple[int, float] = (20, 2.0)
    atr_window: int = 14
    stochastic: tuple[int, int] = (14, 3)
    volume_windows: tuple[int, ...] = (5, 20)

    include_volume_features: bool = True
    neutral_band: float = 0.0
    """Classification labels inside +/- neutral_band are dropped (0 keeps all)."""

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.neutral_band < 0:
            raise ValueError("neutral_band must be non-negative")
        windows = (
            *self.return_lags,
            *self.sma_windows,
            *self.ema_windows,
            *self.momentum_windows,
            *self.volatility_windows,
            self.rsi_window,
            *self.macd,
            self.bollinger[0],
            self.atr_window,
            *self.stochastic,
            *self.volume_windows,
        )
        if any(int(w) < 1 for w in windows):
            raise ValueError("all feature windows must be >= 1")
        if self.bollinger[1] <= 0:
            raise ValueError("bollinger k must be positive")


@dataclass(frozen=True)
class WalkForwardConfig:
    """Rolling-origin evaluation schedule."""

    train_size: int = 750
    test_size: int = 125
    step: int | None = None
    """Advance per fold; defaults to ``test_size`` (non-overlapping tests)."""
    expanding: bool = True
    """Expanding window (use all history) vs. fixed-length rolling window."""
    embargo: int | None = None
    """Bars skipped between train and test; defaults to the target horizon."""
    min_train_size: int | None = None
    """Explicit floor on training bars per fold; defaults to ``train_size``.

    Folds below the floor are skipped. Leaving it unset means a small
    ``train_size`` is honoured rather than silently producing no folds.
    """

    def __post_init__(self) -> None:
        if self.train_size < 1 or self.test_size < 1:
            raise ValueError("train_size and test_size must be >= 1")
        if self.step is not None and self.step < 1:
            raise ValueError("step must be >= 1")
        if self.embargo is not None and self.embargo < 0:
            raise ValueError("embargo must be non-negative")
        if self.min_train_size is not None and self.min_train_size < 1:
            raise ValueError("min_train_size must be >= 1")

    def resolved_step(self) -> int:
        return self.step if self.step is not None else self.test_size

    def resolved_embargo(self, horizon: int) -> int:
        return self.embargo if self.embargo is not None else horizon

    def resolved_min_train_size(self) -> int:
        return self.min_train_size if self.min_train_size is not None else self.train_size


@dataclass(frozen=True)
class BacktestConfig:
    """How predictions become positions, and what trading costs apply."""

    sizing: str = "sign"
    """One of ``sign`` (long/short), ``long_only`` or ``proportional``."""
    threshold: float = 0.0
    """Dead band on the signal; below it the position is flat."""
    max_leverage: float = 1.0
    scale: float = 100.0
    """Signal multiplier used by ``proportional`` sizing before clipping."""
    cost_bps: float = 1.0
    """Commission/fee in basis points of traded notional."""
    slippage_bps: float = 1.0
    """Additional execution cost in basis points of traded notional."""
    allow_short: bool = True
    vol_target: float | None = None
    """Annualised volatility target; scales positions when set."""
    vol_lookback: int = 20
    initial_equity: float = 100_000.0

    _SIZINGS = ("sign", "long_only", "proportional")

    def __post_init__(self) -> None:
        if self.sizing not in self._SIZINGS:
            raise ValueError(f"sizing must be one of {self._SIZINGS}, got {self.sizing!r}")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if self.cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must be non-negative")
        if self.vol_target is not None and self.vol_target <= 0:
            raise ValueError("vol_target must be positive when set")
        if self.vol_lookback < 2:
            raise ValueError("vol_lookback must be >= 2")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")

    @property
    def total_cost_bps(self) -> float:
        return self.cost_bps + self.slippage_bps


@dataclass(frozen=True)
class SimulationConfig:
    """Monte-Carlo and significance-testing settings."""

    n_paths: int = 1000
    horizon_days: int = 252
    method: str = "block_bootstrap"
    """``iid_bootstrap``, ``block_bootstrap`` or ``gaussian``."""
    block_size: int = 20
    n_permutations: int = 500
    seed: int | None = 7
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

    _METHODS = ("iid_bootstrap", "block_bootstrap", "gaussian")

    def __post_init__(self) -> None:
        if self.method not in self._METHODS:
            raise ValueError(f"method must be one of {self._METHODS}, got {self.method!r}")
        if self.n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.n_permutations < 0:
            raise ValueError("n_permutations must be non-negative")
        if any(not 0.0 < q < 1.0 for q in self.quantiles):
            raise ValueError("quantiles must lie strictly inside (0, 1)")


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to reproduce one end-to-end run."""

    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def with_horizon(self, horizon: int) -> ExperimentConfig:
        """Return a copy whose feature horizon is ``horizon``."""
        return replace(self, features=replace(self.features, horizon=horizon))
