"""Synthetic market generator with a *known* predictable component.

Why synthetic data at all? Because on real prices you can never tell whether
a model found signal or merely overfit noise. Here the data-generating process
is written down explicitly, so the pipeline can be validated against ground
truth before it is ever pointed at a real ticker:

* three volatility/drift regimes driven by a Markov chain,
* GARCH(1,1)-style volatility clustering around each regime's level,
* Student-t shocks for fat tails,
* a deliberately small AR(1) + 5-day mean-reversion term - the only edge a
  model can legitimately learn. Set ``ar1 = reversion = 0`` for an efficient
  market where any measured profit is pure luck.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_stock.config import TRADING_DAYS_PER_YEAR, SyntheticConfig

REVERSION_LOOKBACK = 5
"""Window (days) of the cumulative return that drives the reversion term."""


@dataclass(frozen=True)
class SyntheticMarket:
    """A generated market together with the latent state that produced it.

    Attributes
    ----------
    ohlcv:
        The observable data a model is allowed to see.
    regime:
        Latent regime index per day (0 = bull, 1 = sideways, 2 = bear by
        default). Contemporaneously "known" but unobservable in practice.
    conditional_mean:
        E[r_t | information up to t-1] under the true model. The theoretical
        ceiling for any forecaster: no predictor can beat this systematically.
    volatility:
        The realised conditional daily volatility sigma_t.
    """

    ohlcv: pd.DataFrame
    regime: pd.Series
    conditional_mean: pd.Series
    volatility: pd.Series
    config: SyntheticConfig

    @property
    def log_returns(self) -> pd.Series:
        """Realised daily log returns of the close price."""
        return np.log(self.ohlcv["close"]).diff().rename("log_return")

    def theoretical_information_coefficient(self) -> float:
        """Correlation between the true conditional mean and realised returns.

        This is the best information coefficient attainable in this market;
        a walk-forward IC materially above it signals a leak in the pipeline.
        """
        realised = self.log_returns
        aligned = pd.concat([self.conditional_mean, realised], axis=1).dropna()
        if len(aligned) < 3:
            return float("nan")
        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def _standardised_shocks(
    rng: np.random.Generator, size: int, fat_tail_df: float | None
) -> np.ndarray:
    """Draw unit-variance shocks, Student-t when ``fat_tail_df`` is given."""
    if fat_tail_df is None:
        return rng.standard_normal(size)
    scale = np.sqrt(fat_tail_df / (fat_tail_df - 2.0))
    return rng.standard_t(fat_tail_df, size=size) / scale


def _simulate_regimes(rng: np.random.Generator, cfg: SyntheticConfig) -> np.ndarray:
    """Simulate the latent regime path of a persistent Markov chain."""
    n_regimes = len(cfg.regime_drifts)
    regimes = np.empty(cfg.n_days, dtype=np.int64)
    # Start in the least extreme regime available (sideways by default).
    current = min(1, n_regimes - 1)
    for t in range(cfg.n_days):
        regimes[t] = current
        if n_regimes > 1 and rng.random() > cfg.regime_persistence[current]:
            choices = [r for r in range(n_regimes) if r != current]
            current = int(rng.choice(choices))
    return regimes


def _simulate_returns(
    rng: np.random.Generator, cfg: SyntheticConfig, regimes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate log returns, their conditional means and volatilities.

    Returns
    -------
    (returns, conditional_mean, volatility) arrays of length ``cfg.n_days``.
    """
    n = cfg.n_days
    drifts = np.asarray(cfg.regime_drifts, dtype=float)
    vols = np.asarray(cfg.regime_vols, dtype=float)

    # Convert annualised parameters to a daily scale.
    daily_drift = drifts / TRADING_DAYS_PER_YEAR
    daily_vol = vols / np.sqrt(TRADING_DAYS_PER_YEAR)

    shocks = _standardised_shocks(rng, n, cfg.fat_tail_df)

    alpha, beta = cfg.garch_alpha, cfg.garch_beta
    omega = 1.0 - alpha - beta  # keeps E[h_t] = 1 so regime vols stay meaningful

    returns = np.zeros(n)
    cond_mean = np.zeros(n)
    volatility = np.zeros(n)

    h = 1.0  # GARCH variance multiplier, starts at its unconditional mean
    z_prev = 0.0
    for t in range(n):
        h = omega + alpha * z_prev**2 * h + beta * h
        sigma = daily_vol[regimes[t]] * np.sqrt(h)

        prev_return = returns[t - 1] if t >= 1 else 0.0
        lookback = returns[max(0, t - REVERSION_LOOKBACK) : t]
        cumulative = float(lookback.sum())

        mu = daily_drift[regimes[t]] + cfg.ar1 * prev_return + cfg.reversion * cumulative

        cond_mean[t] = mu
        volatility[t] = sigma
        returns[t] = mu + sigma * shocks[t]
        z_prev = shocks[t]

    return returns, cond_mean, volatility


def _build_ohlcv(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    returns: np.ndarray,
    volatility: np.ndarray,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Derive a consistent OHLCV frame from a close-to-close return path."""
    n = len(returns)
    close = cfg.initial_price * np.exp(np.cumsum(returns))
    prev_close = np.concatenate(([cfg.initial_price], close[:-1]))

    gap = cfg.overnight_gap_vol * volatility * rng.standard_normal(n)
    open_ = prev_close * np.exp(gap)

    # Half-normal wicks guarantee high >= max(open, close) and the mirror image
    # for the low, so the bar is always internally consistent.
    up_wick = cfg.intraday_range_mult * volatility * np.abs(rng.standard_normal(n))
    down_wick = cfg.intraday_range_mult * volatility * np.abs(rng.standard_normal(n))
    high = np.maximum(open_, close) * np.exp(up_wick)
    low = np.minimum(open_, close) * np.exp(-down_wick)

    # Volume reacts to the size of the move (vol-of-volume in logs).
    standardised_move = np.abs(returns) / np.maximum(volatility, 1e-12)
    log_volume = (
        np.log(cfg.base_volume)
        + cfg.volume_vol_beta * 0.1 * (standardised_move - np.sqrt(2.0 / np.pi))
        + 0.25 * rng.standard_normal(n)
    )
    volume = np.exp(log_volume)

    frame = pd.DataFrame(
        {
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume": np.round(volume).astype(np.int64),
        },
        index=index,
    )
    # Rounding can nudge a bar out of order; re-clamp the extremes.
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    frame.index.name = "date"
    return frame


def generate_market(config: SyntheticConfig | None = None) -> SyntheticMarket:
    """Generate a synthetic market, latent state included.

    Parameters
    ----------
    config:
        Generator settings; defaults to :class:`SyntheticConfig`.

    Examples
    --------
    >>> market = generate_market(SyntheticConfig(n_days=300, seed=0))
    >>> market.ohlcv.shape
    (300, 5)
    >>> bool((market.ohlcv["high"] >= market.ohlcv["low"]).all())
    True
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    index = pd.bdate_range(start=cfg.start, periods=cfg.n_days, name="date")
    # Drop the inferred frequency so generated data is indistinguishable from
    # data loaded from CSV, where no frequency can be carried.
    index.freq = None
    regimes = _simulate_regimes(rng, cfg)
    returns, cond_mean, volatility = _simulate_returns(rng, cfg, regimes)
    ohlcv = _build_ohlcv(rng, cfg, returns, volatility, index)

    return SyntheticMarket(
        ohlcv=ohlcv,
        regime=pd.Series(regimes, index=index, name="regime"),
        conditional_mean=pd.Series(cond_mean, index=index, name="conditional_mean"),
        volatility=pd.Series(volatility, index=index, name="volatility"),
        config=cfg,
    )


def generate_ohlcv(config: SyntheticConfig | None = None, **overrides: object) -> pd.DataFrame:
    """Generate only the observable OHLCV frame.

    Keyword overrides are applied on top of ``config`` for convenience, e.g.
    ``generate_ohlcv(n_days=500, seed=1)``.
    """
    cfg = config or SyntheticConfig()
    if overrides:
        cfg = SyntheticConfig(**{**cfg.__dict__, **overrides})  # type: ignore[arg-type]
    return generate_market(cfg).ohlcv
