"""Predictive and financial performance metrics.

Two families live here and they answer different questions:

* *predictive* metrics (RMSE, R^2, information coefficient, hit rate) ask how
  close the forecast is to the realised return;
* *financial* metrics (Sharpe, drawdown, turnover) ask whether acting on the
  forecast would have made money after costs.

A model can look good on one and useless on the other, which is exactly why
both are reported side by side.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ai_stock.config import TRADING_DAYS_PER_YEAR

__all__ = [
    "annualised_return",
    "annualised_volatility",
    "calmar_ratio",
    "classification_metrics",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "financial_metrics",
    "information_coefficient",
    "max_drawdown",
    "probabilistic_sharpe_ratio",
    "regression_metrics",
    "sharpe_ratio",
    "sortino_ratio",
]

_EPS = 1e-12


def _as_array(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array


def _aligned(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where either input is missing, keeping the two in lockstep."""
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        frame = pd.concat([y_true.rename("y"), y_pred.rename("p")], axis=1).dropna()
        return frame["y"].to_numpy(float), frame["p"].to_numpy(float)
    a, b = _as_array(y_true), _as_array(y_pred)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _horner(coefficients: tuple[float, ...], x: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * x + coefficient
    return result


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.15e-9)."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    # Central-region coefficients.
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
        1.0,
    )
    # Tail coefficients.
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
        1.0,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return _horner(c, q) / _horner(d, q)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return _horner(a, r) * q / _horner(b, r)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -_horner(c, q) / _horner(d, q)


def information_coefficient(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    method: str = "pearson",
) -> float:
    """Correlation between forecast and realised return.

    ``method`` is ``"pearson"`` or ``"spearman"`` (rank correlation, which is
    robust to the fat tails that dominate return data).
    """
    truth, prediction = _aligned(y_true, y_pred)
    if len(truth) < 3:
        return float("nan")
    if method == "spearman":
        truth = pd.Series(truth).rank().to_numpy()
        prediction = pd.Series(prediction).rank().to_numpy()
    elif method != "pearson":
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    if truth.std() < _EPS or prediction.std() < _EPS:
        return float("nan")
    return float(np.corrcoef(truth, prediction)[0, 1])


def regression_metrics(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> dict[str, float]:
    """Error, correlation and sign-accuracy metrics for a return forecast.

    ``r2`` is the out-of-sample coefficient of determination against the
    realised mean; on daily returns a value of even 0.01 is substantial, and
    negative values (worse than predicting the mean) are the norm.
    """
    truth, prediction = _aligned(y_true, y_pred)
    n = len(truth)
    if n == 0:
        return dict.fromkeys(
            ("n", "mae", "rmse", "r2", "ic_pearson", "ic_spearman", "directional_accuracy"),
            float("nan"),
        ) | {"n": 0.0}

    errors = truth - prediction
    total_variance = float(((truth - truth.mean()) ** 2).sum())
    residual = float((errors**2).sum())

    decided = prediction != 0.0
    directional = (
        float(np.mean(np.sign(prediction[decided]) == np.sign(truth[decided])))
        if decided.any()
        else float("nan")
    )

    return {
        "n": float(n),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": float(1.0 - residual / total_variance) if total_variance > _EPS else float("nan"),
        "ic_pearson": information_coefficient(truth, prediction),
        "ic_spearman": information_coefficient(truth, prediction, method="spearman"),
        "directional_accuracy": directional,
    }


def classification_metrics(
    y_true_direction: pd.Series | np.ndarray, signal: pd.Series | np.ndarray
) -> dict[str, float]:
    """Direction-classification metrics; ``signal > 0`` is read as "up".

    ``base_rate`` is the share of up moves - the accuracy of always predicting
    "up", and the number any classifier must actually beat.
    """
    truth, score = _aligned(y_true_direction, signal)
    keys = ("n", "accuracy", "precision", "recall", "f1", "roc_auc", "base_rate")
    if len(truth) == 0:
        return dict.fromkeys(keys, float("nan")) | {"n": 0.0}

    labels = (truth > 0).astype(int)
    predicted = (score > 0).astype(int)
    base_rate = float(labels.mean())

    try:
        auc = float(roc_auc_score(labels, score)) if 0 < base_rate < 1 else float("nan")
    except ValueError:  # pragma: no cover - guarded by the base-rate check
        auc = float("nan")

    return {
        "n": float(len(truth)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": auc,
        "base_rate": base_rate,
    }


def annualised_return(
    returns: pd.Series | np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Geometric (CAGR-style) annualised return of simple period returns."""
    values = _as_array(returns)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + values))
    if growth <= 0:
        return -1.0  # the account was wiped out
    return float(growth ** (periods_per_year / len(values)) - 1.0)


def annualised_volatility(
    returns: pd.Series | np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualised standard deviation of simple period returns."""
    values = _as_array(returns)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio. ``risk_free_rate`` is an annual rate."""
    values = _as_array(returns)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan")
    excess = values - risk_free_rate / periods_per_year
    deviation = float(np.std(excess, ddof=1))
    if deviation < _EPS:
        return float("nan")
    return float(np.mean(excess) / deviation * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Sharpe's downside-only cousin: only losses count as risk."""
    values = _as_array(returns)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan")
    excess = values - risk_free_rate / periods_per_year
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    if downside_deviation < _EPS:
        return float("nan")
    return float(np.mean(excess) / downside_deviation * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """Largest peak-to-trough fall of an equity curve, as a negative fraction."""
    values = _as_array(equity)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    running_peak = np.maximum.accumulate(values)
    drawdowns = values / np.where(running_peak == 0, np.nan, running_peak) - 1.0
    return float(np.nanmin(drawdowns))


def calmar_ratio(
    returns: pd.Series | np.ndarray,
    equity: pd.Series | np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised return divided by the depth of the worst drawdown."""
    drawdown = max_drawdown(equity)
    if not np.isfinite(drawdown) or abs(drawdown) < _EPS:
        return float("nan")
    return float(annualised_return(returns, periods_per_year) / abs(drawdown))


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    *,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark_sharpe``.

    Follows Bailey & Lopez de Prado (2012): it corrects the observed Sharpe for
    sample length, skewness and kurtosis. A short, fat-tailed, negatively
    skewed track record with a flattering Sharpe lands well below 0.95 here.
    """
    values = _as_array(returns)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 8:
        return float("nan")
    deviation = float(np.std(values, ddof=1))
    if deviation < _EPS:
        return float("nan")

    observed = float(np.mean(values) / deviation)  # per period, not annualised
    target = benchmark_sharpe / math.sqrt(periods_per_year)

    centred = values - np.mean(values)
    population_sd = float(np.std(values))
    skew = float(np.mean(centred**3) / population_sd**3)
    kurtosis = float(np.mean(centred**4) / population_sd**4)

    denominator = 1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed**2
    if denominator <= 0:
        return float("nan")
    statistic = (observed - target) * math.sqrt(n - 1) / math.sqrt(denominator)
    return float(_normal_cdf(statistic))


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Per-period Sharpe the best of ``n_trials`` skill-less attempts would show by chance.

    Bailey & Lopez de Prado (2014). ``trial_sharpe_std`` is the standard
    deviation, across the trials actually run, of their per-period
    (non-annualised) Sharpe ratios - it stands in for the variance pure luck
    would produce across equally-sized configurations. More trials, or more
    disagreement between them, raises the bar the survivor must clear.

    >>> expected_max_sharpe(1, 0.1)
    0.0
    >>> round(expected_max_sharpe(100, 0.1), 4)
    0.2531
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if trial_sharpe_std < 0:
        raise ValueError("trial_sharpe_std must be >= 0")
    if n_trials == 1 or trial_sharpe_std < _EPS:
        return 0.0
    euler_mascheroni = 0.5772156649015329
    return trial_sharpe_std * (
        (1.0 - euler_mascheroni) * _normal_ppf(1.0 - 1.0 / n_trials)
        + euler_mascheroni * _normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    *,
    n_trials: int,
    trial_sharpe_std: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Probability the observed Sharpe is genuine, net of how many configurations were tried.

    `probabilistic_sharpe_ratio` benchmarks one track record against a fixed
    constant (zero, by default). It has no way to know that the model,
    feature set or window it is judging was picked as the best of several
    tried on this same data - the same selection problem `multiple_testing`
    corrects for across symbols, here applied across models or windows on a
    single symbol (Bailey & Lopez de Prado, 2014). This benchmarks against the
    Sharpe pure luck would be expected to produce as the best of ``n_trials``
    equally-sized attempts, rather than against zero.

    With a single trial there is nothing to correct for, and this reduces
    exactly to `probabilistic_sharpe_ratio`:

    >>> import numpy as np
    >>> noise = np.random.default_rng(0).normal(0.0005, 0.01, 300)
    >>> deflated_sharpe_ratio(noise, n_trials=1, trial_sharpe_std=0.0) == (
    ...     probabilistic_sharpe_ratio(noise)
    ... )
    True
    """
    benchmark = expected_max_sharpe(n_trials, trial_sharpe_std) * math.sqrt(periods_per_year)
    return probabilistic_sharpe_ratio(
        returns, benchmark_sharpe=benchmark, periods_per_year=periods_per_year
    )


def financial_metrics(
    returns: pd.Series,
    *,
    equity: pd.Series | None = None,
    positions: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Summarise a strategy's return stream.

    ``positions`` is optional; when supplied, exposure and turnover are added
    so that a high Sharpe achieved by trading every single day is visible as
    the cost problem it usually is.
    """
    series = returns.dropna() if isinstance(returns, pd.Series) else pd.Series(_as_array(returns))
    if equity is None:
        equity = (1.0 + series).cumprod()

    wins = series[series > 0]
    losses = series[series < 0]
    gross_loss = float(-losses.sum())

    metrics = {
        "n_periods": float(len(series)),
        "total_return": float(np.prod(1.0 + series.to_numpy()) - 1.0)
        if len(series)
        else float("nan"),
        "annualised_return": annualised_return(series, periods_per_year),
        "annualised_volatility": annualised_volatility(series, periods_per_year),
        "sharpe": sharpe_ratio(
            series, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        "sortino": sortino_ratio(
            series, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(series, equity, periods_per_year=periods_per_year),
        "hit_rate": float((series > 0).mean()) if len(series) else float("nan"),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > _EPS else float("nan"),
        "probabilistic_sharpe": probabilistic_sharpe_ratio(
            series, periods_per_year=periods_per_year
        ),
    }

    if positions is not None:
        aligned_positions = positions.reindex(series.index).fillna(0.0)
        metrics["avg_exposure"] = float(aligned_positions.abs().mean())
        metrics["annual_turnover"] = (
            float(
                aligned_positions.diff().abs().fillna(aligned_positions.abs().iloc[0]).mean()
                * periods_per_year
            )
            if len(aligned_positions)
            else float("nan")
        )
        metrics["time_in_market"] = float((aligned_positions != 0).mean())
    return metrics
