"""Holding several single-asset strategies at once.

Every other module here evaluates one symbol at a time, and the universe
screen ranks those standalone studies side by side. Trading them together is
a different question. Four names from the same industry, driven by the same
memory cycle, are not four independent bets - but adding four backtests
together quietly assumes they are, and a portfolio Sharpe computed that way
inherits a diversification the correlation never delivered.

This module makes the assumption measurable. It puts the per-symbol net
return streams on the calendar they share, then reports how much
diversification the realised correlation actually buys: a diversification
ratio, an effective number of bets, and the Sharpe the same sleeves would
have shown had they moved independently.

Nothing here optimises weights. Mean-variance weights fitted to the sample
they are then scored on are one of the most reliable ways to manufacture a
backtest, which is exactly what this project exists not to do. Both schemes
on offer - equal weight and inverse volatility - ignore the correlation
matrix entirely, so neither can overfit it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_stock.config import TRADING_DAYS_PER_YEAR
from ai_stock.evaluation.metrics import financial_metrics

__all__ = [
    "ROLLING_WINDOW",
    "STRESS_QUANTILE",
    "WEIGHT_SCHEMES",
    "PortfolioResult",
    "align_sleeves",
    "build_portfolio",
    "diversification_ratio",
    "effective_number_of_bets",
    "sleeve_weights",
]

_EPS = 1e-12

WEIGHT_SCHEMES = ("equal", "inverse_vol")
"""Weighting rules `build_portfolio` accepts. Neither one reads a correlation."""

ROLLING_WINDOW = 63
"""Trailing window for `PortfolioResult.rolling_effective_bets`, in trading days.

A quarter. Short enough that a correlation spike lasting a few weeks is still
visible rather than averaged away, long enough that a 4x4 covariance estimated
from it is not mostly noise - roughly sixteen observations per estimated
pairwise correlation. Nothing about the choice is optimal; it is a compromise
between two biases that point in opposite directions, and it is a parameter so
that a reader who disagrees can move it.
"""

STRESS_QUANTILE = 0.2
"""Fraction of dates counted as stressed: the deepest fifth of the drawdown.

Deep enough to mean something, wide enough that the stressed set is not one
episode. Both halves of that sentence are judgements, not results.
"""


def _combined(returns_by_symbol: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Outer-join the sleeves, so a date any of them traded appears once."""
    if not returns_by_symbol:
        raise ValueError("no sleeves to combine")
    frame = pd.DataFrame(
        {
            str(symbol): pd.Series(series, dtype=float)
            for symbol, series in returns_by_symbol.items()
        }
    )
    if frame.empty:
        raise ValueError("every sleeve is empty")
    return frame.sort_index()


def _common(combined: pd.DataFrame) -> pd.DataFrame:
    sleeves = combined.dropna(how="any")
    if len(sleeves) < 2:
        raise ValueError(
            "the sleeves share fewer than two common dates; there is no portfolio to measure"
        )
    return sleeves


def align_sleeves(returns_by_symbol: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Put every sleeve's return stream on the calendar they all share.

    The intersection, not the union. A correlation only means something
    between observations made on the same day, and a Taiwan holiday is not a
    day on which ``MU`` and ``2337.TW`` can be compared; filling the gap with
    a zero would invent a quiet, uncorrelated day that nobody traded and pull
    every correlation towards zero. What the intersection costs is reported
    instead, as :attr:`PortfolioResult.common_fraction`.

    Raises
    ------
    ValueError
        If there are no sleeves, or fewer than two dates on which all of them
        traded - a volatility needs two observations and a correlation needs
        rather more.
    """
    return _common(_combined(returns_by_symbol))


def sleeve_weights(sleeves: pd.DataFrame, scheme: str = "equal") -> pd.Series:
    """Allocation weights across sleeves, non-negative and summing to one.

    ``equal`` splits the capital evenly. ``inverse_vol`` splits it in inverse
    proportion to each sleeve's realised volatility, so a sleeve that moves
    twice as much is given half the capital - risk parity's simplest form,
    and the only version that needs no covariance matrix to invert.

    Both ignore the correlations on purpose: weights chosen by looking at the
    same sample they are then scored on is how a backtest is manufactured,
    and the correlation matrix is where that temptation lives.
    """
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"unknown weight scheme {scheme!r}; expected one of {WEIGHT_SCHEMES}")
    if sleeves.empty or not len(sleeves.columns):
        raise ValueError("no sleeves to weight")

    if scheme == "equal":
        raw = pd.Series(1.0, index=sleeves.columns, dtype=float)
    else:
        volatilities = sleeves.std(ddof=1)
        if not np.isfinite(volatilities).all() or bool((volatilities <= _EPS).any()):
            raise ValueError("inverse_vol needs every sleeve to have a positive volatility")
        raw = 1.0 / volatilities
    return (raw / raw.sum()).rename("weight")


def _validated(covariance: pd.DataFrame | np.ndarray, weights: pd.Series | np.ndarray):
    matrix = np.asarray(covariance, dtype=float)
    vector = np.asarray(weights, dtype=float).ravel()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if vector.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"{vector.shape[0]} weight(s) but a {matrix.shape[0]}x{matrix.shape[1]} covariance"
        )
    if bool((vector < 0).any()):
        raise ValueError("weights must be non-negative")
    return matrix, vector


def diversification_ratio(
    covariance: pd.DataFrame | np.ndarray, weights: pd.Series | np.ndarray
) -> float:
    """Weighted average sleeve volatility divided by the portfolio's own volatility.

    Choueifaty & Coignard (2008). It is 1 when the sleeves are perfectly
    correlated - one bet wearing several names - and ``sqrt(n)`` for ``n``
    equally weighted, equally volatile, uncorrelated sleeves.

    >>> import numpy as np
    >>> cov = np.full((4, 4), 0.6) + np.diag(np.full(4, 0.4))  # unit vols, rho = 0.6
    >>> round(diversification_ratio(cov, np.full(4, 0.25)), 4)
    1.1952
    >>> round(diversification_ratio(np.eye(4), np.full(4, 0.25)), 4)
    2.0
    """
    matrix, vector = _validated(covariance, weights)
    variance = float(vector @ matrix @ vector)
    if not np.isfinite(variance) or variance <= _EPS:
        return float("nan")
    volatilities = np.sqrt(np.clip(np.diag(matrix), 0.0, None))
    return float(vector @ volatilities / math.sqrt(variance))


def effective_number_of_bets(
    covariance: pd.DataFrame | np.ndarray, weights: pd.Series | np.ndarray
) -> float:
    """How many independent bets the portfolio really holds: the squared diversification ratio.

    For ``n`` equally weighted sleeves of equal volatility and common
    correlation ``rho`` this is exactly ``n / (1 + (n - 1) * rho)`` - the
    classical count of independent bets - and it generalises that formula to
    unequal weights and an arbitrary correlation matrix.

    Four sleeves correlated at 0.6 are nothing like four bets:

    >>> import numpy as np
    >>> cov = np.full((4, 4), 0.6) + np.diag(np.full(4, 0.4))
    >>> round(effective_number_of_bets(cov, np.full(4, 0.25)), 4)
    1.4286
    >>> round(4 / (1 + 3 * 0.6), 4)
    1.4286

    It can also exceed the number of sleeves, and that is not a defect: two
    sleeves that reliably move against each other hedge one another, which is
    worth strictly more than two independent bets.

    >>> hedged = np.array([[1.0, -0.5], [-0.5, 1.0]])
    >>> round(effective_number_of_bets(hedged, np.full(2, 0.5)), 4)
    4.0

    Read a count above ``n`` the way it reads: over this sample the sleeves
    offset each other. Realised negative correlation between strategies on
    related names is usually a small-sample artefact rather than a hedge that
    survives the next drawdown.
    """
    return float(diversification_ratio(covariance, weights) ** 2)


@dataclass(frozen=True)
class PortfolioResult:
    """Several single-asset strategies held together, and what that costs in diversification."""

    sleeves: pd.DataFrame
    """Net sleeve returns, on the calendar every sleeve traded."""
    weights: pd.Series
    returns: pd.Series
    """Portfolio returns: the sleeves recombined at ``weights``, no leverage."""
    equity: pd.Series
    scheme: str
    common_fraction: float
    """Share of the sleeves' combined calendar on which all of them traded."""
    assets: pd.DataFrame | None = None
    """Buy-and-hold returns of the same symbols, on the same calendar.

    Optional, and the reason it is worth supplying: a strategy sleeve can
    decorrelate from its neighbours even when the underlying shares do not,
    simply because the two models happen to be positioned differently. That
    is the difference between diversification earned by the names and
    diversification manufactured by disagreement.
    """
    periods_per_year: int = TRADING_DAYS_PER_YEAR

    @property
    def symbols(self) -> list[str]:
        return [str(column) for column in self.sleeves.columns]

    def correlation(self) -> pd.DataFrame:
        """Correlation of the sleeve returns - the number the whole module is about."""
        return self.sleeves.corr()

    def covariance(self) -> pd.DataFrame:
        return self.sleeves.cov(ddof=1)

    def asset_correlation(self) -> pd.DataFrame | None:
        """Correlation of the underlying shares, for contrast with the sleeves."""
        return None if self.assets is None else self.assets.corr()

    def drawdown(self) -> pd.Series:
        """Depth below the running peak, as a non-positive fraction, on every date."""
        peak = self.equity.cummax()
        return (self.equity / peak.where(peak.abs() > _EPS) - 1.0).rename("drawdown")

    def rolling_effective_bets(self, window: int = ROLLING_WINDOW) -> pd.Series:
        """:func:`effective_number_of_bets` recomputed on each trailing ``window`` of dates.

        :meth:`metrics`'s ``effective_bets`` is one number for the whole
        sample, which is an average over every regime the sleeves lived
        through. Correlations are not constant, and they do not move
        randomly: they rise in exactly the sell-offs the diversification was
        supposed to cushion. A full-sample count therefore reports the
        diversification available on the average day, which is not the day
        anyone needs it.

        The weights are held fixed at :attr:`weights` rather than refitted
        inside each window, on purpose: the question is what the
        *correlation* did, and a weight that moves too makes the two
        inseparable. For ``inverse_vol`` that means each window is scored at
        weights derived from the full sample - descriptive, like the rest of
        this section, and not a claim about what could have been traded.

        The first ``window - 1`` dates are ``NaN``, and so is every date if
        the sleeves share fewer than ``window`` of them.

        Consecutive values share ``window - 1`` observations, so the series is
        heavily autocorrelated. Read it as a picture of when diversification
        was thin, never as a count of independent readings.
        """
        if window < 2:
            raise ValueError(f"window must cover at least two periods; got {window}")
        values = self.sleeves.to_numpy(dtype=float)
        vector = self.weights.to_numpy(dtype=float)
        counts = np.full(len(values), float("nan"))
        for end in range(window, len(values) + 1):
            block = np.atleast_2d(np.cov(values[end - window : end], rowvar=False, ddof=1))
            counts[end - 1] = effective_number_of_bets(block, vector)
        return pd.Series(counts, index=self.sleeves.index, name="rolling_effective_bets")

    def diversification_under_stress(
        self, window: int = ROLLING_WINDOW, *, quantile: float = STRESS_QUANTILE
    ) -> dict[str, float]:
        """Rolling bet count on the deepest-drawdown dates against all the others.

        Splits :meth:`rolling_effective_bets` by :meth:`drawdown`: the
        ``quantile`` fraction of dates sitting furthest below the equity
        curve's running peak are the stressed ones, the rest are calm.
        ``stress_gap`` is stressed minus calm, so a **negative** gap is the
        failure the full-sample count hides - the sleeves converged into one
        bet precisely while the portfolio was losing money.

        This is a description of what happened over one sample, not a
        prediction. The stressed dates are contiguous by construction - a
        drawdown is a run of days, not a scatter - so a handful of episodes
        can supply the whole stressed set, and their windows overlap besides.
        There is deliberately no p-value here: there is nothing to attach one
        to that would not overstate the sample.
        """
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must lie strictly between 0 and 1; got {quantile}")
        counts = self.rolling_effective_bets(window)
        drawdown = self.drawdown()
        usable = counts.notna() & drawdown.notna()
        empty = {
            "rolling_bets_window": float(window),
            "rolling_bets_min": float("nan"),
            "rolling_bets_median": float("nan"),
            "rolling_bets_stressed": float("nan"),
            "rolling_bets_calm": float("nan"),
            "rolling_bets_stress_gap": float("nan"),
            "rolling_bets_n_stressed": 0.0,
        }
        if not bool(usable.any()):
            return empty

        counts, drawdown = counts[usable], drawdown[usable]
        # `<=` against the quantile puts ties on the stressed side. A strategy
        # that has never lost is all ties at zero, which leaves nothing calm to
        # compare against - reported as NaN rather than as a gap of zero.
        stressed = counts[drawdown <= drawdown.quantile(quantile)]
        calm = counts[drawdown > drawdown.quantile(quantile)]
        comparable = bool(len(stressed)) and bool(len(calm))
        return {
            **empty,
            "rolling_bets_min": float(counts.min()),
            "rolling_bets_median": float(counts.median()),
            "rolling_bets_stressed": float(stressed.mean()) if len(stressed) else float("nan"),
            "rolling_bets_calm": float(calm.mean()) if len(calm) else float("nan"),
            "rolling_bets_stress_gap": float(stressed.mean() - calm.mean())
            if comparable
            else float("nan"),
            "rolling_bets_n_stressed": float(len(stressed)),
        }

    def risk_contributions(self) -> pd.Series:
        """Each sleeve's share of portfolio variance, summing to one.

        Equal capital is not equal risk: a sleeve twice as volatile as its
        neighbours, or one that moves with everything else, carries more of
        the portfolio's variance than its weight suggests.
        """
        matrix = self.covariance().to_numpy()
        vector = self.weights.to_numpy(dtype=float)
        variance = float(vector @ matrix @ vector)
        if not np.isfinite(variance) or variance <= _EPS:
            return pd.Series(float("nan"), index=self.weights.index, name="risk_contribution")
        contributions = vector * (matrix @ vector) / variance
        return pd.Series(contributions, index=self.weights.index, name="risk_contribution")

    def sleeve_table(self) -> pd.DataFrame:
        """Per-sleeve weight, standalone risk and return, and share of portfolio variance."""
        scale = math.sqrt(self.periods_per_year)
        volatilities = self.sleeves.std(ddof=1) * scale
        means = self.sleeves.mean() * self.periods_per_year
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = means / volatilities.replace(0.0, np.nan)
        frame = pd.DataFrame(
            {
                "weight": self.weights,
                "annualised_volatility": volatilities,
                "annualised_return": means,
                "sharpe": sharpe,
                "risk_contribution": self.risk_contributions(),
            }
        )
        frame.index.name = "symbol"
        return frame

    def sharpe_if_independent(self) -> float:
        """The portfolio Sharpe the same sleeves would show with zero correlation.

        The mean is untouched - it is linear in the weights, and correlation
        cannot change it. Only the volatility moves, from the realised
        ``sqrt(w' C w)`` down to the ``sqrt(sum(w_i^2 * s_i^2))`` independence
        would allow. The gap between this and the actual Sharpe is the part of
        a naive four-backtest sum that the correlation never handed over.
        """
        volatilities = self.sleeves.std(ddof=1).to_numpy(dtype=float)
        vector = self.weights.to_numpy(dtype=float)
        independent = float(np.sqrt(np.sum((vector * volatilities) ** 2)))
        if not np.isfinite(independent) or independent <= _EPS:
            return float("nan")
        return float(self.returns.mean() / independent * math.sqrt(self.periods_per_year))

    def metrics(self) -> dict[str, float]:
        """Portfolio performance, plus the diversification the correlation actually delivered."""
        merged = dict(
            financial_metrics(
                self.returns, equity=self.equity, periods_per_year=self.periods_per_year
            )
        )
        correlation = self.correlation().to_numpy(dtype=float)
        off_diagonal = correlation[np.triu_indices(len(self.symbols), k=1)]
        ratio = diversification_ratio(self.covariance(), self.weights)
        independent = self.sharpe_if_independent()

        if self.assets is None:
            asset_correlation = float("nan")
            asset_bets = float("nan")
        else:
            asset_matrix = self.assets.corr().to_numpy(dtype=float)
            asset_off = asset_matrix[np.triu_indices(len(self.symbols), k=1)]
            asset_correlation = float(np.mean(asset_off)) if asset_off.size else float("nan")
            asset_bets = effective_number_of_bets(self.assets.cov(ddof=1), self.weights)

        merged.update(
            {
                "n_sleeves": float(len(self.symbols)),
                "mean_asset_correlation": asset_correlation,
                "asset_effective_bets": asset_bets,
                "common_fraction": float(self.common_fraction),
                "mean_correlation": float(np.mean(off_diagonal))
                if off_diagonal.size
                else float("nan"),
                "max_correlation": float(np.max(off_diagonal))
                if off_diagonal.size
                else float("nan"),
                "min_correlation": float(np.min(off_diagonal))
                if off_diagonal.size
                else float("nan"),
                "diversification_ratio": ratio,
                "effective_bets": float(ratio**2),
                "sharpe_if_independent": independent,
                "sharpe_diversification_gap": float(independent - merged["sharpe"]),
                **self.diversification_under_stress(),
            }
        )
        return merged


def build_portfolio(
    returns_by_symbol: Mapping[str, pd.Series],
    *,
    asset_returns_by_symbol: Mapping[str, pd.Series] | None = None,
    scheme: str = "equal",
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PortfolioResult:
    """Combine per-symbol net return streams into one portfolio.

    Parameters
    ----------
    returns_by_symbol:
        Symbol -> net strategy returns, e.g. ``run.backtest.returns`` for each
        symbol in a screen. Costs are expected to be netted out already; this
        function adds none of its own, because rebalancing between sleeves is
        a decision it does not make.
    asset_returns_by_symbol:
        Optional buy-and-hold returns for the same symbols, e.g.
        ``run.backtest.asset_returns``. Supplying them adds the one contrast
        that stops the sleeve correlation being read as good news on its own:
        whether the sleeves are less correlated than the shares they trade.
    scheme:
        One of :data:`WEIGHT_SCHEMES`. Weights are fixed for the whole sample
        and the portfolio is fully invested across the sleeves, so there is no
        leverage and no rebalancing schedule to model.

    Raises
    ------
    ValueError
        If the sleeves share fewer than two dates, the scheme is unknown, or
        the asset returns cover a different set of symbols.
    """
    combined = _combined(returns_by_symbol)
    sleeves = _common(combined)
    weights = sleeve_weights(sleeves, scheme)

    assets = None
    if asset_returns_by_symbol is not None:
        if set(asset_returns_by_symbol) != set(returns_by_symbol):
            raise ValueError("asset returns must cover exactly the same symbols as the sleeves")
        assets = _combined(asset_returns_by_symbol).reindex(sleeves.index)[sleeves.columns]
        # A gap here would leave `assets.cov()` estimated pair by pair on
        # different samples, which can produce a matrix no portfolio could
        # actually have. Refuse rather than report a diversification number
        # built on one.
        if bool(assets.isna().to_numpy().any()):
            raise ValueError(
                "asset returns must cover every date the sleeves trade; "
                f"{int(assets.isna().any(axis=1).sum())} date(s) are missing"
            )

    returns = (sleeves * weights).sum(axis=1).rename("portfolio_return")
    equity = (1.0 + returns).cumprod().rename("portfolio_equity")
    return PortfolioResult(
        sleeves=sleeves,
        weights=weights,
        returns=returns,
        equity=equity,
        scheme=scheme,
        common_fraction=float(len(sleeves) / len(combined)),
        assets=assets,
        periods_per_year=periods_per_year,
    )
