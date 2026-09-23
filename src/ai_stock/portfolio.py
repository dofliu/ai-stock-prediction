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
    "WEIGHT_SCHEMES",
    "PortfolioResult",
    "align_sleeves",
    "build_portfolio",
    "diversification_ratio",
    "effective_number_of_bets",
    "rolling_effective_bets",
    "sleeve_weights",
    "weekly_returns",
]

_EPS = 1e-12

WEIGHT_SCHEMES = ("equal", "inverse_vol")
"""Weighting rules `build_portfolio` accepts. Neither one reads a correlation."""

ROLLING_WINDOW = 63
"""Trailing window, in trading days, for the rolling diversification view.

One calendar quarter, pre-committed rather than chosen. A window picked
because it made the bet count look steadiest - or because it made the
collapse look worst - would be the same after-the-fact selection this module
refuses for weights, and the rolling series exists precisely to be read as
evidence.
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


def weekly_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Compound daily returns into calendar-week buckets, dropping weeks nobody traded.

    A same-day correlation is blind to a shared move that straddles a date
    boundary, which is exactly what happens between markets in different time
    zones. A shock during New York hours lands on ``MU``'s bar for that date
    and on a Taipei name's bar for the *next* one, because Taipei has already
    closed; matching the two by calendar date splits one move across two
    observations and understates how much they really move together.
    Compounding into weeks lets both halves fall in one observation, which is
    what makes the gap between a daily and a weekly correlation a measure of
    how much the same-day match hid.

    The index must be a :class:`~pandas.DatetimeIndex`. A week in which no
    sleeve traded is dropped rather than reported as a flat week: an empty
    bucket compounds to a zero return, and an invented quiet week pulls every
    correlation towards zero for the same reason :func:`align_sleeves` takes
    the intersection rather than the union.

    >>> import pandas as pd
    >>> idx = pd.bdate_range("2024-01-01", periods=10, name="date")
    >>> daily = pd.DataFrame({"a": [0.01] * 10, "b": [0.01] * 10}, index=idx)
    >>> weekly = weekly_returns(daily)
    >>> weekly.shape
    (2, 2)
    >>> bool((weekly.round(6) == round((1.01**5) - 1, 6)).to_numpy().all())
    True
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("weekly returns need a DatetimeIndex to bucket dates into weeks")
    weekly = (1.0 + returns).resample("W").prod() - 1.0
    traded = returns.resample("W").size().to_numpy() > 0
    return weekly.loc[traded]


def _mean_off_diagonal(correlation: pd.DataFrame) -> float:
    """Mean of a correlation matrix's upper triangle, or NaN when there is no pair."""
    matrix = correlation.to_numpy(dtype=float)
    off_diagonal = matrix[np.triu_indices(len(matrix), k=1)]
    return float(np.mean(off_diagonal)) if off_diagonal.size else float("nan")


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


def rolling_effective_bets(
    sleeves: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Mean pairwise correlation and effective bet count over each trailing window.

    :func:`effective_number_of_bets` on the full sample answers "how many bets
    was this on average". That is the wrong question in the one situation the
    count is bought for: correlations rise in drawdowns, so the
    diversification tends to be thinnest exactly when it was supposed to
    cushion. A trailing window says *when* the count collapsed, the same way a
    rolling ``hit_rate_z`` over the journal says when a decay started.

    The weights are held fixed at the ones passed in rather than re-derived
    inside each window. A portfolio is not re-weighted by this function: an
    ``inverse_vol`` allocation recomputed window by window is a different,
    adaptive strategy that was never traded, and scoring it here would quietly
    hand the section a decision it did not make.

    Both columns are causal at their index date - the window ends there - so
    the series can be read against a drawdown of the same portfolio without a
    look-ahead. What it is not is a series of independent observations:
    consecutive windows share ``window - 1`` days, so the minimum over a long
    sample is a minimum over many overlapping draws and is biased low.

    Returns a frame indexed by each window's last date, with columns
    ``mean_correlation`` and ``effective_bets``. A sample shorter than
    ``window`` yields an empty frame rather than an exception, the same way
    :meth:`~ai_stock.evaluation.walkforward.WalkForwardResult.regime_metrics`
    returns an empty frame when it cannot bin: a rolling view is a diagnostic
    the full-sample numbers can do without.

    >>> import numpy as np, pandas as pd
    >>> index = pd.bdate_range("2020-01-01", periods=8, name="date")
    >>> frame = pd.DataFrame(
    ...     np.random.default_rng(0).standard_normal((8, 2)), index=index, columns=["a", "b"]
    ... )
    >>> weights = pd.Series([0.5, 0.5], index=["a", "b"])
    >>> rolling = rolling_effective_bets(frame, weights, window=4)
    >>> list(rolling.columns)
    ['mean_correlation', 'effective_bets']
    >>> list(rolling.index) == list(index[3:])
    True
    >>> rolling_effective_bets(frame, weights, window=20).empty
    True
    """
    if window < 2:
        raise ValueError(f"a rolling correlation needs a window of at least 2, got {window}")
    values = sleeves.to_numpy(dtype=float)
    vector = np.asarray(weights, dtype=float).ravel()
    if vector.shape[0] != values.shape[1]:
        raise ValueError(f"{vector.shape[0]} weight(s) but {values.shape[1]} sleeve(s)")

    n_rows = values.shape[0]
    columns = ["mean_correlation", "effective_bets"]
    if n_rows < window:
        return pd.DataFrame(columns=columns, index=sleeves.index[:0], dtype=float)

    upper = np.triu_indices(values.shape[1], k=1)
    rows = []
    for end in range(window, n_rows + 1):
        covariance = np.cov(values[end - window : end], rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance)
        deviations = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        outer = np.outer(deviations, deviations)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.where(
                outer > _EPS, covariance / np.where(outer > _EPS, outer, 1.0), np.nan
            )
        pairs = correlation[upper]
        rows.append(
            (
                float(np.mean(pairs)) if pairs.size else float("nan"),
                effective_number_of_bets(covariance, vector),
            )
        )
    return pd.DataFrame(rows, columns=columns, index=sleeves.index[window - 1 :])


def _drawdown_labels(n_bins: int) -> list[str]:
    """Bin names ordered as ``pd.qcut`` orders them: deepest drawdown first."""
    if n_bins == 3:
        return ["deep_drawdown", "mid_drawdown", "shallow_drawdown"]
    return [f"drawdown_q{i + 1}_of_{n_bins}" for i in range(n_bins)]


def _stress_z(deep: pd.Series, shallow: pd.Series, window: int) -> tuple[float, float]:
    """How many standard errors separate two tercile means, deflated and naive.

    A difference of a tenth of a bet between two terciles is not a finding,
    and printing it as one is the failure this whole project is written
    against. The deflated figure is the one to read: rolling windows overlap
    by all but one day, so a tercile holding ``m`` windows does not hold ``m``
    reads of the market. Dividing by ``window`` assumes total redundancy
    within any window-length span, the same worst case
    :func:`~ai_stock.journal.independent_blocks` assumes for the forecast
    journal, and the naive figure - which assumes none - is returned beside it
    for the same reason ``hit_rate_z_naive`` is: the honest answer lies
    between them, and while they disagree, believe the smaller.
    """
    if len(deep) < 2 or len(shallow) < 2:
        return float("nan"), float("nan")
    gap = float(deep.mean() - shallow.mean())
    variances = (float(deep.var(ddof=1)), float(shallow.var(ddof=1)))
    counts = (float(len(deep)), float(len(shallow)))
    if not all(np.isfinite(variances)):
        return float("nan"), float("nan")

    def z_at(scale: float) -> float:
        effective = [max(count / scale, 1.0) for count in counts]
        error = math.sqrt(sum(v / n for v, n in zip(variances, effective, strict=True)))
        return gap / error if error > _EPS else float("nan")

    return z_at(float(window)), z_at(1.0)


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

    def _weekly(self, frame: pd.DataFrame | None) -> pd.DataFrame | None:
        """Weekly-compounded returns, or None when the calendar cannot support them.

        None rather than an exception, because a weekly view is a diagnostic
        the daily numbers can do without: a non-datetime index (a synthetic run
        indexed by bar number) or a sample too short to hold two weeks simply
        means the alignment gap is not reported, not that the portfolio is
        unmeasurable.
        """
        if frame is None or not isinstance(self.sleeves.index, pd.DatetimeIndex):
            return None
        weekly = weekly_returns(frame)
        return weekly if len(weekly) >= 2 else None

    def weekly_correlation(self) -> pd.DataFrame | None:
        """Sleeve-return correlation at weekly frequency, or None if it cannot be formed.

        The gap between this and :meth:`correlation` is the co-movement the
        same-day calendar match could not see. It falls almost entirely on
        cross-market pairs and leaves same-market pairs alone, which is the
        signature of a time-zone artefact rather than a real change in how the
        sleeves move together.
        """
        weekly = self._weekly(self.sleeves)
        return None if weekly is None else weekly.corr()

    def weekly_asset_correlation(self) -> pd.DataFrame | None:
        """Buy-and-hold correlation at weekly frequency, contrasting :meth:`asset_correlation`."""
        weekly = self._weekly(self.assets)
        return None if weekly is None else weekly.corr()

    def drawdown(self) -> pd.Series:
        """Portfolio drawdown from its running peak, at every date.

        Causal by construction: the peak at date ``t`` is the highest equity
        seen up to ``t``, so pairing this with :meth:`rolling_bets` compares
        two quantities that were both knowable on the day.
        """
        equity = self.equity.to_numpy(dtype=float)
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(peak > _EPS, equity / np.where(peak > _EPS, peak, 1.0) - 1.0, np.nan)
        return pd.Series(values, index=self.equity.index, name="drawdown")

    def rolling_bets(self, window: int = ROLLING_WINDOW) -> pd.DataFrame:
        """Rolling diversification, and the drawdown each window ended in.

        :func:`rolling_effective_bets` with the portfolio's own fixed weights,
        plus a ``drawdown`` column so the two can be read together. Empty when
        the shared calendar is shorter than ``window``.
        """
        frame = rolling_effective_bets(self.sleeves, self.weights, window=window)
        if frame.empty:
            frame = frame.copy()
            frame["drawdown"] = pd.Series(dtype=float)
            return frame
        frame = frame.copy()
        frame["drawdown"] = self.drawdown().reindex(frame.index)
        return frame

    def _drawdown_bins(
        self, window: int = ROLLING_WINDOW, n_bins: int = 3
    ) -> tuple[pd.DataFrame, pd.Series, list[str]] | None:
        """The rolling frame, its windows binned by drawdown, and the bin names.

        None when there is nothing to condition on: no rolling view, or a
        drawdown too degenerate to split.
        """
        rolling = self.rolling_bets(window=window)
        if rolling.empty:
            return None
        drawdown = rolling["drawdown"].dropna()
        if len(drawdown.unique()) < 2:
            return None

        bins = pd.qcut(drawdown, min(n_bins, len(drawdown.unique())), duplicates="drop")
        labels = _drawdown_labels(len(bins.cat.categories))
        return rolling, bins.cat.rename_categories(labels), labels

    def bets_by_drawdown(self, window: int = ROLLING_WINDOW, n_bins: int = 3) -> pd.DataFrame:
        """Rolling bet counts grouped into quantile bins of the drawdown they sat in.

        The question the full-sample count cannot answer: is the
        diversification thinner when the portfolio is underwater? Bins are
        quantiles of the drawdown at each window's end, so each holds roughly
        the same number of windows, and they are labelled deepest-first.

        Quantiles, not a threshold. "Underwater by more than x%" invites x to
        be chosen once the answer is visible; a tercile split has nothing to
        choose. Empty when there is no rolling view, or when the drawdown is
        too degenerate to bin - a portfolio that only ever made new highs has
        no stress to condition on.
        """
        binned = self._drawdown_bins(window=window, n_bins=n_bins)
        if binned is None:
            return pd.DataFrame()
        rolling, bins, labels = binned

        rows = []
        for label in labels:
            index = bins[bins == label].index
            rows.append(
                {
                    "regime": label,
                    "n_windows": float(len(index)),
                    "mean_drawdown": float(rolling["drawdown"].reindex(index).mean()),
                    "mean_correlation": float(rolling["mean_correlation"].reindex(index).mean()),
                    "effective_bets": float(rolling["effective_bets"].reindex(index).mean()),
                }
            )
        return pd.DataFrame(rows).set_index("regime")

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
        mean_correlation = float(np.mean(off_diagonal)) if off_diagonal.size else float("nan")
        ratio = diversification_ratio(self.covariance(), self.weights)
        independent = self.sharpe_if_independent()

        if self.assets is None:
            asset_correlation = float("nan")
            asset_bets = float("nan")
        else:
            asset_correlation = _mean_off_diagonal(self.assets.corr())
            asset_bets = effective_number_of_bets(self.assets.cov(ddof=1), self.weights)

        # Same-day correlations understate the co-movement of markets in
        # different time zones; a weekly view sizes how much (see weekly_returns).
        weekly_sleeves = self._weekly(self.sleeves)
        weekly_assets = self._weekly(self.assets)
        n_weeks = float(len(weekly_sleeves)) if weekly_sleeves is not None else float("nan")
        mean_correlation_weekly = (
            _mean_off_diagonal(weekly_sleeves.corr())
            if weekly_sleeves is not None
            else float("nan")
        )
        mean_asset_correlation_weekly = (
            _mean_off_diagonal(weekly_assets.corr()) if weekly_assets is not None else float("nan")
        )

        # A full-sample bet count is an average over calm and stressed alike.
        # The rolling view says whether it collapsed when it mattered.
        rolling = self.rolling_bets()
        binned = self._drawdown_bins()
        deep = shallow = stress_z = stress_z_naive = float("nan")
        if binned is not None:
            frame, bins, labels = binned
            if {"deep_drawdown", "shallow_drawdown"} <= set(labels):
                deep_bets = frame["effective_bets"].reindex(bins[bins == "deep_drawdown"].index)
                shallow_bets = frame["effective_bets"].reindex(
                    bins[bins == "shallow_drawdown"].index
                )
                deep = float(deep_bets.mean())
                shallow = float(shallow_bets.mean())
                stress_z, stress_z_naive = _stress_z(deep_bets, shallow_bets, ROLLING_WINDOW)

        merged.update(
            {
                "n_sleeves": float(len(self.symbols)),
                "rolling_window": float(ROLLING_WINDOW),
                "n_rolling_windows": float(len(rolling)),
                "effective_bets_min": float(rolling["effective_bets"].min())
                if not rolling.empty
                else float("nan"),
                "effective_bets_deep_drawdown": deep,
                "effective_bets_shallow_drawdown": shallow,
                "effective_bets_stress_gap": deep - shallow,
                "effective_bets_stress_z": stress_z,
                "effective_bets_stress_z_naive": stress_z_naive,
                "mean_asset_correlation": asset_correlation,
                "asset_effective_bets": asset_bets,
                "common_fraction": float(self.common_fraction),
                "mean_correlation": mean_correlation,
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
                "n_weeks": n_weeks,
                "mean_correlation_weekly": mean_correlation_weekly,
                "mean_asset_correlation_weekly": mean_asset_correlation_weekly,
                "sleeve_alignment_gap": mean_correlation_weekly - mean_correlation,
                "asset_alignment_gap": mean_asset_correlation_weekly - asset_correlation,
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
