"""Hyper-parameter selection inside each walk-forward fold.

Limitation 1 in ``docs/methodology.md`` says the feature set, model list and
hyper-parameters were chosen by looking at this data, and that no walk-forward
can remove that. That is true of the *first* two and only half true of the
third. Hyper-parameters picked once, on the full sample, and then evaluated
out-of-sample report a number no live run could have produced: the live run
would have had to pick them from the past. Picking them inside each fold, from
the fold's own training bars, closes that gap - not the selection bias in the
grid itself, which is still a human choice made with this data in view, but the
layer where the *winner* was chosen knowing how it did on the test bars.

**The inner selection is itself a walk-forward.** The obvious alternative, a
single hold-out at the end of the training window, picks hyper-parameters on
one stretch of market, which is the fragility this project spends its effort
measuring elsewhere. :class:`TuningConfig` instead carves the last
``inner_fraction`` of each training window into ``n_inner_folds`` validation
windows and runs the ordinary :class:`~ai_stock.evaluation.walkforward.
WalkForwardSplitter` over them, so the embargo between inner train and inner
validation is the same tested code path as the outer one. Nothing outside the
outer training window is ever read.

**What it costs and what it buys.** The fit count multiplies by
``len(grid) * n_inner_folds + 1`` per outer fold, and the reported performance
usually goes *down* relative to fixed hyper-parameters. That drop is the bias
being removed, not a regression, and it is the only reason to do this.

**Read the stability table before the performance table.** If the winning
value changes every fold, the selection is reading noise, and the honest claim
is not "these hyper-parameters work" but "this *procedure*, including its
instability, earns this much". :meth:`~ai_stock.evaluation.walkforward.
WalkForwardResult.selection_stability` is what makes that visible.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["TuningConfig", "describe_params", "expand_grid", "inner_schedule"]


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Every combination of ``grid``, in a deterministic order.

    Key order follows the mapping and value order follows each sequence, so a
    tie between candidates always resolves to the same one - a tie broken at
    random would make the whole walk-forward unreproducible.

    >>> expand_grid({"alpha": [0.1, 1.0], "fit_intercept": [True]})
    [{'alpha': 0.1, 'fit_intercept': True}, {'alpha': 1.0, 'fit_intercept': True}]

    An empty grid is one candidate - the defaults - not zero:

    >>> expand_grid({})
    [{}]
    """
    keys = list(grid)
    if not keys:
        return [{}]
    for key in keys:
        if len(grid[key]) == 0:
            raise ValueError(f"hyper-parameter {key!r} has an empty list of values")
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def describe_params(params: Mapping[str, Any]) -> str:
    """A short, stable label for one candidate, for tables and logs.

    >>> describe_params({"alpha": 1.0, "fit_intercept": True})
    'alpha=1.0, fit_intercept=True'
    >>> describe_params({})
    'defaults'
    """
    if not params:
        return "defaults"
    return ", ".join(
        f"{key}={value!r}" if isinstance(value, str) else f"{key}={value}"
        for key, value in params.items()
    )


@dataclass(frozen=True)
class TuningConfig:
    """How hyper-parameters are chosen inside each outer training window.

    Attributes
    ----------
    grid:
        Name -> candidate values, passed as keyword arguments to the model
        factory. An empty grid disables selection entirely.
    n_inner_folds:
        Validation windows carved out of the end of each outer training
        window. One is permitted and is a single hold-out, with the fragility
        this module's docstring describes; the default of three is a
        compromise between that and the fit count.
    inner_fraction:
        Share of the outer training window given over to inner validation.
        The rest trains the first inner fold, and the schedule is expanding,
        so later inner folds see more.
    metric:
        Key of :func:`~ai_stock.evaluation.metrics.regression_metrics` or
        :func:`~ai_stock.evaluation.metrics.classification_metrics` that
        candidates are ranked by, averaged across the inner folds.
    higher_is_better:
        Direction of ``metric``. Set ``False`` for error metrics such as
        ``rmse``.
    min_inner_train:
        Floor on the first inner fold's training bars, below which
        :func:`inner_schedule` refuses to split and the caller keeps the
        model's defaults. This is a judgement call and is stated as one: the
        arithmetic of ``inner_fraction`` scales with the window, so it never
        produces a natural break point, and an unconstrained split will
        happily fit a thirty-feature model on twenty bars and report a winner.
        The default of 60 bars is roughly a quarter of trading days; below it
        the fit is noise whatever the hyper-parameters, and a selection made
        on it would be a coin flip dressed as a choice.

    The selection metric is deliberately *not* a backtest Sharpe ratio.
    Sharpe reads the cost model, the sizing rule and the leverage cap as well
    as the forecast, so tuning on it lets a candidate win by suiting the
    trading configuration rather than by predicting - and the trading
    configuration is the one thing downstream of this that a user changes
    freely.
    """

    grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    n_inner_folds: int = 3
    inner_fraction: float = 0.3
    metric: str = "ic_pearson"
    higher_is_better: bool = True
    min_inner_train: int = 60

    def __post_init__(self) -> None:
        if self.n_inner_folds < 1:
            raise ValueError("n_inner_folds must be >= 1")
        if not 0.0 < self.inner_fraction < 1.0:
            raise ValueError("inner_fraction must lie strictly inside (0, 1)")
        if not self.metric:
            raise ValueError("metric must be a non-empty metric name")
        if self.min_inner_train < 1:
            raise ValueError("min_inner_train must be >= 1")

    @property
    def enabled(self) -> bool:
        """Whether there is anything to select between.

        >>> TuningConfig().enabled
        False
        >>> TuningConfig(grid={"alpha": [0.1, 1.0]}).enabled
        True
        """
        return bool(self.grid) and len(self.candidates()) > 1

    def candidates(self) -> list[dict[str, Any]]:
        """The grid expanded, in the order ties are broken."""
        return expand_grid(self.grid)

    def better(self, candidate: float, incumbent: float) -> bool:
        """Whether ``candidate`` beats ``incumbent`` under this config's direction.

        A non-finite candidate never wins: a fold where the metric could not
        be computed is not evidence for the hyper-parameters that produced it.

        >>> TuningConfig().better(0.2, 0.1)
        True
        >>> TuningConfig(metric="rmse", higher_is_better=False).better(0.2, 0.1)
        False
        >>> TuningConfig().better(float("nan"), -1.0)
        False
        """
        if not np.isfinite(candidate):
            return False
        if not np.isfinite(incumbent):
            return True
        return candidate > incumbent if self.higher_is_better else candidate < incumbent


def inner_schedule(n_train: int, embargo: int, config: TuningConfig) -> tuple[int, int] | None:
    """``(train_size, test_size)`` for the inner walk-forward, or ``None``.

    Solves for a schedule that fits exactly ``n_inner_folds`` abutting
    validation windows inside ``n_train`` bars: with an initial training block
    of ``t0`` bars, an embargo of ``embargo`` and ``k`` windows of ``m`` bars
    each, the last window ends at ``t0 + k * m + embargo``, so ``t0 = n_train -
    embargo - k * m``.

    ``m`` is set from ``inner_fraction`` and then floored at one bar. Returns
    ``None`` when the split would leave the first inner fold below
    ``min_inner_train`` bars, or with less training data than it validates on
    - the point at which selection is guessing. The caller then keeps the
    model's defaults rather than pretending to have chosen.

    >>> inner_schedule(750, 5, TuningConfig(grid={"alpha": [0.1, 1.0]}))
    (520, 75)

    A training window too short for the requested folds selects nothing:

    >>> inner_schedule(40, 5, TuningConfig(grid={"alpha": [0.1, 1.0]})) is None
    True
    """
    folds = config.n_inner_folds
    usable = n_train - embargo
    if usable < folds * 2:
        return None
    test_size = max(1, int(n_train * config.inner_fraction) // folds)
    train_size = usable - folds * test_size
    if train_size < max(test_size, config.min_inner_train):
        return None
    return train_size, test_size
