"""Corrections for testing many hypotheses at once.

Screening a universe multiplies the chance of a false positive: test ten
tickers at the 5% level and you expect one to look significant even when none
of them has an edge. Reporting a raw p-value per ticker after ranking by that
same p-value is therefore not just optimistic, it is the standard way people
convince themselves a strategy works.

The corrections here are the minimum needed to read a screen honestly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["benjamini_hochberg", "bonferroni_threshold", "expected_false_positives"]


def benjamini_hochberg(p_values: np.ndarray | list[float]) -> np.ndarray:
    """False-discovery-rate adjusted q-values (Benjamini & Hochberg, 1995).

    A q-value of 0.10 means: if you accept every result at least this strong,
    about 10% of what you accept will be noise. Less conservative than
    Bonferroni, which is the right trade-off when screening a universe where
    a few real edges may exist.

    ``NaN`` inputs are passed through as ``NaN`` and excluded from the ranking,
    so a ticker whose test could not be run does not inflate everyone else's
    correction.

    >>> benjamini_hochberg([0.01, 0.02, 0.03, 0.9]).round(4).tolist()
    [0.04, 0.04, 0.04, 0.9]
    """
    values = np.asarray(p_values, dtype=float).ravel()
    result = np.full(values.shape, np.nan)

    finite = np.isfinite(values)
    if not finite.any():
        return result
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError("p-values must lie in [0, 1]")

    tested = values[finite]
    m = len(tested)
    order = np.argsort(tested)
    ranked = tested[order]

    # q_(i) = min over j >= i of (m / j) * p_(j), enforced monotone from the top.
    scaled = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_tested = np.empty(m)
    q_tested[order] = q_sorted
    result[finite] = q_tested
    return result


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """The per-test p-value needed for family-wise significance at ``alpha``.

    >>> bonferroni_threshold(10)
    0.005
    """
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return alpha / n_tests


def expected_false_positives(n_tests: int, alpha: float = 0.05) -> float:
    """How many "significant" results pure noise would hand you.

    The number to quote before celebrating the best name in a screen.

    >>> expected_false_positives(20)
    1.0
    """
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return n_tests * alpha
