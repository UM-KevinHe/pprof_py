"""Size-based grouping of providers for groupwise empirical-null calibration."""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["assign_groups"]


def assign_groups(size, n_groups: int = 4, *, rule: str = "quantile", order=None) -> np.ndarray:
    """Assign providers to size-based groups.

    Parameters
    ----------
    size : array-like
        Size variable used for grouping, one value per provider (for example
        patient counts). It need not be the model's own row count.
    n_groups : int
        Number of groups.
    rule : {"quantile", "rank"}
        ``"quantile"`` (the default, as in EmpiNull) places providers by
        empirical quantile breaks of size (type 7, as ``numpy.quantile``'s
        linear method); a size equal to a break goes to the lower group. It
        requires finite sizes and distinct breaks.

        ``"rank"`` sorts providers by size and cuts the sorted list into
        consecutive blocks of ``ceil(n / n_groups)`` providers; the last block
        holds the remainder and can be smaller. Missing sizes sort last and
        occupy the final positions, as R's ``order()`` does.
    order : array-like, optional
        Tie-break keys for ``rule="rank"`` (for example provider IDs):
        providers with equal size are ordered by this key. When omitted, ties
        keep their input order, which is what R's stable ``order()`` does, so
        the result then depends on how the input rows happen to be sorted.

    Returns
    -------
    numpy.ndarray of int
        Group labels 1, ..., ``n_groups``.
    """
    if int(n_groups) != n_groups or n_groups < 1:
        raise ValueError("n_groups must be a positive integer.")
    n_groups = int(n_groups)
    size = np.asarray(size, dtype=np.float64).ravel()
    n = size.size
    if n == 0:
        return np.zeros(0, dtype=int)
    if n_groups == 1:
        return np.ones(n, dtype=int)

    if rule == "rank":
        if order is None:
            position = np.argsort(size, kind="stable")  # NaN sorts last
        else:
            key = np.asarray(order).ravel()
            if key.shape != size.shape:
                raise ValueError("order must have one entry per provider.")
            key_rank = np.unique(key, return_inverse=True)[1].ravel()
            missing = np.isnan(size)
            position = np.lexsort((np.arange(n), key_rank, np.where(missing, 0.0, size), missing))
        block = int(np.ceil(n / n_groups))
        labels = np.empty(n, dtype=int)
        labels[position] = np.minimum(np.arange(n) // block, n_groups - 1) + 1
        return labels

    if rule == "quantile":
        if not np.all(np.isfinite(size)):
            raise ValueError("rule='quantile' requires finite sizes.")
        probs = np.linspace(0.0, 1.0, n_groups + 1)[1:n_groups]
        breaks = np.quantile(size, probs, method="linear")
        if np.unique(breaks).size < breaks.size:
            raise ValueError("Quantile breaks are not distinct; grouping would be ambiguous. "
                             "Use rule='rank' or fewer groups.")
        return np.searchsorted(breaks, size, side="left").astype(int) + 1

    raise ValueError("rule must be 'rank' or 'quantile'.")
