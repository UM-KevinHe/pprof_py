"""Efficient grouping by stratum code.

`cox_partial_likelihood`, `compute_baseline_hazard`, and
`martingale_residuals` all need to iterate "each stratum's rows" many
times over the course of a fit (once per Newton-Raphson evaluation, plus
once each for baseline hazard and residuals). The naive way --
`for s in unique(strata_codes): mask = strata_codes == s` -- rescans the
*entire* array once per stratum, which is O(n * n_strata) overall. For
a fit with many strata (e.g. thousands of facilities) that dominates
runtime long before anything about ties, weights, or truncation does:
profiling a 200k-row, 3000-stratum fit showed this masking pattern
costing far more than the actual risk-set math it was feeding. Sorting
once and slicing contiguous groups out of that sort is O(n log n)
regardless of how many strata there are.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def iter_stratum_indices(strata_codes: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (stratum_code, row_indices) for each distinct value in
    `strata_codes`, in ascending order of the code -- the same order
    `np.unique(strata_codes)` would give, and the order this package's
    `strata_labels` are indexed by (see data/validation.py).

    `row_indices` is an int array suitable for fancy-indexing X, start,
    stop, etc. (`X[row_indices]`), not a boolean mask.
    """
    order = np.argsort(strata_codes, kind="mergesort")
    sorted_codes = strata_codes[order]
    unique_codes, group_starts = np.unique(sorted_codes, return_index=True)
    group_ends = np.append(group_starts[1:], len(order))

    for code, start, end in zip(unique_codes, group_starts, group_ends):
        yield int(code), order[start:end]
