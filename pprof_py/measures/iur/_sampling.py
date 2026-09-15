"""Private sampling utilities for IUR estimation."""
from __future__ import annotations

import numpy as np


def _stratified_bootstrap(
    group_sizes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample with replacement within each group (stratified bootstrap).

    Assumes the underlying data is pre-sorted by group.  Returns 0-based
    indices into the sorted data array.

    Parameters
    ----------
    group_sizes : ndarray of shape (n_groups,)
        Number of observations in each group.
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    indices : ndarray of shape (n_total,)
        Resampled row indices (0-based).
    """
    n_total = int(group_sizes.sum())
    offsets = np.empty(len(group_sizes) + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(group_sizes, out=offsets[1:])

    indices = np.empty(n_total, dtype=np.intp)
    for i, size in enumerate(group_sizes):
        start = offsets[i]
        indices[start : start + size] = start + rng.integers(
            0, size, size=size
        )
    return indices


def _split_half_sample(
    group_sizes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample half of each group without replacement.

    Assumes the underlying data is pre-sorted by group.  Returns indices
    for the first half; the complement (all indices NOT in the returned
    array) forms the second half.

    Parameters
    ----------
    group_sizes : ndarray of shape (n_groups,)
        Number of observations in each group.
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    indices : ndarray of shape (sum(floor(size_i / 2)),)
        Row indices (0-based) for the first half of each group.
    """
    offsets = np.empty(len(group_sizes) + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(group_sizes, out=offsets[1:])

    half_sizes = group_sizes // 2
    total_half = int(half_sizes.sum())
    indices = np.empty(total_half, dtype=np.intp)

    pos = 0
    for i, size in enumerate(group_sizes):
        half = int(half_sizes[i])
        start = offsets[i]
        perm = rng.permutation(size)[:half]
        indices[pos : pos + half] = start + perm
        pos += half
    return indices
