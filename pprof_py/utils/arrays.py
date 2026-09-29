"""Array validation used by pprof_py (replacing scikit-learn's ``check_array``)."""
from __future__ import annotations

import numpy as np

__all__ = ["check_array"]


def check_array(x, *, ensure_2d: bool = True, dtype="numeric") -> np.ndarray:
    """Convert to an ndarray and validate it.

    ``dtype="numeric"`` keeps numeric dtypes and converts object arrays to
    float64; another dtype converts to it. Rejects 0-d and >2-d input, NaN or
    infinity in floating arrays, and empty arrays; ``ensure_2d`` requires 2-d.
    """
    a = np.asarray(x)
    if dtype == "numeric":
        if a.dtype.kind == "O":
            a = a.astype(np.float64)
    elif dtype is not None:
        a = a.astype(dtype)
    if a.ndim == 0:
        raise ValueError("Expected an array-like, got a scalar.")
    if a.ndim > 2:
        raise ValueError(f"Found array with dim {a.ndim}. Expected <= 2.")
    if ensure_2d and a.ndim != 2:
        raise ValueError(f"Expected 2D array, got {a.ndim}D array instead.")
    if a.dtype.kind in "fc" and not np.isfinite(a).all():
        raise ValueError("Input contains NaN or infinity.")
    if a.shape[0] < 1 or (ensure_2d and a.shape[1] < 1):
        raise ValueError(f"Found array with shape {a.shape}; at least one sample and feature are required.")
    return a
