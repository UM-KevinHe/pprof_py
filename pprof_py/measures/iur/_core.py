"""Core variance decomposition for IUR estimation.

Provides the ANOVA-style variance decomposition shared by
:class:`BootstrapIUR` and :class:`DirectIUR`, as well as the
decile-table helper used by both classes.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, list, tuple, pd.Series]


class IURDecomposition(NamedTuple):
    """Results of the ANOVA-style variance decomposition.

    Attributes
    ----------
    iur : float
        Overall inter-unit reliability.
    n_groups : int
        Number of groups (facilities).
    iur_groups : ndarray of shape (n_groups,)
        Per-group reliability.
    s2_between : float
        Between-group variance component (signal).
    s2_within : float
        Within-group variance component (noise), scaled by ``n_prime``.
    n_prime : float
        Effective sample size per group.
    """

    iur: float
    n_groups: int
    iur_groups: np.ndarray
    s2_between: float
    s2_within: float
    n_prime: float


def _iur_decomposition(
    sizes: np.ndarray,
    within_variances: np.ndarray,
    measure_values: np.ndarray,
) -> IURDecomposition:
    """ANOVA-style variance decomposition for inter-unit reliability.

    Decomposes the total variance of a group-level measure into
    between-group (signal) and within-group (noise) components, then
    computes ``IUR = signal / total``.

    Parameters
    ----------
    sizes : ndarray of shape (n_groups,)
        Number of observations per group.
    within_variances : ndarray of shape (n_groups,)
        Per-group within-variance estimates.  For bootstrap IUR these
        are the variances of bootstrap replicates; for direct IUR
        these are the squared standard errors.
    measure_values : ndarray of shape (n_groups,)
        Original (non-bootstrapped) group-level measure values.

    Returns
    -------
    decomp : IURDecomposition
        Named tuple with variance components and IUR values.

    Notes
    -----
    The returned ``s2_within`` is the pooled within-group variance
    multiplied by ``n_prime``, following the R convention.  The
    per-group IUR uses the *unscaled* pooled variance:
    ``s2_b / (s2_b + s2_w_raw / n_i)``.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    within_variances = np.asarray(within_variances, dtype=np.float64)
    measure_values = np.asarray(measure_values, dtype=np.float64)

    n_groups = len(sizes)
    total_n = sizes.sum()

    # ── Pooled within-group variance (weighted by d.f.) ──
    weights = np.maximum(sizes - 1.0, 0.0)
    weight_sum = weights.sum()
    s2_w_raw = (
        (weights * within_variances).sum() / weight_sum
        if weight_sum > 0
        else 0.0
    )

    # ── Effective sample size ──
    n_prime = (total_n - (sizes**2).sum() / total_n) / (n_groups - 1)

    # ── Weighted mean of original measures ──
    t_mean = (sizes * measure_values).sum() / total_n

    # ── Total variance ──
    s2_t = (sizes * (measure_values - t_mean) ** 2).sum() / (
        n_prime * (n_groups - 1)
    )

    # ── Between-group variance (signal) ──
    s2_b = s2_t - s2_w_raw

    # ── Overall IUR ──
    iur = s2_b / s2_t if s2_t > 0 else 0.0

    # ── Per-group IUR: s2_b / (s2_b + s2_w_raw / n_i) ──
    denom = s2_b + s2_w_raw / sizes
    with np.errstate(divide="ignore", invalid="ignore"):
        iur_groups = np.where(denom != 0, s2_b / denom, 0.0)

    return IURDecomposition(
        iur=float(iur),
        n_groups=int(n_groups),
        iur_groups=iur_groups,
        s2_between=float(s2_b),
        s2_within=float(s2_w_raw * n_prime),
        n_prime=float(n_prime),
    )


def _compute_decile_table(
    sizes: np.ndarray,
    s2_between: float,
    s2_within: float,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Compute IUR at representative group sizes (min, decile means, max).

    Uses the fitted variance components to compute the expected
    reliability for groups of different sizes.

    Parameters
    ----------
    sizes : ndarray of shape (n_groups,)
        Group sizes (or a stratification variable aggregated to group
        level).
    s2_between : float
        Between-group variance (signal).
    s2_within : float
        Within-group variance (noise), **already scaled by n_prime**.
    n_quantiles : int, default=10
        Number of quantile groups.

    Returns
    -------
    table : DataFrame
        Single-row frame with columns ``min``, ``decile 1`` …
        ``decile K``, ``max``.
    """
    sizes = np.asarray(sizes, dtype=np.float64)

    breaks = np.unique(
        np.quantile(sizes, np.linspace(0, 1, n_quantiles + 1))
    )
    labels = np.digitize(sizes, breaks[1:-1], right=True) + 1
    labels = np.clip(labels, 1, len(breaks) - 1)

    unique_labels = np.sort(np.unique(labels))
    avg_per_group = np.array(
        [sizes[labels == g].mean() for g in unique_labels]
    )

    representative = np.concatenate(
        [[sizes.min()], avg_per_group, [sizes.max()]]
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        iur_values = np.where(
            (s2_between + s2_within / representative) != 0,
            s2_between / (s2_between + s2_within / representative),
            0.0,
        )

    n_actual = len(unique_labels)
    columns = (
        ["min"]
        + [f"decile {i + 1}" for i in range(n_actual)]
        + ["max"]
    )
    return pd.DataFrame([iur_values], columns=columns)
