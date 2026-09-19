"""Likelihood-agnostic penalty primitives and column standardization.

Pure NumPy, no dependence on any particular likelihood engine -- this sits
at the numerical-kernel layer and is deliberately kept separate from
``coordinate_descent.py`` so the penalty *math* and column scaling can
be tested independently of the optimization loop.

Two penalty families are supported:

1. **Elastic net** (for ``PenalizedLogistic``, ``PenalizedLinear``,
   ``PenalizedCoxPH``)::

       lambda * sum_j pf_j * [alpha * |beta_j| + 0.5*(1-alpha) * beta_j^2]

2. **Sparse group lasso** (for ``GroupLassoLogistic``, ``GroupLassoLinear``,
   ``GroupLassoCoxPH``)::

       lambda * [(1-alpha) * sum_g m_g * ||beta_g||_2
                 +  alpha  * sum_j pf_j * |beta_j|           ]

   When alpha=0 this is pure group lasso (Yuan & Lin 2006); when
   alpha=1 it reduces to element-wise lasso (groups ignored);
   intermediate values give the sparse group lasso of Simon et al.
   (2013).

Standardization and rescaling conventions match glmnet 4.1-8.
Group multiplier conventions match grplasso.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Numba availability check (standalone, no dependency on survival module)
# -----------------------------------------------------------------------
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def _njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator when Numba is not installed."""
        def _decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _decorator

njit = _njit


# -----------------------------------------------------------------------
# Element-wise soft-thresholding
# -----------------------------------------------------------------------

def soft_threshold(z: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    """Elementwise soft-thresholding operator: sign(z) * max(|z| - thresh, 0).

    Parameters
    ----------
    z : ndarray
        Input values.
    thresh : ndarray
        Non-negative threshold (broadcasts against ``z``).

    Returns
    -------
    ndarray
        Soft-thresholded values.
    """
    return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)


# -----------------------------------------------------------------------
# Elastic net penalty value
# -----------------------------------------------------------------------

def elastic_net_penalty_value(
    beta: np.ndarray, alpha: float, penalty_factor: np.ndarray
) -> float:
    """Compute sum_j penalty_factor_j * [alpha*|beta_j| + 0.5*(1-alpha)*beta_j^2].

    No lambda factor -- the caller multiplies by lambda separately.
    Matches glmnet's ``pen_function``.

    Parameters
    ----------
    beta : ndarray, shape (p,)
        Coefficient vector.
    alpha : float
        Elastic net mixing (1 = lasso, 0 = ridge).
    penalty_factor : ndarray, shape (p,)
        Per-variable penalty weights.

    Returns
    -------
    float
        Penalty value (>= 0).
    """
    return float(
        np.sum(penalty_factor * (alpha * np.abs(beta) + 0.5 * (1.0 - alpha) * beta**2))
    )


# -----------------------------------------------------------------------
# Column standardization
# -----------------------------------------------------------------------

def weighted_column_scale(
    X: np.ndarray, weight: np.ndarray, standardize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column scale factors for penalized fitting.

    Computes weighted population standard deviation per column.  Does
    NOT center -- for Cox (no intercept) this matches glmnet exactly;
    for GLMs with intercept the centering is handled by the intercept
    update, not by X standardization.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix.
    weight : ndarray, shape (n,)
        Per-observation weights (positive).
    standardize : bool
        If False, returns all-ones scale (no standardization).

    Returns
    -------
    xs : ndarray, shape (p,)
        Scale divisor per column (weighted population std if
        ``standardize``, else 1.0).  Degenerate columns get xs=1.
    degenerate : ndarray of bool, shape (p,)
        True for columns with numerically zero weighted variance.
    """
    w = np.asarray(weight, dtype=np.float64)
    w_sum = w.sum()
    w_norm = w / w_sum
    xm = w_norm @ X
    centered = X - xm
    xv = w_norm @ (centered**2)
    degenerate = xv < 10.0 * np.finfo(np.float64).eps
    xv = np.where(degenerate, 0.0, xv)
    if standardize:
        xs = np.sqrt(xv)
    else:
        xs = np.ones(X.shape[1])
    xs = np.where(degenerate, 1.0, xs)
    return xs, degenerate


# -----------------------------------------------------------------------
# Penalty factor rescaling
# -----------------------------------------------------------------------

def rescale_penalty_factors(penalty_factor: Optional[np.ndarray], p: int) -> np.ndarray:
    """Validate and rescale user-supplied penalty factors to sum to ``p``.

    ``penalty_factor=None`` gives the all-ones default.  A 0 marks an
    always-unpenalized variable and is preserved exactly.  Matches
    glmnet's "penalty factors are internally rescaled to sum to nvars".

    Parameters
    ----------
    penalty_factor : ndarray or None, shape (p,)
        User-supplied per-variable penalty weights.
    p : int
        Number of features.

    Returns
    -------
    ndarray, shape (p,)
        Rescaled penalty factors summing to ``p``.

    Raises
    ------
    ValueError
        If shape mismatch, negative values, or all-zero.
    """
    if penalty_factor is None:
        return np.ones(p)
    pf = np.asarray(penalty_factor, dtype=np.float64)
    if pf.shape != (p,):
        raise ValueError(f"penalty_factor must have shape ({p},), got {pf.shape}")
    if np.any(pf < 0):
        raise ValueError("penalty_factor must be non-negative")
    total = pf.sum()
    if total <= 0:
        raise ValueError(
            "penalty_factor cannot be all-zero (nothing would be penalized)"
        )
    return pf * (p / total)


# -----------------------------------------------------------------------
# Group lasso / sparse group lasso primitives
# -----------------------------------------------------------------------

def validate_groups(
    groups: np.ndarray, p: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Validate and canonicalize group labels.

    Group labels: 0 = unpenalized, positive integers = penalized groups.
    Labels are remapped to contiguous 1..G.

    Parameters
    ----------
    groups : array-like, shape (p,)
        Integer group labels.
    p : int
        Number of features.

    Returns
    -------
    groups : ndarray of int, shape (p,)
        Canonicalized group labels (0 for unpenalized, 1..G).
    group_sizes : ndarray of int, shape (G,)
        Features per penalized group.
    n_groups : int
        Number of distinct penalized groups (G).
    """
    groups = np.asarray(groups)
    if groups.shape != (p,):
        raise ValueError(f"groups must have shape ({p},), got {groups.shape}")
    if not np.issubdtype(groups.dtype, np.integer):
        if np.issubdtype(groups.dtype, np.floating):
            if not np.all(groups == np.floor(groups)):
                raise ValueError(
                    "groups must contain integer labels, got non-integer "
                    "floating-point values"
                )
            groups = groups.astype(np.intp)
        else:
            raise ValueError(
                f"groups must contain integer labels, got dtype {groups.dtype}"
            )
    else:
        groups = groups.astype(np.intp, copy=True)

    if np.any(groups < 0):
        raise ValueError(
            "groups must be non-negative (0 = unpenalized, positive = "
            "penalized group label)"
        )
    unique_labels = np.unique(groups[groups > 0])
    n_groups = len(unique_labels)
    if n_groups == 0:
        raise ValueError(
            "groups must contain at least one positive label (all-zero "
            "means nothing is penalized)"
        )
    label_map = np.zeros(unique_labels.max() + 1, dtype=np.intp)
    for new_label, old_label in enumerate(unique_labels, start=1):
        label_map[old_label] = new_label
    penalized_mask = groups > 0
    groups[penalized_mask] = label_map[groups[penalized_mask]]
    group_sizes = np.empty(n_groups, dtype=np.intp)
    for g in range(1, n_groups + 1):
        group_sizes[g - 1] = np.sum(groups == g)
    return groups, group_sizes, n_groups


def rescale_group_multipliers(
    group_multiplier: Optional[np.ndarray],
    group_sizes: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Validate and default group multipliers.

    Default: ``sqrt(group_sizes)`` per Yuan & Lin (2006).  Group
    multipliers are used as-is (not rescaled to sum to any value),
    matching grplasso's convention.

    Parameters
    ----------
    group_multiplier : ndarray or None, shape (n_groups,)
        Per-group penalty multipliers.
    group_sizes : ndarray of int, shape (n_groups,)
        Features per penalized group.
    n_groups : int
        Number of penalized groups.

    Returns
    -------
    ndarray of float, shape (n_groups,)
        Validated or defaulted group multipliers.
    """
    if group_multiplier is None:
        return np.sqrt(group_sizes.astype(np.float64))
    gm = np.asarray(group_multiplier, dtype=np.float64)
    if gm.shape != (n_groups,):
        raise ValueError(
            f"group_multiplier must have shape ({n_groups},), got {gm.shape}"
        )
    if np.any(gm < 0):
        raise ValueError("group_multiplier must be non-negative")
    if np.all(gm == 0):
        raise ValueError(
            "group_multiplier cannot be all-zero (nothing would be "
            "penalized at the group level)"
        )
    return gm


def sparse_group_lasso_penalty_value(
    beta: np.ndarray,
    alpha: float,
    groups: np.ndarray,
    group_weights: np.ndarray,
    penalty_factor: np.ndarray,
) -> float:
    """Sparse group lasso penalty value (no lambda factor).

    Computes::

        (1 - alpha) * sum_{g=1}^{G} m_g * ||beta_g||_2
        +    alpha  * sum_{j=1}^{p}  pf_j * |beta_j|

    Parameters
    ----------
    beta : ndarray, shape (p,)
        Coefficient vector.
    alpha : float
        Mixing (0 = pure group lasso, 1 = pure lasso).
    groups : ndarray of int, shape (p,)
        Canonicalized group labels.
    group_weights : ndarray, shape (G,)
        Per-group penalty multipliers.
    penalty_factor : ndarray, shape (p,)
        Per-element penalty factors.

    Returns
    -------
    float
        Penalty value (>= 0).
    """
    val = 0.0
    if alpha < 1.0:
        n_groups = len(group_weights)
        for g in range(1, n_groups + 1):
            mask = groups == g
            beta_g = beta[mask]
            norm_g = np.sqrt(np.dot(beta_g, beta_g))
            val += group_weights[g - 1] * norm_g
        val *= (1.0 - alpha)
    if alpha > 0.0:
        val += alpha * float(np.sum(penalty_factor * np.abs(beta)))
    return val


def group_soft_threshold(
    z: np.ndarray, threshold: float
) -> np.ndarray:
    """Group-level soft-thresholding: ``(1 - threshold/||z||_2)_+ * z``.

    Proximal operator of the scaled L2 norm ``threshold * ||.||_2``.

    Parameters
    ----------
    z : ndarray, shape (K,)
        Input vector.
    threshold : float
        Non-negative threshold.

    Returns
    -------
    ndarray, shape (K,)
        Soft-thresholded vector (zero if ``||z||_2 <= threshold``).
    """
    norm_z = np.sqrt(np.dot(z, z))
    if norm_z <= threshold:
        return np.zeros_like(z)
    return z * (1.0 - threshold / norm_z)


def sparse_group_prox(
    z: np.ndarray,
    diag_A: np.ndarray,
    lam: float,
    alpha: float,
    group_weight: float,
    penalty_factor_g: np.ndarray,
) -> np.ndarray:
    """Proximal operator for one group under the sparse group lasso.

    Two-step nested soft-thresholding:
    1. Element-wise L1 shrinkage (sparse part)
    2. Group L2 shrinkage (group part)

    Parameters
    ----------
    z : ndarray, shape (K,)
        Partial residual for the group.
    diag_A : ndarray, shape (K,)
        Diagonal entries of the quadratic model.
    lam : float
        Regularization strength.
    alpha : float
        Mixing (0 = group, 1 = lasso).
    group_weight : float
        This group's penalty multiplier.
    penalty_factor_g : ndarray, shape (K,)
        Per-element penalty factors for this group.

    Returns
    -------
    ndarray, shape (K,)
        Updated coefficient block.
    """
    K = len(z)
    s = np.empty(K)
    for j in range(K):
        thresh_j = lam * alpha * penalty_factor_g[j]
        abs_z = abs(z[j])
        if abs_z <= thresh_j:
            s[j] = 0.0
        else:
            sign_z = 1.0 if z[j] > 0.0 else -1.0
            s[j] = sign_z * (abs_z - thresh_j) / diag_A[j]
    group_thresh = lam * (1.0 - alpha) * group_weight
    if group_thresh <= 0.0:
        return s
    return group_soft_threshold(s, group_thresh)


# -----------------------------------------------------------------------
# Within-group orthogonalization (optional, for MM method)
# -----------------------------------------------------------------------

def within_group_orthogonalize(
    X: np.ndarray,
    groups: np.ndarray,
    weight: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
    """SVD-based within-group orthogonalization.

    For each penalized group g, computes the weighted SVD and replaces
    it with an orthonormalized version such that
    ``X_g.T @ diag(w) @ X_g / sum(w) = I``.

    Follows grplasso's ``orthogonalize`` convention.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix.
    groups : ndarray of int, shape (p,)
        Canonicalized group labels.
    weight : ndarray, shape (n,)
        Per-observation sample weights.

    Returns
    -------
    X_orth : ndarray, shape (n, p)
        Orthogonalized design matrix.
    QL_blocks : list of (group_label, ndarray)
        Per-group back-transformation matrices.
    """
    n, p = X.shape
    w_sum = weight.sum()
    sqrt_w = np.sqrt(weight / w_sum)
    X_orth = X.copy()
    QL_blocks: List[Tuple[int, np.ndarray]] = []
    n_groups = groups.max()
    for g in range(1, n_groups + 1):
        idx = np.where(groups == g)[0]
        K = len(idx)
        if K == 0:
            continue
        X_g_w = sqrt_w[:, np.newaxis] * X[:, idx]
        _U, d, Vt = np.linalg.svd(X_g_w, full_matrices=False)
        tol = max(n, K) * np.finfo(np.float64).eps * d[0] if d[0] > 0 else 0.0
        r = int(np.sum(d > tol))
        if r == 0:
            X_orth[:, idx] = 0.0
            QL_blocks.append((g, np.zeros((K, 0))))
            logger.warning(
                "Group %d has rank 0 after weighted SVD; columns zeroed.", g
            )
            continue
        V_r = Vt[:r, :].T
        d_r = d[:r]
        Q = V_r * (1.0 / d_r)[np.newaxis, :]
        X_g_orth = X[:, idx] @ Q
        if r < K:
            X_orth[:, idx[:r]] = X_g_orth
            X_orth[:, idx[r:]] = 0.0
            logger.info(
                "Group %d has rank %d < size %d; %d columns zeroed.",
                g, r, K, K - r,
            )
        else:
            X_orth[:, idx] = X_g_orth
        QL_blocks.append((g, Q))
    return X_orth, QL_blocks


def unorthogonalize_coefs(
    beta: np.ndarray,
    groups: np.ndarray,
    QL_blocks: List[Tuple[int, np.ndarray]],
) -> np.ndarray:
    """Map coefficients from orthogonalized space back to original.

    Parameters
    ----------
    beta : ndarray, shape (p,)
        Coefficient vector in the orthogonalized space.
    groups : ndarray of int, shape (p,)
        Canonicalized group labels.
    QL_blocks : list of (group_label, ndarray)
        Back-transformation matrices from ``within_group_orthogonalize``.

    Returns
    -------
    ndarray, shape (p,)
        Coefficients in the original feature space.
    """
    beta_orig = beta.copy()
    for g, Q in QL_blocks:
        idx = np.where(groups == g)[0]
        K = len(idx)
        r = Q.shape[1] if Q.ndim == 2 else 0
        if r == 0:
            beta_orig[idx] = 0.0
            continue
        beta_orth_g = beta[idx[:r]]
        beta_orig[idx] = 0.0
        beta_orig[idx] = Q @ beta_orth_g
    return beta_orig


# -----------------------------------------------------------------------
# Group-level index helpers
# -----------------------------------------------------------------------

def compute_group_indices(
    groups: np.ndarray, n_groups: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute start/end column indices for each penalized group.

    Assumes features within each group are contiguous in column ordering.

    Parameters
    ----------
    groups : ndarray of int, shape (p,)
        Canonicalized group labels.
    n_groups : int
        Number of penalized groups.

    Returns
    -------
    group_starts : ndarray of int, shape (G,)
        Starting column index (inclusive) per group.
    group_ends : ndarray of int, shape (G,)
        Ending column index (exclusive) per group.
    """
    group_starts = np.empty(n_groups, dtype=np.intp)
    group_ends = np.empty(n_groups, dtype=np.intp)
    for g in range(1, n_groups + 1):
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            raise ValueError(f"Penalized group {g} has no features")
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
            raise ValueError(
                f"Features in penalized group {g} are not contiguous "
                f"(columns {idx.tolist()}). Permute the design matrix."
            )
        group_starts[g - 1] = idx[0]
        group_ends[g - 1] = idx[-1] + 1
    return group_starts, group_ends
