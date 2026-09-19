"""Penalty primitives and column standardization for penalized Cox PH.

Pure NumPy, no dependence on pandas, the likelihood engine, or the
estimator layer -- this sits at the same architectural level as
`ties.py` (a numerical kernel), and is deliberately kept separate from
`coordinate_descent.py` so the penalty *math* and column scaling can
be tested and reasoned about independently of the optimization loop
that uses them.

Two penalty families are supported:

1. **Elastic net** (for `PenalizedCoxPH`)::

       lambda * sum_j pf_j * [alpha * |beta_j| + 0.5*(1-alpha) * beta_j^2]

2. **Sparse group lasso** (for `GroupLassoCoxPH`)::

       lambda * [(1-alpha) * sum_g m_g * ||beta_g||_2
                 +  alpha  * sum_j pf_j * |beta_j|           ]

   When alpha=0 this is pure group lasso (Yuan & Lin 2006); when
   alpha=1 it reduces to element-wise lasso (groups ignored);
   intermediate values give the sparse group lasso of Simon et al.
   (2013). The ``alpha`` parameter on ``GroupLassoCoxPH`` controls
   the mixing between the group L2 norm and element-wise L1 norm --
   distinct from ``PenalizedCoxPH.alpha`` (which mixes L1 and L2²
   in the elastic-net sense).

Every convention here is matched against glmnet 4.1-8's actual R
source (not the paper or the docs alone -- see
``docs/R_COMPATIBILITY.md`` for file/line references):

* Standardization divides each column by its *weighted population*
  standard deviation (denominator = sum(weight), not sum(weight) - 1)
  and does NOT center. `glmnet`'s `weighted_mean_sd()`
  (R/glmnetFlex.R) computes the mean only as an intermediate for the
  variance formula; the Cox family never subtracts it from `x`
  (`xm <- rep(0.0, nvars)` unconditionally in `coxpath.R`). This is
  specific to the Cox model's lack of an intercept -- see this
  package's own `CoxPH`, which centers `X` for Newton-Raphson
  conditioning but proves (utils/numerical.py) that centering never
  changes `coef_`. The same argument shows centering would be
  harmless here too, but we match glmnet's exact convention (scale
  only) so the fitted path lines up with `glmnet()` at intermediate
  lambda values, not just in the limit.
* Penalty factors are rescaled so they sum to `p` (the number of
  columns), matching glmnet's documented "penalty factors are
  internally rescaled to sum to nvars".
* Group multipliers default to sqrt(group_size) per Yuan & Lin
  (2006), which equalizes the per-group penalty across groups of
  different sizes. Group multipliers are *not* rescaled to sum to
  anything -- they are used as-is (matching ``grplasso``'s convention).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def soft_threshold(z: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    """Elementwise soft-thresholding operator: sign(z) * max(|z| - thresh, 0).

    `thresh` broadcasts against `z`; both scalar and per-coordinate
    thresholds (the latter needed once `penalty_factor` varies by
    column) are supported.
    """
    return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)


def elastic_net_penalty_value(
    beta: np.ndarray, alpha: float, penalty_factor: np.ndarray
) -> float:
    """sum_j penalty_factor_j * [alpha*|beta_j| + 0.5*(1-alpha)*beta_j^2].

    This is glmnet's `pen_function` (R/glmnetFlex.R): no dependence on
    lambda -- the caller multiplies by lambda separately, since the
    same penalty *shape* is reused across the whole regularization
    path and inside the per-lambda coordinate-descent objective.
    """
    return float(np.sum(penalty_factor * (alpha * np.abs(beta) + 0.5 * (1.0 - alpha) * beta**2)))


def weighted_column_scale(
    X: np.ndarray, weight: np.ndarray, standardize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column scale factors for penalized fitting.

    Returns
    -------
    xs : ndarray, shape (p,)
        Divisor applied to each column of X before fitting (weighted
        population standard deviation if `standardize`, else 1.0 for
        every column). Coefficients are divided by `xs` at the end to
        report them in the original units of X (see coordinate_descent.py).
    degenerate : ndarray of bool, shape (p,)
        True for columns whose weighted variance is numerically zero
        (matches glmnet's `xv[xv < 10*eps] <- 0` safeguard). These
        columns cannot be standardized meaningfully; callers must
        exclude them from penalized fitting (forced coefficient 0)
        rather than divide by a near-zero scale.
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
    # Degenerate columns get xs=1 (never divide by ~0); the caller is
    # responsible for excluding them from the penalized fit entirely.
    xs = np.where(degenerate, 1.0, xs)
    return xs, degenerate


def rescale_penalty_factors(penalty_factor: Optional[np.ndarray], p: int) -> np.ndarray:
    """Validate and rescale user-supplied penalty factors to sum to `p`.

    `penalty_factor=None` gives the all-ones default (every variable
    penalized equally). A 0 marks an always-unpenalized variable and
    is preserved exactly (never rescaled away from 0) -- this is the
    mechanism for "clearly defined penalized vs. unpenalized
    variables" the estimator exposes. Matches glmnet's documented
    "penalty factors are internally rescaled to sum to nvars"; unlike
    glmnet we do not silently treat negative factors as 0 -- a
    negative penalty factor is almost always a user error (it would
    make the penalty *encourage* a nonzero coefficient), so it raises.
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
        raise ValueError("penalty_factor cannot be all-zero (nothing would be penalized; "
                          "use an unpenalized CoxPH fit instead)")
    return pf * (p / total)


# ---------------------------------------------------------------------------
# Group lasso / sparse group lasso primitives
# ---------------------------------------------------------------------------

def validate_groups(
    groups: np.ndarray, p: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Validate and canonicalize group labels.

    Group labels follow a simple convention:

    * **0** marks unpenalized features (never shrunk toward zero,
      analogous to ``penalty_factor=0`` in elastic-net fitting).
    * **Positive integers** (1, 2, ..., G) mark penalized groups.
      All features sharing the same label are updated as a block by
      the group lasso solver and are either jointly zero or jointly
      nonzero.

    Labels need not be contiguous (e.g. ``[0, 1, 1, 5, 5]`` is fine);
    they are remapped internally to a contiguous 1..G range so that
    downstream code can use them as array indices.

    Parameters
    ----------
    groups : array-like, shape (p,)
        Integer group labels.
    p : int
        Number of features (must match ``len(groups)``).

    Returns
    -------
    groups : ndarray of int, shape (p,)
        Canonicalized group labels (0 for unpenalized, 1..G for
        penalized groups, contiguous).
    group_sizes : ndarray of int, shape (G,)
        Number of features in each penalized group (ordered by
        canonical label 1..G).
    n_groups : int
        Number of distinct penalized groups (G).

    Raises
    ------
    ValueError
        If ``groups`` has wrong shape, contains negative labels,
        non-integer labels, or has no penalized groups (all-zero).
    """
    groups = np.asarray(groups)
    if groups.shape != (p,):
        raise ValueError(
            f"groups must have shape ({p},), got {groups.shape}"
        )
    # Integer check: allow float arrays that happen to be integer-valued
    # (e.g. np.array([0.0, 1.0, 1.0])), but reject genuinely non-integer.
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

    # Identify the set of penalized group labels (everything except 0).
    unique_labels = np.unique(groups[groups > 0])
    n_groups = len(unique_labels)
    if n_groups == 0:
        raise ValueError(
            "groups must contain at least one positive label (all-zero "
            "means nothing is penalized; use an unpenalized CoxPH fit instead)"
        )

    # Remap to contiguous 1..G.  Unpenalized (0) stays 0.
    label_map = np.zeros(unique_labels.max() + 1, dtype=np.intp)
    for new_label, old_label in enumerate(unique_labels, start=1):
        label_map[old_label] = new_label
    penalized_mask = groups > 0
    groups[penalized_mask] = label_map[groups[penalized_mask]]

    # Compute group sizes.
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

    If ``group_multiplier`` is None, returns ``sqrt(group_sizes)``
    (the standard default from Yuan & Lin 2006, which equalizes the
    per-group penalty contribution across groups of different sizes).
    Otherwise validates shape and non-negativity.

    Unlike element-wise penalty factors (which are rescaled to sum to
    ``p``), group multipliers are used as-is -- they are *not*
    rescaled to sum to any particular value.  This matches
    ``grplasso``'s convention: the user controls the relative penalty
    weight between groups via ``group_multiplier``, and the absolute
    strength is controlled by ``lambda``.

    Parameters
    ----------
    group_multiplier : ndarray or None, shape (n_groups,)
        User-supplied per-group penalty multipliers.
    group_sizes : ndarray of int, shape (n_groups,)
        Number of features per penalized group.
    n_groups : int
        Number of penalized groups.

    Returns
    -------
    ndarray of float, shape (n_groups,)
        Validated (or defaulted) group multipliers.

    Raises
    ------
    ValueError
        If ``group_multiplier`` has wrong shape, contains negative
        values, or is all-zero.
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

    When ``alpha=0`` this is pure group lasso; when ``alpha=1`` it
    reduces to element-wise lasso (groups ignored).  Intermediate
    alpha gives the sparse group lasso of Simon et al. (2013).

    This is the penalty *shape* (no lambda factor); the caller
    multiplies by lambda separately, matching the convention in
    ``elastic_net_penalty_value``.

    Parameters
    ----------
    beta : ndarray, shape (p,)
        Coefficient vector.
    alpha : float
        Sparse group lasso mixing parameter in [0, 1].
        0 = pure group lasso, 1 = pure lasso.
    groups : ndarray of int, shape (p,)
        Canonicalized group labels (from ``validate_groups``;
        0 = unpenalized, 1..G = penalized groups).
    group_weights : ndarray, shape (G,)
        Per-group penalty multipliers (from ``rescale_group_multipliers``).
    penalty_factor : ndarray, shape (p,)
        Per-element penalty factors (from ``rescale_penalty_factors``).
        Used for the L1 (element-wise) part of the penalty.

    Returns
    -------
    float
        Penalty value (>= 0).
    """
    val = 0.0

    # Group L2 part: (1 - alpha) * sum_g m_g * ||beta_g||_2
    if alpha < 1.0:
        n_groups = len(group_weights)
        for g in range(1, n_groups + 1):
            mask = groups == g
            beta_g = beta[mask]
            norm_g = np.sqrt(np.dot(beta_g, beta_g))
            val += group_weights[g - 1] * norm_g
        val *= (1.0 - alpha)

    # Element-wise L1 part: alpha * sum_j pf_j * |beta_j|
    if alpha > 0.0:
        val += alpha * float(np.sum(penalty_factor * np.abs(beta)))

    return val


def group_soft_threshold(
    z: np.ndarray, threshold: float
) -> np.ndarray:
    """Group-level soft-thresholding: ``(1 - threshold/||z||_2)_+ * z``.

    If ``||z||_2 <= threshold``, returns the zero vector (the entire
    group is killed).  Otherwise returns
    ``z * (1 - threshold / ||z||_2)``.

    This is the proximal operator of the (scaled) L2 norm
    ``threshold * ||.||_2``, used as the group-level shrinkage step
    in the block coordinate descent update for group lasso and sparse
    group lasso.

    Parameters
    ----------
    z : ndarray, shape (K,)
        Input vector (typically the group's partial residual after
        element-wise L1 shrinkage in the sparse group lasso, or
        the raw partial residual for pure group lasso).
    threshold : float
        Non-negative threshold (typically ``lam * (1 - alpha) * m_g``
        for the sparse group lasso, or ``lam * m_g`` for pure group
        lasso).

    Returns
    -------
    ndarray, shape (K,)
        Soft-thresholded vector.
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

    Given the partial residual vector ``z`` (shape ``(K,)`` for a group
    of size K), the diagonal of the quadratic model ``diag_A``, and the
    penalty parameters, applies the two-step nested soft-thresholding:

    1. **Element-wise L1 shrinkage** (the "sparse" part)::

           s_j = sign(z_j) * max(|z_j| - lam * alpha * pf_j, 0) / A_jj

    2. **Group L2 shrinkage** (the "group" part)::

           beta_g = group_soft_threshold(s, lam * (1 - alpha) * m_g)

    When ``alpha=0`` step 1 is a no-op (s = z / diag_A); when
    ``alpha=1`` step 2 is a no-op (no group shrinkage).

    This is the proximal update for one block in the block coordinate
    descent solver (``coordinate_descent.py``).  It is separated here
    so the penalty math can be tested independently of the iteration
    machinery.

    Parameters
    ----------
    z : ndarray, shape (K,)
        Partial residual for the group (``linear_term_g - off_diag_g
        + diag_A * beta_g_old``, in the coordinate descent notation).
    diag_A : ndarray, shape (K,)
        Diagonal entries of the quadratic model for this group.
        For the proximal-Newton method these come from the information
        matrix diagonal; for the MM method they are all ``v`` (the
        constant majorizer: 1.0 for Cox, 0.25 for logistic).
    lam : float
        Regularization strength (a single lambda from the path).
    alpha : float
        Sparse group lasso mixing (0 = pure group, 1 = pure lasso).
    group_weight : float
        This group's penalty multiplier ``m_g``.
    penalty_factor_g : ndarray, shape (K,)
        Per-element penalty factors for this group.

    Returns
    -------
    ndarray, shape (K,)
        Updated coefficient block for the group.
    """
    K = len(z)

    # Step 1: element-wise L1 shrinkage.
    s = np.empty(K)
    for j in range(K):
        thresh_j = lam * alpha * penalty_factor_g[j]
        abs_z = abs(z[j])
        if abs_z <= thresh_j:
            s[j] = 0.0
        else:
            sign_z = 1.0 if z[j] > 0.0 else -1.0
            s[j] = sign_z * (abs_z - thresh_j) / diag_A[j]

    # Step 2: group L2 shrinkage.
    group_thresh = lam * (1.0 - alpha) * group_weight
    if group_thresh <= 0.0:
        # alpha == 1.0: pure lasso, no group shrinkage.
        return s

    return group_soft_threshold(s, group_thresh)


# ---------------------------------------------------------------------------
# Within-group orthogonalization (optional, for MM method)
# ---------------------------------------------------------------------------

def within_group_orthogonalize(
    X: np.ndarray,
    groups: np.ndarray,
    weight: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
    """SVD-based within-group orthogonalization.

    For each penalized group g, computes the weighted SVD of ``X_g``
    and replaces it with an orthonormalized version such that::

        X_g_orth.T @ diag(w) @ X_g_orth / sum(w)  =  I

    within that group's columns.  This ensures the block coordinate
    descent update for the MM method has a simple closed form (no
    matrix inversion needed within each group), since the within-group
    Gram matrix is identity.

    Follows ``grplasso``'s ``Relevant_Functions.R::orthogonalize``::

        SVD <- svd(Z[, ind, drop=FALSE], nu=0)
        QL[[j]] <- sweep(SVD$v[, r, drop=FALSE], 2, sqrt(n)/SVD$d[r], "*")
        orthog.Z[, ind[r]] <- Z[, ind] %*% QL[[j]]

    Unpenalized columns (``groups == 0``) are left unchanged.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix (already column-standardized via
        ``weighted_column_scale``).
    groups : ndarray of int, shape (p,)
        Canonicalized group labels (from ``validate_groups``;
        0 = unpenalized, 1..G = penalized).
    weight : ndarray, shape (n,)
        Per-observation sample weights (positive, not necessarily
        summing to 1).

    Returns
    -------
    X_orth : ndarray, shape (n, p)
        Orthogonalized design matrix.  Unpenalized columns are
        identical to the input.  Each penalized group's columns
        satisfy ``X_g.T @ diag(w) @ X_g / sum(w) = I``.
    QL_blocks : list of (group_label, ndarray)
        Per-group back-transformation matrices.  Each entry is
        ``(g, Q)`` where ``g`` is the canonical group label (1..G)
        and ``Q`` is the matrix such that
        ``beta_original_g = Q @ beta_orthogonalized_g``.
        Use ``unorthogonalize_coefs`` to apply this mapping.

    Notes
    -----
    Within-group orthogonalization is optional and only beneficial for
    the MM method (``method="MM"`` in ``GroupLassoCoxPH``), where the
    majorizer uses a scalar curvature bound rather than the exact
    Hessian.  For the default proximal-Newton method
    (``method="proximal_newton"``), orthogonalization provides no
    algorithmic benefit (the exact information matrix already captures
    within-group correlations) and adds unnecessary computational
    cost.

    Rank-deficient groups (e.g., perfectly collinear columns) are
    handled by retaining only the ``r`` non-negligible singular
    values.  The remaining columns within that group are zeroed out
    in ``X_orth`` and the back-transformation matrix ``Q`` maps the
    reduced-rank orthogonal space back to the full group.
    """
    n, p = X.shape
    w_sum = weight.sum()
    sqrt_w = np.sqrt(weight / w_sum)  # (n,), for weighted SVD

    X_orth = X.copy()
    QL_blocks: List[Tuple[int, np.ndarray]] = []

    n_groups = groups.max()  # canonical labels are 1..G
    for g in range(1, n_groups + 1):
        idx = np.where(groups == g)[0]
        K = len(idx)
        if K == 0:
            continue

        # Weighted design block: sqrt(w/w_sum) * X_g, shape (n, K).
        X_g_w = sqrt_w[:, np.newaxis] * X[:, idx]  # (n, K)

        # SVD:  X_g_w = U @ diag(d) @ Vt
        # We only need V and d (not U).
        # Using full_matrices=False for efficiency.
        _U, d, Vt = np.linalg.svd(X_g_w, full_matrices=False)

        # Rank determination: keep singular values above a relative
        # tolerance (matches grplasso's implicit rank handling via SVD).
        tol = max(n, K) * np.finfo(np.float64).eps * d[0] if d[0] > 0 else 0.0
        r = int(np.sum(d > tol))
        if r == 0:
            # Entire group is numerically zero; zero out and skip.
            X_orth[:, idx] = 0.0
            QL_blocks.append((g, np.zeros((K, 0))))
            logger.warning(
                "Group %d has rank 0 after weighted SVD; all columns "
                "zeroed in the orthogonalized design matrix.", g
            )
            continue

        # V_r: (K, r), d_r: (r,)
        V_r = Vt[:r, :].T  # (K, r)
        d_r = d[:r]

        # Back-transformation matrix Q: (K, r)
        # grplasso: QL[[j]] <- sweep(SVD$v[, r], 2, sqrt(n)/SVD$d[r], "*")
        # In our weighted convention, SVD is on sqrt(w/w_sum)*X_g, whose
        # singular values d_w relate to the raw SVD's d_raw via
        #   d_w = d_raw * sqrt(1/w_sum)  (for uniform weights w=1).
        # grplasso uses Q = V * sqrt(n)/d_raw = V * 1/d_w, which gives
        #   X_orth.T @ diag(w) @ X_orth / w_sum = I.
        # So: Q = V_r / d_r (not V_r * sqrt(w_sum) / d_r).
        Q = V_r * (1.0 / d_r)[np.newaxis, :]  # (K, r)

        # Orthogonalized columns: X_g @ Q
        X_g_orth = X[:, idx] @ Q  # (n, r)

        # If rank < K, the group effectively has fewer active columns.
        # We place the orthogonalized columns in the first r slots and
        # zero the remainder.
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

    Applies the block-diagonal back-transformation::

        beta_original_g = Q_g @ beta_orth_g

    for each penalized group, using the ``Q`` matrices produced by
    ``within_group_orthogonalize``.

    Unpenalized coefficients (``groups == 0``) are returned unchanged.

    Parameters
    ----------
    beta : ndarray, shape (p,)
        Coefficient vector in the orthogonalized space.
    groups : ndarray of int, shape (p,)
        Canonicalized group labels.
    QL_blocks : list of (group_label, ndarray)
        Back-transformation matrices from
        ``within_group_orthogonalize``.

    Returns
    -------
    ndarray, shape (p,)
        Coefficients in the original (non-orthogonalized) feature space.
    """
    beta_orig = beta.copy()
    for g, Q in QL_blocks:
        idx = np.where(groups == g)[0]
        K = len(idx)
        r = Q.shape[1] if Q.ndim == 2 else 0
        if r == 0:
            beta_orig[idx] = 0.0
            continue
        # beta_orth_g occupies the first r slots of the group.
        beta_orth_g = beta[idx[:r]]
        # Clear the full group, then write back the transformed coefs.
        beta_orig[idx] = 0.0
        # Q: (K, r), beta_orth_g: (r,) -> beta_orig_g: (K,)
        beta_orig[idx] = Q @ beta_orth_g
    return beta_orig


# ---------------------------------------------------------------------------
# Group-level index helpers (for coordinate descent)
# ---------------------------------------------------------------------------

def compute_group_indices(
    groups: np.ndarray, n_groups: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute start/end column indices for each penalized group.

    Assumes features within each group are contiguous in the column
    ordering.  If they are not, the caller should permute the design
    matrix columns (and the corresponding group labels / penalty
    factors) before calling this function.

    Parameters
    ----------
    groups : ndarray of int, shape (p,)
        Canonicalized group labels (from ``validate_groups``).
    n_groups : int
        Number of penalized groups (G).

    Returns
    -------
    group_starts : ndarray of int, shape (G,)
        Starting column index (inclusive) for each penalized group.
    group_ends : ndarray of int, shape (G,)
        Ending column index (exclusive) for each penalized group.

    Raises
    ------
    ValueError
        If any penalized group's features are not contiguous in the
        column ordering.
    """
    group_starts = np.empty(n_groups, dtype=np.intp)
    group_ends = np.empty(n_groups, dtype=np.intp)

    for g in range(1, n_groups + 1):
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            raise ValueError(f"Penalized group {g} has no features")
        # Check contiguity: indices should form a consecutive range.
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
            raise ValueError(
                f"Features in penalized group {g} are not contiguous "
                f"(columns {idx.tolist()}). Permute the design matrix "
                f"columns so that each group occupies a contiguous block."
            )
        group_starts[g - 1] = idx[0]
        group_ends[g - 1] = idx[-1] + 1

    return group_starts, group_ends
