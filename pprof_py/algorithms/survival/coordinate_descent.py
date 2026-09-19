"""Penalized (ridge / LASSO / elastic net) Cox regression solver.

Architecture
------------
This module fulfills ``docs/ARCHITECTURE.md``'s prediction that
"penalization... slot[s] into optimization.py (a penalty term added to
the objective)... without touching the likelihood engine".  It lives
alongside ``optimization.py`` at the algorithm/numerical-kernel
layer, in its own file rather than appended to `optimization.py`
itself, because the *mechanism* differs (proximal Newton + coordinate
descent for a non-smooth objective, vs. plain Newton-Raphson for a
smooth one) even though both consume the exact same
`objective(beta) -> (log_likelihood, score, information)` callable
built from `cox_partial_likelihood`. Neither `optimization.py` nor
this module duplicates any Breslow/Efron/strata/offset/weight/
start-stop math -- both are pure consumers of
`algorithms/partial_likelihood.py`.

Algorithm
---------
For a fixed lambda and alpha, we minimize

    J(beta) = -c * log_lik(beta) + lambda * sum_j pf_j * [alpha*|beta_j| + 0.5*(1-alpha)*beta_j^2]

via a proximal-Newton (IRLS-like) outer loop: at the current iterate
`beta_cur`, `cox_partial_likelihood` gives the exact
`(log_lik, score, information)`, which defines a quadratic Newton approximation

    Q(beta) = const - c*score.(beta-beta_cur) + 0.5*c*(beta-beta_cur).information.(beta-beta_cur)

`Q(beta) + lambda*penalty(beta)` is then approximately minimized by cyclic
coordinate descent (a dense-matrix generalization of the standard
LASSO coordinate-descent update -- see `solve_penalized_quadratic`),
giving the next outer iterate. This is mathematically the same class
of algorithm as glmnet's own Cox solver (`R/coxpath.R`'s `cox.fit`,
which builds an IRLS working response/weight pair and calls
`elnet.fit`), but uses the *exact*, already-R-validated dense
information matrix as the quadratic model instead of glmnet's cheaper
diagonal approximation. Since the penalized negative log-partial-
likelihood is convex, both approaches minimize the same objective and
converge to the same answer for a given (standardized) X, lambda, and
alpha; they differ only in the internal iteration path, not the
target.  See ``docs/R_COMPATIBILITY.md`` for the empirical comparison
against ``glmnet`` itself.

The `c` scaling factor (glmnet's `1/sum(weight)`), standardization,
and penalty-factor rescaling conventions are all matched to glmnet
4.1-8's actual source, not the paper alone -- see `penalty.py` and
`docs/R_COMPATIBILITY.md`.

A note on `c`, since it is easy to get wrong (an earlier version of
this module used `2/sum(weight)` and was wrong by exactly a factor of
2): glmnet's own `cox_obj_function` literally computes
`coxnet.deviance(...) + lambda*pen_function(...)`, and
`coxnet.deviance = 2*(lsat - loglik)`, which reads as if the smooth
part should be `-(2/sum(weight))*loglik_raw(beta)`. But the *lambda
sequence glmnet actually fits at* is calibrated by
`get_cox_lambda_max`, which derives its gradient from `coxgrad(eta, y,
weights)` -- and `coxgrad`'s own `std.weights=TRUE` normalizes weights
to sum to *1* (`w <- w/sum(w)`, R/coxgrad.R), a different convention
from `coxnet.deviance`'s `std.weights=TRUE` (sum to *nobs*,
R/coxnet.deviance.R). The upshot, confirmed empirically against
`glmnet()`'s own auto-generated (unforced) lambda sequence and
coefficient path (not merely a lambda sequence we fed back into
`glmnet`, which only proves `glmnet` echoes its `lambda=` argument):
the value of lambda that actually zeroes every coefficient is
`max_j(|score_j| / sum(weight) / penalty_factor_j) / max(alpha,
1e-3)`, i.e. `c = 1/sum(weight)`, not `2/sum(weight)`. Coefficients
along the resulting path match `glmnet(family="cox")` to ~1e-8 with
this value; see ``docs/R_COMPATIBILITY.md`` for the full numerical
comparison.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

import logging

from .penalty import (
    soft_threshold,
    elastic_net_penalty_value,
    sparse_group_lasso_penalty_value,
    compute_group_indices,
)
from .risk_sets import njit, _HAS_NUMBA

logger = logging.getLogger(__name__)

ObjectiveFn = Callable[[np.ndarray], "tuple[float, np.ndarray, np.ndarray]"]


class PenalizedFitResult(NamedTuple):
    """Result of fitting a single (lambda, alpha) point.

    Mirrors `algorithms/optimization.py`'s `NewtonRaphsonResult` in
    spirit (same role: what the outer loop converged to), with fields
    specific to the penalized objective. `information` is the exact
    (unpenalized-likelihood) information matrix at `beta`, kept for
    potential downstream use (e.g. approximate post-selection
    inference on the active set) but not otherwise consumed by this
    package.
    """

    beta: np.ndarray
    log_likelihood: float
    objective_value: float
    n_outer_iter: int
    n_inner_iter_total: int
    converged: bool
    message: str
    information: np.ndarray


# ----------------------------------------------------------------------
# Inner loop: cyclic coordinate descent on a fixed dense quadratic model
# ----------------------------------------------------------------------
@njit(cache=True)
def _coordinate_descent_numba(A, linear_term, beta, lam, alpha, penalty_factor, tol, max_iter):
    p = beta.shape[0]
    diagA = np.empty(p)
    for k in range(p):
        d = A[k, k]
        diagA[k] = d if d > 1e-12 else 1e-12

    A_beta = A @ beta
    n_iter = 0
    max_change = 0.0
    for iteration in range(max_iter):
        max_change = 0.0
        for k in range(p):
            old_beta = beta[k]
            resid_k = linear_term[k] - A_beta[k] + A[k, k] * old_beta
            thresh_k = lam * alpha * penalty_factor[k]
            denom_k = diagA[k] + lam * (1.0 - alpha) * penalty_factor[k]
            if resid_k > thresh_k:
                beta_k_new = (resid_k - thresh_k) / denom_k
            elif resid_k < -thresh_k:
                beta_k_new = (resid_k + thresh_k) / denom_k
            else:
                beta_k_new = 0.0
            change = abs(beta_k_new - old_beta)
            if change > max_change:
                max_change = change
            if change != 0.0:
                beta[k] = beta_k_new
                delta = beta_k_new - old_beta
                for j in range(p):
                    A_beta[j] += A[j, k] * delta
        n_iter = iteration + 1
        if max_change < tol:
            break
    return beta, n_iter, max_change


def _coordinate_descent_python(A, linear_term, beta, lam, alpha, penalty_factor, tol, max_iter):
    """Pure Python fallback using the same incremental A @ beta update."""
    p = beta.shape[0]
    diagA = np.maximum(np.diag(A), 1e-12)
    A_beta = A @ beta
    n_iter = 0
    max_change = 0.0
    for iteration in range(max_iter):
        max_change = 0.0
        for k in range(p):
            old_beta = beta[k]
            resid_k = linear_term[k] - A_beta[k] + A[k, k] * old_beta
            thresh_k = lam * alpha * penalty_factor[k]
            denom_k = diagA[k] + lam * (1.0 - alpha) * penalty_factor[k]
            if resid_k > thresh_k:
                beta_k_new = (resid_k - thresh_k) / denom_k
            elif resid_k < -thresh_k:
                beta_k_new = (resid_k + thresh_k) / denom_k
            else:
                beta_k_new = 0.0
            change = abs(beta_k_new - old_beta)
            if change > max_change:
                max_change = change
            if change != 0.0:
                beta[k] = beta_k_new
                A_beta += A[:, k] * (beta_k_new - old_beta)
        n_iter = iteration + 1
        if max_change < tol:
            break
    return beta, n_iter, max_change



def solve_penalized_quadratic(
    A: np.ndarray,
    linear_term: np.ndarray,
    beta_init: np.ndarray,
    lam: float,
    alpha: float,
    penalty_factor: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 1000,
):
    """Approximately minimize the penalized quadratic subproblem by cyclic
    coordinate descent, dispatching to the Numba kernel when available.

    `A` is a dense p x p PSD matrix (the outer loop's `c * information`).
    The implementation maintains `A @ beta` incrementally during each
    coordinate sweep, avoiding a fresh full matrix-vector product for
    every coordinate while preserving the same quadratic objective.
    for the derivation. Returns (beta, n_iter, max_change_at_stop).
    """
    beta = np.array(beta_init, dtype=np.float64, copy=True)
    lam = float(lam)
    alpha = float(alpha)
    pf = np.asarray(penalty_factor, dtype=np.float64)
    if _HAS_NUMBA:
        beta, n_iter, max_change = _coordinate_descent_numba(
            np.asarray(A, dtype=np.float64), np.asarray(linear_term, dtype=np.float64),
            beta, lam, alpha, pf, float(tol), int(max_iter),
        )
    else:
        beta, n_iter, max_change = _coordinate_descent_python(
            np.asarray(A, dtype=np.float64), np.asarray(linear_term, dtype=np.float64),
            beta, lam, alpha, pf, float(tol), int(max_iter),
        )
    return beta, n_iter, max_change


# ----------------------------------------------------------------------
# Outer loop: proximal Newton on the exact partial-likelihood engine
# ----------------------------------------------------------------------
def fit_single_lambda(
    objective_fn: ObjectiveFn,
    beta_init: np.ndarray,
    c: float,
    lam: float,
    alpha: float,
    penalty_factor: np.ndarray,
    outer_max_iter: int = 100,
    outer_tol: float = 1e-9,
    inner_max_iter: int = 1000,
    inner_tol: float = 1e-10,
    max_halvings: int = 30,
) -> PenalizedFitResult:
    """Fit the elastic-net-penalized Cox model at one (lambda, alpha)
    point, warm-started from `beta_init`.

    `objective_fn(beta) -> (log_likelihood, score, information)` is
    expected to already be a closure over the (already column-scaled)
    design matrix, offset, weights, strata, and tie method -- exactly
    the same shape of closure `CoxPH.fit` builds around
    `cox_partial_likelihood` (see `models/coxph.py`). `c` is
    `1 / sum(sample_weight)` (see module docstring for why this, and
    not glmnet's `coxnet.deviance`'s literal factor of 2, is the value
    that matches glmnet's actual fitted path).
    """
    p = beta_init.shape[0]
    beta = np.array(beta_init, dtype=np.float64, copy=True)
    pf = np.asarray(penalty_factor, dtype=np.float64)

    def exact_objective(b, loglik):
        return -c * loglik + lam * elastic_net_penalty_value(b, alpha, pf)

    loglik, score, info = objective_fn(beta)
    obj_val = exact_objective(beta, loglik)

    n_outer_iter = 0
    n_inner_iter_total = 0
    converged = False
    message = "max_outer_iter reached without convergence"

    for outer_iter in range(1, outer_max_iter + 1):
        n_outer_iter = outer_iter
        A = c * info
        g = c * score
        linear_term = A @ beta + g

        beta_candidate, n_inner, inner_change = solve_penalized_quadratic(
            A, linear_term, beta, lam, alpha, pf, tol=inner_tol, max_iter=inner_max_iter,
        )
        n_inner_iter_total += n_inner

        loglik_candidate, score_candidate, info_candidate = objective_fn(beta_candidate)
        obj_candidate = exact_objective(beta_candidate, loglik_candidate)
        if not (np.isfinite(obj_candidate) and np.all(np.isfinite(beta_candidate))):
            return PenalizedFitResult(
                beta=beta, log_likelihood=loglik, objective_value=obj_val,
                n_outer_iter=n_outer_iter, n_inner_iter_total=n_inner_iter_total,
                converged=False, message="non-finite objective or coefficients encountered",
                information=info,
            )

        # Step-halving safeguard: the quadratic Newton approximation is only a
        # local approximation, so a full proximal-Newton step can
        # occasionally overshoot early in the path (same rationale as
        # optimization.py's Newton-Raphson step-halving). Interpolate
        # back toward the current point until the exact penalized
        # objective actually improves.
        objective_tol = 1e-12 * max(1.0, abs(obj_val))
        step = 1.0
        halvings = 0
        while obj_candidate > obj_val + objective_tol and halvings < max_halvings:
            step *= 0.5
            halvings += 1
            beta_trial = beta + step * (beta_candidate - beta)
            loglik_trial, score_trial, info_trial = objective_fn(beta_trial)
            obj_trial = exact_objective(beta_trial, loglik_trial)
            beta_candidate, loglik_candidate = beta_trial, loglik_trial
            score_candidate, info_candidate, obj_candidate = score_trial, info_trial, obj_trial

        if obj_candidate > obj_val + objective_tol:
            return PenalizedFitResult(
                beta=beta, log_likelihood=loglik, objective_value=obj_val,
                n_outer_iter=n_outer_iter, n_inner_iter_total=n_inner_iter_total,
                converged=False, message="step-halving failed to find an improving step",
                information=info,
            )

        if p > 0:
            beta_scale = np.maximum(1.0, np.abs(beta))
            relative_beta_change = np.max(np.abs(beta_candidate - beta) / beta_scale)
        else:
            relative_beta_change = 0.0
        rel_obj_change = abs(obj_candidate - obj_val) / max(1.0, abs(obj_val))

        beta, loglik, score, info, obj_val = (
            beta_candidate, loglik_candidate, score_candidate, info_candidate, obj_candidate,
        )

        # Check stationarity of the exact penalized objective. This is more
        # reliable than coefficient-change alone, especially near zero
        # coefficients on an L1 path.
        g_exact = c * score
        if p > 0:
            kkt = np.empty(p)
            for j in range(p):
                if beta[j] == 0.0:
                    kkt[j] = max(abs(g_exact[j]) - lam * alpha * pf[j], 0.0)
                else:
                    kkt[j] = abs(
                        g_exact[j]
                        - lam * pf[j] * (alpha * np.sign(beta[j]) + (1.0 - alpha) * beta[j])
                    )
            kkt_scale = max(1.0, float(np.max(np.abs(g_exact))))
            kkt_violation = float(np.max(kkt) / kkt_scale)
        else:
            kkt_violation = 0.0

        if kkt_violation < max(10.0 * outer_tol, 1e-12):
            converged = True
            message = "converged"
            break

    return PenalizedFitResult(
        beta=beta,
        log_likelihood=loglik,
        objective_value=obj_val,
        n_outer_iter=n_outer_iter,
        n_inner_iter_total=n_inner_iter_total,
        converged=converged,
        message=message,
        information=info,
    )


# ----------------------------------------------------------------------
# Lambda sequence
# ----------------------------------------------------------------------
def compute_lambda_max(
    score_at_null: np.ndarray, c: float, penalty_factor: np.ndarray, alpha: float,
) -> float:
    """Smallest lambda at which every penalized coefficient is exactly
    zero, given the score of the (unpenalized-scale) log-likelihood
    evaluated at the "null" point.

    `score_at_null` should be the score returned by `objective_fn` at
    beta=0 in the common case (no unpenalized variables); when some
    columns have `penalty_factor == 0`, the caller should instead
    supply the score evaluated at the point where those unpenalized
    columns sit at their own (unpenalized) MLE and every penalized
    column is 0 -- matching glmnet's `get_cox_lambda_max`, which
    pre-fits the unpenalized variables via `survival::coxph` for
    exactly this reason (see `models/penalized_coxph.py`).

    Derivation: beta=0 (restricted to the penalized columns) is
    optimal for a given lambda iff the KKT stationarity condition for
    the L1 part holds at every penalized coordinate:
    `|c * score_j| <= lambda * alpha * penalty_factor_j`. The smallest
    lambda for which this holds for *all* j is therefore
    `max_j(|c*score_j| / penalty_factor_j) / alpha`, with `alpha`
    floored at 1e-3 (glmnet's own floor) since alpha=0 (pure ridge)
    never sets coefficients exactly to 0 and lambda_max would
    otherwise be infinite.
    """
    pf = np.asarray(penalty_factor, dtype=np.float64)
    g = np.abs(c * np.asarray(score_at_null, dtype=np.float64))
    penalized = pf > 0
    if not np.any(penalized):
        return 0.0
    ratios = g[penalized] / pf[penalized]
    return float(np.max(ratios) / max(alpha, 1e-3))


def build_lambda_sequence(
    lambda_max: float,
    lambda_min_ratio: float,
    n_lambda: int,
    lambda_pad: float = 0.0,
) -> np.ndarray:
    """Log-spaced descending sequence from `lambda_max` to
    `lambda_max * lambda_min_ratio`.

    `lambda_pad` adds a small positive offset to the starting
    `lambda_max` before constructing the log-spaced grid.  This is used
    for R `grplasso::set.lambda.cox()` compatibility, where the Cox
    group-lasso path starts at `lambda_max + 1e-5` so the first path
    point is guaranteed to be all-zero even at the KKT boundary.

    A `lambda_max` of exactly 0 (e.g. every column unpenalized) returns
    an all-zero sequence -- there is nothing to regularize.
    """
    if isinstance(n_lambda, (bool, np.bool_)) or int(n_lambda) != n_lambda or int(n_lambda) < 1:
        raise ValueError("n_lambda must be a positive integer")
    if not np.isfinite(lambda_max) or lambda_max < 0:
        raise ValueError("lambda_max must be finite and non-negative")
    if not np.isfinite(lambda_min_ratio) or not (0.0 < lambda_min_ratio <= 1.0):
        raise ValueError("lambda_min_ratio must be finite and in (0, 1]")
    if not np.isfinite(lambda_pad) or lambda_pad < 0:
        raise ValueError("lambda_pad must be finite and non-negative")
    if lambda_max == 0:
        return np.zeros(int(n_lambda), dtype=np.float64)
    lambda_start = lambda_max + lambda_pad
    if n_lambda == 1:
        return np.array([lambda_start])
    log_max = np.log(lambda_start)
    log_min = np.log(lambda_start * lambda_min_ratio)
    return np.exp(np.linspace(log_max, log_min, n_lambda))


def fit_regularization_path(
    objective_fn: ObjectiveFn,
    p: int,
    c: float,
    alpha: float,
    lambda_sequence: Sequence[float],
    penalty_factor: np.ndarray,
    beta_warm_start: Optional[np.ndarray] = None,
    outer_max_iter: int = 100,
    outer_tol: float = 1e-9,
    inner_max_iter: int = 1000,
    inner_tol: float = 1e-10,
) -> "list[PenalizedFitResult]":
    """Fit the full path, warm-starting each lambda from the previous
    (larger) lambda's solution -- the standard glmnet path strategy,
    and the reason the sequence must be traversed largest-to-smallest
    (the caller is responsible for sorting `lambda_sequence`
    descending; this function does not re-sort, so a path fit with an
    explicit user-supplied sequence still warm-starts in whatever
    order it is given).
    """
    lam_arr = np.asarray(lambda_sequence, dtype=np.float64)
    if lam_arr.ndim != 1 or lam_arr.size == 0:
        raise ValueError("lambda_sequence must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(lam_arr)) or np.any(lam_arr < 0):
        raise ValueError("lambda_sequence must contain finite non-negative values")
    if lam_arr.size > 1 and np.any(np.diff(lam_arr) > 0):
        raise ValueError("lambda_sequence must be sorted in descending order")
    beta = np.zeros(p) if beta_warm_start is None else np.array(beta_warm_start, dtype=np.float64, copy=True)
    if beta.shape != (p,):
        raise ValueError(f"beta_warm_start must have shape ({p},)")
    results = []
    for lam in lam_arr:
        result = fit_single_lambda(
            objective_fn, beta, c, float(lam), alpha, penalty_factor,
            outer_max_iter=outer_max_iter, outer_tol=outer_tol,
            inner_max_iter=inner_max_iter, inner_tol=inner_tol,
        )
        results.append(result)
        beta = result.beta
    return results


# ======================================================================
# Sparse group lasso / group lasso solver
# ======================================================================
#
# The sparse group lasso penalty for Cox PH is:
#
#     J(beta) = -c * loglik(beta)
#               + lam * [(1-alpha) * sum_g m_g * ||beta_g||_2
#                        +  alpha  * sum_j pf_j * |beta_j|       ]
#
# The outer loop is proximal-Newton (same architecture as the elastic-
# net solver above): at the current iterate, the exact information
# matrix defines a quadratic model, which is approximately minimized
# by block coordinate descent.  Both solvers share the same
# ``objective_fn(beta) -> (loglik, score, information)`` callable,
# step-halving logic, and warm-starting conventions.
#
# The inner (block CD) update for each group g is a two-step nested
# soft-thresholding:
#
#   1. Compute partial residual z_j for each j in group g.
#   2. Element-wise L1 shrinkage: s_j = ST(z_j, lam*alpha*pf_j) / A_jj
#   3. Group L2 shrinkage:  beta_g = group_ST(s_g, lam*(1-alpha)*m_g)
#
# This reduces to standard lasso (groups ignored) when alpha=1 and to
# pure group lasso (no within-group sparsity) when alpha=0.
# ======================================================================

class GroupPenalizedFitResult(NamedTuple):
    """Result of fitting a single (lambda, alpha) point for
    group / sparse group lasso.

    Extends ``PenalizedFitResult`` with group-level attributes.
    ``active_groups`` and ``group_norms`` are indexed by penalized-
    group label (1..G), stored in arrays of length G.
    """

    beta: np.ndarray                # shape (p,)
    log_likelihood: float
    objective_value: float
    n_outer_iter: int
    n_inner_iter_total: int
    converged: bool
    message: str
    information: np.ndarray         # shape (p, p)
    active_groups: np.ndarray       # boolean mask, shape (n_groups,)
    group_norms: np.ndarray         # ||beta_g||_2 per group, shape (n_groups,)
    df: float                       # effective degrees of freedom


# ----------------------------------------------------------------------
# Inner loop: block coordinate descent for sparse group lasso
# ----------------------------------------------------------------------
#
# Incremental A_beta update strategy (sequential within-group):
#   We maintain A_beta = A @ beta across the sweep.  For each group g
#   we do *sequential* element-wise L1 shrinkage (updating A_beta after
#   each element, exactly like the elastic-net solver), then apply the
#   group L2 shrinkage as a correction step at the end.  This ensures
#   each element within the group sees the already-updated values of
#   preceding elements, which is essential for convergence when A has
#   non-trivial off-diagonal entries (proximal-Newton with exact
#   information matrix).  Cost is O(K*p) per group, same as K
#   element-wise CD updates.
# ----------------------------------------------------------------------

@njit(cache=True)
def _sparse_group_coordinate_descent_numba(
    A, linear_term, beta, lam, alpha,
    penalty_factor, group_starts, group_ends, group_weights,
    active_groups, tol, max_iter,
):
    """Numba-accelerated block coordinate descent for the sparse group
    lasso penalized quadratic subproblem.

    Minimizes::

        0.5 * beta^T A beta - linear_term^T beta
        + lam * [(1-alpha) * sum_g m_g * ||beta_g||_2
                 + alpha * sum_j pf_j * |beta_j|]

    over the active groups only.  ``beta`` is modified in place and
    returned.

    Within each group, element-wise L1 shrinkage is applied
    *sequentially* (each element sees the already-updated values of
    preceding elements via the incremental ``A_beta`` update), then
    group L2 shrinkage is applied to the whole block.

    Special case: for singleton groups (K=1) the sparse-group penalty
    collapses to a single weighted L1 term,

        lam * [alpha * pf_j + (1-alpha) * m_g] * |beta_j|,

    so the exact coordinate update is available in closed form.  This
    matters for validations where every variable is its own group: the
    generic two-stage "divide by A_jj, then apply group shrinkage"
    update is not mathematically equivalent when A_jj != 1 and solves
    the wrong quadratic subproblem.
    """
    p = beta.shape[0]
    n_groups = group_starts.shape[0]

    # Safeguard diagonal.
    diagA = np.empty(p)
    for k in range(p):
        d = A[k, k]
        diagA[k] = d if d > 1e-12 else 1e-12

    A_beta = A @ beta

    n_iter = 0
    max_change = 0.0

    for iteration in range(max_iter):
        max_change = 0.0

        for g_idx in range(n_groups):
            if not active_groups[g_idx]:
                continue

            gs = group_starts[g_idx]
            ge = group_ends[g_idx]
            K = ge - gs
            mw = group_weights[g_idx]

            # Save pre-update group betas for max-change tracking.
            pre_beta_g = np.empty(K)
            for jj in range(K):
                pre_beta_g[jj] = beta[gs + jj]

            # Singleton group: the sparse-group penalty reduces to a
            # single weighted L1 term, so use the exact one-coordinate
            # soft-threshold update.
            if K == 1:
                j = gs
                old_beta_j = beta[j]
                z_j = linear_term[j] - A_beta[j] + diagA[j] * old_beta_j
                thresh_j = lam * (alpha * penalty_factor[j] + (1.0 - alpha) * mw)
                abs_z = abs(z_j)
                if abs_z <= thresh_j:
                    beta_j_new = 0.0
                else:
                    sign_z = 1.0 if z_j > 0.0 else -1.0
                    beta_j_new = sign_z * (abs_z - thresh_j) / diagA[j]
                delta_j = beta_j_new - old_beta_j
                if delta_j != 0.0:
                    beta[j] = beta_j_new
                    for i in range(p):
                        A_beta[i] += A[i, j] * delta_j
                    if abs(delta_j) > max_change:
                        max_change = abs(delta_j)
                continue

            # KKT zero pre-check for multi-member groups: if the
            # group is currently all-zero and the quadratic-model
            # KKT at zero is satisfied, skip the update entirely.
            # The two-stage CD update (element-wise solve then group
            # shrink) can produce O(eps) non-zeros at the KKT
            # boundary because element-wise Step 1 divides by A_jj,
            # introducing Hessian scaling that the raw-gradient-based
            # lambda_max does not account for.
            all_zero_g = True
            for jj in range(K):
                if pre_beta_g[jj] != 0.0:
                    all_zero_g = False
                    break
            if all_zero_g:
                kkt_sq = 0.0
                for jj in range(K):
                    j = gs + jj
                    r_j = linear_term[j] - A_beta[j]
                    val = abs(r_j) - lam * alpha * penalty_factor[j]
                    if val > 0.0:
                        kkt_sq += val * val
                if kkt_sq ** 0.5 <= lam * (1.0 - alpha) * mw:
                    continue

            # Step 1: sequential element-wise L1 shrinkage.
            # Each element sees the updated A_beta from preceding
            # elements (Gauss-Seidel within the group).
            for jj in range(K):
                j = gs + jj
                old_beta_j = beta[j]
                z_j = linear_term[j] - A_beta[j] + diagA[j] * old_beta_j
                thresh_j = lam * alpha * penalty_factor[j]
                abs_z = abs(z_j)
                if abs_z <= thresh_j:
                    beta_j_new = 0.0
                else:
                    sign_z = 1.0 if z_j > 0.0 else -1.0
                    beta_j_new = sign_z * (abs_z - thresh_j) / diagA[j]
                delta_j = beta_j_new - old_beta_j
                if delta_j != 0.0:
                    beta[j] = beta_j_new
                    for i in range(p):
                        A_beta[i] += A[i, j] * delta_j

            # Step 2: group L2 shrinkage.
            group_thresh = lam * (1.0 - alpha) * mw
            if group_thresh > 0.0:
                s_norm_sq = 0.0
                for jj in range(K):
                    s_norm_sq += beta[gs + jj] * beta[gs + jj]
                s_norm = s_norm_sq ** 0.5

                if s_norm <= group_thresh:
                    # Zero the entire group.
                    for jj in range(K):
                        j = gs + jj
                        old_j = beta[j]
                        if old_j != 0.0:
                            beta[j] = 0.0
                            for i in range(p):
                                A_beta[i] -= A[i, j] * old_j
                elif s_norm > 0.0:
                    shrink = group_thresh / s_norm
                    for jj in range(K):
                        j = gs + jj
                        old_j = beta[j]
                        beta[j] = old_j * (1.0 - shrink)
                        delta_j = beta[j] - old_j
                        if delta_j != 0.0:
                            for i in range(p):
                                A_beta[i] += A[i, j] * delta_j

            # Track max change for convergence (vs pre-update values).
            for jj in range(K):
                change_j = abs(beta[gs + jj] - pre_beta_g[jj])
                if change_j > max_change:
                    max_change = change_j

        n_iter = iteration + 1
        if max_change < tol:
            break

    return beta, n_iter, max_change


def _sparse_group_coordinate_descent_python(
    A, linear_term, beta, lam, alpha,
    penalty_factor, group_starts, group_ends, group_weights,
    active_groups, tol, max_iter,
):
    """Pure-Python fallback for the sparse group lasso block CD.

    Identical algorithm to the Numba kernel above; kept separate so
    the package works (more slowly) without Numba installed.
    """
    p = beta.shape[0]
    n_groups = group_starts.shape[0]
    diagA = np.maximum(np.diag(A), 1e-12)
    A_beta = A @ beta

    n_iter = 0
    max_change = 0.0

    for iteration in range(max_iter):
        max_change = 0.0

        for g_idx in range(n_groups):
            if not active_groups[g_idx]:
                continue

            gs = group_starts[g_idx]
            ge = group_ends[g_idx]
            K = ge - gs
            mw = group_weights[g_idx]

            pre_beta_g = beta[gs:ge].copy()

            # Singleton group: exact one-coordinate update.
            if K == 1:
                j = gs
                old_beta_j = beta[j]
                z_j = linear_term[j] - A_beta[j] + diagA[j] * old_beta_j
                thresh_j = lam * (alpha * penalty_factor[j] + (1.0 - alpha) * mw)
                abs_z = abs(z_j)
                if abs_z <= thresh_j:
                    beta_j_new = 0.0
                else:
                    sign_z = 1.0 if z_j > 0.0 else -1.0
                    beta_j_new = sign_z * (abs_z - thresh_j) / diagA[j]
                delta_j = beta_j_new - old_beta_j
                if delta_j != 0.0:
                    beta[j] = beta_j_new
                    A_beta += A[:, j] * delta_j
                    if abs(delta_j) > max_change:
                        max_change = abs(delta_j)
                continue

            # KKT zero pre-check (see Numba kernel for rationale).
            if np.all(pre_beta_g == 0.0):
                r_g = linear_term[gs:ge] - A_beta[gs:ge]
                st_vals = np.maximum(
                    np.abs(r_g) - lam * alpha * penalty_factor[gs:ge], 0.0,
                )
                if np.sqrt(np.dot(st_vals, st_vals)) <= lam * (1.0 - alpha) * mw:
                    continue

            # Step 1: sequential element-wise L1 shrinkage.
            for jj in range(K):
                j = gs + jj
                old_beta_j = beta[j]
                z_j = linear_term[j] - A_beta[j] + diagA[j] * old_beta_j
                thresh_j = lam * alpha * penalty_factor[j]
                abs_z = abs(z_j)
                if abs_z <= thresh_j:
                    beta_j_new = 0.0
                else:
                    sign_z = 1.0 if z_j > 0.0 else -1.0
                    beta_j_new = sign_z * (abs_z - thresh_j) / diagA[j]
                delta_j = beta_j_new - old_beta_j
                if delta_j != 0.0:
                    beta[j] = beta_j_new
                    A_beta += A[:, j] * delta_j

            # Step 2: group L2 shrinkage.
            group_thresh = lam * (1.0 - alpha) * mw
            if group_thresh > 0.0:
                s_g = beta[gs:ge]
                s_norm = np.sqrt(np.dot(s_g, s_g))

                if s_norm <= group_thresh:
                    delta_g = -s_g.copy()
                    beta[gs:ge] = 0.0
                    A_beta += A[:, gs:ge] @ delta_g
                elif s_norm > 0.0:
                    shrink = group_thresh / s_norm
                    old_g = s_g.copy()
                    beta[gs:ge] = old_g * (1.0 - shrink)
                    delta_g = beta[gs:ge] - old_g
                    A_beta += A[:, gs:ge] @ delta_g

            # Track max change.
            change_g = float(np.max(np.abs(beta[gs:ge] - pre_beta_g)))
            if change_g > max_change:
                max_change = change_g

        n_iter = iteration + 1
        if max_change < tol:
            break

    return beta, n_iter, max_change


def solve_sparse_group_penalized_quadratic(
    A: np.ndarray,
    linear_term: np.ndarray,
    beta_init: np.ndarray,
    lam: float,
    alpha: float,
    penalty_factor: np.ndarray,
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    group_weights: np.ndarray,
    active_groups: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> "tuple[np.ndarray, int, float]":
    """Solve the sparse-group-lasso-penalized quadratic subproblem.

    Dispatches to the Numba kernel when available, with pure-Python
    fallback.  Analogous to ``solve_penalized_quadratic`` for the
    elastic-net case.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        PSD quadratic model (``c * information``).
    linear_term : ndarray, shape (p,)
        Linear term (``A @ beta_cur + c * score``).
    beta_init : ndarray, shape (p,)
        Warm-start coefficients (copied before modification).
    lam : float
        Regularization strength.
    alpha : float
        Sparse group lasso mixing (0 = group, 1 = lasso).
    penalty_factor : ndarray, shape (p,)
        Per-element penalty factors for the L1 part.
    group_starts, group_ends : ndarray of int, shape (G,)
        Start (inclusive) / end (exclusive) column indices per group.
    group_weights : ndarray, shape (G,)
        Per-group penalty multipliers for the L2 part.
    active_groups : ndarray of bool, shape (G,) or None
        Which groups to update.  ``None`` means all groups active.
    tol : float
        Convergence tolerance (max absolute change in beta).
    max_iter : int
        Maximum full sweeps.

    Returns
    -------
    beta : ndarray, shape (p,)
    n_iter : int
    max_change : float
    """
    beta = np.array(beta_init, dtype=np.float64, copy=True)
    lam = float(lam)
    alpha = float(alpha)
    pf = np.asarray(penalty_factor, dtype=np.float64)
    gs = np.asarray(group_starts, dtype=np.intp)
    ge = np.asarray(group_ends, dtype=np.intp)
    gw = np.asarray(group_weights, dtype=np.float64)
    n_groups = len(gs)

    if active_groups is None:
        ag = np.ones(n_groups, dtype=np.bool_)
    else:
        ag = np.asarray(active_groups, dtype=np.bool_)

    if _HAS_NUMBA:
        beta, n_iter, max_change = _sparse_group_coordinate_descent_numba(
            np.asarray(A, dtype=np.float64),
            np.asarray(linear_term, dtype=np.float64),
            beta, lam, alpha, pf, gs, ge, gw, ag,
            float(tol), int(max_iter),
        )
    else:
        beta, n_iter, max_change = _sparse_group_coordinate_descent_python(
            np.asarray(A, dtype=np.float64),
            np.asarray(linear_term, dtype=np.float64),
            beta, lam, alpha, pf, gs, ge, gw, ag,
            float(tol), int(max_iter),
        )
    return beta, n_iter, max_change


# ----------------------------------------------------------------------
# Lambda-max for sparse group lasso
# ----------------------------------------------------------------------

def compute_group_lambda_max(
    score_at_null: np.ndarray,
    c: float,
    groups: np.ndarray,
    group_weights: np.ndarray,
    penalty_factor: np.ndarray,
    alpha: float,
) -> float:
    """Smallest lambda at which every penalized group is exactly zero.

    Derivation: at beta=0 the KKT stationarity condition for penalized
    group g under the sparse group lasso is::

        ||ST(c * score_g,  lam * alpha * pf_g)||_2
            <=  lam * (1 - alpha) * m_g

    where ``ST`` is element-wise soft-thresholding and ``m_g`` is the
    group weight.  We need the smallest ``lam`` satisfying this for all
    penalized groups simultaneously.

    **Special cases** (closed-form):

    * ``alpha = 0`` (pure group lasso): no L1 shrinkage, so the
      condition is ``||c * score_g||_2 <= lam * m_g``, giving
      ``lambda_max = max_g(||c * score_g||_2 / m_g)``.
    * ``alpha = 1`` (standard lasso, groups ignored): falls back to the
      element-wise KKT ``lambda_max = max_j(|c*score_j| / pf_j)``.

    **General case** (``0 < alpha < 1``): the KKT condition is implicit
    in lambda (the L1 threshold in ``ST`` itself depends on lambda).
    For each group, the critical lambda is the root of a monotone
    function, computed via bisection.  ``lambda_max`` is then the
    maximum over all groups.

    Parameters
    ----------
    score_at_null : ndarray, shape (p,)
        Score (gradient of log-partial-likelihood) at the null point
        (penalized betas = 0, unpenalized betas at their MLE).
    c : float
        Objective scaling (``1 / sum(weight)``).
    groups : ndarray of int, shape (p,)
        Canonicalized group labels (0 = unpenalized, 1..G = penalized).
    group_weights : ndarray, shape (G,)
        Per-group penalty multipliers.
    penalty_factor : ndarray, shape (p,)
        Per-element penalty factors.
    alpha : float
        Sparse group lasso mixing in [0, 1].

    Returns
    -------
    float
        ``lambda_max`` (non-negative; 0 if nothing is penalized).
    """
    pf = np.asarray(penalty_factor, dtype=np.float64)
    g_abs = np.abs(c * np.asarray(score_at_null, dtype=np.float64))
    n_groups = len(group_weights)

    if n_groups == 0:
        return 0.0

    # Special case: alpha = 1 (standard lasso, groups ignored).
    if alpha >= 1.0:
        penalized = pf > 0
        if not np.any(penalized):
            return 0.0
        return float(np.max(g_abs[penalized] / pf[penalized]))

    # Special case: alpha = 0 (pure group lasso).
    if alpha <= 0.0:
        lam_max = 0.0
        for g in range(1, n_groups + 1):
            mask = groups == g
            score_g = g_abs[mask]
            norm_g = np.sqrt(np.dot(score_g, score_g))
            mw = group_weights[g - 1]
            if mw > 0:
                candidate = norm_g / mw
                if candidate > lam_max:
                    lam_max = candidate
        return lam_max

    # General case: 0 < alpha < 1.  For each group g, we need the
    # smallest lam_g such that
    #     ||ST(g_abs_g, lam_g * alpha * pf_g)||_2 <= lam_g * (1-alpha) * m_g.
    #
    # Define f_g(lam) = ||ST(g_abs_g, lam*alpha*pf_g)||_2 - lam*(1-alpha)*m_g.
    # f_g is continuous, piecewise linear, and decreasing in lam (the
    # L1 threshold grows, the group tolerance grows, and the
    # soft-thresholded norm shrinks).  f_g(0) = ||g_abs_g||_2 >= 0.
    # f_g(lam) -> -inf as lam -> inf.  So there is a unique root.
    # We use bisection (reliable, 50 iterations = 2^{-50} relative
    # precision, which is beyond float64 resolution).
    lam_max = 0.0

    for g in range(1, n_groups + 1):
        mask = groups == g
        g_abs_g = g_abs[mask]
        pf_g = pf[mask]
        mw = group_weights[g - 1]
        K = len(g_abs_g)

        if mw <= 0 and alpha < 1.0:
            # Group is effectively unpenalized at the group level;
            # only element-wise L1 applies.  lambda_max for this group
            # is max_j(|g_abs_j| / pf_j) (the element-wise KKT).
            pen_j = pf_g > 0
            if np.any(pen_j):
                candidate = float(np.max(g_abs_g[pen_j] / pf_g[pen_j]))
                if candidate > lam_max:
                    lam_max = candidate
            continue

        # Upper bound for bisection.  We need a lambda at which the
        # KKT is satisfied: ||ST(g_abs, lam*alpha*pf)||_2 <= lam*(1-a)*m.
        # Two candidate upper bounds:
        #  (a) Pure-group-lasso lambda_max: norm_g / m_g  (LHS has no
        #      L1 shrinkage; valid only when (1-alpha) >= 1, i.e. alpha=0).
        #  (b) Element-wise L1 lambda that zeros every element:
        #      max_j(g_abs_j / (alpha * pf_j)).  At this lambda, LHS = 0.
        # We take the max of both for robustness.
        norm_g = np.sqrt(np.dot(g_abs_g, g_abs_g))
        if norm_g == 0.0:
            continue
        bound_group = norm_g / mw if mw > 0 else norm_g / 1e-10
        # Element-wise bound (only for penalized elements with pf > 0).
        bound_elem = 0.0
        for jj in range(K):
            if pf_g[jj] > 0 and alpha > 0:
                candidate_j = g_abs_g[jj] / (alpha * pf_g[jj])
                if candidate_j > bound_elem:
                    bound_elem = candidate_j
        lam_hi = max(bound_group, bound_elem) * 1.01  # small safety margin
        lam_lo = 0.0

        for _bisect_iter in range(60):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            # Compute ||ST(g_abs_g, lam_mid * alpha * pf_g)||_2
            st_norm_sq = 0.0
            for jj in range(K):
                thresh_j = lam_mid * alpha * pf_g[jj]
                val = g_abs_g[jj] - thresh_j
                if val > 0.0:
                    st_norm_sq += val * val
            st_norm = st_norm_sq ** 0.5
            rhs = lam_mid * (1.0 - alpha) * mw
            if st_norm > rhs:
                # KKT violated: need larger lambda.
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
            if lam_hi - lam_lo < 1e-14 * max(lam_hi, 1.0):
                break

        candidate = lam_hi
        if candidate > lam_max:
            lam_max = candidate

    return lam_max


# ----------------------------------------------------------------------
# Outer loop: proximal Newton for sparse group lasso at one lambda
# ----------------------------------------------------------------------

def _compute_group_norms(
    beta: np.ndarray, groups: np.ndarray, n_groups: int,
) -> np.ndarray:
    """||beta_g||_2 for each penalized group."""
    norms = np.empty(n_groups)
    for g in range(1, n_groups + 1):
        bg = beta[groups == g]
        norms[g - 1] = np.sqrt(np.dot(bg, bg))
    return norms


def _compute_group_kkt_violation(
    score_scaled: np.ndarray,
    beta: np.ndarray,
    lam: float,
    alpha: float,
    groups: np.ndarray,
    group_weights: np.ndarray,
    penalty_factor: np.ndarray,
    n_groups: int,
) -> float:
    """Maximum scaled KKT violation across all penalized groups.

    For a nonzero group g the stationarity condition is:
        c*score_j = lam * [alpha*pf_j*sign(beta_j)
                           + (1-alpha)*m_g * beta_j / ||beta_g||_2]
    for each j in g.

    For a zero group g:
        ||ST(c*score_g, lam*alpha*pf_g)||_2 <= lam*(1-alpha)*m_g

    Returns the maximum absolute violation, scaled by the gradient
    magnitude for numerical stability.
    """
    kkt_max = 0.0
    scale = max(1.0, float(np.max(np.abs(score_scaled))))

    for g in range(1, n_groups + 1):
        mask = groups == g
        beta_g = beta[mask]
        score_g = score_scaled[mask]
        pf_g = penalty_factor[mask]
        mw = group_weights[g - 1]
        norm_g = np.sqrt(np.dot(beta_g, beta_g))

        if norm_g == 0.0:
            # Zero group: check if any would become nonzero.
            # ||ST(score_g, lam*alpha*pf_g)||_2 <= lam*(1-alpha)*m_g
            st_norm_sq = 0.0
            for jj in range(len(score_g)):
                val = abs(score_g[jj]) - lam * alpha * pf_g[jj]
                if val > 0.0:
                    st_norm_sq += val * val
            st_norm = st_norm_sq ** 0.5
            violation = max(st_norm - lam * (1.0 - alpha) * mw, 0.0)
        else:
            # Nonzero group: check stationarity per element.
            # For beta_j != 0: exact stationarity via subgradient.
            # For beta_j == 0 within an active group: the subdifferential
            #   of the L1 part is [-1,1], so the KKT is
            #   |score_j - lam*(1-a)*m_g*0/||beta_g|||
            #       <= lam*alpha*pf_j,
            #   i.e. max(|score_j| - lam*alpha*pf_j, 0).
            violation = 0.0
            for jj in range(len(beta_g)):
                if abs(beta_g[jj]) > 0.0:
                    subgrad = (
                        lam * alpha * pf_g[jj] * np.sign(beta_g[jj])
                        + lam * (1.0 - alpha) * mw * beta_g[jj] / norm_g
                    )
                    elem_viol = abs(score_g[jj] - subgrad)
                else:
                    # Zero element in active group: L1 subdifferential.
                    elem_viol = max(
                        abs(score_g[jj]) - lam * alpha * pf_g[jj], 0.0,
                    )
                if elem_viol > violation:
                    violation = elem_viol

        kkt_viol_scaled = violation / scale
        if kkt_viol_scaled > kkt_max:
            kkt_max = kkt_viol_scaled

    return kkt_max


def _screen_inactive_groups_quadratic(
    A: np.ndarray,
    linear_term: np.ndarray,
    beta: np.ndarray,
    lam: float,
    alpha: float,
    groups: np.ndarray,
    group_weights: np.ndarray,
    penalty_factor: np.ndarray,
    n_groups: int,
    active_groups: np.ndarray,
) -> np.ndarray:
    """Screen inactive groups using the quadratic-model residual.

    After the inner CD converges on the active set, the residual for
    an inactive (zero) group g is ``q_g = linear_term_g - (A @ beta)_g``.
    The group should enter the active set if the quadratic-model KKT
    is violated::

        ||ST(q_g,  lam*alpha*pf_g)||_2  >  lam*(1-alpha)*m_g

    This is more reliable than screening with the exact gradient
    because the quadratic model is what the CD is actually solving.
    Returns a boolean mask of shape ``(n_groups,)``.
    """
    q = linear_term - A @ beta
    newly_active = np.zeros(n_groups, dtype=np.bool_)

    for g in range(1, n_groups + 1):
        g_idx = g - 1
        if active_groups[g_idx]:
            continue

        mask = groups == g
        q_g = q[mask]
        pf_g = penalty_factor[mask]
        mw = group_weights[g_idx]

        st_norm_sq = 0.0
        for jj in range(len(q_g)):
            val = abs(q_g[jj]) - lam * alpha * pf_g[jj]
            if val > 0.0:
                st_norm_sq += val * val
        st_norm = st_norm_sq ** 0.5

        if st_norm > lam * (1.0 - alpha) * mw:
            newly_active[g_idx] = True

    return newly_active


def fit_single_lambda_group(
    objective_fn: ObjectiveFn,
    beta_init: np.ndarray,
    c: float,
    lam: float,
    alpha: float,
    penalty_factor: np.ndarray,
    groups: np.ndarray,
    group_weights: np.ndarray,
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    n_groups: int,
    outer_max_iter: int = 100,
    outer_tol: float = 1e-9,
    inner_max_iter: int = 1000,
    inner_tol: float = 1e-10,
    max_halvings: int = 30,
    use_active_set: bool = True,
) -> GroupPenalizedFitResult:
    """Fit the sparse-group-lasso-penalized Cox model at one
    (lambda, alpha) point, warm-started from ``beta_init``.

    Same proximal-Newton architecture as ``fit_single_lambda``
    (quadratic model from exact information matrix, inner solve,
    step-halving safeguard, KKT convergence check), but uses the
    group-aware block CD inner solver and group-level KKT check.

    Parameters
    ----------
    objective_fn : callable
        ``objective_fn(beta) -> (log_likelihood, score, information)``.
    beta_init : ndarray, shape (p,)
    c : float
        ``1 / sum(weight)``.
    lam : float
        Regularization strength.
    alpha : float
        Sparse group lasso mixing.
    penalty_factor : ndarray, shape (p,)
    groups : ndarray of int, shape (p,)
        Canonicalized group labels (0 = unpenalized, 1..G = penalized).
    group_weights : ndarray, shape (G,)
    group_starts, group_ends : ndarray of int, shape (G,)
    n_groups : int
    outer_max_iter, outer_tol, inner_max_iter, inner_tol, max_halvings
        Convergence and safety parameters.

    Returns
    -------
    GroupPenalizedFitResult
    """
    p = beta_init.shape[0]
    beta = np.array(beta_init, dtype=np.float64, copy=True)
    pf = np.asarray(penalty_factor, dtype=np.float64)
    gw = np.asarray(group_weights, dtype=np.float64)
    gs_arr = np.asarray(group_starts, dtype=np.intp)
    ge_arr = np.asarray(group_ends, dtype=np.intp)

    def exact_objective(b, loglik):
        return -c * loglik + lam * sparse_group_lasso_penalty_value(
            b, alpha, groups, gw, pf,
        )

    loglik, score, info = objective_fn(beta)
    obj_val = exact_objective(beta, loglik)

    n_outer_iter = 0
    n_inner_iter_total = 0
    converged = False
    message = "max_outer_iter reached without convergence"

    # Active-set initialization (§4.4.1).
    if use_active_set:
        gnorms_init = _compute_group_norms(beta, groups, n_groups)
        active_groups = gnorms_init > 0
        # Ensure at least one group is active.
        if not np.any(active_groups):
            active_groups[:] = True
    else:
        active_groups = np.ones(n_groups, dtype=np.bool_)

    for outer_iter in range(1, outer_max_iter + 1):
        n_outer_iter = outer_iter
        A = c * info
        g = c * score
        linear_term = A @ beta + g

        # ----- Inner active-set CD loop (§4.4.1) -----
        # Solve the quadratic subproblem on the active groups, then
        # screen inactive groups using the quadratic-model residual.
        # Repeat until no new groups are needed.  The inner screening
        # uses the *quadratic model* (A, linear_term) -- not the exact
        # gradient -- so it is consistent with what the CD is solving
        # and catches marginal groups that gradient-based screening
        # would miss due to Hessian coupling.
        inner_active = active_groups.copy() if use_active_set else active_groups
        for _as_round in range(n_groups + 1):
            beta_candidate, n_inner, inner_change = \
                solve_sparse_group_penalized_quadratic(
                    A, linear_term, beta, lam, alpha, pf,
                    gs_arr, ge_arr, gw, inner_active,
                    tol=inner_tol, max_iter=inner_max_iter,
                )
            n_inner_iter_total += n_inner

            if not use_active_set or np.all(inner_active):
                break

            newly_active = _screen_inactive_groups_quadratic(
                A, linear_term, beta_candidate, lam, alpha,
                groups, gw, pf, n_groups, inner_active,
            )
            if not np.any(newly_active):
                break
            inner_active |= newly_active
            logger.debug(
                "Inner active-set expansion: %d new group(s), "
                "%d/%d active (outer iter %d)",
                int(np.sum(newly_active)),
                int(np.sum(inner_active)),
                n_groups, outer_iter,
            )
        # Propagate the expanded active set to the outer loop.
        if use_active_set:
            active_groups = inner_active

        loglik_cand, score_cand, info_cand = objective_fn(beta_candidate)
        obj_cand = exact_objective(beta_candidate, loglik_cand)
        if not (np.isfinite(obj_cand) and np.all(np.isfinite(beta_candidate))):
            gnorms = _compute_group_norms(beta, groups, n_groups)
            df = float(np.sum(beta != 0))
            return GroupPenalizedFitResult(
                beta=beta, log_likelihood=loglik,
                objective_value=obj_val,
                n_outer_iter=n_outer_iter,
                n_inner_iter_total=n_inner_iter_total,
                converged=False,
                message="non-finite objective or coefficients encountered",
                information=info,
                active_groups=gnorms > 0,
                group_norms=gnorms, df=df,
            )

        # Step-halving (same logic as elastic-net outer loop).
        objective_tol = 1e-12 * max(1.0, abs(obj_val))
        step = 1.0
        halvings = 0
        while obj_cand > obj_val + objective_tol and halvings < max_halvings:
            step *= 0.5
            halvings += 1
            beta_trial = beta + step * (beta_candidate - beta)
            loglik_trial, score_trial, info_trial = objective_fn(beta_trial)
            obj_trial = exact_objective(beta_trial, loglik_trial)
            beta_candidate, loglik_cand = beta_trial, loglik_trial
            score_cand, info_cand, obj_cand = (
                score_trial, info_trial, obj_trial,
            )

        if obj_cand > obj_val + objective_tol:
            gnorms = _compute_group_norms(beta, groups, n_groups)
            df = float(np.sum(beta != 0))
            return GroupPenalizedFitResult(
                beta=beta, log_likelihood=loglik,
                objective_value=obj_val,
                n_outer_iter=n_outer_iter,
                n_inner_iter_total=n_inner_iter_total,
                converged=False,
                message="step-halving failed to find an improving step",
                information=info,
                active_groups=gnorms > 0,
                group_norms=gnorms, df=df,
            )

        # Track convergence metrics before updating.
        if p > 0:
            beta_scale = np.maximum(1.0, np.abs(beta))
            relative_beta_change = float(
                np.max(np.abs(beta_candidate - beta) / beta_scale)
            )
        else:
            relative_beta_change = 0.0
        rel_obj_change = abs(obj_cand - obj_val) / max(1.0, abs(obj_val))

        beta, loglik, score, info, obj_val = (
            beta_candidate, loglik_cand, score_cand, info_cand, obj_cand,
        )

        # Update the active set from the accepted beta (§4.4.1).
        # After the first full-pass iteration discovers the correct
        # active set, subsequent iterations use the reduced set.
        if use_active_set:
            gnorms_cur = _compute_group_norms(beta, groups, n_groups)
            active_groups = gnorms_cur > 0
            if not np.any(active_groups):
                active_groups = np.ones(n_groups, dtype=np.bool_)

        g_exact = c * score

        # ----- Convergence: two-tier check -----
        # Tier 1 (cheap): relative beta and objective change.
        # Tier 2 (rigorous): group-aware KKT stationarity.
        # Declare convergence if *either* tier is satisfied. The KKT
        # check is the gold standard but can be overly strict near
        # non-smooth boundaries of the group L2 penalty (the
        # proximal-Newton quadratic model doesn't perfectly capture
        # the penalty geometry, leading to residual KKT violations of
        # O(1e-2) even when the objective and beta have converged to
        # machine precision). The beta/obj tier catches this case.
        kkt_violation = _compute_group_kkt_violation(
            g_exact, beta, lam, alpha, groups, gw, pf, n_groups,
        )

        if kkt_violation < max(10.0 * outer_tol, 1e-12):
            # Safe verification: before declaring convergence when
            # active-set screening was used, do one full-set pass
            # (all groups active).  The gradient at the reduced
            # solution can mask marginal groups whose contribution
            # is only visible when the full Hessian coupling is
            # present. One full iteration catches them.
            if use_active_set and not np.all(active_groups):
                active_groups[:] = True
                logger.debug(
                    "Active-set full verification pass (iter %d)",
                    outer_iter,
                )
                continue
            converged = True
            message = "converged (KKT)"
            break

        if (relative_beta_change < outer_tol
                and rel_obj_change < outer_tol):
            if use_active_set and not np.all(active_groups):
                active_groups[:] = True
                logger.debug(
                    "Active-set full verification pass (beta/obj, "
                    "iter %d)", outer_iter,
                )
                continue
            converged = True
            message = "converged (beta/obj)"
            break

    gnorms = _compute_group_norms(beta, groups, n_groups)
    df = float(np.sum(beta != 0))
    return GroupPenalizedFitResult(
        beta=beta,
        log_likelihood=loglik,
        objective_value=obj_val,
        n_outer_iter=n_outer_iter,
        n_inner_iter_total=n_inner_iter_total,
        converged=converged,
        message=message,
        information=info,
        active_groups=gnorms > 0,
        group_norms=gnorms,
        df=df,
    )


# ----------------------------------------------------------------------
# Full regularization path for sparse group lasso
# ----------------------------------------------------------------------

def fit_group_regularization_path(
    objective_fn: ObjectiveFn,
    p: int,
    c: float,
    alpha: float,
    lambda_sequence: Sequence[float],
    groups: np.ndarray,
    group_weights: np.ndarray,
    penalty_factor: np.ndarray,
    n_groups: int,
    beta_warm_start: Optional[np.ndarray] = None,
    outer_max_iter: int = 100,
    outer_tol: float = 1e-9,
    inner_max_iter: int = 1000,
    inner_tol: float = 1e-10,
    use_active_set: bool = True,
) -> "list[GroupPenalizedFitResult]":
    """Fit the full group-lasso regularization path, warm-starting each
    lambda from the previous (larger) lambda's solution.

    Analogous to ``fit_regularization_path`` for the elastic-net case.
    The lambda sequence must be sorted in descending order (the caller
    is responsible for this; the function does not re-sort).

    Parameters
    ----------
    objective_fn : callable
        ``objective_fn(beta) -> (log_likelihood, score, information)``.
    p : int
        Number of features.
    c : float
        ``1 / sum(weight)``.
    alpha : float
        Sparse group lasso mixing.
    lambda_sequence : sequence of float
        Descending sequence of lambda values.
    groups : ndarray of int, shape (p,)
    group_weights : ndarray, shape (G,)
    penalty_factor : ndarray, shape (p,)
    n_groups : int
    beta_warm_start : ndarray or None
    outer_max_iter, outer_tol, inner_max_iter, inner_tol
        Convergence parameters.

    Returns
    -------
    list of GroupPenalizedFitResult
        One result per lambda value.
    """
    lam_arr = np.asarray(lambda_sequence, dtype=np.float64)
    if lam_arr.ndim != 1 or lam_arr.size == 0:
        raise ValueError(
            "lambda_sequence must be a non-empty one-dimensional sequence"
        )
    if not np.all(np.isfinite(lam_arr)) or np.any(lam_arr < 0):
        raise ValueError(
            "lambda_sequence must contain finite non-negative values"
        )
    if lam_arr.size > 1 and np.any(np.diff(lam_arr) > 0):
        raise ValueError(
            "lambda_sequence must be sorted in descending order"
        )

    groups = np.asarray(groups, dtype=np.intp)
    gw = np.asarray(group_weights, dtype=np.float64)
    pf = np.asarray(penalty_factor, dtype=np.float64)

    # Precompute group start/end index arrays.
    gs_arr, ge_arr = compute_group_indices(groups, n_groups)

    beta = (
        np.zeros(p, dtype=np.float64)
        if beta_warm_start is None
        else np.array(beta_warm_start, dtype=np.float64, copy=True)
    )
    if beta.shape != (p,):
        raise ValueError(f"beta_warm_start must have shape ({p},)")

    results: list[GroupPenalizedFitResult] = []
    for lam in lam_arr:
        result = fit_single_lambda_group(
            objective_fn, beta, c, float(lam), alpha, pf,
            groups, gw, gs_arr, ge_arr, n_groups,
            outer_max_iter=outer_max_iter, outer_tol=outer_tol,
            inner_max_iter=inner_max_iter, inner_tol=inner_tol,
            use_active_set=use_active_set,
        )
        results.append(result)
        beta = result.beta
    return results
