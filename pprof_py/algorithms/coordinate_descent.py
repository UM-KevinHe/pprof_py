"""Likelihood-agnostic penalized regression solver: proximal Newton + CD.

Algorithm
---------
For a fixed lambda and alpha, we minimize

    J(beta) = -c * log_lik(beta) + lambda * penalty(beta)

via a proximal-Newton (IRLS-like) outer loop.  At the current iterate
``beta_cur``, the likelihood engine provides ``(log_lik, score, info)``
which define a quadratic Newton approximation.  The penalized quadratic
is then approximately minimized by cyclic coordinate descent (element-
wise for elastic net, block for group lasso), giving the next iterate.

This architecture is the same across all likelihood families (logistic,
linear, Cox, discrete survival).  Each family provides its own
``objective_fn(beta) -> (loglik, score, info)`` closure; this module's
job is the penalty-aware optimization machinery.

The ``c`` scaling, standardization, and penalty-factor conventions all
match glmnet 4.1-8's actual source.
"""
from __future__ import annotations

import logging
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

from .penalty import (
    soft_threshold,
    elastic_net_penalty_value,
    sparse_group_lasso_penalty_value,
    compute_group_indices,
    njit, _HAS_NUMBA,
)

logger = logging.getLogger(__name__)

ObjectiveFn = Callable[[np.ndarray], "tuple[float, np.ndarray, np.ndarray]"]


# ======================================================================
# Result containers
# ======================================================================

class PenalizedFitResult(NamedTuple):
    """Result of fitting a single (lambda, alpha) point (elastic net)."""
    beta: np.ndarray
    log_likelihood: float
    objective_value: float
    n_outer_iter: int
    n_inner_iter_total: int
    converged: bool
    message: str
    information: np.ndarray
    kkt_violation: float = float("nan")
    """Relative KKT stationarity residual at the returned ``beta``.

    ``converged`` is True when this falls below the solver's threshold.
    When it does not, this value says *how far* from stationary the
    returned point is, which ``converged=False`` alone cannot.
    """


class GroupPenalizedFitResult(NamedTuple):
    """Result of fitting a single (lambda, alpha) point (group lasso)."""
    beta: np.ndarray
    log_likelihood: float
    objective_value: float
    n_outer_iter: int
    n_inner_iter_total: int
    converged: bool
    message: str
    information: np.ndarray
    active_groups: np.ndarray
    group_norms: np.ndarray
    df: float
    kkt_violation: float = float("nan")
    """Relative KKT stationarity residual at the returned ``beta``.

    ``converged`` is True when this falls below the solver's threshold.
    When it does not, this value says *how far* from stationary the
    returned point is, which ``converged=False`` alone cannot.
    """



# ======================================================================
# Inner loop: cyclic coordinate descent (elastic net)
# ======================================================================

@njit(cache=True)
def _coordinate_descent_numba(A, linear_term, beta, lam, alpha, penalty_factor, tol, max_iter):
    """Numba-accelerated elastic net CD on a dense quadratic model."""
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
    """Pure Python fallback for elastic net CD."""
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


# Active-set variants: identical logic but skip inactive variables.

@njit(cache=True)
def _coordinate_descent_active_numba(
    A, linear_term, beta, lam, alpha, penalty_factor,
    active_vars, tol, max_iter,
):
    """Numba CD with active-set masking (element-wise elastic net)."""
    p = beta.shape[0]
    diagA = np.empty(p, dtype=np.float64)
    for j in range(p):
        diagA[j] = max(A[j, j], 1e-12)
    A_beta = A @ beta
    n_iter = 0
    max_change = 0.0
    for iteration in range(max_iter):
        max_change = 0.0
        for k in range(p):
            if not active_vars[k]:
                continue
            old_beta = beta[k]
            resid_k = linear_term[k] - A_beta[k] + diagA[k] * old_beta
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


def _coordinate_descent_active_python(
    A, linear_term, beta, lam, alpha, penalty_factor,
    active_vars, tol, max_iter,
):
    """Pure Python CD with active-set masking."""
    p = beta.shape[0]
    diagA = np.maximum(np.diag(A), 1e-12)
    A_beta = A @ beta
    n_iter = 0
    max_change = 0.0
    for iteration in range(max_iter):
        max_change = 0.0
        for k in range(p):
            if not active_vars[k]:
                continue
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
    active_vars: Optional[np.ndarray] = None,
) -> "tuple[np.ndarray, int, float]":
    """Minimize the elastic-net-penalized quadratic by cyclic CD.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        PSD quadratic model (e.g. ``c * info`` or ``X.T @ diag(w) @ X``).
    linear_term : ndarray, shape (p,)
        Linear term (``A @ beta_cur + c * score``).
    beta_init : ndarray, shape (p,)
        Warm-start coefficients.
    lam, alpha : float
        Regularization strength and elastic net mixing.
    penalty_factor : ndarray, shape (p,)
        Per-variable penalty weights.
    tol : float
        Convergence tolerance (max absolute change in beta).
    max_iter : int
        Maximum sweeps.
    active_vars : ndarray of bool, shape (p,) or None
        If provided, only update variables where active_vars[j] is True.
        Inactive variables are frozen at their current value.  When None,
        all variables are updated (original behaviour).

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
    A_arr = np.asarray(A, dtype=np.float64)
    lt_arr = np.asarray(linear_term, dtype=np.float64)
    if active_vars is not None:
        av = np.asarray(active_vars, dtype=np.bool_)
        if _HAS_NUMBA:
            beta, n_iter, max_change = _coordinate_descent_active_numba(
                A_arr, lt_arr, beta, lam, alpha, pf, av,
                float(tol), int(max_iter),
            )
        else:
            beta, n_iter, max_change = _coordinate_descent_active_python(
                A_arr, lt_arr, beta, lam, alpha, pf, av,
                float(tol), int(max_iter),
            )
    else:
        if _HAS_NUMBA:
            beta, n_iter, max_change = _coordinate_descent_numba(
                A_arr, lt_arr, beta, lam, alpha, pf,
                float(tol), int(max_iter),
            )
        else:
            beta, n_iter, max_change = _coordinate_descent_python(
                A_arr, lt_arr, beta, lam, alpha, pf,
                float(tol), int(max_iter),
            )
    return beta, n_iter, max_change


# ======================================================================
# Outer loop: proximal Newton at one lambda (elastic net)
# ======================================================================

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
    use_active_set: bool = False,
) -> PenalizedFitResult:
    """Fit the elastic-net-penalized model at one (lambda, alpha) point.

    ``objective_fn(beta) -> (log_likelihood, score, information)`` is a
    closure over the likelihood engine (logistic, linear, or Cox).
    ``c`` is the objective scaling (typically ``1 / sum(weight)``).

    Parameters
    ----------
    objective_fn : callable
        Returns ``(loglik, score, info)`` at ``beta``.
    beta_init : ndarray, shape (p,)
    c : float
    lam, alpha : float
    penalty_factor : ndarray, shape (p,)
    outer_max_iter, outer_tol, inner_max_iter, inner_tol, max_halvings
        Convergence parameters.
    use_active_set : bool, default=False
        If True, use active-set screening to skip variables that are
        zero and satisfy the KKT subgradient condition.  Variables are
        re-activated when their KKT condition is violated.  Results are
        identical to the full solve (within tolerance) but faster for
        sparse solutions.

    Returns
    -------
    PenalizedFitResult
    """
    p = beta_init.shape[0]
    beta = np.array(beta_init, dtype=np.float64, copy=True)
    pf = np.asarray(penalty_factor, dtype=np.float64)

    # Active-set initialisation.
    if use_active_set and p > 0:
        active = (beta != 0.0) | (pf == 0.0)
        if not np.any(active):
            active[:] = True  # first lambda: all active
    else:
        active = None

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
            A, linear_term, beta, lam, alpha, pf,
            tol=inner_tol, max_iter=inner_max_iter,
            active_vars=active,
        )
        n_inner_iter_total += n_inner

        loglik_cand, score_cand, info_cand = objective_fn(beta_candidate)
        obj_cand = exact_objective(beta_candidate, loglik_cand)
        if not (np.isfinite(obj_cand) and np.all(np.isfinite(beta_candidate))):
            return PenalizedFitResult(
                beta=beta, log_likelihood=loglik, objective_value=obj_val,
                n_outer_iter=n_outer_iter,
                n_inner_iter_total=n_inner_iter_total,
                converged=False,
                message="non-finite objective or coefficients encountered",
                information=info,
            )

        # Step-halving safeguard.
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
            score_cand, info_cand, obj_cand = score_trial, info_trial, obj_trial

        if obj_cand > obj_val + objective_tol:
            return PenalizedFitResult(
                beta=beta, log_likelihood=loglik, objective_value=obj_val,
                n_outer_iter=n_outer_iter,
                n_inner_iter_total=n_inner_iter_total,
                converged=False,
                message="step-halving failed to find an improving step",
                information=info,
            )

        if p > 0:
            beta_scale = np.maximum(1.0, np.abs(beta))
            relative_beta_change = np.max(np.abs(beta_candidate - beta) / beta_scale)
        else:
            relative_beta_change = 0.0
        rel_obj_change = abs(obj_cand - obj_val) / max(1.0, abs(obj_val))

        beta, loglik, score, info, obj_val = (
            beta_candidate, loglik_cand, score_cand, info_cand, obj_cand,
        )

        # KKT stationarity check.
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

        kkt_threshold = max(10.0 * outer_tol, 1e-12)
        # Tolerance for the supplementary stall test below.  Looser than
        # ``kkt_threshold`` (the real stationarity test) so it can still
        # absorb intercept jitter, but floored at an absolute 1e-6 and never
        # looser than the KKT test itself -- so it cannot become a loophole
        # that accepts a materially non-stationary point at loose outer_tol.
        stall_kkt_tol = max(kkt_threshold, 1e-6)

        if active is not None:
            # Active-set: check if any inactive variable violates KKT.
            inactive_violations = (~active) & (kkt > kkt_threshold * kkt_scale)
            if np.any(inactive_violations):
                # Activate violated variables and continue iterating.
                active |= inactive_violations
                continue

        if kkt_violation < kkt_threshold:
            converged = True
            message = "converged"
            break

        # Supplementary convergence: if beta and the objective are both
        # stable, declare converged even when the KKT residual does not
        # fully settle.  This can happen when fit_intercept=True, because
        # the intercept Newton update is folded into objective_fn as a side
        # effect and keeps perturbing the score the KKT check evaluates.
        #
        # REV-001: on its own this test cannot tell "converged" apart from
        # "the step has collapsed to zero at a non-stationary point", and it
        # was firing in the latter state.  Require the KKT residual to be
        # small in absolute terms as well -- loose enough to still catch the
        # intercept-jitter case, tight enough that a genuinely non-stationary
        # point is reported as non-converged instead of silently accepted.
        if (
            relative_beta_change < outer_tol
            and rel_obj_change < outer_tol
            and kkt_violation < stall_kkt_tol
        ):
            converged = True
            message = "converged (relative change)"
            break

        if active is not None:
            # After convergence check, update active set for next iter.
            active = (beta != 0.0) | (pf == 0.0)
            if not np.any(active):
                active[:] = True

    return PenalizedFitResult(
        beta=beta, log_likelihood=loglik, objective_value=obj_val,
        n_outer_iter=n_outer_iter, n_inner_iter_total=n_inner_iter_total,
        converged=converged, message=message, information=info,
        kkt_violation=float(kkt_violation),
    )


# ======================================================================
# Lambda sequence utilities
# ======================================================================

def compute_lambda_max(
    score_at_null: np.ndarray, c: float, penalty_factor: np.ndarray, alpha: float,
) -> float:
    """Smallest lambda at which every penalized coefficient is zero.

    Parameters
    ----------
    score_at_null : ndarray, shape (p,)
        Score at the null point.
    c : float
        Objective scaling.
    penalty_factor : ndarray, shape (p,)
    alpha : float

    Returns
    -------
    float
    """
    pf = np.asarray(penalty_factor, dtype=np.float64)
    g = np.abs(c * np.asarray(score_at_null, dtype=np.float64))
    penalized = pf > 0
    if not np.any(penalized):
        return 0.0
    ratios = g[penalized] / pf[penalized]
    return float(np.max(ratios) / max(alpha, 1e-3))


def build_lambda_sequence(
    lambda_max: float, lambda_min_ratio: float, n_lambda: int
) -> np.ndarray:
    """Log-spaced descending lambda sequence.

    Matches glmnet's ``exp(seq(log(lambda_max), log(lambda_max*lambda.min.ratio),
    length.out=nlambda))``.

    Parameters
    ----------
    lambda_max : float
    lambda_min_ratio : float
    n_lambda : int

    Returns
    -------
    ndarray, shape (n_lambda,)
    """
    if isinstance(n_lambda, (bool, np.bool_)) or int(n_lambda) != n_lambda or int(n_lambda) < 1:
        raise ValueError("n_lambda must be a positive integer")
    if not np.isfinite(lambda_max) or lambda_max < 0:
        raise ValueError("lambda_max must be finite and non-negative")
    if not np.isfinite(lambda_min_ratio) or not (0.0 < lambda_min_ratio <= 1.0):
        raise ValueError("lambda_min_ratio must be finite and in (0, 1]")
    if lambda_max == 0:
        return np.zeros(int(n_lambda), dtype=np.float64)
    if n_lambda == 1:
        return np.array([lambda_max])
    log_max = np.log(lambda_max)
    log_min = np.log(lambda_max * lambda_min_ratio)
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
    use_active_set: bool = False,
) -> "list[PenalizedFitResult]":
    """Fit the full elastic net path, warm-starting each lambda.

    Parameters
    ----------
    objective_fn : callable
    p : int
    c : float
    alpha : float
    lambda_sequence : sequence of float
        Must be sorted descending.
    penalty_factor : ndarray, shape (p,)
    beta_warm_start : ndarray or None
    outer_max_iter, outer_tol, inner_max_iter, inner_tol
    use_active_set : bool, default=False
        If True, use active-set screening in the inner CD solver.

    Returns
    -------
    list of PenalizedFitResult
    """
    lam_arr = np.asarray(lambda_sequence, dtype=np.float64)
    if lam_arr.ndim != 1 or lam_arr.size == 0:
        raise ValueError("lambda_sequence must be a non-empty 1D sequence")
    if not np.all(np.isfinite(lam_arr)) or np.any(lam_arr < 0):
        raise ValueError("lambda_sequence must contain finite non-negative values")
    if lam_arr.size > 1 and np.any(np.diff(lam_arr) > 0):
        raise ValueError("lambda_sequence must be sorted descending")
    beta = (
        np.zeros(p) if beta_warm_start is None
        else np.array(beta_warm_start, dtype=np.float64, copy=True)
    )
    if beta.shape != (p,):
        raise ValueError(f"beta_warm_start must have shape ({p},)")
    results = []
    for lam in lam_arr:
        result = fit_single_lambda(
            objective_fn, beta, c, float(lam), alpha, penalty_factor,
            outer_max_iter=outer_max_iter, outer_tol=outer_tol,
            inner_max_iter=inner_max_iter, inner_tol=inner_tol,
            use_active_set=use_active_set,
        )
        results.append(result)
        beta = result.beta
    return results


# ======================================================================
# Block CD for sparse group lasso
# ======================================================================

@njit(cache=True)
def _sparse_group_coordinate_descent_numba(
    A, linear_term, beta, lam, alpha,
    penalty_factor, group_starts, group_ends, group_weights,
    active_groups, tol, max_iter,
):
    """Numba-accelerated block CD for sparse group lasso."""
    p = beta.shape[0]
    n_groups = group_starts.shape[0]
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
            pre_beta_g = np.empty(K)
            for jj in range(K):
                pre_beta_g[jj] = beta[gs + jj]
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
    """Pure-Python fallback for sparse group lasso block CD."""
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
    """Solve the sparse-group-lasso-penalized quadratic by block CD.

    Parameters
    ----------
    A : ndarray, shape (p, p)
    linear_term : ndarray, shape (p,)
    beta_init : ndarray, shape (p,)
    lam, alpha : float
    penalty_factor : ndarray, shape (p,)
    group_starts, group_ends : ndarray of int, shape (G,)
    group_weights : ndarray, shape (G,)
    active_groups : ndarray of bool, shape (G,) or None
    tol : float
    max_iter : int

    Returns
    -------
    beta, n_iter, max_change
    """
    beta = np.array(beta_init, dtype=np.float64, copy=True)
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
            beta, float(lam), float(alpha), pf, gs, ge, gw, ag,
            float(tol), int(max_iter),
        )
    else:
        beta, n_iter, max_change = _sparse_group_coordinate_descent_python(
            np.asarray(A, dtype=np.float64),
            np.asarray(linear_term, dtype=np.float64),
            beta, float(lam), float(alpha), pf, gs, ge, gw, ag,
            float(tol), int(max_iter),
        )
    return beta, n_iter, max_change


# ======================================================================
# Lambda-max for group lasso
# ======================================================================

def compute_group_lambda_max(
    score_at_null: np.ndarray,
    c: float,
    groups: np.ndarray,
    group_weights: np.ndarray,
    penalty_factor: np.ndarray,
    alpha: float,
) -> float:
    """Smallest lambda at which every penalized group is exactly zero.

    Parameters
    ----------
    score_at_null : ndarray, shape (p,)
    c : float
    groups : ndarray of int, shape (p,)
    group_weights : ndarray, shape (G,)
    penalty_factor : ndarray, shape (p,)
    alpha : float

    Returns
    -------
    float
    """
    pf = np.asarray(penalty_factor, dtype=np.float64)
    g_abs = np.abs(c * np.asarray(score_at_null, dtype=np.float64))
    n_groups = len(group_weights)
    if n_groups == 0:
        return 0.0
    if alpha >= 1.0:
        penalized = pf > 0
        if not np.any(penalized):
            return 0.0
        return float(np.max(g_abs[penalized] / pf[penalized]))
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
    # General case: 0 < alpha < 1.  Bisection per group.
    lam_max = 0.0
    for g in range(1, n_groups + 1):
        mask = groups == g
        g_abs_g = g_abs[mask]
        pf_g = pf[mask]
        mw = group_weights[g - 1]
        K = len(g_abs_g)
        if mw <= 0 and alpha < 1.0:
            pen_j = pf_g > 0
            if np.any(pen_j):
                candidate = float(np.max(g_abs_g[pen_j] / pf_g[pen_j]))
                if candidate > lam_max:
                    lam_max = candidate
            continue
        norm_g = np.sqrt(np.dot(g_abs_g, g_abs_g))
        if norm_g == 0.0:
            continue
        bound_group = norm_g / mw if mw > 0 else norm_g / 1e-10
        bound_elem = 0.0
        for jj in range(K):
            if pf_g[jj] > 0 and alpha > 0:
                candidate_j = g_abs_g[jj] / (alpha * pf_g[jj])
                if candidate_j > bound_elem:
                    bound_elem = candidate_j
        lam_hi = max(bound_group, bound_elem) * 1.01
        lam_lo = 0.0
        for _bisect_iter in range(60):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            st_norm_sq = 0.0
            for jj in range(K):
                thresh_j = lam_mid * alpha * pf_g[jj]
                val = g_abs_g[jj] - thresh_j
                if val > 0.0:
                    st_norm_sq += val * val
            st_norm = st_norm_sq ** 0.5
            rhs = lam_mid * (1.0 - alpha) * mw
            if st_norm > rhs:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
            if lam_hi - lam_lo < 1e-14 * max(lam_hi, 1.0):
                break
        candidate = lam_hi
        if candidate > lam_max:
            lam_max = candidate
    return lam_max


# ======================================================================
# Group-level helpers
# ======================================================================

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
    """Maximum scaled KKT violation across all penalized groups."""
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
            st_norm_sq = 0.0
            for jj in range(len(score_g)):
                val = abs(score_g[jj]) - lam * alpha * pf_g[jj]
                if val > 0.0:
                    st_norm_sq += val * val
            st_norm = st_norm_sq ** 0.5
            violation = max(st_norm - lam * (1.0 - alpha) * mw, 0.0)
        else:
            violation = 0.0
            for jj in range(len(beta_g)):
                if abs(beta_g[jj]) > 0.0:
                    subgrad = (
                        lam * alpha * pf_g[jj] * np.sign(beta_g[jj])
                        + lam * (1.0 - alpha) * mw * beta_g[jj] / norm_g
                    )
                    elem_viol = abs(score_g[jj] - subgrad)
                else:
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
    """Screen inactive groups using the quadratic-model residual."""
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


# ======================================================================
# Outer loop: proximal Newton at one lambda (group lasso)
# ======================================================================

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
    """Fit the sparse-group-lasso model at one (lambda, alpha) point.

    Parameters
    ----------
    objective_fn : callable
    beta_init : ndarray, shape (p,)
    c : float
    lam, alpha : float
    penalty_factor : ndarray, shape (p,)
    groups : ndarray of int, shape (p,)
    group_weights : ndarray, shape (G,)
    group_starts, group_ends : ndarray of int, shape (G,)
    n_groups : int
    outer_max_iter, outer_tol, inner_max_iter, inner_tol, max_halvings
    use_active_set : bool

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

    if use_active_set:
        gnorms_init = _compute_group_norms(beta, groups, n_groups)
        active_groups = gnorms_init > 0
        if not np.any(active_groups):
            active_groups[:] = True
    else:
        active_groups = np.ones(n_groups, dtype=np.bool_)

    for outer_iter in range(1, outer_max_iter + 1):
        n_outer_iter = outer_iter
        A = c * info
        g = c * score
        linear_term = A @ beta + g

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
        if use_active_set:
            active_groups = inner_active

        loglik_cand, score_cand, info_cand = objective_fn(beta_candidate)
        obj_cand = exact_objective(beta_candidate, loglik_cand)
        if not (np.isfinite(obj_cand) and np.all(np.isfinite(beta_candidate))):
            gnorms = _compute_group_norms(beta, groups, n_groups)
            df = float(np.sum(beta != 0))
            return GroupPenalizedFitResult(
                beta=beta, log_likelihood=loglik, objective_value=obj_val,
                n_outer_iter=n_outer_iter,
                n_inner_iter_total=n_inner_iter_total,
                converged=False,
                message="non-finite objective or coefficients",
                information=info, active_groups=gnorms > 0,
                group_norms=gnorms, df=df,
            )

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
            score_cand, info_cand, obj_cand = score_trial, info_trial, obj_trial

        if obj_cand > obj_val + objective_tol:
            gnorms = _compute_group_norms(beta, groups, n_groups)
            df = float(np.sum(beta != 0))
            return GroupPenalizedFitResult(
                beta=beta, log_likelihood=loglik, objective_value=obj_val,
                n_outer_iter=n_outer_iter,
                n_inner_iter_total=n_inner_iter_total,
                converged=False,
                message="step-halving failed",
                information=info, active_groups=gnorms > 0,
                group_norms=gnorms, df=df,
            )

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

        if use_active_set:
            gnorms_cur = _compute_group_norms(beta, groups, n_groups)
            active_groups = gnorms_cur > 0
            if not np.any(active_groups):
                active_groups = np.ones(n_groups, dtype=np.bool_)

        g_exact = c * score
        kkt_violation = _compute_group_kkt_violation(
            g_exact, beta, lam, alpha, groups, gw, pf, n_groups,
        )
        kkt_threshold = max(10.0 * outer_tol, 1e-12)
        # See the matching comment in ``fit_single_lambda``.
        stall_kkt_tol = max(kkt_threshold, 1e-6)
        if kkt_violation < kkt_threshold:
            if use_active_set and not np.all(active_groups):
                active_groups[:] = True
                continue
            converged = True
            message = "converged (KKT)"
            break
        # REV-001: guard the supplementary test with the KKT residual, so a
        # collapsed step at a non-stationary point is no longer reported as
        # convergence.  See the matching comment in ``fit_single_lambda``.
        if (
            relative_beta_change < outer_tol
            and rel_obj_change < outer_tol
            and kkt_violation < stall_kkt_tol
        ):
            if use_active_set and not np.all(active_groups):
                active_groups[:] = True
                continue
            converged = True
            message = "converged (beta/obj)"
            break

    gnorms = _compute_group_norms(beta, groups, n_groups)
    df = float(np.sum(beta != 0))
    return GroupPenalizedFitResult(
        beta=beta, log_likelihood=loglik, objective_value=obj_val,
        n_outer_iter=n_outer_iter, n_inner_iter_total=n_inner_iter_total,
        converged=converged, message=message, information=info,
        active_groups=gnorms > 0, group_norms=gnorms, df=df,
        kkt_violation=float(kkt_violation),
    )


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
    """Fit the full group-lasso regularization path.

    Parameters
    ----------
    objective_fn : callable
    p : int
    c, alpha : float
    lambda_sequence : sequence of float (descending)
    groups : ndarray of int, shape (p,)
    group_weights : ndarray, shape (G,)
    penalty_factor : ndarray, shape (p,)
    n_groups : int
    beta_warm_start : ndarray or None
    outer_max_iter, outer_tol, inner_max_iter, inner_tol
    use_active_set : bool

    Returns
    -------
    list of GroupPenalizedFitResult
    """
    lam_arr = np.asarray(lambda_sequence, dtype=np.float64)
    if lam_arr.ndim != 1 or lam_arr.size == 0:
        raise ValueError("lambda_sequence must be a non-empty 1D sequence")
    if not np.all(np.isfinite(lam_arr)) or np.any(lam_arr < 0):
        raise ValueError("lambda_sequence must contain finite non-negative values")
    if lam_arr.size > 1 and np.any(np.diff(lam_arr) > 0):
        raise ValueError("lambda_sequence must be sorted descending")
    groups_arr = np.asarray(groups, dtype=np.intp)
    gw = np.asarray(group_weights, dtype=np.float64)
    pf = np.asarray(penalty_factor, dtype=np.float64)
    gs_arr, ge_arr = compute_group_indices(groups_arr, n_groups)

    # ISSUE-010: include unpenalized (group=0) columns as a pseudo-group
    # with zero weight and zero penalty factor so the block-CD kernel
    # updates them with no shrinkage instead of silently skipping them.
    unpen_mask = groups_arr == 0
    _has_unpen = np.any(unpen_mask)
    if _has_unpen:
        unpen_cols = np.where(unpen_mask)[0]
        if not np.array_equal(
            unpen_cols, np.arange(unpen_cols[0], unpen_cols[0] + len(unpen_cols))
        ):
            raise ValueError(
                "Unpenalized (group=0) features must be contiguous "
                f"in column ordering (got columns {unpen_cols.tolist()})"
            )
        pseudo_label = n_groups + 1
        groups_arr = groups_arr.copy()
        groups_arr[unpen_mask] = pseudo_label
        gs_arr = np.append(gs_arr, np.intp(unpen_cols[0]))
        ge_arr = np.append(ge_arr, np.intp(unpen_cols[-1] + 1))
        gw = np.append(gw, 0.0)
        pf = pf.copy()
        pf[unpen_mask] = 0.0
        n_groups += 1

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
            groups_arr, gw, gs_arr, ge_arr, n_groups,
            outer_max_iter=outer_max_iter, outer_tol=outer_tol,
            inner_max_iter=inner_max_iter, inner_tol=inner_tol,
            use_active_set=use_active_set,
        )
        # Strip the pseudo-group from the result so callers see the
        # original n_groups-sized arrays.
        if _has_unpen:
            result = GroupPenalizedFitResult(
                beta=result.beta,
                log_likelihood=result.log_likelihood,
                objective_value=result.objective_value,
                n_outer_iter=result.n_outer_iter,
                n_inner_iter_total=result.n_inner_iter_total,
                converged=result.converged,
                message=result.message,
                information=result.information,
                active_groups=result.active_groups[:-1],
                group_norms=result.group_norms[:-1],
                df=result.df,
                kkt_violation=result.kkt_violation,
            )
        results.append(result)
        beta = result.beta
    return results
