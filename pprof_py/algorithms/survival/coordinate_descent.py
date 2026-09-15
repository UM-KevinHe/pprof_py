"""Penalized (ridge / LASSO / elastic net) Cox regression solver.

Architecture
------------
This module is Phase 3's answer to `docs/ARCHITECTURE.md`'s prediction
that "penalization... slot[s] into optimization.py (a penalty term
added to the objective)... without touching the likelihood engine". It
lives alongside `optimization.py` at the algorithm/numerical-kernel
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
target. See `docs/R_COMPATIBILITY.md`, Phase 3 section, for the
empirical comparison against `glmnet` itself.

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
this value; see `docs/R_COMPATIBILITY.md`, Phase 3 section, for the
full numerical comparison.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

from .penalty import soft_threshold, elastic_net_penalty_value
from .risk_sets import njit, _HAS_NUMBA

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


def build_lambda_sequence(lambda_max: float, lambda_min_ratio: float, n_lambda: int) -> np.ndarray:
    """Log-spaced descending sequence from `lambda_max` to
    `lambda_max * lambda_min_ratio`, matching glmnet's
    `exp(seq(log(lambda_max), log(lambda_max*lambda.min.ratio), length.out=nlambda))`.

    A `lambda_max` of exactly 0 (e.g. every column unpenalized) returns
    an all-zero sequence -- there is nothing to regularize.
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
