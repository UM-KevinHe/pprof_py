"""Logistic likelihood engine for penalized logistic regression.

Provides the Newton/IRLS primitives that plug into the generic
proximal-Newton + coordinate-descent solver in
``algorithms/coordinate_descent.py``.

The key function is ``build_logistic_objective``, which returns a
closure ``objective_fn(beta) -> (loglik, score, info)`` matching the
same signature used by the Cox partial-likelihood engine.  This
closure is then passed to ``fit_single_lambda`` or
``fit_regularization_path`` without any Cox-specific code.

For the IRLS working-response interpretation:
  - Working response:  z_i = eta_i + (y_i - p_i) / w_i
  - Working weights:   w_i = p_i * (1 - p_i)
  - Score:             grad_j = sum_i x_{ij} * (y_i - p_i) * weight_i
  - Information:       H_{jk} = sum_i x_{ij} * x_{ik} * w_i * weight_i

The score and info returned correspond to the *positive* log-likelihood
(so the penalized objective is ``-c*loglik + penalty``).  This matches
the Cox engine's convention.
"""
from __future__ import annotations

import logging
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np

from ..penalty import njit, _HAS_NUMBA

logger = logging.getLogger(__name__)

# Clamp eta to prevent exp overflow (same bound as glmnet).
_ETA_MAX = 30.0


# -----------------------------------------------------------------------
# Numerically stable logistic primitives
# -----------------------------------------------------------------------

def _safe_expit(eta: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-eta)).

    Avoids overflow by clamping eta.

    Parameters
    ----------
    eta : ndarray
        Linear predictor values.

    Returns
    -------
    ndarray
        Probabilities in (eps, 1-eps).
    """
    eta_clamp = np.clip(eta, -_ETA_MAX, _ETA_MAX)
    p = 1.0 / (1.0 + np.exp(-eta_clamp))
    # Clamp away from exact 0 and 1 to avoid log(0) and division by 0.
    eps = np.finfo(np.float64).eps
    return np.clip(p, eps, 1.0 - eps)


def _safe_log(x: np.ndarray) -> np.ndarray:
    """log(x) with floor at log(eps) to avoid -inf."""
    return np.log(np.maximum(x, np.finfo(np.float64).tiny))


# -----------------------------------------------------------------------
# Core likelihood computations
# -----------------------------------------------------------------------

def logistic_loglik(
    y: np.ndarray, eta: np.ndarray, weight: np.ndarray,
) -> float:
    """Weighted binomial log-likelihood.

    ell = sum_i w_i * [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]

    Parameters
    ----------
    y : ndarray, shape (n,)
        Binary outcome (0 or 1).
    eta : ndarray, shape (n,)
        Linear predictor.
    weight : ndarray, shape (n,)
        Per-observation weights.

    Returns
    -------
    float
        Log-likelihood value.
    """
    p = _safe_expit(eta)
    ll = np.sum(weight * (y * _safe_log(p) + (1.0 - y) * _safe_log(1.0 - p)))
    return float(ll)


def logistic_score(
    X: np.ndarray, y: np.ndarray, eta: np.ndarray, weight: np.ndarray,
) -> np.ndarray:
    """Score (gradient of log-likelihood w.r.t. beta).

    grad_j = sum_i x_{ij} * (y_i - p_i) * w_i

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix.
    y : ndarray, shape (n,)
        Binary outcome.
    eta : ndarray, shape (n,)
        Linear predictor.
    weight : ndarray, shape (n,)
        Per-observation weights.

    Returns
    -------
    ndarray, shape (p,)
        Score vector.
    """
    p = _safe_expit(eta)
    resid = weight * (y - p)  # (n,)
    return X.T @ resid  # (p,)


def logistic_information(
    X: np.ndarray, eta: np.ndarray, weight: np.ndarray,
) -> np.ndarray:
    """Observed information matrix (negative Hessian of log-likelihood).

    H_{jk} = sum_i x_{ij} * x_{ik} * w_i * p_i*(1-p_i)
           = X.T @ diag(weight * p * (1-p)) @ X

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix.
    eta : ndarray, shape (n,)
        Linear predictor.
    weight : ndarray, shape (n,)
        Per-observation weights.

    Returns
    -------
    ndarray, shape (p, p)
        Information matrix (PSD).
    """
    p = _safe_expit(eta)
    w = weight * p * (1.0 - p)  # working weights, (n,)
    # Safeguard: floor tiny weights to avoid numerical issues.
    w = np.maximum(w, 1e-12)
    XtW = X.T * w[np.newaxis, :]  # (p, n)
    return XtW @ X  # (p, p)


def logistic_deviance(
    y: np.ndarray, eta: np.ndarray, weight: np.ndarray,
) -> float:
    """Binomial deviance: -2 * loglik.

    Parameters
    ----------
    y : ndarray, shape (n,)
    eta : ndarray, shape (n,)
    weight : ndarray, shape (n,)

    Returns
    -------
    float
        Deviance (non-negative).
    """
    return -2.0 * logistic_loglik(y, eta, weight)


def logistic_null_deviance(
    y: np.ndarray, weight: np.ndarray,
) -> float:
    """Null deviance (intercept-only model).

    Parameters
    ----------
    y : ndarray, shape (n,)
    weight : ndarray, shape (n,)

    Returns
    -------
    float
    """
    w_sum = np.sum(weight)
    p_bar = np.sum(weight * y) / w_sum
    p_bar = np.clip(p_bar, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    eta_null = np.full(len(y), np.log(p_bar / (1.0 - p_bar)))
    return logistic_deviance(y, eta_null, weight)


# -----------------------------------------------------------------------
# Objective closure builder
# -----------------------------------------------------------------------

def build_logistic_objective(
    X: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    offset: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]]:
    """Build an objective closure for penalized logistic regression.

    Returns a function with signature:
        ``objective_fn(beta) -> (loglik, score, info)``

    This matches the interface expected by ``fit_single_lambda`` and
    ``fit_regularization_path`` in ``coordinate_descent.py``.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix (already standardized if needed).
    y : ndarray, shape (n,)
        Binary outcome.
    weight : ndarray, shape (n,)
        Per-observation weights.
    offset : ndarray or None, shape (n,)
        Fixed offset added to X @ beta.

    Returns
    -------
    callable
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if offset is None:
        offset_arr = np.zeros(X.shape[0], dtype=np.float64)
    else:
        offset_arr = np.asarray(offset, dtype=np.float64)

    def objective_fn(beta: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        eta = X @ beta + offset_arr
        ll = logistic_loglik(y, eta, weight)
        sc = logistic_score(X, y, eta, weight)
        info = logistic_information(X, eta, weight)
        return ll, sc, info

    return objective_fn


# -----------------------------------------------------------------------
# Intercept update for logistic regression
# -----------------------------------------------------------------------

def logistic_intercept_update(
    y: np.ndarray, eta: np.ndarray, weight: np.ndarray,
) -> float:
    """Newton step for the intercept (unpenalized).

    beta_0_new = beta_0_old + sum(w * (y - p)) / sum(w * p * (1-p))

    Parameters
    ----------
    y : ndarray, shape (n,)
    eta : ndarray, shape (n,)
    weight : ndarray, shape (n,)

    Returns
    -------
    float
        Intercept increment (Newton step).
    """
    p = _safe_expit(eta)
    resid = weight * (y - p)
    w_var = weight * p * (1.0 - p)
    denom = np.sum(w_var)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(resid) / denom)


# -----------------------------------------------------------------------
# Working response and weights (for IRLS interpretation)
# -----------------------------------------------------------------------

class IRLSComponents(NamedTuple):
    """Working response and weights for IRLS sub-problem."""
    z: np.ndarray       # working response, shape (n,)
    w: np.ndarray       # working weights, shape (n,)
    p_hat: np.ndarray   # predicted probabilities, shape (n,)


def logistic_irls_components(
    y: np.ndarray, eta: np.ndarray, weight: np.ndarray,
) -> IRLSComponents:
    """Compute IRLS working response and weights.

    z_i = eta_i + (y_i - p_i) / [p_i * (1 - p_i)]
    w_i = weight_i * p_i * (1 - p_i)

    Parameters
    ----------
    y : ndarray, shape (n,)
    eta : ndarray, shape (n,)
    weight : ndarray, shape (n,)

    Returns
    -------
    IRLSComponents
    """
    p = _safe_expit(eta)
    variance = p * (1.0 - p)
    variance = np.maximum(variance, 1e-12)
    z = eta + (y - p) / variance
    w = weight * variance
    return IRLSComponents(z=z, w=w, p_hat=p)


# -----------------------------------------------------------------------
# Null-point score for lambda_max computation
# -----------------------------------------------------------------------

def logistic_null_score(
    X: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    offset: Optional[np.ndarray] = None,
    fit_intercept: bool = True,
) -> Tuple[np.ndarray, float]:
    """Score at the null point (beta=0, intercept at MLE if present).

    Used for lambda_max computation.  When ``fit_intercept=True``,
    the intercept is set to the MLE of the intercept-only model
    (logit of the weighted mean of y), and the score is evaluated
    at (intercept_MLE, beta=0).

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
    weight : ndarray, shape (n,)
    offset : ndarray or None, shape (n,)
    fit_intercept : bool

    Returns
    -------
    score : ndarray, shape (p,)
        Score vector at the null point.
    intercept : float
        Intercept value at the null point (0.0 if not fitting intercept).
    """
    n = X.shape[0]
    if offset is None:
        off = np.zeros(n, dtype=np.float64)
    else:
        off = np.asarray(offset, dtype=np.float64)

    if fit_intercept:
        # Intercept-only MLE: logit(weighted mean of y).
        w_sum = np.sum(weight)
        y_bar = np.sum(weight * y) / w_sum
        y_bar = np.clip(y_bar, 1e-6, 1.0 - 1e-6)
        intercept = float(np.log(y_bar / (1.0 - y_bar)))
    else:
        intercept = 0.0

    eta = np.full(n, intercept) + off
    score = logistic_score(X, y, eta, weight)
    return score, intercept


def logistic_unpenalized_null_fit(
    X: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    unpenalized: np.ndarray,
    offset: Optional[np.ndarray] = None,
    fit_intercept: bool = True,
    max_iter: int = 50,
    tol: float = 1e-11,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Null point for ``lambda_max``: unpenalized columns fitted, rest at 0.

    The correct null point for a regularization path is not ``beta=0``
    everywhere -- it is the point where every *penalized* coefficient is
    zero and every *unpenalized* one (plus the intercept) sits at its own
    MLE.  Evaluating the score there gives a ``lambda_max`` that genuinely
    zeroes the penalized coefficients, and gives the path a warm start
    that already has the unpenalized coefficients right at the top of the
    path instead of leaving them near zero for the first several lambdas.

    Mirrors the reference R implementation's ``SerBIN.residuals``, which
    fits the ``group == 0`` columns before taking the null residual.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix (already standardized).
    y : ndarray, shape (n,)
    weight : ndarray, shape (n,)
    unpenalized : ndarray of bool, shape (p,)
        True for columns that carry no penalty at any lambda.
    offset : ndarray or None, shape (n,)
    fit_intercept : bool
    max_iter, tol : int, float
        Newton iteration controls for the restricted fit.

    Returns
    -------
    beta_null : ndarray, shape (p,)
        Zero except at ``unpenalized``, where it holds the restricted MLE.
    score : ndarray, shape (p,)
        Score of the full coefficient vector evaluated at the null point.
    intercept : float
        Intercept at the null point (0.0 if ``fit_intercept`` is False).
    """
    n, p = X.shape
    off = (np.zeros(n, dtype=np.float64) if offset is None
           else np.asarray(offset, dtype=np.float64))
    unpen = np.asarray(unpenalized, dtype=bool)
    idx = np.flatnonzero(unpen)

    _score0, intercept = logistic_null_score(
        X, y, weight, offset=offset, fit_intercept=fit_intercept,
    )
    beta_null = np.zeros(p, dtype=np.float64)

    if idx.size:
        Xu = X[:, idx]
        for _ in range(max_iter):
            eta = Xu @ beta_null[idx] + intercept + off
            if fit_intercept:
                intercept += logistic_intercept_update(y, eta, weight)
                eta = Xu @ beta_null[idx] + intercept + off
            g = logistic_score(Xu, y, eta, weight)
            H = logistic_information(Xu, eta, weight)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            beta_null[idx] += step
            if float(np.max(np.abs(step))) < tol:
                break

    eta = X @ beta_null + intercept + off
    score = logistic_score(X, y, eta, weight)
    return beta_null, score, float(intercept)
