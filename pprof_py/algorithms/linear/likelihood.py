"""Linear (Gaussian) likelihood engine for penalized linear regression.

Simplest case: the IRLS reduces to ordinary weighted least squares
with constant weights.  No iteration is needed -- a single CD pass
per lambda suffices.

Provides ``build_linear_objective`` which returns a closure matching
the ``objective_fn(beta) -> (loglik, score, info)`` interface.

"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def linear_loglik(
    y: np.ndarray, eta: np.ndarray, weight: np.ndarray,
) -> float:
    """Weighted Gaussian log-likelihood (up to additive constant).

    ell = -0.5 * sum_i w_i * (y_i - eta_i)^2

    Parameters
    ----------
    y : ndarray, shape (n,)
    eta : ndarray, shape (n,)
    weight : ndarray, shape (n,)

    Returns
    -------
    float
    """
    resid = y - eta
    return float(-0.5 * np.sum(weight * resid**2))


def linear_score(
    X: np.ndarray, y: np.ndarray, eta: np.ndarray, weight: np.ndarray,
) -> np.ndarray:
    """Score: grad_j = sum_i x_{ij} * w_i * (y_i - eta_i).

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
    eta : ndarray, shape (n,)
    weight : ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (p,)
    """
    resid = weight * (y - eta)
    return X.T @ resid


def linear_information(
    X: np.ndarray, weight: np.ndarray,
) -> np.ndarray:
    """Information matrix: H = X.T @ diag(w) @ X.

    For Gaussian, the Hessian is constant (does not depend on eta),
    so this needs to be computed only once.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    weight : ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (p, p)
    """
    XtW = X.T * weight[np.newaxis, :]
    return XtW @ X


def linear_deviance(
    y: np.ndarray, eta: np.ndarray, weight: np.ndarray,
) -> float:
    """Weighted RSS (deviance for Gaussian).

    Parameters
    ----------
    y, eta, weight : ndarray

    Returns
    -------
    float
    """
    return float(np.sum(weight * (y - eta)**2))


def linear_null_deviance(
    y: np.ndarray, weight: np.ndarray,
) -> float:
    """Null deviance (intercept-only model)."""
    y_bar = np.sum(weight * y) / np.sum(weight)
    return float(np.sum(weight * (y - y_bar)**2))


def build_linear_objective(
    X: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    offset: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]]:
    """Build an objective closure for penalized linear regression.

    Returns ``objective_fn(beta) -> (loglik, score, info)``.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
    weight : ndarray, shape (n,)
    offset : ndarray or None, shape (n,)

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

    # Information is constant for Gaussian -- precompute.
    info = linear_information(X, weight)

    def objective_fn(beta: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        eta = X @ beta + offset_arr
        ll = linear_loglik(y, eta, weight)
        sc = linear_score(X, y, eta, weight)
        return ll, sc, info

    return objective_fn


def linear_null_score(
    X: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    offset: Optional[np.ndarray] = None,
    fit_intercept: bool = True,
) -> Tuple[np.ndarray, float]:
    """Score at the null point for lambda_max computation.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
    weight : ndarray, shape (n,)
    offset : ndarray or None
    fit_intercept : bool

    Returns
    -------
    score : ndarray, shape (p,)
    intercept : float
    """
    n = X.shape[0]
    if offset is None:
        off = np.zeros(n, dtype=np.float64)
    else:
        off = np.asarray(offset, dtype=np.float64)
    if fit_intercept:
        intercept = float(np.sum(weight * (y - off)) / np.sum(weight))
    else:
        intercept = 0.0
    eta = np.full(n, intercept) + off
    score = linear_score(X, y, eta, weight)
    return score, intercept


def linear_intercept_update(
    y: np.ndarray, eta: np.ndarray, weight: np.ndarray,
) -> float:
    """Newton step for the intercept."""
    resid = weight * (y - eta)
    denom = np.sum(weight)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(resid) / denom)


def linear_unpenalized_null_fit(
    X: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    unpenalized: np.ndarray,
    offset: Optional[np.ndarray] = None,
    fit_intercept: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Null point for ``lambda_max``: unpenalized columns fitted, rest at 0.

    Gaussian counterpart of
    :func:`~pprof_py.algorithms.logistic.likelihood.logistic_unpenalized_null_fit`.
    The restricted fit is a weighted least squares solve, so no iteration
    is needed.  See that function for why the null point is not ``beta=0``.

    Returns
    -------
    beta_null : ndarray, shape (p,)
    score : ndarray, shape (p,)
    intercept : float
    """
    n, p = X.shape
    off = (np.zeros(n, dtype=np.float64) if offset is None
           else np.asarray(offset, dtype=np.float64))
    unpen = np.asarray(unpenalized, dtype=bool)
    idx = np.flatnonzero(unpen)

    beta_null = np.zeros(p, dtype=np.float64)
    if idx.size:
        # Weighted LS of (y - offset) on the unpenalized columns, with an
        # intercept column appended when one is being fitted.
        Xu = X[:, idx]
        design = np.column_stack([Xu, np.ones(n)]) if fit_intercept else Xu
        sw = np.sqrt(weight)
        sol, *_ = np.linalg.lstsq(design * sw[:, None], (y - off) * sw,
                                  rcond=None)
        beta_null[idx] = sol[:idx.size]
        intercept = float(sol[-1]) if fit_intercept else 0.0
    else:
        _score0, intercept = linear_null_score(
            X, y, weight, offset=offset, fit_intercept=fit_intercept,
        )

    eta = X @ beta_null + intercept + off
    score = linear_score(X, y, eta, weight)
    return beta_null, score, float(intercept)
