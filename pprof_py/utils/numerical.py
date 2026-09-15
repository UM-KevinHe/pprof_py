"""Shared numerical primitives used across model families.

`col_means`/`safe_exp` originated in the CoxPH engine, where they are the
only numerically "clever" primitives needed (overflow/underflow handled
explicitly rather than left to np.exp() to fail silently into inf/nan).
`sigmoid` is the analogous primitive for the logistic-family models.
"""
from __future__ import annotations

import numpy as np


def col_means(X: np.ndarray) -> np.ndarray:
    """Simple (unweighted) column means of X, used only to re-center
    covariates before optimization for numerical conditioning.

    Centering every row of X by the same constant vector shifts every
    subject's linear predictor eta_i by the same additive constant,
    which rescales every risk score exp(eta_i) by the same
    multiplicative factor. The Cox partial likelihood depends on X only
    through *ratios* of risk scores within a risk set (see
    algorithms/ties.py), so that constant factor cancels exactly:
    centering changes nothing about beta, its standard errors, or the
    log-likelihood at convergence. It only improves the conditioning of
    the Newton-Raphson iterations when covariates are far from zero
    (e.g. a raw "age" column centered around 60-80), which is why R's
    own coxph centers covariates internally before fitting.

    Because centering is provably inert for every *statistical* output,
    this implementation uses plain (unweighted) column means rather than
    trying to reproduce whatever exact centering constant R's C code
    happens to use -- the two are numerically different intermediate
    values but produce identical beta, SE, and log-likelihood. See
    docs/R_COMPATIBILITY.md, question 14.
    """
    return X.mean(axis=0)


def safe_exp(eta: np.ndarray, clip: float = 700.0) -> np.ndarray:
    """exp(eta) guarded against overflow.

    float64's exp() overflows to inf above eta ~ 709.78; clipping at 700
    leaves headroom while only ever triggering for genuinely pathological
    linear predictors (a converged, centered model fit to reasonably
    scaled covariates should never approach this). An inf risk score
    would silently corrupt every downstream risk-set sum, so clipping
    (with the resulting risk score simply very large rather than
    infinite) is preferable to letting that happen.
    """
    return np.exp(np.clip(eta, -clip, clip))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid function element-wise.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Sigmoid of x.
    """
    return 1 / (1 + np.exp(-x))
