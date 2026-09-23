"""Shared numerical primitives used across model families.

`col_means`/`safe_exp` originated in the CoxPH engine, where they are the
only numerically "clever" primitives needed (overflow/underflow handled
explicitly rather than left to np.exp() to fail silently into inf/nan).
`sigmoid` is the analogous primitive for the logistic-family models.
"""
from __future__ import annotations

import warnings

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
    """Compute the logistic sigmoid function element-wise, overflow-safe.

    An alias for the package's guarded logistic helper
    (``algorithms.logistic.likelihood._safe_expit``), so every logistic
    probability on the SRR / standardized-measure path uses one convention:
    the linear predictor is clamped to +/-30 (glmnet's bound) and the result
    to ``[eps, 1 - eps]``.

    REV-010: this was previously ``1 / (1 + exp(-x))`` with no guard.  At
    extreme linear predictors it overflowed and returned exactly 0 or 1, so
    ``p * (1 - p)`` became exactly 0 -- which the logit-scale delta method
    divides by -- and ``log(1 - p)`` became ``-inf``.  For ``|x| <= 30`` the
    result is bit-identical to the old formula (neither clamp is active
    there); only saturated inputs change, and only from non-finite /
    exactly-degenerate values to finite ones.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Sigmoid of x, in ``[eps, 1 - eps]``.
    """
    # Imported lazily: utils is the lower layer, and a module-level import of
    # algorithms here would create an import cycle at package load.
    from ..algorithms.logistic.likelihood import _safe_expit
    return _safe_expit(x)


class SingularInformationWarning(RuntimeWarning):
    """A linear system was singular and a pseudo-inverse fallback was used.

    Results computed through the fallback (standard errors in particular)
    are numerically defined but should not be trusted: a singular
    information matrix usually means collinear or aliased covariates, or a
    fit that has not converged.
    """


_FALLBACK_MSG = (
    "{what} is singular; falling back to a pseudo-inverse. Treat any "
    "resulting standard errors as unreliable and check for collinear or "
    "aliased covariates, or non-convergence."
)


def covariance_from_information(information, warn=False, what="Information matrix"):
    """Invert an information matrix, falling back to a pseudo-inverse.

    The success path is exactly ``np.linalg.inv``, so results are unchanged
    whenever the matrix is invertible.  On ``LinAlgError`` it returns
    ``np.linalg.pinv`` instead of raising, so fitting still returns
    something inspectable -- but implausibly large variances should be
    read as a collinearity signal, not as trustworthy standard errors.

    Parameters
    ----------
    information : ndarray, shape (p, p)
    warn : bool, default False
        Emit :class:`SingularInformationWarning` when the fallback fires.
        Defaults to False to preserve the historical silent behaviour of
        existing callers; call sites that previously *raised* on a singular
        matrix pass True, so that a loud failure does not become a silent
        one (REV-011).
    what : str
        Description used in the warning message.
    """
    try:
        return np.linalg.inv(information)
    except np.linalg.LinAlgError:
        if warn:
            warnings.warn(_FALLBACK_MSG.format(what=what),
                          SingularInformationWarning, stacklevel=2)
        return np.linalg.pinv(information)


def solve_information(A, b, warn=False, what="Linear system"):
    """Solve ``A x = b``, falling back to least squares if ``A`` is singular.

    The success path is exactly ``np.linalg.solve`` -- deliberately *not*
    ``inv(A) @ b``, which is numerically different -- so results are
    unchanged whenever ``A`` is invertible.  On ``LinAlgError`` it returns
    the minimum-norm least-squares solution.  See
    :func:`covariance_from_information` for ``warn``.
    """
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        if warn:
            warnings.warn(_FALLBACK_MSG.format(what=what),
                          SingularInformationWarning, stacklevel=2)
        return np.linalg.lstsq(A, b, rcond=None)[0]
