"""Robust location-scale estimation: the single estimator behind empirical-null calibration.

Implements the intercept-only fits of R's ``MASS::rlm``:

* ``method="M"`` (``rlm(z ~ 1, method = "M", scale.est = "MAD")``): IRLS with
  the scale re-estimated at every iteration as ``median(|resid|) / 0.6745``
  (residuals about the current location; MASS's literal constant), weights
  ``psi(u) / u`` for Huber's psi or Tukey's bisquare, convergence by MASS's
  ``irls.delta`` on the residuals, and the scale reported from the final
  iteration (``fit$s``).
* ``method="MM"`` (``rlm(z ~ 1, method = "MM")``): an S-estimate of location
  and scale (MASS ``lqs(method = "S", k0 = 1.548)``, with its bisquare
  reweighting refinement), followed by bisquare M-steps with the scale held at
  the S-estimate.

Presets reproduce MASS and EmpiNull defaults exactly, including iteration
counts and convergence flags; any other combination is available through
:class:`MEstimator` or :func:`robust_location_scale`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "LocationScale", "MEstimator", "robust_location_scale",
    "HUBER_RLM", "BISQUARE_RLM", "MM_RLM", "DEFAULT_ESTIMATOR",
]

_MAD_CONSTANT = 0.6745
_DEFAULT_TUNING = {"huber": 1.345, "bisquare": 4.685}
_S_K0, _S_BETA = 1.548, 0.5          # MASS lqs(method = "S"): bisquare chi with k0, breakdown 0.5


@dataclass(frozen=True)
class LocationScale:
    """Result of a robust location-scale fit.

    Attributes
    ----------
    location, scale : float
        Estimated centre and spread (``coef(fit)`` and ``fit$s`` in MASS).
    n_used : int
        Number of finite values the fit used.
    n_iter : int
        M-step iterations performed (``length(fit$conv)`` in MASS).
    converged : bool
        Whether the M-step convergence criterion was met within ``maxiter``.
    """

    location: float
    scale: float
    n_used: int
    n_iter: int
    converged: bool


def _weights(u: np.ndarray, psi: str, c: float) -> np.ndarray:
    if psi == "huber":
        with np.errstate(divide="ignore"):
            return np.minimum(1.0, c / np.abs(u))  # MASS psi.huber: pmin(1, k / abs(u))
    return (1.0 - np.minimum(1.0, np.abs(u / c)) ** 2) ** 2  # MASS psi.bisquare


def _chi(u: np.ndarray) -> np.ndarray:
    """Integrated bisquare used by the S-estimator (MASS lqs.c ``chi``), with ``u = r / (k0 * s)``."""
    x = u * u
    return np.where(x > 1.0, 1.0, x * (3.0 + x * (-3.0 + x)))


def _s_estimate(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """MASS ``lqs(x ~ 1, method = "S", k0 = 1.548)``: residuals and scale after refinement.

    Every observation is tried as a candidate location, in order, which is what
    MASS does whenever it can enumerate all candidates (n < 5000); for larger n
    MASS samples 500 candidates at random, so its result then depends on the
    random seed while this one does not.
    """
    n = x.size
    target = (n - 1) * _S_BETA
    best, best_coef = np.inf, np.nan
    for i in range(n):
        res = x - x[i]
        if i == 0:
            old = np.partition(np.abs(res), n // 2)[n // 2] / _MAD_CONSTANT
        else:
            if np.sum(_chi(res / (_S_K0 * best))) > target:
                continue
            old = best
        new = old
        for _ in range(30):
            total = np.sum(_chi(res / (_S_K0 * old)))
            new = np.sqrt(total / target) * old
            if abs(total / target - 1.0) < 1e-4:
                break
            old = new
        if new < best:
            best, best_coef = new, x[i]
    # IWLS refinement (MASS lqs.R). MASS keeps the refined residuals and scale; the
    # refined coefficient is stored under a misspelled name and is not used downstream.
    resid, scale = x - best_coef, best
    for _ in range(30):
        w = (1.0 - np.minimum(1.0, np.abs(resid / scale / _S_K0)) ** 2) ** 2
        resid = x - np.sum(w * x) / np.sum(w)
        s2 = scale * np.sqrt(np.sum(_chi(resid / scale / _S_K0)) / ((n - 1) * _S_BETA))
        if abs(s2 / scale - 1.0) < 1e-5:
            break
        scale = s2
    return resid, float(scale)


def robust_location_scale(
    z,
    *,
    method: str = "M",
    psi: str = "huber",
    tuning: Optional[float] = None,
    init: Union[str, float] = "mean",
    maxiter: int = 20,
    tol: float = 1e-4,
) -> LocationScale:
    """Estimate a robust location and scale.

    Parameters
    ----------
    z : array-like
        Values to summarise. Non-finite values are excluded (as R's
        ``na.action = na.omit`` does for missing values).
    method : {"M", "MM"}
        ``"M"``: M-estimation with MAD scale re-estimated each iteration.
        ``"MM"``: S-estimate start and scale, then bisquare M-steps with the
        scale fixed; requires ``psi="bisquare"`` and ignores ``init``.
    psi : {"huber", "bisquare"}
        Psi function. Huber's is convex, so the M-estimate does not depend on
        the start; the bisquare redescends and can have local optima.
    tuning : float, optional
        Tuning constant; defaults to 1.345 (Huber) or 4.685 (bisquare), the
        MASS defaults. For MM it must exceed 1.548.
    init : {"mean", "median"} or float
        Starting location for ``method="M"``; ``"mean"`` is MASS's default.
    maxiter, tol : int, float
        M-step iteration limit and convergence tolerance (MASS ``maxit``, ``acc``).

    Returns
    -------
    LocationScale
        A zero scale (more than half the values equal the start) is returned
        as is with ``converged=True``, as MASS does; callers decide how to treat it.
    """
    if method not in ("M", "MM"):
        raise ValueError("method must be 'M' or 'MM'.")
    if psi not in _DEFAULT_TUNING:
        raise ValueError(f"psi must be one of {sorted(_DEFAULT_TUNING)}, got {psi!r}.")
    if method == "MM" and psi != "bisquare":
        raise ValueError("method='MM' uses the bisquare psi (as MASS does); pass psi='bisquare'.")
    c = _DEFAULT_TUNING[psi] if tuning is None else float(tuning)
    if not np.isfinite(c) or c <= 0 or (method == "MM" and c <= _S_K0):
        raise ValueError("tuning must be positive and finite (and greater than 1.548 for MM).")
    if int(maxiter) != maxiter or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    if not tol > 0:
        raise ValueError("tol must be positive.")

    x = np.asarray(z, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        raise ValueError("robust_location_scale needs at least one finite value.")

    if method == "MM":
        if n < 2:
            raise ValueError("method='MM' needs at least two finite values.")
        resid, s = _s_estimate(x)
        mu = np.nan
        if not (np.isfinite(s) and s > 0):
            return LocationScale(float(np.median(x)), float(s) if np.isfinite(s) else 0.0, n, 0, False)
    else:
        if isinstance(init, str):
            if init not in ("mean", "median"):
                raise ValueError("init must be 'mean', 'median', or a number.")
            mu = np.mean(x) if init == "mean" else np.median(x)
        else:
            mu = np.float64(init)
        resid = x - mu
        s = np.median(np.abs(resid)) / _MAD_CONSTANT

    n_iter, converged = 0, False
    for it in range(1, int(maxiter) + 1):
        if method == "M":
            s = np.median(np.abs(resid)) / _MAD_CONSTANT
            if s == 0.0:
                converged = True  # MASS: done <- TRUE; break (before this iteration's update)
                break
        w = _weights(resid / s, psi, c)
        w_sum = np.sum(w)
        if not w_sum > 0:  # bisquare: every point beyond the tuning constant
            break
        mu_new = np.sum(w * x) / w_sum
        resid_new = x - mu_new
        delta = np.sqrt(np.sum((resid - resid_new) ** 2) / max(1e-20, np.sum(resid ** 2)))
        mu, resid, n_iter = mu_new, resid_new, it
        if delta <= tol:
            converged = True
            break
    if method == "MM" and n_iter == 0:
        mu = float(np.median(x))
    return LocationScale(float(mu), float(s), n, n_iter, converged)


@dataclass(frozen=True)
class MEstimator:
    """A configured robust location-scale estimator, callable on a vector.

    Instances are what :meth:`EmpiricalNull.fit` expects as ``estimator``;
    any callable returning a :class:`LocationScale` or a ``(location, scale)``
    tuple can be used instead.
    """

    psi: str = "huber"
    tuning: Optional[float] = None
    init: Union[str, float] = "mean"
    maxiter: int = 20
    tol: float = 1e-4
    method: str = "M"

    def __call__(self, z) -> LocationScale:
        return robust_location_scale(z, method=self.method, psi=self.psi, tuning=self.tuning,
                                     init=self.init, maxiter=self.maxiter, tol=self.tol)


HUBER_RLM = MEstimator(psi="huber", tuning=1.345, init="mean", maxiter=20, tol=1e-4)
"""``MASS::rlm(z ~ 1, method = "M", psi = psi.huber)`` with MASS defaults."""

BISQUARE_RLM = MEstimator(psi="bisquare", tuning=4.685, init="mean", maxiter=20, tol=1e-4)
"""``MASS::rlm(z ~ 1, method = "M", psi = psi.bisquare)`` with MASS defaults."""

MM_RLM = MEstimator(psi="bisquare", tuning=4.685, maxiter=20, tol=1e-4, method="MM")
"""``MASS::rlm(z ~ 1, method = "MM")`` with MASS defaults."""

DEFAULT_ESTIMATOR = MEstimator(psi="bisquare", tuning=4.685, init="mean", maxiter=1000, tol=1e-8)
"""Bisquare M-estimation run to tight convergence: EmpiNull's robust-calibration default."""
