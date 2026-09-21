"""Time-varying-coefficient Cox partial likelihood.

The standard Cox model assumes beta(t) == beta.  A time-varying coefficient
model replaces selected terms beta_j * x_j with

    beta_j * x_j * g_j(t),

where the transformation g_j is evaluated at the event time.  This is the
same mathematical idea exposed by R's ``coxph(..., tt=...)`` interface, but
this package keeps a plain Python callable API rather than a formula parser.

The numerical core below deliberately lives outside the model class.  It
reuses the package's validated (start, stop] risk-set convention, sample
weights, offsets, strata, and Breslow/Efron tie definitions, while allowing
the design vector in a risk set to depend on the current event time.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .ties import TieMethod
from ...utils.numerical import safe_exp
from ...utils.grouping import iter_stratum_indices


def _resolve_transform_specs(X, transforms: Mapping, feature_names: Sequence[str]):
    """Normalize a mapping of feature -> callable into index/function pairs."""
    specs = []
    for key, func in transforms.items():
        if not callable(func):
            raise TypeError(f"Time transform for {key!r} must be callable")
        if isinstance(key, (int, np.integer)):
            j = int(key)
            if j < 0 or j >= X.shape[1]:
                raise IndexError(f"Time-transform feature index {j} is outside [0, {X.shape[1]})")
            name = str(feature_names[j])
        else:
            if key not in feature_names:
                raise ValueError(f"Unknown time-transform feature {key!r}")
            j = list(feature_names).index(key)
            name = str(key)
        specs.append((j, name, func))
    return specs


def _augmented_design_at_time(X, beta, offset, t, specs):
    """Return eta(t) and the transformed design for one event time."""
    n = X.shape[0]
    q = len(specs)
    Z = np.empty((n, X.shape[1] + q), dtype=np.float64)
    Z[:, : X.shape[1]] = X
    eta = X @ beta[: X.shape[1]] + offset
    for k, (j, _name, func) in enumerate(specs):
        g = np.asarray(func(X[:, j], t), dtype=np.float64)
        if g.ndim == 0:
            g = np.full(n, float(g))
        if g.shape != (n,):
            raise ValueError(
                f"Time transform for feature {j} must return shape ({n},), got {g.shape}"
            )
        if not np.all(np.isfinite(g)):
            raise ValueError(f"Time transform for feature {j} returned non-finite values at t={t}")
        Z[:, X.shape[1] + k] = X[:, j] * g
        eta += beta[X.shape[1] + k] * Z[:, X.shape[1] + k]
    return Z, eta


def time_varying_cox_partial_likelihood(
    X: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    beta: np.ndarray,
    offset: np.ndarray,
    weight: np.ndarray,
    strata: np.ndarray,
    transforms: Mapping,
    feature_names: Sequence[str],
    ties: str = "breslow",
):
    """Evaluate log partial likelihood, score, and information.

    ``transforms`` maps selected original feature names/indices to callables
    ``f(x, t)``.  The resulting added term is ``x * f(x, t)`` (the callable
    normally ignores the first argument except when a subject-specific
    transform is desired) and its coefficient is appended after the ordinary
    Cox coefficients.

    The implementation evaluates transformed covariates at each distinct
    event time, which avoids the future-information error that occurs when a
    subject's final follow-up time is used as a proxy for event time.
    """
    if ties not in {"breslow", "efron"}:
        raise NotImplementedError("Time-varying coefficients currently support ties='breslow' or 'efron'.")

    specs = _resolve_transform_specs(X, transforms, feature_names)
    p0 = X.shape[1]
    p = p0 + len(specs)
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape != (p,):
        raise ValueError(f"beta must have length {p}, got shape {beta.shape}")

    total_loglik = 0.0
    total_score = np.zeros(p)
    total_info = np.zeros((p, p))

    for _code, idx in iter_stratum_indices(strata):
        st = start[idx]
        sp = stop[idx]
        ev = event[idx].astype(bool)
        w = weight[idx]
        Xs = X[idx]
        os = offset[idx]
        event_times = np.unique(sp[ev])

        for t in event_times:
            risk = (st < t) & (sp >= t)
            deaths = ev & (sp == t)
            if not np.any(deaths):
                continue

            Z_all, eta_all = _augmented_design_at_time(Xs, beta, os, float(t), specs)
            r = safe_exp(eta_all)
            wr = w * r
            wr_risk = wr[risk]
            Zr = Z_all[risk]
            s0 = float(np.sum(wr_risk))
            if s0 <= 0 or not np.isfinite(s0):
                raise FloatingPointError(
                    "Empty or non-finite risk-set denominator in time-varying Cox fit."
                )

            Zd = Z_all[deaths]
            wd = w[deaths]
            eta_d = eta_all[deaths]
            d_star = float(np.sum(wd))
            total_loglik += float(np.sum(wd * eta_d))

            s1 = np.sum(wr_risk[:, None] * Zr, axis=0)
            xbar = s1 / s0
            s2_full = (Zr.T * wr_risk) @ Zr

            if ties == "breslow" or np.sum(deaths) == 1:
                total_loglik -= d_star * np.log(s0)
                total_score += np.sum(wd[:, None] * Zd, axis=0) - d_star * xbar
                total_info += d_star * (s2_full / s0 - np.outer(xbar, xbar))
                continue

            # Efron: interpolate removal of the tied deaths using RAW death
            # count, while weighting the event contribution by total death mass.
            d_raw = int(np.sum(deaths))
            wrd = wd * safe_exp(eta_d)
            d0 = float(np.sum(wrd))
            d1 = np.sum(wrd[:, None] * Zd, axis=0)
            d2 = (Zd.T * wrd) @ Zd
            meanwt = d_star / d_raw
            event_Z_weighted = np.sum(wd[:, None] * Zd, axis=0)

            for k in range(1, d_raw + 1):
                frac = k / d_raw
                q0 = s0 - d0 + frac * d0
                q1 = s1 - d1 + frac * d1
                q2 = s2_full - d2 + frac * d2
                if q0 <= 0 or not np.isfinite(q0):
                    raise FloatingPointError("Non-positive Efron denominator in time-varying Cox fit.")
                mean = q1 / q0
                total_loglik -= meanwt * np.log(q0)
                total_score -= meanwt * mean
                total_info += meanwt * (q2 / q0 - np.outer(mean, mean))

            total_score += event_Z_weighted

    return total_loglik, total_score, total_info
