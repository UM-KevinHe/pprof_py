"""Numerical helpers for shared Gamma-frailty Cox models.

The public estimator lives in ``models/survival/frailty_coxph.py``.  This module
contains the Gamma-frailty EM calculations only: posterior frailty moments,
frailty-variance updates, and cluster-level baseline-exposure calculations.
The Cox partial-likelihood calculation itself is deliberately delegated to
``CoxPH`` so the package keeps one source of truth for risk sets, strata,
weights, offsets, start/stop data, and Breslow/Efron ties.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma


def gamma_posterior_moments(
    theta: float,
    event_count: np.ndarray,
    exposure: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return posterior mean and expected log-frailty under Gamma frailty.

    The shared frailty is parameterized as ``z_g ~ Gamma(1/theta, theta)``
    with mean 1 and variance ``theta``.  Conditional on the current Cox
    baseline cumulative exposure ``Lambda_g`` and weighted event count
    ``D_g``, the posterior is

        z_g | data ~ Gamma(1/theta + D_g,
                          rate=1/theta + Lambda_g).

    ``exposure`` is the integrated baseline hazard multiplied by the fixed
    effects' risk score, before multiplying by frailty.
    """
    if theta <= 0 or not np.isfinite(theta):
        raise ValueError("theta must be finite and strictly positive")

    event_count = np.asarray(event_count, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    if event_count.shape != exposure.shape:
        raise ValueError("event_count and exposure must have the same shape")
    if np.any(event_count < 0) or not np.all(np.isfinite(event_count)):
        raise ValueError("event_count must be finite and non-negative")
    if np.any(exposure < 0) or not np.all(np.isfinite(exposure)):
        raise ValueError("exposure must be finite and non-negative")

    shape = 1.0 / theta + event_count
    rate = 1.0 / theta + exposure
    mean = shape / rate
    elog = digamma(shape) - np.log(rate)
    return mean, elog


def update_gamma_theta(
    theta: float,
    posterior_mean: np.ndarray,
    posterior_log_mean: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> float:
    """Update the Gamma-frailty variance by Newton iteration.

    For ``z_g ~ Gamma(a, rate=a)``, ``theta = 1/a`` and the EM M-step solves

        log(a) - psi(a) + 1 + mean(E[log z] - E[z]) = 0.

    A damped positive Newton update is used because theta can be close to
    zero when the random effect is weak.
    """
    posterior_mean = np.asarray(posterior_mean, dtype=np.float64)
    posterior_log_mean = np.asarray(posterior_log_mean, dtype=np.float64)
    if posterior_mean.shape != posterior_log_mean.shape:
        raise ValueError("posterior_mean and posterior_log_mean must have the same shape")
    if posterior_mean.size == 0:
        return float(theta)
    if np.any(~np.isfinite(posterior_mean)) or np.any(~np.isfinite(posterior_log_mean)):
        raise ValueError("posterior frailty moments must be finite")

    a = 1.0 / float(theta)
    target = 1.0 + float(np.mean(posterior_log_mean - posterior_mean))

    for _ in range(max_iter):
        if a <= 0 or not np.isfinite(a):
            break
        f = np.log(a) - digamma(a) + target
        fp = 1.0 / a - polygamma(1, a)
        if not np.isfinite(f) or not np.isfinite(fp) or fp == 0:
            break
        step = f / fp
        new_a = a - step
        if not np.isfinite(new_a) or new_a <= 0:
            new_a = a / 2.0
        # Backtracking for numerical stability.
        while new_a > 0:
            new_f = np.log(new_a) - digamma(new_a) + target
            if np.isfinite(new_f) and abs(new_f) <= abs(f):
                break
            new_a = 0.5 * (new_a + a)
            if abs(new_a - a) <= np.finfo(float).eps * max(1.0, a):
                break
        if abs(new_a - a) / max(1.0, abs(a)) < tol:
            a = new_a
            break
        a = new_a

    if not np.isfinite(a) or a <= 0:
        return float(theta)
    return float(1.0 / a)


def baseline_hazard_lookup(
    query_times: np.ndarray,
    event_times: np.ndarray,
    cumulative_hazard: np.ndarray,
) -> np.ndarray:
    """Evaluate a right-continuous step-function cumulative baseline hazard.

    ``event_times``/``cumulative_hazard`` are a Breslow-type baseline table:
    one row per distinct event time, sorted ascending, with
    ``cumulative_hazard`` non-decreasing. This estimator is a step function
    -- flat between event times, jumping at each one -- so the value at any
    query time is the value at the last jump at or before it.

    This is deliberately NOT linear interpolation between table rows.
    Linear interpolation would overstate H0 at any query time strictly
    between two listed event times, which is the common case for most
    start/stop row boundaries -- censoring times in particular essentially
    never land exactly on another subject's event time. See
    ``group_exposure`` for where this matters.

    ``event_times`` must be sorted ascending, as produced by
    ``compute_baseline_hazard``.
    """
    query_times = np.asarray(query_times, dtype=np.float64)
    if event_times.size == 0:
        return np.zeros_like(query_times, dtype=np.float64)
    idx = np.searchsorted(event_times, query_times, side="right") - 1
    return np.where(idx >= 0, cumulative_hazard[np.clip(idx, 0, None)], 0.0)


def group_exposure(
    eta: np.ndarray,
    baseline_raw: pd.DataFrame,
    group_codes: np.ndarray,
    n_groups: int,
    start: np.ndarray,
    stop: np.ndarray,
    weight: np.ndarray,
    strata_codes: np.ndarray,
    strata_labels,
) -> np.ndarray:
    """Compute each frailty group's baseline cumulative exposure.

    For row i, the contribution is

        exp(eta_i) * [H0(stop_i) - H0(start_i)],

    with H0 evaluated as the right-continuous step function it actually is
    (``baseline_hazard_lookup``), not by linear interpolation between the
    tabulated event times.

    The baseline hazard is stratum-specific, so cumulative exposure is
    accumulated within each fitted stratum before collapsing by frailty
    group. This takes plain arrays (rather than a ``SurvivalData`` object)
    so it stays a pure, independently testable function like the other two
    helpers in this module -- see ``models/survival/frailty_coxph.py`` for
    how the fields of ``SurvivalData`` map onto ``start``/``stop``/``weight``/
    ``strata_codes``/``strata_labels`` at the call site.
    """
    exposure = np.zeros(n_groups, dtype=np.float64)
    if baseline_raw.empty:
        return exposure

    score = np.exp(np.clip(eta, -745.0, 709.0))
    for code, stratum_label in enumerate(strata_labels):
        idx = np.flatnonzero(strata_codes == code)
        if idx.size == 0:
            continue
        table = baseline_raw[baseline_raw["stratum"] == stratum_label]
        if table.empty:
            continue
        times = table["time"].to_numpy(dtype=np.float64)
        cumulative = table["hazard"].to_numpy(dtype=np.float64)
        h_stop = baseline_hazard_lookup(stop[idx], times, cumulative)
        h_start = baseline_hazard_lookup(start[idx], times, cumulative)
        row_exposure = score[idx] * np.maximum(h_stop - h_start, 0.0)
        exposure += np.bincount(
            group_codes[idx], weights=weight[idx] * row_exposure, minlength=n_groups
        )
    return exposure
