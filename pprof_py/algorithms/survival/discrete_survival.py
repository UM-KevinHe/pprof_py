"""Discrete-time survival model: likelihood, baseline hazard, and residuals.

Architecture
------------
This module implements the numerical kernels for discrete-time survival
models with a logistic link.  It sits alongside ``coordinate_descent.py``
and ``provider_effects.py`` at the algorithm/numerical-kernel layer.

The discrete-time model expresses the conditional hazard at each discrete
timepoint t_k as::

    logit(h(t_k | Z_i)) = alpha_k + eta_i

where ``alpha_k`` are baseline hazard parameters (one per distinct
timepoint, on the logit scale), and ``eta_i = Z_i @ beta + gamma_i`` is
the subject-level linear predictor (covariate effects + optional provider
effect).

The log-likelihood for subject i with event indicator ``delta_i`` and
follow-up spanning discrete timepoints ``t_1, ..., t_{T_i}`` is::

    ell_i = delta_i * log(p_{i,T_i}) + sum_{k < T_i} log(1 - p_{i,k})
          + (1 - delta_i) * log(1 - p_{i,T_i})

where ``p_{i,k} = expit(alpha_k + eta_i)``.

Equivalently, on the person-period expanded data (one row per
subject-timepoint), this is the standard Bernoulli log-likelihood
``y_{ik} * log(p_{ik}) + (1 - y_{ik}) * log(1 - p_{ik})``
with ``y_{ik} = 1`` only for the event row.

Implementation note
-------------------
Following the ``grplasso`` R package (``disc_surv_lasso.cpp``), the
internal fitting routines do **not** physically expand the data to
person-period format.  Instead, they keep one row per subject and
iterate over the relevant timepoints in inner loops (more memory-
efficient for large datasets).  A separate ``person_period_expand()``
function is provided for CV prediction and diagnostics.

The module provides both Numba-accelerated and pure-Python
implementations; the Numba path is selected automatically when
available, following the pattern established in ``risk_sets.py``.

References
----------
.. [1] He, K., Kalbfleisch, J., Li, Y., et al. (2013). Evaluating
   hospital readmission rates in dialysis facilities; adjusting for
   hospital effects. *Lifetime Data Analysis*, 19, 490-512.
.. [2] Shao, Y. & He, K. (2026). grplasso R package.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np

from .risk_sets import njit, _HAS_NUMBA

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class DiscreteSurvivalLikelihoodResult(NamedTuple):
    """Result of a discrete-time survival likelihood evaluation.

    Attributes
    ----------
    neg_loglik : float
        Negative log-likelihood.
    score_beta : ndarray, shape (p,)
        Gradient of the negative log-likelihood w.r.t. eta (= Z @ beta
        part), summed over timepoints.  This is the per-subject
        quantity ``sum_k p_{ik} - delta_i``.
    working_weights : ndarray, shape (n,)
        IRLS working weights for the beta update.  For Newton:
        ``sum_k p_{ik} * (1 - p_{ik})``.  For MM: ``v * T_i`` where
        ``v = 0.25``.
    """
    neg_loglik: float
    score_beta: np.ndarray
    working_weights: np.ndarray


# ---------------------------------------------------------------------------
# Data preparation: discretize times and compute KM-based alpha init
# ---------------------------------------------------------------------------

def discretize_times(
    time: np.ndarray,
    event: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map observed times to contiguous integer indices ``1..K``.

    Parameters
    ----------
    time : ndarray, shape (n,)
        Follow-up times (can be float or int).
    event : ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).

    Returns
    -------
    time_int : ndarray of int, shape (n,)
        Integer-coded times in ``{1, 2, ..., K}``.
    timepoint_map : ndarray, shape (K,)
        Sorted unique original timepoints; ``timepoint_map[k-1]``
        gives the original value for integer code ``k``.
    n_events_per_timepoint : ndarray of int, shape (K,)
        Number of events at each discrete timepoint.
    """
    timepoints = np.sort(np.unique(time))
    # Map: original time -> 1-based index
    time_int = np.searchsorted(timepoints, time) + 1  # 1-based
    K = len(timepoints)
    n_events = np.zeros(K, dtype=np.int64)
    for k in range(K):
        mask = (time_int == (k + 1)) & (event == 1)
        n_events[k] = int(np.sum(mask))
    return time_int.astype(np.int64), timepoints, n_events


def initialize_baseline_hazard(
    n_events_per_timepoint: np.ndarray,
    n_at_risk_per_timepoint: np.ndarray,
) -> np.ndarray:
    """Initialize baseline hazard parameters from KM-like estimates.

    Computes the logit of the crude hazard at each timepoint:
    ``alpha_k = log(h_k / (1 - h_k))`` where ``h_k = d_k / n_k``.

    Parameters
    ----------
    n_events_per_timepoint : ndarray, shape (K,)
        Number of events at each discrete timepoint.
    n_at_risk_per_timepoint : ndarray, shape (K,)
        Number at risk at each discrete timepoint.

    Returns
    -------
    alpha : ndarray, shape (K,)
        Initial baseline hazard parameters (logit scale).
    """
    h = n_events_per_timepoint / np.maximum(n_at_risk_per_timepoint, 1.0)
    # Clamp to avoid log(0) or log(inf)
    h = np.clip(h, 1e-10, 1.0 - 1e-10)
    return np.log(h / (1.0 - h))


def compute_n_at_risk(
    time_int: np.ndarray,
    K: int,
) -> np.ndarray:
    """Compute the number of subjects at risk at each discrete timepoint.

    Subject i is at risk at timepoint k if ``time_int[i] >= k``,
    i.e. they have not yet been censored or had an event before k.

    Parameters
    ----------
    time_int : ndarray of int, shape (n,)
        Integer-coded follow-up times (1-based).
    K : int
        Number of distinct timepoints.

    Returns
    -------
    n_at_risk : ndarray, shape (K,)
    """
    # Vectorised via reverse cumsum of histogram — O(n + K) instead of O(n·K)
    hist = np.bincount(time_int, minlength=K + 1)  # hist[k] = count of time_int == k
    # n_at_risk[k] = number with time_int >= k+1 = total - cumsum of hist[1..k]
    n_at_risk = np.cumsum(hist[1:][::-1])[::-1].astype(np.float64)
    return n_at_risk


# ---------------------------------------------------------------------------
# Person-period expansion (for CV prediction and diagnostics)
# ---------------------------------------------------------------------------

def person_period_expand(
    time_int: np.ndarray,
    event: np.ndarray,
    X: Optional[np.ndarray] = None,
    eta: Optional[np.ndarray] = None,
) -> dict:
    """Expand subject-level data to person-period (long) format.

    Each subject i contributes ``time_int[i]`` rows, one for each
    discrete timepoint ``k = 1, ..., time_int[i]``.  The binary
    outcome ``y_{ik}`` is 1 only when ``k == time_int[i]`` and
    ``event[i] == 1``.

    Parameters
    ----------
    time_int : ndarray of int, shape (n,)
        Integer-coded follow-up times (1-based).
    event : ndarray, shape (n,)
        Event indicators.
    X : ndarray or None, shape (n, p)
        If provided, expanded to long format.
    eta : ndarray or None, shape (n,) or (n, n_lambda)
        If provided, expanded to long format.

    Returns
    -------
    dict with keys:
        'y' : ndarray, shape (N_expanded,)
            Binary outcome in long format.
        'timepoint' : ndarray of int, shape (N_expanded,)
            Timepoint index (1-based) for each expanded row.
        'subject_idx' : ndarray of int, shape (N_expanded,)
            Original subject index for each expanded row.
        'X' : ndarray or None, shape (N_expanded, p)
        'eta' : ndarray or None, shape (N_expanded,) or (N_expanded, n_lambda)
    """
    n = len(time_int)
    time_int_i = time_int.astype(np.intp)
    N = int(np.sum(time_int_i))

    # Vectorised expansion — no Python loops
    subject_idx = np.repeat(np.arange(n, dtype=np.int64), time_int_i)
    timepoint = np.concatenate(
        [np.arange(1, int(Ti) + 1, dtype=np.int64) for Ti in time_int_i]
    )

    # y_{ik} = 1 only at the last row of each event subject
    y = np.zeros(N, dtype=np.float64)
    cum = np.cumsum(time_int_i)
    event_mask = np.asarray(event, dtype=bool)
    if np.any(event_mask):
        y[cum[event_mask] - 1] = 1.0

    result = {
        'y': y,
        'timepoint': timepoint,
        'subject_idx': subject_idx,
        'X': X[subject_idx] if X is not None else None,
        'eta': eta[subject_idx] if eta is not None else None,
    }
    return result


# ---------------------------------------------------------------------------
# Core likelihood computation (Numba-accelerated + Python fallback)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _discrete_loglik_numba(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
    use_mm: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Numba kernel for discrete-time survival log-likelihood.

    Parameters
    ----------
    time_int : ndarray of int, shape (n,)
        Integer-coded follow-up times (1-based).
    event : ndarray, shape (n,)
        Event indicators (1.0 = event, 0.0 = censored).
    alpha : ndarray, shape (K,)
        Baseline hazard parameters (logit scale).
    eta : ndarray, shape (n,)
        Linear predictor (Z @ beta + provider effect + offset).
    use_mm : bool
        If True, use MM working weights (v * T_i, v=0.25);
        if False, use Newton working weights (exact Hessian diagonal).

    Returns
    -------
    neg_loglik : float
    score_beta : ndarray, shape (n,)
        Per-subject gradient: sum_k p_{ik} - delta_i.
    working_weights : ndarray, shape (n,)
        Per-subject IRLS working weights.
    """
    n = len(time_int)
    v = 0.25
    omega_min = 1e-10
    neg_ll = 0.0
    score = np.empty(n)
    weights = np.empty(n)

    for j in range(n):
        Ti = time_int[j]
        sum_p = 0.0
        sum_pq = 0.0
        ll_j = 0.0
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            # Numerically stable sigmoid
            if lin >= 0.0:
                p_jk = 1.0 / (1.0 + np.exp(-lin))
            else:
                exp_lin = np.exp(lin)
                p_jk = exp_lin / (1.0 + exp_lin)
            sum_p += p_jk
            sum_pq += p_jk * (1.0 - p_jk)
            # Log-likelihood contribution
            if k == Ti - 1 and event[j] == 1.0:
                # Event row: y=1
                ll_j += np.log(max(p_jk, 1e-15))
            else:
                # Non-event row: y=0
                ll_j += np.log(max(1.0 - p_jk, 1e-15))
        neg_ll -= ll_j
        # Score for beta update: sum_k p_{ik} - delta_i
        score[j] = sum_p - event[j]
        if use_mm:
            weights[j] = v * Ti
        else:
            weights[j] = max(sum_pq, omega_min)

    return neg_ll, score, weights


def _discrete_loglik_python(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
    use_mm: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Pure-Python fallback for discrete-time survival log-likelihood."""
    n = len(time_int)
    v = 0.25
    omega_min = 1e-10
    neg_ll = 0.0
    score = np.empty(n)
    weights = np.empty(n)

    for j in range(n):
        Ti = int(time_int[j])
        sum_p = 0.0
        sum_pq = 0.0
        ll_j = 0.0
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            p_jk = 1.0 / (1.0 + np.exp(-np.clip(lin, -500, 500)))
            sum_p += p_jk
            sum_pq += p_jk * (1.0 - p_jk)
            if k == Ti - 1 and event[j] == 1.0:
                ll_j += np.log(max(p_jk, 1e-15))
            else:
                ll_j += np.log(max(1.0 - p_jk, 1e-15))
        neg_ll -= ll_j
        score[j] = sum_p - event[j]
        if use_mm:
            weights[j] = v * Ti
        else:
            weights[j] = max(sum_pq, omega_min)

    return neg_ll, score, weights


def discrete_loglik(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
    use_mm: bool = False,
) -> DiscreteSurvivalLikelihoodResult:
    """Compute the discrete-time survival negative log-likelihood.

    Dispatches to the Numba kernel when available.

    Parameters
    ----------
    time_int : ndarray of int, shape (n,)
        Integer-coded follow-up times (1-based).
    event : ndarray, shape (n,)
        Event indicators.
    alpha : ndarray, shape (K,)
        Baseline hazard parameters (logit scale).
    eta : ndarray, shape (n,)
        Linear predictor.
    use_mm : bool, default False
        Use MM working weights (v=0.25 * T_i) instead of Newton.

    Returns
    -------
    DiscreteSurvivalLikelihoodResult
    """
    if _HAS_NUMBA:
        neg_ll, score, wt = _discrete_loglik_numba(
            time_int, event.astype(np.float64), alpha, eta, use_mm,
        )
    else:
        neg_ll, score, wt = _discrete_loglik_python(
            time_int, event.astype(np.float64), alpha, eta, use_mm,
        )
    return DiscreteSurvivalLikelihoodResult(
        neg_loglik=float(neg_ll),
        score_beta=score,
        working_weights=wt,
    )


# ---------------------------------------------------------------------------
# Baseline hazard (alpha_k) update — Newton step
# ---------------------------------------------------------------------------

@njit(cache=True)
def _baseline_hazard_score_info_numba(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
    n_events_per_timepoint: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute score and information for baseline hazard parameters.

    Returns
    -------
    score : ndarray, shape (K,)
        ``sum_{i: T_i >= k} p_{ik} - d_k`` (gradient of neg loglik
        w.r.t. alpha_k).
    info : ndarray, shape (K,)
        ``-sum_{i: T_i >= k} p_{ik} * (1 - p_{ik})`` (negative
        diagonal of Hessian).
    """
    n = len(time_int)
    K = len(alpha)
    score = np.empty(K)
    info = np.empty(K)
    for k in range(K):
        score[k] = -float(n_events_per_timepoint[k])
        info[k] = 0.0
    for j in range(n):
        Ti = time_int[j]
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            if lin >= 0.0:
                p_jk = 1.0 / (1.0 + np.exp(-lin))
            else:
                exp_lin = np.exp(lin)
                p_jk = exp_lin / (1.0 + exp_lin)
            score[k] += p_jk
            info[k] -= p_jk * (1.0 - p_jk)
    return score, info


def _baseline_hazard_score_info_python(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
    n_events_per_timepoint: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-Python fallback."""
    n = len(time_int)
    K = len(alpha)
    score = -n_events_per_timepoint.astype(np.float64).copy()
    info = np.zeros(K, dtype=np.float64)
    for j in range(n):
        Ti = int(time_int[j])
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            p_jk = 1.0 / (1.0 + np.exp(-np.clip(lin, -500, 500)))
            score[k] += p_jk
            info[k] -= p_jk * (1.0 - p_jk)
    return score, info


def baseline_hazard_update(
    alpha: np.ndarray,
    time_int: np.ndarray,
    event: np.ndarray,
    eta: np.ndarray,
    n_events_per_timepoint: np.ndarray,
    bound: float = 10.0,
    backtrack: bool = False,
) -> np.ndarray:
    """One Newton step for baseline hazard parameters alpha_k.

    Parameters
    ----------
    alpha : ndarray, shape (K,)
        Current baseline hazard parameters.
    time_int : ndarray of int, shape (n,)
        Integer-coded follow-up times (1-based).
    event : ndarray, shape (n,)
        Event indicators.
    eta : ndarray, shape (n,)
        Current linear predictor (Z @ beta + provider_effect).
    n_events_per_timepoint : ndarray of int, shape (K,)
        Events at each discrete timepoint.
    bound : float, default 10.0
        Bounding constraint: ``alpha_k`` clamped to
        ``median(alpha) +/- bound``.
    backtrack : bool, default False
        Use Armijo backtracking line search.

    Returns
    -------
    alpha_new : ndarray, shape (K,)
        Updated baseline hazard parameters.
    """
    omega_min = 1e-10
    if _HAS_NUMBA:
        score, info = _baseline_hazard_score_info_numba(
            time_int, event.astype(np.float64), alpha, eta, n_events_per_timepoint,
        )
    else:
        score, info = _baseline_hazard_score_info_python(
            time_int, event.astype(np.float64), alpha, eta, n_events_per_timepoint,
        )

    # Newton direction: d_alpha = score / min(-omega_min, info)
    safe_info = np.minimum(info, -omega_min)
    d_alpha = score / safe_info

    if backtrack:
        # Armijo backtracking line search
        ll_current = _discrete_total_loglik(time_int, event, alpha, eta)
        s, t_bt = 0.01, 0.8
        u = 1.0
        k_dot = np.dot(score, d_alpha)
        for _ in range(50):
            alpha_try = alpha + u * d_alpha
            ll_try = _discrete_total_loglik(time_int, event, alpha_try, eta)
            if ll_try - ll_current >= s * u * k_dot:
                break
            u *= t_bt
        alpha_new = alpha + u * d_alpha
    else:
        alpha_new = alpha + d_alpha

    # Bound: clamp to median(alpha) +/- bound
    med = np.median(alpha_new)
    alpha_new = np.clip(alpha_new, med - bound, med + bound)
    return alpha_new


@njit(cache=True)
def _discrete_total_loglik_numba(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
) -> float:
    """Numba kernel for total log-likelihood."""
    n = len(time_int)
    ll = 0.0
    for j in range(n):
        Ti = time_int[j]
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            if lin >= 0.0:
                p_jk = 1.0 / (1.0 + np.exp(-lin))
            else:
                exp_lin = np.exp(lin)
                p_jk = exp_lin / (1.0 + exp_lin)
            if k == Ti - 1 and event[j] == 1.0:
                ll += np.log(max(p_jk, 1e-15))
            else:
                ll += np.log(max(1.0 - p_jk, 1e-15))
    return ll


def _discrete_total_loglik_python(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
) -> float:
    """Pure-Python fallback."""
    n = len(time_int)
    ll = 0.0
    for j in range(n):
        Ti = int(time_int[j])
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            p_jk = 1.0 / (1.0 + np.exp(-np.clip(lin, -500, 500)))
            if k == Ti - 1 and event[j] == 1.0:
                ll += np.log(max(p_jk, 1e-15))
            else:
                ll += np.log(max(1.0 - p_jk, 1e-15))
    return ll


def _discrete_total_loglik(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
) -> float:
    """Total log-likelihood (for backtracking line search)."""
    evt = event.astype(np.float64)
    if _HAS_NUMBA:
        return float(_discrete_total_loglik_numba(time_int, evt, alpha, eta))
    return float(_discrete_total_loglik_python(time_int, evt, alpha, eta))


# ---------------------------------------------------------------------------
# Discrete-time residuals
# ---------------------------------------------------------------------------

@njit(cache=True)
def _discrete_residuals_numba(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Numba kernel for discrete-time residuals."""
    n = len(time_int)
    r = np.empty(n, dtype=np.float64)
    for j in range(n):
        Ti = time_int[j]
        sum_p = 0.0
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            if lin >= 0.0:
                sum_p += 1.0 / (1.0 + np.exp(-lin))
            else:
                exp_lin = np.exp(lin)
                sum_p += exp_lin / (1.0 + exp_lin)
        r[j] = event[j] - sum_p
    return r


def _discrete_residuals_python(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Pure-Python fallback for residuals."""
    n = len(time_int)
    r = np.empty(n, dtype=np.float64)
    for j in range(n):
        Ti = int(time_int[j])
        sum_p = 0.0
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            sum_p += 1.0 / (1.0 + np.exp(-np.clip(lin, -500, 500)))
        r[j] = event[j] - sum_p
    return r


def discrete_residuals(
    time_int: np.ndarray,
    event: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Martingale-like residuals for the discrete-time model.

    For subject i::

        r_i = delta_i - sum_{k=1}^{T_i} p_{ik}

    where ``p_{ik} = expit(alpha_k + eta_i)``.

    Matches ``grplasso``'s ``DiscSurv_residuals`` (disc_surv_lasso.cpp).

    Parameters
    ----------
    time_int : ndarray of int, shape (n,)
        Integer-coded follow-up times (1-based).
    event : ndarray, shape (n,)
        Event indicators.
    alpha : ndarray, shape (K,)
        Baseline hazard parameters.
    eta : ndarray, shape (n,)
        Linear predictor.

    Returns
    -------
    residuals : ndarray, shape (n,)
    """
    evt = event.astype(np.float64)
    if _HAS_NUMBA:
        return _discrete_residuals_numba(time_int, evt, alpha, eta)
    return _discrete_residuals_python(time_int, evt, alpha, eta)


# ---------------------------------------------------------------------------
# Lambda-max computation for discrete-time model
# ---------------------------------------------------------------------------

def compute_discrete_lambda_max(
    residuals: np.ndarray,
    X: np.ndarray,
    penalty_factor: np.ndarray,
    groups: Optional[np.ndarray] = None,
    group_weights: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> float:
    """Smallest lambda that zeros all penalized coefficients at beta=0.

    For individual (element-wise) penalties::

        lambda_max = max_j(|X_j^T r| / (n * pf_j))

    For group penalties, uses the same KKT formula as
    ``compute_group_lambda_max``.

    Parameters
    ----------
    residuals : ndarray, shape (n,)
        Residuals at beta=0 (from ``discrete_residuals``).
    X : ndarray, shape (n, p)
        Design matrix.
    penalty_factor : ndarray, shape (p,)
        Per-feature penalty factors.
    groups : ndarray or None, shape (p,)
        Group labels (0 = unpenalized). Required for group penalties.
    group_weights : ndarray or None, shape (n_groups,)
        Per-group multipliers. Required for group penalties.
    alpha : float, default 1.0
        Sparse group lasso mixing (1.0 for individual lasso).

    Returns
    -------
    lambda_max : float
    """
    n = X.shape[0]
    gradient = X.T @ residuals  # shape (p,)

    if groups is None:
        # Element-wise lambda max
        penalized = penalty_factor > 0
        if not np.any(penalized):
            return 0.0
        lam_max = np.max(np.abs(gradient[penalized]) / (n * penalty_factor[penalized]))
        return float(lam_max)
    else:
        # Group-level lambda max (same formula as compute_group_lambda_max)
        from .penalty import compute_group_indices
        group_starts, group_ends = compute_group_indices(groups)
        n_groups = len(group_starts)
        lam_max = 0.0
        for g in range(n_groups):
            s, e = group_starts[g], group_ends[g]
            grad_g = gradient[s:e] / n
            if alpha > 0 and alpha < 1.0:
                # Sparse group lasso KKT at beta=0
                # ||soft_threshold(grad_g, lam*alpha*pf_g)||_2 <= lam*(1-alpha)*m_g
                # Binary search or closed-form bound
                grad_norm = np.linalg.norm(grad_g)
                mg = group_weights[g] if group_weights is not None else 1.0
                # Upper bound: lambda >= ||grad_g||_2 / ((1-alpha)*mg + alpha*min(pf_g))
                pf_g = penalty_factor[s:e]
                lam_g = grad_norm / ((1.0 - alpha) * mg)
                lam_max = max(lam_max, lam_g)
            elif alpha == 0.0:
                # Pure group lasso
                grad_norm = np.linalg.norm(grad_g)
                mg = group_weights[g] if group_weights is not None else 1.0
                lam_g = grad_norm / mg
                lam_max = max(lam_max, lam_g)
            else:
                # alpha == 1.0: standard lasso
                pf_g = penalty_factor[s:e]
                penalized = pf_g > 0
                if np.any(penalized):
                    lam_g = np.max(np.abs(grad_g[penalized]) / pf_g[penalized])
                    lam_max = max(lam_max, lam_g)
        return float(lam_max)


# ---------------------------------------------------------------------------
# Coordinate descent for beta (IRLS-style, reusing working response)
# ---------------------------------------------------------------------------

def compute_working_response(
    score_beta: np.ndarray,
    working_weights: np.ndarray,
) -> np.ndarray:
    """Compute the IRLS working response for the beta coordinate descent.

    The working response is ``r = -score / w`` where score is the
    per-subject gradient and w are the working weights.  This is the
    input to the coordinate descent solver.

    Parameters
    ----------
    score_beta : ndarray, shape (n,)
        Per-subject gradient (``sum_k p_{ik} - delta_i``).
    working_weights : ndarray, shape (n,)
        Per-subject working weights.

    Returns
    -------
    working_response : ndarray, shape (n,)
    """
    return -score_beta / working_weights


@njit(cache=True)
def _discrete_cd_numba(
    X: np.ndarray,
    working_response: np.ndarray,
    working_weights: np.ndarray,
    beta: np.ndarray,
    lam: float,
    penalty_factor: np.ndarray,
    tol: float,
    max_iter: int,
) -> Tuple[np.ndarray, float, float]:
    """Numba kernel for weighted coordinate descent with soft-threshold."""
    n = X.shape[0]
    p = X.shape[1]
    r = working_response.copy()
    old_beta = beta.copy()
    max_change = 0.0
    df = 0.0

    for iteration in range(max_iter):
        max_change = 0.0
        df = 0.0
        for j in range(p):
            # Weighted inner products  (explicit loop for Numba)
            wz_sum = 0.0
            wxx_sum = 0.0
            for i in range(n):
                wXij = working_weights[i] * X[i, j]
                wz_sum += wXij * r[i]
                wxx_sum += wXij * X[i, j]

            if wxx_sum == 0.0:
                continue

            z_j = wz_sum / wxx_sum + old_beta[j]

            if penalty_factor[j] == 0.0:
                new_beta_j = z_j
            else:
                threshold = n * lam * penalty_factor[j] / wxx_sum
                if z_j > threshold:
                    new_beta_j = z_j - threshold
                elif z_j < -threshold:
                    new_beta_j = z_j + threshold
                else:
                    new_beta_j = 0.0

            if new_beta_j != old_beta[j]:
                change = new_beta_j - old_beta[j]
                if abs(change) > max_change:
                    max_change = abs(change)
                for i in range(n):
                    r[i] -= change * X[i, j]
                beta[j] = new_beta_j

            if new_beta_j != 0.0 and z_j != 0.0:
                df += abs(new_beta_j) / abs(z_j)

        for j in range(p):
            old_beta[j] = beta[j]
        if max_change < tol:
            break

    return beta, max_change, df


def _discrete_cd_python(
    X: np.ndarray,
    working_response: np.ndarray,
    working_weights: np.ndarray,
    beta: np.ndarray,
    lam: float,
    penalty_factor: np.ndarray,
    tol: float,
    max_iter: int,
) -> Tuple[np.ndarray, float, float]:
    """Pure-Python fallback for weighted coordinate descent."""
    n, p = X.shape
    r = working_response.copy()
    old_beta = beta.copy()
    max_change = 0.0
    df = 0.0

    for iteration in range(max_iter):
        max_change = 0.0
        df = 0.0
        for j in range(p):
            wXj = working_weights * X[:, j]
            denom = np.dot(wXj, X[:, j])
            if denom == 0.0:
                continue
            z_j = np.dot(wXj, r) / denom + old_beta[j]

            if penalty_factor[j] == 0.0:
                new_beta_j = z_j
            else:
                threshold = n * lam * penalty_factor[j] / denom
                new_beta_j = np.sign(z_j) * max(abs(z_j) - threshold, 0.0)

            if new_beta_j != old_beta[j]:
                change = new_beta_j - old_beta[j]
                if abs(change) > max_change:
                    max_change = abs(change)
                r -= change * X[:, j]
                beta[j] = new_beta_j

            if new_beta_j != 0.0 and z_j != 0.0:
                df += abs(new_beta_j) / abs(z_j)

        old_beta = beta.copy()
        if max_change < tol:
            break

    return beta, max_change, df


def discrete_coordinate_descent_step(
    X: np.ndarray,
    working_response: np.ndarray,
    working_weights: np.ndarray,
    beta: np.ndarray,
    lam: float,
    penalty_factor: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 10000,
) -> Tuple[np.ndarray, float, float]:
    """One pass of weighted coordinate descent for beta.

    Solves the penalized weighted least squares problem::

        min_beta  0.5 * sum_i w_i (r_i - X_i @ beta)^2
                  + n * lam * sum_j pf_j |beta_j|

    via cyclic coordinate descent with soft-thresholding.
    Dispatches to Numba kernel when available.

    This implements the element-wise L1 penalty.  For group penalties,
    use the existing ``solve_sparse_group_penalized_quadratic`` from
    ``coordinate_descent.py`` after constructing the appropriate
    working-response / weight pair.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix.
    working_response : ndarray, shape (n,)
        IRLS working response.
    working_weights : ndarray, shape (n,)
        IRLS working weights.
    beta : ndarray, shape (p,)
        Current coefficients (modified in place).
    lam : float
        Regularization strength.
    penalty_factor : ndarray, shape (p,)
        Per-feature penalty factors (0 for unpenalized).
    tol : float, default 1e-4
        Convergence tolerance on max |delta_beta|.
    max_iter : int, default 10000
        Maximum number of full sweeps.

    Returns
    -------
    beta : ndarray, shape (p,)
        Updated coefficients.
    max_change : float
        Maximum absolute change in any coefficient.
    df : float
        Effective degrees of freedom (sum of |beta_j/z_j| indicators).
    """
    if _HAS_NUMBA:
        return _discrete_cd_numba(
            X, working_response, working_weights, beta,
            lam, penalty_factor, tol, max_iter,
        )
    return _discrete_cd_python(
        X, working_response, working_weights, beta,
        lam, penalty_factor, tol, max_iter,
    )


# ---------------------------------------------------------------------------
# Single-lambda fit for discrete-time survival
# ---------------------------------------------------------------------------

class DiscreteFitResult(NamedTuple):
    """Result of fitting one lambda for a discrete-time survival model.

    Attributes
    ----------
    beta : ndarray, shape (p,)
        Fitted covariate coefficients.
    alpha : ndarray, shape (K,)
        Fitted baseline hazard parameters.
    neg_loglik : float
        Negative log-likelihood at the converged iterate.
    df : float
        Effective degrees of freedom.
    n_iter : int
        Number of outer iterations.
    converged : bool
        Whether the algorithm converged.
    eta : ndarray, shape (n,)
        Converged linear predictor.
    """
    beta: np.ndarray
    alpha: np.ndarray
    neg_loglik: float
    df: float
    n_iter: int
    converged: bool
    eta: np.ndarray


def fit_single_lambda_discrete(
    X: np.ndarray,
    time_int: np.ndarray,
    event: np.ndarray,
    n_events_per_timepoint: np.ndarray,
    lam: float,
    penalty_factor: np.ndarray,
    beta_init: np.ndarray,
    alpha_init: np.ndarray,
    eta_init: Optional[np.ndarray] = None,
    use_mm: bool = False,
    bound: float = 10.0,
    backtrack: bool = False,
    tol: float = 1e-4,
    max_iter: int = 10000,
    active_set: bool = True,
) -> DiscreteFitResult:
    """Fit a single lambda for the discrete-time survival model.

    Alternates between:
    1. Baseline hazard (alpha) Newton update
    2. Covariate (beta) IRLS + coordinate descent update

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix (standardized).
    time_int : ndarray of int, shape (n,)
        Integer-coded follow-up times (1-based).
    event : ndarray, shape (n,)
        Event indicators.
    n_events_per_timepoint : ndarray of int, shape (K,)
        Events at each discrete timepoint.
    lam : float
        Regularization strength.
    penalty_factor : ndarray, shape (p,)
        Per-feature penalty factors.
    beta_init : ndarray, shape (p,)
        Initial covariate coefficients.
    alpha_init : ndarray, shape (K,)
        Initial baseline hazard parameters.
    eta_init : ndarray or None, shape (n,)
        Initial linear predictor.  If None, computed as ``X @ beta_init``.
    use_mm : bool, default False
        Use MM working weights.
    bound : float, default 10.0
        Bounding constraint for baseline hazard.
    backtrack : bool, default False
        Armijo backtracking for baseline hazard updates.
    tol : float, default 1e-4
        Convergence tolerance.
    max_iter : int, default 10000
        Maximum number of outer iterations.
    active_set : bool, default True
        Use active-set screening for beta.

    Returns
    -------
    DiscreteFitResult
    """
    beta = beta_init.copy()
    alpha = alpha_init.copy()
    n, p = X.shape
    event_f = event.astype(np.float64)

    if eta_init is not None:
        eta = eta_init.copy()
    else:
        eta = X @ beta

    # Active set tracking
    if active_set:
        active = penalty_factor == 0  # unpenalized are always active
        # Initialize active set: smallest penalized variable
        pen_idx = np.where(penalty_factor > 0)[0]
        if len(pen_idx) > 0:
            active[pen_idx[0]] = True
    else:
        active = np.ones(p, dtype=bool)

    converged = False
    n_iter = 0
    last_df = 0.0

    for outer_iter in range(max_iter):
        n_iter += 1

        # --- Step 1: Update alpha (baseline hazard) ---
        alpha = baseline_hazard_update(
            alpha, time_int, event_f, eta,
            n_events_per_timepoint, bound=bound, backtrack=backtrack,
        )

        # --- Step 2: Update beta via IRLS + CD ---
        ll_result = discrete_loglik(time_int, event_f, alpha, eta, use_mm=use_mm)
        wr = compute_working_response(ll_result.score_beta, ll_result.working_weights)

        old_beta = beta.copy()

        # Build active X subset
        if active_set:
            active_idx = np.where(active)[0]
            X_active = X[:, active_idx]
            pf_active = penalty_factor[active_idx]
            beta_active = beta[active_idx].copy()

            beta_active, max_change, last_df = discrete_coordinate_descent_step(
                X_active, wr, ll_result.working_weights,
                beta_active, lam, pf_active, tol=tol, max_iter=1000,
            )

            # Write back
            for idx_i, col_i in enumerate(active_idx):
                delta = beta_active[idx_i] - beta[col_i]
                if delta != 0:
                    eta += delta * X[:, col_i]
                beta[col_i] = beta_active[idx_i]
        else:
            beta, max_change, last_df = discrete_coordinate_descent_step(
                X, wr, ll_result.working_weights,
                beta, lam, penalty_factor, tol=tol, max_iter=1000,
            )
            eta = X @ beta  # recompute for numerical stability

        max_change_total = np.max(np.abs(beta - old_beta)) if p > 0 else 0.0

        if max_change_total < tol:
            if active_set:
                # Check inactive variables for KKT violations
                inactive_idx = np.where(~active & (penalty_factor > 0))[0]
                if len(inactive_idx) == 0:
                    converged = True
                    break

                # Screen inactive variables
                ll_result2 = discrete_loglik(time_int, event_f, alpha, eta, use_mm=use_mm)
                wr2 = compute_working_response(ll_result2.score_beta, ll_result2.working_weights)
                n_added = 0
                for j_inactive in inactive_idx:
                    wXj = ll_result2.working_weights * X[:, j_inactive]
                    denom = np.dot(wXj, X[:, j_inactive])
                    if denom == 0:
                        continue
                    z_j = np.dot(wXj, wr2) / denom
                    threshold = n * lam * penalty_factor[j_inactive] / denom
                    if abs(z_j) > threshold:
                        active[j_inactive] = True
                        n_added += 1
                if n_added == 0:
                    converged = True
                    break
            else:
                converged = True
                break

    neg_ll = discrete_loglik(time_int, event_f, alpha, eta, use_mm=use_mm).neg_loglik

    return DiscreteFitResult(
        beta=beta,
        alpha=alpha,
        neg_loglik=neg_ll,
        df=last_df,
        n_iter=n_iter,
        converged=converged,
        eta=eta,
    )


# ---------------------------------------------------------------------------
# Full regularization path
# ---------------------------------------------------------------------------

def fit_discrete_regularization_path(
    X: np.ndarray,
    time_int: np.ndarray,
    event: np.ndarray,
    n_events_per_timepoint: np.ndarray,
    lambda_sequence: np.ndarray,
    penalty_factor: np.ndarray,
    alpha_init: np.ndarray,
    use_mm: bool = False,
    bound: float = 10.0,
    backtrack: bool = False,
    tol: float = 1e-4,
    max_iter: int = 10000,
    active_set: bool = True,
    nvar_max: Optional[int] = None,
) -> list:
    """Fit the full discrete-time survival regularization path.

    Warm-starts both beta and alpha along the decreasing lambda
    sequence.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix.
    time_int : ndarray of int, shape (n,)
        Integer-coded follow-up times.
    event : ndarray, shape (n,)
        Event indicators.
    n_events_per_timepoint : ndarray of int, shape (K,)
    lambda_sequence : ndarray, shape (n_lambda,)
        Decreasing lambda values.
    penalty_factor : ndarray, shape (p,)
    alpha_init : ndarray, shape (K,)
    use_mm : bool
    bound : float
    backtrack : bool
    tol : float
    max_iter : int
    active_set : bool
    nvar_max : int or None
        Maximum number of nonzero variables; stop path early if exceeded.

    Returns
    -------
    results : list of DiscreteFitResult
    """
    n, p = X.shape
    if nvar_max is None:
        nvar_max = p

    beta = np.zeros(p, dtype=np.float64)
    alpha = alpha_init.copy()
    eta = X @ beta

    results = []
    for l_idx, lam in enumerate(lambda_sequence):
        result = fit_single_lambda_discrete(
            X, time_int, event, n_events_per_timepoint,
            lam=lam,
            penalty_factor=penalty_factor,
            beta_init=beta,
            alpha_init=alpha,
            eta_init=eta,
            use_mm=use_mm,
            bound=bound,
            backtrack=backtrack,
            tol=tol,
            max_iter=max_iter,
            active_set=active_set,
        )
        results.append(result)

        # Warm-start next lambda
        beta = result.beta.copy()
        alpha = result.alpha.copy()
        eta = result.eta.copy()

        # Early stop: too many variables
        n_nonzero = int(np.sum(np.abs(beta) > 0))
        if n_nonzero > nvar_max:
            logger.info(
                "Early stop at lambda %d/%d: %d nonzero > nvar_max=%d",
                l_idx + 1, len(lambda_sequence), n_nonzero, nvar_max,
            )
            break

        if not result.converged:
            logger.warning(
                "Lambda %d/%d (%.6g) failed to converge in %d iterations",
                l_idx + 1, len(lambda_sequence), lam, result.n_iter,
            )

    return results


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

@njit(cache=True)
def _predict_discrete_hazard_numba(
    alpha: np.ndarray,
    eta: np.ndarray,
    time_int: np.ndarray,
) -> np.ndarray:
    """Numba kernel for person-period hazard prediction."""
    n = len(time_int)
    N = 0
    for j in range(n):
        N += time_int[j]
    hazard = np.empty(N, dtype=np.float64)
    pos = 0
    for j in range(n):
        Ti = time_int[j]
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            if lin >= 0.0:
                hazard[pos] = 1.0 / (1.0 + np.exp(-lin))
            else:
                exp_lin = np.exp(lin)
                hazard[pos] = exp_lin / (1.0 + exp_lin)
            pos += 1
    return hazard


def _predict_discrete_hazard_python(
    alpha: np.ndarray,
    eta: np.ndarray,
    time_int: np.ndarray,
) -> np.ndarray:
    """Pure-Python fallback."""
    n = len(time_int)
    N = int(np.sum(time_int))
    hazard = np.empty(N, dtype=np.float64)
    pos = 0
    for j in range(n):
        Ti = int(time_int[j])
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            hazard[pos] = 1.0 / (1.0 + np.exp(-np.clip(lin, -500, 500)))
            pos += 1
    return hazard


def predict_discrete_hazard(
    alpha: np.ndarray,
    eta: np.ndarray,
    time_int: np.ndarray,
) -> np.ndarray:
    """Predicted conditional hazard probabilities per subject-timepoint.

    Returns a 1D array of length ``sum(time_int)`` containing
    ``expit(alpha_k + eta_j)`` for each (subject j, timepoint k)
    pair.

    Parameters
    ----------
    alpha : ndarray, shape (K,)
        Baseline hazard parameters.
    eta : ndarray, shape (n,)
        Linear predictor.
    time_int : ndarray of int, shape (n,)

    Returns
    -------
    hazard : ndarray, shape (N_expanded,)
    """
    if _HAS_NUMBA:
        return _predict_discrete_hazard_numba(alpha, eta, time_int)
    return _predict_discrete_hazard_python(alpha, eta, time_int)


@njit(cache=True)
def _predict_survival_probability_numba(
    alpha: np.ndarray,
    eta: np.ndarray,
    time_int: np.ndarray,
) -> np.ndarray:
    """Numba kernel for survival probability prediction."""
    n = len(time_int)
    surv = np.empty(n, dtype=np.float64)
    for j in range(n):
        Ti = time_int[j]
        s = 1.0
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            if lin >= 0.0:
                p_jk = 1.0 / (1.0 + np.exp(-lin))
            else:
                exp_lin = np.exp(lin)
                p_jk = exp_lin / (1.0 + exp_lin)
            s *= (1.0 - p_jk)
        surv[j] = s
    return surv


def _predict_survival_probability_python(
    alpha: np.ndarray,
    eta: np.ndarray,
    time_int: np.ndarray,
) -> np.ndarray:
    """Pure-Python fallback."""
    n = len(time_int)
    surv = np.empty(n, dtype=np.float64)
    for j in range(n):
        Ti = int(time_int[j])
        s = 1.0
        for k in range(Ti):
            lin = alpha[k] + eta[j]
            p_jk = 1.0 / (1.0 + np.exp(-np.clip(lin, -500, 500)))
            s *= (1.0 - p_jk)
        surv[j] = s
    return surv


def predict_survival_probability(
    alpha: np.ndarray,
    eta: np.ndarray,
    time_int: np.ndarray,
) -> np.ndarray:
    """Predicted survival probability S(t_k | Z_i) per subject.

    Returns an array of shape (n,) with the survival probability at
    each subject's last observed timepoint::

        S(T_i) = prod_{k=1}^{T_i} (1 - p_{ik})

    Parameters
    ----------
    alpha : ndarray, shape (K,)
    eta : ndarray, shape (n,)
    time_int : ndarray of int, shape (n,)

    Returns
    -------
    surv : ndarray, shape (n,)
    """
    if _HAS_NUMBA:
        return _predict_survival_probability_numba(alpha, eta, time_int)
    return _predict_survival_probability_python(alpha, eta, time_int)
