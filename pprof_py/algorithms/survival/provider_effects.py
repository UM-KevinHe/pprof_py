"""Provider-effect update kernels for the two-layer architecture.

This module implements the outer-layer provider-effect (γ) updates
used by ``ProviderPenalizedCoxPH``.  Provider effects are *not*
penalized; they are updated via one-step Newton, with optional
bounding and backtracking line search.

The inner-layer covariate (β) updates are handled by the existing
``coordinate_descent.py`` solvers (element-wise for elastic net,
block for group / sparse group lasso).

Model
-----
The linear predictor for observation j is::

    η_j = γ_{provider(j)} + X_j @ β + offset_j

with a **shared** baseline hazard across all providers (optionally
stratified by an external variable, but NOT by provider).  This
makes γ_i identifiable from the Cox partial likelihood: it captures
the provider-specific log-hazard ratio relative to the common
baseline, after risk-adjusting for covariates.

Score for γ_i (Breslow, right-censored or start-stop)::

    ∂ℓ/∂γ_i = observed_i − expected_i

where::

    observed_i = Σ_{k: event_k=1, provider(k)=i} w_k
    expected_i = Σ_{k: provider(k)=i} w_k exp(η_k) Ĥ_k

and Ĥ_k is the Breslow cumulative hazard exposure for observation k
(H(stop_k) for right-censored, H(stop_k)−H(start_k) for left-
truncated data).  This is the weighted sum of per-observation
martingale residuals within provider i.

The MM information bound (diagonal majorization, v=1) is::

    info_mm_i = expected_i

Architecture
------------
Sits alongside ``coordinate_descent.py`` and ``optimization.py`` at
the algorithm / numerical-kernel layer.  Consumes the same
``objective(beta) -> (log_likelihood, score, information)`` callable
built from ``cox_partial_likelihood``, plus per-stratum index arrays
from ``precompute_stratum_indices``.

References
----------
He, K., Kalbfleisch, J., Li, Y., et al. (2013). Evaluating hospital
readmission rates in dialysis facilities; adjusting for hospital
effects. *Lifetime Data Analysis*, 19, 490-512.

Shao, Y. & He, K. (2026). grplasso R package.
"""
from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Tuple

import numpy as np

from .risk_sets import njit, _HAS_NUMBA

logger = logging.getLogger(__name__)


# ======================================================================
# Result container
# ======================================================================

class ProviderUpdateResult(NamedTuple):
    """Result of a single round of provider-effect updates.

    Attributes
    ----------
    gamma : ndarray, shape (n_providers,)
        Updated provider-effect estimates.
    max_change : float
        Largest absolute change in any γ_i this round.
    converged : bool
        True if ``max_change <= tol``.
    """
    gamma: np.ndarray
    max_change: float
    converged: bool


# ======================================================================
# Provider-ID utilities
# ======================================================================

def validate_provider_ids(
    provider_ids: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, int]:
    """Canonicalize provider labels to contiguous 0 .. M-1.

    Parameters
    ----------
    provider_ids : array-like, shape (n,)
        Raw provider labels (integers or strings).
    n : int
        Expected number of observations.

    Returns
    -------
    provider_idx : ndarray of int, shape (n,)
        Integer labels in 0 .. n_providers-1.
    n_providers : int
        Number of distinct providers.
    """
    provider_ids = np.asarray(provider_ids)
    if provider_ids.shape != (n,):
        raise ValueError(
            f"provider_ids must have shape ({n},), "
            f"got {provider_ids.shape}"
        )
    unique_labels, provider_idx = np.unique(
        provider_ids, return_inverse=True,
    )
    n_providers = len(unique_labels)
    if n_providers < 2:
        raise ValueError(
            "At least 2 providers are required for provider-effect "
            f"modeling; got {n_providers}"
        )
    return provider_idx.astype(np.intp), int(n_providers)


def precompute_provider_indices(
    provider_idx: np.ndarray,
    n_providers: int,
) -> list:
    """Per-provider observation index arrays.

    Returns a list of length ``n_providers``, where element i is a
    1-D int64 array of the row indices assigned to provider i.
    """
    indices = [[] for _ in range(n_providers)]  # type: ignore[var-annotated]
    for k, p in enumerate(provider_idx):
        indices[p].append(k)
    return [np.array(lst, dtype=np.int64) for lst in indices]


# ======================================================================
# Bounding
# ======================================================================

def provider_bound_clamp(
    gamma: np.ndarray,
    bound: float,
) -> np.ndarray:
    """Clamp provider effects to ``median(γ) ± bound``.

    Matches ``grplasso``'s ``clamp(gamma, median(gamma) - bound,
    median(gamma) + bound)`` which prevents extreme provider-effect
    inflation in small-sample strata.
    """
    med = float(np.median(gamma))
    return np.clip(gamma, med - bound, med + bound)


# ======================================================================
# Breslow per-provider score kernels
# ======================================================================

@njit(cache=True)
def _breslow_provider_scores_stratum_numba(
    stop_desc: np.ndarray,
    event_desc: np.ndarray,
    weight_desc: np.ndarray,
    eta_desc: np.ndarray,
    provider_desc: np.ndarray,
    start_desc: np.ndarray,
    n_providers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba kernel: per-provider score and MM info within one stratum.

    All arrays are pre-sorted by **descending** stop time.
    Handles right-censored and start-stop (left-truncated) data.

    Algorithm
    ---------
    1. Descending pass: accumulate S0 at each unique event time,
       handling Breslow ties (all obs at same stop enter risk set
       before processing events).
    2. Ascending pass over unique event times: build forward
       cumulative hazard H(t).
    3. For each observation k, compute H-exposure = H(stop_k) −
       H(start_k) and accumulate per-provider observed / expected.

    Returns ``(score, info_mm)`` each of shape ``(n_providers,)``.
    """
    n = len(stop_desc)

    # --- Phase 1: identify unique event times and compute S0 ---
    # Worst case: every observation is an event at a unique time.
    ev_time = np.empty(n)
    ev_d = np.empty(n)
    ev_S0 = np.empty(n)
    n_ev_times = 0

    S0 = 0.0
    k = 0
    while k < n:
        t = stop_desc[k]
        # Batch all observations with the same stop time.
        j = k
        while j < n and stop_desc[j] == t:
            S0 += weight_desc[j] * np.exp(eta_desc[j])
            j += 1
        # Total weighted events at this time.
        d = 0.0
        for m in range(k, j):
            if event_desc[m] > 0.5:
                d += weight_desc[m]
        if d > 0.0:
            ev_time[n_ev_times] = t
            ev_d[n_ev_times] = d
            ev_S0[n_ev_times] = S0
            n_ev_times += 1
        k = j

    # --- Phase 2: cumulative hazard at each event time ---
    # ev_time[0] = latest, ev_time[n_ev_times-1] = earliest.
    # H(t) = sum of d_j/S0(t_j) for t_j <= t.
    # In descending order: H at index i = sum from i to end.
    cum_haz = np.empty(n_ev_times)
    running = 0.0
    for i in range(n_ev_times - 1, -1, -1):
        running += ev_d[i] / ev_S0[i]
        cum_haz[i] = running
    # cum_haz[i] = H(ev_time[i]).

    # --- Phase 3: per-observation H-exposure and per-provider sums ---
    observed = np.zeros(n_providers)
    expected = np.zeros(n_providers)

    # For H(stop_k): scan ev_time (descending) with a pointer.
    # For H(start_k): similar, but we need a separate lookup.
    # Since observations are already sorted by descending stop,
    # the stop-pointer only advances forward.

    ev_ptr = 0  # pointer into ev_time for stop lookup
    for k in range(n):
        # Advance pointer while ev_time[ev_ptr] > stop_desc[k].
        while ev_ptr < n_ev_times and ev_time[ev_ptr] > stop_desc[k]:
            ev_ptr += 1
        H_stop = cum_haz[ev_ptr] if ev_ptr < n_ev_times else 0.0

        # H(start_k): need to find the first event time <= start_k.
        # start_k could be 0 (right-censored), in which case H = 0.
        H_start = 0.0
        if start_desc[k] > 0.0:
            # Binary search in ev_time (descending) for start_k.
            lo, hi = 0, n_ev_times
            while lo < hi:
                mid = (lo + hi) // 2
                if ev_time[mid] > start_desc[k]:
                    lo = mid + 1
                else:
                    hi = mid
            # lo = first index where ev_time[lo] <= start_k.
            H_start = cum_haz[lo] if lo < n_ev_times else 0.0

        H_exposure = H_stop - H_start
        if H_exposure < 0.0:
            H_exposure = 0.0  # numerical safety

        p = provider_desc[k]
        if event_desc[k] > 0.5:
            observed[p] += weight_desc[k]
        expected[p] += weight_desc[k] * np.exp(eta_desc[k]) * H_exposure

    score = observed - expected
    info_mm = expected.copy()

    return score, info_mm


def _breslow_provider_scores_stratum_python(
    stop_desc: np.ndarray,
    event_desc: np.ndarray,
    weight_desc: np.ndarray,
    eta_desc: np.ndarray,
    provider_desc: np.ndarray,
    start_desc: np.ndarray,
    n_providers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-Python fallback for per-provider scores (one stratum)."""
    # Identical algorithm, just not @njit-decorated.
    return _breslow_provider_scores_stratum_numba(
        stop_desc, event_desc, weight_desc, eta_desc,
        provider_desc, start_desc, n_providers,
    )


_breslow_scores_impl = (
    _breslow_provider_scores_stratum_numba
    if _HAS_NUMBA
    else _breslow_provider_scores_stratum_python
)


# ======================================================================
# Public dispatcher
# ======================================================================

def compute_provider_scores(
    eta: np.ndarray,
    event: np.ndarray,
    weight: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    provider_idx: np.ndarray,
    n_providers: int,
    strata_codes: Optional[np.ndarray] = None,
    stratum_indices: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-provider Cox score and MM information.

    The score is :math:`\\partial \\ell / \\partial \\gamma_i` (observed
    minus expected events for provider i).  The MM information is
    :math:`\\text{expected}_i`, a valid upper bound on the diagonal
    of :math:`-\\partial^2 \\ell / \\partial \\gamma_i^2`.

    Parameters
    ----------
    eta : ndarray, shape (n,)
        Current linear predictor (γ[provider] + X @ β + offset).
    event, weight, start, stop : ndarray, shape (n,)
        Survival data (right-censored: all start = 0).
    provider_idx : ndarray of int, shape (n,)
        Provider assignment per observation (0 .. n_providers-1).
    n_providers : int
    strata_codes : ndarray of int or None
        External stratum labels (NOT provider).  ``None`` = single
        stratum.
    stratum_indices : list of ndarray or None
        Precomputed per-stratum index arrays.

    Returns
    -------
    score : ndarray, shape (n_providers,)
        :math:`\\partial \\ell / \\partial \\gamma_i`.
    info_mm : ndarray, shape (n_providers,)
        MM diagonal information (= expected_i).
    """
    n = len(eta)
    score = np.zeros(n_providers)
    info_mm = np.zeros(n_providers)

    # Resolve strata.
    if stratum_indices is None:
        if strata_codes is None:
            stratum_indices = [np.arange(n, dtype=np.int64)]
        else:
            from .cox_likelihood import precompute_stratum_indices
            stratum_indices = precompute_stratum_indices(strata_codes)

    for idx in stratum_indices:
        # Sort within stratum by descending stop time.
        sub_stop = stop[idx]
        order = np.argsort(-sub_stop)
        sorted_idx = idx[order]

        s_score, s_info = _breslow_scores_impl(
            stop[sorted_idx],
            event[sorted_idx],
            weight[sorted_idx],
            eta[sorted_idx],
            provider_idx[sorted_idx],
            start[sorted_idx],
            n_providers,
        )
        score += s_score
        info_mm += s_info

    return score, info_mm


# ======================================================================
# Newton step
# ======================================================================

def provider_newton_step(
    gamma: np.ndarray,
    score: np.ndarray,
    info_mm: np.ndarray,
    bound: float = 10.0,
    tol: float = 1e-8,
    min_info: float = 1e-12,
) -> ProviderUpdateResult:
    """One-step Newton update for all provider effects.

    Applies::

        γ_i += score_i / max(info_i, min_info)
        γ   = clamp(γ, median(γ) ± bound)

    Parameters
    ----------
    gamma : ndarray, shape (n_providers,)
        Current provider effects (modified in-place).
    score : ndarray, shape (n_providers,)
        Per-provider score :math:`\\partial \\ell / \\partial \\gamma_i`.
    info_mm : ndarray, shape (n_providers,)
        MM diagonal information.
    bound : float
        Clamp to ``median(γ) ± bound``.
    tol : float
        Convergence tolerance on max |Δγ_i|.
    min_info : float
        Floor for information to prevent division by zero
        (providers with no risk-set exposure).

    Returns
    -------
    ProviderUpdateResult
    """
    gamma_old = gamma.copy()
    step = score / np.maximum(info_mm, min_info)
    gamma += step
    gamma[:] = provider_bound_clamp(gamma, bound)
    max_change = float(np.max(np.abs(gamma - gamma_old)))
    return ProviderUpdateResult(
        gamma=gamma,
        max_change=max_change,
        converged=max_change <= tol,
    )
