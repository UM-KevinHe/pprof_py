"""Shared inference utilities for provider profiling.

This module provides the parametric bootstrap and empirical null calibration
algorithm used by all three model classes (FE, RE, Mixed).

Empirical null (Huber M-estimator):
- huber_location_scale(): Standalone Huber M-estimator matching R's MASS::rlm
- estimate_empirical_null(): Groupwise or global EN estimation wrapper

Resampling inference (He et al. 2013):
- resample_pvalue(): Two-sided p-value for a single provider via bootstrap
- poibin_exact_pvalue(): Exact two-sided p-value via Poisson-binomial DP
- pvalues_to_zscores(): Convert p-values to signed z-scores
- calibrate_empirical_null(): Stratified EN wrapper returning calibrated p-values
- assign_flags(): Directional flags from p-values

Usage:
    from pprof_py.inference.empirical_null import huber_location_scale, estimate_empirical_null
    from pprof_py.inference.empirical_null import resample_pvalue, poibin_exact_pvalue
    from pprof_py.inference.empirical_null import pvalues_to_zscores
    from pprof_py.inference.empirical_null import calibrate_empirical_null, assign_flags
"""

import numpy as np
from scipy.special import expit as plogis
from scipy.stats import norm
from typing import Dict, Optional, Tuple


def huber_location_scale(
    z: np.ndarray,
    k: float = 1.345,
    maxiter: int = 20,
    tol: float = 1e-4,
) -> Tuple[float, float]:
    """Huber M-estimator for location and scale (intercept-only model).

    Replicates R's MASS::rlm(z ~ 1, method='M', psi=psi.huber, scale.est='MAD')
    exactly, including:
    - Scale recomputed (MAD) at every IRLS iteration
    - Convergence via residual vector norm (test.vec='resid')
    - Literal 0.6745 constant (matching R's MASS)

    Parameters
    ----------
    z : np.ndarray
        Array of z-scores (NaN values should be removed before calling).
    k : float, default=1.345
        Huber tuning constant. Observations with |u| > k are downweighted.
    maxiter : int, default=20
        Maximum IRLS iterations (R's rlm default: maxit=20).
    tol : float, default=1e-4
        Convergence tolerance (R's rlm default: acc=1e-4).

    Returns
    -------
    Tuple[float, float]
        (location, scale) — the estimated center and dispersion of the
        empirical null distribution.

    Notes
    -----
    The scale returned is from the FINAL iteration (not the initial MAD).
    R's rlm stores this as ``fit$s``.

    The MAD formula used is::

        s = median(|resid|) / 0.6745

    where resid = z - mu (centered on the current location estimate, NOT
    on the median of residuals). This differs from Python's standard MAD
    which centers on the median.
    """
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if n == 0:
        return 0.0, 1.0

    # Initial LS estimate (intercept-only = mean)
    mu = np.mean(z)
    resid = z - mu

    # Initial scale (will be updated each iteration)
    s = np.median(np.abs(resid)) / 0.6745

    for _ in range(maxiter):
        # Recompute MAD scale at every iteration (R's scale.est="MAD" behavior)
        s = np.median(np.abs(resid)) / 0.6745
        if s == 0.0:
            break

        # Huber weights: psi(u)/u = min(1, k/|u|)
        u = resid / s
        abs_u = np.abs(u)
        w = np.where(abs_u <= k, 1.0, k / abs_u)

        # Weighted LS for intercept: mu = sum(w*z) / sum(w)
        mu_new = np.sum(w * z) / np.sum(w)
        resid_new = z - mu_new

        # Convergence: R's test.vec='resid' — residual vector norm
        delta = (
            np.sqrt(np.sum((resid - resid_new) ** 2))
            / max(1e-20, np.sqrt(np.sum(resid ** 2)))
        )

        mu = mu_new
        resid = resid_new

        if delta <= tol:
            break

    return mu, s


def estimate_empirical_null(
    z_scores: np.ndarray,
    group_sizes: Optional[np.ndarray] = None,
    group_labels: Optional[np.ndarray] = None,
    n_groups: int = 4,
    outlier_mask: Optional[np.ndarray] = None,
    k: float = 1.345,
    maxiter: int = 20,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate empirical null parameters, optionally per size-based group.

    Wraps huber_location_scale() with groupwise stratification logic
    matching R's SFR_Test quartile approach.

    Parameters
    ----------
    z_scores : np.ndarray
        Z-scores for each provider. May contain NaN (excluded from fitting
        but included in group sizing, matching R's na.action=na.omit).
    group_sizes : np.ndarray, optional
        Provider sizes (e.g., sfrpats_f) for sorting into groups.
        Required if group_labels is not provided and n_groups > 1.
    group_labels : np.ndarray, optional
        Pre-computed integer group labels (0-indexed). If provided,
        group_sizes and n_groups are ignored.
    n_groups : int, default=4
        Number of size-based groups. Providers are sorted by group_sizes
        and split into approximately equal groups.
    outlier_mask : np.ndarray of bool, optional
        If provided, True entries are excluded from the Huber fit
        (but still occupy group positions). Use for extreme providers.
    k : float, default=1.345
        Huber tuning constant.
    maxiter : int, default=20
        Maximum IRLS iterations.
    tol : float, default=1e-4
        Convergence tolerance.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (location, scale) arrays of shape (n_providers,). Each provider
        receives the location/scale of its group.

    Examples
    --------
    Global EN (single fit for all providers)::

        loc, scl = estimate_empirical_null(z_scores, n_groups=1)

    Groupwise EN (SFR pattern)::

        loc, scl = estimate_empirical_null(
            z_scores, group_sizes=sfrpats_f, n_groups=4
        )

    With pre-assigned groups::

        loc, scl = estimate_empirical_null(
            z_scores, group_labels=en_group_array
        )
    """
    z_scores = np.asarray(z_scores, dtype=np.float64)
    n = len(z_scores)
    location = np.zeros(n)
    scale = np.ones(n)

    # Prepare working z-scores (mask outliers as NaN for fitting)
    z_work = z_scores.copy()
    if outlier_mask is not None:
        z_work[outlier_mask] = np.nan

    # Determine group labels
    if group_labels is not None:
        labels = np.asarray(group_labels, dtype=int)
        unique_groups = np.unique(labels[~np.isnan(labels.astype(float))])
    elif n_groups <= 1 or group_sizes is None:
        # Global: single group for all providers
        labels = np.zeros(n, dtype=int)
        unique_groups = [0]
    else:
        # Sort by group_sizes, assign groups (R's quartile approach)
        # NaN group_sizes sort to end (matching R's order(SFRPATS_F))
        group_sizes = np.asarray(group_sizes, dtype=np.float64)
        sort_idx = np.argsort(group_sizes, kind="mergesort")
        split_size = int(np.ceil(n / n_groups))
        labels = np.zeros(n, dtype=int)
        for g in range(n_groups):
            start = g * split_size
            end = min((g + 1) * split_size, n)
            labels[sort_idx[start:end]] = g
        unique_groups = range(n_groups)

    # Fit Huber per group
    for g in unique_groups:
        mask_g = labels == g
        z_group = z_work[mask_g]
        valid = ~np.isnan(z_group)

        if np.sum(valid) < 3:
            # Too few valid observations — fall back to defaults
            location[mask_g] = 0.0
            scale[mask_g] = 1.0
            continue

        z_valid = z_group[valid]
        loc, scl = huber_location_scale(z_valid, k=k, maxiter=maxiter, tol=tol)
        location[mask_g] = loc
        scale[mask_g] = scl

    return location, scale


# ======================================================================
# Resampling inference (He et al. 2013 parametric bootstrap)
# ======================================================================


def resample_pvalue(
    obs_sum: float,
    eta_fixed: np.ndarray,
    re_mean: np.ndarray,
    re_var: np.ndarray,
    null_effect: float,
    n_resample: int = 10000,
    seed: int = 1,
) -> float:
    """Compute two-sided p-value for a single provider via parametric bootstrap.

    Implements He et al. (2013) steps (ii)-(iv):
      (ii)   Sample random effects from posterior: N(re_mean, re_var)
      (iii)  Generate Y ~ Bernoulli(expit(null_effect + re_sample + eta_fixed))
      (iv)   Compare observed sum(Y) to simulated sums

    Parameters
    ----------
    obs_sum : float
        Observed sum of Y for this provider.
    eta_fixed : np.ndarray, shape (n_obs,)
        Per-observation fixed linear predictor (offset + xbeta),
        EXCLUDING the target group effect and the sampled RE.
    re_mean : np.ndarray, shape (n_obs,)
        Per-observation posterior mean of the random effect to sample
        (e.g., hospital alpha_mean for each observation).
    re_var : np.ndarray, shape (n_obs,)
        Per-observation posterior variance of that random effect.
    null_effect : float
        The null value for the target group (e.g., median gamma).
    n_resample : int, default=10000
        Number of Monte Carlo resamples.
    seed : int, default=1
        Random seed for reproducibility.

    Returns
    -------
    float
        Two-sided p-value (matching R convention: average of >= and >).
    """
    rng = np.random.default_rng(seed)
    n_obs = len(eta_fixed)

    # Vectorized simulation: tile obs-level quantities across resamples
    re_samples = rng.normal(
        loc=np.tile(re_mean, n_resample),
        scale=np.tile(np.sqrt(np.maximum(re_var, 0)), n_resample),
    )
    eta_fixed_tiled = np.tile(eta_fixed, n_resample)

    # Linear predictor under null
    eta_sim = null_effect + re_samples + eta_fixed_tiled
    probs = plogis(eta_sim)

    # Simulate Y and compute sum per resample
    Y_sim = rng.binomial(1, probs).reshape(n_resample, n_obs)
    sim_sums = Y_sim.sum(axis=1)

    # Two-sided p-value (matching R convention)
    p_upper = np.mean(sim_sums >= obs_sum) + np.mean(sim_sums > obs_sum)
    p_lower = np.mean(sim_sums <= obs_sum) + np.mean(sim_sums < obs_sum)
    return min(p_upper, p_lower)


def poibin_exact_pvalue(
    obs_sum: float,
    probs: np.ndarray,
) -> float:
    """Exact two-sided p-value via Poisson-binomial distribution.

    Drop-in replacement for ``resample_pvalue`` when null probabilities
    are deterministic (no posterior sampling needed).  Uses the
    ``fast_poibin`` DP algorithm for O(n²) exact computation.

    The p-value convention matches ``resample_pvalue`` (He et al. 2013):
        min( P(X >= k) + P(X > k),  P(X <= k) + P(X < k) )
    so results are directly comparable.

    Parameters
    ----------
    obs_sum : float
        Observed sum of Y for this provider.
    probs : np.ndarray, shape (n_obs,)
        Null probability for each observation (deterministic).

    Returns
    -------
    float
        Two-sided p-value.
    """
    from fast_poibin import PoiBin

    probs = np.clip(np.asarray(probs, dtype=float), 1e-10, 1 - 1e-10)
    obs = int(round(obs_sum))
    pb = PoiBin(probs)

    cdf_k = pb.cdf[obs]
    cdf_km1 = pb.cdf[obs - 1] if obs > 0 else 0.0

    p_upper = (1 - cdf_km1) + (1 - cdf_k)   # P(X>=k) + P(X>k)
    p_lower = cdf_k + cdf_km1               # P(X<=k) + P(X<k)

    return min(p_upper, p_lower)


def pvalues_to_zscores(
    p_values: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Convert p-values to signed z-scores.

    Parameters
    ----------
    p_values : np.ndarray
        Two-sided p-values per provider.
    direction : np.ndarray
        Directional indicator (e.g., SRR). >= 1 gets positive z.

    Returns
    -------
    np.ndarray
        Signed z-scores.
    """
    p_min = 1e-2 * np.min(p_values[p_values > 0]) if np.any(p_values > 0) else 1e-10
    p_safe = np.where(p_values > 0, p_values, p_min)
    z_abs = norm.ppf(1 - p_safe / 2)
    return np.where(direction >= 1, z_abs, -z_abs)


def calibrate_empirical_null(
    z_scores: np.ndarray,
    strata_var: Optional[np.ndarray] = None,
    n_strata: int = 4,
) -> Tuple[np.ndarray, Dict]:
    """Apply empirical null calibration, returning calibrated p-values.

    Wraps estimate_empirical_null() and converts location/scale to p-values.

    Parameters
    ----------
    z_scores : np.ndarray
        Signed z-scores per provider.
    strata_var : np.ndarray, optional
        Provider-level variable for stratification (e.g., facility size).
    n_strata : int, default=4
        Number of size-based strata.

    Returns
    -------
    Tuple[np.ndarray, Dict]
        (calibrated_pvalues, params_dict)
    """
    location, scale = estimate_empirical_null(
        z_scores,
        group_sizes=strata_var,
        n_groups=n_strata if strata_var is not None else 1,
    )

    p_upper = norm.sf(z_scores, loc=location, scale=scale)
    p_calibrated = 2.0 * np.minimum(p_upper, 1.0 - p_upper)

    # Build params dict
    params: Dict = {}
    unique_locs = np.unique(np.column_stack([location, scale]), axis=0)
    for i, (loc, scl) in enumerate(unique_locs):
        params[i] = {'mean': float(loc), 'sd': float(scl)}

    return p_calibrated, params


def assign_flags(
    p_values: np.ndarray,
    direction: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Assign directional flags from p-values.

    Parameters
    ----------
    p_values : np.ndarray
        P-values per provider.
    direction : np.ndarray
        Directional indicator (e.g., SRR). >= 1 means worse.
    alpha : float, default=0.05
        Significance threshold.

    Returns
    -------
    np.ndarray of int
        -1 = significantly worse, 0 = as expected, 1 = significantly better
    """
    flags = np.zeros(len(p_values), dtype=int)
    sig = p_values < alpha
    flags[sig & (direction >= 1)] = -1
    flags[sig & (direction < 1)] = 1
    return flags
