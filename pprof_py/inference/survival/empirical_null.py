"""Empirical-null calibration for provider-level standardized measures.

This module provides three tightly coupled post-model inference utilities:

1. Mid-p Poisson Z-score construction.
2. Empirical-null fitting and adjustment (global or quantile-stratified).
3. Confidence-interval helpers — root-finding for count-based measures
   and closed-form log-normal intervals for ratio-based measures.

Public API
----------
* `poisson_midp_zscore(obs, exp)`
* `log_ratio_zscore(ratio, stderr)`
* `fit_empirical_null(...)`
* `fit_grouped_empirical_null(...)`
* `adjust_empirical_null(...)`
* `poisson_confidence_bounds(...)`
* `log_ratio_confidence_intervals(...)`

Empirical-null fitting uses the shared null layer
(:mod:`pprof_py.inference.empirical_null`): one robust estimator
(``MASS::rlm``-exact, Huber or bisquare, M or MM) and one quantile-grouping
rule, run here with R's settings for these functions (bisquare, least-squares
start, ``maxit = 1000``, ``acc = 1e-8``).
"""
from __future__ import annotations
import warnings

from ..empirical_null import EmpiricalNull, EmpiricalNullWarning, MEstimator, assign_groups

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, poisson


ArrayLike = Union[np.ndarray, list, tuple]

_DEFAULT_ALPHA = 0.05
_DEFAULT_UPPER_CAP = 10000.0

__all__ = [
    "poisson_midp_zscore",
    "log_ratio_zscore",
    "fit_empirical_null",
    "fit_grouped_empirical_null",
    "adjust_empirical_null",
    "poisson_confidence_bounds",
    "log_ratio_confidence_intervals",
]


def _as_float_array(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr



def _estimator(psi: str, method: str, tuning: Optional[float], maxiter: int, tol: float) -> MEstimator:
    # R's rlm(z ~ 1, psi, method, maxit, acc) with its default least-squares start
    return MEstimator(psi=psi, tuning=tuning, init="mean", maxiter=maxiter, tol=tol, method=method)


def _quantile_labels(size: np.ndarray, n_groups: int) -> np.ndarray:
    """Quantile groups of the finite sizes; missing sizes get no group (as R's cut() gives NA)."""
    labels = np.full(size.shape[0], np.nan)
    finite = np.isfinite(size)
    labels[finite] = np.asarray(assign_groups(size[finite], n_groups, rule="quantile"), dtype=np.float64)
    return labels


def fit_empirical_null(
    z: ArrayLike,
    psi: str = "bisquare",
    method: str = "M",
    tuning: Optional[float] = None,
    maxiter: int = 1000,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """All-provider empirical null: intercept and scale of ``rlm(z ~ 1)`` (R ``empirical_null_overall``).

    Uses the shared estimator (:func:`pprof_py.inference.robust_location_scale`)
    with R's defaults here: bisquare, least-squares start, ``maxit = 1000``,
    ``acc = 1e-8``. ``method="MM"`` is available.
    """
    res = _estimator(psi, method, tuning, maxiter, tol)(_as_float_array(z, "z"))
    return {"intercept": res.location, "scale": res.scale}


def fit_grouped_empirical_null(
    z: ArrayLike,
    size: ArrayLike,
    n_groups: int = 4,
    psi: str = "bisquare",
    method: str = "M",
    tuning: Optional[float] = None,
    maxiter: int = 1000,
    tol: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Quantile-stratified empirical null (R ``empirical_null_groupwise``).

    Providers are grouped by quantiles of ``size`` (a size equal to a break goes
    to the lower group; missing sizes get no group) and ``rlm(z ~ 1)`` is fitted
    in each group, which needs at least two finite z-values.

    Returns
    -------
    dict
        ``intercept`` and ``scale`` per group (groups 1..n_groups) and ``group``
        per provider (NaN where the size is missing).
    """
    z_arr = _as_float_array(z, "z")
    size_arr = _as_float_array(size, "size")
    if z_arr.shape[0] != size_arr.shape[0]:
        raise ValueError("z and size must have the same length.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2; for a single group use fit_empirical_null().")
    labels = _quantile_labels(size_arr, n_groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EmpiricalNullWarning)   # missing sizes are expected here, as in R
        null = EmpiricalNull.fit(z_arr, groups=labels, estimator=_estimator(psi, method, tuning, maxiter, tol),
                                 min_group_size=2, small_group="error")
    d = null.diagnostics.set_index("group").reindex(np.arange(1.0, n_groups + 1))
    return {"intercept": d.null_mean.to_numpy(), "scale": d.null_sd.to_numpy(), "group": labels}


def adjust_empirical_null(
    z: ArrayLike,
    size: Optional[ArrayLike] = None,
    n_groups: int = 4,
    group_labels: Optional[ArrayLike] = None,
    common_mean: Union[bool, float] = False,
    psi: str = "bisquare",
    method: str = "M",
    tuning: Optional[float] = None,
    maxiter: int = 1000,
    tol: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Estimate an empirical null, adjust the z-statistics, and return two-sided p-values
    (R ``empirical_null_adjust``).

    Parameters
    ----------
    size, n_groups, group_labels
        Explicit ``group_labels``, or quantile groups of ``size`` (``n_groups > 1``),
        or one overall group (``n_groups == 1``).
    common_mean : bool or float
        ``False``: each group's fitted intercept. ``True``: the mean of all finite
        z-values. A number: that value. Group scales are kept as fitted.

    Returns
    -------
    dict
        ``z_adj``, ``p_value``, per-provider ``intercept`` and ``scale``, ``group``,
        and ``params`` (one row per group).
    """
    z_arr = _as_float_array(z, "z")
    if n_groups < 1:
        raise ValueError("n_groups must be a positive integer.")
    if not isinstance(common_mean, (bool, int, float, np.bool_, np.integer, np.floating)):
        raise ValueError("common_mean must be bool or numeric.")
    if group_labels is not None:
        labels = _as_float_array(group_labels, "group_labels")
        if labels.shape[0] != z_arr.shape[0]:
            raise ValueError("group_labels and z must have the same length.")
    elif n_groups == 1:
        labels = np.ones(z_arr.shape[0])
    else:
        if size is None:
            raise ValueError("size must be supplied when n_groups > 1.")
        labels = _quantile_labels(_as_float_array(size, "size"), n_groups)
    cm = None if common_mean is False else (True if common_mean is True else float(common_mean))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EmpiricalNullWarning)
        null = EmpiricalNull.fit(z_arr, groups=labels, estimator=_estimator(psi, method, tuning, maxiter, tol),
                                 min_group_size=2, small_group="error", common_mean=cm)
    z_adj = (z_arr - null.mean) / null.sd
    params = null.diagnostics[["group", "null_mean", "null_sd"]].rename(
        columns={"null_mean": "intercept", "null_sd": "scale"})
    return {"z_adj": z_adj, "p_value": 2.0 * norm.sf(np.abs(z_adj)), "intercept": null.mean,
            "scale": null.sd, "group": labels, "params": params}

def poisson_midp_zscore(obs: ArrayLike, exp: ArrayLike) -> np.ndarray:
    """Convert observed and expected Poisson counts to mid-p Z-scores.

    Computes:

        P_min = 2 * cdf(obs, exp) - pmf(obs, exp)
        P_max = 2 * (1 - cdf(obs - 1, exp)) - pmf(obs, exp)
        p     = max(1e-6, min(P_min, P_max) / 2)

    The sign is chosen by which tail is smaller.
    """
    obs_arr, exp_arr = np.broadcast_arrays(
        _as_float_array(obs, "obs"),
        _as_float_array(exp, "exp"),
    )
    p_min = 2.0 * poisson.cdf(obs_arr, exp_arr) - poisson.pmf(obs_arr, exp_arr)
    p_max = 2.0 * (1.0 - poisson.cdf(obs_arr - 1.0, exp_arr)) - poisson.pmf(obs_arr, exp_arr)
    pval_temp = np.maximum(1e-6, np.minimum(p_min, p_max) / 2.0)
    z_raw = norm.ppf(pval_temp)
    return np.where(p_min <= p_max, z_raw, -z_raw)


def log_ratio_zscore(
    ratio: ArrayLike,
    stderr: ArrayLike,
    zero_method: str = "score",
) -> np.ndarray:
    """Construct Z-scores from a standardized ratio and its standard error.

    For positive ratios this is simply ``log(ratio) / stderr``.  When
    ``ratio == 0`` the log-Wald statistic is undefined (``-inf / inf``).
    The ``zero_method`` parameter controls the fallback:

    * ``"score"`` (default): use ``-1 / stderr``, the score-function
      approximation evaluated at the null.  This keeps zero-event
      facilities within the empirical-null framework rather than
      switching to a different inference method.
    * ``"exclude"``: return ``NaN`` for zero-ratio providers, leaving
      the decision to the caller.

    Parameters
    ----------
    ratio : array-like
        Provider ratio estimates (e.g. SHR, STrR).  Must be >= 0.
    stderr : array-like
        Standard errors.  Must be finite and positive for every provider
        (the caller is responsible for supplying a valid stderr even when
        events are zero — see Notes).
    zero_method : {"score", "exclude"}, default "score"
        Strategy for zero-ratio providers.

    Returns
    -------
    np.ndarray
        Z-scores, one per provider.

    Notes
    -----
    The caller must ensure ``stderr`` is finite for zero-event facilities.
    Some dispersion-based stderr formulas (e.g. ``sqrt(phi / Yi)``) diverge
    when ``Yi = 0``; these must be corrected in the measure code before
    calling this function.
    """
    ratio_arr, stderr_arr = np.broadcast_arrays(
        _as_float_array(ratio, "ratio"),
        _as_float_array(stderr, "stderr"),
    )
    if np.any(ratio_arr < 0.0):
        raise ValueError("ratio must be non-negative.")
    if np.any(~np.isfinite(stderr_arr)) or np.any(stderr_arr <= 0.0):
        raise ValueError(
            "stderr must be finite and positive for every provider. "
            "If events are zero, the measure code must supply a corrected "
            "stderr before calling this function."
        )

    zero_method = zero_method.lower()
    if zero_method not in ("score", "exclude"):
        raise ValueError("zero_method must be 'score' or 'exclude'.")

    positive = ratio_arr > 0.0
    z = np.empty(ratio_arr.shape, dtype=np.float64)
    z[positive] = np.log(ratio_arr[positive]) / stderr_arr[positive]

    zero = ~positive
    if zero_method == "score":
        z[zero] = -1.0 / stderr_arr[zero]
    else:
        z[zero] = np.nan

    return z


def _calibrated_poisson_pvalue(
    expected_count: float,
    obs_count: float,
    intercept: float,
    scale: float,
) -> float:
    z_score = float(poisson_midp_zscore([obs_count], [expected_count])[0])
    return float(2.0 * (1.0 - norm.cdf(abs(z_score - intercept) / scale)))


def _solve_poisson_bound(
    obs_count: float,
    intercept: float,
    scale: float,
    lower_bound: float,
    upper_bound: float,
    alpha: float,
) -> float:
    def objective(expected_count: float) -> float:
        return _calibrated_poisson_pvalue(expected_count, obs_count, intercept, scale) - alpha

    f_lower = objective(lower_bound)
    f_upper = objective(upper_bound)
    if np.isclose(f_lower, 0.0):
        return float(lower_bound)
    if np.isclose(f_upper, 0.0):
        return float(upper_bound)
    if f_lower * f_upper > 0.0:
        raise ValueError(
            "Calibrated Poisson CI root is not bracketed. "
            f"obs={obs_count}, lower={lower_bound}, upper={upper_bound}, "
            f"f(lower)={f_lower}, f(upper)={f_upper}."
        )
    return float(brentq(objective, lower_bound, upper_bound))


def _poisson_interval_scalar(
    obs_count: float,
    expected_count: float,
    p_value: float,
    intercept: float,
    scale: float,
    alpha: float,
    upper_cap: float,
) -> Tuple[float, float]:
    if obs_count == 0:
        lower = 0.0
    elif obs_count <= expected_count:
        lower = _solve_poisson_bound(
            obs_count,
            intercept,
            scale,
            lower_bound=0.0,
            upper_bound=obs_count,
            alpha=alpha,
        )
    elif p_value > alpha:
        lower = _solve_poisson_bound(
            obs_count,
            intercept,
            scale,
            lower_bound=0.0,
            upper_bound=expected_count,
            alpha=alpha,
        )
    else:
        lower = _solve_poisson_bound(
            obs_count,
            intercept,
            scale,
            lower_bound=expected_count,
            upper_bound=obs_count,
            alpha=alpha,
        )

    if obs_count <= expected_count:
        if p_value > alpha:
            upper = _solve_poisson_bound(
                obs_count,
                intercept,
                scale,
                lower_bound=expected_count,
                upper_bound=upper_cap,
                alpha=alpha,
            )
        else:
            upper = _solve_poisson_bound(
                obs_count,
                intercept,
                scale,
                lower_bound=obs_count,
                upper_bound=expected_count,
                alpha=alpha,
            )
    else:
        upper = _solve_poisson_bound(
            obs_count,
            intercept,
            scale,
            lower_bound=obs_count,
            upper_bound=upper_cap,
            alpha=alpha,
        )

    return lower, upper


def poisson_confidence_bounds(
    obs: ArrayLike,
    exp: ArrayLike,
    p_value: ArrayLike,
    intercept: ArrayLike,
    scale: ArrayLike,
    alpha: float = _DEFAULT_ALPHA,
    upper_cap: float = _DEFAULT_UPPER_CAP,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibrated confidence bounds on expected counts via root-finding.

    For each facility the function finds the expected-count values at which
    the empirical-null-adjusted p-value equals ``alpha``, using Brent's
    method. The returned lower/upper bounds are on the expected-count scale;
    callers can divide by ``exp`` to obtain ratio-scale confidence limits.
    """
    obs_arr, exp_arr, p_arr, intercept_arr, scale_arr = np.broadcast_arrays(
        _as_float_array(obs, "obs"),
        _as_float_array(exp, "exp"),
        _as_float_array(p_value, "p_value"),
        _as_float_array(intercept, "intercept"),
        _as_float_array(scale, "scale"),
    )

    lower = np.empty(obs_arr.shape[0], dtype=np.float64)
    upper = np.empty(obs_arr.shape[0], dtype=np.float64)

    for i in range(obs_arr.shape[0]):
        lower[i], upper[i] = _poisson_interval_scalar(
            obs_count=float(obs_arr[i]),
            expected_count=float(exp_arr[i]),
            p_value=float(p_arr[i]),
            intercept=float(intercept_arr[i]),
            scale=float(scale_arr[i]),
            alpha=float(alpha),
            upper_cap=float(upper_cap),
        )

    return lower, upper


def log_ratio_confidence_intervals(
    ratio: ArrayLike,
    log_ratio_z: ArrayLike,
    stderr: ArrayLike,
    intercept: ArrayLike,
    scale: ArrayLike,
    alpha: float = _DEFAULT_ALPHA,
) -> Dict[str, np.ndarray]:
    """Compute empirical-null p-values and log-normal confidence intervals.

    For providers with ``ratio > 0``, computes closed-form log-normal
    intervals using the empirical-null-adjusted location and scale.
    For providers with ``ratio == 0`` (zero events), the log-normal
    formula is undefined; these providers receive the score-based linear
    approximation (``lower = 0``, ``upper = z_alpha * scale * stderr``)
    that keeps inference consistent with the empirical-null framework.

    Parameters
    ----------
    ratio : array-like
        Provider ratio estimate on the original scale.
    log_ratio_z : array-like
        Z-statistic formed as ``log(ratio) / stderr`` for positive ratios,
        or from ``log_ratio_zscore(..., zero_method="score")`` which
        substitutes ``-1/stderr`` for zero-ratio providers.
    stderr : array-like
        Standard error used for the log-normal interval.
    intercept, scale : array-like
        Empirical-null parameters, either scalar or provider-level.
    alpha : float, default 0.05
        Two-sided significance level.

    Returns
    -------
    dict
        Keys: ``test_stat``, ``p_value``, ``upper``, ``lower``.
    """
    ratio_arr, z_arr, stderr_arr, intercept_arr, scale_arr = np.broadcast_arrays(
        _as_float_array(ratio, "ratio"),
        _as_float_array(log_ratio_z, "log_ratio_z"),
        _as_float_array(stderr, "stderr"),
        _as_float_array(intercept, "intercept"),
        _as_float_array(scale, "scale"),
    )
    if np.any(scale_arr <= 0.0):
        raise ValueError("All empirical-null scale values must be positive.")
    if np.any(~np.isfinite(stderr_arr)) or np.any(stderr_arr <= 0.0):
        raise ValueError("stderr must be finite and positive for every provider.")

    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    test_stat = (z_arr - intercept_arr) / scale_arr
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(test_stat)))

    positive_ratio = ratio_arr > 0.0
    log_ratio = np.full(ratio_arr.shape, np.nan, dtype=np.float64)
    log_ratio[positive_ratio] = np.log(ratio_arr[positive_ratio])

    upper = np.exp((log_ratio - intercept_arr * stderr_arr) + (z_alpha * scale_arr * stderr_arr))
    lower = np.exp((log_ratio - intercept_arr * stderr_arr) - (z_alpha * scale_arr * stderr_arr))

    # Zero-ratio providers: log-normal formula is undefined (log(0) = -inf).
    # Apply the score-based linear approximation to keep all facilities in
    # the same inference framework.
    zero_mask = ratio_arr == 0.0
    if np.any(zero_mask):
        upper = np.where(zero_mask, z_alpha * scale_arr * stderr_arr, upper)
        lower = np.where(zero_mask, 0.0, lower)

    return {
        "test_stat": test_stat,
        "p_value": p_value,
        "upper": upper,
        "lower": lower,
    }


# R names (EmpiNull / empirical_null.R) for the same functions.
cal_Z_htaz = poisson_midp_zscore
empirical_null_overall = fit_empirical_null
empirical_null_groupwise = fit_grouped_empirical_null
empirical_null_adjust = adjust_empirical_null
smr_ci_bounds = poisson_confidence_bounds
log_normal_en_ci = log_ratio_confidence_intervals
