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
* `assign_quantile_groups(size, n_groups=4)`
* `fit_robust_location_scale(z, psi="bisquare")`
* `fit_empirical_null(...)`
* `fit_grouped_empirical_null(...)`
* `adjust_empirical_null(...)`
* `poisson_confidence_bounds(...)`
* `log_ratio_confidence_intervals(...)`

The robust location/scale estimator is an intercept-only IRLS routine with
MAD scale updates at every iteration, supporting both Huber and Tukey
bisquare psi functions.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, poisson


ArrayLike = Union[np.ndarray, list, tuple]

_LITERAL_MAD_CONSTANT = 0.6745
_DEFAULT_HUBER_K = 1.345
_DEFAULT_BISQUARE_C = 4.685
_DEFAULT_ALPHA = 0.05
_DEFAULT_UPPER_CAP = 10000.0

__all__ = [
    "poisson_midp_zscore",
    "log_ratio_zscore",
    "assign_quantile_groups",
    "fit_robust_location_scale",
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


def _resolve_psi(psi: str, tuning: Optional[float]) -> Tuple[str, float]:
    psi_key = psi.lower().replace("_", "").replace("-", "")
    aliases = {
        "huber": "huber",
        "psihuber": "huber",
        "bisquare": "bisquare",
        "bisq": "bisquare",
        "biweight": "bisquare",
        "tukey": "bisquare",
        "tukeybisquare": "bisquare",
        "psibisquare": "bisquare",
    }
    if psi_key not in aliases:
        raise ValueError(
            "Unsupported psi. Use 'huber' or 'bisquare'."
        )
    psi_name = aliases[psi_key]
    if tuning is None:
        tuning = _DEFAULT_HUBER_K if psi_name == "huber" else _DEFAULT_BISQUARE_C
    if tuning <= 0:
        raise ValueError("tuning must be positive.")
    return psi_name, float(tuning)


def _validate_method(method: str) -> None:
    if method.upper() != "M":
        raise NotImplementedError(
            "Only method='M' is currently implemented for empirical-null fitting."
        )


def _mad_scale(residuals: np.ndarray) -> float:
    scale = np.median(np.abs(residuals)) / _LITERAL_MAD_CONSTANT
    if not np.isfinite(scale):
        return 0.0
    return float(scale)


def _psi_weights(u: np.ndarray, psi_name: str, tuning: float) -> np.ndarray:
    abs_u = np.abs(u)
    if psi_name == "huber":
        return np.where(abs_u <= tuning, 1.0, tuning / abs_u)
    mask = abs_u < tuning
    u_scaled = np.zeros_like(u)
    u_scaled[mask] = u[mask] / tuning
    w = np.zeros_like(u)
    w[mask] = (1.0 - u_scaled[mask] ** 2) ** 2
    return w


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


def assign_quantile_groups(size: ArrayLike, n_groups: int = 4) -> np.ndarray:
    """Assign 1-indexed quantile groups based on empirical quantile breaks.

    The breaks are formed from ``np.quantile(size, probs)`` with implicit
    ``(-inf, ..., inf)`` sentinels. Values falling on a break boundary are
    assigned to the lower group (right-closed bins ``(a, b]``), matching R's
    ``cut(size, breaks, right=TRUE)`` convention.
    """
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")

    size_arr = _as_float_array(size, "size")
    if np.any(~np.isfinite(size_arr)):
        raise ValueError("size must contain only finite values for quantile grouping.")

    probs = np.linspace(0.0, 1.0, n_groups + 1)[1:n_groups]
    quantiles = np.quantile(size_arr, probs, method="linear")

    finite_breaks = quantiles[np.isfinite(quantiles)]
    if np.unique(finite_breaks).size != finite_breaks.size:
        raise ValueError(
            "Non-unique quantile breaks encountered; grouping would be ambiguous."
        )

    return np.searchsorted(quantiles, size_arr, side="left").astype(int) + 1


def fit_robust_location_scale(
    z: ArrayLike,
    psi: str = "bisquare",
    method: str = "M",
    tuning: Optional[float] = None,
    maxiter: int = 1000,
    tol: float = 1e-8,
) -> Tuple[float, float]:
    """Fit an intercept-only robust location/scale model.

    Parameters
    ----------
    z : array-like
        Z-scores to fit.
    psi : {"bisquare", "huber"}, default "bisquare"
        Robust weighting function.
    method : str, default "M"
        Only M-estimation is currently supported.
    tuning : float, optional
        Tuning constant. Defaults to 4.685 for bisquare and 1.345 for huber.
    maxiter : int, default 1000
        Maximum number of IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance on the residual-vector norm.

    Returns
    -------
    tuple[float, float]
        Estimated intercept and scale.
    """
    _validate_method(method)
    psi_name, tuning_value = _resolve_psi(psi, tuning)

    z_arr = _as_float_array(z, "z")
    z_fit = z_arr[np.isfinite(z_arr)]
    if z_fit.size < 2:
        raise ValueError("At least two finite z values are required for robust fitting.")

    # Initialise with median (matching R's rlm which uses lqs/L1 for method="M").
    # For non-convex psi (bisquare), the starting point matters because
    # the objective has local optima.
    mu = float(np.median(z_fit))
    residuals = z_fit - mu
    scale = _mad_scale(residuals)

    if scale == 0.0:
        return mu, 0.0

    for _ in range(maxiter):
        scale = _mad_scale(residuals)
        if scale == 0.0:
            break

        u = residuals / scale
        weights = _psi_weights(u, psi_name=psi_name, tuning=tuning_value)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            break

        mu_new = float(np.sum(weights * z_fit) / weight_sum)
        residuals_new = z_fit - mu_new

        denom = max(1e-20, float(np.sqrt(np.sum(residuals ** 2))))
        delta = float(np.sqrt(np.sum((residuals - residuals_new) ** 2)) / denom)

        mu = mu_new
        residuals = residuals_new

        if delta <= tol:
            break

    return mu, scale


def fit_empirical_null(
    z: ArrayLike,
    psi: str = "bisquare",
    method: str = "M",
    tuning: Optional[float] = None,
    maxiter: int = 1000,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """Estimate a single empirical-null intercept and scale for all facilities."""
    intercept, scale = fit_robust_location_scale(
        z,
        psi=psi,
        method=method,
        tuning=tuning,
        maxiter=maxiter,
        tol=tol,
    )
    return {"intercept": intercept, "scale": scale}


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
    """Estimate group-specific empirical-null parameters by quantiles of size."""
    z_arr = _as_float_array(z, "z")
    size_arr = _as_float_array(size, "size")
    if z_arr.shape[0] != size_arr.shape[0]:
        raise ValueError("z and size must have the same length.")

    group = assign_quantile_groups(size_arr, n_groups=n_groups)
    intercept = np.empty(n_groups, dtype=np.float64)
    scale = np.empty(n_groups, dtype=np.float64)

    for g in range(1, n_groups + 1):
        z_group = z_arr[group == g]
        z_group = z_group[np.isfinite(z_group)]
        if z_group.size < 2:
            raise ValueError(
                f"Group {g} has fewer than 2 finite facilities; robust fit cannot be computed."
            )
        intercept[g - 1], scale[g - 1] = fit_robust_location_scale(
            z_group,
            psi=psi,
            method=method,
            tuning=tuning,
            maxiter=maxiter,
            tol=tol,
        )

    return {
        "intercept": intercept,
        "scale": scale,
        "group": group,
    }


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
    """Estimate empirical-null parameters and return adjusted Z-scores/p-values.

    Parameters
    ----------
    z : array-like
        Z-scores for each provider.  NaN values are excluded from the
        robust fit but included in the output (their adjusted values will
        be NaN).
    size : array-like, optional
        Provider-level size variable for quantile-based grouping.
        Required when ``n_groups > 1`` and ``group_labels`` is not supplied.
    n_groups : int, default 4
        Number of quantile groups.  Ignored when ``group_labels`` is
        supplied or when ``n_groups == 1``.
    group_labels : array-like of int, optional
        Pre-computed 1-indexed group assignments.  When provided, ``size``
        and ``n_groups`` are ignored and the robust fit is performed per
        unique label.  Use this when group breaks must come from a
        different population than the one being calibrated.
    common_mean : bool or float, default False
        Intercept override:
        * ``False``: keep the fitted intercept for each group.
        * ``True``: replace all intercepts with ``mean(z)``.
        * numeric scalar: force all intercepts to that value.
    psi, method, tuning, maxiter, tol
        Passed through to ``fit_robust_location_scale``.

    Returns
    -------
    dict
        ``z_adj`` : EN-adjusted Z-scores.
        ``p_value`` : Two-sided EN p-values.
        ``intercept`` : Per-provider intercept used.
        ``scale`` : Per-provider scale used.
        ``group`` : Per-provider group index.
        ``params`` : Dict of group-level arrays (``group``, ``intercept``,
        ``scale``).
    """
    z_arr = _as_float_array(z, "z")

    if n_groups < 1:
        raise ValueError("n_groups must be a positive integer.")
    if not isinstance(common_mean, (bool, int, float, np.bool_, np.integer, np.floating)):
        raise ValueError("common_mean must be bool or numeric.")

    rlm_kwargs = dict(psi=psi, method=method, tuning=tuning, maxiter=maxiter, tol=tol)

    if group_labels is not None:
        # --- Pre-computed groups path ---
        group = _as_float_array(group_labels, "group_labels").astype(int)
        if group.shape[0] != z_arr.shape[0]:
            raise ValueError("group_labels and z must have the same length.")
        unique_groups = np.sort(np.unique(group))
        n_unique = unique_groups.shape[0]
        params_intercept = np.empty(n_unique, dtype=np.float64)
        params_scale = np.empty(n_unique, dtype=np.float64)
        for idx, g in enumerate(unique_groups):
            z_group = z_arr[group == g]
            z_group = z_group[np.isfinite(z_group)]
            if z_group.size < 2:
                raise ValueError(
                    f"Group {g} has fewer than 2 finite values; robust fit cannot be computed."
                )
            params_intercept[idx], params_scale[idx] = fit_robust_location_scale(
                z_group, **rlm_kwargs,
            )
        params_group = unique_groups
        # Build lookup: group label → index into params arrays
        label_to_idx = {int(g): idx for idx, g in enumerate(unique_groups)}
        param_idx = np.array([label_to_idx[int(g)] for g in group], dtype=int)

    elif n_groups == 1:
        fit = fit_empirical_null(z_arr, **rlm_kwargs)
        params_group = np.array([1], dtype=int)
        params_intercept = np.array([fit["intercept"]], dtype=np.float64)
        params_scale = np.array([fit["scale"]], dtype=np.float64)
        group = np.ones(z_arr.shape[0], dtype=int)
        param_idx = np.zeros(z_arr.shape[0], dtype=int)

    else:
        if size is None:
            raise ValueError("size must be supplied when n_groups > 1 and group_labels is not provided.")
        fit = fit_grouped_empirical_null(
            z_arr, size=size, n_groups=n_groups, **rlm_kwargs,
        )
        params_group = np.arange(1, n_groups + 1, dtype=int)
        params_intercept = np.asarray(fit["intercept"], dtype=np.float64)
        params_scale = np.asarray(fit["scale"], dtype=np.float64)
        group = np.asarray(fit["group"], dtype=int)
        param_idx = group - 1

    if isinstance(common_mean, (bool, np.bool_)) and common_mean:
        params_intercept = np.full(
            params_intercept.shape,
            np.nanmean(z_arr),
            dtype=np.float64,
        )
    elif not isinstance(common_mean, (bool, np.bool_)):
        params_intercept = np.full(
            params_intercept.shape,
            float(common_mean),
            dtype=np.float64,
        )

    intercept = params_intercept[param_idx]
    scale = params_scale[param_idx]
    if np.any(scale <= 0.0):
        raise ValueError("All empirical-null scale estimates must be positive.")

    z_adj = (z_arr - intercept) / scale
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(z_adj)))

    return {
        "z_adj": z_adj,
        "p_value": p_value,
        "intercept": intercept,
        "scale": scale,
        "group": group,
        "params": {
            "group": params_group,
            "intercept": params_intercept,
            "scale": params_scale,
        },
    }


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


# Compatibility aliases for the initial R-to-Python migration.
cal_Z_htaz = poisson_midp_zscore
quantile_groups = assign_quantile_groups
rlm_location_scale = fit_robust_location_scale
empirical_null_overall = fit_empirical_null
empirical_null_groupwise = fit_grouped_empirical_null
empirical_null_adjust = adjust_empirical_null
smr_ci_bounds = poisson_confidence_bounds
log_normal_en_ci = log_ratio_confidence_intervals
