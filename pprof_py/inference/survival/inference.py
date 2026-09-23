"""Statistical inference for CoxPH models.

Wald-type inference (covariance, standard errors, z-statistics, p-values,
confidence intervals) computed from the observed information matrix at
convergence, plus Poisson-based exact tests for standardised count ratios.

See ``docs/source/survival/R_COMPATIBILITY.md`` for why observed
information is the correct matrix for both the Newton-Raphson step *and*
the covariance estimate in this model (no separate "expected information"
step is needed).  Robust/sandwich variance is in
``inference/survival/robust.py``.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# REV-011: promoted to pprof_py.utils.numerical so the logistic/linear
# fixed-effect code can share it.  Re-exported here so existing imports
# keep working; the default (warn=False) preserves the historical
# silent-fallback behaviour for these callers exactly.
from ...utils.numerical import covariance_from_information  # noqa: E402,F401


def standard_errors(covariance: np.ndarray) -> np.ndarray:
    diag = np.diag(covariance)
    if np.any(diag < 0):
        raise ValueError(
            "Negative variance on the diagonal of the covariance matrix -- the "
            "information matrix is not positive definite. This usually means the "
            "model has not converged or has a collinearity/identifiability "
            "problem; check `n_iter_` and `converged_` before trusting any other "
            "output of this fit."
        )
    return np.sqrt(diag)


def wald_statistics(beta: np.ndarray, se: np.ndarray):
    """Two-sided Wald z-test for each coefficient, matching the test
    R's summary.coxph() reports by default (as opposed to the likelihood
    ratio or score test, which are computed separately if needed)."""
    z = beta / se
    p = 2.0 * stats.norm.sf(np.abs(z))
    return z, p


def confidence_intervals(beta: np.ndarray, se: np.ndarray, level: float = 0.95):
    """Wald confidence intervals beta_hat +/- z_(alpha/2) * SE."""
    alpha = 1.0 - level
    crit = stats.norm.ppf(1.0 - alpha / 2.0)
    lower = beta - crit * se
    upper = beta + crit * se
    return lower, upper


# ---------------------------------------------------------------------------
# Poisson exact test for standardised count ratios (e.g. SWR)
# ---------------------------------------------------------------------------

def poisson_exact_test(
    obs: np.ndarray,
    exp: np.ndarray,
    alpha: float = 0.05,
    normal_threshold: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided Poisson test with exact or normal-approximation CIs.

    For each facility, computes the standardised ratio ``obs / exp``, a
    two-sided p-value from the Poisson distribution, and a confidence
    interval on the ratio scale.

    Matches the R functions ``compute_pvalue``, ``compute_poisson_exact_CI``,
    and ``compute_normal_approx_CI`` in ``SMR_Empirical_Null.R``.

    Parameters
    ----------
    obs : array_like
        Observed event counts per facility (non-negative integers).
    exp : array_like
        Expected event counts per facility (positive reals).
    alpha : float, optional
        Significance level for the confidence interval (default 0.05 →
        95 % CI).
    normal_threshold : float, optional
        Expected-count cutoff above which the normal approximation is
        used instead of the exact chi-square–based interval
        (default 100, matching R's ``SMR_Empirical_Null.R``).

    Returns
    -------
    p_value : ndarray
        Two-sided Poisson p-values, clipped to [0, 0.999].
    lower_ratio : ndarray
        Lower confidence-interval bound on the ratio scale.
    upper_ratio : ndarray
        Upper confidence-interval bound on the ratio scale.

    Notes
    -----
    *   **P-value** (standard two-sided Poisson, NOT mid-p):

        - ``ratio > 1``:  ``min(0.999, 2 * P(X >= obs | exp))``
        - ``ratio <= 1``: ``min(0.999, 2 * P(X <= obs | exp))``

    *   **CI when exp < normal_threshold** (exact Poisson):

        - lower = ``chi2.ppf(alpha/2, 2*obs) / 2 / exp``  (0 when obs=0)
        - upper = ``chi2.ppf(1-alpha/2, 2*(obs+1)) / 2 / exp``

    *   **CI when exp >= normal_threshold** (normal approximation):

        - Uses the cube-root transformation of Byar (1986).
    """
    obs = np.asarray(obs, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    n = obs.size

    ratio = obs / exp
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    # ---- p-value (standard two-sided Poisson) ----
    p_value = np.full(n, np.nan)
    high = ratio > 1
    low = ~high  # ratio <= 1 (includes ratio == 1)
    # P(X >= obs) = 1 - P(X <= obs-1) = sf(obs-1)
    p_value[high] = np.minimum(
        0.999, 2.0 * stats.poisson.sf(obs[high] - 1, exp[high])
    )
    # P(X <= obs) = cdf(obs)
    p_value[low] = np.minimum(
        0.999, 2.0 * stats.poisson.cdf(obs[low], exp[low])
    )

    # ---- confidence intervals ----
    lower = np.zeros(n)
    upper = np.zeros(n)

    exact = exp < normal_threshold
    approx = ~exact

    # -- exact Poisson CI (chi-square–based) --
    if exact.any():
        obs_e = obs[exact]
        exp_e = exp[exact]
        nonzero = obs_e > 0
        # lower: 0 when obs==0, else chi2.ppf(alpha/2, 2*obs) / 2 / exp
        lower_e = np.zeros_like(obs_e)
        if nonzero.any():
            lower_e[nonzero] = (
                stats.chi2.ppf(alpha / 2, 2 * obs_e[nonzero]) / 2.0 / exp_e[nonzero]
            )
        # upper: chi2.ppf(1-alpha/2, 2*(obs+1)) / 2 / exp
        upper_e = stats.chi2.ppf(1.0 - alpha / 2, 2 * (obs_e + 1)) / 2.0 / exp_e
        lower[exact] = lower_e
        upper[exact] = upper_e

    # -- normal approximation CI (Byar's cube-root) --
    if approx.any():
        obs_a = obs[approx]
        exp_a = exp[approx]
        # lower: (obs/exp) * (1 - 1/(9*obs) - z/(3*sqrt(obs)))^3
        # handle obs==0 → lower=0
        nz = obs_a > 0
        lower_a = np.zeros_like(obs_a)
        if nz.any():
            lower_a[nz] = (
                (obs_a[nz] / exp_a[nz])
                * (1 - 1 / (9 * obs_a[nz]) - z_crit / (3 * np.sqrt(obs_a[nz]))) ** 3
            )
        # upper: ((obs+1)/exp) * (1 - 1/(9*(obs+1)) + z/(3*sqrt(obs+1)))^3
        obs1 = obs_a + 1
        upper_a = (
            (obs1 / exp_a)
            * (1 - 1 / (9 * obs1) + z_crit / (3 * np.sqrt(obs1))) ** 3
        )
        lower[approx] = lower_a
        upper[approx] = upper_a

    return p_value, lower, upper
