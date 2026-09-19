"""Cox model deviance, matched to glmnet's `coxnet.deviance()`.

Deviance = 2*(saturated_log_likelihood - log_likelihood) is a
beta-independent recentering of the partial log-likelihood onto a
"lower is better, 0 is a perfect fit" scale.  It is used two ways:

* Reporting `deviance_ratio_path_` alongside each lambda (glmnet's
  `dev.ratio`), a scale-free goodness-of-fit summary independent of
  the raw log-likelihood's units.
* Cross-validation: `PenalizedCoxPHCV` computes, per fold, the
  Verweij & Van Houwelingen (1993) grouped deviance residual
  `deviance(all data; beta_hat_-k) - deviance(training-only data;
  beta_hat_-k)`, exactly matching glmnet's `cv.glmnet(family="cox")`
  default (`grouped=TRUE`) -- confirmed against glmnet 4.1-8's actual
  ``R/buildPredmat.coxnetlist.R`` and ``R/cv.coxnet.R``, not the paper
  or vignette alone (see ``docs/R_COMPATIBILITY.md``).

The saturated log-likelihood formula below (`-sum(wd*log(wd))` over
per-stratum tied-event-time weight sums) is copied from glmnet's own
`coxnet.deviance2`/`coxnet.deviance3` (R/coxnet.deviance.R), which the
glmnet authors' own comment states "gives the same output as
coxnet.deviance0() [the production Fortran-backed path] ... written
completely in R" -- i.e. it is glmnet's own cross-checked reference
form, not a re-derivation from first principles.

Grouping is by `stop` time among `event == 1` rows in every case
(ordinary right-censored data is the start=0 special case of
start/stop data throughout this package -- see
`algorithms/partial_likelihood.py`), matching glmnet's use of
`stop_time` for ties in its (start, stop] deviance routine.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.grouping import iter_stratum_indices


def saturated_log_likelihood(
    stop: np.ndarray, event: np.ndarray, weight: np.ndarray, strata_codes: np.ndarray
) -> float:
    """Sum over strata of -sum_k(wd_k * log(wd_k)), where wd_k is the
    total sample weight of the (possibly tied) events at the k-th
    distinct event time within that stratum.
    """
    total = 0.0
    event_bool = event == 1
    for _, idx in iter_stratum_indices(strata_codes):
        ev = event_bool[idx]
        if not np.any(ev):
            continue
        times = stop[idx][ev]
        w = weight[idx][ev]
        order = np.argsort(times, kind="mergesort")
        times_sorted = times[order]
        w_sorted = w[order]
        _, start_idx = np.unique(times_sorted, return_index=True)
        wd = np.add.reduceat(w_sorted, start_idx)
        wd = wd[wd > 0]
        if wd.size:
            total += -np.sum(wd * np.log(wd))
    return float(total)


def cox_deviance(log_likelihood: float, lsat: float) -> float:
    """2 * (lsat - log_likelihood)."""
    return 2.0 * (lsat - log_likelihood)


def deviance_ratio(log_likelihood: float, log_likelihood_null: float, lsat: float) -> float:
    """1 - deviance(fit) / deviance(null) -- glmnet's `dev.ratio`.

    A value of 1 means the model perfectly reproduces the saturated
    (event-order) log-likelihood; 0 means it does no better than the
    null (beta=0) model.
    """
    dev = cox_deviance(log_likelihood, lsat)
    dev_null = cox_deviance(log_likelihood_null, lsat)
    if dev_null == 0:
        return 0.0
    return 1.0 - dev / dev_null


# ======================================================================
# Bootstrap SE for cross-validated deviance  (grplasso §4.3.2)
# ======================================================================

def _loglik_from_eta_multi(
    eta_matrix: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    weight: np.ndarray,
    strata_codes: np.ndarray,
    ties: str = "breslow",
) -> np.ndarray:
    """Log-partial-likelihood for *multiple* eta vectors at once.

    Vectorised across columns of ``eta_matrix`` (one column per
    lambda).  This avoids repeated sorting and stratum iteration
    when evaluating the same resampled dataset at many lambda values
    (the bootstrap inner loop).

    Parameters
    ----------
    eta_matrix : ndarray, shape ``(n, n_lambda)``
        Pre-computed linear predictors (one column per lambda).
    start, stop, event, weight, strata_codes : ndarray, shape ``(n,)``
    ties : ``'breslow'`` (Efron not supported here; use the
        ``cox_partial_likelihood`` fallback for Efron bootstrap).

    Returns
    -------
    loglik : ndarray, shape ``(n_lambda,)``
    """
    if ties != "breslow":
        raise NotImplementedError(
            "_loglik_from_eta_multi only supports Breslow ties; "
            f"got ties={ties!r}"
        )
    n, n_lambda = eta_matrix.shape
    loglik = np.zeros(n_lambda, dtype=np.float64)

    for _, idx in iter_stratum_indices(strata_codes):
        n_s = len(idx)
        if n_s == 0:
            continue

        order = np.argsort(stop[idx], kind="mergesort")
        ix = idx[order]

        eta_s = eta_matrix[ix]               # (n_s, n_lambda)
        start_s = start[ix]
        stop_s = stop[ix]
        event_s = event[ix]
        weight_s = weight[ix]

        # Clamp for numerical safety (matches safe_exp range)
        eta_clamped = np.clip(eta_s, -500.0, 500.0)
        hazard = weight_s[:, None] * np.exp(eta_clamped)  # (n_s, n_lambda)

        ev_mask = event_s == 1
        if not np.any(ev_mask):
            continue

        no_left_trunc = np.all(start_s <= 0.0)

        if no_left_trunc:
            # Right-censored: efficient reverse cumulative sum
            risk_sum = np.cumsum(hazard[::-1], axis=0)[::-1]  # (n_s, nlam)
            loglik += np.sum(
                weight_s[ev_mask, None]
                * (eta_clamped[ev_mask] - np.log(
                    np.maximum(risk_sum[ev_mask], 1e-300)
                )),
                axis=0,
            )
        else:
            # Start-stop data: per-event risk-set, vectorised across
            # lambda within each event.
            ev_indices = np.flatnonzero(ev_mask)
            for j_ev in ev_indices:
                t_j = stop_s[j_ev]
                at_risk = (start_s < t_j) & (stop_s >= t_j)
                risk_j = np.sum(hazard[at_risk], axis=0)  # (n_lambda,)
                loglik += weight_s[j_ev] * (
                    eta_clamped[j_ev]
                    - np.log(np.maximum(risk_j, 1e-300))
                )

    return loglik


def bootstrap_cv_se(
    eta_matrix: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    weight: np.ndarray,
    strata_codes: np.ndarray,
    n_bootstrap: int = 100,
    random_state: Optional[int] = None,
    ties: str = "breslow",
) -> np.ndarray:
    """Bootstrap SE for cross-validated deviance.

    Implements the approach from ``grplasso``'s ``se.strat_cox``:
    resample observations with replacement (preserving time ordering
    by sorting indices), recompute the deviance per-event-weight on
    each bootstrap replicate, and return the standard deviation
    across replicates as the SE estimate.

    The deviance is ``2 * (lsat - loglik)`` divided by the bootstrap
    sample's total event weight, matching the scale of the analytical
    ``cvm`` / ``cvsd`` from ``_PenalizedCoxPHCVBase._compute_cv_
    statistics``.

    Parameters
    ----------
    eta_matrix : ndarray, shape ``(n, n_lambda)``
        Out-of-fold linear predictors.  ``eta_matrix[i, j]`` is the
        linear predictor for observation *i* at lambda index *j*,
        computed using the beta from the fold where *i* was held out.
    start, stop, event, weight, strata_codes : ndarray, shape ``(n,)``
        Full-data survival arrays.
    n_bootstrap : int, default 100
        Number of bootstrap replicates.
    random_state : int or None
        Seed for reproducibility.
    ties : ``'breslow'``
        Tie-handling method.  Only Breslow is supported by the fast
        vectorised path; Efron raises ``NotImplementedError``.

    Returns
    -------
    se : ndarray, shape ``(n_lambda,)``
        Bootstrap standard error of the per-event-weight deviance at
        each lambda.
    """
    rng = np.random.RandomState(random_state)
    n = len(event)
    n_lambda = eta_matrix.shape[1]
    losses = np.empty((n_bootstrap, n_lambda), dtype=np.float64)

    for b in range(n_bootstrap):
        idx = np.sort(rng.choice(n, n, replace=True))

        loglik_b = _loglik_from_eta_multi(
            eta_matrix[idx], start[idx], stop[idx],
            event[idx], weight[idx], strata_codes[idx],
            ties=ties,
        )
        lsat_b = saturated_log_likelihood(
            stop[idx], event[idx], weight[idx], strata_codes[idx],
        )
        dev_b = 2.0 * (lsat_b - loglik_b)          # (n_lambda,)
        ew_b = float(np.sum(weight[idx] * event[idx]))
        losses[b] = dev_b / max(ew_b, 1e-12)

    return np.std(losses, axis=0, ddof=1)


# ---------------------------------------------------------------------------
# Discrete-time survival loss (binary cross-entropy deviance)
# ---------------------------------------------------------------------------

def discrete_survival_loss(
    y_expanded: np.ndarray,
    p_hat_expanded: np.ndarray,
) -> np.ndarray:
    """Binary cross-entropy deviance for discrete-time survival.

    Computes the per-observation deviance on person-period expanded
    data, matching ``grplasso``'s ``loss.Disc.Surv`` (R/loss.R)::

        loss_i = -2 * [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]

    Predicted probabilities are clamped to ``[1e-5, 1 - 1e-5]`` to
    avoid log(0), consistent with the R implementation's clamping
    to ``[0.00001, 0.99999]``.

    Parameters
    ----------
    y_expanded : ndarray, shape (N,)
        Binary outcomes in person-period format (1 = event row,
        0 = non-event row).
    p_hat_expanded : ndarray, shape (N,) or (N, n_lambda)
        Predicted hazard probabilities.  If 2D, loss is computed
        independently for each lambda column.

    Returns
    -------
    loss : ndarray, same shape as ``p_hat_expanded``
        Per-observation binary cross-entropy deviance.
    """
    p = np.clip(p_hat_expanded, 1e-5, 1.0 - 1e-5)
    loss = np.zeros_like(p, dtype=np.float64)
    if p.ndim == 1:
        mask1 = y_expanded == 1
        mask0 = y_expanded == 0
        loss[mask1] = -2.0 * np.log(p[mask1])
        loss[mask0] = -2.0 * np.log(1.0 - p[mask0])
    else:
        mask1 = y_expanded == 1
        mask0 = y_expanded == 0
        loss[mask1] = -2.0 * np.log(p[mask1])
        loss[mask0] = -2.0 * np.log(1.0 - p[mask0])
    return loss
