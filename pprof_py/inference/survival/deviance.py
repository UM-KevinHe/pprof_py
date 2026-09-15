"""Cox model deviance, matched to glmnet's `coxnet.deviance()`.

Deviance = 2*(saturated_log_likelihood - log_likelihood) is a
beta-independent recentering of the partial log-likelihood onto a
"lower is better, 0 is a perfect fit" scale. It is used two ways in
Phase 3:

* Reporting `deviance_ratio_path_` alongside each lambda (glmnet's
  `dev.ratio`), a scale-free goodness-of-fit summary independent of
  the raw log-likelihood's units.
* Cross-validation: `PenalizedCoxPHCV` computes, per fold, the
  Verweij & Van Houwelingen (1993) grouped deviance residual
  `deviance(all data; beta_hat_-k) - deviance(training-only data;
  beta_hat_-k)`, exactly matching glmnet's `cv.glmnet(family="cox")`
  default (`grouped=TRUE`) -- confirmed against glmnet 4.1-8's actual
  `R/buildPredmat.coxnetlist.R` and `R/cv.coxnet.R`, not the paper or
  vignette alone (see `docs/R_COMPATIBILITY.md`, Phase 3 section).

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

import numpy as np

from ...utils.grouping import iter_stratum_indices


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
