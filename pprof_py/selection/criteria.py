"""Model-selection criteria for CoxPHSelector.

Computed directly from a fitted CoxPH model's own reported statistics --
no separate or duplicated partial-likelihood computation happens here,
per the requirement that the selector reuse CoxPH rather than
reimplementing any of its statistics.

Matched to R's actual `extractAIC.coxph` (a base-R S3 method registered
by `survival`, inspected directly via `getAnywhere(extractAIC.coxph)`
rather than assumed):

    edf    <- sum(!is.na(fit$coefficients))
    loglik <- fit$loglik[length(fit$loglik)]
    AIC    <- -2 * loglik + k * edf

`edf` is the number of estimated (non-aliased) coefficients -- this
package's CoxPH never produces an aliased/NA coefficient (a
near-singular information matrix falls back to a pseudo-inverse rather
than dropping a term), so `edf` is simply `n_features_in_`. `loglik` is
the log-likelihood *at the fitted coefficients*, not the null model --
`log_likelihood_`, not `log_likelihood_null_`.

For BIC, R's `step()` is called with `k = log(nobs(fit))`, and
`nobs.coxph(fit)` was confirmed directly (not assumed) to return
`fit$nevent` -- the number of EVENTS, not `fit$n` (the number of
observations). This matters in every survival dataset with censoring
(i.e. essentially always): using `n_obs` instead of `n_events` for the
BIC penalty gives a different, non-R-matching number. See
docs/R_COMPATIBILITY.md for the R session that established this.
"""
from __future__ import annotations

import numpy as np


def aic(model) -> float:
    """-2*logLik(beta_hat) + 2*edf, matching R's extractAIC.coxph(fit)
    (its default k=2)."""
    edf = model.n_features_in_
    return -2.0 * model.log_likelihood_ + 2.0 * edf


def bic(model) -> float:
    """-2*logLik(beta_hat) + log(n_events)*edf, matching
    extractAIC.coxph(fit, k=log(nobs(fit))) with R's nobs.coxph
    convention (n_events, not n_obs) -- see module docstring."""
    edf = model.n_features_in_
    n_events = model.n_events_
    return -2.0 * model.log_likelihood_ + np.log(n_events) * edf


CRITERIA = {"aic": aic, "bic": bic}
