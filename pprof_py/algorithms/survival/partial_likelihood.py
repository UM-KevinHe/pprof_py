"""Core Cox partial-likelihood engine.

Computes log-likelihood, score, and observed information for a given
beta.  This module knows nothing about pandas, optimization, or the
user-facing API -- it is pure NumPy in, NumPy out.  That boundary is
the seam a future distributed backend would replace: each partition
computes its own per-stratum ``StratumContribution`` and the results
are summed across partitions, unchanged from how strata are already
summed here.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .ties import TieMethod, BreslowTies, EfronTies, ExactTies
from ...utils.grouping import iter_stratum_indices

_TIE_METHODS = {"breslow": BreslowTies, "efron": EfronTies, "exact": ExactTies}


def precompute_stratum_indices(strata_codes: np.ndarray):
    """Pre-compute per-stratum index arrays via a single sort.

    Returns a list of 1-D int64 arrays, one per stratum, each containing
    the row indices belonging to that stratum (in ascending stratum-code
    order -- see `utils/grouping.py::iter_stratum_indices`, which this
    wraps). Callers that iterate over strata can fancy-index with these
    small arrays instead of creating a full-length boolean mask per
    stratum per Newton-Raphson iteration -- the dominant overhead when
    the number of strata is large (e.g. 7000+ provider strata in a
    stratified Cox model).
    """
    return [idx for _, idx in iter_stratum_indices(strata_codes)]


def cox_partial_likelihood(
    X: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    beta: np.ndarray,
    offset: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
    ties: Union[str, TieMethod] = "breslow",
    stratum_indices=None,
):
    """Evaluate the Cox partial log-likelihood, score, and information
    matrix at a given coefficient vector `beta`.

    Parameters mirror R's internal coxfit routine as closely as a
    Python/NumPy signature allows: `start`/`stop`/`event` define the
    (start, stop] risk intervals (ordinary right-censored data is simply
    start=0 for every row), `offset` enters the linear predictor
    unpenalized, `weight` gives case weights, and `strata` partitions
    observations into independent risk-set groups that share `beta`.

    Returns
    -------
    log_likelihood : float
    score : ndarray, shape (p,)
        Gradient of the log partial likelihood w.r.t. beta.
    information : ndarray, shape (p, p)
        Negative Hessian of the log partial likelihood -- the matrix
        whose inverse, evaluated at the MLE, is the model-based
        covariance of beta_hat.
    """
    n, p = X.shape
    if offset is None:
        offset = np.zeros(n)
    if weight is None:
        weight = np.ones(n)
    if strata is None:
        strata_codes = np.zeros(n, dtype=np.int64)
    else:
        strata_codes = strata

    if isinstance(ties, str):
        if ties not in _TIE_METHODS:
            raise ValueError(f"Unknown ties method {ties!r}; expected one of {list(_TIE_METHODS)}")
        tie_method = _TIE_METHODS[ties]()
    else:
        tie_method = ties

    eta = X @ beta + offset

    total_loglik = 0.0
    total_score = np.zeros(p)
    total_information = np.zeros((p, p))

    if stratum_indices is None:
        stratum_indices = precompute_stratum_indices(strata_codes)

    for idx in stratum_indices:
        contribution = tie_method.stratum_contribution(
            X[idx], start[idx], stop[idx], event[idx], weight[idx], eta[idx],
        )
        total_loglik += contribution.log_likelihood
        total_score += contribution.score
        total_information += contribution.information

    return total_loglik, total_score, total_information
