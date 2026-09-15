"""A deliberately naive, O(n*K)-or-worse reimplementation of the Breslow
partial likelihood, score, and Hessian -- written by directly transcribing
the math with no incremental-sweep cleverness, no vectorization trick, and
no shared code with algorithms/risk_sets.py or algorithms/ties.py.

This exists purely so that BreslowTies (the efficient, production
implementation) can be checked against something whose correctness is
obvious by inspection, on data with strata / offsets / weights / left
truncation in every combination -- independent of whether R happens to
be installed. If this and BreslowTies disagree, the bug is almost
certainly in the efficient sweep, not in R-compatibility.
"""
from __future__ import annotations

import numpy as np


def brute_force_partial_likelihood(X, start, stop, event, beta, offset, weight, strata):
    n, p = X.shape
    eta = X @ beta + offset
    r = np.exp(eta)

    total_ll = 0.0
    total_score = np.zeros(p)
    total_info = np.zeros((p, p))

    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        for i in idx:
            if event[i]:
                total_ll += weight[i] * eta[i]

        event_times = sorted({stop[i] for i in idx if event[i]})
        for t in event_times:
            # naive O(n) scan of the risk set for THIS single time point,
            # no reuse of work done for other time points
            risk_idx = [i for i in idx if start[i] < t <= stop[i]]
            death_idx = [i for i in idx if event[i] and stop[i] == t]
            d_star = sum(weight[i] for i in death_idx)

            S0 = sum(weight[i] * r[i] for i in risk_idx)
            S1 = sum(weight[i] * r[i] * X[i] for i in risk_idx)
            S2 = sum(weight[i] * r[i] * np.outer(X[i], X[i]) for i in risk_idx)

            total_ll -= d_star * np.log(S0)
            xbar = S1 / S0
            total_score += sum(weight[i] * X[i] for i in death_idx) - d_star * xbar
            total_info += d_star * (S2 / S0 - np.outer(xbar, xbar))

    return total_ll, total_score, total_info
