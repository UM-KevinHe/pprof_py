"""Baseline cumulative hazard and survival, dispatched through whichever
tie method actually fit the model.

Matches `survival::basehaz(fit, centered = FALSE)`: the hazard reported
is at X = 0 (plus whatever offset applies), in the covariates' original,
uncentered units -- not at X = mean(X), which is what a naive
implementation that forgot to un-translate the optimizer's internal
centering would silently produce instead. See utils/numerical.py and
docs/R_COMPATIBILITY.md, question 14.

The estimator itself is tie-method-specific, not just the likelihood:
Breslow's `dH0(t_j) = d*_j / S0(t_j)` and Efron's average-of-reciprocals
formula (see algorithms/ties.py::EfronTies.baseline_hazard_increments)
give visibly different numbers whenever there are ties, even though both
are perfectly valid "baseline hazard for this fitted model" estimators --
using the wrong one is a genuine mismatch with R, not a matter of taste.
This module contains no tie-method-specific math at all for exactly that
reason: it only loops over strata and calls
`tie_method.baseline_hazard_increments(...)`, so a tie method implemented
correctly in algorithms/ties.py is automatically correct here too.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from ...algorithms.survival.ties import TieMethod, BreslowTies, EfronTies, ExactTies
from ...utils.grouping import iter_stratum_indices

_TIE_METHODS = {"breslow": BreslowTies, "efron": EfronTies, "exact": ExactTies}


def compute_baseline_hazard(
    X: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    eta: np.ndarray,
    weight: np.ndarray,
    strata_codes: np.ndarray,
    strata_labels: np.ndarray,
    ties: Union[str, TieMethod] = "breslow",
) -> pd.DataFrame:
    """Return a DataFrame with columns [stratum, time, hazard, survival],
    one row per distinct event time per stratum. `hazard` is the
    cumulative baseline hazard H0(t); `survival` is exp(-H0(t)).

    `eta` must already be X @ beta + offset evaluated on the *original*
    (uncentered) X -- see models/coxph.py, which is the only caller and
    is responsible for that. `ties` must be the SAME tie method the model
    was actually fit with (models/coxph.py passes `self.ties` through) --
    the hazard estimator is not interchangeable across tie methods.
    """
    if isinstance(ties, str):
        if ties not in _TIE_METHODS:
            raise ValueError(f"Unknown ties method {ties!r}; expected one of {list(_TIE_METHODS)}")
        tie_method = _TIE_METHODS[ties]()
    else:
        tie_method = ties

    rows = []
    for code, idx in iter_stratum_indices(strata_codes):
        label = strata_labels[code]
        event_times, dH0 = tie_method.baseline_hazard_increments(
            X[idx], start[idx], stop[idx], event[idx], weight[idx], eta[idx],
        )
        if len(event_times) == 0:
            continue
        H0 = np.cumsum(dH0)
        rows.append(
            pd.DataFrame(
                {
                    "stratum": label,
                    "time": event_times,
                    "hazard": H0,
                    "survival": np.exp(-H0),
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["stratum", "time", "hazard", "survival"])
    return pd.concat(rows, ignore_index=True)

