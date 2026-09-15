"""Built-in measure functions for IUR estimation.

A **measure function** computes a group-level summary statistic from
patient-level observed and expected outcomes.  The standard signature
is::

    measure_fn(obs, exp, groups) -> measures

where *obs*, *exp*, and *groups* are 1-d arrays of equal length
(**sorted by group**), and the return value is a 1-d array of length
``n_groups`` (one value per unique group, in sorted order).

Custom measure functions may be passed to any IUR estimator via the
``measure_fn`` parameter.
"""
from __future__ import annotations

import numpy as np


def ratio_measure(
    obs: np.ndarray,
    exp: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Observed-to-expected ratio per group.

    Computes ``sum(obs_g) / sum(exp_g)`` for each group *g*.  This is
    the standard measure for ratio-based quality indicators (e.g.,
    standardised mortality, hospitalisation, or readmission ratios).

    Parameters
    ----------
    obs : ndarray of shape (n_samples,)
        Observed outcomes (numerator contributions).
    exp : ndarray of shape (n_samples,)
        Expected outcomes (denominator contributions).
    groups : ndarray of shape (n_samples,)
        Group identifiers.  Data **must** be sorted by group.

    Returns
    -------
    measures : ndarray of shape (n_groups,)
        Ratio measure for each group, in sorted group order.
    """
    _, idx = np.unique(groups, return_index=True)
    obs_sum = np.add.reduceat(obs, idx)
    exp_sum = np.add.reduceat(exp, idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(exp_sum != 0, obs_sum / exp_sum, np.nan)
