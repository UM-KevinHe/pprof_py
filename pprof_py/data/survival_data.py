"""Internal representation of validated survival data.

Everything downstream of validation (risk sets, the partial-likelihood
engine, optimization) operates on this plain-NumPy container rather than
on a DataFrame.  That boundary is deliberate: it is the seam a future
distributed backend would replace (each partition computing its own
partial sums over this same field layout), without the user-facing
``CoxPH`` class or anything in ``algorithms/`` needing to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SurvivalData:
    X: np.ndarray
    start: np.ndarray
    stop: np.ndarray
    event: np.ndarray
    offset: np.ndarray
    weight: np.ndarray
    strata_codes: np.ndarray
    strata_labels: np.ndarray
    feature_names: List[str]

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_strata(self) -> int:
        return len(self.strata_labels)
