"""Direct IUR estimation from pre-computed estimates and standard errors.

For measures where group-level estimates and their standard errors are
already available (e.g., from a logistic fixed-effect model), IUR can
be computed directly without bootstrap resampling.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from ...base import ProviderModel

from ._core import ArrayLike, _compute_decile_table, _iur_decomposition


class DirectIUR(ProviderModel):
    """Direct IUR from pre-computed group-level estimates and SEs.

    This estimator does not require patient-level data.  It computes
    the variance decomposition using the squared standard errors as the
    within-group variance estimates.

    Attributes
    ----------
    iur_ : float
        Overall inter-unit reliability.
    n_groups_ : int
        Number of groups.
    s2_between_ : float
        Between-group variance component (signal).
    s2_within_ : float
        Within-group variance component (noise), scaled by ``n_prime_``.
    n_prime_ : float
        Effective sample size per group.
    sizes_ : ndarray of shape (n_groups_,)
        Group sizes used during fitting.

    Examples
    --------
    >>> from pprof_py.measures.iur import DirectIUR
    >>> model = DirectIUR()
    >>> model.fit(sizes=counts, estimates=pie_j, standard_errors=se_pie_j)
    >>> print(f"IUR = {model.iur_:.4f}")
    """

    def fit(
        self,
        sizes: ArrayLike,
        estimates: ArrayLike,
        standard_errors: ArrayLike,
    ) -> "DirectIUR":
        """Compute IUR from group-level summaries.

        Parameters
        ----------
        sizes : array-like of shape (n_groups,)
            Number of observations per group.
        estimates : array-like of shape (n_groups,)
            Group-level measure estimates (e.g., adjusted rates).
        standard_errors : array-like of shape (n_groups,)
            Standard errors of the group-level estimates.

        Returns
        -------
        self
            Fitted estimator.
        """
        sizes = np.asarray(sizes, dtype=np.float64)
        estimates = np.asarray(estimates, dtype=np.float64)
        se = np.asarray(standard_errors, dtype=np.float64)

        if not (len(sizes) == len(estimates) == len(se)):
            raise ValueError(
                "sizes, estimates, and standard_errors must have the "
                "same length."
            )

        within_variances = se**2
        decomp = _iur_decomposition(sizes, within_variances, estimates)

        self.iur_ = decomp.iur
        self.n_groups_ = decomp.n_groups
        self.s2_between_ = decomp.s2_between
        self.s2_within_ = decomp.s2_within
        self.n_prime_ = decomp.n_prime
        self.sizes_ = sizes

        return self

    def decile_table(self, n_quantiles: int = 10) -> pd.DataFrame:
        """Compute IUR at representative group sizes.

        Uses the fitted variance components to compute the expected
        reliability for groups of different sizes (min, decile
        averages, max).

        Parameters
        ----------
        n_quantiles : int, default=10
            Number of quantile groups.

        Returns
        -------
        table : DataFrame
            Single-row frame with IUR at each representative size.
        """
        self._require_fitted("iur_")
        return _compute_decile_table(
            self.sizes_,
            self.s2_between_,
            self.s2_within_,
            n_quantiles=n_quantiles,
        )
