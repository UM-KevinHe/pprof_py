"""The three-stage logistic provider model (He et al. 2013), as R's ``glmm.fac.hosp``.

Providers (for example facilities) share clusters (for example hospitals), and the
clusters carry their own effects. The model estimates fixed provider effects with
random cluster effects in three stages:

1. **Stage 1**: :class:`LogisticFixedEffectModel` with one effect per provider x cluster
   cell, on the cells with more than ``cutoff`` records and the raw outcome, gives the
   covariate effects beta.
2. **Stage 2**: :class:`LogisticRandomEffectModel` with crossed random intercepts for
   providers and clusters and the offset ``x @ beta``, on the adjusted outcome, gives the
   cluster SD sigma, the providers' BLUPs and the intercept.
3. **Stage 3**: :class:`LogisticFERandomClusterModel` estimates the provider effects with
   beta and sigma held fixed, starting from Stage 2's provider effects plus its intercept.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ...base import ProviderModel
from ...data.glmm_prep import GLMMPreparedData, glmm_data_prep
from .fe_random_cluster import LogisticFERandomClusterModel
from .fixed_effect import LogisticFixedEffectModel
from .random_effect import LogisticRandomEffectModel

OFFSET = "stage1_offset"


class LogisticThreeStageModel(ProviderModel):
    """Three-stage logistic model: fixed provider effects, random cluster effects.

    Parameters
    ----------
    cutoff : int, default 10
        Providers, and Stage 1's provider x cluster cells, need more than ``cutoff``
        records (see :func:`~pprof_py.data.glmm_data_prep`).
    n_nodes, max_iter, tol, bound, bound_mode, convergence_criterion, estimator
        Stage 3's settings (see :class:`LogisticFERandomClusterModel`).

    Attributes
    ----------
    prep_ : GLMMPreparedData
        The prepared data; ``data_`` is ``prep_.data`` with the Stage 1 offset column
        ``stage1_offset``.
    stage1_ : LogisticFixedEffectModel
    stage2_ : LogisticRandomEffectModel
    stage3_ : LogisticFERandomClusterModel
        The fitted stages. ``test()``, ``calculate_standardized_measures()``,
        ``calculate_confidence_intervals()`` and ``summary()`` are Stage 3's.
    """

    def __init__(
        self,
        cutoff: int = 10,
        n_nodes: int = 20,
        max_iter: int = 10000,
        tol: float = 1e-5,
        bound: float = 10.0,
        bound_mode: str = "relative",
        convergence_criterion: str = "max_delta_gamma",
        estimator: str = "he2013",
    ):
        self.cutoff = cutoff
        self.n_nodes = n_nodes
        self.max_iter = max_iter
        self.tol = tol
        self.bound = bound
        self.bound_mode = bound_mode
        self.convergence_criterion = convergence_criterion
        self.estimator = estimator
        self.prep_: Optional[GLMMPreparedData] = None
        self.data_: Optional[pd.DataFrame] = None
        self.stage1_: Optional[LogisticFixedEffectModel] = None
        self.stage2_: Optional[LogisticRandomEffectModel] = None
        self.stage3_: Optional[LogisticFERandomClusterModel] = None

    def fit(self, data: pd.DataFrame, y_var: str, x_vars: List[str], provider_var: str, cluster_var: str,
            verbose: bool = False) -> "LogisticThreeStageModel":
        """Prepare the data and fit the three stages.

        Parameters
        ----------
        data : pandas.DataFrame
            One row per record, with the binary outcome, the covariates and the provider
            and cluster IDs.
        y_var : str
            Binary (0/1) outcome.
        x_vars : list of str
            Covariates.
        provider_var, cluster_var : str
            Provider (facility) and cluster (hospital) IDs.
        verbose : bool, default False
            Print Stage 3's iteration progress.

        Returns
        -------
        self
        """
        x_vars = list(x_vars)
        if OFFSET in data.columns:
            raise ValueError(f"data must not have a column named {OFFSET!r}; the model adds it.")
        prep = glmm_data_prep(data, y_var, provider_var, cluster_var, cutoff=self.cutoff)
        d = prep.data
        stage1 = LogisticFixedEffectModel(use_dataprep=False, screen_providers=False)
        stage1.fit(X=d[d["included"] == 1], y_var=y_var, x_vars=x_vars, provider_var="cell_id")
        beta = pd.Series(np.asarray(stage1.coefficients_["beta"], dtype=np.float64).ravel(),
                         index=stage1.covariate_names_).reindex(x_vars).to_numpy()
        d = d.assign(**{OFFSET: d[x_vars].to_numpy(dtype=np.float64) @ beta})
        stage2 = LogisticRandomEffectModel(verbose=False)
        stage2.fit(d, y_var="y_adj", x_vars=None, provider_var=provider_var, cluster_vars=[cluster_var],
                   offset_var=OFFSET, verbose=False)
        stage3 = LogisticFERandomClusterModel(n_nodes=self.n_nodes, max_iter=self.max_iter, tol=self.tol,
                                              bound=self.bound, bound_mode=self.bound_mode,
                                              convergence_criterion=self.convergence_criterion,
                                              estimator=self.estimator)
        stage3.fit(d, "y_adj", x_vars, provider_var, cluster_var, stage1=stage1, stage2=stage2, obs_var=y_var,
                   verbose=verbose)
        self.prep_, self.data_ = prep, d
        self.stage1_, self.stage2_, self.stage3_ = stage1, stage2, stage3
        return self

    def _stage3(self) -> LogisticFERandomClusterModel:
        if self.stage3_ is None:
            raise ValueError("The model must be fitted first.")
        return self.stage3_

    def test(self, *args, **kwargs) -> pd.DataFrame:
        """Stage 3's provider tests; see :meth:`LogisticFERandomClusterModel.test`."""
        return self._stage3().test(*args, **kwargs)

    def calculate_standardized_measures(self, *args, **kwargs):
        """Stage 3's standardized measures; see :meth:`LogisticFERandomClusterModel.calculate_standardized_measures`."""
        return self._stage3().calculate_standardized_measures(*args, **kwargs)

    def calculate_confidence_intervals(self, *args, **kwargs):
        """Stage 3's intervals; see :meth:`LogisticFERandomClusterModel.calculate_confidence_intervals`."""
        return self._stage3().calculate_confidence_intervals(*args, **kwargs)

    def summary(self, *args, **kwargs) -> pd.DataFrame:
        """The covariate Wald table of Stage 1; see :meth:`LogisticFERandomClusterModel.summary`."""
        return self._stage3().summary(*args, **kwargs)
