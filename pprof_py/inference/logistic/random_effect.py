"""Covariate-level inference and posterior BLUP variance/SE for
``LogisticRandomEffectModel``.  Mixed into the model class so that
``models/logistic/random_effect.py`` stays focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

Array = np.ndarray


class RandomEffectInferenceMixin:
    """Fixed-effect summary and posterior BLUP variance/SE for
    `LogisticRandomEffectModel`."""

    def summary(self) -> pd.DataFrame:
        self._check_is_fitted()
        fe = self.coefficients_["beta"]
        vcov = self.variances_["beta"]
        se = np.sqrt(np.maximum(np.diag(vcov.to_numpy()), 0.0))
        z = np.divide(fe.to_numpy(), se, out=np.full_like(fe.to_numpy(), np.nan), where=se > 0)
        p = 2.0 * norm.sf(np.abs(z))
        return pd.DataFrame(
            {"Estimate": fe.to_numpy(), "Std.Error": se, "z value": z, "Pr(>|z|)": p},
            index=fe.index,
        )

    # ------------------------------------------------------------------
    # Posterior standard errors (conditional variance of BLUPs)
    # ------------------------------------------------------------------

    def _get_posterior_var(self, group_var: Optional[str] = None) -> Array:
        """Posterior variances of the random-effect BLUPs.

        For the spherical parameterization u ~ N(0, I) with b = sigma * u,
        the conditional variance of b given y is approximately:

            Var(b | y) = sigma^2 * diag(H^{-1})

        where H = I + A' W A is the conditional random-effect Hessian
        from the final PIRLS iteration.

        Parameters
        ----------
        group_var : str, optional
            Which grouping factor's posterior variances to return.
            If None, returns all (concatenated).

        Returns
        -------
        np.ndarray
            Posterior variances for each level of the group.
        """
        self._check_is_fitted()

        # Rebuild H at final estimates
        b = self._u_to_random_effects(self._sigma, self._u)
        eta = self._eta(self._beta, b)
        mu = expit(eta)
        w = np.maximum(mu * (1.0 - mu), 1e-12)
        H, _ = self._build_H_C(self._sigma, w)

        # Compute diagonal of H^{-1} via sparse LU, batched across all unit
        # vector columns at once (mathematically identical to solving one
        # column at a time, since `lu` is a single fixed factorization).
        H_csc = H.tocsc()
        from scipy.sparse.linalg import splu
        lu = splu(H_csc)
        Hinv_cols = lu.solve(np.eye(self._q))
        diag_Hinv = np.diag(Hinv_cols)

        # Var(b_k[j]) = sigma_k^2 * diag_Hinv[j_idx]
        if group_var is not None:
            k = self._group_vars.index(group_var)
            sl = self._q_slices[k]
            sigma_k = self._sigma[k]
            return sigma_k**2 * diag_Hinv[sl]

        # Return all
        posterior_vars = np.empty(self._q)
        for k, gv in enumerate(self._group_vars):
            sl = self._q_slices[k]
            posterior_vars[sl] = self._sigma[k] ** 2 * diag_Hinv[sl]
        return posterior_vars

    def _get_posterior_se(self, group_var: Optional[str] = None) -> pd.Series:
        """Posterior standard errors of BLUPs for a grouping factor.

        Parameters
        ----------
        group_var : str, optional
            Which grouping factor. If None and only one exists, uses that.

        Returns
        -------
        pd.Series
            Standard errors indexed by group level labels.
        """
        if group_var is None:
            if len(self._group_vars) == 1:
                group_var = self._group_vars[0]
            else:
                raise ValueError(
                    f"Specify group_var; available: {self._group_vars}"
                )
        k = self._group_vars.index(group_var)
        pvar = self._get_posterior_var(group_var=group_var)
        se = np.sqrt(np.maximum(pvar, 0.0))
        return pd.Series(se, index=self._group_labels[k], name="posterior_se")
