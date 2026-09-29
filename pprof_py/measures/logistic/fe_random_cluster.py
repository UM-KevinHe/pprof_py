"""Standardized measures for `LogisticFERandomClusterModel`: direct and indirect
standardized rates and ratios. Provider-effect tests and confidence intervals
are in `pprof_py.inference.logistic.fe_random_cluster`. Mixed into
the model class so that `models/logistic/fe_random_cluster.py` can stay
focused on configuration, fitting, and prediction.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import expit as plogis



class LogisticFERandomClusterMeasuresMixin:
    """Standardized measures and provider-effect hypothesis tests for
    `LogisticFERandomClusterModel`."""

    def calculate_standardized_measures(
        self,
        providers: Optional[Union[List, np.ndarray]] = None,
        stdz: Union[str, List[str]] = "indirect",
        reference: Union[str, float] = "median",
    ) -> dict:
        """Calculate indirect/direct standardized ratios and rates.

        Matches the API of ``LogisticFixedEffectModel`` and
        ``LogisticRandomEffectModel.calculate_standardized_measures()``.

        For indirect standardization (SRR):
            expected_k = sum(expit(gamma_null + alpha_mean_i + xbeta_i))
            indirect_ratio = observed_k / expected_k
            indirect_rate  = indirect_ratio * population_rate

        For direct standardization:
            predicted_k = sum(expit(gamma_k + alpha_mean_i + xbeta_i))
                          across ALL observations
            direct_ratio = predicted_k / total_observed
            direct_rate  = predicted_k / N * 100

        Parameters
        ----------
        providers : list or np.ndarray, optional
            Subset of provider IDs. If None, all providers are included.
        stdz : str or list of str, default="indirect"
            Standardization method(s): "indirect" and/or "direct".
        reference : {'median', 'mean'} or float, default="median"
            Null value for gamma in expected computation (indirect)
            and population norm (direct).
            - 'median': uses median(gamma)
            - 'mean': uses mean(gamma)
            - float: uses this value directly

        Returns
        -------
        dict
            Keys for each requested standardization:
            - 'indirect' -> DataFrame [provider_id, indirect_ratio,
              indirect_rate, observed, expected]
            - 'direct' -> DataFrame [provider_id, direct_ratio,
              direct_rate, observed, expected]
        """
        self._check_is_fitted()

        if isinstance(stdz, str):
            stdz = [stdz]
        if not any(m in stdz for m in ["indirect", "direct"]):
            raise ValueError("stdz must include 'indirect' and/or 'direct'.")

        # Determine null gamma
        if reference == "median":
            gamma_null = float(np.median(self.gamma_))
        elif reference == "mean":
            gamma_null = float(np.mean(self.gamma_))
        elif isinstance(reference, (int, float)):
            gamma_null = float(reference)
        else:
            raise ValueError("null must be 'median', 'mean', or a numeric value.")

        prov_idx = self._provider_idx
        n_obs = len(self._obs)
        results = {}

        # --- Indirect standardization ---
        if "indirect" in stdz:
            exp_prob = plogis(gamma_null + self.alpha_mean_ + self.xbeta_)
            expected = np.bincount(prov_idx, weights=exp_prob,
                                   minlength=self.n_providers_)
            observed = np.bincount(prov_idx, weights=self._obs,
                                   minlength=self.n_providers_)

            indirect_ratio = np.where(
                expected > 0, observed / expected, np.nan
            )
            population_rate = observed.sum() / n_obs * 100.0
            indirect_rate = np.clip(
                indirect_ratio * population_rate, 0.0, 100.0
            )

            indirect_df = pd.DataFrame({
                "provider_id": self.provider_ids_,
                "indirect_ratio": indirect_ratio,
                "indirect_rate": indirect_rate,
                "observed": observed,
                "expected": expected,
            })

            if providers is not None:
                indirect_df = indirect_df[
                    indirect_df["provider_id"].isin(np.asarray(providers))
                ].reset_index(drop=True)
            results["indirect"] = indirect_df

        # --- Direct standardization ---
        if "direct" in stdz:
            obs_total = float(np.sum(self._obs))
            population_rate = obs_total / n_obs * 100.0

            direct_preds = np.empty(self.n_providers_)
            for j in range(self.n_providers_):
                p_j = plogis(
                    self.gamma_[j] + self.alpha_mean_ + self.xbeta_
                )
                direct_preds[j] = float(np.sum(p_j))

            direct_ratio = (
                direct_preds / obs_total if obs_total > 0
                else np.full(self.n_providers_, np.nan)
            )
            direct_rate = np.clip(
                direct_preds / n_obs * 100.0, 0.0, 100.0
            )

            direct_df = pd.DataFrame({
                "provider_id": self.provider_ids_,
                "direct_ratio": direct_ratio,
                "direct_rate": direct_rate,
                "observed": np.full(self.n_providers_, obs_total),
                "expected": direct_preds,
            })

            if providers is not None:
                direct_df = direct_df[
                    direct_df["provider_id"].isin(np.asarray(providers))
                ].reset_index(drop=True)
            results["direct"] = direct_df

        return results

    #: Gauss-Hermite nodes for the posterior mixtures in ``test()``.
    _POSTERIOR_NODES = 32
