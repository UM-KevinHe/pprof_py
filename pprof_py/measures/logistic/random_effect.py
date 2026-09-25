"""Standardized measures for `LogisticRandomEffectModel`: direct and
indirect standardized ratios and rates. Provider-effect tests and
confidence intervals are in `pprof_py.inference.logistic.random_effect`. Mixed into the model class so that
`models/logistic/random_effect.py` can stay focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import expit

Array = np.ndarray


class LogisticRandomEffectMeasuresMixin:
    """Standardized measures, provider-level confidence intervals, and
    provider-effect hypothesis tests for `LogisticRandomEffectModel`."""

    def calculate_standardized_measures(
        self,
        providers: Optional[Union[List, Array]] = None,
        stdz: Union[str, List[str]] = "indirect",
        reference: Union[str, float] = "median",
    ) -> dict:
        """Calculate indirect/direct standardized ratios and rates.

        Matches the API of ``LogisticFixedEffectModel.calculate_standardized_measures()``.

        For indirect standardization:
            expected_k = sum(expit(gamma_null + xbeta_i)) for obs in group k
            indirect_ratio = observed_k / expected_k
            indirect_rate  = indirect_ratio * population_rate

        For direct standardization:
            predicted_k = sum(expit(blup_k + xbeta_i)) across ALL observations
            direct_ratio = predicted_k / total_observed
            direct_rate  = predicted_k / N * 100

        Parameters
        ----------
        providers : list or np.ndarray, optional
            Subset of provider/group IDs. If None, all are included.
        stdz : str or list of str, default="indirect"
            Standardization method(s): "indirect" and/or "direct".
        reference : {'median', 'mean'} or float, default="median"
            Baseline for expected counts (indirect) and population norm (direct).
            - 'median': uses median of BLUPs
            - 'mean': uses mean of BLUPs
            - float: uses this value as the null random effect

        Returns
        -------
        dict
            Keys for each requested standardization:
            - 'indirect' -> DataFrame [provider_id, indirect_ratio, indirect_rate,
              observed, expected]
            - 'direct' -> DataFrame [provider_id, direct_ratio, direct_rate,
              observed, expected]
        """
        group_var = self._provider_var
        self._check_is_fitted()

        if isinstance(stdz, str):
            stdz = [stdz]
        if not any(m in stdz for m in ["indirect", "direct"]):
            raise ValueError("stdz must include 'indirect' and/or 'direct'.")


        k = self._group_vars.index(group_var)
        idx = self._group_indices[k]
        labels = self._group_labels[k]
        n_levels = self._n_groups[k]
        blups = self.get_random_effects(group_var).values

        # Determine null BLUP value
        if reference == "median":
            gamma_null = float(np.median(blups))
        elif reference == "mean":
            gamma_null = float(np.mean(blups))
        elif isinstance(reference, (int, float)):
            gamma_null = float(reference)
        else:
            raise ValueError("null must be 'median', 'mean', or a numeric value.")

        # xbeta is offset + X@beta (no random effects)
        xbeta = self.xbeta_
        n_samples = len(self._y)

        results = {}

        # --- Indirect standardization ---
        if "indirect" in stdz:
            # Expected under null BLUP for each obs within that group
            p_null = expit(gamma_null + xbeta)
            expected_by_group = np.bincount(idx, weights=p_null, minlength=n_levels)
            observed_by_group = np.bincount(idx, weights=self._y, minlength=n_levels)

            indirect_ratio = np.where(
                expected_by_group > 0,
                observed_by_group / expected_by_group,
                np.nan,
            )
            population_rate = observed_by_group.sum() / n_samples * 100.0
            indirect_rate = np.clip(indirect_ratio * population_rate, 0.0, 100.0)

            indirect_df = pd.DataFrame({
                "provider_id": labels,
                "indirect_ratio": indirect_ratio,
                "indirect_rate": indirect_rate,
                "observed": observed_by_group,
                "expected": expected_by_group,
            })

            if providers is not None:
                indirect_df = indirect_df[
                    indirect_df["provider_id"].isin(np.asarray(providers))
                ].reset_index(drop=True)
            results["indirect"] = indirect_df

        # --- Direct standardization ---
        if "direct" in stdz:
            obs_total = float(np.sum(self._y))
            population_rate = obs_total / n_samples * 100.0

            direct_preds = np.empty(n_levels)
            for j in range(n_levels):
                p_j = expit(blups[j] + xbeta)
                direct_preds[j] = float(np.sum(p_j))

            direct_ratio = direct_preds / obs_total if obs_total > 0 else np.full(n_levels, np.nan)
            direct_rate = np.clip(direct_preds / n_samples * 100.0, 0.0, 100.0)

            direct_df = pd.DataFrame({
                "provider_id": labels,
                "direct_ratio": direct_ratio,
                "direct_rate": direct_rate,
                "observed": np.full(n_levels, obs_total),
                "expected": direct_preds,
            })

            if providers is not None:
                direct_df = direct_df[
                    direct_df["provider_id"].isin(np.asarray(providers))
                ].reset_index(drop=True)
            results["direct"] = direct_df

        return results
