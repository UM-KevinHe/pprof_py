"""Provider standardization and measure-specific workflows for
`LogisticMixedEffectModel`: standardized rates/ratios (direct/indirect)
and provider-effect significance testing (resampling or exact
Poisson-binomial, with optional empirical-null calibration). Mixed into
the model class so that `models/logistic/mixed_effect.py` can stay
focused on configuration, fitting, and prediction (see AGENTS.md
Sections 9 and 16).
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import expit as plogis

from ...inference.empirical_null import (
    poibin_exact_pvalue,
    resample_pvalue,
    pvalues_to_zscores,
    calibrate_empirical_null,
    assign_flags,
)


class MixedEffectMeasuresMixin:
    """Standardized measures and provider-effect hypothesis tests for
    `LogisticMixedEffectModel`."""

    def calculate_standardized_measures(
        self,
        providers: Optional[Union[List, np.ndarray]] = None,
        stdz: Union[str, List[str]] = "indirect",
        null: Union[str, float] = "median",
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
        null : {'median', 'mean'} or float, default="median"
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
        if null == "median":
            gamma_null = float(np.median(self.gamma_))
        elif null == "mean":
            gamma_null = float(np.mean(self.gamma_))
        elif isinstance(null, (int, float)):
            gamma_null = float(null)
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

    def test(
        self,
        test_method: str = 'resampling',
        n_resample: int = 10000,
        empirical_null: bool = True,
        n_strata: int = 4,
        strata_var: Optional[np.ndarray] = None,
        level: float = 0.95,
        seed: int = 1,
    ) -> pd.DataFrame:
        """Significance test for provider effects.

        Supported test methods:
        - "resampling": Parametric bootstrap (He et al. 2013) incorporating
          posterior uncertainty of cluster random effects.
        - "poibin_exact": Exact Poisson-binomial test using point estimates
          (posterior mean) for cluster effects.  Deterministic — no Monte
          Carlo error or seed dependence.

        Both methods share the same post-processing pipeline:
        1. Compute two-sided p-value per provider.
        2. Convert p-values to z-scores.
        3. (Optional) Apply empirical null calibration via Huber M-estimator,
           stratified by provider size or a user-supplied grouping variable.

        Parameters
        ----------
        test_method : {'resampling', 'poibin_exact'}, default='resampling'
            Inference method.  ``'poibin_exact'`` requires the ``fast_poibin``
            package.  ``n_resample`` and ``seed`` are ignored when using
            ``'poibin_exact'``.
        n_resample : int, default=10000
            Number of Monte Carlo resamples per provider (resampling only).
        empirical_null : bool, default=True
            Whether to apply empirical null calibration.
        n_strata : int, default=4
            Number of strata (quantiles) for empirical null grouping.
        strata_var : np.ndarray, optional
            Provider-level variable for stratification (e.g., facility size).
            If None, uses provider sample size (number of observations).
        level : float, default=0.95
            Confidence level for flagging (significance level = 1 - level).
        seed : int, default=1
            Random seed for reproducibility (resampling only).

        Returns
        -------
        pd.DataFrame with columns:
            provider_id, gamma, srr, obs, exp, p_theo, z_score,
            p_empi (if empirical_null), flag
        """
        self._check_is_fitted()

        if test_method not in ('resampling', 'poibin_exact'):
            raise ValueError(
                f"test_method='{test_method}' is not supported. "
                "Use 'resampling' or 'poibin_exact'."
            )

        prov_idx = self._provider_idx
        gamma_median = np.median(self.gamma_)
        sm = self.calculate_standardized_measures()
        indirect = sm['indirect']
        srr = indirect['indirect_ratio'].values

        # Build per-observation arrays sorted by provider
        sort_order = np.argsort(prov_idx)
        y_sorted = self._obs[sort_order]
        xbeta_sorted = self.xbeta_[sort_order]
        alpha_mean_sorted = self.alpha_mean_[sort_order]
        alpha_var_sorted = self.alpha_var_[sort_order]

        # Provider boundaries
        prov_sizes = np.bincount(prov_idx, minlength=self.n_providers_)
        prov_starts = np.concatenate([[0], np.cumsum(prov_sizes[:-1])])

        # Compute p-values per provider
        p_theo = np.zeros(self.n_providers_)

        for j in range(self.n_providers_):
            start = prov_starts[j]
            end = start + prov_sizes[j]
            obs_j = y_sorted[start:end].sum()

            if test_method == 'poibin_exact':
                null_probs = plogis(
                    gamma_median
                    + alpha_mean_sorted[start:end]
                    + xbeta_sorted[start:end]
                )
                p_theo[j] = poibin_exact_pvalue(obs_j, null_probs)
            else:  # resampling
                p_theo[j] = resample_pvalue(
                    obs_sum=obs_j,
                    eta_fixed=xbeta_sorted[start:end],
                    re_mean=alpha_mean_sorted[start:end],
                    re_var=alpha_var_sorted[start:end],
                    null_effect=gamma_median,
                    n_resample=n_resample,
                    seed=seed,
                )

        # Convert to z-scores
        z_score = pvalues_to_zscores(p_theo, srr)

        # Empirical null calibration
        if empirical_null:
            strata = strata_var if strata_var is not None else prov_sizes.astype(float)
            p_empi, _ = calibrate_empirical_null(z_score, strata, n_strata)
        else:
            p_empi = p_theo

        # Flag assignment
        flag = assign_flags(p_empi, srr, 1 - level)

        # Build results
        results = pd.DataFrame({
            'provider_id': self.provider_ids_,
            'gamma': self.gamma_,
            'srr': srr,
            'obs': indirect['observed'].values,
            'exp': indirect['expected'].values,
            'p_theo': p_theo,
            'z_score': z_score,
            'p_empi': p_empi,
            'flag': flag,
        })

        return results
