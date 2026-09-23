"""Provider standardization and measure-specific workflows for
`LogisticMixedEffectModel`: standardized rates/ratios (direct/indirect)
and provider-effect significance testing (resampling or exact
Poisson-binomial, with optional empirical-null calibration). Mixed into
the model class so that `models/logistic/mixed_effect.py` can stay
focused on configuration, fitting, and prediction.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import expit as plogis

from ...inference.effect_tests import (EXACT_P_FLOOR, effect_test, normalize_alternative, poibin_tails,
                                       reference_effect, resample_tails, z_from_tails)


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
        providers=None,
        *,
        test_method: str = "resampling",
        reference="median",
        null_model=None,
        alternative: str = "two_sided",
        level: float = 0.95,
        critical: Optional[float] = None,
        n_resample: int = 10000,
        seed=None,
    ) -> pd.DataFrame:
        """Test each provider's Stage 3 effect against the reference effect gamma_0.

        Parameters
        ----------
        providers : array-like, optional
            Report only these providers; gamma_0 and any empirical null use all.
        test_method : {"resampling", "poibin_exact"}
            ``"resampling"``: the provider's event count against simulated
            counts with its effect at gamma_0 and the cluster effects drawn
            from their posterior (He et al. 2013, Section 3.3).
            ``"poibin_exact"``: exact Poisson-binomial test with the cluster
            effects at their posterior means.
        reference : "median", "mean", or float
            The reference effect gamma_0: the median of the estimated effects,
            their size-weighted mean, or a value on the effect scale.
        null_model : NullModel or callable, optional
            Null for the z-statistics: :class:`~pprof_py.inference.TheoreticalNull`
            by default, or an instance such as ``FixedNull(sd=...)``, or a callable
            that receives the z-statistics, such as ``EmpiricalNull.fitter(...)``.
        alternative, level, critical
            As in :func:`~pprof_py.inference.provider_test`.
            He et al.'s empirical-null step is
            ``null_model=EmpiricalNull.fitter(size=sizes, n_groups=4, grouping="rank",
            estimator=HUBER_RLM, small_group="theoretical")``.
        n_resample, seed : int, optional
            Monte Carlo draws and seed for ``"resampling"``.

        Returns
        -------
        pandas.DataFrame
            Indexed by provider with columns
            :data:`~pprof_py.inference.PROVIDER_TEST_COLUMNS`: ``flag`` is +1
            above gamma_0, -1 below, 0 not significant, NA not tested.
        """
        self._check_is_fitted()
        alt = normalize_alternative(alternative)
        if test_method not in ("resampling", "poibin_exact"):
            raise ValueError(f"test_method={test_method!r} is not supported; use 'resampling' or 'poibin_exact'.")
        idx = np.asarray(self._provider_idx).ravel()
        g0 = reference_effect(self.gamma_, np.bincount(idx, minlength=self.n_providers_), reference)
        order = np.argsort(idx, kind="stable")
        edges = np.r_[0, np.cumsum(np.bincount(idx, minlength=self.n_providers_))]
        rng = np.random.default_rng(seed) if test_method == "resampling" else None
        tails = np.empty((self.n_providers_, 4))
        for j in range(self.n_providers_):
            rows = order[edges[j]:edges[j + 1]]
            obs = self._obs[rows].sum()
            if test_method == "poibin_exact":
                tails[j] = poibin_tails(obs, plogis(g0 + self.alpha_mean_[rows] + self.xbeta_[rows]))
            else:
                tails[j] = resample_tails(obs, self.xbeta_[rows], self.alpha_mean_[rows], self.alpha_var_[rows],
                                          g0, n_resample, rng)
        two = alt == "two_sided"
        z = z_from_tails(tails[:, 0] if two else tails[:, 2], tails[:, 1] if two else tails[:, 3], alt,
                         EXACT_P_FLOOR if test_method == "poibin_exact" else 0.5 / n_resample)
        return effect_test(self.provider_ids_, self.gamma_, z, g0, null_model=null_model, alternative=alt,
                           level=level, critical=critical, providers=providers, test_method=test_method)
