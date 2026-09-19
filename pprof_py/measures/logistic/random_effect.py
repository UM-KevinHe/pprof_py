"""Provider standardization and measure-specific workflows for
`LogisticRandomEffectModel`: standardized ratios/rates (direct/indirect),
provider-level confidence intervals, and provider-effect hypothesis
testing. Mixed into the model class so that
`models/logistic/random_effect.py` can stay focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

Array = np.ndarray


class RandomEffectMeasuresMixin:
    """Standardized measures, provider-level confidence intervals, and
    provider-effect hypothesis tests for `LogisticRandomEffectModel`."""

    def calculate_standardized_measures(
        self,
        group_var: Optional[str] = None,
        providers: Optional[Union[List, Array]] = None,
        stdz: Union[str, List[str]] = "indirect",
        null: Union[str, float] = "median",
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
        group_var : str, optional
            Grouping variable. If None and only one exists, uses that.
        providers : list or np.ndarray, optional
            Subset of provider/group IDs. If None, all are included.
        stdz : str or list of str, default="indirect"
            Standardization method(s): "indirect" and/or "direct".
        null : {'median', 'mean'} or float, default="median"
            Baseline for expected counts (indirect) and population norm (direct).
            - 'median': uses median of BLUPs
            - 'mean': uses mean of BLUPs
            - float: uses this value as the null random effect

        Returns
        -------
        dict
            Keys for each requested standardization:
            - 'indirect' -> DataFrame [group_id, indirect_ratio, indirect_rate,
              observed, expected]
            - 'direct' -> DataFrame [group_id, direct_ratio, direct_rate,
              observed, expected]
        """
        self._check_is_fitted()

        if isinstance(stdz, str):
            stdz = [stdz]
        if not any(m in stdz for m in ["indirect", "direct"]):
            raise ValueError("stdz must include 'indirect' and/or 'direct'.")

        if group_var is None:
            if len(self._group_vars) == 1:
                group_var = self._group_vars[0]
            else:
                raise ValueError(f"Specify group_var; available: {self._group_vars}")

        k = self._group_vars.index(group_var)
        idx = self._group_indices[k]
        labels = self._group_labels[k]
        n_levels = self._n_groups[k]
        blups = self.get_random_effects(group_var).values

        # Determine null BLUP value
        if null == "median":
            gamma_null = float(np.median(blups))
        elif null == "mean":
            gamma_null = float(np.mean(blups))
        elif isinstance(null, (int, float)):
            gamma_null = float(null)
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
                "group_id": labels,
                "indirect_ratio": indirect_ratio,
                "indirect_rate": indirect_rate,
                "observed": observed_by_group,
                "expected": expected_by_group,
            })

            if providers is not None:
                indirect_df = indirect_df[
                    indirect_df["group_id"].isin(np.asarray(providers))
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
                "group_id": labels,
                "direct_ratio": direct_ratio,
                "direct_rate": direct_rate,
                "observed": np.full(n_levels, obs_total),
                "expected": direct_preds,
            })

            if providers is not None:
                direct_df = direct_df[
                    direct_df["group_id"].isin(np.asarray(providers))
                ].reset_index(drop=True)
            results["direct"] = direct_df

        return results

    # ------------------------------------------------------------------
    # Hypothesis testing (aligned with LogisticFixedEffectModel API)
    # ------------------------------------------------------------------

    def test(
        self,
        group_var: Optional[str] = None,
        providers: Optional[Union[List, Array]] = None,
        level: float = 0.95,
        test_method: str = "wald",
        null: Union[str, float] = "median",
        alternative: str = "two_sided",
        n_resample: int = 10000,
        empirical_null: bool = False,
        n_strata: int = 4,
        strata_var: Optional[Array] = None,
        seed: int = 1,
    ) -> pd.DataFrame:
        """Test random effects (BLUPs) for significance.

        Supported test methods:
        - "wald": Z-test using posterior standard errors from condVar.
        - "resampling": Parametric bootstrap (He et al. 2013) with
          optional empirical null calibration.
        - "poibin_exact": Exact Poisson-binomial test (deterministic,
          no Monte Carlo error). Uses point estimates for other REs.

        Parameters
        ----------
        group_var : str, optional
            Grouping variable to test. If None and only one exists, uses that.
        providers : list or np.ndarray, optional
            Subset of provider/group IDs to test. If None, all are tested.
        level : float, default 0.95
            Confidence level => alpha = 1 - level is significance threshold.
        test_method : {'wald', 'resampling', 'poibin_exact'}, default='wald'
            Testing approach.
        null : {'median', 'mean'} or float, default='median'
            Null hypothesis value for the random effect.
        alternative : {'two_sided', 'greater', 'less'}, default='two_sided'
            Direction of the test (only for 'wald' method).
        n_resample : int, default=10000
            Number of bootstrap resamples (only for 'resampling' method).
        empirical_null : bool, default=False
            Apply empirical null calibration (only for 'resampling' method).
        n_strata : int, default=4
            Number of strata for empirical null grouping.
        strata_var : np.ndarray, optional
            Provider-level variable for stratification. If None, uses
            provider sample size.
        seed : int, default=1
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Indexed by group_id, with columns depending on test_method.
        """
        from ...inference.empirical_null import (
            resample_pvalue,
            poibin_exact_pvalue,
            pvalues_to_zscores,
            calibrate_empirical_null,
            assign_flags,
        )

        self._check_is_fitted()
        if test_method not in ("wald", "resampling", "poibin_exact"):
            raise ValueError(
                f"test_method='{test_method}' is not supported. "
                "Use 'wald', 'resampling', or 'poibin_exact'."
            )

        if group_var is None:
            if len(self._group_vars) == 1:
                group_var = self._group_vars[0]
            else:
                raise ValueError(f"Specify group_var; available: {self._group_vars}")

        blups = self.get_random_effects(group_var)
        se = self._get_posterior_se(group_var)

        # Determine null value
        if null == "median":
            gamma_null = float(np.median(blups.values))
        elif null == "mean":
            gamma_null = float(np.mean(blups.values))
        elif isinstance(null, (int, float)):
            gamma_null = float(null)
        else:
            raise ValueError("null must be 'median', 'mean', or a numeric value.")

        alpha = 1.0 - level

        # ---- Wald test ----
        if test_method == "wald":
            if alternative not in ("two_sided", "greater", "less"):
                raise ValueError("alternative must be 'two_sided', 'greater', or 'less'")

            z = (blups.values - gamma_null) / np.maximum(se.values, 1e-15)
            flags = np.zeros(len(z), dtype=int)

            if alternative == "two_sided":
                pvals = 2.0 * norm.sf(np.abs(z))
                flags[pvals < alpha] = np.where(z[pvals < alpha] > 0, 1, -1)
            elif alternative == "greater":
                pvals = norm.sf(z)
                flags[pvals < alpha] = 1
            else:
                pvals = norm.cdf(z)
                flags[pvals < alpha] = -1

            result = pd.DataFrame({
                "flag": pd.Categorical(flags, categories=[-1, 0, 1]),
                "p_value": pvals,
                "stat": z,
                "std_error": se.values,
            }, index=blups.index)
            result.index.name = "group_id"

        # ---- Resampling test ----
        else:  # test_method in ("resampling", "poibin_exact")
            k = self._group_vars.index(group_var)
            idx = self._group_indices[k]
            n_levels = self._n_groups[k]

            # Compute SRR (for direction)
            sm = self.calculate_standardized_measures(
                group_var=group_var, stdz="indirect", null=null
            )
            srr = sm["indirect"]["indirect_ratio"].values

            # Per-observation quantities for bootstrap:
            # eta_fixed = offset + xbeta (everything EXCEPT target group BLUP)
            # For other REs, use their BLUPs as the mean to sample around
            xbeta = self.xbeta_  # offset + X @ beta

            # Build per-obs RE mean/var for other group(s)
            # (If target group is the only group, use zeros)
            other_re_mean = np.zeros(self._n, dtype=float)
            other_re_var = np.zeros(self._n, dtype=float)
            for j, gv in enumerate(self._group_vars):
                if gv == group_var:
                    continue
                # Add other groups' BLUPs as fixed contribution to eta
                other_blups = self.get_random_effects(gv).values
                other_se = self._get_posterior_se(gv).values
                other_re_mean += other_blups[self._group_indices[j]]
                other_re_var += (other_se[self._group_indices[j]]) ** 2

            # Sort by target group for provider-wise iteration
            sort_order = np.argsort(idx)
            y_sorted = self._y[sort_order]
            eta_sorted = xbeta[sort_order]
            re_mean_sorted = other_re_mean[sort_order]
            re_var_sorted = other_re_var[sort_order]
            idx_sorted = idx[sort_order]

            # Provider boundaries
            prov_sizes = np.bincount(idx, minlength=n_levels)
            prov_starts = np.concatenate([[0], np.cumsum(prov_sizes[:-1])])

            # Compute p-values per provider
            p_theo = np.zeros(n_levels)
            for j in range(n_levels):
                start = prov_starts[j]
                end = start + prov_sizes[j]
                obs_j = y_sorted[start:end].sum()

                if test_method == "poibin_exact":
                    null_probs = expit(
                        gamma_null
                        + re_mean_sorted[start:end]
                        + eta_sorted[start:end]
                    )
                    p_theo[j] = poibin_exact_pvalue(obs_j, null_probs)
                else:  # resampling
                    p_theo[j] = resample_pvalue(
                        obs_sum=obs_j,
                        eta_fixed=eta_sorted[start:end],
                        re_mean=re_mean_sorted[start:end],
                        re_var=re_var_sorted[start:end],
                        null_effect=gamma_null,
                        n_resample=n_resample,
                        seed=seed,
                    )

            # Convert to z-scores
            z_score = pvalues_to_zscores(p_theo, srr)

            # Empirical null calibration
            if empirical_null:
                strata = strata_var if strata_var is not None else prov_sizes.astype(float)
                p_final, _ = calibrate_empirical_null(z_score, strata, n_strata)
            else:
                p_final = p_theo

            # Flags
            flags = assign_flags(p_final, srr, alpha)

            result = pd.DataFrame({
                "flag": pd.Categorical(flags, categories=[-1, 0, 1]),
                "p_value": p_final,
                "p_theo": p_theo,
                "z_score": z_score,
                "srr": srr,
            }, index=blups.index)
            result.index.name = "group_id"

        # Filter to requested providers
        if providers is not None:
            providers_set = set(np.asarray(providers))
            result = result[result.index.isin(providers_set)]

        return result

    # ------------------------------------------------------------------
    # Confidence intervals (aligned with LogisticFixedEffectModel API)
    # ------------------------------------------------------------------

    def calculate_confidence_intervals(
        self,
        group_var: Optional[str] = None,
        providers: Optional[Union[List, Array]] = None,
        level: float = 0.95,
        option: str = "SM",
        stdz: Union[str, List[str]] = "indirect",
        null: Union[str, float] = "median",
        measure: Union[str, List[str]] = ("rate", "ratio"),
        alternative: str = "two_sided",
    ) -> dict:
        """Compute confidence intervals for BLUPs or standardized measures.

        Matches the API of ``LogisticFixedEffectModel.calculate_confidence_intervals()``.

        Parameters
        ----------
        group_var : str, optional
            Grouping variable. If None and only one exists, uses that.
        providers : list or np.ndarray, optional
            Subset of providers. If None, all are included.
        level : float, default 0.95
            Confidence level.
        option : {'alpha', 'SM'}, default='SM'
            - 'alpha': CIs for BLUPs on the log-odds scale.
            - 'SM': CIs for standardized measures (ratio and/or rate).
        stdz : {'indirect', 'direct'} or list, default='indirect'
            Standardization method(s) if option='SM'. Ignored if option='alpha'.
        null : {'median', 'mean'} or float, default='median'
            Baseline norm for standardization. Ignored if option='alpha'.
        measure : str or list, default=('rate', 'ratio')
            Measures to produce CIs for if option='SM'.
        alternative : {'two_sided', 'greater', 'less'}, default='two_sided'
            Interval type. Must be 'two_sided' if option='alpha'.

        Returns
        -------
        dict
            If option='alpha':
                {'alpha_ci': DataFrame [group_id, alpha, alpha_lower, alpha_upper]}
            If option='SM':
                May include keys: 'indirect_ratio', 'indirect_rate',
                'direct_ratio', 'direct_rate' depending on stdz and measure.
        """
        self._check_is_fitted()
        if option not in ("alpha", "SM"):
            raise ValueError("option must be 'alpha' or 'SM'.")

        if group_var is None:
            if len(self._group_vars) == 1:
                group_var = self._group_vars[0]
            else:
                raise ValueError(f"Specify group_var; available: {self._group_vars}")

        blups = self.get_random_effects(group_var)
        se = self._get_posterior_se(group_var)

        # --- option = 'alpha': CIs for BLUPs on log-odds scale ---
        if option == "alpha":
            if alternative != "two_sided":
                raise ValueError("For option='alpha', only two_sided is supported.")
            z_crit = norm.ppf(1.0 - (1.0 - level) / 2.0)
            ci_lower = blups.values - z_crit * se.values
            ci_upper = blups.values + z_crit * se.values

            alpha_ci = pd.DataFrame({
                "group_id": blups.index,
                "alpha": blups.values,
                "alpha_lower": ci_lower,
                "alpha_upper": ci_upper,
            })
            if providers is not None:
                alpha_ci = alpha_ci[
                    alpha_ci["group_id"].isin(np.asarray(providers))
                ].reset_index(drop=True)
            return {"alpha_ci": alpha_ci}

        # --- option = 'SM': CIs for standardized measures ---
        if isinstance(stdz, str):
            stdz = [stdz]
        if isinstance(measure, str):
            measure = [measure]

        # Determine null BLUP
        if null == "median":
            gamma_null = float(np.median(blups.values))
        elif null == "mean":
            gamma_null = float(np.mean(blups.values))
        elif isinstance(null, (int, float)):
            gamma_null = float(null)
        else:
            raise ValueError("null must be 'median', 'mean', or numeric.")

        # Compute BLUP CI bounds (on log-odds scale)
        alpha_val = 1.0 - level
        if alternative == "two_sided":
            z_crit = norm.ppf(1.0 - alpha_val / 2.0)
            blup_lower = blups.values - z_crit * se.values
            blup_upper = blups.values + z_crit * se.values
        elif alternative == "greater":
            z_crit = norm.ppf(1.0 - alpha_val)
            blup_lower = blups.values - z_crit * se.values
            blup_upper = np.full_like(blups.values, np.inf)
        else:  # less
            z_crit = norm.ppf(1.0 - alpha_val)
            blup_lower = np.full_like(blups.values, -np.inf)
            blup_upper = blups.values + z_crit * se.values

        k = self._group_vars.index(group_var)
        idx = self._group_indices[k]
        n_levels = self._n_groups[k]
        labels = self._group_labels[k]
        xbeta = self.xbeta_
        n_samples = len(self._y)
        obs_total = float(np.sum(self._y))
        population_rate = obs_total / n_samples * 100.0

        results = {}

        # --- Indirect SM CIs ---
        if "indirect" in stdz:
            # Expected under null for each group's own observations
            p_null = expit(gamma_null + xbeta)
            expected_by_group = np.bincount(idx, weights=p_null, minlength=n_levels)
            observed_by_group = np.bincount(idx, weights=self._y, minlength=n_levels)

            # Transform BLUP CI bounds → indirect ratio CI
            # ratio = obs / expected(gamma); as gamma increases, expected increases
            # Compute expected at lower and upper BLUP bounds
            ratio_lower = np.empty(n_levels)
            ratio_upper = np.empty(n_levels)
            for j in range(n_levels):
                mask_j = (idx == j)
                xbeta_j = xbeta[mask_j]
                obs_j = observed_by_group[j]
                # Lower bound of gamma → lower expected → HIGHER ratio
                # Upper bound of gamma → higher expected → LOWER ratio
                # But for indirect: expected uses gamma_null, not the provider's gamma
                # The BLUP CI translates via Delta method on log(SR)
                exp_j = expected_by_group[j]
                if exp_j > 0 and obs_j > 0:
                    sr_j = obs_j / exp_j
                    log_sr = np.log(sr_j)
                    se_j = se.values[j]
                    ratio_lower[j] = np.exp(log_sr - z_crit * se_j)
                    ratio_upper[j] = np.exp(log_sr + z_crit * se_j)
                else:
                    ratio_lower[j] = np.nan
                    ratio_upper[j] = np.nan

            if "ratio" in measure:
                sr = np.where(
                    expected_by_group > 0,
                    observed_by_group / expected_by_group,
                    np.nan,
                )
                df_ir = pd.DataFrame({
                    "group_id": labels,
                    "indirect_ratio": sr,
                    "lower": ratio_lower,
                    "upper": ratio_upper,
                })
                if providers is not None:
                    df_ir = df_ir[df_ir["group_id"].isin(np.asarray(providers))].reset_index(drop=True)
                results["indirect_ratio"] = df_ir

            if "rate" in measure:
                sr = np.where(
                    expected_by_group > 0,
                    observed_by_group / expected_by_group,
                    np.nan,
                )
                rate = np.clip(sr * population_rate, 0.0, 100.0)
                rate_lower = np.clip(ratio_lower * population_rate, 0.0, 100.0)
                rate_upper = np.clip(ratio_upper * population_rate, 0.0, 100.0)
                df_irate = pd.DataFrame({
                    "group_id": labels,
                    "indirect_rate": rate,
                    "lower": rate_lower,
                    "upper": rate_upper,
                })
                if providers is not None:
                    df_irate = df_irate[df_irate["group_id"].isin(np.asarray(providers))].reset_index(drop=True)
                results["indirect_rate"] = df_irate

        # --- Direct SM CIs ---
        if "direct" in stdz:
            # Direct: predicted_k = sum(expit(blup_k + xbeta_i)) over all obs
            # CI: replace blup_k with blup_lower/upper
            direct_pred = np.empty(n_levels)
            direct_pred_lower = np.empty(n_levels)
            direct_pred_upper = np.empty(n_levels)
            for j in range(n_levels):
                direct_pred[j] = float(np.sum(expit(blups.values[j] + xbeta)))
                direct_pred_lower[j] = float(np.sum(expit(blup_lower[j] + xbeta)))
                direct_pred_upper[j] = float(np.sum(expit(blup_upper[j] + xbeta)))

            if "ratio" in measure:
                dr = direct_pred / obs_total if obs_total > 0 else np.full(n_levels, np.nan)
                dr_lower = direct_pred_lower / obs_total if obs_total > 0 else np.full(n_levels, np.nan)
                dr_upper = direct_pred_upper / obs_total if obs_total > 0 else np.full(n_levels, np.nan)
                df_dr = pd.DataFrame({
                    "group_id": labels,
                    "direct_ratio": dr,
                    "lower": dr_lower,
                    "upper": dr_upper,
                })
                if providers is not None:
                    df_dr = df_dr[df_dr["group_id"].isin(np.asarray(providers))].reset_index(drop=True)
                results["direct_ratio"] = df_dr

            if "rate" in measure:
                drate = np.clip(direct_pred / n_samples * 100.0, 0.0, 100.0)
                drate_lower = np.clip(direct_pred_lower / n_samples * 100.0, 0.0, 100.0)
                drate_upper = np.clip(direct_pred_upper / n_samples * 100.0, 0.0, 100.0)
                df_drate = pd.DataFrame({
                    "group_id": labels,
                    "direct_rate": drate,
                    "lower": drate_lower,
                    "upper": drate_upper,
                })
                if providers is not None:
                    df_drate = df_drate[df_drate["group_id"].isin(np.asarray(providers))].reset_index(drop=True)
                results["direct_rate"] = df_drate

        return results
