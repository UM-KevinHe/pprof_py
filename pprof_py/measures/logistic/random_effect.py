"""Provider standardization and measure-specific workflows for
`LogisticRandomEffectModel`: standardized ratios/rates (direct/indirect),
provider-level confidence intervals, and provider-effect hypothesis
testing. Mixed into the model class so that
`models/logistic/random_effect.py` can stay focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations
from ...inference.effect_tests import (EXACT_P_FLOOR, effect_test, normalize_alternative, poibin_tails,
                                       reference_effect, resample_tails, z_from_tails)

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
        providers=None,
        *,
        group_var: Optional[str] = None,
        test_method: str = "wald",
        reference=0.0,
        null_model=None,
        alternative: str = "two_sided",
        level: float = 0.95,
        critical: Optional[float] = None,
        interval: str = "inversion",
        n_resample: int = 10000,
        seed=None,
    ) -> pd.DataFrame:
        """Test each group's random effect against the reference effect gamma_0.

        Parameters
        ----------
        providers : array-like, optional
            Report only these groups; gamma_0 and any empirical null use all.
        group_var : str, optional
            Grouping factor to test (required with crossed random effects).
        test_method : {"wald", "resampling", "poibin_exact"}
            ``"wald"``: ``(b_j - gamma_0) / SE(b_j)`` with the BLUP's posterior
            SE. ``"poibin_exact"``: exact Poisson-binomial test of the group's
            event count with its effect set to gamma_0 and other random effects
            at their posterior means. ``"resampling"``: the same test drawing
            the other random effects from their posterior (He et al. 2013).
        reference : "median", "mean", or float
            The reference effect gamma_0 (default 0, the random-effect mean, as R pprof): the median of the estimated effects,
            their size-weighted mean, or a value on the effect scale.
        null_model : NullModel or callable, optional
            Null for the z-statistics: :class:`~pprof_py.inference.TheoreticalNull`
            by default, or an instance such as ``FixedNull(sd=...)``, or a callable
            that receives the z-statistics, such as ``EmpiricalNull.fitter(...)``.
        alternative, level, critical, interval
            As in :func:`~pprof_py.inference.provider_test`.
            Intervals are available for the Wald test.
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
        if test_method not in ("wald", "resampling", "poibin_exact"):
            raise ValueError(f"test_method={test_method!r} is not supported; use 'wald', 'resampling', or 'poibin_exact'.")
        if group_var is None:
            if len(self._group_vars) == 1:
                group_var = self._group_vars[0]
            else:
                raise ValueError(f"Specify group_var; available: {self._group_vars}")
        blups = self.get_random_effects(group_var)
        post_se = self._get_posterior_se(group_var)
        k = self._group_vars.index(group_var)
        idx = np.asarray(self._group_indices[k]).ravel()
        n_levels = self._n_groups[k]
        g0 = reference_effect(blups.values, np.bincount(idx, minlength=n_levels), reference)
        se = None
        if test_method == "wald":
            se = np.asarray(post_se.values, dtype=np.float64)
            z = (blups.values - g0) / np.maximum(se, 1e-15)
        else:
            re_mean = np.zeros(self._n)
            re_var = np.zeros(self._n)
            for j, gv in enumerate(self._group_vars):
                if gv == group_var:
                    continue
                re_mean += self.get_random_effects(gv).values[self._group_indices[j]]
                re_var += self._get_posterior_se(gv).values[self._group_indices[j]] ** 2
            order = np.argsort(idx, kind="stable")
            edges = np.r_[0, np.cumsum(np.bincount(idx, minlength=n_levels))]
            rng = np.random.default_rng(seed) if test_method == "resampling" else None
            tails = np.empty((n_levels, 4))
            for j in range(n_levels):
                rows = order[edges[j]:edges[j + 1]]
                obs = self._y[rows].sum()
                if test_method == "poibin_exact":
                    tails[j] = poibin_tails(obs, expit(g0 + re_mean[rows] + self.xbeta_[rows]))
                else:
                    tails[j] = resample_tails(obs, self.xbeta_[rows], re_mean[rows], re_var[rows], g0, n_resample, rng)
            two = alt == "two_sided"
            z = z_from_tails(tails[:, 0] if two else tails[:, 2], tails[:, 1] if two else tails[:, 3], alt,
                             EXACT_P_FLOOR if test_method == "poibin_exact" else 0.5 / n_resample)
        return effect_test(blups.index, blups.values, z, g0, se=se, null_model=null_model, alternative=alt,
                           level=level, critical=critical, interval=interval, providers=providers,
                           test_method=test_method)

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
