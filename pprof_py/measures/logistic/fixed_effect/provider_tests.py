"""Provider-effect hypothesis testing for logistic fixed-effect models.

Contains ``_ProviderTestMethods``, a mixin fragment providing
:meth:`test` (gamma-level tests) and :meth:`test_standardized`
(standardized-measure tests with empirical-null calibration).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from fast_poibin import PoiBin

from ....utils.numerical import sigmoid

logger = logging.getLogger(__name__)


class _ProviderTestMethods:
    """Mixin fragment: provider-effect hypothesis tests."""

    def test(
        self,
        providers: Optional[Union[list, np.ndarray]] = None,
        level: float = 0.95,
        test_method: str = "poibin_exact",
        score_modified: bool = True,
        null: Union[str, float] = "median",
        n_bootstrap: int = 10000,
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """Conduct hypothesis tests on provider effects.

        Supported test methods:
        - "poibin_exact"  (exact test using Poisson-binomial DP approach)
        - "bootstrap_exact" (exact test via bootstrap resampling)
        - "score"         (score test; can be "modified" or standard)
        - "wald"          (wald test; disclaim for outlying providers)

        Parameters
        ----------
        providers : list or np.ndarray, optional
            Subset of provider IDs to test. If None, all providers are included.
        level : float, default=0.95
            Confidence level => alpha = 1 - level is significance.
        test_method : {"poibin_exact","bootstrap_exact","score","wald"}, default="poibin_exact"
            Which testing approach to use.
        score_modified : bool, default=True
            If True, uses a simpler "modified" score approach that does not re-fit 
            restricted models for each provider. If False, you would do the standard 
            approach (placeholder or partial).
        null : {"median"} or float, default="median"
            The null hypothesis value for gamma. If "median", uses median(gamma_hat).
            If numeric, that numeric is used instead.
        n_bootstrap : int, default=10000
            Resample size for "bootstrap_exact" approach.
        alternative : {"two_sided","greater","less"}, default="two_sided"
            Direction of test. "two_sided" is default.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ["flag", "p_value", "stat", "std_error"] (if applicable)
            indexed by provider ID. Also has an attribute "provider_size".

        Raises
        ------
        ValueError
            If the model is not fitted or if arguments are invalid.
        """

        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("The model must be fitted with valid coefficients_ and variances_.")

        alpha = 1.0 - level
        gamma_vals = self.coefficients_["gamma"].flatten()
        se_gamma = np.sqrt(self.variances_["gamma"].flatten())
        n_obs = self.outcome_.size if self.outcome_ is not None else 0
        p = len(self.coefficients_["beta"])
        m = len(gamma_vals)
        df = n_obs - (m + p)

        if null == "median":
            gamma_null = np.median(gamma_vals)
        elif isinstance(null, (int, float)):
            gamma_null = float(null)
        else:
            raise ValueError("Argument 'null' must be 'median' or a numeric value.")

        full_index = np.arange(m)
        if providers is not None:
            mask = np.isin(self.groups_, providers)
            indices = full_index[mask]
        else:
            indices = full_index

        tested_groups = self.groups_[indices]
        size_dict = dict(zip(self.groups_, self.group_sizes_))

        if test_method == "wald":
            flags, pvals, stats, se = self._compute_wald_gamma(indices, gamma_null, alpha, alternative, gamma_vals, se_gamma, df)
        elif test_method == "score":
            flags, pvals, stats, se = self._compute_score_gamma(indices, gamma_null, alpha, alternative, score_modified)
        elif test_method == "poibin_exact":
            flags, pvals, stats, se = self._compute_poibin_gamma(indices, gamma_null, alpha, alternative)
        elif test_method == "bootstrap_exact":
            flags, pvals, stats, se = self._compute_bootstrap_gamma(indices, gamma_null, alpha, alternative, n_bootstrap)
        else:
            raise ValueError("test_method must be one of {'poibin_exact','bootstrap_exact','score','wald'}.")

        df_res = pd.DataFrame({
            "flag": pd.Categorical(flags, categories=[-1, 0, 1]),
            "p_value": pvals,
            "stat": stats,
            "std_error": se
        }, index=tested_groups)
        df_res.attrs["provider_size"] = {gid: size_dict[gid] for gid in tested_groups}
        return df_res

    def _compute_wald_gamma(
        self, 
        indices: np.ndarray,
        gamma_null: float, 
        alpha: float, 
        alternative: str, 
        gamma_vals: np.ndarray, 
        se_gamma: np.ndarray, 
        df: int
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """Compute Wald test for gamma coefficients.

        Parameters
        ----------
        indices : np.ndarray
            Indices of tested groups.
        gamma_null : float
            Null hypothesis value.
        alpha : float
            Significance level.
        alternative : str
            Hypothesis type ("two_sided", "greater", or "less").
        gamma_vals : np.ndarray
            Gamma coefficient values.
        se_gamma : np.ndarray
            Standard errors for gamma.
        df : int
            Degrees of freedom.

        Returns
        -------
        Tuple[List[int], List[float], List[float], List[float]]
            Flags, p-values, test statistics, and standard errors.
        """
        tested_gamma = gamma_vals[indices]
        tested_se = se_gamma[indices]
        wald_stat = (tested_gamma - gamma_null) / tested_se
        prob = t.sf(wald_stat, df=df) if df > 0 else norm.sf(wald_stat)

        flags, pvals, stats, se = [], [], [], []
        for i, st in enumerate(wald_stat):
            pr = prob[i]
            if alternative == "two_sided":
                f_ = 1 if pr < alpha / 2 else -1 if pr > 1 - alpha / 2 else 0
                p_val = 2 * min(pr, 1 - pr)
            elif alternative == "greater":
                f_ = 1 if pr < alpha else 0
                p_val = pr
            elif alternative == "less":
                f_ = -1 if (1 - pr) < alpha else 0
                p_val = 1 - pr
            else:
                raise ValueError("Argument 'alternative' must be 'two_sided','greater','less'.")
            flags.append(f_)
            pvals.append(round(p_val, 7))
            stats.append(st)
            se.append(tested_se[i])
        return flags, pvals, stats, se

    def _compute_score_gamma(
        self, 
        indices: np.ndarray, 
        gamma_null: float, 
        alpha: float, 
        alternative: str, 
        score_modified: bool
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """Compute Score test for gamma coefficients.

        Parameters
        ----------
        indices : np.ndarray
            Indices of tested groups.
        gamma_null : float
            Null hypothesis value.
        alpha : float
            Significance level.
        alternative : str
            Hypothesis type ("two_sided", "greater", or "less").
        score_modified : bool
            Use modified score approach if True.

        Returns
        -------
        Tuple[List[int], List[float], List[float], List[float]]
            Flags, p-values, test statistics, and standard errors.
        """
        if not score_modified:
            raise NotImplementedError("Standard (unmodified) score test not implemented. Use score_modified=True.")

        pvec = 1.0 / (1.0 + np.exp(-(gamma_null + self.xbeta_)))
        pvec = np.clip(pvec, 1e-10, 1 - 1e-10)

        flags, pvals, stats, se = [], [], [], []
        for g_ind in indices:
            mask_g = (self.group_indices_ == g_ind)
            obs_count = np.sum(self.outcome_[mask_g])
            sum_p = np.sum(pvec[mask_g])
            sum_var = np.sum(pvec[mask_g] * (1 - pvec[mask_g]))
            zscore = (obs_count - sum_p) / np.sqrt(sum_var) if sum_var >= 1e-14 else 0.0

            if alternative == "two_sided":
                p_one_side = norm.sf(abs(zscore))
                p_val = 2 * p_one_side
                f_ = 1 if p_one_side < alpha / 2 and zscore > 0 else -1 if p_one_side < alpha / 2 else 0
            elif alternative == "greater":
                p_val = norm.sf(zscore)
                f_ = 1 if p_val < alpha else 0
            elif alternative == "less":
                p_val = norm.cdf(zscore)
                f_ = -1 if p_val < alpha else 0
            else:
                raise ValueError("Argument 'alternative' must be 'two_sided','greater','less'.")
            flags.append(f_)
            pvals.append(round(p_val, 7))
            stats.append(zscore)
            se.append(np.nan)
        return flags, pvals, stats, se

    def _compute_poibin_gamma(
        self, 
        indices: np.ndarray, 
        gamma_null: float, 
        alpha: float, 
        alternative: str
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """Compute Poisson-Binomial exact test for gamma coefficients.

        Parameters
        ----------
        indices : np.ndarray
            Indices of tested groups.
        gamma_null : float
            Null hypothesis value.
        alpha : float
            Significance level.
        alternative : str
            Hypothesis type ("two_sided", "greater", or "less").

        Returns
        -------
        Tuple[List[int], List[float], List[float], List[float]]
            Flags, p-values, test statistics, and standard errors.
        """
        flags, pvals, stats, se = [], [], [], []
        for g_ind in indices:
            mask_g = (self.group_indices_ == g_ind)
            x_mat = self.xbeta_[mask_g]
            pvec = 1.0 / (1.0 + np.exp(-(gamma_null + x_mat)))
            pvec = np.clip(pvec, 1e-10, 1 - 1e-10)
            obs = int(np.sum(self.outcome_[mask_g]))  # Convert obs to integer for indexing

            pb = PoiBin(pvec)
            cdf_obs = pb.cdf[obs] 
            pmf_obs = pb.pmf[obs]          
            cdf_obs_minus_1 = pb.cdf[obs - 1] if obs > 0 else 0.0  # Handle edge case

            if alternative == "two_sided":
                pr = 1.0 - cdf_obs + 0.5 * pmf_obs
                zscore = norm.isf(pr)
                f_ = 1 if pr < alpha / 2 else -1 if pr > 1 - alpha / 2 else 0
                p_val = 2 * min(pr, 1 - pr)
            elif alternative == "greater":
                pr = 1.0 - cdf_obs_minus_1 if obs > 0 else 1.0
                zscore = norm.isf(pr)
                p_val = pr
                f_ = 1 if pr < alpha else 0
            elif alternative == "less":
                pr = cdf_obs
                zscore = norm.ppf(pr)
                p_val = pr
                f_ = -1 if pr < alpha else 0
            else:
                raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")
            flags.append(f_)
            pvals.append(round(p_val, 7))
            stats.append(zscore)
            se.append(np.nan)
        return flags, pvals, stats, se

    def _compute_bootstrap_gamma(
        self, 
        indices: np.ndarray, 
        gamma_null: float, 
        alpha: float, 
        alternative: str, 
        n_bootstrap: int
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """Compute Bootstrap exact test for gamma coefficients.

        Parameters
        ----------
        indices : np.ndarray
            Indices of tested groups.
        gamma_null : float
            Null hypothesis value.
        alpha : float
            Significance level.
        alternative : str
            Hypothesis type ("two_sided", "greater", or "less").
        n_bootstrap : int
            Number of bootstrap resamples.

        Returns
        -------
        Tuple[List[int], List[float], List[float], List[float]]
            Flags, p-values, test statistics, and standard errors.
        """
        flags, pvals, stats, se = [], [], [], []
        for g_ind in indices:
            mask_g = (self.group_indices_ == g_ind)
            x_mat = self.xbeta_[mask_g]
            pvec = 1.0 / (1.0 + np.exp(-(gamma_null + x_mat)))
            pvec = np.clip(pvec, 1e-10, 1 - 1e-10)
            obs = np.sum(self.outcome_[mask_g])

            draws = np.empty(n_bootstrap, dtype=np.int_)
            group_size = pvec.size
            for i_bs in range(n_bootstrap):
                r = np.random.rand(group_size)
                draws[i_bs] = np.sum(r < pvec)

            if alternative == "two_sided":
                bigger = np.sum(draws > obs)
                equal = np.sum(draws == obs)
                pr = (bigger + 0.5 * equal) / n_bootstrap
                zscore = norm.isf(pr)
                f_ = 1 if pr < alpha / 2 else -1 if pr > 1 - alpha / 2 else 0
                p_val = 2 * min(pr, 1 - pr)
            elif alternative == "greater":
                pr = np.sum(draws >= obs) / n_bootstrap
                zscore = norm.isf(pr)
                p_val = pr
                f_ = 1 if pr < alpha else 0
            elif alternative == "less":
                pr = np.sum(draws <= obs) / n_bootstrap
                zscore = norm.ppf(pr)
                p_val = pr
                f_ = -1 if pr < alpha else 0
            else:
                raise ValueError("Argument 'alternative' must be 'two_sided','greater','less'.")
            flags.append(f_)
            pvals.append(round(p_val, 7))
            stats.append(zscore)
            se.append(np.nan)
        return flags, pvals, stats, se

    # -------------------------------------------------------------------------
    # STANDARDIZED MEASURE SE AND EMPIRICAL NULL TESTING
    # -------------------------------------------------------------------------
    def _compute_se_standardized(
        self,
        measure: str = "direct_rate",
        null: Union[str, float] = "median",
        variance_type: str = "model",
        include_extreme_obs: bool = False,
        extreme_obs_total_n: Optional[float] = None,
    ) -> pd.DataFrame:
        """Compute standard errors for standardized measures via the delta method.

        For direct measures, the SE is derived from the sensitivity of the
        standardized quantity to gamma_j (the provider fixed effect). For indirect
        measures, the SE is derived under a Poisson-binomial variance assumption.

        Supports N-weighted computation for binomial models (auto-detected from
        self.N_) and optionally includes extreme observations (xbeta=0) in the
        population denominator.

        Parameters
        ----------
        measure : str, default="direct_rate"
            Standardized measure for which to compute SE. Must be one of
            {"direct_rate", "indirect_rate", "direct_ratio", "indirect_ratio"}.
        null : {'median', 'mean'} or float, default="median"
            Baseline norm for indirect standardization. Used to define the
            expected counts under the null for indirect measures.
        variance_type : {'model', 'robust'}, default="model"
            Type of variance estimator for gamma to use in the delta method.
            - 'model': information-matrix-based SE (assumes independence).
            - 'robust': cluster-robust sandwich SE (requires obs_id_var at fit).
        include_extreme_obs : bool, default=False
            If True, include extreme provider observations (xbeta=0) in the
            population sum for direct standardization.
        extreme_obs_total_n : float, optional
            Total binomial N across extreme provider observations.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ["group_id", "estimate", "se"] containing
            the point estimate and its standard error for each provider.

        Raises
        ------
        ValueError
            If the model is not fitted, if 'measure' is invalid, or if
            variance_type='robust' but robust variances are not available.
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("Model must be fitted before computing SE of standardized measures.")

        valid_measures = {"direct_rate", "indirect_rate", "direct_ratio", "indirect_ratio"}
        if measure not in valid_measures:
            raise ValueError(f"Argument 'measure' must be one of {valid_measures}.")
        if variance_type not in {"model", "robust"}:
            raise ValueError("Argument 'variance_type' must be 'model' or 'robust'.")
        if variance_type == "robust" and self.robust_variances_ is None:
            raise ValueError(
                "Robust variances not available. Fit the model with obs_id_var "
                "to enable cluster-robust variance estimation."
            )

        gamma_vals = self.coefficients_["gamma"].flatten()
        if variance_type == "robust":
            se_gamma = np.sqrt(self.robust_variances_["gamma"].flatten())
        else:
            se_gamma = np.sqrt(self.variances_["gamma"].flatten())
        n_samples = len(self.outcome_)

        # N-weighting for binomial models
        N_obs = self.N_ if self.N_ is not None else np.ones(n_samples)
        Ntot_model = np.sum(N_obs)
        extreme_n = float(extreme_obs_total_n) if (include_extreme_obs and extreme_obs_total_n) else 0.0
        Ntot = Ntot_model + extreme_n

        obs_total = np.sum(self.outcome_)
        population_rate = obs_total / Ntot

        # Determine gamma_null for indirect measures
        if null == "median":
            gamma_null = np.median(gamma_vals)
        elif null == "mean":
            gamma_null = np.average(gamma_vals, weights=self.group_sizes_)
        elif isinstance(null, (int, float)):
            gamma_null = float(null)
        else:
            raise ValueError("Argument 'null' must be 'median', 'mean', or a numeric value.")

        records = []

        if measure in ("direct_rate", "direct_ratio"):
            # Delta method for direct standardization (N-weighted):
            # direct_rate_j = sum(N_i * sigmoid(gamma_j + xbeta_i)) / Ntot
            # d(rate_j)/d(gamma_j) = sum(N_i * p_ij * (1 - p_ij)) / Ntot
            # se(rate_j) = [sum(N_i * p_ij * (1-p_ij)) / Ntot] * se(gamma_j)
            for j, gid in enumerate(self.groups_):
                p_all = sigmoid(gamma_vals[j] + self.xbeta_)
                pred_model = np.sum(N_obs * p_all)
                deriv_model = np.sum(N_obs * p_all * (1.0 - p_all))

                # Extreme obs contribution (xbeta=0)
                if extreme_n > 0:
                    p_ext = sigmoid(gamma_vals[j])
                    pred_model += extreme_n * p_ext
                    deriv_model += extreme_n * p_ext * (1.0 - p_ext)

                if measure == "direct_rate":
                    estimate = pred_model / Ntot
                    se_estimate = (deriv_model / Ntot) * se_gamma[j]
                else:  # direct_ratio
                    estimate = pred_model / obs_total
                    se_estimate = (deriv_model / obs_total) * se_gamma[j]

                records.append({
                    "group_id": gid,
                    "estimate": estimate,
                    "se": se_estimate
                })

        else:  # indirect_rate or indirect_ratio
            # For indirect measures, the estimate is O_j / E_j (ratio) or
            # (O_j / E_j) * population_rate (rate), where E_j depends on gamma_null.
            # SE is derived from the Poisson-binomial variance of O_j:
            # Var(O_j) = sum_i(N_i * p_ij * (1 - p_ij)) for i in group j
            # se(ISR_j) = sqrt(Var(O_j)) / E_j
            for j, gid in enumerate(self.groups_):
                mask = (self.group_indices_ == j)
                xbeta_group = self.xbeta_[mask]
                N_group = N_obs[mask]

                # Expected under null (N-weighted)
                p_null = sigmoid(gamma_null + xbeta_group)
                expected_j = np.sum(N_group * p_null)

                # Observed
                observed_j = np.sum(self.outcome_[mask])

                # Variance of observed (N-weighted Poisson-binomial)
                p_fitted = sigmoid(gamma_vals[j] + xbeta_group)
                var_o_j = np.sum(N_group * p_fitted * (1.0 - p_fitted))

                if expected_j > 1e-10:
                    isr_j = observed_j / expected_j
                    se_isr_j = np.sqrt(var_o_j) / expected_j
                else:
                    isr_j = np.nan
                    se_isr_j = np.nan

                if measure == "indirect_ratio":
                    records.append({"group_id": gid, "estimate": isr_j, "se": se_isr_j})
                else:  # indirect_rate
                    records.append({
                        "group_id": gid,
                        "estimate": isr_j * population_rate,
                        "se": se_isr_j * population_rate
                    })

        return pd.DataFrame(records)

    def _estimate_empirical_null(
        self,
        z_scores: np.ndarray,
        group_sizes: np.ndarray,
        groupwise: bool = True,
        n_groups: int = 4,
        remove_outliers: bool = True,
        outlier_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate empirical null parameters using Huber robust regression.

        Fits a robust intercept-only model (M-estimation with Huber's psi) to
        the Z-scores to estimate the location (intercept) and scale of the
        empirical null distribution. Optionally performs groupwise estimation
        by partitioning providers into size-based quartiles.

        Uses the custom huber_location_scale() implementation that replicates
        R's MASS::rlm(z ~ 1, method='M', psi=psi.huber, scale.est='MAD')
        exactly (scale recomputed every iteration, residual-based convergence).

        Parameters
        ----------
        z_scores : np.ndarray
            Array of Z-scores for each provider.
        group_sizes : np.ndarray
            Array of provider sizes (number of patients), used for groupwise
            stratification.
        groupwise : bool, default=True
            If True, estimate separate empirical null parameters for each
            size-based group (quartile). If False, estimate a single global
            empirical null.
        n_groups : int, default=4
            Number of groups for groupwise estimation. Providers are sorted
            by size and split into approximately equal-sized groups.
        remove_outliers : bool, default=True
            If True, exclude providers flagged by 'outlier_mask' from the
            robust regression (they receive the group's estimated parameters
            but do not influence the fit).
        outlier_mask : np.ndarray of bool, optional
            Boolean array indicating which providers are outliers (e.g.,
            zero-event or all-event providers). If None, no providers are
            excluded from estimation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (intercept, scale) arrays of shape (n_providers,) containing the
            empirical null location and scale for each provider.
        """
        from ...inference.empirical_null import huber_location_scale

        n = len(z_scores)
        intercept = np.zeros(n)
        scale = np.ones(n)

        # Prepare working Z-scores (mask outliers as NaN)
        z_work = z_scores.copy().astype(float)
        if remove_outliers and outlier_mask is not None:
            z_work[outlier_mask] = np.nan

        if groupwise:
            # Sort by group size and assign group labels
            sort_idx = np.argsort(group_sizes)
            split_size = int(np.ceil(n / n_groups))
            group_labels = np.zeros(n, dtype=int)
            for g in range(n_groups):
                start = g * split_size
                end = min((g + 1) * split_size, n)
                group_labels[sort_idx[start:end]] = g

            # Fit Huber M-estimator per group
            for g in range(n_groups):
                mask_g = (group_labels == g)
                z_group = z_work[mask_g]
                valid = ~np.isnan(z_group)

                if np.sum(valid) < 3:
                    # Too few valid observations; fall back to defaults
                    intercept[mask_g] = 0.0
                    scale[mask_g] = 1.0
                    continue

                z_valid = z_group[valid]
                loc, scl = huber_location_scale(z_valid)
                intercept[mask_g] = loc
                scale[mask_g] = scl
        else:
            # Global Huber M-estimator
            valid = ~np.isnan(z_work)
            if np.sum(valid) < 3:
                return intercept, scale

            z_valid = z_work[valid]
            loc, scl = huber_location_scale(z_valid)
            intercept[:] = loc
            scale[:] = scl

        return intercept, scale

    def test_standardized(
        self,
        providers: Optional[Union[list, np.ndarray]] = None,
        measure: str = "direct_rate",
        null: Union[str, float] = "mean",
        level: float = 0.95,
        variance_type: str = "model",
        empirical_null: bool = False,
        groupwise: bool = True,
        n_groups: int = 4,
        remove_outliers: bool = True,
        scale: float = 1.81,
        alternative: str = "two_sided",
        z_scale: str = "auto",
        include_extreme_obs: bool = False,
        extreme_obs_total_n: Optional[float] = None,
    ) -> pd.DataFrame:
        """Test provider effects using standardized measures with optional empirical null calibration.

        Computes Z-scores on an appropriate transformed scale and optionally
        applies empirical null calibration via Huber robust regression to adjust
        for overdispersion in large-scale simultaneous testing.

        This method generalizes R's PPPW_Test, SFR_Test, and AOH flagging to
        work with any standardized measure: direct rate, indirect rate, direct
        ratio, indirect ratio, or raw gamma.

        Parameters
        ----------
        providers : list or np.ndarray, optional
            Subset of provider IDs to include. If None, all providers are tested.
        measure : str, default="direct_rate"
            Standardized measure to test. Must be one of {"direct_rate",
            "indirect_rate", "direct_ratio", "indirect_ratio", "gamma"}.
        null : {'mean', 'median', 'crude'} or float, default="mean"
            Null hypothesis reference value.
            - 'mean': population average (mean of provider estimates).
            - 'median': median of provider estimates.
            - 'crude': sum(Y) / sum(N) — the crude population rate. Used by
              AOH where pie_l = sum(Y_all) / sum(N_all).
            - If float: uses that value directly (on the z_scale being used).
        level : float, default=0.95
            Confidence level for intervals and flagging.
        variance_type : {'model', 'robust'}, default="model"
            Type of variance estimator for gamma.
        empirical_null : bool, default=False
            If True, estimate the null distribution from the data using Huber
            robust regression. If False, use a fixed scale (default 1.81).
        groupwise : bool, default=True
            If True and empirical_null=True, estimate separate null parameters
            per provider-size quartile.
        n_groups : int, default=4
            Number of size-based groups for groupwise empirical null.
        remove_outliers : bool, default=True
            If True and empirical_null=True, exclude extreme providers
            (zero-event/all-event with gamma at boundary) from the robust
            regression fit.
        scale : float, default=1.81
            Fixed scale factor when empirical_null=False. A value of 1.0
            corresponds to the theoretical null (no overdispersion).
        alternative : {'two_sided', 'greater', 'less'}, default="two_sided"
            Direction of the hypothesis test.
        z_scale : {'auto', 'identity'}, default="auto"
            Scale on which to compute Z-scores.
            - 'auto': logit transform for rates, log for ratios, identity for
              gamma (the default PPPW behavior).
            - 'identity': compute Z directly on the rate/ratio scale without
              any transform: z = (estimate - null) / se. Required for SFR and
              AOH where R uses rate-scale z-tests.
        include_extreme_obs : bool, default=False
            If True, include extreme provider observations (xbeta=0) in the
            population for SE computation. Passed to _compute_se_standardized.
        extreme_obs_total_n : float, optional
            Total binomial N across extreme provider observations. Required
            when include_extreme_obs=True.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by provider ID with columns:
            - estimate : point estimate on the original scale
            - se : standard error on the original scale
            - transformed : point estimate on the working scale
            - se_transformed : SE on the working scale
            - null_value : null hypothesis value on the working scale
            - z_score : raw Z-score (before EN calibration)
            - intercept : empirical null location (0 if EN disabled)
            - scale : empirical null scale (fixed 'scale' if EN disabled)
            - z_calibrated : calibrated Z-score = (z_score - intercept) / scale
            - flag : integer flag (1=higher, 0=as expected, -1=lower)
            - p_value : two-sided or one-sided p-value
            - ci_lower : lower confidence limit on the original scale
            - ci_upper : upper confidence limit on the original scale

        Raises
        ------
        ValueError
            If the model is not fitted or if arguments are invalid.

        Notes
        -----
        When z_scale='auto', the transform depends on the measure:
        - Rates (bounded in (0,1)): logit transform for approximate normality.
        - Ratios (positive): log transform for approximate normality.
        - Gamma (unbounded): identity (no transform).

        When z_scale='identity', no transform is applied regardless of measure.
        This is used by SFR (z = (sfr_f - sfr_u) / sfr_se) and AOH
        (z = (pie_j - pie_l) / se_pie_j) where the rate-scale z-test is standard.

        The confidence intervals are constructed on the working scale as:
            transformed ± z_crit * scale * se_transformed
        and then back-transformed to the original scale (if z_scale='auto').
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("Model must be fitted before calling 'test_standardized'.")

        valid_measures = {"direct_rate", "indirect_rate", "direct_ratio", "indirect_ratio", "gamma"}
        if measure not in valid_measures:
            raise ValueError(f"Argument 'measure' must be one of {valid_measures}.")
        if alternative not in {"two_sided", "greater", "less"}:
            raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")
        if z_scale not in {"auto", "identity"}:
            raise ValueError("Argument 'z_scale' must be 'auto' or 'identity'.")
        if variance_type not in {"model", "robust"}:
            raise ValueError("Argument 'variance_type' must be 'model' or 'robust'.")
        if variance_type == "robust" and self.robust_variances_ is None:
            raise ValueError(
                "Robust variances not available. Fit the model with obs_id_var "
                "to enable cluster-robust variance estimation."
            )

        gamma_vals = self.coefficients_["gamma"].flatten()
        alpha = 1.0 - level

        # --- Step 1: Obtain estimates and SEs on the original scale ---
        if measure == "gamma":
            estimates = gamma_vals.copy()
            if variance_type == "robust":
                se_estimates = np.sqrt(self.robust_variances_["gamma"].flatten())
            else:
                se_estimates = np.sqrt(self.variances_["gamma"].flatten())
        else:
            se_df = self._compute_se_standardized(
                measure=measure, null=null, variance_type=variance_type,
                include_extreme_obs=include_extreme_obs,
                extreme_obs_total_n=extreme_obs_total_n,
            )
            estimates = se_df["estimate"].values
            se_estimates = se_df["se"].values

        # --- Step 2: Transform to approximate normality ---
        if z_scale == "identity" or measure == "gamma":
            # Identity: work directly on the rate/ratio/gamma scale
            # Used by SFR (z = (sfr_f - sfr_u) / sfr_se) and AOH
            transformed = estimates.copy()
            se_transformed = se_estimates.copy()
        elif measure in ("direct_rate", "indirect_rate"):
            # Logit transform: f(x) = log(x / (1 - x))
            est_clipped = np.clip(estimates, 1e-10, 1.0 - 1e-10)
            transformed = np.log(est_clipped / (1.0 - est_clipped))
            # Delta method: se(logit(x)) = se(x) / (x * (1 - x))
            se_transformed = se_estimates / (est_clipped * (1.0 - est_clipped))
        elif measure in ("direct_ratio", "indirect_ratio"):
            # Log transform: f(x) = log(x)
            est_clipped = np.maximum(estimates, 1e-10)
            transformed = np.log(est_clipped)
            # Delta method: se(log(x)) = se(x) / x
            se_transformed = se_estimates / est_clipped

        # --- Step 3: Compute null value on the working scale ---
        if isinstance(null, (int, float)):
            null_transformed = float(null)
        elif null == "crude":
            # Crude population rate: sum(Y) / sum(N)
            # For binomial: sum(Y_all) / sum(N_all) including extreme obs
            N_obs = self.N_ if self.N_ is not None else np.ones(len(self.outcome_))
            Ntot_model = np.sum(N_obs)
            extreme_n = float(extreme_obs_total_n) if (include_extreme_obs and extreme_obs_total_n) else 0.0
            Ntot = Ntot_model + extreme_n
            crude_rate = np.sum(self.outcome_) / Ntot
            # Apply same transform as Step 2
            if z_scale == "identity" or measure == "gamma":
                null_transformed = crude_rate
            elif measure in ("direct_rate", "indirect_rate"):
                crude_clipped = np.clip(crude_rate, 1e-10, 1.0 - 1e-10)
                null_transformed = np.log(crude_clipped / (1.0 - crude_clipped))
            elif measure in ("direct_ratio", "indirect_ratio"):
                null_transformed = np.log(max(crude_rate, 1e-10))
        elif null == "mean":
            if z_scale == "identity":
                null_transformed = np.mean(estimates)
            elif measure in ("direct_rate", "indirect_rate"):
                mean_est = np.mean(estimates)
                mean_est = np.clip(mean_est, 1e-10, 1.0 - 1e-10)
                null_transformed = np.log(mean_est / (1.0 - mean_est))
            elif measure in ("direct_ratio", "indirect_ratio"):
                mean_est = np.mean(estimates)
                mean_est = max(mean_est, 1e-10)
                null_transformed = np.log(mean_est)
            else:  # gamma
                null_transformed = np.average(gamma_vals, weights=self.group_sizes_)
        elif null == "median":
            if z_scale == "identity":
                null_transformed = np.median(estimates)
            elif measure in ("direct_rate", "indirect_rate"):
                med_est = np.median(estimates)
                med_est = np.clip(med_est, 1e-10, 1.0 - 1e-10)
                null_transformed = np.log(med_est / (1.0 - med_est))
            elif measure in ("direct_ratio", "indirect_ratio"):
                med_est = np.median(estimates)
                med_est = max(med_est, 1e-10)
                null_transformed = np.log(med_est)
            else:  # gamma
                null_transformed = np.median(gamma_vals)
        else:
            raise ValueError("Argument 'null' must be 'mean', 'median', 'crude', or a numeric value.")

        # --- Step 4: Compute Z-scores ---
        # Avoid division by zero for providers with negligible SE
        se_safe = np.where(se_transformed > 1e-14, se_transformed, np.nan)
        z_scores = (transformed - null_transformed) / se_safe

        # --- Step 5: Empirical null calibration ---
        if empirical_null:
            # Build outlier mask: providers at gamma boundaries (±17 or similar extremes)
            outlier_mask = None
            if remove_outliers:
                gamma_bound = self.algorithm.bound if self.algorithm is not None else 10.0
                outlier_mask = (np.abs(gamma_vals) >= gamma_bound - 0.1)

            en_intercept, en_scale = self._estimate_empirical_null(
                z_scores=z_scores,
                group_sizes=self.group_sizes_,
                groupwise=groupwise,
                n_groups=n_groups,
                remove_outliers=remove_outliers,
                outlier_mask=outlier_mask
            )
        else:
            en_intercept = np.zeros(len(gamma_vals))
            en_scale = np.full(len(gamma_vals), scale)

        # --- Step 6: Calibrated Z-scores, flagging, and p-values ---
        z_calibrated = (z_scores - en_intercept) / en_scale

        if alternative == "two_sided":
            p_values = 2.0 * norm.sf(np.abs(z_calibrated))
            flags = np.where(z_calibrated > norm.ppf(1.0 - alpha / 2.0), 1,
                             np.where(z_calibrated < norm.ppf(alpha / 2.0), -1, 0))
        elif alternative == "greater":
            p_values = norm.sf(z_calibrated)
            flags = np.where(z_calibrated > norm.ppf(1.0 - alpha), 1, 0)
        else:  # less
            p_values = norm.cdf(z_calibrated)
            flags = np.where(z_calibrated < norm.ppf(alpha), -1, 0)

        # --- Step 7: Confidence intervals on transformed scale, back-transformed ---
        if alternative == "two_sided":
            z_crit = norm.ppf(1.0 - alpha / 2.0)
            ci_lower_t = transformed - z_crit * en_scale * se_transformed
            ci_upper_t = transformed + z_crit * en_scale * se_transformed
        elif alternative == "greater":
            z_crit = norm.ppf(1.0 - alpha)
            ci_lower_t = transformed - z_crit * en_scale * se_transformed
            ci_upper_t = np.full_like(transformed, np.inf)
        else:  # less
            z_crit = norm.ppf(1.0 - alpha)
            ci_lower_t = np.full_like(transformed, -np.inf)
            ci_upper_t = transformed + z_crit * en_scale * se_transformed

        # Back-transform CIs to original scale
        if z_scale == "identity" or measure == "gamma":
            # No transform was applied — CIs already on original scale
            ci_lower = ci_lower_t
            ci_upper = ci_upper_t
        elif measure in ("direct_rate", "indirect_rate"):
            # Inverse logit: expit(x) = exp(x) / (1 + exp(x))
            ci_lower = sigmoid(ci_lower_t)
            ci_upper = sigmoid(ci_upper_t)
        elif measure in ("direct_ratio", "indirect_ratio"):
            # Inverse log: exp(x)
            ci_lower = np.exp(ci_lower_t)
            ci_upper = np.exp(ci_upper_t)

        # --- Step 8: Subset to requested providers ---
        result_df = pd.DataFrame({
            "estimate": estimates,
            "se": se_estimates,
            "transformed": transformed,
            "se_transformed": se_transformed,
            "null_value": null_transformed,
            "z_score": z_scores,
            "intercept": en_intercept,
            "scale": en_scale,
            "z_calibrated": z_calibrated,
            "flag": flags.astype(int),
            "p_value": p_values,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }, index=self.groups_)
        result_df.index.name = "group_id"

        if providers is not None:
            result_df = result_df.loc[result_df.index.isin(providers)]

        return result_df
