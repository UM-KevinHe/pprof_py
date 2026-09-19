"""Provider standardization and measure-specific workflows for
`LinearRandomEffectModel`: standardized differences (direct/indirect),
provider-level confidence intervals, and provider-effect hypothesis
testing. Mixed into the model class so that
`models/linear/random_effect.py` can stay focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


class _LinearREMeasuresHost(Protocol):
    """Attribute contract that `RandomEffectMeasuresMixin` expects from its
    host class (`LinearRandomEffectModel`).
    """
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    groups_: Optional[np.ndarray]
    group_indices_: Optional[np.ndarray]
    group_sizes_: Optional[np.ndarray]
    sigma_: Optional[float]
    xbeta_: Optional[np.ndarray]

    def _check_is_fitted(self) -> None: ...
    def calculate_standardized_measures(self, **kwargs: Any) -> dict: ...


class RandomEffectMeasuresMixin:
    """Standardized measures, provider-level confidence intervals, and
    provider-effect hypothesis tests for `LinearRandomEffectModel`."""

    def calculate_standardized_measures(
        self, providers: Optional[Union[list, np.ndarray]] = None,
        stdz: Union[str, list] = "indirect",
        null: str = "median"
    ) -> dict:
        """Calculate direct/indirect standardized differences for the random effect model.

        Parameters
        ----------
        providers : Optional[Union[list, np.ndarray]]
            Specifies a subset of providers (groups) for which the measures are calculated.
            Defaults to all providers.
        stdz : Union[str, list], default="indirect"
            Standardization method(s); can be "indirect", "direct", or both.
        null : Union[str, float], default="median"
            Baseline norm used for standardization; can be "median", "mean", or a specific numeric value.

        Returns
        -------
        dict
            A dictionary containing DataFrames of standardized differences and observed/expected outcomes
            grouped by method. The keys will be "indirect" and/or "direct" based on the selected methods.
        """

        self._check_is_fitted()
        if isinstance(stdz, str):
            stdz = [stdz]

        if not any(method in stdz for method in ["indirect", "direct"]):
            raise ValueError("Argument 'stdz' must include 'indirect' and/or 'direct'.")

        # Extract model components
        random_effects = self.coefficients_["alpha"]
        group_names = self.groups_
        group_indices = self.group_indices_
        group_sizes = self.group_sizes_
        total_samples = len(self.fitted_)

        # Determine the null value for random effects
        if null == "median":
            re_null = np.median(random_effects)
        elif null == "mean":
            re_null = np.average(random_effects, weights=group_sizes)
        elif isinstance(null, (int, float)):
            re_null = null
        else:
            raise ValueError("Invalid 'null' argument provided. Must be 'median', 'mean', or a numeric value.")

        # If providers are specified, select those groups; otherwise, use all groups
        if providers is not None:
            mask = np.isin(group_names, providers)
            selected_groups = group_names[mask]
        else:
            selected_groups = group_names

        results = {}

        # Indirect Standardization
        if "indirect" in stdz:
            n_groups = len(group_names)
            # Compute expected outcomes by group by excluding random effects
            expected_by_group = np.bincount(group_indices, weights=self.xbeta_, minlength=n_groups)
            # Compute observed outcomes using the full fitted values including random effects
            observed_by_group = np.bincount(group_indices, weights=self.fitted_, minlength=n_groups)
            # Indirect standardized difference
            indirect_diff = (observed_by_group - expected_by_group) / group_sizes

            indirect_df = pd.DataFrame({
                "group_id": group_names,
                "indirect_difference": indirect_diff,
                "observed": observed_by_group,
                "expected": expected_by_group
            })

            if providers is not None:
                indirect_df = indirect_df[indirect_df['group_id'].isin(selected_groups)].reset_index(drop=True)

            results["indirect"] = indirect_df

        # Direct Standardization
        if "direct" in stdz:
            # Overall observed is the total sum of the fitted values
            total_observed = self.fitted_.sum()
            # Calculate direct expected outcomes using group-specific random effects:
            # sum(xbeta + re_val) == xbeta.sum() + re_val * n_samples.
            xbeta_sum = self.xbeta_.sum()
            expected_direct_by_group = xbeta_sum + random_effects * total_samples
            # Direct standardized difference
            direct_diff = (expected_direct_by_group - total_observed) / total_samples

            direct_df = pd.DataFrame({
                "group_id": group_names,
                "direct_difference": direct_diff,
                "observed": np.full(len(group_names), total_observed),
                "expected": expected_direct_by_group
            })

            if providers is not None:
                direct_df = direct_df[direct_df['group_id'].isin(selected_groups)].reset_index(drop=True)

            results["direct"] = direct_df

        return results
    
    def _compute_ci_bounds(
        self, 
        re_coef: np.ndarray, 
        se: np.ndarray, 
        level: float, 
        alternative: str
    ) -> tuple:
        """Compute lower and upper bounds for confidence intervals given estimates,
        their standard errors, and the desired alternative using the normal approximation.
        
        Parameters
        ----------
        re_coef : np.ndarray
            The point estimates.
        se : np.ndarray
            Standard errors corresponding to re_coef.
        level : float
            Confidence level (e.g., 0.95).
        alternative : str
            One of "two_sided", "greater", or "less".
            
        Returns
        -------
        tuple
            (lower, upper) as np.ndarray of the same shape as re_coef.
        """
        alpha = 1 - level
        if alternative == "two_sided":
            crit_value = norm.ppf(1 - alpha / 2)
            lower = re_coef - crit_value * se
            upper = re_coef + crit_value * se
        elif alternative == "greater":
            crit_value = norm.ppf(1 - alpha)
            lower = re_coef - crit_value * se
            upper = np.full_like(re_coef, np.inf)
        elif alternative == "less":
            crit_value = norm.ppf(1 - alpha)
            lower = np.full_like(re_coef, -np.inf)
            upper = re_coef + crit_value * se
        else:
            raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")
        return lower, upper
    
    def calculate_confidence_intervals(
        self,
        providers: Optional[Union[list, np.ndarray]] = None,
        level: float = 0.95,
        option: str = "SM",
        stdz: Union[str, list] = "indirect",
        null: Union[str, float] = "median",
        alternative: str = "two_sided"
    ) -> dict:
        """Calculate confidence intervals for provider (random) effects or standardized measures.

        Parameters
        ----------
        providers : Optional[Union[list, np.ndarray]]
            Subset of providers for which confidence intervals are calculated. Defaults to all providers.
        level : float, default=0.95
            Confidence level.
        option : str, default="SM"
            Specifies whether to provide confidence intervals for "alpha" (provider effects) or "SM" (standardized measures).
        stdz : Union[str, list], default="indirect"
            Standardization method(s) if option is "SM"; must include "indirect" and/or "direct".
        null : Union[str, float], default="median"
            Baseline norm for calculating standardized measures.
        alternative : str, default="two_sided"
            One of "two_sided", "greater", or "less".
            Note: The "alpha" option only supports two-sided intervals.

        Returns
        -------
        dict
            Contains confidence intervals for specified options.
        """
        self._check_is_fitted()

        if isinstance(stdz, str):
            stdz = [stdz]
        if option not in {"alpha", "SM"}:
            raise ValueError("Argument 'option' must be 'alpha' or 'SM'.")
        if (option == "SM") and not any(m in stdz for m in ["indirect", "direct"]):
            raise ValueError("Argument 'stdz' must include 'indirect' and/or 'direct'.")
        if option == "alpha" and alternative != "two_sided":
            raise ValueError("Provider effect 'alpha' only supports two-sided confidence intervals.")

        alpha = 1 - level
        result = {}

        random_effects = self.coefficients_["alpha"]
        var_alpha = self.variances_["alpha"].values[0, 0]  # Extract the scalar value

        # Compute the residual variance.
        sigma_sq = self.sigma_ ** 2

        # self.group_sizes_ is an array with the number of observations for each provider.
        n_prov = self.group_sizes_

        # Compute the shrinkage factor for each provider.
        shrinkage_factor = var_alpha / (var_alpha + sigma_sq / n_prov)

        # Now compute the standard error for each provider effect.
        se_alpha = np.sqrt(shrinkage_factor * sigma_sq / n_prov)

        lower_alpha, upper_alpha = self._compute_ci_bounds(
                    random_effects.to_numpy(), se_alpha, level, alternative
        )

        # Confidence Intervals for Provider (random) Effects ("alpha")
        if option == "alpha":
            alpha_ci = pd.DataFrame({
                "group_id": self.groups_,
                "alpha": random_effects,
                "alpha_lower": lower_alpha,
                "alpha_upper": upper_alpha
            })

            if providers is not None:
                alpha_ci = alpha_ci[alpha_ci["group_id"].isin(providers)].reset_index(drop=True)
            result["alpha_ci"] = alpha_ci

        # Confidence Intervals for Standardized Measures (SM)
        if option == "SM":
            sm_results = self.calculate_standardized_measures(stdz=stdz, null=null)
            
            if "indirect" in stdz:
                lower_obs = np.repeat(lower_alpha, n_prov) + self.xbeta_.flatten()
                upper_obs = np.repeat(upper_alpha, n_prov) + self.xbeta_.flatten()

                lower_prov = np.bincount(self.group_indices_, weights=lower_obs)
                upper_prov = np.bincount(self.group_indices_, weights=upper_obs)

                indirect_df = sm_results["indirect"].copy()
                expected_indirect = indirect_df["expected"].to_numpy()
                indirect_df["lower"] = (lower_prov - expected_indirect) / n_prov
                indirect_df["upper"] = (upper_prov - expected_indirect) / n_prov
                
                if providers is not None:
                    indirect_df = indirect_df[indirect_df["group_id"].isin(providers)].reset_index(drop=True)
                result["indirect_ci"] = indirect_df

            # Direct standardization: 
            if "direct" in stdz:
                # Instead of simply subtracting a baseline (re_null) from lower/upper bounds,
                # we aggregate the fixed-effect predictions with the bounds.
                # In the R code:
                #   Exp.direct(gamma) = sum(linear_pred) + n_total * gamma
                #   lower_direct = (Exp.direct(lower_gamma) - Obs_direct) / n_total
                # Here, we define:
                n_total = self.fitted_.size  # total number of observations
                constant_sum = np.sum(self.xbeta_)  # sum of fixed-effect predictions

                # Compute aggregated bounds:
                lower_prov = constant_sum + n_total * lower_alpha
                upper_prov = constant_sum + n_total * upper_alpha

                # Retrieve the observed overall effect from the SM output:
                obs_direct = sm_results["direct"]["observed"].values 

                if alternative == "two_sided":
                    lower_direct = (lower_prov - obs_direct) / n_total
                    upper_direct = (upper_prov - obs_direct) / n_total
                elif alternative == "greater":
                    lower_direct = (lower_prov - obs_direct) / n_total
                    upper_direct = np.full_like(lower_direct, np.inf)
                elif alternative == "less":
                    lower_direct = np.full_like(lower_prov, -np.inf)
                    upper_direct = (upper_prov - obs_direct) / n_total
                else:
                    raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")

                direct_df = sm_results["direct"].copy()
                direct_df["lower"] = lower_direct
                direct_df["upper"] = upper_direct
                if providers is not None:
                    direct_df = direct_df[direct_df["group_id"].isin(providers)].reset_index(drop=True)
                result["direct_ci"] = direct_df

        return result

    def test(
        self,
        providers: Optional[Union[list, np.ndarray]] = None,
        level: float = 0.95,
        null: float = 0,
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """Conduct hypothesis tests on provider (random) effects to identify outliers.

        Parameters
        ----------
        providers : Optional[Union[list, np.ndarray]]
            A subset of provider IDs for which tests are conducted.
            If None, tests are conducted for all provider IDs.
        level : float, default=0.95
            Confidence level for the hypothesis tests.
        null : float, default=0
            Null hypothesis value for provider effects.
        alternative : str, default="two_sided"
            Alternative hypothesis: Must be one of "two_sided", "greater", or "less".

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns "flag", "p_value", "stat", and "std_error" for each provider.
            - "flag": Indicates whether a provider is an outlier with respect to the null hypothesis. 
            Values are 1 (outlier above), -1 (outlier below), or 0 (not an outlier).
            - "p_value": P-value of the hypothesis test for each provider.
            - "stat": Test statistic (Z-score) calculated for each provider.
            - "std_error": Standard error of the estimated provider effect.
        """
        self._check_is_fitted()

        alpha = 1 - level
        random_effects = self.coefficients_["alpha"]
        var_alpha = self.variances_["alpha"].values[0, 0]
        sigma_sq = self.sigma_ ** 2

        n_prov = self.group_sizes_  # array with sample sizes per provider

        # Calculate the shrinkage (or reliability) factor for each provider
        shrinkage_factor = var_alpha / (var_alpha + sigma_sq / n_prov)
        se_alpha = np.sqrt(shrinkage_factor * sigma_sq / n_prov)
        # Compute test statistic (Z-score)
        Z_score = (random_effects - null) / se_alpha

        # use the upper-tail probability to mimic R's pnorm(..., lower.tail=F)
        p = 1 - norm.cdf(Z_score)

        if alternative == "two_sided":
            p_value = 2 * np.minimum(p, 1 - p)
            flag = np.where(p < alpha / 2, 1, np.where(p > 1 - alpha / 2, -1, 0))
        elif alternative == "greater":
            p_value = p
            flag = np.where(p < alpha, 1, 0)
        elif alternative == "less":
            p_value = 1 - p
            flag = np.where(1 - p < alpha, -1, 0)
        else:
            raise ValueError("Argument 'alternative' should be 'two_sided', 'greater', or 'less'.")

        result = pd.DataFrame({
            "flag": pd.Categorical(flag),
            "p_value": np.round(p_value, 7),
            "stat": Z_score,
            "std_error": se_alpha
        }, index=random_effects.index)

        if providers is not None:
            if isinstance(providers, (list, np.ndarray)):
                result = result.loc[providers]
            else:
                raise ValueError("Argument 'providers' should be a list or array of provider IDs.")

        return result
