"""Provider standardization and measure-specific workflows for
`LinearFixedEffectModel`: standardized differences (direct/indirect),
provider-level confidence intervals, and provider-effect hypothesis
testing. Mixed into the model class so that
`models/linear/fixed_effect.py` can stay focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations
from ...inference.effect_tests import effect_test, normalize_alternative, reference_effect
from ...inference.zstat import t_to_z

from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
import pandas as pd
from scipy.stats import t


class _LinearFEMeasuresHost(Protocol):
    """Attribute contract that `FixedEffectMeasuresMixin` expects from its
    host class (`LinearFixedEffectModel`).
    """
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    provider_ids_: Optional[np.ndarray]
    provider_indices_: Optional[np.ndarray]
    provider_sizes_: Optional[np.ndarray]
    outcome_: Optional[np.ndarray]
    xbeta_: Optional[np.ndarray]

    def _check_is_fitted(self) -> None: ...
    def calculate_standardized_measures(self, **kwargs: Any) -> dict: ...


class FixedEffectMeasuresMixin:
    """Standardized measures, provider-level confidence intervals, and
    provider-effect hypothesis tests for `LinearFixedEffectModel`."""

    def calculate_standardized_measures(
        self, providers=None, stdz="indirect", reference="median"
    ) -> dict:
        """Calculate direct/indirect standardized differences for a fixed effects linear model.

        Parameters
        ----------
        providers : Optional[list or np.ndarray]
            The specific groups or provider identifiers for which the differences should be calculated.
            If None, calculates for all groups.
        stdz : Union[str, list], default="indirect"
            Methods for standardization; can be "indirect", "direct", or both.
        reference : Union[str, float], default="median"
            Baseline norm used for standardization; can be "median", "mean", or a specific numeric value.

        Returns
        -------
        dict
            A dictionary containing DataFrames of standardized differences and observed/expected outcomes
            grouped by method. The keys will be "indirect" and/or "direct" based on the selected methods.
        """

        if self.coefficients_ is None or self.fitted_ is None:
            raise ValueError("The model must be fitted before calculating standardized differences.")

        if self.outcome_ is None:
            raise ValueError("Original outcomes were not stored during fitting.")

        if isinstance(stdz, str):
            stdz = [stdz]

        if not any(method in stdz for method in ["indirect", "direct"]):
            raise ValueError("Argument 'stdz' must include 'indirect' and/or 'direct'.")
        
        # Extract model components
        gamma = self.coefficients_["gamma"].flatten()  # shape (m,)
        n_samples = len(self.outcome_)
        group_sizes = self.provider_sizes_
        
        # Determine the null value for gamma
        if reference == "median":
            gamma_null = np.median(gamma)
        elif reference == "mean":
            gamma_null = np.average(gamma, weights=group_sizes)
        elif isinstance(reference, (int, float)):
            gamma_null = reference
        else:
            raise ValueError("Invalid 'null' argument provided. Must be 'median', 'mean', or a numeric value.")
        
        # If providers are specified, select those groups; otherwise, use all groups.
        if providers is not None:
            mask = np.isin(self.provider_ids_, providers)
            selected_groups = self.provider_ids_[mask]
        else:
            selected_groups = self.provider_ids_
        
        results = {}
        
        # Indirect Standardization
        if "indirect" in stdz:
            n_groups = len(self.provider_ids_)
            # For each observation, expected outcome = gamma_null + linear predictor.
            expected = gamma_null + self.xbeta_.flatten()
            # Sum expected and observed by group.
            expected_by_group = np.bincount(self.provider_indices_, weights=expected, minlength=n_groups)
            observed_by_group = np.bincount(self.provider_indices_, weights=self.outcome_, minlength=n_groups)
            # Standardized difference is the (Obs - Exp) divided by the group size.
            indirect_diff = (observed_by_group - expected_by_group) / group_sizes
            
            indirect_df = pd.DataFrame({
                "provider_id": self.provider_ids_,
                "indirect_difference": indirect_diff,
                "observed": observed_by_group,
                "expected": expected_by_group
            })
            if providers is not None:
                indirect_df = indirect_df[indirect_df['provider_id'].isin(selected_groups)].reset_index(drop=True)
            results["indirect"] = indirect_df
        
        # Direct Standardization
        if "direct" in stdz:
            # Overall observed is the sum over all observations of (gamma_null + xbeta).
            obs_direct_total = (gamma_null + self.xbeta_.flatten()).sum()
            # For each group, compute expected sum using the group-specific gamma:
            # sum(gamma_val + xbeta) == gamma_val * n_samples + xbeta.sum().
            xbeta_sum = self.xbeta_.sum()
            exp_direct_by_group = gamma * n_samples + xbeta_sum
            # Standardized difference is (expected_by_group - overall observed) divided by total sample size.
            direct_diff = (exp_direct_by_group - obs_direct_total) / n_samples
            
            direct_df = pd.DataFrame({
                "provider_id": self.provider_ids_,
                "direct_difference": direct_diff,
                "observed": np.full(len(self.provider_ids_), obs_direct_total),
                "expected": exp_direct_by_group
            })
            if providers is not None:
                direct_df = direct_df[direct_df['provider_id'].isin(selected_groups)].reset_index(drop=True)
            results["direct"] = direct_df
            
        return results
    
    # --- Confidence Intervals ---
    def _compute_ci_bounds(self, gamma: np.ndarray, se: np.ndarray, df: int, level: float, alternative: str) -> tuple:
        """Compute lower and upper bounds for confidence intervals given estimates,
        their standard errors, degrees of freedom, and the desired alternative.

        Parameters
        ----------
        gamma : np.ndarray
            The point estimates.
        se : np.ndarray
            Standard errors corresponding to gamma.
        df : int
            Degrees of freedom.
        level : float
            Confidence level (e.g., 0.95).
        alternative : str
            One of "two_sided", "greater", or "less".

        Returns
        -------
        tuple
            (lower, upper) as np.ndarray of the same shape as gamma.
        """
        alpha = 1 - level
        if alternative == "two_sided":
            crit_value = t.ppf(1 - alpha / 2, df)
            lower = gamma - crit_value * se
            upper = gamma + crit_value * se
        elif alternative == "greater":
            crit_value = t.ppf(1 - alpha, df)
            lower = gamma - crit_value * se
            upper = np.full_like(gamma, np.inf)
        elif alternative == "less":
            crit_value = t.ppf(1 - alpha, df)
            lower = np.full_like(gamma, -np.inf)
            upper = gamma + crit_value * se
        else:
            raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")
        return lower, upper

    def calculate_confidence_intervals(
        self,
        providers=None,
        level: float = 0.95,
        option: str = "SM",
        stdz: Union[str, list] = "indirect",
        reference: Union[str, float] = "median",
        alternative: str = "two_sided"
    ) -> dict:
        """Calculate confidence intervals for provider effects (gamma) or standardized measures (SM).

        Parameters
        ----------
        providers : Optional[list or np.ndarray]
            Subset of group identifiers for which confidence intervals are calculated.
        level : float, default=0.95
            Confidence level.
        option : str, default="SM"
            Either "gamma" for provider effects or "SM" for standardized measures.
        stdz : Union[str, list], default="indirect"
            Standardization method(s) if option is "SM"; must include "indirect" and/or "direct".
        reference : Union[str, float], default="median"
            Baseline norm for calculating standardized measures.
        alternative : str, default="two_sided"
            One of "two_sided", "greater", or "less". (Note: gamma option only supports two_sided.)

        Returns
        -------
        dict
            Dictionary containing DataFrames with confidence intervals.
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("The model must be fitted before calculating confidence intervals.")

        if isinstance(stdz, str):
            stdz = [stdz]
        if option not in {"gamma", "SM"}:
            raise ValueError("Argument 'option' must be 'gamma' or 'SM'.")
        if (option == "SM") and not any(m in stdz for m in ["indirect", "direct"]):
            raise ValueError("Argument 'stdz' must include 'indirect' and/or 'direct'.")
        if option == "gamma" and alternative != "two_sided":
            raise ValueError("Provider effect 'gamma' only supports two-sided confidence intervals.")

        # Degrees of freedom: total observations minus (number of groups + number of predictors)
        n = self.fitted_.size
        p = len(self.coefficients_["beta"])
        m = len(self.coefficients_["gamma"])
        df = n - m - p

        # Compute standard errors for gamma using the flattened variances
        se_gamma = np.sqrt(self.variances_["gamma"].flatten())
        gamma = self.coefficients_["gamma"].flatten()
        
        # Use the helper to compute CI bounds for gamma
        lower_gamma, upper_gamma = self._compute_ci_bounds(gamma, se_gamma, df, level, alternative)
        
        result = {}
        
        if option == "gamma":
            gamma_ci = pd.DataFrame({
                "provider_id": self.provider_ids_,
                "gamma": gamma,
                "lower": lower_gamma,
                "upper": upper_gamma
            })
            if providers is not None:
                gamma_ci = gamma_ci[gamma_ci["provider_id"].isin(providers)].reset_index(drop=True)
            result["gamma_ci"] = gamma_ci
        
        if option == "SM":
            # First get the standardized measures
            sm_results = self.calculate_standardized_measures(stdz=stdz, reference=reference)
            
            # For indirect SM, we need to aggregate CI bounds from gamma over observations.
            # Replicate group-level lower/upper bounds for each observation.
            lower_obs = lower_gamma[self.provider_indices_] + self.xbeta_.flatten()
            upper_obs = upper_gamma[self.provider_indices_] + self.xbeta_.flatten()
            # Sum over each group using np.bincount
            lower_prov = np.bincount(self.provider_indices_, weights=lower_obs)
            upper_prov = np.bincount(self.provider_indices_, weights=upper_obs)
            
            if "indirect" in stdz:
                indirect_df = sm_results["indirect"].copy()
                expected_indirect = indirect_df["expected"].to_numpy()
                # Standardized difference: (observed - expected) normalized by group size.
                lower_indirect = (lower_prov - expected_indirect) / self.provider_sizes_
                upper_indirect = (upper_prov - expected_indirect) / self.provider_sizes_
                indirect_df["lower"] = lower_indirect
                indirect_df["upper"] = upper_indirect
                if providers is not None:
                    # Filter by providers
                    indirect_df = indirect_df[indirect_df["provider_id"].isin(providers)]
                result["indirect_ci"] = indirect_df
            
            if "direct" in stdz:
                # In direct standardization, the standardized difference equals gamma - gamma_null.
                # So the CI for the direct measure is simply:
                # lower_direct = lower_gamma - gamma_null
                # upper_direct = upper_gamma - gamma_null
                if reference == "median":
                    gamma_null = np.median(gamma)
                elif reference == "mean":
                    gamma_null = np.average(gamma, weights=self.provider_sizes_)
                elif isinstance(reference, (int, float)):
                    gamma_null = reference
                else:
                    raise ValueError("Invalid 'null' argument provided.")
                direct_df = sm_results["direct"].copy()
                lower_direct = lower_gamma - gamma_null
                upper_direct = upper_gamma - gamma_null
                direct_df["lower"] = lower_direct
                direct_df["upper"] = upper_direct
                if providers is not None:
                    # Filter direct CIs by providers; assume that the direct_df has a column named "provider_id"
                    direct_df = direct_df[direct_df["provider_id"].isin(providers)]
                result["direct_ci"] = direct_df
            
        return result

    def test(
        self,
        providers=None,
        *,
        reference="median",
        null_model=None,
        alternative: str = "two_sided",
        level: float = 0.95,
        critical: Optional[float] = None,
        interval: str = "inversion",
    ) -> pd.DataFrame:
        """Wald test of each provider's effect against the reference effect gamma_0.

        ``(gamma_j - gamma_0) / SE(gamma_j)`` with a Student-t reference on
        ``n - p - m`` degrees of freedom; p-values, flags, and intervals follow
        the t distribution.

        Parameters
        ----------
        providers : array-like, optional
            Report only these providers; gamma_0 and any empirical null use all.
        reference : "median", "mean", or float
            The reference effect gamma_0: the median of the estimated effects,
            their size-weighted mean, or a value on the effect scale.
        null_model : NullModel or callable, optional
            Null for the z-statistics: :class:`~pprof_py.inference.TheoreticalNull`
            by default, or an instance such as ``FixedNull(sd=...)``, or a callable
            that receives the z-statistics, such as ``EmpiricalNull.fitter(...)``.
        alternative, level, critical, interval
            As in :func:`~pprof_py.inference.provider_test`.

        Returns
        -------
        pandas.DataFrame
            Indexed by provider with columns
            :data:`~pprof_py.inference.PROVIDER_TEST_COLUMNS`: ``flag`` is +1
            above gamma_0, -1 below, 0 not significant, NA not tested.
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("The model must be fitted before testing.")
        gamma = np.asarray(self.coefficients_["gamma"], dtype=np.float64).ravel()
        se = np.sqrt(np.asarray(self.variances_["gamma"], dtype=np.float64).ravel())
        df = self.fitted_.size - len(self.coefficients_["beta"]) - gamma.size
        g0 = reference_effect(gamma, self.provider_sizes_, reference)
        z = t_to_z((gamma - g0) / se, df)
        return effect_test(self.provider_ids_, gamma, z, g0, se=se, df=df, null_model=null_model,
                           alternative=normalize_alternative(alternative), level=level, critical=critical,
                           interval=interval, providers=providers, test_method="wald")
