"""Covariate-level inference, provider-effect tests (``test()``) and
confidence intervals for ``LinearRandomEffectModel``.
Mixed into the model class so that ``models/linear/random_effect.py``
stays focused on configuration, fitting, and prediction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Union

import numpy as np
import pandas as pd
from scipy.stats import t
from ..effect_tests import effect_test, normalize_alternative, reference_effect
from scipy.stats import norm


class _LinearREInferenceHost(Protocol):
    """Attribute contract that `LinearRandomEffectInferenceMixin` expects from its
    host class (`LinearRandomEffectModel`).
    """
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]

    def _check_is_fitted(self) -> None: ...


class LinearRandomEffectInferenceMixin:
    """Fixed-effect (covariate) statistical inference for
    `LinearRandomEffectModel`."""

    def summary(
        self, covariates: Optional[Union[list, np.ndarray]] = None, 
        level: float = 0.95,
        null: float = 0, 
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """Provide summary statistics for the fixed effects in the model.

        Parameters:
        ----------
        - covariates: Optional[Union[list, np.ndarray]]
            Subset of covariates for which summary statistics are provided. Defaults to all.
        - level: float, default=0.95
            Confidence level for the intervals.
        - null: float, default=0
            Null hypothesis value for the parameter estimates.
        - alternative: str, default="two_sided"
            The alternative hypothesis ("two_sided", "greater", or "less").

        Returns:
        -------
        - summary_df: pd.DataFrame
            A DataFrame with columns:
              - estimate
              - std_error
              - stat
              - p_value
              - ci_lower
              - ci_upper
        """
        self._check_is_fitted()

        fe_estimates = self.coefficients_["beta"]
        se_fe = np.sqrt(np.diag(self.variances_["beta"]))
        stat = (fe_estimates - null) / se_fe

        n = len(self.fitted_)
        p = len(fe_estimates)
        m = len(self.coefficients_["alpha"])
        df = n - p - m  # consistent with fixed_effect model
        
        if alternative == "two_sided":
            p_value = 2 * (1 - t.cdf(np.abs(stat), df=df))
            crit_value = t.ppf(1 - (1 - level) / 2, df=df)
            ci_lower = fe_estimates - crit_value * se_fe
            ci_upper = fe_estimates + crit_value * se_fe
        elif alternative == "greater":
            p_value = 1 - t.cdf(stat, df=df)
            crit_value = t.ppf(1 - (1 - level), df=df)
            ci_lower = fe_estimates - crit_value * se_fe
            ci_upper = np.inf
        elif alternative == "less":
            p_value = t.cdf(stat, df=df)
            crit_value = t.ppf(1 - (1 - level), df=df)
            ci_lower = -np.inf
            ci_upper = fe_estimates + crit_value * se_fe
        else:
            raise ValueError("argument 'alternative' should be 'two_sided', 'greater', or 'less'.")
        
        p_value = np.round(p_value, 7)
        summary_df = pd.DataFrame({
            "estimate": fe_estimates,
            "std_error": se_fe,
            "stat": stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }, index=fe_estimates.index)
        
        if covariates is not None:
            if isinstance(covariates, (list, np.ndarray)):
                summary_df = summary_df.loc[covariates]
            else:
                raise ValueError("argument 'covariates' should be a list or array of covariate names or indices.")
        return summary_df


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
        reference: Union[str, float] = "median",
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
        reference : Union[str, float], default="median"
            Baseline norm for calculating standardized measures.
        alternative : str, default="two_sided"
            One of "two_sided", "greater", or "less".
            Note: The "alpha" option only supports two-sided intervals.

        Returns
        -------
        dict
            Contains confidence intervals for specified options.
        """
        self._require_one_factor()
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

        # self.provider_sizes_ is an array with the number of observations for each provider.
        n_prov = self.provider_sizes_

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
                "provider_id": self.provider_ids_,
                "alpha": random_effects,
                "alpha_lower": lower_alpha,
                "alpha_upper": upper_alpha
            })

            if providers is not None:
                alpha_ci = alpha_ci[alpha_ci["provider_id"].isin(providers)].reset_index(drop=True)
            result["alpha_ci"] = alpha_ci

        # Confidence Intervals for Standardized Measures (SM)
        if option == "SM":
            sm_results = self.calculate_standardized_measures(stdz=stdz, reference=reference)

            if "indirect" in stdz:
                lower_obs = lower_alpha[self.provider_indices_] + self.xbeta_.flatten()
                upper_obs = upper_alpha[self.provider_indices_] + self.xbeta_.flatten()

                lower_prov = np.bincount(self.provider_indices_, weights=lower_obs)
                upper_prov = np.bincount(self.provider_indices_, weights=upper_obs)

                indirect_df = sm_results["indirect"].copy()
                expected_indirect = indirect_df["expected"].to_numpy()
                indirect_df["lower"] = (lower_prov - expected_indirect) / n_prov
                indirect_df["upper"] = (upper_prov - expected_indirect) / n_prov

                if providers is not None:
                    indirect_df = indirect_df[indirect_df["provider_id"].isin(providers)].reset_index(drop=True)
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
                    direct_df = direct_df[direct_df["provider_id"].isin(providers)].reset_index(drop=True)
                result["direct_ci"] = direct_df

        return result

    def test(
        self,
        providers=None,
        *,
        reference=0.0,
        null_model=None,
        alternative: str = "two_sided",
        level: float = 0.95,
        critical: Optional[float] = None,
        interval: str = "inversion",
    ) -> pd.DataFrame:
        """Wald test of each provider's random effect against the reference effect gamma_0.

        ``(alpha_j - gamma_0) / SE(alpha_j)`` with the shrinkage-adjusted SE
        and a normal reference.

        Parameters
        ----------
        providers : array-like, optional
            Report only these providers; gamma_0 and any empirical null use all.
        reference : "median", "mean", or float
            The reference effect gamma_0 (default 0, the random-effect mean): the median of the estimated effects,
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
        self._require_one_factor()
        self._check_is_fitted()
        effects = self.coefficients_["alpha"]
        values = np.asarray(effects, dtype=np.float64).ravel()
        var_alpha = self.variances_["alpha"].values[0, 0]
        sigma_sq = self.sigma_ ** 2
        n_prov = np.asarray(self.provider_sizes_, dtype=np.float64)
        g0 = reference_effect(values, n_prov, reference)
        se = np.sqrt(var_alpha / (var_alpha + sigma_sq / n_prov) * sigma_sq / n_prov)
        index = effects.index if hasattr(effects, "index") else np.arange(values.size)
        return effect_test(index, values, (values - g0) / se, g0, se=se, null_model=null_model,
                           alternative=normalize_alternative(alternative), level=level, critical=critical,
                           interval=interval, providers=providers, test_method="wald")
