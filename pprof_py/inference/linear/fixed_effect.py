"""Variance estimation, covariate-level inference, provider-effect tests
(``test()``) and confidence intervals for ``LinearFixedEffectModel``.  Mixed into the model class so that
``models/linear/fixed_effect.py`` stays focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import numpy as np
from ...utils.numerical import covariance_from_information, solve_information
import pandas as pd
from scipy.stats import t
from ..effect_tests import effect_test, normalize_alternative, reference_effect
from ..zstat import t_to_z
from typing import Union


class _LinearFEInferenceHost(Protocol):
    """Attribute contract that `LinearFixedEffectInferenceMixin` expects from its
    host class (`LinearFixedEffectModel`).
    """
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    sigma_: Optional[float]
    covariate_names_: list
    gamma_var_option: str

    def _check_is_fitted(self) -> None: ...


class LinearFixedEffectInferenceMixin:
    """Variance estimation and covariate-level statistical inference for
    `LinearFixedEffectModel`."""

    def _estimate_sigma(
        self, residuals: np.ndarray, n_samples: int, n_groups: int, n_features: int
    ) -> float:
        """Estimate the standard deviation of residuals (sigma).

        Parameters:
        ----------
        - residuals: np.ndarray, shape (n_samples,)
            Residuals from the model.
        - n_samples: int
            Total number of samples.
        - n_groups: int
            Number of groups.
        - n_features: int
            Number of predictors.

        Returns:
        -------
        - sigma: float
            Estimated standard deviation of residuals.
        """
        residual_sum_squares = np.sum(residuals**2)
        sigma_hat_sq = residual_sum_squares / (n_samples - n_groups - n_features)
        return np.sqrt(sigma_hat_sq)

    def _estimate_variances(
        self, Q: np.ndarray, X: np.ndarray, X_means: np.ndarray, beta: np.ndarray, group_sizes: np.ndarray
    ) -> dict:
        """Estimate variances of beta and gamma coefficients for inferential statistics.

        Parameters
        ----------
        Q : np.ndarray
            Block diagonal matrix for demeaning group effects.
        X : np.ndarray
            Design matrix.
        X_means : np.ndarray
            Group-level means of predictors.
        beta : np.ndarray
            Estimated regression coefficients.
        group_sizes : np.ndarray
            Number of samples in each group.

        Returns
        -------
        dict
            Variances of beta and gamma coefficients.
        """
        sigma_hat_sq = self.sigma_**2
        # Variance of beta:
        var_beta = sigma_hat_sq * covariance_from_information(
            (Q @ X).T @ (Q @ X), warn=True, what="Within-provider X'X")
        
        if self.gamma_var_option == "complete":
            # Remove the sigma factor from var_beta before forming the quadratic term:
            inv_term = covariance_from_information(
            (Q @ X).T @ (Q @ X), warn=True, what="Within-provider X'X")  # This equals var_beta/sigma_hat_sq.
            # Compute diag(Z_bar %*% inv_term %*% t(Z_bar))
            quad = np.sum((X_means @ inv_term) * X_means, axis=1)  # vector of length n_groups
            var_gamma = sigma_hat_sq * (1 / group_sizes + quad)
            var_gamma = var_gamma.reshape(-1, 1)
        else:  # "simplified"
            var_gamma = (sigma_hat_sq / group_sizes).reshape(-1, 1)
        
        return {"beta": var_beta, "gamma": var_gamma}

    def summary(
        self,
        covariates: list = None,
        level: float = 0.95,
        null: float = 0,
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """Provides summary statistics for the covariate estimates in a fitted fixed effects model.

        Parameters
        ----------
        covariates : list of str or int, optional
            A subset of covariates for which summary statistics are returned.
            Can be specified as a list of covariate names or indices.
            By default, summary statistics for all covariates are returned.
        level : float, default=0.95
            The confidence level for the hypothesis test.
        null : float, default=0
            The null hypothesis value for the covariate coefficients.
        alternative : str, default="two_sided"
            Specifies the alternative hypothesis; must be "two_sided", "greater", or "less".

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns "estimate", "std_error", "stat", "p_value",
            "ci_lower", and "ci_upper" for each covariate.
        """
        self._check_is_fitted()

        # Extract covariate estimates and compute standard errors.
        beta = self.coefficients_["beta"].flatten()
        se_beta = np.sqrt(np.diag(self.variances_["beta"]))
        
        # Get degrees of freedom: total observations minus (number of predictors + number of groups)
        n = self.fitted_.size
        p = len(beta)
        m = len(self.coefficients_["gamma"])
        df = n - p - m
        alpha = 1 - level

        # Compute test statistics.
        stat = (beta - null) / se_beta

        # Compute p-values and confidence intervals based on the specified alternative.
        if alternative == "two_sided":
            p_val = 2 * (1 - t.cdf(np.abs(stat), df))
            crit_val = t.ppf(1 - alpha / 2, df)
            ci_lower = beta - crit_val * se_beta
            ci_upper = beta + crit_val * se_beta
        elif alternative == "greater":
            p_val = 1 - t.cdf(stat, df)
            crit_val = t.ppf(level, df)
            ci_lower = beta - crit_val * se_beta
            ci_upper = np.full_like(beta, np.inf)
        elif alternative == "less":
            p_val = t.cdf(stat, df)
            crit_val = t.ppf(level, df)
            ci_lower = np.full_like(beta, -np.inf)
            ci_upper = beta + crit_val * se_beta
        else:
            raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")

        # Use self.covariate_names_ (or assign default names if not available)
        cov_names = getattr(self, "covariate_names_", [f"X{i}" for i in range(p)])

        # Build the summary DataFrame using lower-case column names.
        summary_df = pd.DataFrame({
            "estimate": beta,
            "std_error": se_beta,
            "stat": stat,
            "p_value": p_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }, index=cov_names)

        # If a subset of covariates is requested, filter the rows accordingly.
        if covariates is not None:
            if all(isinstance(x, str) for x in covariates):
                summary_df = summary_df.loc[covariates]
            elif all(isinstance(x, int) for x in covariates):
                summary_df = summary_df.iloc[covariates]
            else:
                raise ValueError("Argument 'covariates' must be a list of names or indices.")

        return summary_df


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
