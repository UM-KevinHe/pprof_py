"""Variance estimation and covariate-level statistical inference for
``LinearFixedEffectModel``.  Mixed into the model class so that
``models/linear/fixed_effect.py`` stays focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import numpy as np
import pandas as pd
from scipy.stats import t


class _LinearFEInferenceHost(Protocol):
    """Attribute contract that `FixedEffectInferenceMixin` expects from its
    host class (`LinearFixedEffectModel`).
    """
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    sigma_: Optional[float]
    covariate_names_: list
    gamma_var_option: str

    def _check_is_fitted(self) -> None: ...


class FixedEffectInferenceMixin:
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
        var_beta = sigma_hat_sq * np.linalg.inv((Q @ X).T @ (Q @ X))
        
        if self.gamma_var_option == "complete":
            # Remove the sigma factor from var_beta before forming the quadratic term:
            inv_term = np.linalg.inv((Q @ X).T @ (Q @ X))  # This equals var_beta/sigma_hat_sq.
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
