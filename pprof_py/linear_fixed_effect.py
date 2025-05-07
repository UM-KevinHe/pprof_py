from typing import Optional, Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import norm, t, probplot
from scipy.linalg import block_diag

from .base_model import BaseModel
from .mixins import SummaryMixin, PlotMixin, TestMixin
from .plotting import plot_caterpillar


class LinearFixedEffectModel(BaseModel, SummaryMixin, PlotMixin, TestMixin):
    """
    Linear Fixed Effect model.

    The Linear Fixed Effect model is a linear regression model that includes fixed effects.
    The model is fitted using the weighted least squares method.

    Parameters:
    - gamma_var_option: str, default="complete"
        Option for variance calculation. Must be "complete" or "simplified".

    Examples:
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'y': [1, 2, 3, 4, 5, 6],
    ...     'x1': [1, 0, 1, 0, 1, 0],
    ...     'x2': [0, 1, 0, 1, 0, 1],
    ...     'group': [1, 1, 2, 2, 3, 3]
    ... })
    >>> model = LinearFixedEffectModel()
    >>> model.fit(data, x_vars=['x1', 'x2'], y_var='y', group_var='group')
    >>> predictions = model.predict(data[['x1', 'x2']], groups=data['group'])
    >>> print(predictions)
    """

    def __init__(self, gamma_var_option: str = "complete") -> None:
        if gamma_var_option not in {"complete", "simplified"}:
            raise ValueError("'gamma_var_option' must be 'complete' or 'simplified'.")
        self.gamma_var_option = gamma_var_option

    
    def _preprocess_groups(
        self, X: np.ndarray, y: np.ndarray, group_indices: np.ndarray, n_groups: int
    ) -> tuple:
        """
        Preprocess groups to construct the block diagonal matrix and compute group means.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Design matrix.
        - y: np.ndarray, shape (n_samples,)
            Response variable.
        - group_indices: np.ndarray, shape (n_samples,)
            Group indices for each sample.
        - n_groups: int
            Number of groups.

        Returns:
        - Q: np.ndarray
            Block diagonal matrix for demeaning group effects.
        - y_means: np.ndarray, shape (n_groups,)
            Mean of the response variable for each group.
        - X_means: np.ndarray, shape (n_groups, n_features)
            Mean of predictors for each group.
        """
        group_sizes = np.bincount(group_indices)
        Q = self._construct_block_diag_matrix(group_sizes)
        # Compute group means for y and X:
        y_means = np.array([np.mean(y[group_indices == g]) for g in range(n_groups)])
        X_means = np.array([np.mean(X[group_indices == g], axis=0) for g in range(n_groups)])
        return Q, y_means, X_means

    def _calculate_residuals(
        self, xbeta: np.ndarray, gamma: np.ndarray, y: np.ndarray, group_indices: np.ndarray
    ) -> tuple:
        """
        Calculate predictions and residuals.

        This function computes the predicted values by adding the group-level fixed effects to
        the linear predictor (xbeta) and computes the residuals by subtracting the predictions
        from the observed response values.

        Parameters
        ----------
        xbeta : np.ndarray, shape (n_samples,)
            The linear predictor values (X @ beta) for each sample.
        gamma : np.ndarray, shape (n_groups, 1)
            Fixed effects for each group.
        y : np.ndarray, shape (n_samples,)
            Observed response variable.
        group_indices : np.ndarray, shape (n_samples,)
            Group indices for each sample, indicating which group each sample belongs to.

        Returns
        -------
        tuple
            predictions : np.ndarray, shape (n_samples,)
                The predicted response values.
            residuals : np.ndarray, shape (n_samples,)
                The residuals of the model, calculated as the difference between observed and predicted values.
        """
        gamma_obs = gamma[group_indices].flatten()
        predictions = xbeta.flatten() + gamma_obs
        residuals = y - predictions
        return predictions, residuals

    def _compute_model_statistics(
        self, residuals: np.ndarray, n_samples: int, n_groups: int, n_features: int
    ) -> tuple:
        """
        Compute model statistics including AIC and BIC.

        Parameters:
        - residuals: np.ndarray, shape (n_samples,)
            Residuals from the model.
        - n_samples: int
            Total number of samples.
        - n_groups: int
            Number of groups.
        - n_features: int
            Number of predictors.

        Returns:
        - aic: float
            Akaike Information Criterion.
        - bic: float
            Bayesian Information Criterion.
        """
        residual_sum_squares = np.sum(residuals**2)
        log_likelihood = (
            -n_samples / 2 * np.log(2 * np.pi)
            - n_samples / 2 * np.log(residual_sum_squares / n_samples)
            - residual_sum_squares / (2 * residual_sum_squares / n_samples)
        )
        aic = -2 * log_likelihood + 2 * (n_groups + n_features + 1)
        bic = -2 * log_likelihood + (n_groups + n_features + 1) * np.log(n_samples)
        return aic, bic

    def _construct_block_diag_matrix(self, group_sizes: np.ndarray) -> np.ndarray:
        """
        Construct a block diagonal matrix for demeaning group effects.

        Parameters:
        - group_sizes: np.ndarray, shape (n_groups,)
            Number of samples in each group.

        Returns:
        - Q: np.ndarray
            Block diagonal matrix for demeaning group effects.
        """
        Q_blocks = [np.eye(n) - np.ones((n, n)) / n for n in group_sizes]
        return block_diag(*Q_blocks)

    def _perform_weighted_least_squares(self, Q: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Perform weighted least squares to estimate coefficients.

        Parameters:
        - Q: np.ndarray, shape (n_samples, n_samples)
            Block diagonal matrix for demeaning group effects.
        - X: np.ndarray, shape (n_samples, n_features)
            Design matrix.
        - y: np.ndarray, shape (n_samples,)
            Response variable.

        Returns:
        - beta: np.ndarray, shape (n_features, 1)
            Estimated regression coefficients.
        """
        QX = Q @ X
        Qy = Q @ y.reshape(-1, 1)
        beta = np.linalg.solve(QX.T @ QX, QX.T @ Qy)
        return beta

    def _estimate_sigma(
        self, residuals: np.ndarray, n_samples: int, n_groups: int, n_features: int
    ) -> float:
        """
        Estimate the standard deviation of residuals (sigma).

        Parameters:
        - residuals: np.ndarray, shape (n_samples,)
            Residuals from the model.
        - n_samples: int
            Total number of samples.
        - n_groups: int
            Number of groups.
        - n_features: int
            Number of predictors.

        Returns:
        - sigma: float
            Estimated standard deviation of residuals.
        """
        residual_sum_squares = np.sum(residuals**2)
        sigma_hat_sq = residual_sum_squares / (n_samples - n_groups - n_features)
        return np.sqrt(sigma_hat_sq)

    def _estimate_variances(
        self, Q: np.ndarray, X: np.ndarray, X_means: np.ndarray, beta: np.ndarray, group_sizes: np.ndarray
    ) -> dict:
        """
        Estimate variances of beta and gamma coefficients for inferential statistics.

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
        
    def fit(self, X, y=None, groups=None, x_vars=None, y_var=None, group_var=None) -> "LinearFixedEffectModel":
        """
        Fit the LinearFixedEffect model.

        Parameters:
        - X: array-like, shape (n_samples, n_features) or pd.DataFrame
            Design matrix (covariates) or complete dataset.
        - y: array-like, shape (n_samples,) or None
            Response variable or None if X is a DataFrame.
        - groups: array-like, shape (n_samples,) or None
            Group identifiers for fixed effects or None if X is a DataFrame.
        - x_vars: list of str, optional
            Column names in X to be used as predictors.
        - y_var: str, optional
            Column name in X to be used as the response variable.
        - group_var: str, optional
            Column name in X to be used as group identifiers.

        Returns:
        - self: LinearFixedEffectModel
            The fitted model instance.
        """
        X, y, groups = self._validate_and_convert_inputs(X, y, groups, x_vars, y_var, group_var)

        self.outcome_ = y

        self.groups_, self.group_indices_ = np.unique(groups, return_inverse=True)
        self.group_sizes_ = np.bincount(self.group_indices_)
        n_groups = len(self.groups_)
        n_samples, n_features = X.shape

        # Group preprocessing: Block diagonal matrix, group means
        Q, y_means, X_means = self._preprocess_groups(X, y, self.group_indices_, n_groups)

        # Weighted least squares
        beta = self._perform_weighted_least_squares(Q, X, y)

        self.xbeta_ = X @ beta  # Storing the linear predictor

        # Calculate fixed effects
        gamma = y_means.reshape(-1, 1) - X_means @ beta

        # Store results
        self.coefficients_ = {"beta": beta, "gamma": gamma}
        self.fitted_, self.residuals_ = self._calculate_residuals(self.xbeta_, gamma, y, self.group_indices_)

        self.sigma_ = self._estimate_sigma(self.residuals_, n_samples, n_groups, n_features)

        # Compute variances and statistics
        self.variances_ = self._estimate_variances(Q, X, X_means, beta, self.group_sizes_)

        self.aic_, self.bic_ = self._compute_model_statistics(self.residuals_, n_samples, n_groups, n_features)

        return self

    def predict(self, X, groups=None, x_vars=None, group_var=None) -> np.ndarray:
        """
        Predict using the LinearFixedEffect model.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Design matrix (covariates) or complete dataset.
        groups : array-like or None
            Group identifiers or None if `group_var` is specified in a DataFrame.
        x_vars : list of str, optional
            Column names in X to be used as predictors, required if X is a DataFrame.
        group_var : str, optional
            Column name in X to be used as group identifiers, required if X is a DataFrame.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self.coefficients_ is None:
            raise ValueError("Model is not fitted. Call 'fit' before 'predict'.")

        # Validate and convert inputs
        X, _, groups = self._validate_and_convert_inputs(X, None, groups, x_vars, None, group_var)

        # Align groups with fitted model's groups, need to fix this one
        group_indices = np.searchsorted(self.groups_, groups)

        # Retrieve regression coefficients
        beta = self.coefficients_["beta"]
        gamma = self.coefficients_["gamma"]

        # Calculate predictions
        xbeta = X @ beta
        gamma_obs = gamma[group_indices].flatten()
        predictions = xbeta.flatten() + gamma_obs
        
        return predictions

    def score(self, X, y, groups) -> float:
        """
        Compute the R^2 score for the model.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Design matrix (covariates).
        - y: array-like, shape (n_samples,)
            True target values.
        - groups: array-like, shape (n_samples,)
            Group identifiers for the fixed effects.

        Returns:
        - r2: float
            R^2 score of the model.
        """
        y_pred = self.predict(X, groups)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        return 1 - (ss_residual / ss_total)

    def get_params(self) -> dict:
        """
        Return model parameters.

        Returns:
        - params: dict, Model parameters including coefficients, variances, sigma, AIC, and BIC.
        """
        return {
            "coefficients": self.coefficients_,
            "variances": self.variances_,
            "sigma": self.sigma_,
            "aic": self.aic_,
            "bic": self.bic_,
        }

    def summary(
        self,
        covariates: list = None,
        level: float = 0.95,
        null: float = 0,
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """
        Provides summary statistics for the covariate estimates in a fitted fixed effects model.

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
        if self.coefficients_ is None:
            raise ValueError("The model must be fitted before calling 'summary'.")

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

        # Format p-values for readability.
        p_val_formatted = [f"{min(pv, 1.0):.7g}" for pv in p_val]

        # Use self.covariate_names_ (or assign default names if not available)
        cov_names = getattr(self, "covariate_names_", [f"X{i}" for i in range(p)])

        # Build the summary DataFrame using lower-case column names.
        summary_df = pd.DataFrame({
            "estimate": beta,
            "std_error": se_beta,
            "stat": stat,
            "p_value": p_val_formatted,
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

    def calculate_standardized_measures(
        self, group_ids=None, stdz="indirect", null="median"
    ) -> dict:
        """
        Calculate direct/indirect standardized differences for a fixed effects linear model.

        Parameters
        ----------
        group_ids : Optional[list or np.ndarray]
            The specific groups or provider identifiers for which the differences should be calculated.
            If None, calculates for all groups.
        stdz : Union[str, list], default="indirect"
            Methods for standardization; can be "indirect", "direct", or both.
        null : Union[str, float], default="median"
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
        group_sizes = self.group_sizes_
        
        # Determine the null value for gamma
        if null == "median":
            gamma_null = np.median(gamma)
        elif null == "mean":
            gamma_null = np.average(gamma, weights=group_sizes)
        elif isinstance(null, (int, float)):
            gamma_null = null
        else:
            raise ValueError("Invalid 'null' argument provided. Must be 'median', 'mean', or a numeric value.")
        
        # If group_ids are specified, select those groups; otherwise, use all groups.
        if group_ids is not None:
            mask = np.isin(self.groups_, group_ids)
            selected_groups = self.groups_[mask]
        else:
            selected_groups = self.groups_
        
        results = {}
        
        # Indirect Standardization
        if "indirect" in stdz:
            # For each observation, expected outcome = gamma_null + linear predictor.
            expected = gamma_null + self.xbeta_.flatten()
            # Sum expected and observed by group.
            expected_by_group = np.array([
                expected[self.group_indices_ == i].sum() 
                for i in range(len(self.groups_))
            ])
            observed_by_group = np.array([
                self.outcome_[self.group_indices_ == i].sum() 
                for i in range(len(self.groups_))
            ])
            # Standardized difference is the (Obs - Exp) divided by the group size.
            indirect_diff = (observed_by_group - expected_by_group) / group_sizes
            
            indirect_df = pd.DataFrame({
                "group_id": self.groups_,
                "indirect_difference": indirect_diff,
                "observed": observed_by_group,
                "expected": expected_by_group
            })
            if group_ids is not None:
                indirect_df = indirect_df[indirect_df['group_id'].isin(selected_groups)].reset_index(drop=True)
            results["indirect"] = indirect_df
        
        # Direct Standardization
        if "direct" in stdz:
            # Overall observed is the sum over all observations of (gamma_null + xbeta).
            obs_direct_total = (gamma_null + self.xbeta_.flatten()).sum()
            # For each group, compute expected sum using the group-specific gamma.
            exp_direct_by_group = np.array([
                np.sum(gamma_val + self.xbeta_) for gamma_val in gamma
            ])
            # Standardized difference is (expected_by_group - overall observed) divided by total sample size.
            direct_diff = (exp_direct_by_group - obs_direct_total) / n_samples
            
            direct_df = pd.DataFrame({
                "group_id": self.groups_,
                "direct_difference": direct_diff,
                "observed": np.full(len(self.groups_), obs_direct_total),
                "expected": exp_direct_by_group
            })
            if group_ids is not None:
                direct_df = direct_df[direct_df['group_id'].isin(selected_groups)].reset_index(drop=True)
            results["direct"] = direct_df
            
        return results
    
    # --- Confidence Intervals ---
    def _compute_ci_bounds(self, gamma: np.ndarray, se: np.ndarray, df: int, level: float, alternative: str) -> tuple:
        """
        Compute lower and upper bounds for confidence intervals given estimates,
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
        group_ids=None,
        level: float = 0.95,
        option: str = "SM",
        stdz: Union[str, list] = "indirect",
        null: Union[str, float] = "median",
        alternative: str = "two_sided"
    ) -> dict:
        """
        Calculate confidence intervals for provider effects (gamma) or standardized measures (SM).

        Parameters
        ----------
        group_ids : Optional[list or np.ndarray]
            Subset of group identifiers for which confidence intervals are calculated.
        level : float, default=0.95
            Confidence level.
        option : str, default="SM"
            Either "gamma" for provider effects or "SM" for standardized measures.
        stdz : Union[str, list], default="indirect"
            Standardization method(s) if option is "SM"; must include "indirect" and/or "direct".
        null : Union[str, float], default="median"
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
                "group_id": self.groups_,
                "gamma": gamma,
                "lower": lower_gamma,
                "upper": upper_gamma
            })
            if group_ids is not None:
                gamma_ci = gamma_ci[gamma_ci["group_id"].isin(group_ids)].reset_index(drop=True)
            result["gamma_ci"] = gamma_ci
        
        if option == "SM":
            # First get the standardized measures
            sm_results = self.calculate_standardized_measures(stdz=stdz, null=null)
            
            # For indirect SM, we need to aggregate CI bounds from gamma over observations.
            # Replicate group-level lower/upper bounds for each observation.
            lower_obs = np.repeat(lower_gamma, self.group_sizes_) + self.xbeta_.flatten()
            upper_obs = np.repeat(upper_gamma, self.group_sizes_) + self.xbeta_.flatten()
            # Sum over each group using np.bincount
            lower_prov = np.bincount(self.group_indices_, weights=lower_obs)
            upper_prov = np.bincount(self.group_indices_, weights=upper_obs)
            
            if "indirect" in stdz:
                indirect_df = sm_results["indirect"].copy()
                expected_indirect = indirect_df["expected"].to_numpy()
                # Standardized difference: (observed - expected) normalized by group size.
                lower_indirect = (lower_prov - expected_indirect) / self.group_sizes_
                upper_indirect = (upper_prov - expected_indirect) / self.group_sizes_
                indirect_df["lower"] = lower_indirect
                indirect_df["upper"] = upper_indirect
                if group_ids is not None:
                    # Filter by group IDs
                    indirect_df = indirect_df[indirect_df[self.char_list["ID.char"]].isin(group_ids)]
                result["indirect_ci"] = indirect_df
            
            if "direct" in stdz:
                # In direct standardization, the standardized difference equals gamma - gamma_null.
                # So the CI for the direct measure is simply:
                # lower_direct = lower_gamma - gamma_null
                # upper_direct = upper_gamma - gamma_null
                if null == "median":
                    gamma_null = np.median(gamma)
                elif null == "mean":
                    gamma_null = np.average(gamma, weights=self.group_sizes_)
                elif isinstance(null, (int, float)):
                    gamma_null = null
                else:
                    raise ValueError("Invalid 'null' argument provided.")
                direct_df = sm_results["direct"].copy()
                lower_direct = lower_gamma - gamma_null
                upper_direct = upper_gamma - gamma_null
                direct_df["lower"] = lower_direct
                direct_df["upper"] = upper_direct
                if group_ids is not None:
                    # Filter direct CIs by group_ids; assume that the direct_df has a column named "group_id"
                    direct_df = direct_df[direct_df["group_id"].isin(group_ids)]
                result["direct_ci"] = direct_df
            
        return result

    def test(
        self,
        providers: Optional[Union[list, np.ndarray]] = None,
        level: float = 0.95,
        null: Union[str, float] = "median",
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """
        Conduct hypothesis tests on provider effects and identify outlying providers.

        Parameters
        ----------
        providers : Optional[Union[list, np.ndarray]]
            A subset of provider IDs for which tests are conducted. If None, tests are conducted for all providers.
        level : float, default=0.95
            Confidence level for the hypothesis tests.
        null : Union[str, float], default="median"
            Null hypothesis value for provider effects; options are "median", "mean", or a numeric value.
        alternative : str, default="two_sided"
            Alternative hypothesis: "two_sided", "greater", or "less".

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns "flag", "p_value", "stat", and "std_error" for each provider.
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("The model must be fitted before testing.")
        
        alpha = 1 - level
        gamma = self.coefficients_["gamma"].flatten()   # shape (m,)
        se_gamma = np.sqrt(self.variances_["gamma"].flatten())
        n_prov = len(gamma)
        total_samples = self.fitted_.size
        p = len(self.coefficients_["beta"])
        df = total_samples - p - n_prov

        # Compute gamma_null using the actual group sizes (self.group_sizes_) if needed
        if null == "median":
            gamma_null = np.median(gamma)
        elif null == "mean":
            if self.group_sizes_ is None:
                raise ValueError("Group sizes are not available for computing a weighted mean.")
            gamma_null = np.average(gamma, weights=self.group_sizes_)
        elif isinstance(null, (int, float)):
            gamma_null = null
        else:
            raise ValueError("Argument 'null' must be 'median', 'mean', or a numeric value.")

        # Compute test statistics.
        stat = (gamma - gamma_null) / se_gamma

        # Use the survival function so that a high test statistic yields a small probability,
        # consistent with R's pt(..., lower.tail = F)
        prob = t.sf(stat, df=df)  # equivalent to 1 - t.cdf(stat, df=df)

        # Determine flags and p-values based on the alternative hypothesis.
        if alternative == "two_sided":
            # Flag: 1 if probability < alpha/2, -1 if probability > 1 - alpha/2, else 0.
            flag = np.where(prob < alpha / 2, 1,
                            np.where(prob > 1 - alpha / 2, -1, 0))
            p_value = 2 * np.minimum(prob, 1 - prob)
        elif alternative == "greater":
            flag = np.where(prob < alpha, 1, 0)
            p_value = prob
        elif alternative == "less":
            flag = np.where(1 - prob < alpha, -1, 0)
            p_value = 1 - prob
        else:
            raise ValueError("Argument 'alternative' should be 'two_sided', 'greater', or 'less'.")

        # Build the result DataFrame, using self.groups_ as the index.
        result = pd.DataFrame({
            "flag": pd.Categorical(flag),
            "p_value": np.round(p_value, 7),
            "stat": stat,
            "std_error": se_gamma
        }, index=self.groups_)

        # Filter results if a subset of providers is specified.
        if providers is not None:
            if isinstance(providers, (list, np.ndarray)):
                result = result.loc[result.index.isin(providers)]
            else:
                raise ValueError("Argument 'providers' should be a list or ndarray matching provider IDs.")

        # Optionally, attach the provider sizes as an attribute.
        result.attrs["provider_size"] = dict(zip(self.groups_, self.group_sizes_))
        
        return result
    

    def plot_funnel(
        self,
        stdz: str = "indirect",
        null: Union[str, float] = "median",
        target: float = 0.0,
        alpha: Union[float, List[float]] = 0.05,
        labels: List[str] = ["Lower", "Expected", "Higher"],
        point_colors: List[str] = ["#E69F00", "#56B4E9", "#009E73"],
        point_shapes: List[str] = ['v', 'o', '^'],
        point_size: float = 2.0,
        point_alpha: float = 0.8,
        line_size: float = 0.8,
        target_linestyle: str = '--',
        font_size: float = 12,
        tick_label_size: float = 10,
        cl_line_colors: Optional[Union[str, List[str]]] = "grey",
        cl_line_styles: Optional[Union[str, List[str]]] = None,
        fill_color: str = "#A6CEE3",
        fill_alpha: float = 0.25,
        edge_color: Optional[str] = "grey",
        edge_linewidth: float = 0.5,
        add_grid: bool = True,
        grid_style: str = ':',
        grid_alpha: float = 0.6,
        remove_top_right_spines: bool = True,
        figure_size: Tuple[float, float] = (8, 6),
        plot_title: str = "Funnel Plot of Standardized Differences",
        xlab: str = "Precision (Group Size)",
        ylab: str = "Standardized Difference",
        legend_location: str = 'best'
    ) -> None:
        """
        Create a funnel plot for standardized differences.

        For LinearFixedEffectModel, this plots the indirect standardized difference
        (gamma_i - gamma_null) against group size (as a proxy for precision).
        Control limits are based on the overall model's residual standard deviation (sigma).

        Parameters:
        -----------
        stdz : str, default="indirect"
            Standardization method. Currently, only "indirect" is meaningfully
            supported as both indirect and direct differences simplify to gamma_i - gamma_null.
        null : str or float, default="median"
            Baseline for provider effects (gamma) used in calculating the difference
            and for flagging. Can be "median", "mean", or a specific float value.
        target : float, default=0.0
            Reference value for the difference (target line on the plot).
        alpha : float or List[float], default=0.05
            Significance level(s) for control limits.
        labels : List[str], default=["Lower", "Expected", "Higher"]
            Labels for provider performance categories based on flags (-1, 0, 1).
        point_colors : List[str]
            Colors for provider points based on performance flag.
        point_shapes : List[str]
            Marker shapes for provider points based on performance flag.
        point_size : float, default=2.0
            Scaling factor for marker size.
        point_alpha : float, default=0.8
            Marker transparency.
        line_size : float, default=0.8
            Thickness for target and control limit lines.
        target_linestyle : str, default='--'
            Line style for the target reference line.
        font_size : float, default=12
            Base font size for labels and title.
        tick_label_size : float, default=10
            Font size for axis tick labels.
        cl_line_colors : str or List[str], optional
            Color(s) for the control limit lines. Defaults to "grey".
        cl_line_styles : str or List[str], optional
            Line style(s) for control limits. Defaults based on number of alphas.
        fill_color : str, default="#A6CEE3"
            Fill color for the area between the outermost control limits.
        fill_alpha : float, default=0.25
            Transparency of the control limit fill area.
        edge_color : str or None, default="grey"
            Edge color for scatter points.
        edge_linewidth : float, default=0.5
            Line width for scatter point edges.
        add_grid : bool, default=True
            Whether to add a background grid.
        grid_style : str, default=':'
            Line style for the grid.
        grid_alpha : float, default=0.6
            Transparency for the grid lines.
        remove_top_right_spines : bool, default=True
            Whether to remove the top and right axis lines.
        figure_size : Tuple[float, float], default=(8, 6)
            Figure size in inches.
        plot_title : str, default="Funnel Plot of Standardized Differences"
            Title for the plot.
        xlab : str, default="Precision (Group Size)"
            Label for the x-axis.
        ylab : str, default="Standardized Difference"
            Label for the y-axis.
        legend_location : str, default='best'
            Location string for the legend.
        """
        if self.coefficients_ is None or self.sigma_ is None or self.groups_ is None or self.group_sizes_ is None:
            raise ValueError("Model must be fitted and sigma estimated before plotting funnel plot.")
        if stdz != "indirect":
            warnings.warn("Funnel plot for LinearFixedEffectModel is primarily designed for 'indirect' standardized differences.")

        a_list = sorted([alpha] if isinstance(alpha, (float, int)) else alpha)
        alpha_test = min(a_list)

        sm_info = self.calculate_standardized_measures(stdz=stdz, null=null)
        if stdz not in sm_info or sm_info[stdz].empty:
            print(f"Warning: No standardized measure data found for '{stdz}'. Cannot plot.")
            return
        df = sm_info[stdz].copy()
        if 'group_id' in df.columns: df.set_index('group_id', inplace=True)
        
        precision_map = pd.Series(self.group_sizes_, index=self.groups_)
        df["precision"] = df.index.map(precision_map)
        df.dropna(subset=['precision'], inplace=True)

        test_df = self.test(null=null, level=1.0 - alpha_test, alternative="two_sided")
        df = df.merge(test_df[['flag']], left_index=True, right_index=True, how='left')
        df["flag"] = df["flag"].fillna(0).astype(int)

        limits_list = []
        for a_val in a_list:
            z_val = norm.ppf(1 - a_val / 2)
            se_for_limits = self.sigma_ / np.sqrt(df["precision"])
            se_for_limits.replace([np.inf, -np.inf], np.nan, inplace=True)
            se_for_limits.fillna(0, inplace=True)
            control_lower = target - z_val * se_for_limits
            control_upper = target + z_val * se_for_limits
            limits_df_a = pd.DataFrame({
                "precision": df["precision"], 
                "control_lower": control_lower,
                "control_upper": control_upper, 
                "alpha": a_val
            }, index=df.index)
            limits_list.append(limits_df_a)
        limits_all_alphas = pd.concat(limits_list)

        fig, ax = plt.subplots(figsize=figure_size)
        limits_all_alphas = limits_all_alphas.sort_values("precision")
        outer_alpha = min(a_list)

        if cl_line_styles is None: 
            default_styles = ['-', '--', ':', '-.']
            cl_line_styles = [default_styles[i % len(default_styles)] for i in range(len(a_list))]
        elif isinstance(cl_line_styles, str): 
            cl_line_styles = [cl_line_styles] * len(a_list)

        if cl_line_colors is None or isinstance(cl_line_colors, str): 
            cl_line_colors = [cl_line_colors or 'grey'] * len(a_list)
        
        sorted_alphas = sorted(a_list)
        style_map = dict(zip(sorted_alphas, cl_line_styles)); color_map = dict(zip(sorted_alphas, cl_line_colors))
        legend_handles, legend_labels_list = [], []

        outer_limits = limits_all_alphas[limits_all_alphas["alpha"] == outer_alpha]
        label_outer_ci = f'{int((1-outer_alpha)*100)}% CI'
        fill_handle = ax.fill_between(outer_limits["precision"], outer_limits["control_lower"], outer_limits["control_upper"], color=fill_color, alpha=fill_alpha, label=label_outer_ci)
        legend_handles.append(fill_handle); legend_labels_list.append(label_outer_ci)

        for a_val in sorted_alphas:
            subset_lim = limits_all_alphas[limits_all_alphas["alpha"] == a_val]
            line_label = f'{int((1-a_val)*100)}% CI' if a_val != outer_alpha else None
            line_lower, = ax.plot(subset_lim["precision"], subset_lim["control_lower"], linestyle=style_map[a_val], color=color_map[a_val], linewidth=line_size, label=line_label)
            ax.plot(subset_lim["precision"], subset_lim["control_upper"], linestyle=style_map[a_val], color=color_map[a_val], linewidth=line_size)
            if line_label: legend_handles.append(line_lower); legend_labels_list.append(line_label)
        
        ax.axhline(y=target, color="black", linestyle=target_linestyle, linewidth=line_size)

        present_flags = sorted(df["flag"].unique())
        flag_map = {-1: 0, 0: 1, 1: 2}
        for flag_val in present_flags:
            subset_pts = df[df["flag"] == flag_val]
            label_idx = flag_map.get(flag_val,1)
            count = len(subset_pts)
            point_label = f"{labels[label_idx]} ({count})"
            scatter_handle = ax.scatter(subset_pts["precision"], 
                                        subset_pts[f"{stdz}_difference"],
                                        marker=point_shapes[label_idx % len(point_shapes)], 
                                        color=point_colors[label_idx % len(point_colors)],
                                        s=point_size*30, 
                                        alpha=point_alpha, 
                                        edgecolor=edge_color, 
                                        linewidth=edge_linewidth if edge_color else 0, 
                                        label=point_label)
            legend_handles.append(scatter_handle); 
            legend_labels_list.append(point_label)

        ax.set_xlabel(xlab, fontsize=font_size)
        ax.set_ylabel(ylab, fontsize=font_size)
        ax.set_title(plot_title, fontsize=font_size + 2, pad=15)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        
        all_y_values = pd.concat([df[f"{stdz}_difference"], limits_all_alphas["control_lower"], limits_all_alphas["control_upper"]]).dropna()
        if not all_y_values.empty:
            min_y_val = min(all_y_values.min(), target)
            max_y_val = max(all_y_values.max(), target)
            padding = (max_y_val - min_y_val) * 0.1 if (max_y_val - min_y_val) > 1e-6 else 0.1 # Ensure padding is positive
            ax.set_ylim(min_y_val - padding, max_y_val + padding)
        
        max_x = df["precision"].max(skipna=True)
        ax.set_xlim(left=0, right=(max_x * 1.05 if pd.notna(max_x) and max_x > 0 else 1))

        if add_grid: 
            ax.grid(True, linestyle=grid_style, alpha=grid_alpha, axis='both', color='lightgrey')
        if remove_top_right_spines: 
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
        
        unique_handles_labels = {}; 
        for handle, label in zip(legend_handles, legend_labels_list):
            if label not in unique_handles_labels: 
                unique_handles_labels[label] = handle
        ax.legend(handles=unique_handles_labels.values(), 
                  labels=unique_handles_labels.keys(), 
                  fontsize=font_size - 2, 
                  loc=legend_location, 
                  title="Flag")
        plt.tight_layout()
        plt.show()

    def plot_provider_effects(
        self, 
        group_ids=None, 
        level: float = 0.95,
        use_flags: bool = True, 
        null: Union[str, float] = 'median',
        test_method: Optional[str] = None, # Added for consistency with LogisticFE
        **plot_kwargs
    ) -> None:
        """
        Plots provider fixed effects (gamma) using the plot_caterpillar helper function.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot. If None, all providers are included.
        level : float, default=0.95
            Confidence level for intervals.
        use_flags : bool, default=True
            Whether to color-code providers based on flags from the test method.
        null : str or float, default='median'
            Null hypothesis for gamma used for flagging. Can be 'median', 'mean', or a float.
        test_method : str, optional
             Test method used specifically for generating flags ('wald' is the only one for LinearFE's .test()).
             If None, defaults to 'wald'.
        **plot_kwargs
            Additional arguments passed to plot_caterpillar (e.g., plot_title, orientation).
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("Model must be fitted first.")

        # Get gamma CIs (always two-sided for plotting)
        # For LinearFixedEffectModel, test_method in calculate_CIs is implicitly Wald-like (t-dist)
        ci_results = self.calculate_confidence_intervals(
            group_ids=group_ids, 
            level=level, 
            option='gamma', 
            alternative='two_sided'
        )
        if 'gamma_ci' not in ci_results or ci_results['gamma_ci'].empty:
            print("Warning: No gamma CI data. Cannot plot.")
            return
        
        df_plot = ci_results['gamma_ci'] # Has 'group_id', 'gamma', 'lower', 'upper'

        flag_col_name = None
        if use_flags:
            flag_col_name = 'flag'
            # LinearFEModel's test method is t-test based (Wald-like)
            current_test_method = test_method if test_method else 'wald' # Default for LinearFE
            if current_test_method != 'wald':
                warnings.warn(f"LinearFixedEffectModel.test uses a t-test (Wald-like). test_method '{current_test_method}' for flagging will use this underlying test.")

            try:
                test_df = self.test(
                    providers=df_plot['group_id'].unique().tolist(),
                    level=level, 
                    null=null, 
                    alternative='two_sided'
                )
                # Merge flags using left_on='group_id' and right_index=True since test_df is indexed by provider IDs
                df_plot = df_plot.merge(test_df[['flag']], left_on='group_id', right_index=True, how='left')
                df_plot[flag_col_name] = df_plot[flag_col_name].fillna(0).astype(int)
            except Exception as e:
                warnings.warn(f"Could not generate flags. Plotting without flags. Error: {e}")
                flag_col_name = None
        
        gamma_vals = self.coefficients_["gamma"].flatten()
        if null == "median": gamma_null_val = np.median(gamma_vals)
        elif null == "mean": gamma_null_val = np.average(gamma_vals, weights=self.group_sizes_ if self.group_sizes_ is not None else None)
        else: gamma_null_val = float(null)

        # Default orientation is vertical (groups on Y, estimates on X)
        orientation = plot_kwargs.pop('orientation', 'vertical')
        if orientation == 'vertical':
            plot_kwargs.setdefault('xlab', 'Gamma Estimate (Fixed Effect)')
            plot_kwargs.setdefault('ylab', 'Provider')
        else: # horizontal
            plot_kwargs.setdefault('xlab', 'Provider')
            plot_kwargs.setdefault('ylab', 'Gamma Estimate (Fixed Effect)')

        # Use 'plot_title' instead of 'title' to match plot_caterpillar's parameter
        plot_kwargs.setdefault('plot_title', 'Provider Effects (Gamma)')
        plot_kwargs.setdefault('refline_value', gamma_null_val)
        plot_kwargs.setdefault('orientation', orientation)

        plot_caterpillar(
            df=df_plot, 
            estimate_col='gamma', 
            ci_lower_col='lower', 
            ci_upper_col='upper',
            group_col='group_id', 
            flag_col=flag_col_name, 
            **plot_kwargs
        )

    def plot_standardized_measures(
        self,
        group_ids=None, 
        level: float = 0.95, 
        stdz: str = 'indirect',
        measure: str = 'difference',
        use_flags: bool = True, null: Union[str, float] = 'median',
        test_method: Optional[str] = None,
        **plot_kwargs
    ) -> None:
        """
        Plots standardized differences using plot_caterpillar.
        For LinearFixedEffectModel, standardized measures are differences (gamma_i - gamma_null).

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot.
        level : float, default=0.95
            Confidence level for intervals.
        stdz : str, default='indirect'
            Standardization method ('indirect' or 'direct'). Both result in gamma_i - gamma_null.
        measure : str, default='difference'
            The measure to plot. For linear models, this is always 'difference'.
        use_flags : bool, default=True
            Whether to color-code providers based on flags from the gamma test method.
        null : str or float, default='median'
            Null hypothesis for gamma used for flagging and calculating the difference.
        test_method : str, optional
             Test method used specifically for generating flags. Defaults to 'wald' (t-test).
        **plot_kwargs
            Additional arguments passed to plot_caterpillar.
        """
        if self.coefficients_ is None: raise ValueError("Model must be fitted.")
        if measure != 'difference':
            warnings.warn("For LinearFixedEffectModel, standardized 'measure' is 'difference'.")
        
        # Get SM CIs (which are for the difference: gamma_i - gamma_null)
        ci_results = self.calculate_confidence_intervals(
            group_ids=group_ids, 
            level=level, 
            option='SM', 
            stdz=stdz, 
            null=null, 
            alternative='two_sided'
        )
        ci_key = f"{stdz}_ci"
        if ci_key not in ci_results or ci_results[ci_key].empty:
            print(f"Warning: No SM CI data for '{ci_key}'. Cannot plot.")
            return
        
        df_plot = ci_results[ci_key] # This df should have 'group_id', '{stdz}_difference', 'lower', 'upper'
        estimate_col_name = f"{stdz}_difference"
                
        if estimate_col_name not in df_plot.columns or 'lower' not in df_plot.columns or 'upper' not in df_plot.columns:
             raise ValueError(f"Required columns ('{estimate_col_name}', 'lower', 'upper') not found in SM CI results. Available: {df_plot.columns}")

        flag_col_name = None
        if use_flags:
            flag_col_name = 'flag'
            current_test_method = test_method if test_method else 'wald'
            if current_test_method != 'wald':
                 warnings.warn(f"LinearFixedEffectModel.test uses a t-test (Wald-like). test_method '{current_test_method}' for flagging will use this.")
            try:
                test_df = self.test(providers=df_plot['group_id'].unique().tolist(), 
                                    level=level, 
                                    null=null, 
                                    alternative='two_sided')
                # Merge using left_on='group_id' and right_index=True
                df_plot = df_plot.merge(test_df[['flag']], left_on='group_id', right_index=True, how='left')
                df_plot[flag_col_name] = df_plot[flag_col_name].fillna(0).astype(int)
            except Exception as e:
                warnings.warn(f"Could not generate flags. Plotting without flags. Error: {e}")
                flag_col_name = None

        orientation = plot_kwargs.pop('orientation', 'vertical')
        default_title = f"{stdz.capitalize()} Standardized Difference"
        if orientation == 'vertical':
            default_xlab = f"{stdz.capitalize()} Difference Estimate"
            default_ylab = "Provider"
        else:
            default_xlab = "Provider"
            default_ylab = f"{stdz.capitalize()} Difference Estimate"
        default_refline = 0.0

        plot_kwargs.setdefault('plot_title', default_title)
        plot_kwargs.setdefault('xlab', default_xlab)
        plot_kwargs.setdefault('ylab', default_ylab)
        plot_kwargs.setdefault('refline_value', default_refline)
        plot_kwargs.setdefault('orientation', orientation)

        plot_caterpillar(
            df=df_plot, 
            estimate_col=estimate_col_name,
            ci_lower_col='lower', 
            ci_upper_col='upper',
            group_col='group_id', 
            flag_col=flag_col_name, 
            **plot_kwargs
        )

    def plot_coefficient_forest(
        self,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        refline_value: Optional[float] = 0.0,
        point_color: str = "#34495E",
        point_alpha: float = 0.8,
        edge_color: Optional[str] = None,
        edge_linewidth: float = 0,
        point_size: float = 0.05,
        error_color: str = "#95A5A6",
        capsize: float = 5,
        errorbar_size: float = 0.5,
        errorbar_alpha: float = 0.5,
        line_color: str = "red",
        line_style: str = "--",
        line_size: float = 0.8,
        font_size: float = 12,
        tick_label_size: float = 10,
        add_grid: bool = True,
        grid_style: str = ":",
        grid_alpha: float = 0.6,
        remove_top_right_spines: bool = True,
        figure_size: Tuple[float, float] = (8, 6),
        plot_title: str = "Forest Plot of Covariate Coefficients",
        xlab: str = "Coefficient Estimate",
        ylab: str = "Covariate",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> None:
        """
        Create a forest plot of covariate coefficients with 95% confidence intervals.

        Plots each covariate's coefficient estimate and its confidence interval
        in a vertical or horizontal layout, with a reference line at a specified value.

        Parameters
        ----------
        orientation : {'vertical','horizontal'}, default 'vertical'
            'vertical': covariate names on the y-axis, estimates on the x-axis.
            'horizontal': covariate names on the x-axis, estimates on the y-axis.
        refline_value : float or None, default 0.0
            Draws a reference line at this value (vertical or horizontal). None disables it.
        point_color : str, default "#34495E"
            Color of the coefficient marker.
        point_alpha : float, default 0.8
            Opacity of the coefficient marker.
        edge_color : str or None, default None
            Edge color of the marker. None for no edge.
        edge_linewidth : float, default 0
            Width of the marker edge.
        point_size : float, default 0.5
            Scale factor for marker size.
        error_color : str, default "#95A5A6"
            Color of the error bars.
        capsize : float, default 5
            Cap size for error bars.
        errorbar_size : float, default 0.5
            Thickness of the error bar lines.
        errorbar_alpha : float, default 0.5
            Opacity of the error bars.
        line_color : str, default "red"
            Color of the reference line.
        line_style : str, default "--"
            Line style of the reference line.
        line_size : float, default 0.8
            Thickness of the reference line.
        font_size : float, default 12
            Font size for labels and title.
        tick_label_size : float, default 10
            Font size for tick labels.
        add_grid : bool, default True
            Whether to draw a light grid.
        grid_style : str, default ":"
            Line style for the grid.
        grid_alpha : float, default 0.6
            Opacity of the grid.
        remove_top_right_spines : bool, default True
            Hide the top and right spines.
        figure_size : tuple, default (10, 6)
            Size of the figure in inches.
        plot_title : str, default "Forest Plot of Covariate Coefficients"
            Title of the plot.
        xlab : str, default "Coefficient Estimate"
            Label for the x-axis (or y-axis if horizontal).
        ylab : str, default "Covariate"
            Label for the y-axis (or x-axis if horizontal).
        save_path : str or None, default None
            File path to save the figure. If None, the plot is shown.
        dpi : int, default 300
            Resolution (dots per inch) for saving.

        Raises
        ------
        ValueError
            If the model is not fitted or if an invalid orientation is provided.
        """
        # Preconditions
        if self.coefficients_ is None or self.variances_ is None or self.covariate_names_ is None:
            raise ValueError("Model must be fitted before plotting coefficients.")
        if orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        # Compute estimates and 95% CIs
        beta = self.coefficients_["beta"].flatten()
        se_beta = np.sqrt(np.diag(self.variances_["beta"]))
        df_denom = self.fitted_.size - len(beta) - len(self.coefficients_["gamma"])
        crit = t.ppf(1 - 0.05 / 2, df_denom)
        lower = beta - crit * se_beta
        upper = beta + crit * se_beta

        coef_df = (
            pd.DataFrame({
                "covariate": self.covariate_names_,
                "estimate": beta,
                "ci_lower": lower,
                "ci_upper": upper
            })
            .sort_values("estimate")
            .reset_index(drop=True)
        )

        # Positions
        n = len(coef_df)
        positions = np.arange(n)

        # Prepare coordinates and errors
        if orientation == "vertical":
            x_vals, y_vals = coef_df["estimate"], positions
            xerr = [
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"]
            ]
        else:
            x_vals, y_vals = positions, coef_df["estimate"]
            yerr = [
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"]
            ]

        # Plot setup
        fig, ax = plt.subplots(figsize=figure_size)

        # Draw errorbars and points
        if orientation == "vertical":
            ax.errorbar(
                x_vals, y_vals,
                xerr=xerr,
                fmt="o",
                color=point_color,
                ecolor=error_color,
                capsize=capsize,
                markersize=point_size * 30,
                alpha=point_alpha,
                linewidth=edge_linewidth if edge_color else 0,
                markeredgecolor=edge_color
            )
        else:
            ax.errorbar(
                x_vals, y_vals,
                yerr=yerr,
                fmt="o",
                color=point_color,
                ecolor=error_color,
                capsize=capsize,
                markersize=point_size * 30,
                alpha=point_alpha,
                linewidth=edge_linewidth if edge_color else 0,
                markeredgecolor=edge_color
            )

        # Reference line
        if refline_value is not None:
            if orientation == "vertical":
                ax.axvline(refline_value, color=line_color, linestyle=line_style, linewidth=line_size)
            else:
                ax.axhline(refline_value, color=line_color, linestyle=line_style, linewidth=line_size)

        # Labels, ticks, grid
        if orientation == "vertical":
            ax.set_xlabel(xlab, fontsize=font_size)
            ax.set_ylabel(ylab, fontsize=font_size)
            ax.set_yticks(positions)
            ax.set_yticklabels(coef_df["covariate"], fontsize=tick_label_size)
            ax.tick_params(axis="x", labelsize=tick_label_size)
            if add_grid:
                ax.grid(True, axis="x", linestyle=grid_style, alpha=grid_alpha, color="lightgrey")
        else:
            ax.set_xlabel(ylab, fontsize=font_size)
            ax.set_ylabel(xlab, fontsize=font_size)
            ax.set_xticks(positions)
            ax.set_xticklabels(coef_df["covariate"], rotation=45, ha="right", fontsize=tick_label_size)
            ax.tick_params(axis="y", labelsize=tick_label_size)
            if add_grid:
                ax.grid(True, axis="y", linestyle=grid_style, alpha=grid_alpha, color="lightgrey")

        # Spines
        if remove_top_right_spines:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        if orientation == "vertical":
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_linewidth(0.8)
        else:
            ax.spines["bottom"].set_linewidth(0.8)
            ax.spines["left"].set_linewidth(0.8)

        # Title & layout
        ax.set_title(plot_title, fontsize=font_size + 2, pad=15)
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def plot_residuals(
        self,
        figsize: tuple = (8, 5),
        point_color: str = "#1F78B4",
        point_alpha: float = 0.6,
        edge_color: Optional[str] = "grey",
        edge_linewidth: float = 0.5,
        point_size: float = 30,
        line_color: str = "red",
        line_style: str = "--",
        line_width: float = 1.5,
        xlabel: str = "Fitted Values",
        ylabel: str = "Residuals",
        title: str = "Residuals vs. Fitted Values",
        font_size: float = 12,
        tick_label_size: float = 10,
        add_grid: bool = True,
        grid_style: str = ':',
        grid_alpha: float = 0.6,
        remove_top_right_spines: bool = True
    ) -> None:
        """
        Plots residuals versus fitted values.

        This diagnostic plot helps assess model assumptions such as linearity
        and homoscedasticity (constant variance of errors). Ideally, the points
        should show no discernible pattern and be randomly scattered around the
        horizontal line at zero.

        Parameters
        ----------
        figsize : tuple, default=(8, 5)
            Size of the figure to create.
        point_color : str, default="#1F78B4"
            Color for the residual points.
        point_alpha : float, default=0.6
            Transparency level for the points.
        edge_color : str or None, default="grey"
            Edge color for the points. None means no edge.
        edge_linewidth : float, default=0.5
            Width of the point edges if `edge_color` is specified.
        point_size : float, default=30
            Size of the scatter plot markers.
        line_color : str, default="red"
            Color of the horizontal reference line at zero.
        line_style : str, default="--"
            Line style for the reference line.
        line_width : float, default=1.5
            Width of the reference line.
        xlabel : str, default="Fitted Values"
            Label for the x-axis.
        ylabel : str, default="Residuals"
            Label for the y-axis.
        title : str, default="Residuals vs. Fitted Values"
            Title of the plot.
        font_size : float, default=12
            Base font size for labels and title.
        tick_label_size : float, default=10
            Font size for axis tick labels.
        add_grid : bool, default=True
            Whether to add a background grid.
        grid_style : str, default=':'
            Line style for the grid.
        grid_alpha : float, default=0.6
            Transparency for the grid lines.
        remove_top_right_spines : bool, default=True
            Whether to remove the top and right plot spines.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.fitted_ is None or self.residuals_ is None:
            raise ValueError("Model must be fitted before plotting residuals.")

        # Create a new figure and axes for the plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            self.fitted_.flatten(), self.residuals_.flatten(),
            color=point_color, alpha=point_alpha,
            edgecolor=edge_color, linewidth=edge_linewidth if edge_color else 0,
            s=point_size
        )
        ax.axhline(0, color=line_color, linestyle=line_style, linewidth=line_width)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.set_title(title, fontsize=font_size + 2, pad=15)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if add_grid: ax.grid(True, linestyle=grid_style, alpha=grid_alpha, color='lightgrey')
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8); ax.spines['bottom'].set_linewidth(0.8)

        plt.tight_layout()
        plt.show()
        # No return value

    def plot_qq(
        self,
        figsize: tuple = (7, 6),
        title: str = "Normal Q-Q Plot of Residuals",
        xlabel: str = "Theoretical Normal Quantiles",
        ylabel: str = "Ordered Residuals",
        font_size: float = 12,
        tick_label_size: float = 10,
        point_color: str = "#1F78B4",
        line_color: str = "red",
        add_grid: bool = False,
        remove_top_right_spines: bool = True
    ) -> None:
        """
        Creates a Q-Q plot of residuals against a Normal distribution.

        This plot helps assess the assumption of normally distributed errors,
        which is important for the validity of t-tests and confidence intervals
        in linear models. Points falling approximately along the diagonal line
        suggest normality.

        Parameters
        ----------
        figsize : tuple, default=(7, 6)
            Size of the figure to create.
        title : str, default="Normal Q-Q Plot of Residuals"
            Title of the plot.
        xlabel : str, default="Theoretical Normal Quantiles"
            Label for the x-axis.
        ylabel : str, default="Ordered Residuals"
            Label for the y-axis.
        font_size : float, default=12
            Base font size for labels and title.
        tick_label_size : float, default=10
            Font size for axis tick labels.
        point_color : str, default="#1F78B4"
            Color for the points representing residuals.
        line_color : str, default="red"
            Color for the diagonal reference line.
        add_grid : bool, default=False
            Whether to add a background grid.
        remove_top_right_spines : bool, default=True
            Whether to remove the top and right plot spines.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.residuals_ is None:
            raise ValueError("Model must be fitted before plotting QQ plot.")

        # Create a new figure and axes for the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Ensure residuals are 1D array
        residuals_flat = self.residuals_.flatten()

        # Create the Q-Q plot using scipy.stats.probplot
        try:
            (osm, osr), (slope, intercept, r_sq) = probplot(residuals_flat, dist="norm", fit=True, plot=None)
            # Plot the ordered residuals against theoretical quantiles
            ax.plot(osm, osr, 'o', color=point_color, markersize=5, alpha=0.7)
            # Plot the fitted line
            ax.plot(osm, slope * osm + intercept, color=line_color, linestyle='-', linewidth=1.5)
        except Exception as e:
            warnings.warn(f"Could not generate Q-Q plot data, possibly due to issues with residuals: {e}")
            # Plot empty axes as placeholder if data generation fails
            pass

        ax.set_title(title, fontsize=font_size + 2, pad=15)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if add_grid: ax.grid(True, linestyle=':', alpha=0.6, color='lightgrey')
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8); ax.spines['bottom'].set_linewidth(0.8)

        plt.tight_layout()
        plt.show()
