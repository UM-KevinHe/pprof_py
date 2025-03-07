from typing import Optional, Union
from scipy.stats import norm, t, probplot
from scipy.linalg import block_diag
import numpy as np
import pandas as pd
from sklearn.utils import check_array
import matplotlib.pyplot as plt

from .base_model import BaseModel
from .mixins import SummaryMixin, PlotMixin, TestMixin


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
        stdz: Union[str, List[str]] = "indirect",
        null: Union[str, float] = "median",
        target: float = 0,
        alpha: Union[float, List[float]] = 0.05,
        labels: List[str] = ["lower", "expected", "higher"],
        # Options for indirect:
        point_colors: List[str] = ["#D73027", "#4575B4", "#1A9850"],
        point_shapes: List[str] = ['o', 's', 'D'],
        # Options for direct:
        direct_marker: str = 'X',
        direct_color: str = "purple",
        # Common plotting parameters:
        point_size: float = 2,
        point_alpha: float = 0.85,
        line_size: float = 2,
        target_linestyle: str = 'dashdot',
        font_size: float = 12,
        tick_label_size: float = 10  # Tick labels
    ) -> None:
        """
        Create a funnel plot comparing provider performance.
        
        This method plots either the indirect standardized differences, the direct standardized differences,
        or both—in separate subplots if requested via the `stdz` parameter.
        
        Parameters:
            stdz: Either a string ("indirect" or "direct") or a list of such strings.
                - "indirect": Plots the indirect standardized differences (with control limits).
                - "direct": Plots the direct standardized differences (provider effect minus baseline).
            null: Baseline for provider effects (e.g. "median" or a numeric value).
            target: The reference performance value (drawn as a horizontal line).
            alpha: Significance level(s) for control limits (can be a float or list of floats).
            labels: Labels for provider performance categories (used for indirect).
            point_colors: Colors for provider points (for indirect).
            point_shapes: Marker shapes for provider points (for indirect).
            direct_marker: Marker shape for direct standardized differences.
            direct_color: Color for direct standardized differences.
            point_size: Scaling factor for marker size.
            point_alpha: Marker transparency.
            line_size: Thickness for lines.
            target_linestyle: Line style for the target reference line.
        """
        # Ensure stdz is a list.
        if isinstance(stdz, str):
            stdz = [stdz]
        # Determine how many plots to produce.
        n_plots = len(stdz)
        
        # Create a figure with one axis if only one measure is requested,
        # or a subplot with one row and n_plots columns if more than one.
        if n_plots == 1:
            fig, axes = plt.subplots(figsize=(10, 6))
            axes = [axes]
        else:
            fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 6))
        
        # Loop over each requested standardization method.
        for i, method in enumerate(stdz):
            ax = axes[i]
            if method == "indirect":
                # --- Indirect Standardization Plot ---
                # Calculate indirect standardized measures.
                sm_results = self.calculate_standardized_measures(stdz="indirect", null=null)
                indirect_df: pd.DataFrame = sm_results["indirect"].copy()
                indirect_df.index = self.groups_
                indirect_df["precision"] = self.group_sizes_

                # Merge provider flags from test() (assumed to be indexed by group)
                test_df = self.test(null=null, level=0.95)
                indirect_df["flag"] = test_df["flag"]

                # Ensure alpha is a list.
                a_list = [alpha] if isinstance(alpha, (float, int)) else sorted(alpha)
                # Compute control limits based on sigma and precision.
                limits_list = []
                for a in a_list:
                    z_val = norm.ppf(1 - a / 2)
                    control_lower = target - z_val * self.sigma_ / np.sqrt(indirect_df["precision"])
                    control_upper = target + z_val * self.sigma_ / np.sqrt(indirect_df["precision"])
                    limits_df = pd.DataFrame({
                        "precision": indirect_df["precision"],
                        "control_lower": control_lower,
                        "control_upper": control_upper,
                        "alpha": a
                    }, index=self.groups_)
                    limits_list.append(limits_df)
                limits_df = pd.concat(limits_list)

                # Plot control limit areas.
                for a in a_list:
                    subset = limits_df[limits_df["alpha"] == a]
                    ax.fill_between(subset["precision"], subset["control_lower"], subset["control_upper"],
                                    color="#A6CEE3", alpha=0.25,
                                    label=f"Indirect limits (α={a})" if a == min(a_list) else "")

                # Map flags to colors and shapes.
                unique_flags = np.sort(indirect_df["flag"].unique())
                colors_map = {flag: point_colors[i % len(point_colors)] for i, flag in enumerate(unique_flags)}
                shapes_map = {flag: point_shapes[i % len(point_shapes)] for i, flag in enumerate(unique_flags)}

                # Plot provider points.
                for flag in unique_flags:
                    subset = indirect_df[indirect_df["flag"] == flag]
                    label_idx = int(flag + 1)  # if flags are -1, 0, 1
                    ax.scatter(subset["precision"], subset["indirect_difference"],
                            color=colors_map[flag],
                            marker=shapes_map[flag],
                            s=point_size * 50,
                            alpha=point_alpha,
                            edgecolor="k",
                            label=f"{labels[label_idx]} (Indirect, n={len(subset)})")
                ax.set_ylabel("Indirect Standardized Difference", fontsize=font_size)
                ax.set_title("Funnel Plot (Indirect Standardization)", fontsize=font_size + 2)
            
            elif method == "direct":
                # --- Direct Standardization Plot ---
                # Compute direct standardized measures.
                sm_results = self.calculate_standardized_measures(stdz="direct", null=null)
                direct_df: pd.DataFrame = sm_results["direct"].copy()
                direct_df.index = self.groups_
                # For direct standardization in the fixed-effects context,
                # the standardized difference is computed as: gamma - gamma_null.
                # Here we simply plot these values against precision (group sizes).
                ax.scatter(self.group_sizes_,  # using group size as a proxy for precision
                        direct_df["direct_difference"],
                        color=direct_color,
                        marker=direct_marker,
                        s=point_size * 50,
                        alpha=point_alpha,
                        edgecolor="k",
                        label="Direct Standardization")
                ax.set_ylabel("Direct Standardized Difference", fontsize=font_size)
                ax.set_title("Funnel Plot (Direct Standardization)", fontsize=font_size + 2)
            
            else:
                raise ValueError("stdz must be 'indirect' and/or 'direct'.")
            
            # Common plotting settings.
            ax.axhline(y=target, color="black", linestyle=target_linestyle, linewidth=line_size)
            ax.set_xlabel("Precision (Group Size)", fontsize=font_size)
            ax.tick_params(axis="both", labelsize=tick_label_size)  # Adjust tick labels
            ax.legend(fontsize=font_size - 2)
        
        plt.tight_layout()
        plt.show()

    def plot_residuals(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (8, 5),
        point_color: str = "#1F78B4",
        point_alpha: float = 0.75,
        edge_color: str = "k",
        point_size: float = 50,  # new parameter to control point size
        line_color: str = "red",
        line_style: str = "--",
        line_width: float = 2,
        xlabel: str = "Fitted Values",
        ylabel: str = "Residuals",
        title: str = "Residuals vs. Fitted Values",
        font_size: float = 12,
        tick_label_size: float = 10  # Tick labels
    ) -> None:
        """
        Plot residuals versus fitted values to assess model fit.

        Parameters:
            ax: Optional matplotlib Axes to plot on.
            figsize: Size of the figure if a new figure is created.
            point_color: Color for residual points.
            point_alpha: Transparency for residual points.
            edge_color: Edge color for residual points.
            point_size: Marker size for the points.
            line_color: Color for the horizontal reference line.
            line_style: Style for the reference line.
            line_width: Line width for the reference line.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Plot title.

        Returns:
            None. Displays a scatter plot of residuals.
        """
        if self.fitted_ is None or self.residuals_ is None:
            raise ValueError("Model must be fitted before plotting residuals.")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            self.fitted_.flatten(),
            self.residuals_.flatten(),
            color=point_color,
            alpha=point_alpha,
            edgecolor=edge_color,
            s=point_size
        )
        ax.axhline(0, color=line_color, linestyle=line_style, linewidth=line_width)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.set_title(title, fontsize=font_size + 2)
        ax.tick_params(axis="both", labelsize=tick_label_size)  # Adjust tick labels
        plt.tight_layout()
        plt.show()


    def plot_qq(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (8, 6),
        title: str = "Q-Q Plot of Residuals",
        xlabel: str = "Theoretical Quantiles",
        ylabel: str = "Ordered Residuals",
        font_size: float = 12,
        tick_label_size: float = 10  # Tick labels
    ) -> None:
        """
        Create a Q-Q plot of the residuals to assess normality.

        Parameters:
            ax: Optional matplotlib Axes to plot on.
            figsize: Size of the figure if a new figure is created.
            title: Plot title.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.

        Returns:
            None. Displays a Q-Q plot.
        """
        if self.residuals_ is None:
            raise ValueError("Model must be fitted before plotting QQ plot.")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        probplot(self.residuals_.flatten(), dist="norm", plot=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)  # Adjust tick labels
        plt.tight_layout()
        plt.show()

    def plot_provider_effects(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (8, 6),
        point_color: str = "#475569",      
        error_color: str = "#94a3b8",       
        marker_size: float = 6,            
        capsize: float = 0,     
        font_size: float = 12,
        tick_label_size: float = 10,  # Tick labels          
        hline_color: str = "#64748b",
        hline_style: str = "dashed", 
        hline_width: float = 1,            # Reference line width
        xlabel: str = "Provider Effect Estimate",
        ylabel: str = "Provider",
        title: str = "Caterpillar Plot of Provider Effects"
    ) -> None:
        """
        Create a caterpillar (dot-and-error bar) plot for provider effects.
        
        Providers are sorted by their estimated effect (γ) and displayed with 95%
        confidence intervals. The default style (colors, line types, and sizes) are chosen
        to mimic the R version of the caterpillar plot.

        Parameters:
            ax: Optional matplotlib Axes object to plot on.
            figsize: Size of the figure if a new figure is created.
            point_color: Color for provider points (default "#475569").
            error_color: Color for error bars (default "#94a3b8").
            marker_size: Size of the marker points (default 6).
            capsize: Size of error bar caps (default 0 to mimic R's errorbar_width = 0).
            hline_color: Color for the vertical reference line (default "#64748b").
            hline_style: Style for the reference line (default "dashed").
            hline_width: Line width for the reference line (default 1).
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Plot title.
            
        Returns:
            None. Displays the caterpillar plot.
        """
        if self.coefficients_ is None or self.variances_ is None or self.groups_ is None:
            raise ValueError("Model must be fitted before plotting provider effects.")

        # Extract provider effects and compute their standard errors and 95% CIs.
        gamma = self.coefficients_["gamma"].flatten()
        se_gamma = np.sqrt(self.variances_["gamma"].flatten())
        groups = self.groups_
        df_model = self.fitted_.size - len(self.coefficients_["beta"]) - len(gamma)
        crit = t.ppf(0.975, df_model)
        ci_lower = gamma - crit * se_gamma
        ci_upper = gamma + crit * se_gamma

        # Create a DataFrame with provider information.
        df_gamma = pd.DataFrame({
            "group": groups,
            "gamma": gamma,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })
        # Sort providers by their effect estimate.
        df_gamma_sorted = df_gamma.sort_values("gamma").reset_index(drop=True)
        # Use the sorted index as y-axis positions.
        y_positions = np.arange(len(df_gamma_sorted))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot error bars for each provider.
        ax.errorbar(
            df_gamma_sorted["gamma"],
            y_positions,
            xerr=[df_gamma_sorted["gamma"] - df_gamma_sorted["ci_lower"],
                df_gamma_sorted["ci_upper"] - df_gamma_sorted["gamma"]],
            fmt="o",
            color=point_color,
            ecolor=error_color,
            capsize=capsize,
            markersize=marker_size
        )
        # Draw a vertical reference line at 0.
        ax.axvline(0, color=hline_color, linestyle=hline_style, linewidth=hline_width)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        # Set y-ticks to display the provider names in sorted order.
        # Remove Y-axis labels (both ticks and labels)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both", labelsize=tick_label_size)  # Adjust tick labels
        ax.set_title(title, fontsize=font_size+ 2)
        plt.tight_layout()
        plt.show()

    def plot_coefficient_forest(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (10, 6),
        point_color: str = "#34495E",
        error_color: str = "#95A5A6",
        point_size: float = 8,  # controls marker size
        capsize: float = 5,
        hline_color: str = "red",
        hline_style: str = "--",
        hline_width: float = 2,
        xlabel: str = "Coefficient Estimate",
        ylabel: str = "Covariate",
        title: str = "Forest Plot of Covariate Coefficients"
    ) -> None:
        """
        Create a forest plot of covariate coefficients with 95% confidence intervals.

        Parameters:
            ax: Optional matplotlib Axes to plot on.
            figsize: Size of the figure if a new figure is created.
            point_color: Color for the coefficient points.
            error_color: Color for the error bars.
            point_size: Marker size for the points.
            capsize: Size of the error bar caps.
            hline_color: Color for the vertical reference line.
            hline_style: Style for the reference line.
            hline_width: Line width for the reference line.
            xlabel, ylabel, title: Axis labels and plot title.

        Returns:
            None. Displays the forest plot.
        """
        if self.coefficients_ is None or self.variances_ is None or self.covariate_names_ is None:
            raise ValueError("Model must be fitted before plotting coefficients.")
        beta = self.coefficients_["beta"].flatten()
        se_beta = np.sqrt(np.diag(self.variances_["beta"]))
        df = self.fitted_.size - len(beta) - len(self.coefficients_["gamma"])
        crit = t.ppf(0.975, df)
        ci_lower = beta - crit * se_beta
        ci_upper = beta + crit * se_beta

        coef_df = pd.DataFrame({
            "covariate": self.covariate_names_,
            "estimate": beta,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })
        coef_df.sort_values("estimate", inplace=True)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(
            coef_df["estimate"],
            coef_df["covariate"],
            xerr=[coef_df["estimate"] - coef_df["ci_lower"], coef_df["ci_upper"] - coef_df["estimate"]],
            fmt="o",
            color=point_color,
            ecolor=error_color,
            capsize=capsize,
            markersize=point_size
        )
        ax.axvline(0, color=hline_color, linestyle=hline_style, linewidth=hline_width)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()