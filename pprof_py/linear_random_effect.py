import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t, norm
from statsmodels.regression.mixed_linear_model import MixedLMResults
from typing import Optional, Union

from .base_model import BaseModel
from .mixins import SummaryMixin, PlotMixin, TestMixin


class LinearRandomEffectModel(BaseModel, SummaryMixin, PlotMixin, TestMixin):
    """
    Linear Random Effect Model.

    The Linear Random Effect Model is a linear regression model that incorporates
    random effects using a mixed-effects framework (via statsmodels' MixedLM).
    It estimates both fixed effects and random effects. The model can be fitted
    using either a formula-based interface (with a DataFrame of variables) or by
    providing separate design matrices and vectors. This class provides methods
    for model fitting, prediction, generating summary statistics, confidence
    intervals, hypothesis tests, and various diagnostic plots.

    Parameters
    ----------
    X : array-like or pd.DataFrame, optional
        Design matrix (or complete dataset) containing covariates. If a DataFrame is provided,
        then 'x_vars' and 'group_var' must be specified.
    y : array-like or pd.Series, optional
        Response variable (if not provided via y_var when X is a DataFrame).
    groups : array-like or pd.Series, optional
        Group identifiers for random effects (if not provided via group_var when X is a DataFrame).
    x_vars : list of str, optional
        Column names in X to be used as predictors (required if X is a DataFrame).
    y_var : str, optional
        Column name in X to be used as the response variable.
    group_var : str, optional
        Column name in X to be used as group identifiers.
    formula : str, optional
        A formula specifying the model. Used when both formula and data are provided.
    data : pd.DataFrame, optional
        DataFrame containing the variables used in the formula.
    **kwargs : dict
        Additional keyword arguments to be passed to statsmodels' MixedLM.

    Attributes
    ----------
    model : statsmodels.regression.mixed_linear_model.MixedLM or None
        The mixed-effects model instance.
    result : MixedLMResults or None
        The fitted model results.
    coefficients_ : dict
        Dictionary containing model coefficients:
            - fixed_effect: Fixed effects estimates.
            - random_effect: Random effects estimates (converted to a pandas Series).
    variances_ : dict
        Dictionary containing variance matrices:
            - fe_var_cov: Variance-covariance matrix of fixed effects.
            - re_var: Variance (or covariance matrix) of random effects.
    fitted_ : np.ndarray or pd.Series
        Fitted values from the model.
    residuals_ : np.ndarray or pd.Series
        Residuals from the model.
    sigma_ : float
        Estimated standard deviation of the residuals.
    aic_ : float
        Akaike Information Criterion for the model.
    bic_ : float
        Bayesian Information Criterion for the model.
    groups_ : np.ndarray
        Unique group identifiers.
    group_indices_ : np.ndarray
        Array mapping each observation to a group index.
    group_sizes_ : np.ndarray
        Number of observations per group.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'y': [1, 2, 3, 4, 5, 6],
    ...     'x1': [1, 0, 1, 0, 1, 0],
    ...     'x2': [0, 1, 0, 1, 0, 1],
    ...     'group': [1, 1, 2, 2, 3, 3]
    ... })
    >>> model = LinearRandomEffectModel()
    >>> # Using a DataFrame input:
    >>> model.fit(data, x_vars=['x1', 'x2'], y_var='y', group_var='group')
    >>> # model fitting complete.
    >>> predictions = model.predict(data[['x1', 'x2']])
    >>> print(predictions)
    """
    def __init__(self) -> None:
        """
        Initialize the LinearRandomEffectModel.
        Prepares placeholders for the model and its results.
        """
        super().__init__()
        self.model: Optional[sm.MixedLM] = None
        self.result: Optional[sm.regression.mixed_linear_model.MixedLMResults] = None

    def fit(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, 
            groups: Optional[pd.Series] = None, x_vars: Optional[list] = None, 
            y_var: Optional[str] = None, group_var: Optional[str] = None, 
            use_reml: bool = True, **kwargs) -> "LinearRandomEffectModel":
        """
        Fit the random effect linear model.

        Parameters
        ----------
        X : Optional[pd.DataFrame]
            Design matrix (or complete dataset) containing covariates.
            If a DataFrame is provided, then 'x_vars' and 'group_var' must be specified.
        y : Optional[pd.Series]
            Response variable (if not provided via y_var in a DataFrame).
        groups : Optional[pd.Series]
            Group identifiers for random effects (if not provided via group_var in a DataFrame).
        x_vars : Optional[list]
            Column names in X to be used as predictors (required if X is a DataFrame).
        y_var : Optional[str]
            Column name in X to be used as the response variable.
        group_var : Optional[str]
            Column name in X to be used as group identifiers.
        **kwargs : dict
            Additional keyword arguments for statsmodels' MixedLM.

        Returns
        -------
        self : LinearRandomEffectModel
            The fitted model instance.

        Raises
        ------
        ValueError
            If neither a DataFrame (with x_vars and group_var) nor separate X, y, and groups are provided.
        """

        # Validate and convert inputs.
        X, y, groups = self._validate_and_convert_inputs(X, y, groups, x_vars, y_var, group_var)
        
        # Convert X to a DataFrame (if it isn't one already) and ensure the needed columns are there.
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=self.covariate_names_)
        else:
            df = X.copy()

        # Ensure that the response and group columns are present.
        df[y_var] = y
        df[group_var] = groups

        # # (Optional) sort by group to mimic R's ordering.
        # df = df.sort_values(by=group_var)
        
        # Build a formula string.
        # This automatically adds an intercept unless you remove it (e.g., "y ~ 0 + x1 + x2 + ...").
        formula = f"{y_var} ~ " + " + ".join(x_vars)
        
        # Fit the mixed effects model using the formula interface.
        # Note: The groups argument here must be the actual group labels (not the column name),
        # so we pass the appropriate column from the DataFrame.
        self.model = sm.MixedLM.from_formula(formula, data=df, groups=df[group_var], **kwargs)
        self.result = self.model.fit(reml=use_reml)
        self._store_results(y, groups)
        
        # Calculate the fixed-effect linear predictor.
        # This uses the fixed effects coefficients and the design matrix for fixed effects.
        self.xbeta_ = np.dot(self.model.exog, self.result.fe_params)
        print("Model fitting complete.")
        return self

    def _store_results(self, y: pd.Series, groups: pd.Series) -> None:
        """
        Store the results from the fitted model in class attributes.

        Parameters:
        - y: pd.Series
            The response variable used in fitting the model.
        - groups: pd.Series
            The group identifiers used in the model.
        """
        # Convert random effects (a dict) to a Series (assume random intercept model).
        re_series = pd.Series({k: (v[0] if isinstance(v, (np.ndarray, list, pd.Series)) else v)
                               for k, v in self.result.random_effects.items()})
        self.coefficients_ = {
            "fixed_effect": self.result.fe_params,
            "random_effect": re_series
        }

        fixed_effect_names = self.result.fe_params.index
        fe_cov = self.result.cov_params().loc[fixed_effect_names, fixed_effect_names]

        self.variances_ = {
            "fe_var_cov": fe_cov,
            "re_var": self.result.cov_re  # For a random intercept model, a 1x1 matrix.
        }
        self.fitted_ = self.result.fittedvalues
        self.residuals_ = y - self.fitted_
        
        self.sigma_ = np.sqrt(self.result.scale) # REML residual std dev
        self.aic_ = self.result.aic
        self.bic_ = self.result.bic
        self.groups_, self.group_indices_ = np.unique(groups, return_inverse=True)
        self.group_sizes_ = np.bincount(self.group_indices_)

    def predict(
        self, X: Optional[pd.DataFrame] = None, 
        x_vars: Optional[list] = None
    ) -> np.ndarray:
        """
        Predict outcomes for new data using the fitted model.

        Parameters:
        - X: pd.DataFrame
            DataFrame containing new data for prediction. Variables must match those in the model.

        Returns:
        - predictions: np.ndarray
            Array of predicted values based on the fitted model.
        - x_vars : Optional[list]
            Column names in X to be used as predictors (required if X is a DataFrame).
        Raises:
        - ValueError: If the model has not been fitted prior to prediction.
        """
        if X is None:
            raise ValueError("X cannot be None for prediction.")
        if isinstance(X, pd.DataFrame):
            if x_vars is None:
                # Assume the same covariate names as during training.
                x_vars = self.covariate_names_

            X = X[x_vars].to_numpy()
        else:
            X = check_array(X, ensure_2d=True, dtype=np.float64)
        predictions = X @ self.result.fe_params
        return predictions
    
        
    def summary(
        self, covariates: Optional[Union[list, np.ndarray]] = None, 
        level: float = 0.95,
        null: float = 0, 
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """
        Provide summary statistics for the fixed effects in the model.

        Parameters:
        - covariates: Optional[Union[list, np.ndarray]]
            Subset of covariates for which summary statistics are provided. Defaults to all.
        - level: float, default=0.95
            Confidence level for the intervals.
        - null: float, default=0
            Null hypothesis value for the parameter estimates.
        - alternative: str, default="two_sided"
            The alternative hypothesis ("two_sided", "greater", or "less").

        Returns:
        - summary_df: pd.DataFrame
            A DataFrame with columns:
              - estimate
              - std_error
              - stat
              - p_value
              - ci_lower
              - ci_upper
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("model must be fitted before summarizing.")
        
        fe_estimates = self.coefficients_["fixed_effect"]
        se_fe = np.sqrt(np.diag(self.variances_["fe_var_cov"]))
        stat = (fe_estimates - null) / se_fe
        
        n = len(self.fitted_)
        p = len(fe_estimates)
        m = len(self.coefficients_["random_effect"])
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

    def calculate_standardized_measures(
        self, group_ids: Optional[Union[list, np.ndarray]] = None,
        stdz: Union[str, list] = "indirect", 
        null: str = "median"
    ) -> dict:
        """
        Calculate direct/indirect standardized differences for the random effect model.

        Parameters
        ----------
        group_ids : Optional[Union[list, np.ndarray]]
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

        if self.coefficients_ is None or self.fitted_ is None:
            raise ValueError("The model must be fitted before calculating standardized differences.")

        if isinstance(stdz, str):
            stdz = [stdz]

        if not any(method in stdz for method in ["indirect", "direct"]):
            raise ValueError("Argument 'stdz' must include 'indirect' and/or 'direct'.")

        # Extract model components
        random_effects = self.coefficients_["random_effect"]
        group_names = self.groups_
        group_indices = self.group_indices_
        group_sizes = self.group_sizes_
        totalower_samples = len(self.fitted_)

        # Determine the null value for random effects
        if null == "median":
            re_null = np.median(random_effects)
        elif null == "mean":
            re_null = np.average(random_effects, weights=group_sizes)
        elif isinstance(null, (int, float)):
            re_null = null
        else:
            raise ValueError("Invalid 'null' argument provided. Must be 'median', 'mean', or a numeric value.")

        # If group_ids are specified, select those groups; otherwise, use all groups
        if group_ids is not None:
            mask = np.isin(group_names, group_ids)
            selected_groups = group_names[mask]
        else:
            selected_groups = group_names

        results = {}

        # Indirect Standardization
        if "indirect" in stdz:
            # Compute expected outcomes by group by excluding random effects
            expected_by_group = np.array([
                self.xbeta_[group_indices == i].sum() for i in range(len(group_names))
            ])
            # Compute observed outcomes using the full fitted values including random effects
            observed_by_group = np.array([
                self.fitted_[group_indices == i].sum() for i in range(len(group_names))
            ])
            # Indirect standardized difference
            indirect_diff = (observed_by_group - expected_by_group) / group_sizes

            indirect_df = pd.DataFrame({
                "group_id": group_names,
                "indirect_difference": indirect_diff,
                "observed": observed_by_group,
                "expected": expected_by_group
            })

            if group_ids is not None:
                indirect_df = indirect_df[indirect_df['group_id'].isin(selected_groups)].reset_index(drop=True)

            results["indirect"] = indirect_df

        # Direct Standardization
        if "direct" in stdz:
            # Overall observed is the total sum of the fitted values
            totalower_observed = self.fitted_.sum()
            # Calculate direct expected outcomes using group-specific random effects
            expected_direct_by_group = np.array([
                np.sum(self.xbeta_ + re_val) for re_val in random_effects
            ])
            # Direct standardized difference
            direct_diff = (expected_direct_by_group - totalower_observed) / totalower_samples

            direct_df = pd.DataFrame({
                "group_id": group_names,
                "direct_difference": direct_diff,
                "observed": np.full(len(group_names), totalower_observed),
                "expected": expected_direct_by_group
            })

            if group_ids is not None:
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
        """
        Compute lower and upper bounds for confidence intervals given estimates,
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
            upper = np.fullower_like(re_coef, np.inf)
        elif alternative == "less":
            crit_value = norm.ppf(1 - alpha)
            lower = np.fullower_like(re_coef, -np.inf)
            upper = re_coef + crit_value * se
        else:
            raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")
        return lower, upper
    
    def calculate_confidence_intervals(
        self,
        group_ids: Optional[Union[list, np.ndarray]] = None,
        level: float = 0.95,
        option: str = "SM",
        stdz: Union[str, list] = "indirect",
        null: Union[str, float] = "median",
        alternative: str = "two_sided"
    ) -> dict:
        """
        Calculate confidence intervals for provider (random) effects or standardized measures.

        Parameters
        ----------
        group_ids : Optional[Union[list, np.ndarray]]
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
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("The model must be fitted before calculating confidence intervals.")

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

        random_effects = self.coefficients_["random_effect"]
        var_alpha = self.variances_["re_var"].values[0, 0]  # Extract the scalar value

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
                "lower": lower_alpha,
                "upper": upper_alpha
            })

            if group_ids is not None:
                alpha_ci = alpha_ci[alpha_ci["group_id"].isin(group_ids)].reset_index(drop=True)
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
                
                if group_ids is not None:
                    indirect_df = indirect_df[indirect_df["group_id"].isin(group_ids)].reset_index(drop=True)
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
                    upper_direct = np.fullower_like(lower_direct, np.inf)
                elif alternative == "less":
                    lower_direct = np.fullower_like(lower_prov, -np.inf)
                    upper_direct = (upper_prov - obs_direct) / n_total
                else:
                    raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")

                direct_df = sm_results["direct"].copy()
                direct_df["lower"] = lower_direct
                direct_df["upper"] = upper_direct
                if group_ids is not None:
                    direct_df = direct_df[direct_df["group_id"].isin(group_ids)].reset_index(drop=True)
                result["direct_ci"] = direct_df

        return result

    def test(
        self,
        group_ids: Optional[Union[list, np.ndarray]] = None,
        level: float = 0.95,
        null: float = 0,
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """
        Conduct hypothesis tests on provider (random) effects to identify outliers.

        Parameters
        ----------
        group_ids : Optional[Union[list, np.ndarray]]
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
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("The model must be fitted before testing.")

        alpha = 1 - level
        random_effects = self.coefficients_["random_effect"]
        var_alpha = self.variances_["re_var"].values[0, 0]
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

        if group_ids is not None:
            if isinstance(group_ids, (list, np.ndarray)):
                result = result.loc[group_ids]
            else:
                raise ValueError("Argument 'group_ids' should be a list or array of provider IDs.")

        return result
    
    def plot_funnel(
        self,
        null: Union[str, float] = "median",
        target: float = 0,
        alpha: Union[float, List[float]] = 0.05,
        labels: List[str] = ["lower", "expected", "higher"],
        point_colors: List[str] = ["#D73027", "#4575B4", "#1A9850"],
        point_shapes: List[str] = ['o', 's', 'D'],
        point_size: float = 2,
        point_alpha: float = 0.85,
        line_size: float = 2,
        target_line_type: str = 'dashdot'
    ) -> None:
        """
        Create a funnel plot comparing provider performance for random effect models.
        """
        if self.coefficients_ is None or self.sigma_ is None or self.groups_ is None:
            raise ValueError("Model must be fitted before plotting.")

        if isinstance(alpha, (float, int)):
            alpha = [alpha]

        SM_info = self.calculate_standardized_measures(stdz="indirect")
        indirect_df: pd.DataFrame = SM_info["indirect"].copy()
        indirect_df.index = self.groups_
        indirect_df["precision"] = self.group_sizes_

        test_df = self.test(level=0.95, null=null)
        indirect_df["flag"] = test_df["flag"]

        limits_list = []
        for a in sorted(alpha):
            z_val = norm.ppf(1 - a / 2)
            ci_lower = target - z_val * self.sigma_ / np.sqrt(indirect_df["precision"])
            ci_upper = target + z_val * self.sigma_ / np.sqrt(indirect_df["precision"])
            limits_df = pd.DataFrame({
                "precision": indirect_df["precision"],
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "alpha": a
            }, index=self.groups_)
            limits_list.append(limits_df)
        limits_df = pd.concat(limits_list)

        plt.figure(figsize=(10, 6))
        for a in sorted(alpha):
            subset = limits_df[limits_df["alpha"] == a]
            plt.fillower_between(subset["precision"], subset["ci_lower"], subset["ci_upper"],
                             color="#A6CEE3", alpha=0.25,
                             label=f"Control limits (α={a})" if a == min(alpha) else "")
        unique_flags = np.sort(indirect_df["flag"].unique())
        colors_map = {flag: point_colors[i % len(point_colors)] for i, flag in enumerate(unique_flags)}
        shapes_map = {flag: point_shapes[i % len(point_shapes)] for i, flag in enumerate(unique_flags)}

        for flag in unique_flags:
            subset = indirect_df[indirect_df["flag"] == flag]
            labeloweshrinkage_factordx = int(flag + 1)  # assuming flags -1, 0, 1
            plt.scatter(subset["precision"], subset["Indirect_standardized.difference"],
                        color=colors_map[flag],
                        marker=shapes_map[flag],
                        s=point_size * 50,
                        alpha=point_alpha,
                        edgecolor="k",
                        label=f"{labels[labeloweshrinkage_factordx]} ({len(subset)})")
        plt.axhline(y=target, color="black", linestyle=target_line_type, linewidth=line_size)
        plt.xlabel("Precision (Group Size)")
        plt.ylabel("Performance Indicator")
        plt.title("Funnel Plot for Random Effects Model")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_residuals(self) -> None:
        """
        Plot residuals versus fitted values to assess model fit.
        """
        if self.fitted_ is None or self.residuals_ is None:
            raise ValueError("Model must be fitted before plotting residuals.")
        plt.figure(figsize=(10, 6))
        plt.scatter(self.fitted_.values, self.residuals_.values,
                    color="#1F78B4", alpha=0.75, edgecolor="k")
        plt.axhline(0, color="red", linestyle="--", linewidth=2)
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs. Fitted Values")
        plt.tight_layout()
        plt.show()

    def plot_qq(self) -> None:
        """
        Create a Q-Q plot of the residuals to assess normality.
        """
        if self.residuals_ is None:
            raise ValueError("Model must be fitted before plotting QQ plot.")
        plt.figure(figsize=(8, 6))
        stats.probplot(self.residuals_.values, dist="norm", plot=plt)
        plt.title("Q-Q Plot of Residuals")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Ordered Residuals")
        plt.tight_layout()
        plt.show()

    def plot_provider_effects(self) -> None:
        """
        Create a caterpillar plot (dot-and-error bar) for provider (random) effects.
        """
        if self.coefficients_ is None or self.variances_ is None or self.groups_ is None:
            raise ValueError("Model must be fitted before plotting provider effects.")
        re_coefs = self.coefficients_["Random Effects"]
        se_alpha = np.sqrt(self.variances_["RE Variance"][0, 0])
        n = len(self.fitted_)
        df = n - len(self.coefficients_["Fixed Effects"]) - len(re_coefs) + 1
        crit = t.ppf(0.975, df)
        lower_bounds = re_coefs - crit * se_alpha
        upper_bounds = re_coefs + crit * se_alpha
        df_re = pd.DataFrame({
            "group": re_coefs.index,
            "Random Effect": re_coefs.values,
            "ci_lower": lower_bounds,
            "ci_upper": upper_bounds
        }).sort_values("Random Effect")
        plt.figure(figsize=(10, 8))
        plt.errorbar(df_re["Random Effect"], df_re["group"],
                     xerr=[df_re["Random Effect"] - df_re["ci_lower"], df_re["ci_upper"] - df_re["Random Effect"]],
                     fmt="o", color="#2C3E50", ecolor="#E74C3C", capsize=5, markersize=8)
        plt.axvline(0, color="grey", linestyle="--", linewidth=2)
        plt.xlabel("Provider Effect Estimate")
        plt.ylabel("Provider")
        plt.title("Caterpillar Plot of Provider Effects")
        plt.tight_layout()
        plt.show()

    def plot_coefficient_forest(self) -> None:
        """
        Create a forest plot of covariate coefficients.
        """
        if self.coefficients_ is None or self.variances_ is None or self.covariate_names_ is None:
            raise ValueError("Model must be fitted before plotting coefficients.")
        fe_estimates = self.coefficients_["Fixed Effects"]
        se_fe = np.sqrt(np.diag(self.variances_["FE Variance-Covariance"]))
        n = len(self.fitted_)
        df = n - len(fe_estimates) - len(self.coefficients_["Random Effects"]) + 1
        crit = t.ppf(0.975, df)
        lower_bound = fe_estimates - crit * se_fe
        upper_bound = fe_estimates + crit * se_fe
        coef_df = pd.DataFrame({
            "covariate": self.covariate_names_,
            "estimate": fe_estimates,
            "ci_lower": lower_bound,
            "ci_upper": upper_bound
        })
        coef_df.sort_values("estimate", inplace=True)
        plt.figure(figsize=(10, 6))
        plt.errorbar(coef_df["estimate"], coef_df["covariate"],
                     xerr=[coef_df["estimate"] - coef_df["ci_lower"], coef_df["ci_upper"] - coef_df["estimate"]],
                     fmt="o", color="#34495E", ecolor="#95A5A6", capsize=5, markersize=8)
        plt.axvline(0, color="red", linestyle="--", linewidth=2)
        plt.xlabel("Coefficient Estimate")
        plt.ylabel("Covariate")
        plt.title("Forest Plot of Covariate Coefficients")
        plt.tight_layout()
        plt.show()
