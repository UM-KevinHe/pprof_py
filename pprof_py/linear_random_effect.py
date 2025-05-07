import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import t, norm, probplot
from typing import Optional, Union, List, Tuple

from .base_model import BaseModel
from .mixins import SummaryMixin, PlotMixin, TestMixin
from .utils import check_array
from .plotting import plot_caterpillar



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
        Create a funnel plot for standardized differences in random effect models.

        This plot visualizes the standardized differences (e.g., alpha_i - alpha_null) against
        group size (as a proxy for precision). Control limits are based on the overall model's
        residual standard deviation (sigma).

        Parameters:
        -----------
        stdz : str, default="indirect"
            Standardization method. For random effects, this can be "indirect" or "direct".
        null : str or float, default="median"
            Baseline for provider effects (alpha) used in calculating the difference
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

        # Ttransform any "median"/"mean" null into a numeric value
        if isinstance(null, str):
            # Grab the random effects as you do elsewhere
            random_effects = self.coefficients_["random_effect"]
            if null == "median":
                null = np.median(random_effects)
            elif null == "mean":
                null = np.mean(random_effects)
            else:
                raise ValueError("If you pass a string to 'null', it must be 'median' or 'mean'.")
        else:
            # If it's already numeric, just ensure float
            null = float(null)

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
        null: Union[str, float] = 0,
        test_method: Optional[str] = None,
        **plot_kwargs
    ) -> None:
        """
        Plots provider random effects (alpha) using the plot_caterpillar helper function.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot. If None, all providers are included.
        level : float, default=0.95
            Confidence level for intervals.
        use_flags : bool, default=True
            Whether to color-code providers based on flags from the test method.
        null : str or float, default=0
            Null hypothesis for alpha used for flagging. Can be 'median', 'mean', or a float.
        test_method : str, optional
            Test method used specifically for generating flags ('wald' is the only one for LinearRE's .test()).
            If None, defaults to 'wald'.
        **plot_kwargs
            Additional arguments passed to plot_caterpillar (e.g., plot_title, orientation).
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("Model must be fitted first.")

        # Ttransform any "median"/"mean" null into a numeric value
        if isinstance(null, str):
            # Grab the random effects as you do elsewhere
            random_effects = self.coefficients_["random_effect"]
            if null == "median":
                null = np.median(random_effects)
            elif null == "mean":
                null = np.mean(random_effects)
            else:
                raise ValueError("If you pass a string to 'null', it must be 'median' or 'mean'.")
        else:
            # If it's already numeric, just ensure float
            null = float(null)

        ci_results = self.calculate_confidence_intervals(
            group_ids=group_ids, level=level, option='alpha', alternative='two_sided'
        )
        if 'alpha_ci' not in ci_results or ci_results['alpha_ci'].empty:
            print("Warning: No alpha CI data. Cannot plot.")
            return

        df_plot = ci_results['alpha_ci']  # Has 'group_id', 'alpha', 'lower', 'upper'

        flag_col_name = None
        if use_flags:
            flag_col_name = 'flag'
            current_test_method = test_method if test_method else 'wald'
            if current_test_method != 'wald':
                warnings.warn(f"LinearRandomEffectModel.test uses a t-test (Wald-like). test_method '{current_test_method}' for flagging will use this underlying test.")

            try:
                test_df = self.test(
                    group_ids=df_plot['group_id'].unique().tolist(),
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

        alpha_vals = self.coefficients_["random_effect"].values.flatten()
        if null == "median":
            alpha_null_val = np.median(alpha_vals)
        elif null == "mean":
            alpha_null_val = np.average(alpha_vals, weights=self.group_sizes_ if self.group_sizes_ is not None else None)
        else:
            alpha_null_val = float(null)

        orientation = plot_kwargs.pop('orientation', 'vertical')
        if orientation == 'vertical':
            plot_kwargs.setdefault('xlab', 'Alpha Estimate (Random Effect)')
            plot_kwargs.setdefault('ylab', 'Provider')
        else:
            plot_kwargs.setdefault('xlab', 'Provider')
            plot_kwargs.setdefault('ylab', 'Alpha Estimate (Random Effect)')

        plot_kwargs.setdefault('plot_title', 'Provider Effects (Alpha)')
        plot_kwargs.setdefault('refline_value', alpha_null_val)
        plot_kwargs.setdefault('orientation', orientation)

        plot_caterpillar(
            df=df_plot,
            estimate_col='alpha',
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
        use_flags: bool = True, 
        null: Union[str, float] = 'median',
        test_method: Optional[str] = None,
        **plot_kwargs
    ) -> None:
        """
        Plots standardized differences using plot_caterpillar.
        For LinearRandomEffectModel, standardized measures are differences (alpha_i - alpha_null).

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot.
        level : float, default=0.95
            Confidence level for intervals.
        stdz : str, default='indirect'
            Standardization method ('indirect' or 'direct'). Both result in alpha_i - alpha_null.
        measure : str, default='difference'
            The measure to plot. For linear models, this is always 'difference'.
        use_flags : bool, default=True
            Whether to color-code providers based on flags from the alpha test method.
        null : str or float, default='median'
            Null hypothesis for alpha used for flagging and calculating the difference.
        test_method : str, optional
            Test method used specifically for generating flags. Defaults to 'wald' (t-test).
        **plot_kwargs
            Additional arguments passed to plot_caterpillar.
        """
        if self.coefficients_ is None: 
            raise ValueError("Model must be fitted.")

        # Ttransform any "median"/"mean" null into a numeric value
        if isinstance(null, str):
            # Grab the random effects as you do elsewhere
            random_effects = self.coefficients_["random_effect"]
            if null == "median":
                null = np.median(random_effects)
            elif null == "mean":
                null = np.mean(random_effects)
            else:
                raise ValueError("If you pass a string to 'null', it must be 'median' or 'mean'.")
        else:
            # If it's already numeric, just ensure float
            null = float(null)

        if measure != 'difference':
            warnings.warn("For LinearRandomEffectModel, standardized 'measure' is 'difference'.")

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

        df_plot = ci_results[ci_key]  # This df should have 'group_id', '{stdz}_difference', 'lower', 'upper'
        estimate_col_name = f"{stdz}_difference"

        if estimate_col_name not in df_plot.columns or 'lower' not in df_plot.columns or 'upper' not in df_plot.columns:
            raise ValueError(f"Required columns ('{estimate_col_name}', 'lower', 'upper') not found in SM CI results. Available: {df_plot.columns}")

        flag_col_name = None
        if use_flags:
            flag_col_name = 'flag'
            current_test_method = test_method if test_method else 'wald'
            if current_test_method != 'wald':
                warnings.warn(f"LinearRandomEffectModel.test uses a t-test (Wald-like). test_method '{current_test_method}' for flagging will use this.")
            try:
                test_df = self.test(
                    group_ids=df_plot['group_id'].unique().tolist(), 
                    level=level, 
                    null=null, 
                    alternative='two_sided'
                )

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
        if self.coefficients_ is None or self.variances_ is None or self.covariate_names_ is None:
            raise ValueError("Model must be fitted before plotting coefficients.")
        if orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        # Compute estimates and 95% CIs
        beta = self.coefficients_["fixed_effect"].values.flatten()
        se_beta = np.sqrt(np.diag(self.variances_["fe_var_cov"]))
        df_denom = self.fitted_.size - len(beta) - len(self.coefficients_["random_effect"])
        crit = t.ppf(1 - 0.05 / 2, df_denom)
        lower = beta - crit * se_beta
        upper = beta + crit * se_beta

        coef_df = (
            pd.DataFrame({
                "covariate": self.result.fe_params.index,
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
            self.fitted_.values, self.residuals_.values,
            color=point_color, alpha=point_alpha,
            edgecolor=edge_color, linewidth=edge_linewidth if edge_color else 0,
            s=point_size
        )
        ax.axhline(0, color=line_color, linestyle=line_style, linewidth=line_width)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.set_title(title, fontsize=font_size + 2, pad=15)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if add_grid: 
            ax.grid(True, linestyle=grid_style, alpha=grid_alpha, color='lightgrey')
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)

        plt.tight_layout()
        plt.show()

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
        residuals_flat = self.residuals_.values

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
        if add_grid: 
            ax.grid(True, linestyle=':', alpha=0.6, color='lightgrey')
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)

        plt.tight_layout()
        plt.show()