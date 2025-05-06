import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from scipy.optimize import root_scalar
from sklearn.metrics import roc_auc_score
from typing import Optional, Union, List
from fast_poibin import PoiBin



from .base_model import BaseModel
from .algorithm import SerbinAlgorithm, BanAlgorithm
from .utils import sigmoid
from .mixins import SummaryMixin, TestMixin, PlotMixin
from .plotting import plot_caterpillar


class LogisticFixedEffectModel(BaseModel, SummaryMixin, TestMixin, PlotMixin):
    """
    Logistic Regression Model with Fixed Effects.

    This class implements a logistic regression model with fixed effects for groups (e.g., providers),
    using custom optimization algorithms ('Serbin' or 'Ban'). It supports advanced data preparation
    via the DataPrep class, estimates coefficients for covariates and group effects, and provides
    methods for fitting, prediction, and diagnostics.

    Parameters
    ----------
    use_dataprep : bool, default=True
        Whether to use the DataPrep class for data preparation.
    screen_providers : bool, default=True
        Whether to screen providers (groups) during data preparation.
    log_event_providers : bool, default=True
        Whether to log event providers during data preparation.
    cutoff : int, default=10
        Minimum group size for screening.
    threshold_cor : float, default=0.9
        Correlation threshold for multicollinearity checks.
    threshold_vif : int, default=10
        Variance Inflation Factor threshold for multicollinearity.
    algorithm : str, default='Serbin'
        Optimization algorithm ('Serbin' or 'Ban').

    Attributes
    ----------
    algorithm_type : str
        Optimization algorithm used.
    coefficients_ : dict
        Model coefficients: 'beta' (covariates), 'gamma' (group effects).
    variances_ : dict
        Variance-covariance matrices: 'beta' (for covariates), 'gamma' (for group effects).
    fitted_ : np.ndarray
        Fitted probabilities.
    aic_ : float
        Akaike Information Criterion.
    bic_ : float
        Bayesian Information Criterion.
    groups_ : np.ndarray
        Unique group identifiers.
    """

    def __init__(
        self,
        use_dataprep: bool = True,
        screen_providers: bool = True,
        log_event_providers: bool = True,
        cutoff: int = 10,
        threshold_cor: float = 0.9,
        threshold_vif: int = 10,
        algorithm: str = 'Serbin'
    ):
        """
        Initialize the LogisticFixedEffectModel with data preparation options.

        Parameters
        ----------
        (See class docstring for parameter details)
        """
        # Store data preparation defaults
        self.use_dataprep = use_dataprep
        self.dataprep_params = {
            'screen_providers': screen_providers,
            'log_event_providers': log_event_providers,
            'cutoff': cutoff,
            'threshold_cor': threshold_cor,
            'threshold_vif': threshold_vif,
            'binary_response': True  # Enforce binary response for logistic models
        }
        
        # Validate and set algorithm
        if algorithm not in ['Serbin', 'Ban']:
            raise ValueError("Algorithm must be 'Serbin' or 'Ban'")
        self.algorithm_type = algorithm
        
        # Initialize attributes
        self.algorithm = None
        self.coefficients_ = None
        self.variances_ = None
        self.fitted_ = None
        self.aic_ = None
        self.bic_ = None
        self.auc_ = None
        self.groups_ = None
        self.group_indices_ = None
        self.group_sizes_ = None
        self.xbeta_ = None
        self.outcome_ = None
        self.covariate_names_ = None  # Store covariate names if provided

    def _estimate_variances(self) -> dict:
        """
        Estimate variances of beta and gamma coefficients for inferential statistics.

        Returns
        -------
        dict
            Variances of beta (covariance matrix) and gamma (diagonal variances).

        Raises
        ------
        ValueError
            If the model has not been fitted (i.e., required attributes are None).
        """
        # Check if the model has been fitted
        if self.fitted_ is None or self.X is None or self.group_indices_ is None:
            raise ValueError("Model must be fitted before estimating variances.")

        # Use precomputed predicted probabilities
        p = self.fitted_
        p = np.clip(p, 1e-10, 1 - 1e-10)  # Ensure numerical stability

        q = p * (1 - p)

        # Number of groups
        n_groups = len(self.groups_)

        # Information for gamma: inverse of sum(q) per group
        info_gamma_inv = 1 / np.bincount(self.group_indices_, weights=q, minlength=n_groups)

        # Cross-information: sum(X * q) per group for each covariate
        info_beta_gamma = np.array([
            np.bincount(self.group_indices_, weights=q * self.X[:, i], minlength=n_groups)
            for i in range(self.X.shape[1])
        ])  # Shape: (n_covariates, n_groups)

        # Information for beta: X.T @ (q * X)
        info_beta = self.X.T @ (q[:, None] * self.X)

        # Variance for beta
        mat_tmp1 = (info_gamma_inv * info_beta_gamma).T  # Shape: (n_covariates, n_groups)
        schur_complement = info_beta - info_beta_gamma @ mat_tmp1
        info_beta_inv = np.linalg.inv(schur_complement)
        var_beta = info_beta_inv

        # Variance for gamma
        quad_term = np.sum((mat_tmp1 @ info_beta_inv) * mat_tmp1, axis=1)  # Shape: (n_groups,)
        var_gamma = info_gamma_inv + quad_term

        return {"beta": var_beta, "gamma": var_gamma}

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        x_vars: Optional[List[str]] = None,
        y_var: Optional[str] = None,
        group_var: Optional[str] = None,
        use_dataprep: Optional[bool] = None,
        screen_providers: Optional[bool] = None,
        log_event_providers: Optional[bool] = None,
        cutoff: Optional[int] = None,
        threshold_cor: Optional[float] = None,
        threshold_vif: Optional[int] = None,
        max_iter: int = 10000,
        tol: float = 1e-5,
        bound: float = 10.0,
        backtrack: bool = True
    ) -> "LogisticFixedEffectModel":
        """
        Fit the logistic fixed effect model with enhanced data preparation and variance estimation.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Covariates or dataset.
        y : Optional[np.ndarray], default=None
            Binary response variable.
        groups : Optional[np.ndarray], default=None
            Group identifiers.
        x_vars : Optional[List[str]], default=None
            Covariate column names if X is a DataFrame.
        y_var : Optional[str], default=None
            Response column name if X is a DataFrame.
        group_var : Optional[str], default=None
            Group column name if X is a DataFrame.
        use_dataprep : Optional[bool], default=None
            Override default data preparation setting.
        screen_providers : Optional[bool], default=None
            Override provider screening.
        log_event_providers : Optional[bool], default=None
            Override logging of event providers.
        cutoff : Optional[int], default=None
            Override cutoff for group size.
        threshold_cor : Optional[float], default=None
            Override correlation threshold.
        threshold_vif : Optional[int], default=None
            Override VIF threshold.
        max_iter : int, default=10000
            Maximum optimization iterations.
        tol : float, default=1e-5
            Convergence tolerance.
        bound : float, default=10.0
            Bound for group effects.
        backtrack : bool, default=True
            Use backtracking line search.

        Returns
        -------
        LogisticFixedEffectModel
            Fitted model instance.
        """
        # Use stored defaults unless overridden
        use_dataprep = use_dataprep if use_dataprep is not None else self.use_dataprep
        dataprep_params = self.dataprep_params.copy()
        for param, value in {
            'screen_providers': screen_providers,
            'log_event_providers': log_event_providers,
            'cutoff': cutoff,
            'threshold_cor': threshold_cor,
            'threshold_vif': threshold_vif
        }.items():
            if value is not None:
                dataprep_params[param] = value

        # Store covariate names for later use
        self.covariate_names_ = x_vars if x_vars is not None else [f'x{i}' for i in range(X.shape[1])] if isinstance(X, np.ndarray) else None

        # Validate and convert inputs
        X, y, groups = self._validate_and_convert_inputs(
            X, y, groups, x_vars, y_var, group_var,
            use_dataprep=use_dataprep,
            **dataprep_params
        )

        # Store data attributes
        self.X = X
        self.outcome_ = y
        self.groups_, self.group_indices_ = np.unique(groups, return_inverse=True)
        self.group_sizes_ = np.bincount(self.group_indices_)

        # Prepare data for algorithm
        data = np.column_stack((y, groups, X))
        y_index = 0
        prov_index = 1
        n_prov = self.group_sizes_
        gamma_prov = np.repeat(np.log(np.mean(y) / (1 - np.mean(y))), len(n_prov))
        beta = np.zeros(X.shape[1])

        # Instantiate algorithm
        if self.algorithm_type == 'Serbin':
            self.algorithm = SerbinAlgorithm(
                data, y_index, X, prov_index, n_prov, gamma_prov, beta,
                backtrack=backtrack, max_iter=max_iter, bound=bound, tol=tol
            )
        else:
            self.algorithm = BanAlgorithm(
                data, y_index, X, prov_index, n_prov, gamma_prov, beta,
                backtrack=backtrack, max_iter=max_iter, bound=bound, tol=tol
            )

        # Fit the model
        result = self.algorithm.fit()

        self.coefficients_ = {'beta': result['beta'], 'gamma': result['gamma']}

        # Compute fitted values
        self.xbeta_ = np.dot(X, self.coefficients_['beta'])
        gamma_obs = np.repeat(self.coefficients_['gamma'], n_prov)
        linear_pred = self.xbeta_ + gamma_obs
        self.fitted_ = sigmoid(linear_pred)

        # Compute aic and bic
        neg2Loglkd = -2 * np.sum((gamma_obs + self.xbeta_) * y - np.log(1 + np.exp(linear_pred)))
        n_params = len(self.coefficients_['beta']) + len(self.coefficients_['gamma'])
        self.aic_ = 2 * n_params + neg2Loglkd
        self.bic_ = n_params * np.log(len(y)) + neg2Loglkd

        self.auc_ = roc_auc_score(y, self.fitted_)

        # Compute variances
        self.variances_ = self._estimate_variances()

        return self

    def predict(self, X, groups=None, x_vars=None, group_var=None) -> np.ndarray:
        """
        Predict using the LogisticFixedEffect model.

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
            Predicted probabilities.
        """
        if self.coefficients_ is None:
            raise ValueError("Model is not fitted. Call 'fit' before 'predict'.")

        X, _, groups = self._validate_and_convert_inputs(X, None, groups, x_vars, None, group_var)
        group_indices = np.searchsorted(self.groups_, groups)
        beta = self.coefficients_["beta"]
        gamma = self.coefficients_["gamma"]
        xbeta = X @ beta
        gamma_obs = gamma[group_indices].flatten()
        predictions = 1 / (1 + np.exp(-(xbeta + gamma_obs)))
        return predictions

    def score(self, X, y=None, groups=None, x_vars=None, y_var=None, group_var=None):
        """
        Compute the accuracy score for the model.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Design matrix (covariates) or complete dataset.
        y : array-like, shape (n_samples,), optional
            True target values (if X is array-like).
        groups : array-like, shape (n_samples,), optional
            Group identifiers (if X is array-like).
        x_vars : list of str, optional
            Column names for predictors (if X is a DataFrame).
        y_var : str, optional
            Column name for the target variable (if X is a DataFrame).
        group_var : str, optional
            Column name for group identifiers (if X is a DataFrame).

        Returns
        -------
        float
            Accuracy of the model.
        """
        if self.coefficients_ is None:
            raise ValueError("Model is not fitted. Call 'fit' before 'score'.")

        # Validate and convert inputs
        X, y, groups = self._validate_and_convert_inputs(X, y, groups, x_vars, y_var, group_var)

        # Generate predictions (assuming predict returns probabilities)
        y_pred = self.predict(X, groups)

        # Convert probabilities to class labels (adjust threshold if needed)
        y_pred_class = (y_pred > 0.5).astype(int)

        # Compute accuracy
        return np.mean(y_pred_class == y)

    def get_params(self) -> dict:
        """
        Return model parameters.

        Returns
        -------
        dict
            Model parameters including coefficients, variances, aic, and bic.
        """
        return {
            "coefficients": self.coefficients_,
            "variances": self.variances_,
            "aic": self.aic_,
            "bic": self.bic_,
            "auc": self.auc_
        }

    def _compute_wald_beta(self, index: int, null: float = 0, alternative: str = "two_sided", alpha: float = 0.05):
        """
        Perform a Wald test for a specific covariate coefficient.

        Parameters
        ----------
        index : int
            Index of the covariate to test.
        null : float, default=0
            Null hypothesis value for the coefficient.
        alternative : str, default="two_sided"
            Alternative hypothesis: "two_sided", "less", or "greater".
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        dict
            Results including test statistic, p-value, and confidence interval.
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("Model must be fitted before performing tests.")

        beta = self.coefficients_['beta'][index]
        se_beta = np.sqrt(self.variances_['beta'][index, index])
        stat = (beta - null) / se_beta

        if alternative == "two_sided":
            p_value = 2 * (1 - norm.cdf(np.abs(stat)))
            crit_value = norm.ppf(1 - alpha / 2)
            ci_lower = beta - crit_value * se_beta
            ci_upper = beta + crit_value * se_beta
        elif alternative == "less":
            p_value = norm.cdf(stat)
            crit_value = norm.ppf(1 - alpha)
            ci_lower = -np.inf
            ci_upper = beta + crit_value * se_beta
        elif alternative == "greater":
            p_value = 1 - norm.cdf(stat)
            crit_value = norm.ppf(1 - alpha)
            ci_lower = beta - crit_value * se_beta
            ci_upper = np.inf
        else:
            raise ValueError("Alternative must be 'two_sided', 'less', or 'greater'")

        return {
            "statistic": stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "alternative": alternative
        }

    def _compute_lr_beta(self, index: int):
        """
        Perform a Likelihood Ratio test for a specific covariate.

        Parameters
        ----------
        index : int
            Index of the covariate to test.

        Returns
        -------
        dict
            Results including test statistic and p-value.
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted before performing tests.")

        # Full model log-likelihood
        gamma_obs_full = np.repeat(self.coefficients_['gamma'], self.group_sizes_)
        loglik_full = self.algorithm._loglikelihood(gamma_obs_full, self.coefficients_['beta'])

        # Fit reduced model excluding the covariate at index
        reduced_X = np.delete(self.X, index, axis=1)
        reduced_model = LogisticFixedEffectModel(algorithm=self.algorithm_type)
        reduced_model.fit(
            reduced_X, self.outcome_, self.group_indices_,
            max_iter=self.algorithm.max_iter,
            tol=self.algorithm.tol,
            bound=self.algorithm.bound,
            backtrack=self.algorithm.backtrack
        )

        # Reduced model log-likelihood
        gamma_obs_reduced = np.repeat(reduced_model.coefficients_['gamma'], reduced_model.group_sizes_)
        loglik_reduced = reduced_model.algorithm._loglikelihood(gamma_obs_reduced, reduced_model.coefficients_['beta'])

        # Test statistic and p-value
        test_stat = 2 * (loglik_full - loglik_reduced)
        p_value = chi2.sf(test_stat, df=1)

        return {
            "statistic": test_stat,
            "p_value": p_value,
            "df": 1
        }

    def _compute_score_beta(self, index: int):
        """
        Perform a Score test for a specific covariate.

        Parameters
        ----------
        index : int
            Index of the covariate to test.

        Returns
        -------
        dict
            Results including test statistic and p-value.
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted before performing tests.")

        # Fit reduced model excluding the covariate at index
        reduced_X = np.delete(self.X, index, axis=1)
        reduced_model = LogisticFixedEffectModel(algorithm=self.algorithm_type)
        reduced_model.fit(
            reduced_X, self.outcome_, self.group_indices_,
            max_iter=self.algorithm.max_iter,
            tol=self.algorithm.tol,
            bound=self.algorithm.bound,
            backtrack=self.algorithm.backtrack
        )

        # Compute probabilities and weights under the reduced model
        gamma_obs = np.repeat(reduced_model.coefficients_['gamma'], reduced_model.group_sizes_)
        p = sigmoid(gamma_obs + reduced_X @ reduced_model.coefficients_['beta'])
        q = p * (1 - p)

        # Score for the excluded covariate
        score_excluded = np.sum((self.outcome_ - p) * self.X[:, index])

        # Information components
        info_excluded_excluded = np.sum(q * self.X[:, index] ** 2)  # scalar
        info_excluded_beta = (q * self.X[:, index]).T @ reduced_X   # (1, p_reduced)
        info_beta = reduced_X.T @ (q[:, None] * reduced_X)          # (p_reduced, p_reduced)

        # Inverse of the information matrix for beta
        schur_inv = np.linalg.inv(info_beta)  # (p_reduced, p_reduced)

        # Variance of the score
        info_full = info_excluded_excluded - info_excluded_beta @ schur_inv @ info_excluded_beta.T

        # Test statistic and p-value
        test_stat = score_excluded ** 2 / info_full
        p_value = chi2.sf(test_stat, df=1)

        return {
            "statistic": test_stat,
            "p_value": p_value,
            "df": 1
        }

    def summary(self, covariates: list = None, level: float = 0.95, null: float = 0, alternative: str = "two_sided", test_method: str = "wald") -> pd.DataFrame:
        """
        Provides summary statistics for the covariate estimates in a fitted fixed effects model.

        Parameters
        ----------
        covariates : list of str or int, optional
            Covariate names or indices to include in the summary. If None, 
            all are included.
        level : float, default=0.95
            Confidence level for intervals.
        null : float, default=0
            Null hypothesis value for the coefficient. Used directly for 
            the Wald test. For LR and Score tests, only null=0 is implemented.
        alternative : str, default="two_sided"
            Hypothesis type ("two_sided", "greater", or "less") for the Wald test.
            LR and Score tests are implicitly two-sided for coefficient = 0.
        test_method : str, default="wald"
            Testing approach: "wald", "lr", or "score".

        Returns
        -------
        pd.DataFrame
            Summary statistics with columns:
            - estimate : coefficient estimate
            - std_error : standard error (from the variance-covariance matrix)
            - stat : test statistic (from wald_test, lr_test, or score_test)
            - p_value : p-value from wald_test, lr_test, or score_test
            - ci_lower, ci_upper : confidence interval bounds (Wald-based)

        Raises
        ------
        ValueError
            If the model is not fitted or if 'test' is invalid or if 
            'covariates' are mis-specified.
        NotImplementedError
            If 'null' != 0 for LR/Score (only null=0 is implemented).
        """

        # ----------------------------------------------------------------------
        # 1. Ensure the model is fitted
        if self.coefficients_ is None:
            raise ValueError("The model must be fitted before calling 'summary'.")
        if self.variances_ is None:
            raise ValueError("Variances are not available. Ensure the model is fully fitted.")

        # ----------------------------------------------------------------------
        # 2. Check test validity
        if test_method not in ["wald", "lr", "score"]:
            raise ValueError("Argument 'test' must be one of 'wald', 'lr', or 'score'.")

        # For LR and Score, we only handle null=0
        if test_method in ["lr", "score"] and null != 0:
            raise NotImplementedError(
                f"{test_method.upper()} test currently only implemented for null=0."
            )

        # ----------------------------------------------------------------------
        # 3. Identify which covariate indices to summarize
        beta = self.coefficients_["beta"].flatten()
        n_cov = len(beta)

        if covariates is not None:
            # Covariates can be int indices or string names
            if all(isinstance(c, str) for c in covariates):
                indices = [self.covariate_names_.index(cov) for cov in covariates]
            elif all(isinstance(c, int) for c in covariates):
                indices = covariates
            else:
                raise ValueError("Argument 'covariates' must be a list of names or indices.")
        else:
            indices = range(n_cov)

        # Build the display names (if not provided, generate x0, x1, etc.)
        cov_names = (
            [self.covariate_names_[i] for i in indices]
            if self.covariate_names_
            else [f"x{i}" for i in indices]
        )

        # ----------------------------------------------------------------------
        # 4. Prepare containers for summary results
        estimates = []
        std_errors = []
        stats = []
        p_values = []
        ci_lowers = []
        ci_uppers = []

        # Wald-based intervals use alpha = 1 - level
        alpha = 1 - level

        # We'll compute the standard error from the variance matrix
        # but for "wald" we rely on self._compute_wald_beta() for the official
        # statistic, p-value, and intervals. For "lr"/"score" we 
        # use that method for stat/p-value, but still give a wald-based
        # interval for convenience.
        for idx, name in zip(indices, cov_names):
            estimate = beta[idx]
            # Standard error from the diagonal of the covariance matrix
            se = np.sqrt(self.variances_["beta"][idx, idx])

            if test_method == "wald":
                # NOTE: wald_test expects alpha, not level
                wald_res = self._compute_wald_beta(
                    index=idx, null=null, alternative=alternative, alpha=alpha
                )
                stat_val = wald_res["statistic"]
                p_val = wald_res["p_value"]
                ci_lower = wald_res["ci_lower"]
                ci_upper = wald_res["ci_upper"]

            elif test_method == "lr":
                # Use the lr_test method for test stat & p-value
                lr_res = self._compute_lr_beta(idx)
                stat_val = lr_res["statistic"]
                p_val = lr_res["p_value"]
                # Construct Wald-based confidence interval for reference
                # two-sided intervals only
                zcrit = norm.ppf(1 - alpha / 2)
                ci_lower = estimate - zcrit * se
                ci_upper = estimate + zcrit * se

            else:  # score
                score_res = self._compute_score_beta(idx)
                stat_val = score_res["statistic"]
                p_val = score_res["p_value"]
                # Construct Wald-based confidence interval
                zcrit = norm.ppf(1 - alpha / 2)
                ci_lower = estimate - zcrit * se
                ci_upper = estimate + zcrit * se

            # Accumulate results for the summary dataframe
            estimates.append(estimate)
            std_errors.append(se)
            stats.append(stat_val)
            # Round or format p-values
            p_values.append(f"{min(p_val, 1.0):.7g}")
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)

        # ----------------------------------------------------------------------
        # 5. Construct the final DataFrame
        summary_df = pd.DataFrame(
            {
                "estimate": estimates,
                "std_error": std_errors,
                "stat": stats,
                "p_value": p_values,
                "ci_lower": ci_lowers,
                "ci_upper": ci_uppers,
            },
            index=cov_names,
        )

        return summary_df

    # -------------------------------------------------------------------------
    # STANDARDIZED MEASURES
    # -------------------------------------------------------------------------
    def calculate_standardized_measures(self, group_ids=None, stdz="indirect", null="median") -> dict:
        """
        Calculate direct/indirect standardized ratios and rates for a fixed-effects logistic model.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider/group IDs for whom to calculate measures. If None,
            all providers are included.
        stdz : str or list of str, default="indirect"
            Standardization method(s). Must include at least one of {"indirect", "direct"}.
        null : {'median', 'mean'} or float, default="median"
            Defines the population norm if "direct" standardization is requested. 
            - 'median': uses median(gamma)
            - 'mean': uses weighted mean(gamma, weights=group_sizes_)
            - float: uses a user-specified numeric reference level.

        Returns
        -------
        dict
            A dictionary with keys for each requested standardization type:
            - 'indirect' -> DataFrame with columns ["group_id", "indirect_ratio", 
            "indirect_rate", "observed", "expected"].
            - 'direct' -> DataFrame with columns ["group_id", "direct_ratio", 
            "direct_rate", "observed", "expected"].

        Raises
        ------
        ValueError
            If the model is not deemed fitted or if arguments are invalid.
        """
        if self.coefficients_ is None or self.fitted_ is None or self.outcome_ is None:
            raise ValueError("The model must be fitted with stored outcomes, coefficients, and fitted probabilities.")

        if isinstance(stdz, str):
            stdz = [stdz]
        if not any(m in stdz for m in ["indirect", "direct"]):
            raise ValueError("Argument 'stdz' must include 'indirect' and/or 'direct'.")

        gamma = self.coefficients_["gamma"].flatten()
        n_samples = len(self.outcome_)

        # Determine gamma_null if needed
        if null == "median":
            gamma_null = np.median(gamma)
        elif null == "mean":
            gamma_null = np.average(gamma, weights=self.group_sizes_)
        elif isinstance(null, (int, float)):
            gamma_null = float(null)
        else:
            raise ValueError("Invalid 'null' argument for standardization baseline.")

        selected_groups = (
            self.groups_ if group_ids is None
            else self.groups_[np.isin(self.groups_, group_ids)]
        )
        results = {}

        # Indirect standardization
        if "indirect" in stdz:
            expected_prob = 1.0 / (1.0 + np.exp(-(gamma_null + self.xbeta_)))
            expected_by_group = np.array([
                np.sum(expected_prob[self.group_indices_ == i])
                for i in range(len(self.groups_))
            ])
            observed_by_group = np.array([
                np.sum(self.outcome_[self.group_indices_ == i])
                for i in range(len(self.groups_))
            ])
            indirect_ratio = observed_by_group / expected_by_group
            population_rate = observed_by_group.sum() / n_samples * 100.0
            indirect_rate = np.clip(indirect_ratio * population_rate, 0.0, 100.0)

            indirect_df = pd.DataFrame({
                "group_id": self.groups_,
                "indirect_ratio": indirect_ratio,
                "indirect_rate": indirect_rate,
                "observed": observed_by_group,
                "expected": expected_by_group
            })

            if group_ids is not None:
                indirect_df = indirect_df[indirect_df["group_id"].isin(selected_groups)].reset_index(drop=True)
            results["indirect"] = indirect_df

        # Direct standardization
        if "direct" in stdz:
            obs_total = np.sum(self.outcome_)
            population_rate = obs_total / n_samples * 100.0

            direct_preds = []
            for g_val in gamma:  # each provider's gamma
                p_temp = 1.0 / (1.0 + np.exp(-(g_val + self.xbeta_)))
                direct_preds.append(np.sum(p_temp))

            direct_preds = np.array(direct_preds)
            ds_ratio = direct_preds / obs_total
            ds_rate = np.clip(ds_ratio * population_rate, 0.0, 100.0)

            direct_df = pd.DataFrame({
                "group_id": self.groups_,
                "direct_ratio": ds_ratio,
                "direct_rate": ds_rate,
                "observed": np.full(len(self.groups_), obs_total),
                "expected": direct_preds
            })

            if group_ids is not None:
                direct_df = direct_df[direct_df["group_id"].isin(selected_groups)].reset_index(drop=True)
            results["direct"] = direct_df

        return results

    # -------------------------------------------------------------------------
    # BASIC (T-BASED) CI BOUNDS FOR GAMMA
    # -------------------------------------------------------------------------
    def _compute_ci_bounds(self, gamma: np.ndarray, se: np.ndarray, df: int, level: float, alternative: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute two-sided or one-sided confidence interval bounds for gamma using a t-distribution.

        Parameters
        ----------
        gamma : np.ndarray
            Point estimates of provider effects.
        se : np.ndarray
            Standard errors for gamma.
        df : int
            Degrees of freedom for the t-distribution.
        level : float
            Confidence level (e.g., 0.95 for 95% CI).
        alternative : str
            Hypothesis type: 'two_sided', 'greater', or 'less'.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds for the confidence intervals.

        Raises
        ------
        ValueError
            If 'alternative' is not one of the allowed values.
        """
        alpha = 1.0 - level
        if alternative == "two_sided":
            crit_value = t.ppf(1.0 - alpha / 2.0, df)
            lower = gamma - crit_value * se
            upper = gamma + crit_value * se
        elif alternative == "greater":
            crit_value = t.ppf(1.0 - alpha, df)
            lower = gamma - crit_value * se
            upper = np.full_like(gamma, np.inf)
        elif alternative == "less":
            crit_value = t.ppf(1.0 - alpha, df)
            lower = np.full_like(gamma, -np.inf)
            upper = gamma + crit_value * se
        else:
            raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")
        return lower, upper

    def _search_root(self, func, bracket: Tuple[float, float], max_attempts: int = 3) -> Optional[float]:
        """
        Attempt to find a root of 'func' within an initial bracket, expanding if necessary.

        Parameters
        ----------
        func : callable
            Function for which to find the root.
        bracket : Tuple[float, float]
            Initial [left, right] bracket for the root.
        max_attempts : int, default=3
            Number of bracket expansions to attempt.

        Returns
        -------
        Optional[float]
            The root if found, otherwise None.
        """
        left, right = bracket
        for i in range(max_attempts):
            try:
                sol = root_scalar(func, bracket=[left, right], method='bisect')
                if sol.converged:
                    return sol.root
            except ValueError:
                pass
            expand_amt = 5.0 * (i + 1)
            left -= expand_amt
            right += expand_amt
        logger.warning(f"Root-finding failed after {max_attempts} attempts with bracket {bracket}")
        return None

    def _get_no_all_events(self, group_idx: int) -> Tuple[bool, bool]:
        """
        Check if a provider has no events or all events.

        Parameters
        ----------
        group_idx : int
            Index of the provider group.

        Returns
        -------
        Tuple[bool, bool]
            (no_events, all_events) indicating if the provider has no events or all events.
        """
        sum_y = np.sum(self.outcome_[self.group_indices_ == group_idx])
        gsize = self.group_sizes_[group_idx]
        return sum_y == 0, sum_y == gsize

    def _score_ci_for_one_group(self, group_idx: int, alpha: float, alternative: str, gamma_guess: float) -> Tuple[float, float]:
        """
        Compute score-based confidence interval for a single provider's gamma.

        Parameters
        ----------
        group_idx : int
            Provider group index.
        alpha : float
            Significance level.
        alternative : str
            Hypothesis type: 'two_sided', 'greater', or 'less'.
        gamma_guess : float
            Initial guess for gamma.

        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds of the confidence interval.
        """
        # Check for no events or all events
        no_events, all_events = self._get_no_all_events(group_idx)
        observed = np.sum(self.outcome_[self.group_indices_ == group_idx])
        xbeta_group = self.xbeta_[self.group_indices_ == group_idx]

        qnorm_half = norm.ppf(1.0 - alpha / 2.0)
        qnorm_1side = norm.ppf(1.0 - alpha)

        def upper_func(gamma):
            probs = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
            score = (observed - probs.sum()) / np.sqrt((probs * (1.0 - probs)).sum())
            return score + (qnorm_half if alternative == "two_sided" else qnorm_1side if alternative == "less" else 0)

        def lower_func(gamma):
            probs = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
            score = (observed - probs.sum()) / np.sqrt((probs * (1.0 - probs)).sum())
            return score - (qnorm_half if alternative == "two_sided" else qnorm_1side if alternative == "greater" else 0)

        if no_events:
            def no_events_func(gamma):
                probs = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
                return qnorm_1side - probs.sum() / np.sqrt((probs * (1.0 - probs)).sum())
            upper_bound = self._search_root(no_events_func, (gamma_guess - 5.0, gamma_guess + 5.0))
            return (-np.inf, upper_bound if upper_bound is not None else np.inf)

        if all_events:
            def all_events_func(gamma):
                probs = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
                return ((1.0 - probs).sum() / np.sqrt((probs * (1.0 - probs)).sum())) - qnorm_1side
            lower_bound = self._search_root(all_events_func, (gamma_guess - 5.0, gamma_guess + 5.0))
            return (lower_bound if lower_bound is not None else -np.inf, np.inf)

        lower_bound, upper_bound = -np.inf, np.inf
        if alternative in ["two_sided", "less"]:
            upper_bound = self._search_root(upper_func, (gamma_guess, gamma_guess + 5.0)) or np.inf
        if alternative in ["two_sided", "greater"]:
            lower_bound = self._search_root(lower_func, (gamma_guess - 5.0, gamma_guess)) or -np.inf
        return (lower_bound, upper_bound)

    def _exact_ci_for_one_group(self, group_idx: int, alpha: float, alternative: str, gamma_guess: float) -> (float, float):
        """
        Compute 'exact' CI for a single provider's gamma using the fast_poibin package.

        Parameters
        ----------
        group_idx : int
            Provider index in [0, M-1].
        alpha : float
            1 - confidence level.
        alternative : {'two_sided','greater','less'}
            Specifies the alternative hypothesis.
        gamma_guess : float
            An initial guess for gamma.

        Returns
        -------
        (float, float)
            (lower_bound, upper_bound) for this provider's gamma.

        Notes
        -----
        - Uses the fast_poibin.PoiBin class for computing PMF and CDF.
        - For the "mid-p" approach when alternative == "two_sided", we compute:
            lower tail: P(X <= obs-1) + 0.5 * P(X=obs) - alpha/2
            upper tail: P(X >= obs) + 0.5 * P(X=obs) - alpha/2
                      = (1 - P(X <= obs)) + P(X=obs) + 0.5 * P(X=obs) - alpha/2
                      = 1 - P(X <= obs-1) - 0.5 * P(X=obs) - alpha/2
        """
        # Check for no events or all events
        no_events, all_events = self._get_no_all_events(group_idx)
        observed = int(np.sum(self.outcome_[self.group_indices_ == group_idx]))
        xbeta_group = self.xbeta_[self.group_indices_ == group_idx]
        n_trials = len(xbeta_group)

        if n_trials == 0:
            print(f"Warning: Group {group_idx} has no observations. Returning (-inf, inf).")
            return (-np.inf, np.inf)

        if no_events:
            if alternative not in ["two_sided", "less"]:
                return (-np.inf, np.inf)

            def upper_func(gamma):
                pvec = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
                if np.any(np.isnan(pvec)) or np.any(np.isinf(pvec)):
                    return np.nan
                pb = PoiBin(pvec)
                alpha_level = alpha if alternative == "less" else alpha / 2.0
                return 0.5 * pb.pmf[0] - alpha_level if len(pb.pmf) > 0 else np.nan

            upper_bound = self._search_root(upper_func, (gamma_guess - 10.0, gamma_guess + 10.0))
            return (-np.inf, upper_bound if upper_bound is not None else np.inf)

        if all_events:
            if alternative not in ["two_sided", "greater"]:
                return (-np.inf, np.inf)

            def lower_func(gamma):
                pvec = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
                if np.any(np.isnan(pvec)) or np.any(np.isinf(pvec)):
                    return np.nan
                pb = PoiBin(pvec)
                alpha_level = alpha if alternative == "greater" else alpha / 2.0
                pmf_n = pb.pmf[n_trials] if len(pb.pmf) > n_trials else np.nan
                return pmf_n - alpha_level if alternative == "greater" else 0.5 * pmf_n - alpha_level

            lower_bound = self._search_root(lower_func, (gamma_guess - 10.0, gamma_guess + 10.0))
            return (lower_bound if lower_bound is not None else -np.inf, np.inf)

        def upper_func(gamma):
            pvec = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
            if np.any(np.isnan(pvec)) or np.any(np.isinf(pvec)):
                return np.nan
            pb = PoiBin(pvec)
            if len(pb.pmf) <= observed or len(pb.cdf) <= observed:
                print(f"Warning: fast_poibin arrays too short for obs={observed}.")
                return np.nan
            pmf_obs = pb.pmf[observed]
            cdf_minus_1 = pb.cdf[observed - 1] if observed > 0 else 0.0
            if alternative == "two_sided":
                return cdf_minus_1 + 0.5 * pmf_obs - alpha / 2.0
            elif alternative == "less":
                return cdf_minus_1 - alpha
            return 1.0

        def lower_func(gamma):
            pvec = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
            if np.any(np.isnan(pvec)) or np.any(np.isinf(pvec)):
                return np.nan
            pb = PoiBin(pvec)
            if len(pb.pmf) <= observed or len(pb.cdf) <= observed:
                print(f"Warning: fast_poibin arrays too short for obs={observed}.")
                return np.nan
            pmf_obs = pb.pmf[observed]
            cdf_obs = pb.cdf[observed]
            if alternative == "two_sided":
                return (1.0 - cdf_obs) + 0.5 * pmf_obs - alpha / 2.0
            elif alternative == "greater":
                cdf_minus_1 = pb.cdf[observed - 1] if observed > 0 else 0.0
                return (1.0 - cdf_minus_1) - alpha
            return 1.0

        lower_bound, upper_bound = -np.inf, np.inf
        search_interval_half_width = 5.0
        if alternative in ["two_sided", "less"]:
            upper_bound = self._search_root(upper_func, (gamma_guess, gamma_guess + search_interval_half_width)) or np.inf
        if alternative in ["two_sided", "greater"]:
            lower_bound = self._search_root(lower_func, (gamma_guess - search_interval_half_width, gamma_guess)) or -np.inf

        if lower_bound > upper_bound:
            print(f"Warning: Lower bound {lower_bound} > Upper bound {upper_bound} for group {group_idx}. Resetting.")
            return (-np.inf, np.inf)
        return (lower_bound, upper_bound)
    
    def _validate_ci_arguments(self, option: str, stdz: Union[str, list], alternative: str) -> None:
        """
        Validate arguments for confidence interval calculations.

        Parameters
        ----------
        option : {'gamma', 'SM'}
            Whether to calculate intervals for provider effects or standardized measures.
        stdz : str or list
            Standardization method(s).
        alternative : str
            Hypothesis type.

        Raises
        ------
        ValueError 
            If arguments are inconsistent.
        """
        if option not in {"gamma", "SM"}:
            raise ValueError("Argument 'option' must be one of {'gamma','SM'}.")

        if isinstance(stdz, str):
            stdz = [stdz]

        if option == "gamma" and alternative not in {"two_sided", "greater", "less"}:
            raise ValueError("option='gamma' requires 'alternative' in {'two_sided','greater','less'}.")

        if option == "SM" and not any(m in stdz for m in ["indirect", "direct"]):
            raise ValueError("If option='SM', 'stdz' must include at least one of {'indirect','direct'}.")

    def _compute_gamma_intervals(
        self,
        group_ids: np.ndarray,
        level: float,
        alternative: str,
        test_method: str
    ) -> pd.DataFrame:
        """
        Compute intervals for all providers' gamma (option='gamma'), 
        delegating 'wald' to _compute_ci_bounds, 
        and 'score'/'exact' to specialized logic.

        Returns
        -------
        pd.DataFrame with columns ["group_id","gamma","gamma_lower","gamma_upper"].
        """
        gamma_vals = self.coefficients_["gamma"].flatten()
        se_gamma = np.sqrt(self.variances_["gamma"].flatten())
        n_obs = len(self.outcome_)
        p = len(self.coefficients_["beta"])
        m = len(gamma_vals)
        df = n_obs - (m + p)
        alpha = 1.0 - level

        records = []
        for i, gid in enumerate(self.groups_):
            if gid not in group_ids:
                continue

            gamma_est = gamma_vals[i]
            if test_method == "wald":
                lower, upper = -inf, inf
                if alternative in {"two_sided", "less", "greater"}:
                    # Reuse _compute_ci_bounds for just *one* provider if we wish
                    g_lower, g_upper = self._compute_ci_bounds(
                        gamma=np.array([gamma_est]),
                        se=np.array([se_gamma[i]]),
                        df=df,
                        level=level,
                        alternative=alternative
                    )
                    lower, upper = g_lower[0], g_upper[0]
                else:
                    raise ValueError("alternative must be one of {'two_sided','less','greater'}")

            elif test_method == "score":
                lower, upper = self._score_ci_for_one_group(
                    group_idx=i, alpha=alpha, alternative=alternative, gamma_guess=gamma_est
                )
            elif test_method == "exact":
                lower, upper = self._exact_ci_for_one_group(
                    group_idx=i, alpha=alpha, alternative=alternative, gamma_guess=gamma_est
                )
            else:
                raise ValueError("test_method must be wald, score, exact")

            records.append({
                "group_id": gid,
                "gamma": gamma_est,
                "gamma_lower": lower,
                "gamma_upper": upper
            })

        df_res = pd.DataFrame(records)
        return df_res

    def _compute_sm_intervals(
        self,
        group_ids: np.ndarray,
        level: float,
        stdz: Union[str, list],
        measure: Union[str, list],
        alternative: str,
        test_method: str,
        null: Union[str,float]
    ) -> dict:
        """
        Factor out the logic for "option='SM'" to a dedicated helper,
        returning intervals for indirect/direct ratio/rate. 
        Based on your prior code or the approach from your R function.
        """
        alpha = 1.0 - level
        gamma_vals = self.coefficients_["gamma"].flatten()
        n_obs = len(self.outcome_)

        # force measure to list
        if isinstance(measure, str):
            measure = [measure]
        want_ratio = "ratio" in measure
        want_rate  = "rate"  in measure

        # get the raw indirect/direct values
        sm_data = self.calculate_standardized_measures(
            group_ids=None, stdz=stdz, null=null
        )

        # build gamma CI maps
        df_gamma_ci = self._compute_gamma_intervals(
            group_ids=self.groups_,
            level=level,
            alternative=alternative,
            test_method=test_method
        )
        gamma_lower_map = df_gamma_ci.set_index("group_id")["gamma_lower"].to_dict()
        gamma_upper_map = df_gamma_ci.set_index("group_id")["gamma_upper"].to_dict()

        population_rate = np.mean(self.outcome_) * 100.0
        results = {}

        # ---- INDIRECT ----
        if "indirect" in sm_data:
            base = sm_data["indirect"].copy().set_index("group_id")

            # always compute ratio CIs if either ratio OR rate was requested
            if want_ratio or want_rate:
                rl, ru = [], []
                for gid, row in base.iterrows():
                    if gid not in group_ids:
                        rl.append(np.nan); ru.append(np.nan); continue
                    gl = gamma_lower_map[gid]; gu = gamma_upper_map[gid]
                    expv = row["expected"]
                    low_sum = self._sum_logistic_for_provider(gid, gl)
                    up_sum  = self._sum_logistic_for_provider(gid, gu)
                    rl.append(low_sum/expv if expv>0 else np.nan)
                    ru.append(up_sum /expv if expv>0 else np.nan)
                base["ci_ratio_lower"] = rl
                base["ci_ratio_upper"] = ru

            # only export the ratio table if they asked for it
            if want_ratio:
                df_ratio = base.copy()
                df_ratio.attrs.update({
                    "confidence_level": f"{level*100}%",
                    "description": "Indirect Standardized Ratio",
                    "model": type(self).__name__
                })
                results["indirect_ratio"] = df_ratio.copy()

            # now compute rate, reusing ratio bounds
            if want_rate:
                # make a fresh copy so we don't clobber df_ratio
                df_rate = base.copy()
                # ci_rate_lower = ci_ratio_lower * pop_rate
                df_rate["ci_rate_lower"] = np.clip(
                    df_rate["ci_ratio_lower"] * population_rate, 0, 100
                )
                df_rate["ci_rate_upper"] = np.clip(
                    df_rate["ci_ratio_upper"] * population_rate, 0, 100
                )
                df_rate.attrs.update({
                    "confidence_level": f"{level*100}%",
                    "description": "Indirect Standardized Rate",
                    "model": type(self).__name__,
                    "population_rate": population_rate
                })
                results["indirect_rate"] = df_rate.copy()

        # ---- DIRECT ----
        if "direct" in sm_data:
            base = sm_data["direct"].copy().set_index("group_id")

            if want_ratio or want_rate:
                rl, ru = [], []
                for gid, row in base.iterrows():
                    if gid not in group_ids:
                        rl.append(np.nan); ru.append(np.nan); continue
                    gl = gamma_lower_map[gid]; gu = gamma_upper_map[gid]
                    obs  = row["observed"]
                    low_sum = self._sum_logistic_overall(gl)
                    up_sum  = self._sum_logistic_overall(gu)
                    rl.append(low_sum/obs if obs>0 else np.nan)
                    ru.append(up_sum /obs if obs>0 else np.nan)
                base["ci_ratio_lower"] = rl
                base["ci_ratio_upper"] = ru

            if want_ratio:
                df_ratio = base.copy()
                df_ratio.attrs.update({
                    "confidence_level": f"{level*100}%",
                    "description": "Direct Standardized Ratio",
                    "model": type(self).__name__
                })
                results["direct_ratio"] = df_ratio.copy()

            if want_rate:
                df_rate = base.copy()
                df_rate["ci_rate_lower"] = np.clip(
                    df_rate["ci_ratio_lower"] * population_rate, 0, 100
                )
                df_rate["ci_rate_upper"] = np.clip(
                    df_rate["ci_ratio_upper"] * population_rate, 0, 100
                )
                df_rate.attrs.update({
                    "confidence_level": f"{level*100}%",
                    "description": "Direct Standardized Rate",
                    "model": type(self).__name__,
                    "population_rate": population_rate
                })
                results["direct_rate"] = df_rate.copy()

        # finally, turn group_id back into a column
        for k, df in results.items():
            if df.index.name == "group_id":
                results[k] = df.reset_index()

        return results
    
    def _sum_logistic_for_provider(self, gid, gamma_val):
        """Sum logistic(gamma_val + xbeta_) for the observations belonging to provider gid."""
        idx = np.where(self.groups_ == gid)[0]
        if len(idx) == 0: return np.nan
        group_idx = idx[0]
        mask = (self.group_indices_ == group_idx)
        xbeta_group = self.xbeta_[mask]
        pvals = 1.0/(1.0 + np.exp(-(gamma_val + xbeta_group)))
        return pvals.sum()

    def _sum_logistic_overall(self, gamma_val):
        """Sum logistic(gamma_val + xbeta_) over the entire dataset (for direct approach)."""
        pvals = 1.0/(1.0 + np.exp(-(gamma_val + self.xbeta_)))
        return pvals.sum()

    # -------------------------------------------------------------------------
    # PUBLIC METHOD TO COMPUTE CONFIDENCE INTERVALS
    # -------------------------------------------------------------------------
    def calculate_confidence_intervals(
        self,
        group_ids: Optional[Union[list, np.ndarray]] = None,
        level: float = 0.95,
        option: str = "SM",
        stdz: Union[str, list] = "indirect",
        null: Union[str, float] = "median",
        measure: Union[str, list] = ("rate", "ratio"),
        alternative: str = "two_sided",
        test_method: str = "exact"
    ) -> dict:
        """
        Compute confidence intervals for either provider effects (option="gamma") or
        standardized measures (option="SM") in a logistic fixed-effects model.

        When option="gamma", the method calculates two-sided confidence intervals
        for each provider's fixed effect (γ) using a specified test method 
        ("wald", "score", or "exact"). The user must supply alternative="two_sided" 
        if option="gamma", and the method ignores any stdz or measure arguments.

        When option="SM", the method produces confidence intervals for indirect or direct
        standardized measures (ratio and/or rate) at the specified confidence level.
        It calculates provider-specific γ intervals first, then transforms lower/upper
        γ bounds into standardized measure intervals. The user must supply stdz ∈ 
        {"indirect","direct"} and measure ∈ {"ratio","rate"}.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of providers to include. If None, all providers are included.
        level : float, default=0.95
            Confidence level → alpha = 1 - level significance threshold.
        option : {'gamma','SM'}, default='SM'
            Type of intervals to compute:
            - 'gamma': intervals for each provider effect (γ).
            - 'SM': intervals for standardized measures (indirect/direct ratio/rate).
        stdz : {'indirect','direct'} or list, default='indirect'
            Standardization method(s) to use if option="SM". Ignored if option="gamma".
        null : {'median','mean'} or float, default='median'
            Baseline norm for direct standardization. Ignored if option="gamma".
        measure : {'rate','ratio'} or list of these, default=('rate','ratio')
            Which standardized measures to produce intervals for if option="SM".
            Ignored if option="gamma".
        alternative : {'two_sided','greater','less'}, default='two_sided'
            Hypothesis direction. Must be "two_sided" if option="gamma".
            If option="SM", relevant only for how γ intervals are computed in "score" or "exact".
        test_method : {'wald','score','exact'}, default='wald'
            Method for computing γ intervals. 
            - 'wald' uses a t-based approach (via self._compute_ci_bounds).
            - 'score' uses partial derivative root-finding. 
            - 'exact' uses Poisson-binomial root-finding logic.

        Returns
        -------
        dict
            If option="gamma": 
                { "gamma_ci": DataFrame with columns [group_id, gamma, gamma_lower, gamma_upper] }
            If option="SM": 
                Possibly includes keys:
                - "indirect_ratio"
                - "indirect_rate"
                - "direct_ratio"
                - "direct_rate"
            depending on stdz and measure selected.

        Notes
        -----
        - By default, if option="gamma", the method enforces alternative="two_sided".
        - For option="SM", standardization intervals are derived from summing logistic(γ_lower + Xβ)
        or logistic(γ_upper + Xβ), dividing by "expected," and (for rates) multiplying by the 
        population event rate. 
        - DataFrame .attrs usage:
        Storing metadata (e.g., "confidence_level", "description") in DataFrame.attrs is a
        convenient way to keep additional information attached without adding columns. This
        is optional. Some developers prefer returning a separate dictionary for metadata.
        """

        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("Model must be fitted, with valid coefficients_ and variances_.")

        # Enforce that if option="gamma", user must supply two_sided, 
        # and ignore stdz, measure, null
        if option == "gamma":
            # Force the intervals to be two-sided
            if alternative != "two_sided":
                raise ValueError(
                    "For option='gamma', only two-sided intervals are supported."
                )
            # If the user provided stz or measure, that doesn't make sense here
            if stdz is not None and (isinstance(stdz, str) or isinstance(stdz, list)):
                # It's simpler just to ignore them or raise an error:
                # We'll raise an error so user doesn't get confused:
                if (isinstance(stdz, list) and len(stdz) > 0) or (isinstance(stdz, str) and stdz != "indirect"):
                    raise ValueError("For option='gamma', stdz is not applicable.")
            if measure is not None and (
                (isinstance(measure, list) and len(measure) > 0) or
                (isinstance(measure, str))
            ):
                raise ValueError("For option='gamma', 'measure' is not applicable.")
            
            # Subset or gather final group IDs
            final_groups = self.groups_ if group_ids is None else [g for g in self.groups_ if g in group_ids]
            # Compute gamma intervals
            gamma_ci_df = self._compute_gamma_intervals(
                group_ids=np.array(final_groups),
                level=level,
                alternative=alternative,
                test_method=test_method
            )
            return {"gamma_ci": gamma_ci_df}

        elif option == "SM":
            # For standardized measures, we rely on a helper approach
            # that handles indirect/direct ratio/rate intervals:
            final_groups = self.groups_ if group_ids is None else [g for g in self.groups_ if g in group_ids]
            return self._compute_sm_intervals(
                group_ids=np.array(final_groups),
                level=level,
                stdz=stdz,
                measure=measure,
                alternative=alternative,
                test_method=test_method,
                null=null
            )

        else:
            raise ValueError("option must be either 'gamma' or 'SM'.")
    
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
        """
        Conduct hypothesis tests on provider effects.

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
        """
        Compute Wald test for gamma coefficients.

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
        """
        Compute Score test for gamma coefficients.

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
        """
        Compute Poisson-Binomial exact test for gamma coefficients.

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
        """
        Compute Bootstrap exact test for gamma coefficients.

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

    def plot_funnel(
        self,
        test_method: str = "score", # "score" or "poibin_exact"
        null: Union[str, float] = "median",
        target: float = 1.0,
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
        edge_color: Optional[str] = None,
        edge_linewidth: float = 0,
        add_grid: bool = True,
        grid_style: str = ':',
        grid_alpha: float = 0.6,
        remove_top_right_spines: bool = True,
        figure_size: Tuple[float, float] = (8, 6),
        plot_title: str = "Funnel Plot (Indirect Standardization)",
        xlab: str = "Precision",
        ylab: str = "Indirectly Standardized Ratio (O/E)",
        legend_location: str = 'best' # Location for legend
     ) -> None:
        """
        Create a funnel plot comparing provider performance (Indirect Ratio O/E).

        Control limits and point flagging are based on the specified test method
        ('score' or 'poibin_exact'). Precision is calculated as Expected^2 / Variance.

        Parameters:
        -----------
        test_method : str, default="score"
            Method for flagging and control limits: "score" or "poibin_exact".
            "poibin_exact" requires the 'poibin' library.
        null : str or float, default="median"
            Baseline for provider effects (gamma) used in null hypothesis calculations.
            Can be "median" or a specific float value.
        target : float, default=1.0
            Reference performance value (target line on the plot).
        alpha : float or List[float], default=0.05
            Significance level(s) for control limits (e.g., 0.05 for 95% limits).
            If a list is provided, multiple limit boundaries will be drawn.
        labels : List[str], default=["Lower", "Expected", "Higher"]
            Labels for provider performance categories based on flags (-1, 0, 1).
        point_colors : List[str]
            Colors for provider points based on performance flag. Cycles through list.
        point_shapes : List[str]
            Marker shapes for provider points based on performance flag. Cycles through list.
        point_size : float, default=2.0
            Scaling factor for marker size (base size is typically ~20-50).
        point_alpha : float, default=0.8
            Marker transparency (0 to 1).
        line_size : float, default=0.8
            Thickness for target and control limit lines.
        target_linestyle : str, default='--'
            Line style for the target reference line (matplotlib style).
        font_size : float, default=12
            Base font size for labels and title.
        tick_label_size : float, default=10
            Font size for axis tick labels.
        cl_line_colors : str or List[str], optional
            Color(s) for the control limit lines. If a list, cycles through alphas. Defaults to "grey".
        cl_line_styles : str or List[str], optional
            Line style(s) for control limits. If None, defaults based on number of alphas.
            E.g., ['-', '--', ':'] for 3 alpha levels.
        fill_color : str, default="#A6CEE3"
            Fill color for the area between the outermost control limits (smallest alpha).
        fill_alpha : float, default=0.25
            Transparency of the control limit fill area (0 to 1).
        edge_color : str or None, default=None
            Edge color for scatter points. Set to None for no edges.
        edge_linewidth : float, default=0.5
             Line width for scatter point edges.
        add_grid : bool, default=True
             Whether to add a background grid to the plot.
        grid_style : str, default=':'
             Line style for the grid.
        grid_alpha : float, default=0.6
             Transparency for the grid lines.
        remove_top_right_spines : bool, default=True
             Whether to remove the top and right axis lines (spines) for a cleaner look.
        figure_size : Tuple[float, float], default=(8, 6)
            Figure size in inches (width, height).
        plot_title : str, default="Funnel Plot (Indirect Standardization)"
            Title for the plot.
        xlab : str, default="Precision (Expected^2 / Variance)"
            Label for the x-axis.
        ylab : str, default="Indirectly Standardized Ratio (O/E)"
            Label for the y-axis.
                 legend_location : str, default='best'
             Location string for the legend (e.g., 'best', 'upper right', 'lower left',
             'center left', 'upper center', 'center', or tuple (x,y)).

        Raises:
        -------
        ValueError
            If model is not fitted, arguments are invalid, or required attributes missing.

        """

        # --- Input Validation ---
        if self.coefficients_ is None or self.groups_ is None or self.group_sizes_ is None:
            raise ValueError("Model must be fitted with coefficients, groups, and group sizes.")
        if self.outcome_ is None or self.xbeta_ is None or self.group_indices_ is None:
             raise ValueError("Model requires 'outcome_', 'xbeta_', and 'group_indices_' attributes for funnel plot calculations.")
        allowed_tests = ["score", "poibin_exact"]
        if test_method not in allowed_tests:
            raise ValueError(f"Argument 'test_method' must be one of {allowed_tests}.")
        if not isinstance(labels, list) or len(labels) != 3:
            raise ValueError("'labels' must be a list of three strings.")
        if not isinstance(point_colors, list) or len(point_colors) < 1:
            raise ValueError("'point_colors' must be a non-empty list.")
        if not isinstance(point_shapes, list) or len(point_shapes) < 1:
            raise ValueError("'point_shapes' must be a non-empty list.")

        a_list = sorted([alpha] if isinstance(alpha, (float, int)) else alpha)
        if not all(0 < a < 1 for a in a_list):
            raise ValueError("'alpha' must be between 0 and 1.")
        alpha_test = min(a_list) # Use the smallest alpha for flagging

        # --- Data Preparation ---
        # 1. Get Standardized Measures & Set Index
        try:
            sm_results = self.calculate_standardized_measures(stdz="indirect", null=null)
            df = sm_results["indirect"].copy()
            if len(df) == len(self.groups_):
                 df.index = self.groups_
            else: 
                raise ValueError(f"Length mismatch: std measures ({len(df)}) vs groups ({len(self.groups_)}).")
            required_cols = ["indirect_ratio", "observed", "expected"]
            if not all(col in df.columns for col in required_cols): 
                raise ValueError(f"Missing required columns: {required_cols}")
        except (AttributeError, KeyError, ValueError) as e: 
            raise ValueError(f"Failed data prep (std measures). Error: {e}")

        # 2. Calculate Probabilities under Null
        gamma_vals = self.coefficients_["gamma"].flatten()
        if null == "median": 
            gamma_null = np.median(gamma_vals)
        elif isinstance(null, (int, float)): 
            gamma_null = float(null)
        else: 
            raise ValueError("Argument 'null' must be 'median' or a numeric value.")

        pvec_null_all = 1.0 / (1.0 + np.exp(-(gamma_null + self.xbeta_)))
        pvec_null_all = np.clip(pvec_null_all, 1e-10, 1 - 1e-10)
        probs_by_group_idx = [pvec_null_all[self.group_indices_ == i] for i in range(len(self.groups_))]

        # 3. Calculate Variance and Precision
        group_vars_null = np.array([np.sum(p * (1 - p)) for p in probs_by_group_idx])
        group_vars_null_clipped = np.maximum(group_vars_null, 1e-12)
        expected_counts_null = np.array([np.sum(p) for p in probs_by_group_idx])
        df['expected_null'] = expected_counts_null; df['variance_null'] = group_vars_null
        df["precision"] = np.where(df['variance_null'] > 1e-14, np.square(df['expected_null']) / group_vars_null_clipped, np.inf)
        max_finite_precision = df.loc[np.isfinite(df['precision']), 'precision'].max(skipna=True)
        if pd.isna(max_finite_precision): 
            max_finite_precision = 1
        df["precision"].replace(np.inf, max_finite_precision * 1.1, inplace=True)

        # 4. Get Flags
        test_df = self.test(null=null, level=1.0 - alpha_test, test_method=test_method)
        df["flag"] = test_df.loc[df.index, "flag"].astype(int)

        # --- Calculate Control Limits ---
        limits_list = []
        for a in a_list:
            if test_method == "score":
                z_val = norm.ppf(1 - a / 2); se_ratio = np.sqrt(1.0 / df["precision"])
                se_ratio[~np.isfinite(se_ratio)] = 0.0
                control_lower = target - z_val * se_ratio; control_upper = target + z_val * se_ratio
                limits_df = pd.DataFrame(
                    {"precision": df["precision"], "control_lower": np.maximum(0, control_lower), "control_upper": control_upper, "alpha": a}, 
                    index=df.index)
                limits_list.append(limits_df)
            elif test_method == "poibin_exact":
                cl_lower_ratio, cl_upper_ratio, group_precision_list = [], [], []
                for i, g_id in enumerate(df.index):
                     probs_g = probs_by_group_idx[i]
                     expected_g = df.loc[g_id, 'expected_null']
                     precision_g = df.loc[g_id, 'precision']

                     if len(probs_g) == 0 or expected_g < 1e-10:
                         cl_lower_ratio.append(0);
                         cl_upper_ratio.append(np.inf if expected_g < 1e-10 else 0)
                         group_precision_list.append(precision_g)
                         continue
                     pb = PoiBin(probs_g)
                     o_lower = pb.quantile(a / 2)
                     o_upper = pb.quantile(1 - a / 2)
                     limit_lower = o_lower / expected_g
                     limit_upper = o_upper / expected_g
                     cl_lower_ratio.append(limit_lower) 
                     cl_upper_ratio.append(limit_upper)
                     group_precision_list.append(precision_g)
                limits_df = pd.DataFrame(
                    {"precision": group_precision_list, "control_lower": np.maximum(0, cl_lower_ratio), "control_upper": cl_upper_ratio, "alpha": a}, 
                    index=df.index)
                limits_list.append(limits_df)
        limits_all_alphas = pd.concat(limits_list)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=figure_size)

        # Plot control limits
        limits_all_alphas = limits_all_alphas.sort_values("precision")
        outer_alpha = min(a_list)

        # Define line styles and colors (ensure enough styles/colors for alphas)
        if cl_line_styles is None: 
            default_styles = ['-', '--', ':', '-.']
            num_alphas = len(a_list)
            cl_line_styles = [default_styles[i % len(default_styles)] for i in range(num_alphas)]

        elif isinstance(cl_line_styles, str): 
            cl_line_styles = [cl_line_styles] * len(a_list)

        if cl_line_colors is None or isinstance(cl_line_colors, str): 
            cl_line_colors = [cl_line_colors or 'grey'] * len(a_list)

        sorted_alphas = sorted(a_list)
        style_map = dict(zip(sorted_alphas, cl_line_styles))
        color_map = dict(zip(sorted_alphas, cl_line_colors))

        # Store handles and labels for legend creation later
        legend_handles = []
        legend_labels = []

        # Plot fill for outermost limit - ADD LEGEND ENTRY
        outer_limits = limits_all_alphas[limits_all_alphas["alpha"] == outer_alpha]
        label_outer_ci = f'{int((1-outer_alpha)*100)}% CI'
        fill_handle = ax.fill_between(outer_limits["precision"], 
                                      outer_limits["control_lower"], 
                                      outer_limits["control_upper"],
                                      color=fill_color, 
                                      alpha=fill_alpha, 
                                      label=label_outer_ci) # Label for fill
        legend_handles.append(fill_handle)
        legend_labels.append(label_outer_ci)

        # Plot lines for all limits
        for a in sorted_alphas:
            subset = limits_all_alphas[limits_all_alphas["alpha"] == a]
            line_label = None
            if a != outer_alpha: # Only label inner CI lines
                 line_label = f'{int((1-a)*100)}% CI'

            # Plot lines, store handle only if labeled
            line_lower, = ax.plot(subset["precision"], 
                                  subset["control_lower"],
                                  linestyle=style_map[a], 
                                  color=color_map[a], 
                                  linewidth=line_size, 
                                  label=line_label)
            ax.plot(subset["precision"], subset["control_upper"], # Upper line doesn't need label
                    linestyle=style_map[a], color=color_map[a], linewidth=line_size)

            if line_label: # If we added a label, store the handle
                 legend_handles.append(line_lower)
                 legend_labels.append(line_label)

        # Plot target line - NO LABEL
        ax.axhline(y=target, color="black", linestyle=target_linestyle, linewidth=line_size) # No label

        # Plot provider points - ADD LEGEND ENTRIES
        present_flags = sorted(df["flag"].unique())
        flag_map = {-1: 0, 0: 1, 1: 2} # Map flag to index in labels/colors/shapes

        for flag in present_flags:
            subset = df[df["flag"] == flag]
            label_idx = flag_map[flag]
            count = len(subset)
            point_label = f"{labels[label_idx]} ({count})"

            # Plot points and store the handle for the legend
            scatter_handle = ax.scatter(subset["precision"], 
                                        subset["indirect_ratio"],
                                        marker=point_shapes[label_idx % len(point_shapes)],
                                        color=point_colors[label_idx % len(point_colors)],
                                        s=point_size * 30,
                                        alpha=point_alpha,
                                        edgecolor=edge_color,
                                        linewidth=edge_linewidth if edge_color else 0,
                                        label=point_label) # Use label here
            legend_handles.append(scatter_handle)
            legend_labels.append(point_label)


        # --- Final Touches & Prettier Plot ---
        ax.set_xlabel(xlab, fontsize=font_size)
        ax.set_ylabel(ylab, fontsize=font_size)
        ax.set_title(plot_title, fontsize=font_size + 2, pad=15)
        ax.tick_params(axis="both", labelsize=tick_label_size)

        # Set Axis Limits
        min_y = 0
        max_y_points = df["indirect_ratio"].max(skipna=True)
        max_y_cl = limits_all_alphas["control_upper"].replace(np.inf, np.nan).max(skipna=True)
        max_y = max(target, max_y_points if pd.notna(max_y_points) else target, max_y_cl if pd.notna(max_y_cl) else target)
        ax.set_ylim(min_y, max_y * 1.1)
        max_x = df["precision"].max(skipna=True)
        ax.set_xlim(left=0, right=(max_x * 1.05 if pd.notna(max_x) else 1))

        # Add Grid
        if add_grid: ax.grid(True, linestyle=grid_style, alpha=grid_alpha, axis='both', color='lightgrey') # Lighter grid color

        # Remove Spines
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8); ax.spines['bottom'].set_linewidth(0.8)

        # Create Legend using collected handles and labels (automatically handles duplicates)
        ax.legend(handles=legend_handles, 
                  labels=legend_labels,
                  fontsize=font_size - 2, 
                  loc=legend_location) # Add a title to legend


        plt.tight_layout()
        plt.show()

    def plot_provider_effects(
        self,
        group_ids=None,
        level: float = 0.95,
        test_method: str = 'wald',
        use_flags: bool = True,
        null: Union[str, float] = 'median',
        **plot_kwargs
    ) -> None:
        """
        Plot provider-specific effects (gamma) with confidence intervals in a caterpillar plot.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot. If None, all providers are included.
        level : float, default=0.95
            Confidence level for intervals.
        test_method : str, default='wald'
            Method for computing intervals: 'wald', 'score', or 'exact'.
        use_flags : bool, default=True
            Whether to color-code providers based on flags from the test method.
        null : str or float, default='median'
            Null hypothesis for gamma in the test method.
        **plot_kwargs
            Additional arguments passed to plot_caterpillar (e.g., title, point_color).
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted first.")

        # 1) Compute gamma‐CIs
        ci_results = self.calculate_confidence_intervals(
            group_ids=group_ids,
            level=level,
            option='gamma',
            test_method=test_method
        )
        df = ci_results['gamma_ci']

        # 2) Merge in flags based on the same `null`
        if use_flags:
            test_df = self.test(
                providers=group_ids,
                level=level,
                test_method=test_method,
                null=null
            )
            df = df.merge(
                test_df[['flag']],
                left_on='group_id',
                right_index=True,
                how='left'
            )

        # 3) Figure out what “null” really is
        gamma_vals = self.coefficients_["gamma"].flatten()
        if   null == "median":
            gamma_null = np.median(gamma_vals)
        elif null == "mean":
            gamma_null = np.mean(gamma_vals)
        elif isinstance(null, (int, float)):
            gamma_null = float(null)
        else:
            raise ValueError("`null` must be 'median', 'mean', or a numeric value")

        # 4) Orientation & axis‐labels
        orientation = plot_kwargs.pop('orientation', 'vertical')
        if orientation not in ('vertical','horizontal'):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        est_label = "Gamma Estimate"
        grp_label = "Provider"

        if orientation == 'vertical':
            xlab = plot_kwargs.pop('xlabel', est_label)
            ylab = plot_kwargs.pop('ylabel', grp_label)
        else:
            xlab = plot_kwargs.pop('xlabel', grp_label)
            ylab = plot_kwargs.pop('ylabel', est_label)

        # 5) Use gamma_null as the default ref‑line
        refline_value = plot_kwargs.pop('refline_value', gamma_null)

        # 6) Fire off the caterpillar
        plot_caterpillar(
            df=df,
            estimate_col='gamma',
            ci_lower_col='gamma_lower',
            ci_upper_col='gamma_upper',
            group_col='group_id',
            flag_col='flag' if use_flags else None,
            orientation=orientation,
            refline_value=refline_value,
            xlab=xlab,
            ylab=ylab,
            plot_title=plot_kwargs.pop('title', 'Provider Effects (Gamma)'),
            **plot_kwargs
        )
        
    def plot_standardized_measures(
        self,
        group_ids=None,
        level: float = 0.95,
        stdz: str = 'indirect',
        measure: str = 'ratio',
        test_method: str = 'score',
        use_flags: bool = True,
        null: Union[str, float] = 'median',
        **plot_kwargs
    ) -> None:
        """
        Plot standardized measures (e.g., indirect ratio or rate) with confidence intervals in a caterpillar plot.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot. If None, all providers are included.
        level : float, default=0.95
            Confidence level for intervals.
        stdz : str, default='indirect'
            Standardization method: 'indirect' or 'direct'.
        measure : str, default='ratio'
            Measure to plot: 'rate' or 'ratio'.
        test_method : str, default='score'
            Method for computing intervals: 'wald', 'score', or 'exact'.
        use_flags : bool, default=True
            Whether to color-code providers based on flags from the test method.
        null : str or float, default='median'
            Null hypothesis for gamma in the test method.
        **plot_kwargs
            Additional arguments passed to plot_caterpillar (e.g., title, point_color).
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted first.")

        # Compute SM‐CIs
        ci_results = self.calculate_confidence_intervals(
            group_ids=group_ids,
            level=level,
            option='SM',
            stdz=stdz,
            measure=measure,
            test_method=test_method,
            null=null
        )
        key = f"{stdz}_{measure}"
        if key not in ci_results:
            raise ValueError(f"Invalid combination: stdz='{stdz}', measure='{measure}'")
        df = ci_results[key]
        
        # Determine default refline:
        #  - ratios → 1
        #  - rates  → population_rate (stored in df.attrs by _compute_sm_intervals)
        refline_value = plot_kwargs.pop('refline_value', None)
        if refline_value is None:
            if measure == 'ratio':
                refline_value = 1.0
            elif measure == 'rate':
                refline_value = df.attrs.get('population_rate', None)

        # Merge in flags if desired
        if use_flags:
            test_df = self.test(
                providers=group_ids,
                level=level,
                test_method=test_method,
                null=null
            )
            df = df.merge(test_df[['flag']],
                        left_on='group_id',
                        right_index=True,
                        how='left')

        # Orientation
        orientation = plot_kwargs.pop('orientation', 'vertical')
        if orientation not in ('vertical','horizontal'):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")


        # Axis‐labels
        est_label = f"{stdz.capitalize()} {measure.capitalize()}"
        grp_label = "Provider"

        if orientation == 'vertical':
            xlabel = plot_kwargs.pop('xlabel', est_label)
            ylabel = plot_kwargs.pop('ylabel', grp_label)
        else:
            xlabel = plot_kwargs.pop('xlabel', grp_label)
            ylabel = plot_kwargs.pop('ylabel', est_label)

        # Plot
        plot_caterpillar(
            df=df,
            estimate_col=f"{stdz}_{measure}",
            ci_lower_col=f"ci_{measure}_lower",
            ci_upper_col=f"ci_{measure}_upper",
            group_col='group_id',
            flag_col='flag' if use_flags else None,
            orientation=orientation,
            refline_value=refline_value,
            xlab=xlabel,
            ylab=ylabel,
            plot_title=plot_kwargs.pop(
                'title',
                f"{stdz.capitalize()} Standardized {measure.capitalize()}"
            ),
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
        figure_size: Tuple[float, float] = (10, 6),
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

       
    def plot_residuals(self, *args, **kwargs) -> None:
        """
        Plot residuals versus fitted probabilities for logistic regression.

        Raises:
        -------
        NotImplementedError
            This method is not implemented as residuals are not computed by default.
        """
        raise NotImplementedError(
            "plot_residuals is not implemented for LogisticFixedEffectModel. "
            "Residual diagnostics are less standard in logistic regression and "
            "residuals are not computed by default in this model."
        )

    def plot_qq(self, *args, **kwargs) -> None:
        """
        Create a Q-Q plot of the deviance residuals for logistic regression.

        Raises:
        -------
        NotImplementedError
            This method is not implemented as residuals are not computed by default.
        """
        raise NotImplementedError(
            "plot_qq is not implemented for LogisticFixedEffectModel. "
            "Q-Q plots are less meaningful in logistic regression and "
            "residuals are not computed by default in this model."
        )