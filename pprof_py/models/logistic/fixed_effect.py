"""User-facing ``LogisticFixedEffectModel`` estimator.

Fits a logistic regression with provider-specific fixed intercepts
(gamma) via iterative Newton methods (Serbin or Ban algorithm),
and provides covariate-level inference, standardized provider-effect
measures, and diagnostic plots through mixin composition.
"""
import logging
from dataclasses import replace

import numpy as np

from ...base import ProviderModel
import pandas as pd
from ...utils.metrics import roc_auc_score
from typing import Optional, Union, List

from ...data.validation import validate_and_convert_inputs
from ...data.preparation import DataPrepOptions
from ...exceptions import NotFittedError
from ...algorithms.logistic.fixed_effect import SerbinAlgorithm, BanAlgorithm, AlgorithmOptions
from ...utils.numerical import sigmoid
from ...inference.logistic import FixedEffectInferenceMixin
from ...measures.logistic import FixedEffectMeasuresMixin
from ...plotting.logistic import FixedEffectPlottingMixin

logger = logging.getLogger(__name__)


class LogisticFixedEffectModel(
    FixedEffectInferenceMixin,
    FixedEffectMeasuresMixin,
    FixedEffectPlottingMixin,
    ProviderModel,
):
    """Logistic Regression Model with Fixed Effects.

    This class implements a logistic regression model with fixed effects for groups (e.g., providers),
    using custom optimization algorithms ('Serbin' or 'Ban'). It supports advanced data preparation
    via the DataPrep class, estimates coefficients for covariates and group effects, and provides
    methods for fitting, prediction, and diagnostics.

    Responsibilities are split across mixins so this class stays focused on
    configuration, input handling, fitting, and prediction:

    - `FixedEffectInferenceMixin` (`pprof_py.inference.logistic`): variance estimation and
      covariate (beta) hypothesis tests (Wald/LR/score), `summary()`.
    - `FixedEffectMeasuresMixin` (`pprof_py.measures.logistic`): standardized rates/ratios,
      provider-level confidence intervals, provider-effect (gamma) hypothesis testing, `test()`.
    - `FixedEffectPlottingMixin` (`pprof_py.plotting.logistic`): funnel/caterpillar/forest plots.

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
        """Initialize the LogisticFixedEffectModel with data preparation options.

        Parameters
        ----------
        (See class docstring for parameter details)
        """
        # Store data preparation defaults
        self.use_dataprep = use_dataprep
        self.dataprep_options = DataPrepOptions(
            screen_providers=screen_providers,
            log_event_providers=log_event_providers,
            cutoff=cutoff,
            threshold_cor=threshold_cor,
            threshold_vif=threshold_vif,
            binary_response=True,  # Enforce binary response for logistic models
        )
        
        # Validate and set algorithm
        if algorithm not in ['Serbin', 'Ban']:
            raise ValueError("Algorithm must be 'Serbin' or 'Ban'")
        self.algorithm_type = algorithm
        
        # Initialize attributes
        self.algorithm = None
        self.coefficients_ = None
        self.variances_ = None
        self.robust_variances_ = None
        self.fitted_ = None
        self.aic_ = None
        self.bic_ = None
        self.auc_ = None
        self.groups_ = None
        self.group_indices_ = None
        self.group_sizes_ = None
        self.xbeta_ = None
        self.outcome_ = None
        self.obs_ids_ = None
        self.N_ = None  # Binomial trials per observation (None = Bernoulli)
        self.covariate_names_ = None  # Store covariate names if provided

    def _check_is_fitted(self) -> None:
        """Raise `NotFittedError` if the model has not been fitted yet."""
        if self.coefficients_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call `fit` first."
            )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        x_vars: Optional[List[str]] = None,
        y_var: Optional[str] = None,
        group_var: Optional[str] = None,
        n_var: Optional[str] = None,
        obs_id_var: Optional[str] = None,
        use_dataprep: Optional[bool] = None,
        screen_providers: Optional[bool] = None,
        log_event_providers: Optional[bool] = None,
        cutoff: Optional[int] = None,
        threshold_cor: Optional[float] = None,
        threshold_vif: Optional[int] = None,
        max_iter: int = 10000,
        tol: float = 1e-8,
        bound: float = 10.0,
        backtrack: bool = True
    ) -> "LogisticFixedEffectModel":
        """Fit the logistic fixed effect model with enhanced data preparation and variance estimation.

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
        n_var : Optional[str], default=None
            Column name for the number of trials per observation (binomial N).
            If None, defaults to 1 for all observations (Bernoulli model).
            Use this when each row represents aggregated binomial data
            (Y successes out of N trials).
        obs_id_var : Optional[str], default=None
            Column name for observation-level identifiers (e.g., patient ID,
            member ID). Required for cluster-robust variance estimation. If
            provided, enables variance='robust' in test_standardized() and standardized_measure().
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

        # Merge fit-time overrides into the stored DataPrepOptions
        overrides = {
            k: v for k, v in {
                'screen_providers': screen_providers,
                'log_event_providers': log_event_providers,
                'cutoff': cutoff,
                'threshold_cor': threshold_cor,
                'threshold_vif': threshold_vif,
            }.items() if v is not None
        }
        options = (
            replace(self.dataprep_options, **overrides)
            if overrides
            else self.dataprep_options
        )

        # Validate and convert inputs
        validated = validate_and_convert_inputs(
            X, y, groups, x_vars, y_var, group_var,
            n_var=n_var,
            obs_id_var=obs_id_var,
            use_dataprep=use_dataprep,
            dataprep_options=options,
        )
        X, y, groups = validated.X, validated.y, validated.groups
        self.covariate_names_ = validated.covariate_names
        self.obs_ids_ = validated.obs_ids
        self.N_ = validated.N

        # Set binomial N: default to 1 (Bernoulli) if not provided
        if self.N_ is None:
            self.N_ = np.ones(len(y))

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
        Y_bar = np.sum(y) / np.sum(self.N_)
        gamma_prov = np.repeat(np.log(Y_bar / (1 - Y_bar)), len(n_prov))
        beta = np.zeros(X.shape[1])

        # Instantiate algorithm
        algorithm_options = AlgorithmOptions(
            backtrack=backtrack, max_iter=max_iter, bound=bound, tol=tol
        )
        if self.algorithm_type == 'Serbin':
            self.algorithm = SerbinAlgorithm(
                data, y_index, X, prov_index, n_prov, gamma_prov, beta,
                algorithm_options, N=self.N_
            )
        else:
            self.algorithm = BanAlgorithm(
                data, y_index, X, prov_index, n_prov, gamma_prov, beta,
                algorithm_options, N=self.N_
            )

        # Fit the model
        result = self.algorithm.fit()

        self.coefficients_ = {'beta': result['beta'], 'gamma': result['gamma']}

        # Compute fitted values
        self.xbeta_ = np.dot(X, self.coefficients_['beta'])
        gamma_obs = self.coefficients_['gamma'][self.group_indices_]
        linear_pred = self.xbeta_ + gamma_obs
        self.fitted_ = sigmoid(linear_pred)

        # Compute aic and bic
        neg2Loglkd = -2 * np.sum((gamma_obs + self.xbeta_) * y - self.N_ * np.logaddexp(0, linear_pred))
        n_params = len(self.coefficients_['beta']) + len(self.coefficients_['gamma'])
        self.aic_ = 2 * n_params + neg2Loglkd
        self.bic_ = n_params * np.log(len(y)) + neg2Loglkd

        # AUC only meaningful for binary outcomes (Bernoulli); skip for binomial counts
        if len(np.unique(y)) == 2:
            self.auc_ = roc_auc_score(y, self.fitted_)
        else:
            self.auc_ = None

        # Compute variances
        self.variances_ = self._estimate_variances()

        # Compute robust variances if observation IDs are available
        if self.obs_ids_ is not None:
            self.robust_variances_ = self._compute_robust_variances()
        else:
            self.robust_variances_ = None

        return self

    def add_providers(
        self,
        provider_ids,
        gamma,
        se_gamma = 0.01,
        group_sizes = None
    ) -> "LogisticFixedEffectModel":
        """Add external providers to model results.

        Extends the fitted model with providers that were excluded during
        fitting (e.g., zero-event or all-event providers identified from a
        separate indicator file). This replicates R's PPPW_Modelling behavior
        of appending extreme providers with fixed gamma and SE values after
        model fitting.

        The added providers' standardized measures (direct rate, etc.) are
        computed using the same patient-level xbeta as fitted providers:
            direct_rate_j = sum(sigmoid(gamma_j + xbeta_i)) / N

        Parameters
        ----------
        provider_ids : array-like
            Provider identifiers to add.
        gamma : array-like
            Fixed effect values for the added providers (e.g., 17 for
            all-event, -17 for zero-event).
        se_gamma : array-like or float, default=0.01
            Standard errors for the added providers.
        group_sizes : array-like, optional
            Number of patients per provider (for groupwise EN quartiles).
            If None, defaults to 0 for added providers.

        Returns
        -------
        LogisticFixedEffectModel
            Self, for method chaining.
        """
        self._check_is_fitted()

        provider_ids = np.asarray(provider_ids)
        gamma = np.asarray(gamma, dtype=float)
        if np.isscalar(se_gamma) or (hasattr(se_gamma, '__len__') and len(se_gamma) == 1):
            se_gamma = np.full(len(provider_ids), float(se_gamma) if np.isscalar(se_gamma) else float(se_gamma[0]))
        else:
            se_gamma = np.asarray(se_gamma, dtype=float)

        n_add = len(provider_ids)
        if n_add == 0:
            logger.info("No providers to add.")
            return self

        # Separate into providers that already exist (replace) vs new (append)
        existing_set = set(self.groups_.tolist())
        existing_mask = np.array([p in existing_set for p in provider_ids])
        new_mask = ~existing_mask

        # Replace existing providers' gamma and SE
        n_replaced = int(existing_mask.sum())
        if n_replaced > 0:
            for idx in np.where(existing_mask)[0]:
                j = np.where(self.groups_ == provider_ids[idx])[0][0]
                self.coefficients_["gamma"][j] = gamma[idx]
                if self.variances_ is not None and "gamma" in self.variances_:
                    self.variances_["gamma"][j] = se_gamma[idx] ** 2
                if self.robust_variances_ is not None and "gamma" in self.robust_variances_:
                    self.robust_variances_["gamma"][j] = se_gamma[idx] ** 2

        # Append new providers
        n_new = int(new_mask.sum())
        if n_new > 0:
            self.groups_ = np.concatenate([self.groups_, provider_ids[new_mask]])
            self.coefficients_["gamma"] = np.concatenate([
                self.coefficients_["gamma"].flatten(), gamma[new_mask]
            ])
            se_sq_new = se_gamma[new_mask] ** 2
            if self.variances_ is not None and "gamma" in self.variances_:
                self.variances_["gamma"] = np.concatenate([
                    self.variances_["gamma"].flatten(), se_sq_new
                ])
            if self.robust_variances_ is not None and "gamma" in self.robust_variances_:
                self.robust_variances_["gamma"] = np.concatenate([
                    self.robust_variances_["gamma"].flatten(), se_sq_new
                ])
            if group_sizes is not None:
                gs_new = np.asarray(group_sizes, dtype=int)[new_mask]
            else:
                gs_new = np.zeros(n_new, dtype=int)
            self.group_sizes_ = np.concatenate([self.group_sizes_, gs_new])

        # Sort all arrays by provider ID (matches R: gamma_summary[order(rownames), ])
        sort_idx = np.argsort(self.groups_)
        self.groups_ = self.groups_[sort_idx]
        self.coefficients_["gamma"] = self.coefficients_["gamma"].flatten()[sort_idx]
        if self.variances_ is not None and "gamma" in self.variances_:
            self.variances_["gamma"] = self.variances_["gamma"].flatten()[sort_idx]
        if self.robust_variances_ is not None and "gamma" in self.robust_variances_:
            self.robust_variances_["gamma"] = self.robust_variances_["gamma"].flatten()[sort_idx]
        self.group_sizes_ = self.group_sizes_[sort_idx]

        n_pos = int(np.sum(gamma > 0))
        n_neg = int(np.sum(gamma < 0))
        logger.info(f"Added {n_new} new + replaced {n_replaced} existing providers "
                    f"({n_pos} all-event, {n_neg} zero-event). "
                    f"Total providers: {len(self.groups_)}")
        return self

    def predict(self, X, groups=None, x_vars=None, group_var=None) -> np.ndarray:
        """Predict using the LogisticFixedEffect model.

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
        self._check_is_fitted()

        validated = validate_and_convert_inputs(X, None, groups, x_vars, None, group_var)
        X, groups = validated.X, validated.groups
        group_indices = np.searchsorted(self.groups_, groups)
        beta = self.coefficients_["beta"]
        gamma = self.coefficients_["gamma"]
        xbeta = X @ beta
        gamma_obs = gamma[group_indices].flatten()
        predictions = 1 / (1 + np.exp(-(xbeta + gamma_obs)))
        return predictions

    def score(self, X, y=None, groups=None, x_vars=None, y_var=None, group_var=None):
        """Compute the accuracy score for the model.

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
        self._check_is_fitted()

        # Validate and convert inputs
        validated = validate_and_convert_inputs(X, y, groups, x_vars, y_var, group_var)
        X, y, groups = validated.X, validated.y, validated.groups

        # Generate predictions (assuming predict returns probabilities)
        y_pred = self.predict(X, groups)

        # Convert probabilities to class labels (adjust threshold if needed)
        y_pred_class = (y_pred > 0.5).astype(int)

        # Compute accuracy
        return np.mean(y_pred_class == y)

    def get_fitted_params(self) -> dict:
        """Return fitted statistical parameters.

        Returns a dictionary of the estimated model quantities (coefficients,
        variances, information criteria, AUC). These are the *fitted* results,
        not the constructor configuration — see the class ``__init__`` for
        hyper-parameters.

        .. note::
           Renamed from ``get_params`` in 0.4.0 to avoid collision with
           ``ProviderModel.get_params()``, which returns
           constructor arguments.

        Returns
        -------
        dict
            Keys: ``coefficients``, ``variances``, ``aic``, ``bic``, ``auc``.
        """
        return {
            "coefficients": self.coefficients_,
            "variances": self.variances_,
            "aic": self.aic_,
            "bic": self.bic_,
            "auc": self.auc_
        }
