"""User-facing ``LinearFixedEffectModel`` estimator.

Fits a linear regression with provider-specific fixed intercepts
(gamma) via closed-form weighted least squares with group demeaning,
and provides covariate-level inference, provider-effect measures,
and diagnostic plots through mixin composition.
"""
import numpy as np

from ...base import ProviderModel
from ...data.validation import validate_and_convert_inputs
from ...exceptions import NotFittedError
from ...algorithms.linear.fixed_effect import (
    preprocess_groups,
    perform_weighted_least_squares,
    calculate_residuals,
    compute_model_statistics,
)
from ...inference.linear import LinearFixedEffectInferenceMixin
from ...measures.linear import LinearFixedEffectMeasuresMixin
from ...plotting.linear import LinearFixedEffectPlottingMixin


class LinearFixedEffectModel(
    LinearFixedEffectInferenceMixin,
    LinearFixedEffectMeasuresMixin,
    LinearFixedEffectPlottingMixin,
    ProviderModel,
):
    """Linear Fixed Effect model.

    The Linear Fixed Effect model is a linear regression model that includes fixed effects.
    The model is fitted using the weighted least squares method.

    Responsibilities are split across mixins so this class stays focused on
    configuration, input handling, fitting, and prediction:

    - `LinearFixedEffectInferenceMixin` (`pprof_py.inference.linear`): sigma/variance estimation,
      `summary()`; provider-effect tests, `test()`; confidence intervals.
    - `LinearFixedEffectMeasuresMixin` (`pprof_py.measures.linear`): standardized differences,
      `calculate_standardized_measures()`.
    - `LinearFixedEffectPlottingMixin` (`pprof_py.plotting.linear`): funnel/caterpillar/forest/
      residual/Q-Q plots.

    Parameters
    ----------
    - gamma_var_option: str, default="complete"
        Option for variance calculation. Must be "complete" or "simplified".

    Examples:
    ---------
    >>> from pprof_py import LinearFixedEffectModel
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'y': [1, 2, 3, 4, 5, 6],
    ...     'x1': [1, 0, 1, 0, 1, 0],
    ...     'x2': [0, 1, 0, 1, 0, 1],
    ...     'group': [1, 1, 2, 2, 3, 3]
    ... })
    >>> model = LinearFixedEffectModel()
    >>> model.fit(data, x_vars=['x1', 'x2'], y_var='y', provider_var='group')
    >>> predictions = model.predict(data[['x1', 'x2']], groups=data['group'])
    >>> print(predictions)
    """

    def __init__(self, gamma_var_option: str = "complete") -> None:
        """Linear fixed-effect provider model."""
        if gamma_var_option not in {"complete", "simplified"}:
            raise ValueError("'gamma_var_option' must be 'complete' or 'simplified'.")
        self.gamma_var_option = gamma_var_option

        # Fitted attributes (previously inherited from BaseModel)
        self.coefficients_ = None
        self.variances_ = None
        self.fitted_ = None
        self.residuals_ = None
        self.sigma_ = None
        self.aic_ = None
        self.bic_ = None
        self.provider_ids_ = None
        self.provider_indices_ = None
        self.provider_sizes_ = None
        self.xbeta_ = None
        self.outcome_ = None
        self.covariate_names_ = None

    
    def _check_is_fitted(self) -> None:
        """Raise `NotFittedError` if the model has not been fitted yet."""
        if self.coefficients_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call `fit` first."
            )

    def fit(self, X, y=None, provider_id=None, x_vars=None, y_var=None, provider_var=None) -> "LinearFixedEffectModel":
        """Fit the LinearFixedEffect model.

        Parameters
        ----------
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
        - provider_var: str, optional
            Column name in X to be used as group identifiers.

        Returns:
        -------
        - self: LinearFixedEffectModel
            The fitted model instance.
        """
        validated = validate_and_convert_inputs(X, y, provider_id, x_vars, y_var, provider_var)
        X, y, provider_id = validated.X, validated.y, validated.provider_id
        self.covariate_names_ = validated.covariate_names

        self.outcome_ = y

        self.provider_ids_, self.provider_indices_ = np.unique(provider_id, return_inverse=True)
        self.provider_sizes_ = np.bincount(self.provider_indices_)
        n_groups = len(self.provider_ids_)
        n_samples, n_features = X.shape

        # Group preprocessing: Block diagonal matrix, group means
        Q, y_means, X_means = preprocess_groups(X, y, self.provider_indices_, n_groups)

        # Weighted least squares
        beta = perform_weighted_least_squares(Q, X, y)

        self.xbeta_ = X @ beta  # Storing the linear predictor

        # Calculate fixed effects
        gamma = y_means.reshape(-1, 1) - X_means @ beta

        # Store results
        self.coefficients_ = {"beta": beta, "gamma": gamma}
        self.fitted_, self.residuals_ = calculate_residuals(self.xbeta_, gamma, y, self.provider_indices_)

        self.sigma_ = self._estimate_sigma(self.residuals_, n_samples, n_groups, n_features)

        # Compute variances and statistics
        self.variances_ = self._estimate_variances(Q, X, X_means, beta, self.provider_sizes_)

        self.aic_, self.bic_ = compute_model_statistics(self.residuals_, n_samples, n_groups, n_features)

        return self

    def predict(self, X, provider_id=None, x_vars=None, provider_var=None) -> np.ndarray:
        """Predict using the LinearFixedEffect model.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Design matrix (covariates) or complete dataset.
        provider_id : array-like or None
            Group identifiers or None if `provider_var` is specified in a DataFrame.
        x_vars : list of str, optional
            Column names in X to be used as predictors, required if X is a DataFrame.
        provider_var : str, optional
            Column name in X to be used as group identifiers, required if X is a DataFrame.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        self._check_is_fitted()

        # Validate and convert inputs
        validated = validate_and_convert_inputs(X, None, provider_id, x_vars, None, provider_var)
        X, provider_id = validated.X, validated.provider_id

        # Align groups with fitted model's groups — validate membership
        unseen = np.setdiff1d(provider_id, self.provider_ids_)
        if unseen.size > 0:
            raise ValueError(
                f"predict() received group ids not seen during fit: "
                f"{unseen.tolist()}.  LinearFixedEffectModel cannot "
                f"extrapolate to unseen providers."
            )
        # np.searchsorted is safe now — every id is in self.provider_ids_
        group_indices = np.searchsorted(self.provider_ids_, provider_id)

        # Retrieve regression coefficients
        beta = self.coefficients_["beta"]
        gamma = self.coefficients_["gamma"]

        # Calculate predictions
        xbeta = X @ beta
        gamma_obs = gamma[group_indices].flatten()
        predictions = xbeta.flatten() + gamma_obs
        
        return predictions

    def score(self, X, y, provider_id) -> float:
        """Compute the R^2 score for the model.

        Parameters
        ----------
        - X: array-like, shape (n_samples, n_features)
            Design matrix (covariates).
        - y: array-like, shape (n_samples,)
            True target values.
        - groups: array-like, shape (n_samples,)
            Group identifiers for the fixed effects.

        Returns:
        -------
        - r2: float
            R^2 score of the model.
        """
        y_pred = self.predict(X, provider_id)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        return 1 - (ss_residual / ss_total)

    def get_fitted_params(self) -> dict:
        """Return fitted statistical parameters.

        Returns a dictionary of the estimated model quantities (coefficients,
        variances, residual standard error, information criteria). These are
        the *fitted* results, not the constructor configuration — see the
        class ``__init__`` for hyper-parameters.

        .. note::
           Renamed from ``get_params`` in 0.4.0 to avoid collision with
           ``ProviderModel.get_params()``, which returns
           constructor arguments.

        Returns
        -------
        dict
            Keys: ``coefficients``, ``variances``, ``sigma``, ``aic``, ``bic``.
        """
        return {
            "coefficients": self.coefficients_,
            "variances": self.variances_,
            "sigma": self.sigma_,
            "aic": self.aic_,
            "bic": self.bic_,
        }
