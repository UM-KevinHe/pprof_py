from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.utils import check_array

class BaseModel(ABC):
    """
    Abstract Base Class for statistical models.

    This class defines a common interface and provides shared state management 
    for derived model classes. It handles storage and management of model parameters 
    (such as coefficients and variances), fitted values, residuals, and goodness-of-fit 
    metrics (like AIC and BIC). Subclasses must implement model-specific logic for fitting,
    predicting, and calculating diagnostic measures.

    Attributes
    ----------
    coefficients_ : Optional[Dict[str, Any]]
        Dictionary storing model coefficients (parameter estimates).
    variances_ : Optional[Dict[str, Any]]
        Dictionary containing variance-covariance matrices of the coefficients.
    fitted_ : Optional[np.ndarray]
        Array of fitted values from the model.
    residuals_ : Optional[np.ndarray]
        Array of residuals computed from the fitted model.
    sigma_ : Optional[float]
        Estimated standard deviation of the residuals.
    aic_ : Optional[float]
        Akaike Information Criterion value, used as a measure of model fit.
    bic_ : Optional[float]
        Bayesian Information Criterion value, used as a measure of model fit.
    groups_ : Optional[np.ndarray]
        Array of unique group identifiers, typically used for fixed effects.
    group_indices_ : Optional[np.ndarray]
        Array indicating the indices corresponding to each group.
    group_sizes_ : Any
        Information on group sizes.
    xbeta_ : Any
        Linear predictor values (i.e., the product of the design matrix and coefficients).
    outcome_ : Any
        The observed outcomes (response variable values).

    Examples
    --------
    >>> from mypackage.models import BaseModel
    >>> class MyModel(BaseModel):
    ...     def fit(self, X, y, **kwargs):
    ...         # Implement fitting logic here
    ...         self.coefficients_ = {'intercept': 1.0, 'slope': 2.0}
    ...         self.fitted_ = X[:, 0] * self.coefficients_['slope'] + self.coefficients_['intercept']
    ...     def predict(self, X, **kwargs):
    ...         # Implement prediction logic here
    ...         return X[:, 0] * self.coefficients_['slope'] + self.coefficients_['intercept']
    ...     def calculate_standardized_measures(self, **kwargs):
    ...         # Implement standardized measure calculation
    ...         return {'std_diff': 0.0}
    ...     def calculate_confidence_intervals(self, **kwargs):
    ...         # Implement confidence interval calculation
    ...         return {'intercept': (0.5, 1.5), 'slope': (1.5, 2.5)}
    >>> model = MyModel()
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """
    def __init__(self) -> None:
        self.coefficients_: Optional[Dict[str, Any]] = None
        self.variances_: Optional[Dict[str, Any]] = None
        self.fitted_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.sigma_: Optional[float] = None
        self.aic_: Optional[float] = None
        self.bic_: Optional[float] = None
        self.groups_: Optional[np.ndarray] = None
        self.group_indices_: Optional[np.ndarray] = None
        self.group_sizes_ = None
        self.xbeta_ = None
        self.outcome_ = None  # Store the actual observed outcomes

    def _validate_and_convert_inputs(
        self, X, y=None, groups=None, x_vars=None, y_var=None, group_var=None
    ) -> tuple:
        """
        Validate and convert input data to numpy arrays.

        This helper method validates the input data and converts it to the appropriate numpy 
        arrays for further processing in model fitting and prediction. It supports both 
        array-like inputs and pandas DataFrame objects. When a DataFrame is provided, the
        columns corresponding to predictors, response, and group identifiers are extracted 
        based on the provided column names.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Design matrix (covariates) or complete dataset.
        y : array-like, optional
            Response variable. If None and X is a DataFrame, y will be derived from `y_var`.
        groups : array-like, optional
            Group identifiers corresponding to each sample. If None and X is a DataFrame, groups 
            will be derived from `group_var`.
        x_vars : list of str, optional
            Column names in X to be used as predictors. Required if X is a DataFrame.
        y_var : str, optional
            Column name in X to be used as the response variable. Defaults to zeros if not provided.
        group_var : str, optional
            Column name in X to be used as group identifiers. Required if X is a DataFrame.

        Returns
        -------
        tuple
            A tuple containing:
                - X (np.ndarray): Converted design matrix with predictors.
                - y (np.ndarray): Converted response variable array.
                - groups (np.ndarray): Array of group identifiers.

        Raises
        ------
        ValueError
            If `x_vars` or `group_var` are not specified when X is a DataFrame, or if the lengths 
            of groups, y, and X are inconsistent.
        """
        if isinstance(X, pd.DataFrame):
            if x_vars is None or group_var is None:
                raise ValueError(
                    "When providing a DataFrame, `x_vars` and `group_var` must be specified."
                )
            self.covariate_names_ = x_vars  # save the covariate names here!
            groups = X[group_var].to_numpy()
            if y_var is not None:
                y = X[y_var].to_numpy()
            else:
                y = np.zeros(X.shape[0])
            X = X[x_vars].to_numpy()
        else:
            X = check_array(X, ensure_2d=True, dtype=np.float64)
            groups = check_array(groups, ensure_2d=False)
            if y is not None:
                y = check_array(y, ensure_2d=False, dtype=np.float64)
            else:
                y = np.zeros(X.shape[0])
            # Optionally, if the array does not include column names, you can assign default names:
            self.covariate_names_ = [f"X{i}" for i in range(X.shape[1])]
        if len(groups) != len(X):
            raise ValueError("X and groups must have the same number of samples.")
        if len(y) != 0 and len(y) != len(X):
            raise ValueError("y must have the same number of samples as X if provided.")
        return X, y, groups

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> None:
        """
        Fit the model to the provided data.

        Subclasses must implement this method to define how the model is fitted to the input data.
        Typically, this involves estimating model parameters (e.g., coefficients), computing the 
        linear predictor, calculating residuals, and evaluating model fit using metrics such as 
        AIC and BIC.

        Parameters
        ----------
        *args : Any
            Positional arguments required for model fitting.
        **kwargs : Any
            Keyword arguments required for model fitting.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Generate predictions using the fitted model.

        Subclasses must implement this method to define how predictions are computed. The method
        should leverage the fitted parameters to produce predicted values based on the input 
        design matrix or features.

        Parameters
        ----------
        *args : Any
            Positional arguments that may include the design matrix for prediction.
        **kwargs : Any
            Keyword arguments that modify or provide additional prediction parameters.

        Returns
        -------
        np.ndarray
            An array of predicted values corresponding to the input data.
        """
        pass

    @abstractmethod
    def calculate_standardized_measures(self, *args, **kwargs) -> dict:
        """
        Calculate standardized measures for model diagnostics.

        Subclasses should implement this method to compute standardized differences or other 
        diagnostic measures. These measures can be used to assess model fit, detect outliers,
        or evaluate the influence of observations.

        Parameters
        ----------
        *args : Any
            Positional arguments for calculating standardized measures.
        **kwargs : Any
            Keyword arguments for calculating standardized measures.

        Returns
        -------
        dict
            A dictionary containing the calculated standardized measures.
        """
        pass