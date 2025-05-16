from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.utils import check_array

from .data_prep import DataPrep  

class BaseModel(ABC):
    """Abstract Base Class for statistical models.

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
        self, X, y=None, groups=None, x_vars=None, y_var=None, group_var=None, 
        use_dataprep: bool = True, screen_providers: bool = False, 
        log_event_providers: bool = False, cutoff: int = 10, 
        threshold_cor: float = 0.9, threshold_vif: int = 10, **kwargs
    ) -> tuple:
        """Validate and convert input data to NumPy arrays for modeling.

        This method supports both pandas DataFrames and array-like inputs, validates them,
        and optionally applies data preparation using the DataPrep class. It ensures the 
        returned data is in NumPy array format, ready for use in modeling classes.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Design matrix (covariates) or complete dataset if DataFrame.
        y : array-like, optional
            Response variable. If None and X is a DataFrame, derived from `y_var`.
        groups : array-like, optional
            Group identifiers. If None and X is a DataFrame, derived from `group_var`.
        x_vars : list of str, optional
            Column names in X for predictors. Required if X is a DataFrame.
        y_var : str, optional
            Column name in X for the response variable.
        group_var : str, optional
            Column name in X for group identifiers. Required if X is a DataFrame.
        use_dataprep : bool, default=True
            Whether to apply data preparation using DataPrep.
        screen_providers : bool, default=False
            Whether to filter small providers (used if use_dataprep is True).
        log_event_providers : bool, default=False
            Whether to log stats for providers with no/all events (used if use_dataprep is True).
        cutoff : int, default=10
            Minimum number of records per provider for screening.
        threshold_cor : float, default=0.9
            Correlation threshold for multicollinearity checks.
        threshold_vif : int, default=10
            Variance Inflation Factor (VIF) threshold for multicollinearity checks.
        **kwargs : Additional keyword arguments
            Passed to DataPrep for further customization.

        Returns
        -------
        tuple
            (X, y, groups) as NumPy arrays.

        Raises
        ------
        ValueError
            If `x_vars` or `group_var` are missing when X is a DataFrame, or if input lengths 
            are inconsistent.
        """
        # Handle DataFrame inputs
        if isinstance(X, pd.DataFrame):
            if x_vars is None or group_var is None:
                raise ValueError("When providing a DataFrame, `x_vars` and `group_var` must be specified.")
            self.covariate_names_ = x_vars
            data_for_prep = X.copy()  # Use a copy of the DataFrame for DataPrep
            # Extract arrays for initial consistency checks
            X_array = data_for_prep[x_vars].to_numpy()
            y_array = data_for_prep[y_var].to_numpy() if y_var else np.zeros(X.shape[0])
            groups_array = data_for_prep[group_var].to_numpy()
        # Handle array-like inputs
        else:
            X_array = check_array(X, ensure_2d=True, dtype=np.float64)
            groups_array = check_array(groups, ensure_2d=False) if groups is not None else np.zeros(X_array.shape[0])
            y_array = check_array(y, ensure_2d=False, dtype=np.float64) if y is not None else np.zeros(X_array.shape[0])
            self.covariate_names_ = [f"X{i}" for i in range(X_array.shape[1])]
            # Create a temporary DataFrame for DataPrep
            data_for_prep = pd.DataFrame(X_array, columns=self.covariate_names_)
            data_for_prep['y'] = y_array
            data_for_prep['groups'] = groups_array

        # Check input dimensions
        if len(groups_array) != len(X_array):
            raise ValueError("X and groups must have the same number of samples.")
        if len(y_array) != 0 and len(y_array) != len(X_array):
            raise ValueError("y must have the same number of samples as X if provided.")

        # Apply DataPrep if enabled
        if use_dataprep:
            # Set column names for DataPrep based on input type
            if isinstance(X, pd.DataFrame):
                Y_char = y_var if y_var else 'y'
                X_char = x_vars
                prov_char = group_var
            else:
                Y_char = 'y'
                X_char = self.covariate_names_
                prov_char = 'groups'

            # Instantiate and run DataPrep
            dataprep = DataPrep(
                data=data_for_prep,
                Y_char=Y_char,
                X_char=X_char,
                prov_char=prov_char,
                cutoff=cutoff,
                check=True,  # Perform standard checks (e.g., missingness, variation)
                screen_providers=screen_providers,
                log_event_providers=log_event_providers,
                threshold_cor=threshold_cor,
                threshold_vif=threshold_vif,
                **kwargs
            )
            prepared_data = dataprep.data_prep()

            # Extract updated arrays from prepared data
            X_array = prepared_data[X_char].to_numpy()
            y_array = prepared_data[Y_char].to_numpy()
            groups_array = prepared_data[prov_char].to_numpy()

        return X_array, y_array, groups_array

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Fit the model to the provided data.

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
        """Generate predictions using the fitted model.

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
        """Calculate standardized measures for model diagnostics.

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