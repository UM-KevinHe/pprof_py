.. _base_model:

Base Model
=================================

This package provides a modular framework for statistical modeling. It defines a common interface for model fitting, prediction, hypothesis testing, summary reporting, and diagnostic plotting. The design is centered on an abstract base class (``BaseModel``) and several mixin classes to add specialized functionality. Although the package is model-agnostic, many of the concepts (e.g., model coefficients, residuals, information criteria) are familiar from linear or generalized linear models.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
---------
This framework is built to standardize the workflow for fitting statistical models. Its key features include:

- **Unified Interface:** A consistent API for fitting models, predicting outcomes, and evaluating model performance.
- **Diagnostic Tools:** Mixin classes provide methods for generating summaries, hypothesis tests, and a range of diagnostic plots.
- **Extensibility:** By subclassing ``BaseModel`` and mixins, you can create customized models and analyses while reusing common routines.

Each model stores important outputs such as the estimated coefficients, variance-covariance matrices, residuals, and information criteria like AIC and BIC. In addition, the package anticipates the need for group-specific analyses (e.g., fixed effects) by including support for group identifiers and indices.



Core Components
-----------------


The ``BaseModel`` class (from ``pprof_py.base_model``) is an abstract base class that defines the shared interface and state management for all statistical models.

.. py:class:: pprof_py.base_model.BaseModel

   .. attribute:: coefficients_
      :type: Optional[Dict[str, Any]]

      Stores the estimated model parameters (e.g., regression coefficients).
      *Example Formula (Linear Model):*

      .. math::

         \hat{\beta} = (X^T X)^{-1} X^T y

   .. attribute:: variances_
      :type: Optional[Dict[str, Any]]

      Contains the variance-covariance matrix of the coefficients.
      *Example Formula:*

      .. math::

         \hat{\text{Var}}(\hat{\beta}) = \sigma^2 (X^T X)^{-1} \quad \text{where} \quad \sigma^2 = \frac{1}{n - p}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2

   .. attribute:: fitted_
      :type: Optional[np.ndarray]

      The fitted values from the model, :math:`\hat{y}`.

   .. attribute:: residuals_
      :type: Optional[np.ndarray]

      The residuals computed as:

      .. math::

         r = y - \hat{y}

   .. attribute:: sigma_
      :type: Optional[float]

      The estimated standard deviation of the residuals.
      *Formula:*

      .. math::

         \hat{\sigma} = \sqrt{\frac{1}{n - p}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}

   .. attribute:: aic_
      :type: Optional[float]

      Akaike Information Criterion used to assess model fit.
      *Formula:*

      .. math::

         AIC = 2k - 2\ln(\hat{L})

      where :math:`k` is the number of parameters and :math:`\hat{L}` is the maximized likelihood.

   .. attribute:: bic_
      :type: Optional[float]

      Bayesian Information Criterion, which penalizes model complexity more heavily than AIC.
      *Formula:*

      .. math::

         BIC = \ln(n) \cdot k - 2\ln(\hat{L})

      where :math:`n` is the sample size.

   .. attribute:: groups_
      :type: Optional[np.ndarray]

      Array of unique group identifiers for models that incorporate fixed effects.

   Other attributes (such as ``group_indices_``, ``group_sizes_``, ``xbeta_``, and ``outcome_``) are provided for internal state management, especially when dealing with grouped data.

   **Abstract Methods**

   .. abstractmethod:: fit(*args: Any, **kwargs: Any) -> None

      Fit the model to the data. Each subclass must implement the specific fitting logic.

   .. abstractmethod:: predict(*args: Any, **kwargs: Any) -> np.ndarray

      Predict outcomes based on the fitted model.

   .. abstractmethod:: calculate_standardized_measures(*args, **kwargs) -> dict

      Compute standardized metrics. For example, when comparing groups, one might use:

      .. math::

         \delta = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s^2_1 + s^2_2}{2}}}

      where :math:`\bar{x}_i` and :math:`s_i^2` are the mean and variance of each group.

   .. method:: calculate_confidence_intervals(*args, **kwargs) -> dict
      :abstractmethod:

      Calculate confidence intervals for model parameters using:

      .. math::

         \hat{\beta} \pm z_{\alpha/2} \cdot SE(\hat{\beta})

      (or using a :math:`t`-distribution for smaller sample sizes).
      *(Note: In your provided `base_model.py`, this method is not explicitly marked with `@abstractmethod`. If it should be abstract, ensure the decorator is present in the Python code.)*


SummaryMixin
------------

The ``SummaryMixin`` provides an interface for generating a summary of the fitted model. Its main abstract method is:

.. py:method:: summary(level: float = 0.95, null: float = 0, alternative: str = "two.sided") -> Any
   :abstractmethod:

   Returns a detailed summary including:
   - Estimated coefficients,
   - Standard errors,
   - Confidence intervals (using the formula above), and
   - P-values for hypothesis testing.

The summary output is model-defined and can be formatted as a table or a structured report.

TestMixin
---------

The ``TestMixin`` supplies an interface for performing hypothesis tests on model parameters. Its abstract method:

.. py:method:: test(*args, **kwargs) -> Any
   :abstractmethod:

   Conducts statistical tests on the model estimates. For instance, a common test is the Z-test:

   .. math::

      z = \frac{\hat{\beta} - \beta_0}{SE(\hat{\beta})}

   where :math:`\beta_0` is the hypothesized value (often 0). P-values are computed based on the standard normal (or :math:`t`) distribution.

This mixin enables model developers to add custom tests as required.

PlotMixin
---------

The ``PlotMixin`` defines an abstract interface for a suite of diagnostic and summary plots. These plots help in visually assessing the model’s performance and underlying assumptions. The abstract methods include:

.. py:method:: plot_funnel(*args, **kwargs) -> None
   :abstractmethod:

   Generates a funnel plot to compare provider or group performance against expected control limits.

.. py:method:: plot_residuals(*args, **kwargs) -> None
   :abstractmethod:

   Plots residuals versus fitted values, which is essential for checking non-linearity and heteroscedasticity.

.. py:method:: plot_qq(*args, **kwargs) -> None
   :abstractmethod:

   Produces a Q–Q plot to assess if the residuals follow a normal distribution.

.. py:method:: plot_provider_effects(*args, **kwargs) -> None
   :abstractmethod:

   Visualizes fixed effects (or provider effects) along with their confidence intervals.

.. py:method:: plot_coefficient_forest(*args, **kwargs) -> None
   :abstractmethod:

   Creates a forest plot that displays covariate coefficients and their confidence intervals, facilitating comparisons across predictors.

Each plotting method should be implemented in a subclass using your preferred visualization library (e.g., Matplotlib, Seaborn).


Usage Example
-----------------

Below is a brief example demonstrating how a custom model might be implemented using these classes:

.. code-block:: python

   from pprof_py.base_model import BaseModel
   from pprof_py.mixins import SummaryMixin, TestMixin, PlotMixin
   import numpy as np


   class MyCustomModel(BaseModel, SummaryMixin, TestMixin, PlotMixin):
       def fit(self, X, y):
           # Implement fitting logic (e.g., using least squares)
           # Example: self.coefficients_ = {'beta': (X.T @ X)^(-1) @ X.T @ y}
           self.coefficients_ = {"beta": np.linalg.inv(X.T @ X) @ (X.T @ y)}
           self.fitted_ = X @ self.coefficients_["beta"]
           self.residuals_ = y - self.fitted_
           # Calculate sigma_, aic_, bic_ based on formulas above.
           # (For demonstration, these are placeholders.)
           self.sigma_ = np.std(self.residuals_, ddof=X.shape[1]) # ddof for unbiased estimator
           n = len(y)
           p = X.shape[1]
           log_likelihood_placeholder = -n/2 * np.log(2 * np.pi * self.sigma_**2) - np.sum(self.residuals_**2) / (2 * self.sigma_**2) # Approx for Gaussian
           self.aic_ = 2 * p - 2 * log_likelihood_placeholder
           self.bic_ = np.log(n) * p - 2 * log_likelihood_placeholder

       def predict(self, X):
           return X @ self.coefficients_["beta"]

       def calculate_standardized_measures(self, X, y):
           # Implement your standardized measure logic here.
           return {"std_diff": (np.mean(y) - np.mean(self.fitted_)) / np.std(y)}

       def calculate_confidence_intervals(self, level: float = 0.95):
           # Compute confidence intervals using the standard formula.
           # This is a placeholder implementation.
           from scipy.stats import norm
           z = norm.ppf(1 - (1 - level) / 2)
           # A more realistic SE for OLS: sqrt(diag(sigma^2 * (X'X)^-1))
           # For simplicity, using a placeholder SE.
           # This would typically come from self.variances_
           se_placeholder = np.std(self.residuals_) / np.sqrt(len(self.fitted_)) # Very rough placeholder
           beta = self.coefficients_["beta"]
           return {"lower": beta - z * se_placeholder, "upper": beta + z * se_placeholder}

       def summary(self, level: float = 0.95, null: float = 0, alternative: str = "two.sided"):
           # Generate and return a summary dictionary or formatted table.
           ci = self.calculate_confidence_intervals(level)
           summary_data = {
               "coefficients": self.coefficients_["beta"],
               "sigma": self.sigma_,
               "AIC": self.aic_,
               "BIC": self.bic_,
               "confidence_intervals_lower": ci["lower"],
               "confidence_intervals_upper": ci["upper"]
           }
           # This would typically be formatted into a nice string or pandas DataFrame
           return summary_data

       def test(self, null_hyp: float = 0.0): # Renamed from *args, **kwargs for clarity
           # Implement a hypothesis test for the coefficients.
           # For instance, a Z-test for each coefficient.
           from scipy.stats import norm
           test_results = {}
           # Placeholder SE, should come from self.variances_
           se_placeholder = np.std(self.residuals_) / np.sqrt(len(self.fitted_))
           for i, beta_val in enumerate(self.coefficients_["beta"]):
               z_stat = (beta_val - null_hyp) / se_placeholder
               p_val = 2 * (1 - norm.cdf(np.abs(z_stat))) # Two-sided test
               test_results[f"beta_{i}"] = {"z_stat": z_stat, "p_value": p_val}
           return test_results

       # Plotting methods would be implemented here using matplotlib or seaborn
       def plot_residuals(self, *args, **kwargs):
           import matplotlib.pyplot as plt
           plt.figure()
           plt.scatter(self.fitted_, self.residuals_)
           plt.xlabel("Fitted values")
           plt.ylabel("Residuals")
           plt.title("Residuals vs. Fitted")
           plt.axhline(0, color='red', linestyle='--')
           plt.show()
           # In a real scenario, you might return the fig/ax object or save the plot.

   # Example usage:
   if __name__ == "__main__":
       # Dummy data
       X = np.random.rand(100, 3)
       # Add an intercept column to X for a more standard OLS setup
       X_intercept = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
       true_betas = np.array([0.5, 1.5, -2.0, 0.5]) # intercept, beta1, beta2, beta3
       y = X_intercept @ true_betas + np.random.randn(100) * 0.5

       model = MyCustomModel()
       model.fit(X_intercept, y)
       predictions = model.predict(X_intercept)
       print("Model Summary:\n", model.summary())
       print("\nHypothesis Tests (against 0):\n", model.test(null_hyp=0.0))
       # model.plot_residuals() # Uncomment to show plot if running interactively

Extending the Framework
----------------------

To add new models or customize the behavior:
- **Subclass ``BaseModel``:** Provide your own implementation for ``fit``, ``predict``, and any additional calculations.
- **Implement Mixins:** If you require custom summary statistics, tests, or plots, extend the mixin classes and override their abstract methods.
- **Integrate New Formulas:** Adapt or extend the mathematical formulas based on the specific assumptions and characteristics of your model.
