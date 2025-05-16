.. _linear_random_effect_model_stats:

Linear Random Effect Modeling
=============================

.. contents::
   :local:
   :depth: 2

1. Introduction
---------------

When evaluating provider performance using quantitative outcomes, linear mixed-effects models, often called linear random effect (RE) models, offer an alternative to fixed effect (FE) models :cite:`Wooldridge2010Econometric`. RE models are particularly useful when we assume providers are a sample from a larger population of providers and we wish to make inferences about this population or predict effects for individual providers, potentially "borrowing strength" across providers.

Unlike FE models that estimate a distinct parameter for each provider, RE models treat provider effects as random variables drawn from a common distribution. This approach can lead to more efficient estimates, especially when the number of observations per provider is small. However, a key assumption is that the random effects are uncorrelated with the covariates in the model. If this assumption is violated, estimates of covariate effects (:math:`\boldsymbol\beta`) can be biased :cite:`Wooldridge2010Econometric`.

This document details the statistical methodology for linear random effect models as implemented in the ``pprof_test.linear_random_effect.LinearRandomEffectModel`` class, which utilizes ``statsmodels.regression.mixed_linear_model.MixedLM`` for estimation. We cover:

  * The linear random effects model formulation (focusing on random intercepts).
  * Parameter estimation (fixed effects, variance components) via (Restricted) Maximum Likelihood.
  * Prediction of random effects (Best Linear Unbiased Predictors - BLUPs).
  * Calculation of standardized measures for performance comparison.
  * Hypothesis testing procedures for provider effects.
  * Construction of confidence intervals for provider effects and standardized measures.
  * Visualization tools for interpreting results.

2. Methods
----------

2.1. The Linear Random Effects Model (Random Intercept)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`Y_{ij}` be a quantitative outcome for subject :math:`j` (:math:`j = 1, \ldots, n_i`) within provider (group) :math:`i` (:math:`i = 1, \ldots, m`). Let :math:`\mathbf{X}_{ij}` be a :math:`p \times 1` vector of subject-level covariates. The linear random intercept model is:

.. math::

   Y_{ij} = \mathbf{X}_{ij}^\top\boldsymbol\beta + u_i + \epsilon_{ij}

where:

*   :math:`\boldsymbol\beta` is the :math:`p \times 1` vector of fixed regression coefficients for the covariates.
*   :math:`u_i` is the random effect for provider :math:`i`. It represents the deviation of provider :math:`i`'s intercept from the overall intercept (which is part of :math:`\mathbf{X}_{ij}^\top\boldsymbol\beta` if an intercept is included in :math:`\mathbf{X}_{ij}`). It is assumed that :math:`u_i \sim N(0, \sigma^2_u)`, where :math:`\sigma^2_u` is the variance of the provider effects.
*   :math:`\epsilon_{ij}` is the random error term for subject :math:`j` in provider :math:`i`. It is assumed that :math:`\epsilon_{ij} \sim N(0, \sigma^2_e)`, where :math:`\sigma^2_e` is the residual variance.
*   :math:`u_i` and :math:`\epsilon_{ij}` are assumed to be independent of each other and of :math:`\mathbf{X}_{ij}`.

The model aims to estimate the fixed effects :math:`\boldsymbol\beta`, the variance components :math:`\sigma^2_u` and :math:`\sigma^2_e`, and to predict the random effects :math:`u_i`.

2.2. Parameter Estimation and Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters (:math:`\boldsymbol\beta`, :math:`\sigma^2_u`, :math:`\sigma^2_e`) are typically estimated using Maximum Likelihood (ML) or Restricted Maximum Likelihood (REML). REML is often preferred for estimating variance components as it accounts for the degrees of freedom used in estimating fixed effects. The ``LinearRandomEffectModel`` uses ``statsmodels.MixedLM`` for this purpose, which allows specifying REML (default) or ML.

Once the variance components are estimated, the fixed effects :math:`\hat{\boldsymbol\beta}` are estimated. The random effects :math:`u_i` are not directly estimated as parameters but are predicted using Best Linear Unbiased Predictors (BLUPs), denoted as :math:`\hat{u}_i`. BLUPs are empirical Bayes estimates and exhibit shrinkage towards the overall mean (zero in this formulation), especially for providers with fewer observations or less precise estimates.

The BLUP for a random intercept :math:`u_i` is given by:

.. math::

   \hat{u}_i = \frac{n_i \sigma^2_u}{n_i \sigma^2_u + \sigma^2_e} (\bar{Y}_i - \bar{\mathbf{X}}_i^\top\hat{\boldsymbol\beta})

where :math:`\frac{n_i \sigma^2_u}{n_i \sigma^2_u + \sigma^2_e}` is the shrinkage factor.

The implementation stores:
*   Fixed effects: ``coefficients_['fixed_effect']`` (:math:`\hat{\boldsymbol\beta}`)
*   Random effects (BLUPs): ``coefficients_['random_effect']`` (:math:`\hat{u}_i`)
*   Variance-covariance of fixed effects: ``variances_['fe_var_cov']`` (:math:`\widehat{\text{Var}}(\hat{\boldsymbol\beta})`)
*   Variance of random effects: ``variances_['re_var']`` (:math:`\hat{\sigma}^2_u`)
*   Residual standard deviation: ``sigma_`` (:math:`\hat{\sigma}_e`)

2.3. Standardized Measures for Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For linear random effects models, standardized measures quantify how much a provider's total or average outcome differs from what would be expected under a baseline scenario, after adjusting for case mix. These measures are calculated either by comparing observed outcomes to expected outcomes under a baseline random effect (indirect standardization), or by comparing expected outcomes for the entire population under each provider's random effect to those under the baseline (direct standardization).

Let :math:`\hat{\boldsymbol{\beta}}` denote the estimated fixed effects, and :math:`\hat{\alpha}_i` the estimated random effect for provider :math:`i`. Define a reference or baseline random effect :math:`\alpha_0` (e.g., the median or mean of :math:`\hat{\alpha}_i`, as specified by the ``null`` parameter in ``LinearRandomEffectModel.calculate_standardized_measures``).

**2.3.1. Indirect Standardization**

Indirect standardization compares the observed total outcome for a provider to the expected total outcome if that provider had the baseline random effect :math:`\alpha_0`, given its specific patient mix.

*   **Observed Total Outcome for Provider i** (:math:`O_i`):
    The sum of fitted values (including both fixed and random effects) for all :math:`n_i` subjects in provider :math:`i`. This corresponds to the ``observed`` column in the output DataFrame for indirect standardization.

    .. math::

       O_i = \sum_{j=1}^{n_i} \left( \mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \hat{\alpha}_i \right)

*   **Expected Total Outcome for Provider i under Baseline** (:math:`E_i(\alpha_0)`):
    The sum of expected outcomes for provider :math:`i`'s :math:`n_i` subjects, if the provider effect was the baseline :math:`\alpha_0`, adjusted for their specific covariates. This corresponds to the ``expected`` column in the output DataFrame for indirect standardization.

    .. math::

       E_i(\alpha_0) = \sum_{j=1}^{n_i} \left( \mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \alpha_0 \right)

*   **Indirect Standardized Difference for Provider i** (:math:`\text{ISDiff}_i`):
    The average difference between observed and expected outcomes for provider :math:`i`. This is calculated as the total observed outcome minus the total expected outcome, divided by the number of subjects in the provider (:math:`n_i`). This corresponds to the ``indirect_difference`` column in the output DataFrame.

    .. math::

       \text{ISDiff}_i = \frac{O_i - E_i(\alpha_0)}{n_i}

    This difference can also be expressed as the difference between the observed mean and the expected mean for provider :math:`i` under the baseline effect:

    .. math::

       \text{ISDiff}_i = \left( \bar{Y}_i^{\text{fitted}} - (\bar{\mathbf{X}}_i^\top\hat{\boldsymbol{\beta}} + \alpha_0) \right)

    where :math:`\bar{Y}_i^{\text{fitted}}` is the mean fitted value for provider :math:`i`, and :math:`\bar{\mathbf{X}}_i` is the mean covariate vector for provider :math:`i`.

**2.3.2. Direct Standardization**

Direct standardization compares the expected total outcome if the *entire population* experienced provider :math:`k`'s random effect (:math:`\hat{\alpha}_k`) to the expected total outcome if the entire population experienced the baseline effect (:math:`\alpha_0`).

*   **Expected Total Outcome under Provider k's Effect** (:math:`E^{(k)}`):
    The total expected outcome for the entire population if all subjects experienced provider :math:`k`'s random effect (:math:`\hat{\alpha}_k`), adjusted for their specific covariates.

    .. math::

       E^{(k)} = \sum_{i=1}^m \sum_{j=1}^{n_i} \left( \mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \hat{\alpha}_k \right)

*   **Expected Total Outcome under Baseline Effect** (:math:`E^{(0)}`):
    The total expected outcome for the entire population if all subjects experienced the baseline effect (:math:`\alpha_0`), adjusted for their specific covariates.

    .. math::

       E^{(0)} = \sum_{i=1}^m \sum_{j=1}^{n_i} \left( \mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \alpha_0 \right)

*   **Direct Standardized Difference for Provider k** (:math:`\text{DSDiff}_k`):
    The average difference between the total expected outcomes under provider :math:`k`'s effect and the baseline effect, divided by the total sample size (:math:`N = \sum_{i=1}^m n_i`). This corresponds to the ``direct_difference`` column in the output DataFrame.

    .. math::

       \text{DSDiff}_k = \frac{E^{(k)} - E^{(0)}}{N}

    This difference can also be expressed as the difference between the expected mean outcome under provider :math:`k`'s effect and the expected mean outcome under the baseline effect:

    .. math::

       \text{DSDiff}_k = \hat{\alpha}_k - \alpha_0

Therefore, for linear random effects models, both indirect and direct standardized differences ultimately simplify to :math:`\hat{\alpha}_i - \alpha_0` when expressed on a per-subject basis. However, the calculations differ in how they aggregate observed and expected outcomes (by group size for indirect, and by total sample size for direct). The implementation (``LinearRandomEffectModel.calculate_standardized_measures``) calculates and returns these differences along with the observed and expected totals for each method.

.. 2.3. Standardized Measures for Performance Comparison
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Standardized measures quantify how a provider's adjusted performance (:math:`\hat{u}_i`) compares to a baseline or reference level (:math:`u_0`). The baseline :math:`u_0` can be the median or mean of :math:`\hat{u}_i`, or a specified value (e.g., 0), as defined by the ``null`` parameter in relevant methods.

.. Let :math:`\hat{\boldsymbol{\beta}}` be the estimated fixed effects and :math:`\hat{u}_i` be the predicted random effect (BLUP) for provider :math:`i`.

.. **2.3.1. Indirect Standardization**

.. The "indirect standardized difference" for provider :math:`i` is calculated as :math:`\hat{u}_i - u_0`.
.. In the implementation (``calculate_standardized_measures``), the term `indirect_difference` is first computed as :math:`\hat{u}_i` (derived from :math:`(\sum (Y_{ij} - \mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}})) / n_i` after accounting for shrinkage, effectively yielding the BLUP). This is then implicitly compared against :math:`u_0` when interpreting or using in tests/CIs relative to `null`.

.. **2.3.2. Direct Standardization**

.. Similarly, the "direct standardized difference" for provider :math:`k` is calculated as :math:`\hat{u}_k - u_0`.
.. The implementation calculates a `direct_difference` term which also effectively represents :math:`\hat{u}_k` relative to a population average random effect, before comparison with the specific :math:`u_0`.

.. Therefore, for the linear random intercept model, both indirect and direct standardized *differences* essentially represent the predicted provider-specific random effect (BLUP) relative to a chosen baseline random effect: :math:`\hat{u}_i - u_0`.

.. The method ``calculate_standardized_measures`` returns these differences along with observed and expected sum components.

2.4. Hypothesis Testing for Provider Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We test the null hypothesis :math:`H_0: u_i = u_0` against an alternative :math:`H_1`. The test is based on the predicted random effects (BLUPs) :math:`\hat{u}_i` and their standard errors. The test statistic is a Z-score:

.. math::

   Z_i = \frac{\hat{u}_i - u_0}{\widehat{\text{se}}(\hat{u}_i)}

The standard error of the BLUP, :math:`\widehat{\text{se}}(\hat{u}_i)`, is derived from the posterior variance of :math:`u_i` given the data:

.. math::

   \widehat{\text{se}}(\hat{u}_i) = \sqrt{\frac{\hat{\sigma}^2_u}{\hat{\sigma}^2_u + \hat{\sigma}^2_e/n_i} \frac{\hat{\sigma}^2_e}{n_i}}

Under :math:`H_0`, :math:`Z_i` is assumed to follow a standard normal distribution. P-values are calculated based on this distribution according to the specified ``alternative`` ('two_sided', 'less', 'greater') in the ``test`` method.

2.5. Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^

Confidence intervals are constructed for the fixed effects :math:`\boldsymbol\beta` (via ``summary`` method, using t-distribution) and for the provider random effects :math:`u_i` or standardized differences :math:`u_i - u_0` (via ``calculate_confidence_intervals`` method, using normal approximation for BLUPs).

*   **For Fixed Effects** :math:`\beta_k` (see ``summary``):
    Uses t-distribution with degrees of freedom :math:`N - p - m` (total observations - num fixed effects - num groups).

    .. math::

       \hat{\beta}_k \pm t_{1-\alpha/2, df} \times \widehat{\text{se}}(\hat{\beta}_k)

*   **For Random Effects** :math:`u_i` (BLUPs, option ``'alpha'`` in ``calculate_confidence_intervals``):
    Based on the normal approximation for BLUPs.

    .. math::

       \hat{u}_i \pm z_{1-\alpha/2} \times \widehat{\text{se}}(\hat{u}_i)

    This is typically a two-sided interval.
*   **For Standardized Differences** :math:`\hat{u}_i - u_0` (option ``'SM'`` in ``calculate_confidence_intervals``):
    The confidence interval for :math:`\hat{u}_i` is shifted by :math:`-u_0`.

    .. math::

       (\hat{u}_i - u_0) \pm z_{1-\alpha/2} \times \widehat{\text{se}}(\hat{u}_i)

    The implementation handles one-sided and two-sided alternatives.

2.6. Visualization
^^^^^^^^^^^^^^^^^^

The ``LinearRandomEffectModel`` class provides several plotting methods:

*   **Caterpillar Plot for Provider Effects** (``plot_provider_effects``): Displays BLUPs :math:`\hat{u}_i` with their confidence intervals.
*   **Caterpillar Plot for Standardized Measures** (``plot_standardized_measures``): Displays standardized differences :math:`\hat{u}_i - u_0` with confidence intervals.
*   **Funnel Plot** (``plot_funnel``): Plots standardized differences :math:`\hat{u}_i - u_0` against group size :math:`n_i`. Control limits are typically based on the overall residual standard deviation :math:`\hat{\sigma}_e`, e.g., :math:`target \pm z_{1-\alpha/2} \times \frac{\hat{\sigma}_e}{\sqrt{n_i}}`.
*   **Coefficient Forest Plot** (``plot_coefficient_forest``):** Displays estimates and confidence intervals for fixed effect coefficients :math:`\hat{\boldsymbol{\beta}}`.
*   **Residual Plots** (``plot_residuals``): Standard residuals vs. fitted values plot.
*   **Q-Q Plot** (``plot_qq``):** Q-Q plot of residuals against a normal distribution to check normality assumption.

3. Implementation and Usage
---------------------------

The ``LinearRandomEffectModel`` class in ``pprof_test.linear_random_effect`` implements these methods.

3.1. Initialization and Fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import pandas as pd
   from pprof_test.linear_random_effect import LinearRandomEffectModel

   # Example data generation
   np.random.seed(0)
   n_groups = 20
   n_obs_per_group = 50
   n_total_samples = n_groups * n_obs_per_group
   
   data_df = pd.DataFrame({
       'Covariate1': np.random.rand(n_total_samples),
       'Covariate2': np.random.randn(n_total_samples),
       'ProviderID': np.repeat(np.arange(n_groups), n_obs_per_group)
   })
   
   # True parameters
   beta_true = np.array([1.0, -0.5]) # Coefficients for Covariate1, Covariate2
   sigma_u_true = 0.5 # SD of random intercepts
   sigma_e_true = 1.0 # SD of residual error
   
   # Generate random intercepts
   u_true = np.random.normal(0, sigma_u_true, n_groups)
   data_df['u_i'] = data_df['ProviderID'].map(lambda x: u_true[x])
   
   # Generate outcome
   data_df['ContinuousY'] = (1.5 + # Intercept
                             data_df['Covariate1'] * beta_true[0] +
                             data_df['Covariate2'] * beta_true[1] +
                             data_df['u_i'] +
                             np.random.normal(0, sigma_e_true, n_total_samples))

   # Initialize and fit the model
   lre_model = LinearRandomEffectModel()
   lre_model.fit(
       X=data_df,
       y_var='ContinuousY',
       x_vars=['Covariate1', 'Covariate2'], # Intercept is added by default by statsmodels
       group_var='ProviderID',
       use_reml=True
   )
   # Output: Model fitting complete.

3.2. Accessing Results
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Fixed Effects Coefficients
   fixed_effects = lre_model.coefficients_['fixed_effect']
   print("Estimated Fixed Effects (Betas):")
   print(fixed_effects)

   # Predicted Random Effects (BLUPs)
   random_effects_blups = lre_model.coefficients_['random_effect']
   print("\nPredicted Random Effects (BLUPs, first 5):")
   print(random_effects_blups.head())

   # Variance Components
   fe_var_cov = lre_model.variances_['fe_var_cov']
   re_var = lre_model.variances_['re_var'] # Variance of random effects (sigma_u^2)
   sigma_e = lre_model.sigma_ # Residual standard deviation (sigma_e)
   print(f"\nEstimated Variance of Random Effects (sigma_u^2): {re_var.iloc[0,0]:.3f}")
   print(f"Estimated Residual Standard Deviation (sigma_e): {sigma_e:.3f}")

   # Fit statistics
   print(f"\nAIC: {lre_model.aic_:.2f}")
   print(f"BIC: {lre_model.bic_:.2f}")

   # Summary for fixed effects
   # fe_summary = lre_model.summary()
   # print("\n--- Fixed Effects Summary ---")
   # print(fe_summary)

3.3. Prediction
^^^^^^^^^^^^^^^
Predictions are typically based on fixed effects only, or can include specific random effects if known/estimated for the prediction set. The current ``predict`` method uses fixed effects.

.. code-block:: python

   # Predict outcomes using only fixed effects
   # Create a sample new data DataFrame (ensure it has the x_vars)
   new_data_df = pd.DataFrame({
       'Covariate1': np.random.rand(5),
       'Covariate2': np.random.randn(5)
   })
   
   predictions_fe_only = lre_model.predict(
       X=new_data_df,
       x_vars=['Covariate1', 'Covariate2']
   )
   print(f"\nFirst 5 predictions (fixed effects only): {predictions_fe_only}")

3.4. Standardized Measures Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Calculate Indirect Standardized Difference vs median random effect
   sm_results_lre = lre_model.calculate_standardized_measures(
       stdz='indirect', # Can be 'direct' or ['indirect', 'direct']
       null='median'    # Baseline for random effects: 'median', 'mean', or a float
   )
   print("\n--- Linear RE Indirect Measures (vs Median Random Effect) ---")
   if 'indirect' in sm_results_lre:
       print(sm_results_lre['indirect'].head())

3.5. Hypothesis Testing for Provider Effects (``test``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Test provider random effects (:math:`u_i`) against a null value.

.. code-block:: python

   # Test providers vs median random effect
   test_results_lre = lre_model.test(
       null='median', # Null hypothesis for u_i
       level=0.95,
       alternative='two_sided'
   )
   print("\n--- Linear RE Provider Test (vs Median Random Effect) ---")
   print(test_results_lre.head())

3.6. Confidence Interval Calculation (``calculate_confidence_intervals``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Compute CIs for provider random effects (:math:`u_i`) or standardized differences.

.. code-block:: python

   # Get 95% CIs for random effects (alpha option)
   alpha_cis_lre_results = lre_model.calculate_confidence_intervals(
       option='alpha', # For random effects u_i
       level=0.95,
       alternative='two_sided' # 'alpha' option only supports two-sided
   )
   print("\n--- Linear RE Random Effect (u_i) CIs ---")
   if 'alpha_ci' in alpha_cis_lre_results:
       print(alpha_cis_lre_results['alpha_ci'].head())

   # Get 95% CIs for the Indirect Standardized Difference
   isd_cis_lre_results = lre_model.calculate_confidence_intervals(
       option='SM', # For standardized measures
       stdz='indirect',
       level=0.95,
       null='median', # Baseline for u_i
       alternative='two_sided'
   )
   print("\n--- Linear RE Indirect Difference CIs (vs Median Random Effect) ---")
   if 'indirect_ci' in isd_cis_lre_results:
       print(isd_cis_lre_results['indirect_ci'].head())

3.7. Visualization
^^^^^^^^^^^^^^^^^^
Use plotting methods from the ``LinearRandomEffectModel`` instance. (Examples assume plots are shown interactively or saved).

.. code-block:: python

   # Example: Plot provider random effects (alpha/u_i)
   # lre_model.plot_provider_effects(null='median', level=0.95)

   # Example: Plot standardized differences
   # lre_model.plot_standardized_measures(stdz='indirect', null='median', level=0.95)

   # Example: Funnel plot
   # lre_model.plot_funnel(stdz='indirect', null='median', target=0.0, alpha=0.05)
   
   # Example: Forest plot for fixed covariate effects (beta)
   # lre_model.plot_coefficient_forest()

   # Example: Residual plot
   # lre_model.plot_residuals()

   # Example: Q-Q plot of residuals
   # lre_model.plot_qq()

4. Discussion
-------------

Linear random effect models provide a powerful framework for analyzing clustered data, such as patient outcomes within healthcare providers. They allow for the estimation of overall covariate effects while accounting for provider-specific variability. The prediction of random effects (BLUPs) incorporates shrinkage, which can be beneficial for ranking or comparing providers, especially those with small sample sizes.

**Advantages:**

*   **Efficiency:** Can be more efficient than FE models if the random effects assumption holds, particularly with many groups or few observations per group.
*   **Borrowing Strength:** BLUPs "borrow strength" from the overall data, leading to more stable estimates for individual group effects.
*   **Generalizability:** Allows inferences about the population of groups from which the sample is drawn.
*   **Flexibility:** Can model more complex variance structures (though the current implementation focuses on random intercepts).

**Limitations and Assumptions:**

*   **Random Effects Assumption:** Assumes random effects are drawn from a specific distribution (typically normal) and are uncorrelated with covariates. Violation of the latter can lead to biased :math:`\boldsymbol\beta` estimates.
*   **Distributional Assumptions:** Relies on normality assumptions for errors and random effects for exact inference, though estimates can be robust.
*   **Complexity:** Conceptually and computationally more complex than simple OLS or FE models.

5. Conclusion
-------------
The ``LinearRandomEffectModel`` class, leveraging ``statsmodels.MixedLM``, offers a comprehensive tool for provider profiling using linear mixed-effects models. It provides estimation of fixed effects and variance components, prediction of provider-specific random effects (BLUPs), and various methods for inference, standardization, and visualization. This approach is valuable when it is reasonable to assume providers are a sample from a population and when interest lies in both overall effects and provider-specific deviations.

6. References
-------------

.. bibliography:: references.bib
   :list: enumerate
   :filter: docname in docnames