.. _logistic_random_effect_model_stats:

Logistic Random Effect Modeling
================================

.. contents::
   :local:
   :depth: 2

1. Introduction
---------------

When analyzing binary patient outcomes (e.g., mortality, readmission) clustered within healthcare providers, logistic random effect models, also known as Generalized Linear Mixed Models (GLMMs) with a logit link, are a common approach :cite:`Bates2015lme4,McCulloch2001GLMM,Stroup2013GLMM`. These models account for the correlation of outcomes within providers by incorporating provider-specific random effects, typically random intercepts. This allows for provider comparisons while adjusting for patient-level covariates.

The ``pprof_test.logistic_random_effect.LogisticRandomEffectModel`` class implements such a model by leveraging the ``pymer4`` library :cite:`pymer4doc`, which serves as a Python interface to R's powerful ``lme4`` package. This approach combines the statistical robustness of ``lme4`` with the convenience of a Python environment.

This document outlines the statistical methodology underpinning the ``LogisticRandomEffectModel``, covering:

  * The logistic random intercept model formulation.
  * Parameter estimation via ``pymer4`` (interfacing ``lme4``), including fixed effects and variance components, and prediction of random effects (BLUPs).
  * Calculation of standardized measures (e.g., Standardized Mortality/Morbidity Ratios - SMRs, Standardized Rates) for performance comparison.
  * Hypothesis testing procedures for provider random effects.
  * Construction of confidence intervals for fixed effects and random effects.
  * Visualization tools for interpreting model results.

2. Methods
----------

2.1. The Logistic Random Intercept Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`Y_{ij}` be a binary outcome (0 or 1) for subject :math:`j` (:math:`j = 1, \ldots, n_i`) within provider (group) :math:`i` (:math:`i = 1, \ldots, m`). Let :math:`\mathbf{X}_{ij}` be a :math:`p \times 1` vector of subject-level covariates. The probability of success :math:`P(Y_{ij}=1) = p_{ij}` is modeled using a logit link function:

.. math::

   \text{logit}(p_{ij}) = \ln\left(\frac{p_{ij}}{1-p_{ij}}\right) = \eta_{ij} = \mathbf{X}_{ij}^\top\boldsymbol\beta + u_i

where:

*   :math:`\boldsymbol\beta` is the :math:`p \times 1` vector of fixed regression coefficients associated with the covariates. These represent the change in log-odds of the outcome for a one-unit change in the corresponding covariate, holding the provider constant.
*   :math:`u_i` is the random intercept for provider :math:`i`. It represents the deviation of provider :math:`i`'s baseline log-odds from the overall intercept (which is part of :math:`\mathbf{X}_{ij}^\top\boldsymbol\beta` if an intercept term is included in :math:`\mathbf{X}_{ij}`).
*   It is assumed that :math:`u_i \sim N(0, \sigma^2_u)`, where :math:`\sigma^2_u` is the variance of the provider random effects.
*   The random effects :math:`u_i` are assumed to be independent of the covariates :math:`\mathbf{X}_{ij}`.

The model aims to estimate the fixed effects :math:`\boldsymbol\beta`, the random effects variance :math:`\sigma^2_u`, and to predict the individual random effects :math:`u_i`.

2.2. Parameter Estimation and Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``LogisticRandomEffectModel`` utilizes ``pymer4.models.Lmer`` (with ``family='binomial'``) to fit the GLMM. ``lme4`` typically uses (Restricted) Maximum Likelihood estimation, often employing methods like Laplace approximation or Penalized Quasi-Likelihood (PQL) for GLMMs.

*   **Fixed Effects** (:math:`\hat{\boldsymbol\beta}`): Estimates of the population-average covariate effects. Stored in ``coefficients_['fixed_effect']``.
*   **Random Effects** Variance (:math:`\hat{\sigma}^2_u`): Estimate of the variability between providers. Stored in ``variances_['re_var']``.
*   **Random Effects** (BLUPs, :math:`\hat{u}_i`): Predictions of the provider-specific deviations from the overall intercept. These are Best Linear Unbiased Predictors (BLUPs) on the log-odds scale and exhibit shrinkage towards the mean (zero). Stored in ``coefficients_['random_effect']``.
*   **Variance-Covariance of Fixed Effects:** Stored in ``variances_['fe_var_cov']``.

The fitted probabilities :math:`\hat{p}_{ij}` (including random effects) are stored in ``fitted_``. The linear predictor from fixed effects only, :math:`\mathbf{X}_{ij}^\top\hat{\boldsymbol\beta}`, is stored in ``xbeta_``.

2.3. Standardized Measures for Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For logistic random effects models, standardized measures allow for fair comparison of provider performance by adjusting for patient case mix and provider-specific random effects. The ``calculate_standardized_measures`` method computes both indirect and direct standardized ratios and rates, based on the predicted random effects (BLUPs).

Let :math:`\hat{\boldsymbol{\beta}}` denote the estimated fixed effects, and :math:`\hat{\alpha}_i` the estimated random effect (BLUP) for provider :math:`i`. Define a reference or baseline random effect :math:`\alpha_0` (e.g., the median or mean of :math:`\hat{\alpha}_i`, as specified by the ``null`` parameter).

**2.3.1. Indirect Standardization**

Indirect standardization compares the observed outcome for a provider (as predicted by the full model, including random effects) to the expected outcome for that provider if its random effect were set to the baseline value :math:`\alpha_0`, given its specific patient mix.

*   **Observed Outcome for Provider i**:
    The sum of fitted probabilities (including both fixed and random effects) for all :math:`n_i` subjects in provider :math:`i`. This is the ``observed`` column in the output DataFrame.

    .. math::

       O_i = \sum_{j=1}^{n_i} \hat{p}_{ij}^{\text{full}} = \sum_{j=1}^{n_i} \text{logit}^{-1}(\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \hat{\alpha}_i)

*   **Expected Outcome for Provider i under Baseline**:
    The sum of expected probabilities for provider :math:`i`'s :math:`n_i` subjects, if the provider effect was the baseline :math:`\alpha_0`, adjusted for their covariates. This is the ``expected`` column.

    .. math::

       E_i(\alpha_0) = \sum_{j=1}^{n_i} \text{logit}^{-1}(\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \alpha_0)

*   **Indirect Standardized Ratio**:
    The ratio of observed to expected outcomes for provider :math:`i`:

    .. math::

       \text{ISR}_i = \frac{O_i}{E_i(\alpha_0)}

*   **Indirect Standardized Rate**:
    The standardized rate for provider :math:`i`, calculated as the indirect ratio multiplied by the overall population rate (expressed as a percentage):

    .. math::

       \text{ISRate}_i = \text{ISR}_i \times \text{Population Rate}

*   **Indirect Standardized Difference**:
    The difference between the provider's random effect and the baseline:

    .. math::

       \text{ISDiff}_i = \hat{\alpha}_i - \alpha_0

**2.3.2. Direct Standardization**

Direct standardization compares the expected outcome for the entire population if all subjects experienced provider :math:`k`'s random effect (:math:`\hat{\alpha}_k`) to the expected outcome if all subjects experienced the baseline effect (:math:`\alpha_0`).

*   **Expected Outcome under Provider k's Effect**:
    The sum of expected probabilities for all subjects, using provider :math:`k`'s random effect:

    .. math::

       E^{(k)} = \sum_{i=1}^m \sum_{j=1}^{n_i} \text{logit}^{-1}(\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \hat{\alpha}_k)

*   **Expected Outcome under Baseline Effect**:
    The sum of expected probabilities for all subjects, using the baseline effect:

    .. math::

       E^{(0)} = \sum_{i=1}^m \sum_{j=1}^{n_i} \text{logit}^{-1}(\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \alpha_0)

*   **Direct Standardized Ratio**:
    The ratio of expected outcomes under provider :math:`k`'s effect to the observed total outcome:

    .. math::

       \text{DSR}_k = \frac{E^{(k)}}{O_{\text{total}}}

    where :math:`O_{\text{total}}` is the sum of observed outcomes across all subjects.

*   **Direct Standardized Rate**:
    The standardized rate for provider :math:`k`, calculated as the direct ratio multiplied by the overall population rate:

    .. math::

       \text{DSRate}_k = \text{DSR}_k \times \text{Population Rate}

*   **Direct Standardized Difference**:
    The difference between provider :math:`k`'s random effect and the baseline:

    .. math::

       \text{DSDiff}_k = \hat{\alpha}_k - \alpha_0


The implementation (``LogisticRandomEffectModel.calculate_standardized_measures``) returns a dictionary with DataFrames for each standardization method. Each DataFrame contains the group ID, standardized difference, ratio, rate, and the observed and expected outcomes as described above. These measures allow for meaningful provider comparisons while accounting for both patient case mix and provider-specific random effects.

.. 2.3. Standardized Measures for Performance Comparison
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Standardized measures, such as standardized rates or ratios, are calculated to compare provider performance after adjusting for case mix. These are derived using the predicted random effects (BLUPs, :math:`\hat{u}_i`) and a baseline reference level :math:`u_0`. The baseline :math:`u_0` can be the median or mean of :math:`\hat{u}_i`, or a specified value (e.g., 0), as defined by the ``null`` parameter in ``calculate_standardized_measures``.

.. The method ``calculate_standardized_measures`` computes:
.. *   **Difference:** :math:`\hat{u}_i - u_0` on the log-odds scale.
.. *   **Ratio:** For indirect standardization, this is often interpreted as (sum of fitted probabilities for provider *i*) / (sum of expected probabilities for provider *i* under null random effect). For direct standardization, it's (total expected events if all patients had provider *k*'s effect) / (total observed events).
.. *   **Rate:** The ratio multiplied by the overall population rate (e.g., events per 100 patients).

.. The "observed" sum for indirect standardization in the implementation refers to the sum of fitted probabilities (including the provider's random effect), while "expected" refers to the sum of probabilities calculated using only fixed effects (or fixed effects + :math:`u_0`).

2.4. Hypothesis Testing for Provider Random Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To assess if a provider's random effect :math:`u_i` is significantly different from a null value :math:`u_0`, a Z-test is performed:

.. math::

   Z_i = \frac{\hat{u}_i - u_0}{\widehat{\text{se}}(\hat{u}_i)}

where :math:`\hat{u}_i` is the BLUP for provider :math:`i`, and :math:`\widehat{\text{se}}(\hat{u}_i)` is its posterior standard error. These standard errors are obtained from the ``lme4`` model via ``pymer4`` (see ``_get_blup_post_se`` method).

Under the null hypothesis :math:`H_0: u_i = u_0`, the statistic :math:`Z_i` is assumed to follow a standard normal distribution. P-values are calculated based on this distribution according to the specified ``alternative`` ('two_sided', 'less', 'greater') in the ``test`` method.

2.5. Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^

*   **For Fixed Effects** :math:`\boldsymbol\beta`:
    The ``summary`` method provides confidence intervals for fixed effects, typically based on Z-scores and standard errors from the ``lme4`` output.

    .. math::

       \hat{\beta}_k \pm z_{1-\alpha/2} \times \widehat{\text{se}}(\hat{\beta}_k)

*   **For Random Effects** :math:`u_i` (BLUPs):
    The ``calculate_confidence_intervals`` method with ``option='alpha'`` provides approximate confidence intervals for the BLUPs :math:`\hat{u}_i` on the log-odds scale. These are based on the posterior standard errors of the BLUPs and a normal approximation:

    .. math::

       \hat{u}_i \pm z_{1-\alpha/2} \times \widehat{\text{se}}(\hat{u}_i)

*   **For Standardized Measures** (Ratios/Rates):
    Calculating robust confidence intervals for standardized ratios or rates derived from logistic random effect models is statistically complex. It often requires methods like the Delta method on the log/logit scale or extensive bootstrapping. The ``calculate_confidence_intervals`` method with ``option='SM'`` raises a ``NotImplementedError``, advising users to consider these advanced methods or consult a statistician.

2.6. Visualization
^^^^^^^^^^^^^^^^^^

The ``LogisticRandomEffectModel`` class offers several plotting methods:

*   **Caterpillar Plot for Provider Effects** (``plot_provider_effects``): Displays BLUPs :math:`\hat{u}_i` (on the log-odds scale) with their approximate confidence intervals.
*   **Coefficient Forest Plot** (``plot_coefficient_forest``): Displays estimates and confidence intervals for fixed effect coefficients :math:`\hat{\boldsymbol{\beta}}`.
*   **Residual Plots** (``plot_residuals``): Plots response or Pearson residuals against fitted probabilities. Note that residuals from logistic models have specific patterns and are not expected to be homoscedastic or normally distributed like in linear models.

The following plots are **not implemented** due to theoretical complexities or limited utility in the logistic random effect context:

*   **Funnel Plot** (``plot_funnel``): Standard funnel plots with control limits are difficult to define rigorously for random effect models, especially logistic ones.
*   **Q-Q Plot of Residuals** (``plot_qq``): Residuals from logistic regression are not expected to be normally distributed, making standard Q-Q plots less meaningful for overall model assessment.
*   **Caterpillar Plot for Standardized Measures** (``plot_standardized_measures``): This is not implemented because robust CIs for these measures are not provided by the corresponding CI calculation method.

3. Implementation and Usage
---------------------------

The ``LogisticRandomEffectModel`` class in ``pprof_test.logistic_random_effect`` implements these methods.

3.1. Initialization and Fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   import numpy as np
   from pprof_test.logistic_random_effect import LogisticRandomEffectModel

   # Generate synthetic data (from class docstring)
   np.random.seed(123)
   n_groups = 20; n_obs_per_group = 50; N = n_groups * n_obs_per_group
   groups = np.repeat([f'G{i+1}' for i in range(n_groups)], n_obs_per_group)
   x1 = np.random.randn(N); x2 = np.random.binomial(1, 0.4, N)
   true_beta = np.array([0.5, -1.0]); true_intercept = -0.5; true_re_sd = 0.8
   true_re_dict = {f'G{i+1}': np.random.normal(0, true_re_sd) for i in range(n_groups)}
   true_re_obs = np.array([true_re_dict[g] for g in groups])
   lin_pred = true_intercept + x1 * true_beta[0] + x2 * true_beta[1] + true_re_obs
   prob = 1 / (1 + np.exp(-lin_pred)); y = np.random.binomial(1, prob)
   data = pd.DataFrame({'Y': y, 'X1': x1, 'X2': x2, 'GroupID': groups})

   # Initialize and fit the model
   logit_re_model = LogisticRandomEffectModel()
   logit_re_model.fit(X=data, y_var='Y', x_vars=['X1', 'X2'], group_var='GroupID')
   # Output: Fitting Lmer model with formula: Y ~ X1 + X2 + (1 | GroupID)
   # Output: Model fitting complete.

3.2. Accessing Results
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Fixed Effects Coefficients (log-odds scale)
   fixed_effects = logit_re_model.coefficients_['fixed_effect']
   print("Estimated Fixed Effects (Betas):")
   print(fixed_effects)

   # Predicted Random Effects (BLUPs, log-odds scale)
   random_effects_blups = logit_re_model.coefficients_['random_effect']
   print("\nPredicted Random Effects (BLUPs, first 5):")
   print(random_effects_blups.head())

   # Variance Components
   fe_var_cov = logit_re_model.variances_['fe_var_cov']
   re_var = logit_re_model.variances_['re_var'] # Variance of random effects (sigma_u^2)
   print(f"\nEstimated Variance of Random Effects (sigma_u^2): {re_var:.3f}")

   # Fit statistics
   print(f"\nAIC: {logit_re_model.aic_:.2f}")
   print(f"BIC: {logit_re_model.bic_:.2f}")
   print(f"Log-likelihood: {logit_re_model.loglike_:.2f}")

   # Summary for fixed effects
   # fe_summary = logit_re_model.summary()
   # print("\n--- Fixed Effects Summary ---")
   # print(fe_summary)

3.3. Prediction (Fixed Effects Only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Predict probabilities for new data using only fixed effects
   new_data_df = pd.DataFrame({
       'X1': np.random.randn(5),
       'X2': np.random.binomial(1, 0.5, 5)
   })
   
   predictions_fe_only = logit_re_model.predict(
       X=new_data_df,
       x_vars=['X1', 'X2'] # or let it use fitted covariates
   )
   print(f"\nPredicted probabilities (fixed effects only): {predictions_fe_only}")

3.4. Standardized Measures Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Calculate Indirect Standardized Ratios and Rates vs median random effect
   sm_results_logit_re = logit_re_model.calculate_standardized_measures(
       stdz='indirect',
       null='median',
       measure=['ratio', 'rate']
   )
   print("\n--- Logistic RE Indirect Measures (vs Median Random Effect) ---")
   if 'indirect' in sm_results_logit_re:
       print(sm_results_logit_re['indirect'].head())

3.5. Hypothesis Testing for Provider Random Effects (``test``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Test provider random effects (:math:`u_i`, on log-odds scale) against a null value.

.. code-block:: python

   # Test providers vs a null random effect of 0.0
   test_results_logit_re = logit_re_model.test(
       null=0.0, # Null hypothesis for u_i (log-odds)
       level=0.95,
       alternative='two_sided'
   )
   print("\n--- Logistic RE Provider Test (vs Null RE of 0.0) ---")
   print(test_results_logit_re.head())

3.6. Confidence Interval Calculation (``calculate_confidence_intervals``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Compute CIs for provider random effects (:math:`u_i`, on log-odds scale).

.. code-block:: python

   # Get 95% CIs for random effects (BLUPs)
   alpha_cis_logit_re_results = logit_re_model.calculate_confidence_intervals(
       option='alpha', # For random effects u_i
       level=0.95
       # alternative is 'two_sided' by default for 'alpha'
   )
   print("\n--- Logistic RE Random Effect (u_i) CIs (Log-Odds Scale) ---")
   if 'alpha_ci' in alpha_cis_logit_re_results:
       print(alpha_cis_logit_re_results['alpha_ci'].head())
   
   # Note: CIs for standardized measures (option='SM') are not implemented.
   # try:
   #     sm_cis_results = logit_re_model.calculate_confidence_intervals(option='SM')
   # except NotImplementedError as e:
   #     print(f"\nError for SM CIs: {e}")

3.7. Visualization
^^^^^^^^^^^^^^^^^^
Use plotting methods from the ``LogisticRandomEffectModel`` instance.

.. code-block:: python

   # Example: Plot provider random effects (BLUPs on log-odds scale)
   # logit_re_model.plot_provider_effects(null=0.0, level=0.95)

   # Example: Forest plot for fixed covariate effects (beta on log-odds scale)
   # logit_re_model.plot_coefficient_forest()

   # Example: Residual plot (Pearson residuals vs. fitted probabilities)
   # logit_re_model.plot_residuals(residual_type="pearson")
   
   # The following plots will raise NotImplementedError:
   # logit_re_model.plot_funnel()
   # logit_re_model.plot_qq()
   # logit_re_model.plot_standardized_measures()

4. Discussion
-------------

Logistic random effect models, as implemented via ``pymer4`` and ``lme4``, provide a robust framework for analyzing binary outcomes clustered within providers. They account for provider-level heterogeneity through random intercepts, allowing for more nuanced comparisons than models ignoring such clustering.

**Advantages:**

*   Appropriately models binary outcomes and accounts for data clustering.
*   Leverages the well-established ``lme4`` package in R for estimation.
*   Provides estimates of fixed effects (population average) and predictions of random effects (provider-specific deviations, BLUPs).
*   BLUPs incorporate shrinkage, which can be beneficial for ranking or comparing providers, especially those with small sample sizes or extreme raw rates.

**Limitations and Assumptions:**

*   **Software Dependency:** Requires ``pymer4``, R, and the ``lme4`` R package to be correctly installed and configured.
*   **GLMM Assumptions:** Relies on standard GLMM assumptions, including linearity on the logit scale, correct specification of the random effects distribution (typically normal), and independence of random effects from covariates.
*   **Computational Complexity:** Fitting GLMMs can be computationally intensive, especially with large datasets or complex random effects structures.
*   **Interpretation of Standardized Measures:** Standardized ratios/rates derived from logistic models can be complex to interpret, and their confidence intervals are challenging to compute accurately without advanced statistical methods (e.g., bootstrapping). The current implementation provides point estimates for these measures.

5. Conclusion
-------------

The ``LogisticRandomEffectModel`` offers a valuable tool for provider profiling with binary outcomes. By incorporating random effects, it provides a statistically sound approach to adjust for case mix and account for provider-level variability. While the interpretation and inference for standardized measures derived from these models require care, the model's ability to estimate provider-specific effects (BLUPs) on the log-odds scale, along with their approximate confidence intervals, is a key strength for performance evaluation.


References
----------

.. bibliography:: references.bib
   :list: enumerate
   :filter: docname in docnames
