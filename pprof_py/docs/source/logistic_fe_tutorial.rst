.. _logistic_fixed_effect_tutorial:

Tutorial: Logistic Fixed Effect Models for Provider Profiling
==============================================================

.. contents::
   :local:
   :depth: 2

Introduction
------------

This tutorial will guide you through using the ``LogisticFixedEffectModel`` from the ``pprof_py`` package. This model is designed for provider profiling when you have a binary (0/1) outcome, such as mortality, readmission, or complication occurrence. It helps you assess provider performance while adjusting for patient-level risk factors by estimating a unique fixed effect (log-odds adjustment) for each provider.

The key idea is to model the log-odds of an event for patient :math:`j` at provider :math:`i` as:

.. math::

   \text{logit}(P(Y_{ij}=1)) = \gamma_i + \mathbf{X}_{ij}^\top\boldsymbol\beta

Where:

* :math:`\gamma_i` is the unique fixed effect (log-odds intercept adjustment) for provider :math:`i`.
* :math:`\mathbf{X}_{ij}^\top\boldsymbol\beta` represents the effects of patient-level covariates on the log-odds.
* :math:`P(Y_{ij}=1)` is the probability of the event.

This tutorial focuses on practical application: fitting the model, understanding its outputs, and visualizing the results. The model uses custom optimization algorithms ('Serbin' or 'Ban') for estimation.

1. Getting Started: Fitting Your First Model
---------------------------------------------

Let's begin by importing necessary libraries, preparing some example data, and fitting the model.

1.1. Import Libraries
~~~~~~~~~~~~~~~~~~~~~

You'll primarily need ``pandas`` for data manipulation and the ``LogisticFixedEffectModel`` itself.

.. plot::
   :context: close-figs
   :include-source: True

   import pandas as pd
   import numpy as np
   from pprof_py import LogisticFixedEffectModel
   import matplotlib.pyplot as plt
   print("Libraries imported.")

1.2. Prepare Your Data
~~~~~~~~~~~~~~~~~~~~~~

The model expects your data in a ``pandas`` DataFrame with:

* A **binary outcome variable** column (0 or 1).
* One or more **covariate columns** (patient risk factors).
* A **group variable** column (provider identifiers).

Here's how you can simulate some data for this tutorial:

.. plot::
   :context: close-figs
   :include-source: True

   # Simulate data for demonstration
   np.random.seed(420)
   n_providers_logistic = 100
   n_patients_per_provider_logistic = np.random.randint(50, 150, n_providers_logistic)
   n_total_patients_logistic = np.sum(n_patients_per_provider_logistic)

   provider_ids_logistic = []
   for i, count in enumerate(n_patients_per_provider_logistic):
       provider_ids_logistic.extend([f"Clinic_{i+1}"] * count)

   data_lfe = pd.DataFrame({
       'patient_id': range(n_total_patients_logistic),
       'provider_id': provider_ids_logistic,
       'age_patient': np.random.normal(65, 8, n_total_patients_logistic),
       'chronic_conditions': np.random.randint(0, 4, n_total_patients_logistic),
       'prior_admission': np.random.choice([0, 1], n_total_patients_logistic, p=[0.8, 0.2])
   })

   # Simulate true provider effects (log-odds adjustments) and binary outcome (e.g., 30-day readmission)
   provider_log_odds_effect_map = {f"Clinic_{i+1}": np.random.normal(0, 0.5) for i in range(n_providers_logistic)}
   data_lfe['true_provider_log_odds_effect'] = data_lfe['provider_id'].map(provider_log_odds_effect_map)
   
   # Log-odds calculation
   log_odds = (
       -2.5  # Base log-odds
       + 0.02 * (data_lfe['age_patient'] - 65) # Centering age
       + 0.3 * data_lfe['chronic_conditions']
       + 0.6 * data_lfe['prior_admission']
       + data_lfe['true_provider_log_odds_effect']
   )
   
   # Convert log-odds to probability
   probabilities = 1 / (1 + np.exp(-log_odds))
   
   # Simulate binary outcome
   data_lfe['readmitted_30day'] = (np.random.rand(n_total_patients_logistic) < probabilities).astype(int)

   # Define variable names for the model
   outcome_var_lfe = 'readmitted_30day'
   covariate_vars_lfe = ['age_patient', 'chronic_conditions', 'prior_admission']
   group_var_lfe = 'provider_id'

   print("Sample data for Logistic Fixed Effect Model (first 5 rows):")
   print(data_lfe.head())
   print(f"\nOutcome: {outcome_var_lfe}, Covariates: {covariate_vars_lfe}, Group: {group_var_lfe}")
   print(f"\nOverall event rate: {data_lfe[outcome_var_lfe].mean():.3f}")

1.3. Initialize and Fit the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instantiate ``LogisticFixedEffectModel`` and use its ``fit()`` method. You can choose between 'Serbin' and 'Ban' algorithms.

.. plot::
   :context: close-figs
   :include-source: True

   # Initialize the model
   # You can specify algorithm='Serbin' (default) or 'Ban'
   # Other parameters control data preparation steps if use_dataprep=True (default)
   lfe_model_logistic = LogisticFixedEffectModel(
       algorithm='Serbin',
       screen_providers=True, # Example: ensure providers meet minimum size
       cutoff=10              # Example: minimum 10 patients per provider
   )

   # Fit the model
   lfe_model_logistic.fit(
       X=data_lfe,
       y_var=outcome_var_lfe,
       x_vars=covariate_vars_lfe,
       group_var=group_var_lfe,
       max_iter=1000, # Max iterations for the optimization algorithm
       tol=1e-6       # Tolerance for convergence
   )
   # The "Model fitting complete." message comes from the .fit() method.

2. Understanding Model Results
------------------------------
Once the model is fitted, you can access various estimated parameters and statistics.

2.1. Coefficients (Beta and Gamma)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **Beta (:math:`\boldsymbol{\beta}`):** Estimated log-odds ratios for your patient-level covariates.
*   **Gamma (:math:`\boldsymbol{\gamma}`):** Estimated fixed effects (log-odds adjustments) for each provider.

.. plot::
   :context: close-figs
   :include-source: True

   coefficients_lfe = lfe_model_logistic.coefficients_
   beta_coeffs_lfe_df = coefficients_lfe['beta']  # Covariate effects (as Series)
   gamma_coeffs_lfe_df = coefficients_lfe['gamma'] # Provider fixed effects (as Series)

   print("\nEstimated Beta Coefficients (Log-Odds Ratios for Covariates):\n", beta_coeffs_lfe_df)
   print("\nEstimated Gamma Coefficients (Log-Odds Adjustments for Providers, first 5):\n", gamma_coeffs_lfe_df.head())

   # Interpretation:
   # For Beta: A one-unit increase in 'age_patient' is associated with a change of [beta_for_age] in the log-odds of 'readmitted_30day', holding other factors and provider constant.
   # For Gamma: Clinic_X has an adjusted log-odds of 'readmitted_30day' that is [gamma_for_Clinic_X] units different from the reference when covariates are at their baseline.

2.2. Variances
~~~~~~~~~~~~~~

Access variance-covariance estimates for coefficients.

.. plot::
   :context: close-figs
   :include-source: True

   variances_lfe = lfe_model_logistic.variances_
   var_beta_lfe_df = variances_lfe['beta']    # Variance-covariance matrix for beta
   var_gamma_lfe_df = variances_lfe['gamma']  # Variances for gamma (diagonal)

   print("\nVariance-Covariance Matrix for Beta:\n", var_beta_lfe_df)
   print("\nVariances for Gamma (Providers, first 5):\n", var_gamma_lfe_df.head())

2.3. Model Fit Statistics (AIC, BIC, AUC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AIC and BIC for model comparison, and AUC for discriminative ability.

.. plot::
   :context: close-figs
   :include-source: True

   aic_lfe = lfe_model_logistic.aic_
   bic_lfe = lfe_model_logistic.bic_
   auc_lfe = lfe_model_logistic.auc_

   print(f"\nAIC: {aic_lfe:.2f}")
   print(f"BIC: {bic_lfe:.2f}")
   print(f"AUC: {auc_lfe:.4f}")

2.4. Detailed Summary of Covariate Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``summary()`` method provides a table with estimates, standard errors, test statistics, p-values, and confidence intervals for the **beta** coefficients. You can choose 'wald', 'lr' (Likelihood Ratio), or 'score' test methods.

.. plot::
   :context: close-figs
   :include-source: True

   covariate_summary_lfe_df = lfe_model_logistic.summary(test_method='wald') # or 'lr', 'score'
   print("\nSummary of Covariate Effects (Betas - Wald test):\n", covariate_summary_lfe_df)

3. Making Predictions
---------------------

Use the ``predict()`` method to get predicted probabilities.

.. plot::
   :context: close-figs
   :include-source: True

   # Predict on the training data
   data_lfe['predicted_prob_readmission'] = lfe_model_logistic.predict(
       X=data_lfe,
       x_vars=covariate_vars_lfe,
       group_var=group_var_lfe
   )
   print("\nData with predicted probabilities (selected columns, first 5 rows):\n", data_lfe[[group_var_lfe, outcome_var_lfe, 'predicted_prob_readmission']].head())

4. Standardized Measures for Fair Comparison
--------------------------------------------

Standardized measures adjust provider outcomes to a common baseline. For logistic models, these are typically **Standardized Mortality/Morbidity Ratios (SMRs)** or **Standardized Rates**.
The ``calculate_standardized_measures()`` method computes:

*   **Indirect Standardized Ratio (O/E Ratio):** Observed events divided by Expected events if the provider performed at a baseline level (e.g., median provider), given *its own patient mix*.
*   **Direct Standardized Ratio/Rate:** (Conceptually) Compares the expected outcome if the *entire population* experienced a specific provider's effect versus the baseline effect.

.. plot::
   :context: close-figs
   :include-source: True

   # Calculate indirect standardized ratios and rates against the median provider performance
   sm_results_lfe = lfe_model_logistic.calculate_standardized_measures(
       stdz='indirect',  # or 'direct', or ['indirect', 'direct']
       null='median'     # Baseline: 'median' or 'mean' of estimated gammas, or a specific float value for gamma_null
   )

   indirect_sm_lfe_df = sm_results_lfe['indirect']
   print("\nIndirect Standardized Measures (vs median gamma, first 5):\n", indirect_sm_lfe_df.head())
   # Key columns: 'indirect_ratio' (O/E), 'indirect_rate' (Adjusted Rate), 'observed', 'expected'.
   # An indirect_ratio > 1 means more events observed than expected at baseline.

5. Hypothesis Testing: Identifying Outliers
-------------------------------------------

The ``test()`` method performs tests for each provider's estimated effect (:math:`\hat{\gamma}_i`) against a null baseline. Supported methods include 'poibin_exact' (default), 'bootstrap_exact', 'score', and 'wald'.

.. plot::
   :context: close-figs
   :include-source: True

   # Test provider effects against the median gamma using Poisson-Binomial exact test
   provider_test_lfe_results_df = lfe_model_logistic.test(
       null='median',          # Null hypothesis value for gamma
       level=0.95,             # Corresponds to alpha = 0.05 for flagging
       test_method='poibin_exact', # or 'score', 'wald', 'bootstrap_exact'
       alternative='two_sided'
   )

   print("\nProvider Test Results (vs median gamma, poibin_exact, first 5):\n", provider_test_lfe_results_df.head())
   # Key columns: 'flag' (-1, 0, 1), 'p_value', 'stat'.

   print("\nProviders flagged as significantly different (example):\n", provider_test_lfe_results_df[provider_test_lfe_results_df['flag'] != 0].head())

6. Confidence Intervals for Effects and Measures
------------------------------------------------

Calculate confidence intervals for provider effects (:math:`\gamma_i`) or standardized measures.

6.1. CIs for Provider Effects (:math:`\gamma_i`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use `option='gamma'`. Supported `test_method` for CIs: 'wald', 'score', 'exact'.

.. plot::
   :context: close-figs
   :include-source: True

   gamma_ci_lfe_results = lfe_model_logistic.calculate_confidence_intervals(
       option='gamma',
       level=0.95,
       test_method='exact', # 'wald', 'score', or 'exact' (Poisson-Binomial based)
       alternative='two_sided' # Must be 'two_sided' for option='gamma'
   )
   gamma_ci_lfe_df = gamma_ci_lfe_results['gamma_ci']
   print("\nConfidence Intervals for Gamma (Exact method, first 5):\n", gamma_ci_lfe_df.head())

6.2. CIs for Standardized Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use `option='SM'`. CIs for SMs are derived from the CIs of gamma.

.. plot::
   :context: close-figs
   :include-source: True

   sm_ci_lfe_results = lfe_model_logistic.calculate_confidence_intervals(
       option='SM',
       stdz='indirect',
       measure=['ratio', 'rate'], # Can be 'ratio', 'rate', or both
       null='median',
       level=0.95,
       test_method='exact', # Method used for underlying gamma CIs
       alternative='two_sided'
   )
   indirect_ratio_ci_lfe_df = sm_ci_lfe_results.get('indirect_ratio')
   indirect_rate_ci_lfe_df = sm_ci_lfe_results.get('indirect_rate')

   if indirect_ratio_ci_lfe_df is not None:
       print("\nCIs for Indirect Standardized Ratio (first 5):\n", indirect_ratio_ci_lfe_df.head())
   if indirect_rate_ci_lfe_df is not None:
       print("\nCIs for Indirect Standardized Rate (first 5):\n", indirect_rate_ci_lfe_df.head())

7. Visualizing Model Results
----------------------------

Visualizations are crucial for understanding provider performance.

7.1. Caterpillar Plot of Provider Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shows each provider's estimated :math:`\hat{\gamma}_i` (log-odds adjustment) with CIs.

.. plot::
   :context: close-figs
   :include-source: True

   lfe_model_logistic.plot_provider_effects(
       level=0.95,
       test_method='exact', # Method for CI calculation: 'wald', 'score', or 'exact'
       use_flags=True,
       null='median',
       title="Provider Effects: Adjusted Log-Odds (Gamma)",
       figure_size=(8, 6)
   )

7.2. Caterpillar Plot of Standardized Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example for Indirect Standardized Ratio (O/E Ratio).

.. plot::
   :context: close-figs
   :include-source: True

   lfe_model_logistic.plot_standardized_measures(
       stdz='indirect',
       measure='ratio',
       level=0.95,
       test_method='exact', # Method for underlying gamma CIs: 'wald', 'score', or 'exact'
       use_flags=True,
       null='median',
       title="Provider Standardized Ratios (Indirect O/E)",
       figure_size=(8, 6)
   )

7.3. Funnel Plot
~~~~~~~~~~~~~~~~

Plots Indirect Standardized Ratios against provider precision. Control limits can be based on 'score' or 'poibin_exact' methods.

.. plot::
   :context: close-figs
   :include-source: True

   lfe_model_logistic.plot_funnel(
       test_method='poibin_exact', # 'score' or 'poibin_exact' for control limits
       null='median',
       target=1.0,          # Target O/E ratio
       alpha=[0.05, 0.01],  # For 95% and 99% control limits
       plot_title="Funnel Plot: Indirect Standardized Ratios (O/E)",
       ylab="Indirect Standardized Ratio (O/E)"
   )

7.4. Forest Plot of Covariate Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualizes the estimated :math:`\hat{\beta}_k` (log-odds ratios) and their CIs.

.. plot::
   :context: close-figs
   :include-source: True

   lfe_model_logistic.plot_coefficient_forest(
       plot_title="Forest Plot of Covariate Log-Odds Ratios (Beta)"
   )

7.5. Model Diagnostic Plots (Residuals, Q-Q)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard residual plots are less straightforward for logistic regression. The `plot_residuals` and `plot_qq` methods in this class currently raise `NotImplementedError`.

.. code-block:: text

   # lfe_model_logistic.plot_residuals() # Raises NotImplementedError
   # lfe_model_logistic.plot_qq()      # Raises NotImplementedError

   print("\nNote: plot_residuals and plot_qq are not implemented for LogisticFixedEffectModel.")
   print("Residual diagnostics for logistic regression often involve binned residuals,")
   print("Hosmer-Lemeshow tests, or other specialized techniques not included by default.")

Alternative diagnostics might involve:
*   Calibration plots (observed vs. expected probabilities within deciles of risk).
*   Goodness-of-fit tests like Hosmer-Lemeshow (though be cautious with large sample sizes).
*   Examining influence statistics if available.

8. Quick Interpretation Guide
-----------------------------

*   **Covariate Coefficients (:math:`\boldsymbol{\beta}`):** Indicate how the log-odds of the event change for a one-unit change in a covariate. Exponentiating a beta coefficient (:math:`e^\beta`) gives the odds ratio.
*   **Provider Effects (:math:`\boldsymbol{\gamma_i}`):** Represent provider-specific adjustments to the log-odds. Exponentiating gamma (:math:`e^\gamma`) gives the odds ratio for that provider relative to the reference (when covariates are zero or at their mean).
*   **Standardized Ratios (e.g., Indirect O/E Ratio):**

    *   A ratio of 1.0 means the provider performed as expected compared to the baseline.
    *   A ratio > 1.0 means more events observed than expected (e.g., higher readmission rate).
    *   A ratio < 1.0 means fewer events observed than expected.
    
*   **Standardized Rates:** Adjusted event rates for providers, comparable across different case mixes.
*   **Caterpillar Plots:** Visually compare providers based on their adjusted effects or standardized measures.
*   **Funnel Plots:** Identify potential outliers by plotting performance against precision (often related to volume). Providers outside the funnel limits warrant closer inspection.
*   **AUC:** Measures the model's ability to discriminate between events and non-events. Higher is better (1.0 is perfect, 0.5 is no better than chance).

This tutorial provides a foundation for using the ``LogisticFixedEffectModel``. Always refer to the specific method docstrings for detailed parameter explanations and explore the various customization options.