.. _logistic_random_effect_tutorial:

Logistic Random Effect Models for Provider Profiling
===============================================================

.. contents::
   :local:
   :depth: 2

Introduction
------------
This tutorial will guide you through using the ``LogisticRandomEffectModel`` from the ``pprof_py`` package. This model is designed for provider profiling with a binary (0/1) outcome (e.g., mortality, readmission) when you want to model provider variation as random effects. This approach is useful when you have many providers, wish to "borrow strength" across them, or consider providers as a sample from a larger population.

The Logistic Random Effect Model (a type of Generalized Linear Mixed Model - GLMM) models the log-odds of an event for patient :math:`j` at provider :math:`i` as:

.. math::

   \text{logit}(P(Y_{ij}=1)) = \mathbf{X}_{ij}^\top\boldsymbol\beta + u_i

Where:

* :math:`\mathbf{X}_{ij}^\top\boldsymbol\beta` represents the fixed effects of patient-level covariates on the log-odds.
* :math:`u_i` is the random effect for provider :math:`i`, typically assumed :math:`u_i \sim N(0, \sigma_u^2)`.
* :math:`P(Y_{ij}=1)` is the probability of the event.

This tutorial focuses on fitting the model, understanding its outputs, and visualizing results. The implementation is pure Python (lme4-style PIRLS + Laplace approximation) and does not require R or pymer4.

1. Getting Started: Fitting Your First Model
---------------------------------------------
Let's begin by importing libraries, preparing example data, and fitting the model.

1.1. Import Libraries
~~~~~~~~~~~~~~~~~~~~~
You'll need ``pandas``, ``numpy``, and the ``LogisticRandomEffectModel``.

.. plot::
   :context: close-figs
   :include-source: True

   import pandas as pd
   import numpy as np
   # Ensure this path is correct for your package structure
   from pprof_py import LogisticRandomEffectModel
   import matplotlib.pyplot as plt
   print("Libraries imported.")

1.2. Prepare Your Data
~~~~~~~~~~~~~~~~~~~~~~
The model expects data in a ``pandas`` DataFrame with:

* A **binary outcome variable** column (0 or 1).
* One or more **covariate columns** (for fixed effects).
* A **group variable** column (for random effects, e.g., provider identifiers).

Here's how to simulate data:

.. plot::
   :context: close-figs
   :include-source: True

   # Simulate data for demonstration
   np.random.seed(789)
   n_providers_log_re = 25
   n_patients_per_provider_log_re = np.random.randint(40, 120, n_providers_log_re)
   n_total_patients_log_re = np.sum(n_patients_per_provider_log_re)

   provider_ids_log_re = []
   for i, count in enumerate(n_patients_per_provider_log_re):
       provider_ids_log_re.extend([f"Hospital_RE_{i+1}"] * count) # String IDs for provider groups

   data_lre = pd.DataFrame({
       'patient_id': range(n_total_patients_log_re),
       'provider_id': provider_ids_log_re,
       'age_yrs': np.random.normal(60, 10, n_total_patients_log_re),
       'severity_score': np.random.gamma(2.5, 1.2, n_total_patients_log_re),
       'urgent_case': np.random.choice([0, 1], n_total_patients_log_re, p=[0.7, 0.3])
   })

   # Simulate true random provider effects (log-odds adjustments) and binary outcome
   true_re_sd_log = 0.6
   provider_log_odds_re_map = {
       f"Hospital_RE_{i+1}": np.random.normal(0, true_re_sd_log) for i in range(n_providers_log_re)
   }
   data_lre['true_provider_re_log_odds'] = data_lre['provider_id'].map(provider_log_odds_re_map)
   
   # Log-odds calculation
   log_odds_re = (
       -2.0  # Base log-odds (Intercept)
       + 0.03 * (data_lre['age_yrs'] - 60)
       + 0.25 * data_lre['severity_score']
       + 0.5 * data_lre['urgent_case']
       + data_lre['true_provider_re_log_odds'] # Add random effect
   )
   
   probabilities_re = 1 / (1 + np.exp(-log_odds_re))
   data_lre['complication'] = (np.random.rand(n_total_patients_log_re) < probabilities_re).astype(int)

   outcome_var_lre = 'complication'
   covariate_vars_lre = ['age_yrs', 'severity_score', 'urgent_case']
   group_var_lre = 'provider_id'

   print("Sample data for Logistic Random Effect Model (first 5 rows):")
   print(data_lre.head())
   print(f"\nOutcome: {outcome_var_lre}, Covariates: {covariate_vars_lre}, Group: {group_var_lre}")
   print(f"\nOverall event rate: {data_lre[outcome_var_lre].mean():.3f}")

1.3. Initialize and Fit the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate ``LogisticRandomEffectModel`` and use its ``fit()`` method.

.. plot::
   :context: close-figs
   :include-source: True

   # Initialize the model
   lre_model_logistic = LogisticRandomEffectModel()

   # Fit the model
   # The model internally constructs a formula like 'y_var ~ x_var1 + (1 | group_var)'
   lre_model_logistic.fit(
        X=data_lre,
        y_var=outcome_var_lre,
        x_vars=covariate_vars_lre,
        group_var=group_var_lre
    )
    print("Model fitting complete.")


2. Understanding Model Results
------------------------------
Assuming the model fitted successfully, you can access estimated parameters.

2.1. Coefficients (Fixed and Random Effects)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*   **Fixed Effects (:math:`\boldsymbol{\beta}`):** Estimated log-odds ratios for covariates.
*   **Random Effects (:math:`u_i`):** Predicted provider-specific deviations from the overall log-odds (Best Linear Unbiased Predictors - BLUPs).

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       coefficients_lre = lre_model_logistic.coefficients_
       fixed_effects_lre_series = coefficients_lre['beta']    # Fixed effects (as Series)
        random_effects_lre_series = lre_model_logistic.get_random_effects()  # BLUPs (as Series)

        if fixed_effects_lre_series is not None:
            print("\nEstimated Fixed Effects (Log-Odds Ratios for Covariates):\n", fixed_effects_lre_series)
       else:
           print("\nFixed effects not available.")
       
       if random_effects_lre_series is not None:
           print("\nPredicted Random Effects (BLUPs for Providers, first 5):\n", random_effects_lre_series.head())
       else:
           print("\nRandom effects not available.")
   else:
       print("Model not fitted successfully. Cannot display coefficients.")

2.2. Variances
~~~~~~~~~~~~~~
Access variance-covariance for fixed effects and the variance of random effects.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       variances_lre = lre_model_logistic.variances_
       fe_var_cov_lre_df = variances_lre['beta']  # VCV for fixed effects
        re_var_lre = variances_lre['alpha']          # Variance of random effects {group_var: sigma_u^2}

        if fe_var_cov_lre_df is not None:
            print("\nVariance-Covariance Matrix for Fixed Effects:\n", fe_var_cov_lre_df.head())
        else:
            print("\nFixed effects VCV not available.")
            
        if re_var_lre is not None:
            sigma_u2 = list(re_var_lre.values())[0]  # Get sigma_u^2 for first (only) group
            print(f"\nEstimated Variance of Random Effects (sigma_u^2): {sigma_u2:.4f}")
       else:
           print("\nRandom effects variance not available.")
   else:
       print("Model not fitted successfully. Cannot display variances.")

2.3. Model Fit Statistics (AIC, BIC, LogLik)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Access AIC, BIC, and Log-Likelihood from the fitted model.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       aic_lre_log = lre_model_logistic.aic_
       bic_lre_log = lre_model_logistic.bic_
       loglik_lre_log = lre_model_logistic.loglike_

       print(f"\nAIC: {aic_lre_log:.2f}" if aic_lre_log is not None else "\nAIC: Not available")
       print(f"BIC: {bic_lre_log:.2f}" if bic_lre_log is not None else "BIC: Not available")
       print(f"Log-Likelihood: {loglik_lre_log:.2f}" if loglik_lre_log is not None else "Log-Likelihood: Not available")
   else:
       print("Model not fitted successfully. Cannot display fit statistics.")

2.4. Detailed Summary of Fixed Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``summary()`` method provides a table for fixed effects (betas), typically including estimates, SE, Z-statistic, and p-values.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       try:
           fixed_effects_summary_lre_df = lre_model_logistic.summary()
           print("\nSummary of Fixed Effects (Betas):\n", fixed_effects_summary_lre_df)
       except Exception as e:
           print(f"Could not generate fixed effects summary: {e}")
   else:
       print("Model not fitted successfully. Cannot display fixed effects summary.")

3. Making Predictions
---------------------
*   ``predict()``: Predicts probabilities using only fixed effects (for new data).
*   ``fitted_``: Attribute storing fitted probabilities on the training data, including random effects.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       # Predict probabilities using only fixed effects for the training data (as an example)
       data_lre['pred_prob_fixed_only'] = lre_model_logistic.predict(
           X=data_lre,
            x_vars=covariate_vars_lre  # If None, uses covariates from fitting
       )
       print("\nData with predictions (fixed effects only, first 5 rows):\n",
             data_lre[[group_var_lre, outcome_var_lre, 'pred_prob_fixed_only']].head())

       # Fitted values including random effects (BLUPs) are stored in self.fitted_
       if lre_model_logistic.fitted_ is not None:
           data_lre['fitted_prob_with_re'] = lre_model_logistic.fitted_
           print("\nData with fitted probabilities (including RE, first 5 rows):\n",
                 data_lre[[group_var_lre, outcome_var_lre, 'fitted_prob_with_re']].head())
       else:
           print("\nFitted probabilities (including RE) not available.")
   else:
       print("Model not fitted successfully. Cannot make predictions.")

4. Standardized Measures for Fair Comparison
--------------------------------------------
Calculates standardized ratios (O/E) and rates based on BLUPs.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       try:
           sm_results_lre_log = lre_model_logistic.calculate_standardized_measures(
               stdz='indirect',
               null='median'  # Baseline for random effects: 'median', 'mean', or 0.0
           )
           indirect_sm_lre_log_df = sm_results_lre_log.get('indirect')
           if indirect_sm_lre_log_df is not None:
               print("\nIndirect Standardized Measures (vs median BLUP, first 5):\n", indirect_sm_lre_log_df.head())
           else:
               print("\nIndirect standardized measures not calculated.")
       except Exception as e:
           print(f"Could not calculate standardized measures: {e}")
   else:
       print("Model not fitted successfully. Cannot calculate standardized measures.")

5. Hypothesis Testing: Identifying Outlier Providers
----------------------------------------------------
Tests each provider's predicted random effect (:math:`\hat{u}_i`) against a null baseline using Z-tests.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       try:
           provider_test_lre_log_df = lre_model_logistic.test(
               null=0.0, # Test BLUPs against 0 (or 'median', 'mean')
               level=0.95,
               alternative='two_sided'
           )
           print("\nProvider Random Effect Test Results (vs null=0, first 5):\n", provider_test_lre_log_df.head())
           print("\nProviders flagged as significantly different (example):\n",
                 provider_test_lre_log_df[provider_test_lre_log_df['flag'] != 0].head())
       except Exception as e:
           print(f"Could not perform hypothesis tests for providers: {e}")
   else:
       print("Model not fitted successfully. Cannot perform provider tests.")

6. Confidence Intervals for Effects and Measures
------------------------------------------------

6.1. CIs for Provider Random Effects (:math:`u_i`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Calculates approximate CIs for the BLUPs using their posterior standard errors.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       try:
           re_ci_results = lre_model_logistic.calculate_confidence_intervals(
               option='alpha', # 'alpha' for random effects (BLUPs)
               level=0.95
           )
           re_ci_df = re_ci_results.get('alpha_ci')
           if re_ci_df is not None:
               print("\nApproximate CIs for Random Effects (BLUPs, first 5):\n", re_ci_df.head())
           else:
               print("\nRandom effect CIs not available.")
       except Exception as e:
           print(f"Could not calculate CIs for random effects: {e}")
   else:
       print("Model not fitted successfully. Cannot calculate CIs for random effects.")

6.2. CIs for Standardized Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute CIs for standardized ratios and/or rates using a Delta-method transformation of the BLUP posterior standard errors.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       try:
           sm_ci_results = lre_model_logistic.calculate_confidence_intervals(
               option='SM',
               stdz='indirect',
               null='median',
               measure=['ratio', 'rate'],
               level=0.95
           )
           if 'indirect_ratio' in sm_ci_results:
               print("\nCIs for Indirect Standardized Ratio (first 5):")
               print(sm_ci_results['indirect_ratio'].head())
           if 'indirect_rate' in sm_ci_results:
               print("\nCIs for Indirect Standardized Rate (first 5):")
               print(sm_ci_results['indirect_rate'].head())
       except Exception as e:
           print(f"Could not calculate CIs for standardized measures: {e}")
   else:
       print("Model not fitted. Skipping CIs for Standardized Measures.")


7. Visualizing Model Results
----------------------------

The ``LogisticRandomEffectModel`` provides four plotting methods via the ``RandomEffectPlottingMixin``.

7.1. Funnel Plot
~~~~~~~~~~~~~~~~

Plots the indirect standardized ratio (O/E) against expected count.  Poisson-based control limits flag outlier providers.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       lre_model_logistic.plot_funnel(
           test_method='wald',
           null='median',
           target=1.0,
           alpha=[0.05, 0.01],
           plot_title="Funnel Plot: Indirect Standardized Ratio (O/E)",
       )

7.2. Caterpillar Plot of Provider Random Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shows each provider's BLUP on the log-odds scale with confidence intervals.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       lre_model_logistic.plot_provider_effects(
           level=0.95,
           use_flags=True,
           null='median',
           plot_title="Provider Random Effects (BLUPs, Log-Odds Scale)",
       )

7.3. Caterpillar Plot of Standardized Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Displays indirect (or direct) standardized ratios or rates with CIs.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       lre_model_logistic.plot_standardized_measures(
           stdz='indirect',
           measure='ratio',
           level=0.95,
           use_flags=True,
           null='median',
           plot_title="Indirect Standardized Ratio (O/E) with CIs",
       )

7.4. Forest Plot of Fixed Effect Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualizes the fixed-effect estimates with z-based Wald CIs.

.. plot::
   :context: close-figs
   :include-source: True

   if lre_model_logistic.coefficients_ is not None:
       lre_model_logistic.plot_coefficient_forest(
           plot_title="Forest Plot of Covariate Coefficients (Log-Odds)"
       )


8. Quick Interpretation Guide
-----------------------------

*   **Fixed Effect Coefficients (:math:`\boldsymbol{\beta}`):** Log-odds ratios. Exponentiating (:math:`e^\beta`) gives the odds ratio for a one-unit change in a covariate, for an average provider.
*   **Random Effects (:math:`u_i`):** Provider-specific adjustments to the log-odds (BLUPs). Exponentiating (:math:`e^{u_i}`) gives the odds ratio for that provider relative to the average provider. The variance (:math:`\sigma_u^2`) indicates provider-level variability.
*   **Standardized Ratios/Rates:** Compare provider performance (e.g., O/E ratio) after adjusting for case mix and accounting for random effects.
*   **Caterpillar Plot of Random Effects** (``plot_provider_effects``): Visually compare providers based on their BLUPs with CIs and significance flags.
*   **Funnel Plot** (``plot_funnel``): Distinguish true outliers from providers with high variability due to small sample size.
*   **Standardized Measure Caterpillar** (``plot_standardized_measures``): Compare indirect/direct ratios or rates across providers.

This tutorial provides a foundation for using the ``LogisticRandomEffectModel``. The model is implemented in pure Python and does not require external R or pymer4 dependencies. Always refer to method docstrings for details.