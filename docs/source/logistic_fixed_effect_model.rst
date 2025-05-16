.. _logistic_fixed_effect_model:

Logistic Fixed Effect Modeling
============================================================================

.. contents::
   :local:
   :depth: 2

1. Introduction
---------------

Healthcare provider profiling, the process of evaluating and comparing the performance of healthcare entities like hospitals, clinics, or individual practitioners, is crucial for improving quality of care, managing costs, and informing patient choice :cite:`Shahian2011Public,Krumholz2013Administrative`. Performance is often assessed using patient outcomes such as mortality, readmission, or complication rates. However, providers serve different patient populations, necessitating robust risk adjustment methods to account for variations in patient characteristics (case mix) before making fair comparisons :cite:`Iezzoni2001Risk`.

Generalized linear models (GLMs), particularly logistic regression for binary outcomes, are commonly employed for risk adjustment. When dealing with data clustered within providers, incorporating provider-specific effects is essential. Two primary approaches exist: random effects (RE) models and fixed effects (FE) models. RE models assume provider effects are random variables drawn from a common distribution, allowing for "borrowing strength" across providers and potentially increasing efficiency, but rely on the strong assumption that provider effects are uncorrelated with patient covariates :cite:`Neuhaus1991Comparison`. Violation of this assumption, which is common in observational healthcare data, can lead to biased estimates :cite:`Kalbfleisch2013Monitoring`.

Fixed effects models, conversely, treat each provider's effect as a distinct, unknown parameter to be estimated. This approach is robust to the correlation between provider effects and covariates, making it preferable when unbiased estimation of individual provider performance, especially for potentially outlying providers, is the primary goal :cite:`He2013Evaluating`. However, estimating FE models with a large number of providers (:math:`m`) poses significant computational challenges for standard GLM algorithms due to the high dimensionality (:math:`m+p`, where :math:`p` is the number of covariates) of the parameter space :cite:`Greene2004Behaviour`.

This paper details the statistical methodology implemented in our software for fitting logistic fixed effect models efficiently, even with a large number of providers. We focus on:

  * The logistic fixed effects model formulation.
  * Computationally efficient estimation algorithms adapted from methods like SerBIN (wu2022improving).
  * Calculation of standardized measures (Indirect and Direct Standardization) for performance comparison.  
  * Robust hypothesis testing procedures (Wald, Score, Exact Poisson-Binomial, Exact Bootstrap) for identifying providers with performance significantly different from a benchmark.
  * Construction of corresponding confidence intervals for provider effects and standardized measures.
  * Visualization tools for interpreting results.

2. Methods
----------

2.1. The Logistic Fixed Effects Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`Y_{ij}` be the binary outcome variable (e.g., 1 for mortality, 0 for survival) for subject :math:`j` (:math:`j = 1, \ldots, n_i`) within provider :math:`i` (:math:`i = 1, \ldots, m`). Let :math:`\mathbf{X}_{ij}` be a :math:`p \times 1` vector of subject-level covariates. The logistic fixed effects model assumes:

.. math::

   Y_{ij} | \mathbf{X}_{ij}, \gamma_i \sim \text{Bernoulli}(p_{ij})

where the probability :math:`p_{ij}` is modeled via the logit link function:

.. math::

   \text{logit}(p_{ij}) = \log\left(\frac{p_{ij}}{1-p_{ij}}\right) = \eta_{ij} = \mathbf{X}_{ij}^\top\boldsymbol\beta + \gamma_i

Here, :math:`\boldsymbol\beta` is the :math:`p \times 1` vector of regression coefficients for the covariates, representing the change in the log-odds of the outcome associated with a one-unit increase in the corresponding covariate, conditional on the provider. :math:`\gamma_i` is the fixed effect for provider :math:`i`, representing the baseline log-odds for that provider when :math:`\mathbf{X}_{ij} = \mathbf{0}`. The full parameter vector is :math:`\boldsymbol{\theta} = (\boldsymbol{\gamma}^\top, \boldsymbol{\beta}^\top)^\top`, where :math:`\boldsymbol{\gamma} = (\gamma_1, \dots, \gamma_m)^\top`.

The likelihood function for the observed data :math:`D = \{(Y_{ij}, \mathbf{X}_{ij})\}_{i=1}^{m}, {j=1}^{n_i}` is:

.. math::

   L(\boldsymbol{\theta}; D) = \prod_{i=1}^m \prod_{j=1}^{n_i} p_{ij}^{Y_{ij}} (1-p_{ij})^{1-Y_{ij}}

The log-likelihood function is:

.. math::

   \ell(\boldsymbol{\theta}; D) = \sum_{i=1}^m \sum_{j=1}^{n_i} \left[ Y_{ij} \eta_{ij} - \log(1 + e^{\eta_{ij}}) \right]

.. math::

   \ell(\boldsymbol{\theta}; D) = \sum_{i=1}^m \sum_{j=1}^{n_i} \left[ Y_{ij} (\mathbf{X}_{ij}^\top\boldsymbol\beta + \gamma_i) - \log(1 + e^{\mathbf{X}_{ij}^\top\boldsymbol\beta + \gamma_i}) \right]

Maximum likelihood estimation (MLE) involves finding :math:`\hat{\boldsymbol{\theta}}` that maximizes :math:`\ell(\boldsymbol{\theta}; D)`.

2.2. Parameter Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^

Maximizing the log-likelihood typically involves iterative methods like Newton-Raphson or Fisher Scoring. The update step is given by:

.. math::

   \boldsymbol{\theta}^{(k+1)} = \boldsymbol{\theta}^{(k)} + [I(\boldsymbol{\theta}^{(k)})]^{-1} U(\boldsymbol{\theta}^{(k)})

where :math:`U(\boldsymbol{\theta}) = \frac{\partial \ell}{\partial \boldsymbol{\theta}}` is the score vector and :math:`I(\boldsymbol{\theta}) = -E\left[\frac{\partial^2 \ell}{\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}^\top}\right]` is the Fisher information matrix.

The components of the score vector are:

.. math::

   U(\gamma_i) = \frac{\partial \ell}{\partial \gamma_i} = \sum_{j=1}^{n_i} (Y_{ij} - p_{ij})

.. math::

   U(\beta_k) = \frac{\partial \ell}{\partial \beta_k} = \sum_{i=1}^m \sum_{j=1}^{n_i} X_{ijk} (Y_{ij} - p_{ij})

The Fisher information matrix :math:`I(\boldsymbol{\theta})` has a block structure:

.. math::

   I(\boldsymbol{\theta}) = \begin{pmatrix} I_{\gamma\gamma} & I_{\gamma\beta} \\ I_{\beta\gamma} & I_{\beta\beta} \end{pmatrix}

where

* :math:`I_{\gamma\gamma}` is an :math:`m \times m` diagonal matrix with diagonal elements :math:`[I_{\gamma\gamma}]_{ii} = \sum_{j=1}^{n_i} w_{ij} = \sum_{j=1}^{n_i} p_{ij}(1-p_{ij})`.
* :math:`I_{\beta\beta}` is a :math:`p \times p` matrix with elements :math:`[I_{\beta\beta}]_{kl} = \sum_{i=1}^m \sum_{j=1}^{n_i} X_{ijk} X_{ijl} w_{ij}`.
* :math:`I_{\gamma\beta} = I_{\beta\gamma}^\top` is an :math:`m \times p` matrix with elements :math:`[I_{\gamma\beta}]_{ik} = \sum_{j=1}^{n_i} X_{ijk} w_{ij}`.

The key challenge is inverting :math:`I(\boldsymbol{\theta})` when :math:`m` is large. The SerBIN approach :cite:`Wu2022Improving` leverages the fact that :math:`I_{\gamma\gamma}` is diagonal and uses the partitioned inverse formula involving the Schur complement :math:`S = I_{\beta\beta} - I_{\beta\gamma} I_{\gamma\gamma}^{-1} I_{\gamma\beta}`:

.. math::

   [I(\boldsymbol{\theta})]^{-1} = \begin{pmatrix} I_{\gamma\gamma}^{-1} + I_{\gamma\gamma}^{-1} I_{\gamma\beta} S^{-1} I_{\beta\gamma} I_{\gamma\gamma}^{-1} & -I_{\gamma\gamma}^{-1} I_{\gamma\beta} S^{-1} \\ -S^{-1} I_{\beta\gamma} I_{\gamma\gamma}^{-1} & S^{-1} \end{pmatrix}

This form is computationally advantageous because it requires inverting the diagonal :math:`I_{\gamma\gamma}` (trivial) and the :math:`p \times p` matrix :math:`S`. The ``SerbinAlgorithm`` implemented in our package uses this structure for efficient updates. The ``BanAlgorithm`` employs an alternating optimization strategy, updating :math:`\boldsymbol{\gamma}` holding :math:`\boldsymbol{\beta}` fixed, and then updating :math:`\boldsymbol{\beta}` holding :math:`\boldsymbol{\gamma}` fixed, iteratively.

To prevent numerical instability, particularly for providers with few observations or where separation occurs (all :math:`Y_{ij}=0` or all :math:`Y_{ij}=1`), the iterative updates for :math:`\hat{\gamma}_i` are often constrained within a plausible range, such as :math:`\hat{\gamma}_{\text{median}} \pm B`, where :math:`B` is a predefined bound (e.g., 10). The asymptotic variance-covariance matrix of the estimators is given by the inverse of the Fisher information matrix evaluated at the MLEs, :math:`[I(\hat{\boldsymbol{\theta}})]^{-1}`. The diagonal elements corresponding to :math:`\gamma_i` provide :math:`\text{Var}(\hat{\gamma}_i)`.

2.3. Standardized Measures for Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Raw provider effects (:math:`\hat{\gamma}_i`) are adjusted for patient covariates but are on the log-odds scale and depend on the specific patient mix within each provider. Standardized measures are often preferred for comparing providers, as they adjust for case mix and provide a more interpretable metric relative to a benchmark.

Let :math:`\hat{\boldsymbol{\beta}}` and :math:`\hat{\boldsymbol{\gamma}}` be the MLEs. Define a reference or baseline provider effect :math:`\gamma_0`. Common choices for :math:`\gamma_0` include the median or (weighted) mean of the estimated :math:`\hat{\gamma}_i`.

**2.3.1. Indirect Standardization**

Indirect standardization compares the observed number of events in a provider to the number expected if that provider's patients experienced the outcome probability associated with the baseline effect :math:`\gamma_0`, given their specific covariates.

*   **Observed Events**  (:math:`O_i`)::math:`O_i = \sum_{j=1}^{n_i} Y_{ij}`
*   **Expected Events** (:math:`E_i`): The expected count under the null/baseline effect :math:`\gamma_0`.

    .. math::

       E_i(\gamma_0) = \sum_{j=1}^{n_i} P(Y_{ij}=1 | \mathbf{X}_{ij}, \gamma_0, \hat{\boldsymbol{\beta}}) = \sum_{j=1}^{n_i} \frac{e^{\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \gamma_0}}{1 + e^{\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \gamma_0}}

*   **Indirect Standardized Ratio (ISR):**

    .. math::

       \text{ISR}_i = \frac{O_i}{E_i(\gamma_0)}

    An ISR > 1 indicates more events were observed than expected under the baseline performance level, while ISR < 1 indicates fewer events were observed.
*   **Indirect Standardized Rate:** The ISR can be scaled by the overall population event rate (:math:`\bar{Y} = (\sum O_i) / (\sum n_i)`) to get an adjusted rate.

    .. math::

       \text{Indirect Rate}_i = \text{ISR}_i \times \bar{Y} \times 100\%

    This represents the event rate provider :math:`i` would be expected to have if it performed like the baseline, applied to its specific patient mix, and then scaled by the overall rate. *Note: This rate is typically clipped to [0, 100].*

**2.3.2. Direct Standardization**

Direct standardization calculates the expected number of events if the *entire* population had the risk profile associated with a specific provider :math:`k`'s estimated effect :math:`\hat{\gamma}_k`.

*   **Total Observed Events** (:math:`O`): :math:`O = \sum_{i=1}^m O_i = \sum_{i=1}^m \sum_{j=1}^{n_i} Y_{ij}`
*   **Expected Events under Provider** :math:`k`'s Effect (:math:`E^{(k)}`):

    .. math::

       E^{(k)} = \sum_{i=1}^m \sum_{j=1}^{n_i} P(Y_{ij}=1 | \mathbf{X}_{ij}, \hat{\gamma}_k, \hat{\boldsymbol{\beta}}) = \sum_{i=1}^m \sum_{j=1}^{n_i} \frac{e^{\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \hat{\gamma}_k}}{1 + e^{\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \hat{\gamma}_k}}

*   **Direct Standardized Ratio (DSR):**

    .. math::

       \text{DSR}_k = \frac{E^{(k)}}{O}

    A DSR > 1 suggests that if the whole population experienced provider :math:`k`'s specific effect, more events would occur than were actually observed overall.
*   **Direct Standardized Rate:**

    .. math::

       \text{Direct Rate}_k = \text{DSR}_k \times \bar{Y} \times 100\%

    This represents the overall event rate expected if the entire population was subject to provider :math:`k`'s specific effect level.

2.4. Hypothesis Testing for Provider Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To formally assess whether a provider's performance is significantly different from a benchmark, we test the null hypothesis :math:`H_0: \gamma_i = \gamma_0` against an alternative :math:`H_1` (e.g., :math:`\gamma_i \neq \gamma_0`, :math:`\gamma_i > \gamma_0`, or :math:`\gamma_i < \gamma_0`). Several test methods are implemented:

*   **Wald Test:** This test relies on the asymptotic normality of the MLE :math:`\hat{\gamma}_i`. The test statistic is:

    .. math::

       T_W = \frac{\hat{\gamma}_i - \gamma_0}{\widehat{\text{se}}(\hat{\gamma}_i)}

    where :math:`\widehat{\text{se}}(\hat{\gamma}_i)` is the estimated standard error obtained from the square root of the corresponding diagonal element of the inverse Fisher information matrix :math:`[I(\hat{\boldsymbol{\theta}})]^{-1}`. Under :math:`H_0`, :math:`T_W` asymptotically follows a standard Normal distribution, or often approximated by a t-distribution with :math:`N - m - p` degrees of freedom (:math:`N = \sum n_i`) in practice.
    
    *   *Caveat:* This test can be unreliable for providers where :math:`\hat{\gamma}_i` is poorly estimated (e.g., small :math:`n_i`) or infinite (due to separation), as the standard error estimate may be inaccurate or zero.

*   **Score Test (Modified):** This test evaluates the score function :math:`U(\gamma_i)` at the null value :math:`\gamma_0`, using the MLE :math:`\hat{\boldsymbol{\beta}}` from the full model. The statistic is based on the standardized score under the null:

    .. math::

       T_S = \frac{U(\gamma_i)|_{\gamma_i=\gamma_0, \boldsymbol\beta=\hat{\boldsymbol{\beta}}}}{{\sqrt{I_{\gamma\gamma, ii}|_{\gamma_i=\gamma_0, \boldsymbol\beta=\hat{\boldsymbol{\beta}}}}}} = \frac{\sum_{j=1}^{n_i} (Y_{ij} - p_{ij}(\gamma_0, \hat{\boldsymbol{\beta}}))}{\sqrt{\sum_{j=1}^{n_i} p_{ij}(\gamma_0, \hat{\boldsymbol{\beta}})(1-p_{ij}(\gamma_0, \hat{\boldsymbol{\beta}}))}}

    where :math:`p_{ij}(\gamma_0, \hat{\boldsymbol{\beta}})` is the probability calculated under :math:`H_0`. Under :math:`H_0`, :math:`T_S` asymptotically follows a standard Normal distribution. This "modified" version avoids refitting the model under the restriction :math:`\gamma_i = \gamma_0`.

*   **Exact Poisson-Binomial Test:** This test leverages the exact distribution of the observed count :math:`O_i = \sum_{j=1}^{n_i} Y_{ij}` under the null hypothesis :math:`H_0: \gamma_i = \gamma_0`. Conditional on the covariates :math:`\mathbf{X}_{i1}, \dots, \mathbf{X}_{in_i}` and :math:`\hat{\boldsymbol{\beta}}`, the :math:`Y_{ij}` are independent Bernoulli trials with potentially different success probabilities :math:`p_{ij}(\gamma_0, \hat{\boldsymbol{\beta}})`. Therefore, the sum :math:`O_i` follows a Poisson-Binomial distribution. The p-value is calculated by summing the probabilities of outcomes as extreme or more extreme than the observed :math:`O_i` under this distribution:
    
    *   :math:`H_1: \gamma_i > \gamma_0`: :math:`P(S \ge O_i | H_0)`
    *   :math:`H_1: \gamma_i < \gamma_0`: :math:`P(S \le O_i | H_0)`
    *   :math:`H_1: \gamma_i \neq \gamma_0`: :math:`2 \times \min(P(S \ge O_i | H_0), P(S \le O_i | H_0))` (or similar definition for two-sided exact tests).

    The implementation uses efficient algorithms (e.g., FFT-based methods available in libraries like ``poibin``) to compute the Poisson-Binomial PMF/CDF. This test is preferred when asymptotic approximations may be poor.

*   **Exact Bootstrap Test:** This provides an alternative exact test by simulating the null distribution.

      * For a large number of bootstrap replicates :math:`B` (e.g., 10,000):
        a. For each subject :math:`j` in provider :math:`i`, simulate an outcome :math:`Y_{ij}^{(b)} \sim \text{Bernoulli}(p_{ij}(\gamma_0, \hat{\boldsymbol{\beta}}))`.
        b. Calculate the simulated sum :math:`O_i^{(b)} = \sum_{j=1}^{n_i} Y_{ij}^{(b)}`.
      *  The p-value is estimated as the proportion of simulated sums :math:`O_i^{(b)}` that are as extreme or more extreme than the actually observed sum :math:`O_i`, according to the alternative hypothesis. For example, for :math:`H_1: \gamma_i > \gamma_0`, the p-value is estimated by :math:`(\sum_{b=1}^B \mathbb{I}(O_i^{(b)} \ge O_i)) / B`.

2.5. Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^

Confidence intervals provide a range of plausible values for the estimated parameters.

**2.5.1. Confidence Intervals for Provider Effects** (:math:`\gamma_i`)

*   **Wald Interval:** Based on the asymptotic normality of :math:`\hat{\gamma}_i`:

    .. math::

       \hat{\gamma}_i \pm z_{1-\alpha/2} \times \widehat{\text{se}}(\hat{\gamma}_i)

    where :math:`z_{1-\alpha/2}` is the :math:`(1-\alpha/2)` quantile of the standard Normal distribution (or a t-distribution quantile). This is computationally simple but shares the limitations of the Wald test.

*   **Score Interval:** Obtained by inverting the score test. It finds the set of :math:`\gamma_0` values for which the score test statistic :math:`T_S` falls within the acceptance region :math:`[-z_{1-\alpha/2}, z_{1-\alpha/2}]`. This involves numerically solving equations like :math:`T_S(\gamma_0) = \pm z_{1-\alpha/2}` for :math:`\gamma_0`.

*   **Exact (Poisson-Binomial) Interval:** Found by inverting the exact Poisson-Binomial test (analogous to the Clopper-Pearson interval for a binomial proportion). It identifies the range of :math:`\gamma_0` values such that the observed :math:`O_i` is not statistically significant at level :math:`\alpha`. For a two-sided interval :math:`[\gamma_{L}, \gamma_{U}]`, :math:`\gamma_L` is found such that :math:`P(S \ge O_i | \gamma_L, \hat{\boldsymbol{\beta}}) = \alpha/2`, and :math:`\gamma_U` is found such that :math:`P(S \le O_i | \gamma_U, \hat{\boldsymbol{\beta}}) = \alpha/2`. This requires root-finding algorithms.

**2.5.2. Confidence Intervals for Standardized Measures**

Confidence intervals for standardized measures like ISR or DSR can be derived by transforming the confidence interval for the corresponding :math:`\gamma_i` (or :math:`\gamma_k`). Let :math:`[\gamma_{L}, \gamma_{U}]` be a :math:`100(1-\alpha)\%` confidence interval for :math:`\gamma_i`.

*   **ISR Interval:** Assuming a monotonic relationship between :math:`\gamma_i` and :math:`E_i(\gamma_i) = \sum_j p_{ij}(\gamma_i, \hat{\boldsymbol{\beta}})`, the interval for :math:`\text{ISR}_i = O_i / E_i(\gamma_0)` can be approximated by transforming the bounds of the *expected count* derived from the gamma interval:

    .. math::

       \text{CI}(\text{ISR}_i) \approx \left[ \frac{E_i(\gamma_{L})}{E_i(\gamma_0)}, \frac{E_i(\gamma_{U})}{E_i(\gamma_0)} \right]

    *Note: The observed count :math:`O_i` is fixed; the uncertainty comes from the estimation of the expected counts under different plausible values of :math:`\gamma_i`. The interval bounds are derived from :math:`E_i(\gamma_L)` and :math:`E_i(\gamma_U)` relative to the null expectation :math:`E_i(\gamma_0)`.*

*   **DSR Interval:** Similarly, the interval for :math:`\text{DSR}_k = E^{(k)} / O` is approximated by:

    .. math::

       \text{CI}(\text{DSR}_k) \approx \left[ \frac{E^{(k)}(\gamma_{L})}{O}, \frac{E^{(k)}(\gamma_{U})}{O} \right]

    where :math:`E^{(k)}(\gamma)` denotes the direct expected count calculated using :math:`\gamma` instead of :math:`\hat{\gamma}_k`.

*   **Rate Intervals:** Intervals for standardized rates are obtained by scaling the corresponding ratio intervals by the overall event rate :math:`\bar{Y}`.

2.6. Visualization
^^^^^^^^^^^^^^^^^^

Visualizations are essential for interpreting provider profiling results.

*   **Caterpillar Plot:** Displays the point estimate (e.g., :math:`\hat{\gamma}_i`, ISR:math:`_i`, or DSR:math:`_k`) and its confidence interval for each provider, typically sorted by the estimate. This allows for easy visual comparison of performance and uncertainty across providers. Points can be color-coded based on statistical significance flags from hypothesis tests.

*   **Funnel Plot:** Plots a measure of performance (e.g., ISR:math:`_i`) against a measure of precision (e.g., :math:`E_i(\gamma_0)` or :math:`E_i(\gamma_0)^2 / \text{Var}(O_i|H_0)`). Control limits, often based on the expected variation under the null hypothesis (e.g., derived from score or exact tests), form a funnel shape. Providers falling outside the funnel limits are potential outliers. This plot helps distinguish statistical variation from potentially meaningful differences in performance, accounting for provider size/volume.

3. Implementation and Usage
---------------------------

The methods described above are implemented in the ``LogisticFixedEffectModel`` class. This section provides examples of its usage.

3.1. Initialization and Fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instantiate the model and fit it to the data using the ``.fit()`` method.

.. code-block:: python

   # Assuming data_df is prepared as in Section 2.1 example
  from pprof_py.logistic_fixed_effect import LogisticFixedEffectModel

   model = LogisticFixedEffectModel(algorithm='Serbin', cutoff=5)

   model.fit(
       X=data_df,
       y_var='Outcome',
       x_vars=['Covariate1', 'Covariate2', 'Covariate3'],
       group_var='ProviderID',
       max_iter=500,
       tol=1e-4
   )

   print("Fit complete.")

3.2. Accessing Results
^^^^^^^^^^^^^^^^^^^^^^
After fitting, access estimated parameters, variances, and fit statistics via attributes:

.. code-block:: python

   # Coefficients
   betas = model.coefficients_['beta']
   gammas = model.coefficients_['gamma']
   print(f"Estimated Beta coefficients: {betas}")
   # print(f"First 5 Gamma estimates: {gammas[:5]}") # Can be long

   # Variances
   var_beta = model.variances_['beta'] # VCV matrix for beta
   var_gamma = model.variances_['gamma'] # Variances for gamma
   print(f"Beta SEs: {np.sqrt(np.diag(var_beta))}")
   # print(f"First 5 Gamma SEs: {np.sqrt(var_gamma[:5])}")

   # Fit statistics
   print(f"AIC: {model.aic_:.2f}")
   print(f"BIC: {model.bic_:.2f}")
   print(f"AUC: {model.auc_:.3f}")

   # Group information
   # print(f"Groups: {model.groups_[:5]}")
   # print(f"Group Sizes: {model.group_sizes_[:5]}")

3.3. Prediction
^^^^^^^^^^^^^^^
Generate predicted probabilities for new or existing data.

.. code-block:: python

   # Predict probabilities on the training data
   predictions = model.predict(
       X=data_df,
       x_vars=['Covariate1', 'Covariate2', 'Covariate3'],
       group_var='ProviderID'
   )
   print(f"First 5 predictions: {predictions[:5]}")

3.4. Standardized Measures Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calculate ISR, DSR, and corresponding rates using ``.calculate_standardized_measures()``.

.. code-block:: python

   # Calculate Indirect Standardized Ratio and Rate vs median
   sm_indirect = model.calculate_standardized_measures(
       stdz='indirect',
       null='median' # Use median gamma as baseline
   )
   print("\n--- Indirect Measures (vs Median) ---")
   print(sm_indirect['indirect'].head()) # Access the DataFrame

3.5. Hypothesis Testing (.test())
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Test provider effects (:math:`\gamma_i`) against a null value using various methods.

.. code-block:: python

   # Test providers vs median gamma using exact test (default)
   test_results_exact = model.test(
       null='median',
       level=0.95,
       test_method='poibin_exact',
       alternative='two_sided'
   )
   print("\n--- Provider Test (Exact vs Median) ---")
   print(test_results_exact.head())

   # Test vs gamma=0 using score test for specific providers
   # test_results_score = model.test(
   #     providers=['Group_1', 'Group_2'],
   #     null=0.0,
   #     level=0.95,
   #     test_method='score',
   #     alternative='two_sided'
   # )
   # print("\n--- Provider Test (Score vs 0) ---")
   # print(test_results_score)

The output includes flags indicating significance (-1: lower, 0: expected, 1: higher).

3.6. Confidence Interval Calculation (.calculate_confidence_intervals())
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Compute CIs for provider effects (:math:`\gamma`) or standardized measures.

.. code-block:: python

   # Get 95% Wald CIs for gamma
   gamma_cis = model.calculate_confidence_intervals(
       option='gamma',
       level=0.95,
       test_method='wald',
       alternative='two_sided' # Must be 'two_sided' for option='gamma'
   )
   print("\n--- Gamma CIs (Wald) ---")
   print(gamma_cis['gamma_ci'].head())

   # Get 90% CIs for Indirect Ratio, based on transforming 'exact' gamma CIs
   isr_cis = model.calculate_confidence_intervals(
       option='SM',
       stdz='indirect',
       measure='ratio',
       level=0.90,
       test_method='exact', # Base gamma CI method
       null='median',
       alternative='two_sided' # Base gamma CIs must be two-sided
   )
   print("\n--- Indirect Ratio CIs (based on Exact) ---")
   # Access the DataFrame using the key 'indirect_ratio'
   # Check column names, e.g., 'ci_ratio_lower', 'ci_ratio_upper'
   print(isr_cis['indirect_ratio'].head())

3.7. Visualization
^^^^^^^^^^^^^^^^^^
Use the plotting methods to visualize results.

.. code-block:: python

   # Ensure matplotlib is installed: pip install matplotlib
   # Ensure poibin is installed for exact methods: pip install poibin

   # Funnel plot using score test limits
   # model.plot_funnel(test_method='score', alpha=0.05, target=1.0)

   # Caterpillar plot for provider effects (gamma) using Wald CIs
   # model.plot_provider_effects(level=0.95, test_method='wald', use_flags=True)

   # Caterpillar plot for Indirect Standardized Ratio using Score-based CIs
   # model.plot_standardized_measures(
   #     stdz='indirect', measure='ratio', level=0.95, test_method='score', use_flags=True
   # )

   # Forest plot for covariate effects (beta)
   # model.plot_coefficient_forest(level=0.95)

(Note: Plotting examples are commented out as they require matplotlib to run and display output.)

4. Discussion
-------------

The logistic fixed effects model offers a robust approach to provider profiling, particularly valuable when potential confounding between provider characteristics and patient case mix is a concern. By directly estimating provider-specific intercepts (:math:`\gamma_i`), the model effectively controls for all time-invariant provider attributes, whether observed or unobserved. This contrasts with random effects models, which rely on the often-untested assumption that provider effects are uncorrelated with patient covariates. While RE models may offer efficiency gains under specific conditions (large :math:`m`, small :math:`n_i`, and no confounding), the FE approach prioritizes unbiased estimation of provider effects, which is critical for high-stakes applications like public reporting or pay-for-performance :cite:`Kalbfleisch2013Monitoring`.

Our implementation provides several computationally efficient algorithms (Serbin, Ban) that scale well even with a large number of providers, overcoming limitations of standard GLM software. Furthermore, the inclusion of various hypothesis testing methods (Wald, Score, Exact Poisson-Binomial, Bootstrap) allows users to choose the most appropriate inferential tool based on their data characteristics and assumptions. The exact methods, particularly the Poisson-Binomial test, are recommended when asymptotic approximations underlying Wald and Score tests may be inadequate, such as with small providers or low event rates.

Standardized measures (ISR, DSR) facilitate meaningful comparisons by adjusting for case mix and presenting performance relative to a benchmark (e.g., the median provider). Visualizations like caterpillar plots and funnel plots are crucial for interpreting these results. Caterpillar plots effectively display the estimate and uncertainty for each provider, while funnel plots help distinguish random variation from statistically significant deviations, particularly accounting for provider volume or precision. The precision measure used in our funnel plot (:math:`E_i^2 / \text{Var}(O_i | H_0)`) appropriately reflects the information content for each provider under the null hypothesis.

**Limitations and Considerations**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Incidental Parameters Problem**: In FE models, when the number of groups (:math:`m`) grows large while the group size (:math:`n_i`) remains small and fixed, the MLEs for the common parameters (:math:`\beta`) can be inconsistent :cite:`Neyman1948Consistent`. However, for logistic regression, the inconsistency is typically small, and :math:`\hat{\beta}` remains consistent if :math:`n_i \rightarrow \infty` :cite:`Greene2004Behaviour`. In many provider profiling scenarios where :math:`n_i` is reasonably large, this is less of a concern.

**Separation**: Like standard logistic regression, the FE model can suffer from separation (perfect prediction) or quasi-separation, especially within smaller groups or those with zero or all events. This can lead to infinite estimates for some :math:`\gamma_i`. Our implementation includes bounding of :math:`\gamma_i` during optimization to mitigate numerical issues, but users should be aware of providers exhibiting such patterns (e.g., via the ``log_event_providers`` option during data preparation). The Wald test and associated CIs are particularly unreliable in cases of separation. Exact and score-based methods are generally more robust in these situations.

**Computational Intensity**: While the implemented algorithms are efficient, fitting models with extremely large datasets (:math:`N`) and a very large number of providers (:math:`m`) can still require significant computational resources. Exact bootstrap tests are particularly time-consuming.

**Interpretation of** :math:`\gamma_i`: The fixed effects absorb all time-invariant provider characteristics. Therefore, the effects of specific, measured provider-level variables cannot be estimated directly within this model.

**Future Directions**
^^^^^^^^^^^^^^^^^

Potential extensions could include incorporating time-varying covariates, handling different types of outcomes (e.g., count data with Poisson FE models), implementing methods for dynamic profiling over time, and exploring alternative estimation techniques like conditional likelihood approaches for logistic FE models, which can sometimes provide consistent estimates for :math:`\beta` even when :math:`n_i` is small.

5. Conclusion
----------

The ``LogisticFixedEffectModel`` provides a robust and computationally efficient tool for healthcare provider profiling and similar analyses involving clustered binary data. By implementing the fixed effects approach, it avoids strong assumptions about the correlation between group effects and covariates, offering unbiased comparisons critical for fair evaluation. The package integrates efficient estimation algorithms, appropriate standardized measures, a suite of hypothesis tests (including exact methods), confidence interval calculations, and informative visualizations (caterpillar and funnel plots). This comprehensive toolkit empowers researchers and analysts to conduct rigorous provider profiling, identify performance outliers, and ultimately contribute to improving healthcare quality.


References
----------

.. bibliography:: references.bib
   :list: enumerate
   :filter: docname in docnames