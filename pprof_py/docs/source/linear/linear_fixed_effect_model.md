(linear_fixed_effect_model_stats)=
# Linear Fixed Effect Modeling
   

## 1. Introduction

Evaluating and comparing the performance of healthcare providers is essential for quality improvement initiatives, cost management, and informed patient decision-making {cite}`Shahian2011Public,Krumholz2013Administrative`. Provider performance is often assessed using quantitative patient outcomes, such as length of stay, cost of care, or clinical measurements like blood pressure or estimated glomerular filtration rate (eGFR). However, direct comparison of raw outcomes can be misleading due to differences in patient populations served by different providers. Robust risk adjustment methods are therefore necessary to account for patient case mix before drawing conclusions about provider performance {cite}`Iezzoni2001Risk`.

Linear regression models are frequently used for risk adjustment when the outcome of interest is quantitative. When data are clustered within providers, incorporating provider-specific effects is crucial. Similar to binary outcomes, two main approaches exist: random effects (RE) and fixed effects (FE) models. RE models assume provider effects follow a distribution and can be efficient but rest on the strong assumption that provider effects are uncorrelated with patient characteristics {cite}`Neuhaus1991Comparison`. Fixed effects models treat each provider effect as a distinct parameter, offering robustness against potential confounding between provider effects and patient covariates {cite}`Kalbfleisch2013Monitoring`. This robustness makes the FE approach particularly suitable for provider profiling where unbiased estimation of individual provider performance relative to peers is paramount.

Fitting FE models, especially linear ones, can be computationally streamlined compared to their logistic counterparts, but still requires careful implementation. This paper details the statistical methodology implemented in our software for fitting linear fixed effect models. We focus on:

  * The linear fixed effects model formulation.
  * Estimation via Ordinary Least Squares (OLS) using within-group transformations.
  * Calculation of standardized measures (Indirect and Direct Standardized Differences) for performance comparison.
  * Hypothesis testing procedures (t-tests) for identifying providers with performance significantly different from a benchmark.
  * Construction of corresponding confidence intervals for provider effects and standardized measures.
  * Visualization tools for interpreting results.

## 2. Methods

### 2.1. The Linear Fixed Effects Model

Let $Y_{ij}$ denote a quantitative outcome (e.g., cost, length of stay, blood pressure) for subject $j$ ($j = 1, \ldots, n_i$) within provider $i$ ($i = 1, \ldots, m$). Let $\mathbf{X}_{ij}$ be a $p \times 1$ vector of subject-level covariates. The linear fixed effects model is specified as:

$$
Y_{ij} = \gamma_i + \mathbf{X}_{ij}^\top\boldsymbol\beta + \epsilon_{ij}
$$

where:

* $\gamma_i$ is the fixed effect (intercept) for provider $i$. It represents the expected outcome for provider $i$ when $\mathbf{X}_{ij} = \mathbf{0}$.
* $\boldsymbol\beta$ is the $p \times 1$ vector of regression coefficients for the covariates. $\beta_k$ represents the change in the expected outcome for a one-unit increase in the $k$-th covariate, holding the provider fixed.
* $\epsilon_{ij}$ is the random error term for subject $j$ in provider $i$, typically assumed to be independent and identically distributed with $E[\epsilon_{ij}] = 0$ and $\text{Var}(\epsilon_{ij}) = \sigma^2$.

The model aims to estimate $\boldsymbol{\gamma} = (\gamma_1, \dots, \gamma_m)^\top$, $\boldsymbol\beta$, and $\sigma^2$.

### 2.2. Parameter Estimation

The parameters are estimated using Ordinary Least Squares (OLS) by minimizing the sum of squared residuals:

$$
S(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \sum_{i=1}^m \sum_{j=1}^{n_i} (Y_{ij} - \gamma_i - \mathbf{X}_{ij}^\top\boldsymbol\beta)^2
$$

This minimization can be performed efficiently using the within-group transformation (also known as demeaning or fixed effects transformation). Let $\bar{Y}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} Y_{ij}$ and $\bar{\mathbf{X}}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} \mathbf{X}_{ij}$ be the provider-specific means. Averaging the model equation within each provider $i$ gives:

$$
\bar{Y}_i = \gamma_i + \bar{\mathbf{X}}_i^\top\boldsymbol\beta + \bar{\epsilon}_i
$$

Subtracting the mean equation from the original equation yields the within-group transformed model:

$$
(Y_{ij} - \bar{Y}_i) = (\mathbf{X}_{ij} - \bar{\mathbf{X}}_i)^\top\boldsymbol\beta + (\epsilon_{ij} - \bar{\epsilon}_i)
$$

$$
\tilde{Y}_{ij} = \tilde{\mathbf{X}}_{ij}^\top\boldsymbol\beta + \tilde{\epsilon}_{ij}
$$

where $\tilde{Y}_{ij}$, $\tilde{\mathbf{X}}_{ij}$, and $\tilde{\epsilon}_{ij}$ represent the demeaned variables. Crucially, the provider fixed effect $\gamma_i$ is eliminated by this transformation.

The OLS estimator for $\boldsymbol\beta$ is obtained by regressing $\tilde{Y}_{ij}$ on $\tilde{\mathbf{X}}_{ij}$:

$$
\hat{\boldsymbol\beta}_{FE} = \left( \sum_{i=1}^m \sum_{j=1}^{n_i} \tilde{\mathbf{X}}_{ij} \tilde{\mathbf{X}}_{ij}^\top \right)^{-1} \left( \sum_{i=1}^m \sum_{j=1}^{n_i} \tilde{\mathbf{X}}_{ij} \tilde{Y}_{ij} \right)
$$

This can be expressed using the projection matrix $\mathbf{Q}_i = \mathbf{I}_{n_i} - \frac{1}{n_i}\mathbf{1}_{n_i}\mathbf{1}_{n_i}^\top$, which demeans data within group $i$. Let $\mathbf{Y}_i$ and $\mathbf{X}_i$ be the stacked outcome vector and covariate matrix for group $i$. Then $\tilde{\mathbf{Y}}_i = \mathbf{Q}_i \mathbf{Y}_i$ and $\tilde{\mathbf{X}}_i = \mathbf{Q}_i \mathbf{X}_i$. The estimator becomes:

$$
\hat{\boldsymbol\beta}_{FE} = \left( \sum_{i=1}^m \mathbf{X}_i^\top \mathbf{Q}_i \mathbf{X}_i \right)^{-1} \left( \sum_{i=1}^m \mathbf{X}_i^\top \mathbf{Q}_i \mathbf{Y}_i \right)
$$

Once $\hat{\boldsymbol\beta}_{FE}$ is obtained, the fixed effects $\gamma_i$ are estimated using the relationship derived from the mean equation:

$$
\hat{\gamma}_i = \bar{Y}_i - \bar{\mathbf{X}}_i^\top\hat{\boldsymbol\beta}_{FE}
$$

The variance of the error term, $\sigma^2$, is estimated using the residuals from the full model, $e_{ij} = Y_{ij} - \hat{\gamma}_i - \mathbf{X}_{ij}^\top\hat{\boldsymbol\beta}_{FE}$:

$$
\hat{\sigma}^2 = \frac{\sum_{i=1}^m \sum_{j=1}^{n_i} e_{ij}^2}{N - m - p}
$$

where $N = \sum n_i$ is the total sample size. The denominator reflects the degrees of freedom used for estimating $m$ fixed effects and $p$ covariate coefficients.

The variance-covariance matrix for $\hat{\boldsymbol\beta}_{FE}$ is:

$$
\text{Var}(\hat{\boldsymbol\beta}_{FE}) = \hat{\sigma}^2 \left( \sum_{i=1}^m \mathbf{X}_i^\top \mathbf{Q}_i \mathbf{X}_i \right)^{-1}
$$

The variance for $\hat{\gamma}_i$ is given by:

$$
\text{Var}(\hat{\gamma}_i) = \text{Var}(\bar{Y}_i - \bar{\mathbf{X}}_i^\top\hat{\boldsymbol\beta}_{FE}) = \text{Var}(\bar{\epsilon}_i - \bar{\mathbf{X}}_i^\top(\hat{\boldsymbol\beta}_{FE} - \boldsymbol\beta))
$$

$$
\text{Var}(\hat{\gamma}_i) = \frac{\sigma^2}{n_i} + \bar{\mathbf{X}}_i^\top \text{Var}(\hat{\boldsymbol\beta}_{FE}) \bar{\mathbf{X}}_i
$$

The implementation (`LinearFixedEffectModel.__init__`) provides two options via the `gamma_var_option` parameter for estimating the diagonal elements of $\text{Var}(\hat{\boldsymbol\gamma})$:

1. **`complete`:** Uses the full formula above, $\text{Var}(\hat{\gamma}_i) = \frac{\hat{\sigma}^2}{n_i} + \bar{\mathbf{X}}_i^\top \widehat{\text{Var}}(\hat{\boldsymbol\beta}_{FE}) \bar{\mathbf{X}}_i$, including the term involving $\widehat{\text{Var}}(\hat{\boldsymbol\beta}_{FE})$.
2. **`simplified`:** Uses only the first term, $\text{Var}(\hat{\gamma}_i) = \frac{\hat{\sigma}^2}{n_i}$, ignoring the uncertainty in $\hat{\boldsymbol\beta}_{FE}$. This may be appropriate if $p$ is small relative to $N$ or if primary interest is in ranking rather than precise inference.


### 2.3. Standardized Measures for Performance Comparison

For linear models, standardized measures typically represent differences rather than ratios. They quantify how much a provider's total or average outcome differs from what would be expected under a baseline scenario, after adjusting for case mix.

Let $\hat{\boldsymbol{\beta}}$ and $\hat{\boldsymbol{\gamma}}$ be the OLS estimates. Define a reference or baseline provider effect $\gamma_0$ (e.g., median or mean of $\hat{\gamma}_i$, as specified by the `null` parameter in `LinearFixedEffectModel.calculate_standardized_measures`).

#### 2.3.1. Indirect Standardization

Indirect standardization compares the observed total outcome for a provider to the expected total outcome if that provider performed at the baseline level $\gamma_0$, given its specific patient mix.

Observed Total Outcome for Provider $i$ ($O_i$)

The sum of actual outcomes for all $n_i$ subjects in provider $i$. This corresponds to the `observed` column in the output DataFrame for indirect standardization.

$$
O_i = \sum_{j=1}^{n_i} Y_{ij}
$$

Expected Total Outcome for Provider $i$ under Baseline ($E_i(\gamma_0)$)

The sum of expected outcomes for provider $i$'s $n_i$ subjects, if the provider effect was the baseline $\gamma_0$, adjusted for their specific covariates $\mathbf{X}_{ij}$. This corresponds to the `expected` column in the output DataFrame for indirect standardization.

$$
E_i(\gamma_0) = \sum_{j=1}^{n_i} \left( \gamma_0 + \mathbf{X}_{ij}^\top \hat{\boldsymbol{\beta}} \right)
$$

Indirect Standardized Difference for Provider $i$ ($\text{ISDiff}_i$)

The average difference between observed and expected outcomes for provider $i$. This is calculated as the total observed outcome minus the total expected outcome, divided by the number of subjects in the provider ($n_i$). This corresponds to the `indirect_difference` column in the output DataFrame.

$$
\text{ISDiff}_i = \frac{O_i - E_i(\gamma_0)}{n_i}
$$

This difference can also be expressed as the difference between the observed mean, $\bar{Y}_i = O_i / n_i$, and the expected mean for provider $i$ under baseline effect $\gamma_0$, $\bar{E}_i(\gamma_0) = E_i(\gamma_0) / n_i$:

$$
\text{ISDiff}_i = \bar{Y}_i - \left( \gamma_0 + \bar{\mathbf{X}}_i^\top \hat{\boldsymbol{\beta}} \right)
$$

where $\bar{\mathbf{X}}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} \mathbf{X}_{ij}$ is the mean covariate vector for provider $i$.

#### 2.3.2. Direct Standardization

Direct standardization compares the expected total outcome if the *entire population* experienced provider $k$'s effect ($\hat{\gamma}_k$) to the expected total outcome if the entire population experienced the baseline effect ($\gamma_0$).

Expected Total Outcome under Provider $k$'s Effect ($E^{(k)}$)

The total expected outcome for the entire population if all subjects experienced provider $k$'s effect ($\hat{\gamma}_k$), adjusted for their specific covariates.

$$
E^{(k)} = \sum_{i=1}^m \sum_{j=1}^{n_i} \left( \hat{\gamma}_k + \mathbf{X}_{ij}^\top \hat{\boldsymbol{\beta}} \right)
$$

Expected Total Outcome under Baseline Effect ($E^{(0)}$)

The total expected outcome for the entire population if all subjects experienced the baseline effect ($\gamma_0$), adjusted for their specific covariates.

$$
E^{(0)} = \sum_{i=1}^m \sum_{j=1}^{n_i} \left( \gamma_0 + \mathbf{X}_{ij}^\top \hat{\boldsymbol{\beta}} \right)
$$

Direct Standardized Difference for Provider $k$ ($\text{DSDiff}_k$)

The average difference between the total expected outcomes under provider $k$'s effect and the baseline effect, divided by the total sample size ($N = \sum_{i=1}^m n_i$). This corresponds to the `direct_difference` column in the output DataFrame.

$$
\text{DSDiff}_k = \frac{E^{(k)} - E^{(0)}}{N}
$$

This difference can also be expressed as the difference between the expected mean outcome under provider $k$'s effect ($\bar{E}^{(k)} = E^{(k)} / N$) and the expected mean outcome under the baseline effect ($\bar{E}^{(0)} = E^{(0)} / N$):

$$
\text{DSDiff}_k = \hat{\gamma}_k - \gamma_0
$$

Therefore, for linear fixed effect models, both indirect and direct standardized differences ultimately simplify to $\hat{\gamma}_i - \gamma_0$ when expressed on a per-subject basis. However, the calculations differ in how they aggregate observed and expected outcomes (by group size for indirect, and by total sample size for direct). The implementation (`LinearFixedEffectModel.calculate_standardized_measures`) calculates and returns these differences along with the observed and expected totals for each method.

### 2.4. Hypothesis Testing for Provider Effects

We test the null hypothesis $H_0: \gamma_i = \gamma_0$ against an alternative $H_1$. Given the linear model assumptions and the estimation of $\sigma^2$, the natural test is a t-test, as implemented in `LinearFixedEffectModel.test`:

$$
T_i = \frac{\hat{\gamma}_i - \gamma_0}{\widehat{\operatorname{se}}(\hat{\gamma}_i)}
$$

Under $H_0$, the test statistic $T_i$ follows a t-distribution with $N - m - p$ degrees of freedom. P-values are calculated from this distribution according to the specified `alternative` (`'two_sided'`, `'less'`, or `'greater'`).

### 2.5. Confidence Intervals

Confidence intervals for $\boldsymbol{\beta}$ and $\gamma_i$ are constructed using the t-distribution with $N - m - p$ degrees of freedom.

For $\beta_k$ (see `LinearFixedEffectModel.summary`), the confidence interval is

$$
\hat{\beta}_k \pm t_{1-\alpha/2,\, N-m-p} \times \widehat{\operatorname{se}}(\hat{\beta}_k)
$$

where $\widehat{\operatorname{se}}(\hat{\beta}_k)$ is the square root of the $k$-th diagonal element of $\widehat{\operatorname{Var}}(\hat{\boldsymbol{\beta}}_{FE})$.

For $\gamma_i$ (see `LinearFixedEffectModel.calculate_confidence_intervals`), the confidence interval is

$$
\hat{\gamma}_i \pm t_{1-\alpha/2,\, N-m-p} \times \widehat{\operatorname{se}}(\hat{\gamma}_i)
$$

where $\widehat{\operatorname{se}}(\hat{\gamma}_i)$ is the square root of the estimated $\operatorname{Var}(\hat{\gamma}_i)$ using either the `complete` or `simplified` option.

For standardized differences (see `LinearFixedEffectModel.calculate_confidence_intervals`), both indirect and direct standardized differences simplify to $\hat{\gamma}_i - \gamma_0$, where $\gamma_0$ is treated as a fixed value after estimation for confidence-interval construction. The confidence interval is

$$
(\hat{\gamma}_i - \gamma_0) \pm t_{1-\alpha/2,\, N-m-p} \times \widehat{\operatorname{se}}(\hat{\gamma}_i)
$$

This is equivalent to shifting the confidence interval for $\hat{\gamma}_i$ by $-\gamma_0$.

### 2.6. Visualization

The `LinearFixedEffectModel` class provides several plotting methods:

- **Caterpillar Plot (`plot_provider_effects`, `plot_standardized_measures`):** Displays $\hat{\gamma}_i$ (or the standardized difference $\hat{\gamma}_i - \gamma_0$) with its confidence interval for each provider, sorted by the estimate. Helps visualize relative performance and uncertainty.
- **Funnel Plot (`plot_funnel`):** Plots the standardized difference ($\hat{\gamma}_i - \gamma_0$) against group size $n_i$ (as a measure of precision). Control limits are drawn based on the overall model's residual standard deviation $\hat{\sigma}$, typically as $target \pm z_{1-\alpha/2} \times \frac{\hat{\sigma}}{\sqrt{n_i}}$, forming a funnel shape.
- **Coefficient Forest Plot (`plot_coefficient_forest`):** Displays estimates and confidence intervals for covariate coefficients $\hat{\boldsymbol{\beta}}$.
- **Residual Plots (`plot_residuals`, `plot_qq`):** Standard diagnostic plots for model assessment, such as residuals vs. fitted values and Q-Q plot of residuals.

## 3. Implementation and Usage

The `LinearFixedEffectModel` class in `pprof_py` implements these methods.

### 3.1. Initialization and Fitting

```python

# Assuming data_df is a pandas DataFrame with columns:
# 'ContinuousY', 'Covariate1', 'Covariate2', 'Covariate3', 'ProviderID'
# And n_total_samples, linear_pred_true are defined elsewhere for example data generation.
import numpy as np
import pandas as pd
from pprof_py import LinearFixedEffectModel

# Example data generation (conceptual)
# n_total_samples = 1000
# data_df = pd.DataFrame({
#     'Covariate1': np.random.rand(n_total_samples),
#     'Covariate2': np.random.rand(n_total_samples),
#     'Covariate3': np.random.rand(n_total_samples),
#     'ProviderID': np.random.choice(range(10), n_total_samples)
# })
# linear_pred_true = data_df[['Covariate1', 'Covariate2', 'Covariate3']].sum(axis=1) # Example
# data_df['ContinuousY'] = linear_pred_true + np.random.normal(0, 1.0, n_total_samples)

# Initialize and fit the model
lin_model = LinearFixedEffectModel(gamma_var_option='complete')

lin_model.fit(
    X=data_df, # DataFrame containing all necessary columns
    y_var='ContinuousY',
    x_vars=['Covariate1', 'Covariate2', 'Covariate3'],
    group_var='ProviderID'
)

print("Linear FE model fitting complete.")

```

### 3.2. Accessing Results

```python

# Coefficients
betas_lin = lin_model.coefficients_['beta']
gammas_lin = lin_model.coefficients_['gamma']
print(f"Estimated Linear Beta coefficients: {betas_lin.flatten()}")

# Variances and Sigma
var_beta_lin = lin_model.variances_['beta'] # Variance-covariance matrix for beta
var_gamma_lin = lin_model.variances_['gamma'] # Diagonal matrix of variances for gamma
sigma_lin = lin_model.sigma_
print(f"Estimated Sigma (Residual SD): {sigma_lin:.3f}")
print(f"Linear Beta SEs: {np.sqrt(np.diag(var_beta_lin))}")

# Fit statistics
print(f"Linear AIC: {lin_model.aic_:.2f}")
print(f"Linear BIC: {lin_model.bic_:.2f}")

# For summary of covariate coefficients (including SE, p-value, CI)
# summary_df = lin_model.summary()
# print("\n--- Covariate Summary ---")
# print(summary_df)

```

### 3.3. Prediction

```python

# Predict outcomes
# Assuming data_df_new has the same structure for covariates and group IDs
linear_predictions = lin_model.predict(
    X=data_df, # Can be new data
    x_vars=['Covariate1', 'Covariate2', 'Covariate3'],
    group_var='ProviderID'
)
print(f"First 5 linear predictions: {linear_predictions[:5]}")

```

### 3.4. Standardized Measures Calculation

```python

# Calculate Indirect Standardized Difference vs median
sm_results_lin = lin_model.calculate_standardized_measures(
    stdz='indirect', # Can be 'direct' or ['indirect', 'direct']
    null='median'    # Can be 'mean' or a float
)
print("\n--- Linear Indirect Measures (vs Median) ---")
print(sm_results_lin['indirect'].head()) # Access the DataFrame for 'indirect' results

# If stdz=['indirect', 'direct']
# sm_direct_lin = sm_results_lin['direct']

```

### 3.5. Hypothesis Testing (`test`)
Test provider effects ($\gamma_i$) using t-tests.

```python

# Test providers vs median gamma
test_results_lin = lin_model.test(
    null='median',    # Can be 'mean' or a float
    level=0.95,
    alternative='two_sided' # Can be 'less' or 'greater'
)
print("\n--- Linear Provider Test (vs Median) ---")
print(test_results_lin.head())

```

### 3.6. Confidence Interval Calculation (`calculate_confidence_intervals`)
Compute CIs for $\gamma_i$ or standardized differences.

```python

# Get 95% CIs for gamma
gamma_cis_lin_results = lin_model.calculate_confidence_intervals(
    option='gamma',
    level=0.95,
    alternative='two_sided' # Must be two-sided for gamma option
)
print("\n--- Linear Gamma CIs ---")
print(gamma_cis_lin_results['gamma_ci'].head())

# Get 95% CIs for the Indirect Standardized Difference
isd_cis_lin_results = lin_model.calculate_confidence_intervals(
    option='SM',
    stdz='indirect', # Can be 'direct' or ['indirect', 'direct']
    level=0.95,
    null='median',
    alternative='two_sided' # Can be 'less' or 'greater'
)
print("\n--- Linear Indirect Difference CIs (vs Median) ---")
# Access the DataFrame using the key '{stdz}_ci', e.g., 'indirect_ci'
print(isd_cis_lin_results['indirect_ci'].head())

```

### 3.7. Visualization
Use plotting methods from the `LinearFixedEffectModel` instance.

```python

# Funnel plot of standardized difference vs group size
# lin_model.plot_funnel(stdz='indirect', null='median', alpha=0.05, target=0.0)

# Caterpillar plot for provider effects (gamma)
# lin_model.plot_provider_effects(level=0.95, use_flags=True, null='median')

# Caterpillar plot for Indirect Standardized Difference
# lin_model.plot_standardized_measures(
#     stdz='indirect', level=0.95, use_flags=True, null='median'
# )

# Forest plot for covariate effects (beta)
# lin_model.plot_coefficient_forest() # Defaults to 95% CI

# Residual plot
# lin_model.plot_residuals()

# Q-Q plot of residuals
# lin_model.plot_qq()

```

## 4. Discussion

The linear fixed effects model provides a straightforward and robust method for risk adjustment and provider profiling when the outcome variable is quantitative. Its primary advantage over random effects models lies in its robustness to the potential correlation between provider-specific effects and patient covariates, a common scenario in observational healthcare data. By estimating a separate intercept ($\gamma_i$) for each provider, the model effectively controls for all stable provider characteristics.

The estimation via within-group demeaning is computationally efficient and yields unbiased and consistent estimates for the covariate effects ($\boldsymbol\beta$) under standard OLS assumptions (conditional on the fixed effects). The provider effects ($\hat{\gamma}_i$) are subsequently recovered.

Standardized differences (both indirect and direct) simplify to $\hat{\gamma}_i - \gamma_0$ in the linear case, providing an easily interpretable measure of a provider's adjusted performance relative to a benchmark ($\gamma_0$) on the original outcome scale. Inference for these effects is based on standard t-tests and t-distribution confidence intervals, leveraging the estimated residual variance ($\hat{\sigma}^2$). The choice between the 'complete' and 'simplified' variance calculation for $\hat{\gamma}_i$ (via `gamma_var_option`) depends on whether the uncertainty in $\hat{\boldsymbol\beta}$ is considered relevant for the specific application.

**Limitations:**

- **Incidental Parameters:** Similar to the logistic case, if $n_i$ is small and $m$ is large, the estimates of $\gamma_i$ can have large variances, although $\hat{\boldsymbol\beta}$ remains consistent under weaker conditions than in non-linear models.
- **Provider-Level Covariates:** The FE model cannot estimate the effects of variables that are constant within providers (e.g., hospital teaching status, ownership).
- **Assumption of Linearity:** The model assumes a linear relationship between covariates and the outcome, conditional on the provider effect.
- **Homoscedasticity:** Assumes constant error variance ($\sigma^2$) across all observations. Heteroscedasticity might require robust standard errors or alternative modeling approaches.

## 5. Conclusion

The linear fixed effects model, as implemented in the `LinearFixedEffectModel`` class, offers a valuable tool for provider profiling with quantitative outcomes. It provides robust risk adjustment through provider-specific intercepts and facilitates performance comparisons using standardized differences. The implementation includes efficient estimation, standard inference procedures based on t-tests, and methods for calculating confidence intervals. When combined with appropriate visualization tools, it enables researchers and analysts to effectively evaluate and compare provider performance while accounting for patient case mix.

References
----------
```{bibliography} ../references.bib
:list: enumerate
:filter: docname in docnames
:keyprefix: linfe-
```