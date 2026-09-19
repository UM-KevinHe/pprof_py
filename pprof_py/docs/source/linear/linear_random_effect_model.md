(linear_random_effect_model_stats)=

# Linear Random Effect Modeling

```{contents}
:local:
:depth: 2
```

## 1. Introduction

When evaluating provider performance using quantitative outcomes, linear mixed-effects models, often called linear random effect (RE) models, offer an alternative to fixed effect (FE) models {cite}`Wooldridge2010Econometric`. RE models are particularly useful when we assume providers are a sample from a larger population of providers and we wish to make inferences about this population or predict effects for individual providers, potentially "borrowing strength" across providers.

Unlike FE models that estimate a distinct parameter for each provider, RE models treat provider effects as random variables drawn from a common distribution. This approach can lead to more efficient estimates, especially when the number of observations per provider is small. However, a key assumption is that the random effects are uncorrelated with the covariates in the model. If this assumption is violated, estimates of covariate effects ($\boldsymbol\beta$) can be biased {cite}`Wooldridge2010Econometric`.

This document details the statistical methodology for linear random effect models as implemented in the `LinearRandomEffectModel` class, which uses a pure-Python lme4-style (Restricted) Maximum Likelihood solver. We cover:

- The linear random effects model formulation (focusing on random intercepts).
- Parameter estimation (fixed effects, variance components) via (Restricted) Maximum Likelihood.
- Prediction of random effects (Best Linear Unbiased Predictors - BLUPs).
- Calculation of standardized measures for performance comparison.
- Hypothesis testing procedures for provider effects.
- Construction of confidence intervals for provider effects and standardized measures.
- Visualization tools for interpreting results.

## 2. Methods

### 2.1. The Linear Random Effects Model (Random Intercept)

Let $Y_{ij}$ be a quantitative outcome for subject $j$ ($j = 1, \ldots, n_i$) within provider (group) $i$ ($i = 1, \ldots, m$). Let $\mathbf{X}_{ij}$ be a $p \times 1$ vector of subject-level covariates. The linear random intercept model is:

$$
Y_{ij} = \mathbf{X}_{ij}^\top\boldsymbol\beta + u_i + \epsilon_{ij}
$$

where:

- $\boldsymbol\beta$ is the $p \times 1$ vector of fixed regression coefficients for the covariates.
- $u_i$ is the random effect for provider $i$. It represents the deviation of provider $i$'s intercept from the overall intercept (which is part of $\mathbf{X}_{ij}^\top\boldsymbol\beta$ if an intercept is included in $\mathbf{X}_{ij}$). It is assumed that $u_i \sim N(0, \sigma^2_u)$, where $\sigma^2_u$ is the variance of the provider effects.
- $\epsilon_{ij}$ is the random error term for subject $j$ in provider $i$. It is assumed that $\epsilon_{ij} \sim N(0, \sigma^2_e)$, where $\sigma^2_e$ is the residual variance.
- $u_i$ and $\epsilon_{ij}$ are assumed to be independent of each other and of $\mathbf{X}_{ij}$.

The model aims to estimate the fixed effects $\boldsymbol\beta$, the variance components $\sigma^2_u$ and $\sigma^2_e$, and to predict the random effects $u_i$.

### 2.2. Parameter Estimation and Prediction

The parameters ($\boldsymbol\beta$, $\sigma^2_u$, $\sigma^2_e$) are typically estimated using Maximum Likelihood (ML) or Restricted Maximum Likelihood (REML). REML is often preferred for estimating variance components as it accounts for the degrees of freedom used in estimating fixed effects. The `LinearRandomEffectModel` implements a pure-Python solver for this purpose, following the lme4 profiled-deviance formulation. Both REML (default) and ML are supported via the `reml` parameter.

Once the variance components are estimated, the fixed effects $\hat{\boldsymbol\beta}$ are estimated. The random effects $u_i$ are not directly estimated as parameters but are predicted using Best Linear Unbiased Predictors (BLUPs), denoted as $\hat{u}_i$. BLUPs are empirical Bayes estimates and exhibit shrinkage towards the overall mean (zero in this formulation), especially for providers with fewer observations or less precise estimates.

The BLUP for a random intercept $u_i$ is given by:

$$
\hat{u}_i = \frac{n_i \sigma^2_u}{n_i \sigma^2_u + \sigma^2_e} (\bar{Y}_i - \bar{\mathbf{X}}_i^\top\hat{\boldsymbol\beta})
$$

where $\frac{n_i \sigma^2_u}{n_i \sigma^2_u + \sigma^2_e}$ is the shrinkage factor.

The implementation stores:

- Fixed effects: `coefficients_['beta']` ($\hat{\boldsymbol\beta}$) — `pd.Series`
- Random effects (BLUPs): `coefficients_['alpha']` ($\hat{u}_i$) — `pd.Series` (single group) or `dict` of `pd.Series` (multiple groups)
- Variance-covariance of fixed effects: `variances_['beta']` ($\widehat{\text{Var}}(\hat{\boldsymbol\beta})$) — `pd.DataFrame`
- Variance of random effects: `variances_['alpha']` ($\hat{\sigma}^2_u$) — `pd.DataFrame` (single group) or `dict` (multiple groups)
- Residual standard deviation: `sigma_` ($\hat{\sigma}_e$) — `float`
- Random-effect standard deviations: `random_effect_sd_` — `dict` mapping each group variable name to $\hat{\sigma}_u$
- Log-likelihood at convergence: `loglike_`
- REML/ML objective at convergence: `objective_`

### 2.3. Standardized Measures for Performance Comparison

For linear random effects models, standardized measures quantify how much a provider's total or average outcome differs from what would be expected under a baseline scenario, after adjusting for case mix. These measures are calculated either by comparing observed outcomes to expected outcomes under a baseline random effect (indirect standardization), or by comparing expected outcomes for the entire population under each provider's random effect to those under the baseline (direct standardization).

Let $\hat{\boldsymbol{\beta}}$ denote the estimated fixed effects, and $\hat{\alpha}_i$ the estimated random effect for provider $i$. Define a reference or baseline random effect $\alpha_0$ (e.g., the median or mean of $\hat{\alpha}_i$, as specified by the `null` parameter in `LinearRandomEffectModel.calculate_standardized_measures`).

#### 2.3.1. Indirect Standardization

Indirect standardization compares the observed total outcome for a provider to the expected total outcome if that provider had the baseline random effect $\alpha_0$, given its specific patient mix.

Observed Total Outcome for Provider $i$ ($O_i$)

The sum of fitted values (including both fixed and random effects) for all $n_i$ subjects in provider $i$. This corresponds to the `observed` column in the output DataFrame for indirect standardization.

$$
O_i = \sum_{j=1}^{n_i} \left( \mathbf{X}_{ij}^\top \hat{\boldsymbol{\beta}} + \hat{\alpha}_i \right)
$$

Expected Total Outcome for Provider $i$ under Baseline ($E_i(\alpha_0)$)

The sum of expected outcomes for provider $i$'s $n_i$ subjects, if the provider effect was the baseline $\alpha_0$, adjusted for their specific covariates. This corresponds to the `expected` column in the output DataFrame for indirect standardization.

$$
E_i(\alpha_0) = \sum_{j=1}^{n_i} \left( \mathbf{X}_{ij}^\top \hat{\boldsymbol{\beta}} + \alpha_0 \right)
$$

Indirect Standardized Difference for Provider $i$ ($\text{ISDiff}_i$)

The average difference between observed and expected outcomes for provider $i$. This is calculated as the total observed outcome minus the total expected outcome, divided by the number of subjects in the provider ($n_i$). This corresponds to the `indirect_difference` column in the output DataFrame.

$$
\text{ISDiff}_i = \frac{O_i - E_i(\alpha_0)}{n_i}
$$

This difference can also be expressed as the difference between the observed mean and the expected mean for provider $i$ under the baseline effect:

$$
\text{ISDiff}_i = \bar{Y}_i^{\text{fitted}} - \left( \bar{\mathbf{X}}_i^\top \hat{\boldsymbol{\beta}} + \alpha_0 \right)
$$

where $\bar{Y}_i^{\text{fitted}}$ is the mean fitted value for provider $i$, and $\bar{\mathbf{X}}_i$ is the mean covariate vector for provider $i$.

#### 2.3.2. Direct Standardization

Direct standardization compares the expected total outcome if the _entire population_ experienced provider $k$'s random effect ($\hat{\alpha}_k$) to the expected total outcome if the entire population experienced the baseline effect ($\alpha_0$).

Expected Total Outcome under Provider $k$'s Effect ($E^{(k)}$)

The total expected outcome for the entire population if all subjects experienced provider $k$'s random effect ($\hat{\alpha}_k$), adjusted for their specific covariates.

$$
E^{(k)} = \sum_{i=1}^m \sum_{j=1}^{n_i} \left( \mathbf{X}_{ij}^\top \hat{\boldsymbol{\beta}} + \hat{\alpha}_k \right)
$$

Expected Total Outcome under Baseline Effect ($E^{(0)}$)

The total expected outcome for the entire population if all subjects experienced the baseline effect ($\alpha_0$), adjusted for their specific covariates.

$$
E^{(0)} = \sum_{i=1}^m \sum_{j=1}^{n_i} \left( \mathbf{X}_{ij}^\top \hat{\boldsymbol{\beta}} + \alpha_0 \right)
$$

Direct Standardized Difference for Provider $k$ ($\text{DSDiff}_k$)

The average difference between the total expected outcomes under provider $k$'s effect and the baseline effect, divided by the total sample size ($N = \sum_{i=1}^m n_i$). This corresponds to the `direct_difference` column in the output DataFrame.

$$
\text{DSDiff}_k = \frac{E^{(k)} - E^{(0)}}{N}
$$

This difference can also be expressed as the difference between the expected mean outcome under provider $k$'s effect and the expected mean outcome under the baseline effect:

$$
\text{DSDiff}_k = \hat{\alpha}_k - \alpha_0
$$

Therefore, for linear random effects models, both indirect and direct standardized differences ultimately simplify to $\hat{\alpha}_i - \alpha_0$ when expressed on a per-subject basis. However, the calculations differ in how they aggregate observed and expected outcomes (by group size for indirect, and by total sample size for direct). The implementation (`LinearRandomEffectModel.calculate_standardized_measures`) calculates and returns these differences along with the observed and expected totals for each method.

### 2.4. Hypothesis Testing for Provider Effects

We test the null hypothesis $H_0: u_i = u_0$ against an alternative $H_1$. The test is based on the predicted random effects (BLUPs) $\hat{u}_i$ and their standard errors. The test statistic is a Z-score:

$$
Z_i = \frac{\hat{u}_i - u_0}{\widehat{\operatorname{se}}(\hat{u}_i)}
$$

The standard error of the BLUP, $\widehat{\operatorname{se}}(\hat{u}_i)$, is derived from the posterior variance of $u_i$ given the data:

$$
\widehat{\operatorname{se}}(\hat{u}_i) = \sqrt{\frac{\hat{\sigma}^2_u}{\hat{\sigma}^2_u + \hat{\sigma}^2_e / n_i} \cdot \frac{\hat{\sigma}^2_e}{n_i}}
$$

Under $H_0$, $Z_i$ is assumed to follow a standard normal distribution. P-values are calculated from this distribution according to the specified `alternative` (`'two_sided'`, `'less'`, `'greater'`) in the `test` method.

### 2.5. Confidence Intervals

Confidence intervals are constructed for the fixed effects $\boldsymbol{\beta}$ (via `summary` method, using t-distribution) and for the provider random effects $u_i$ or standardized differences $u_i - u_0$ (via `calculate_confidence_intervals` method, using normal approximation for BLUPs).

For fixed effects $\beta_k$ (see `summary`), the t-distribution with degrees of freedom $N - p - m$ (total observations minus number of fixed effects minus number of groups) is used:

$$
\hat{\beta}_k \pm t_{1-\alpha/2,\, df} \times \widehat{\operatorname{se}}(\hat{\beta}_k)
$$

For random effects $u_i$ (BLUPs, option `'alpha'` in `calculate_confidence_intervals`), the normal approximation gives a two-sided interval:

$$
\hat{u}_i \pm z_{1-\alpha/2} \times \widehat{\operatorname{se}}(\hat{u}_i)
$$

For standardized differences $\hat{u}_i - u_0$ (option `'SM'` in `calculate_confidence_intervals`), the confidence interval for $\hat{u}_i$ is shifted by $-u_0$:

$$
(\hat{u}_i - u_0) \pm z_{1-\alpha/2} \times \widehat{\operatorname{se}}(\hat{u}_i)
$$

The implementation handles one-sided and two-sided alternatives.

### 2.6. Visualization

The `LinearRandomEffectModel` class provides several plotting methods:

- **Caterpillar Plot for Provider Effects** (`plot_provider_effects`): Displays BLUPs $\hat{u}_i$ with their confidence intervals.
- **Caterpillar Plot for Standardized Measures** (`plot_standardized_measures`): Displays standardized differences $\hat{u}_i - u_0$ with confidence intervals.
- **Funnel Plot** (`plot_funnel`): Plots standardized differences $\hat{u}_i - u_0$ against group size $n_i$. Control limits are typically based on the overall residual standard deviation $\hat{\sigma}_e$, e.g., $target \pm z_{1-\alpha/2} \times \frac{\hat{\sigma}_e}{\sqrt{n_i}}$.
- **Coefficient Forest Plot** (`plot_coefficient_forest`):\*\* Displays estimates and confidence intervals for fixed effect coefficients $\hat{\boldsymbol{\beta}}$.
- **Residual Plots** (`plot_residuals`): Standard residuals vs. fitted values plot.
- **Q-Q Plot** (`plot_qq`):\*\* Q-Q plot of residuals against a normal distribution to check normality assumption.

## 3. Implementation and Usage

The `LinearRandomEffectModel` class in `pprof_py` implements these methods.

### 3.1. Initialization and Fitting

```python

import numpy as np
import pandas as pd
from pprof_py import LinearRandomEffectModel

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
lre_model = LinearRandomEffectModel(verbose=False)
lre_model.fit(
    data_df,
    y_var='ContinuousY',
    x_vars=['Covariate1', 'Covariate2'],
    group_var='ProviderID',
    reml=True
)

```

### 3.2. Accessing Results

```python

# Fixed Effects Coefficients
fixed_effects = lre_model.coefficients_['beta']
print("Estimated Fixed Effects (Betas):")
print(fixed_effects)

# Predicted Random Effects (BLUPs)
random_effects_blups = lre_model.coefficients_['alpha']
print("\nPredicted Random Effects (BLUPs, first 5):")
print(random_effects_blups.head())

# Variance Components
fe_var_cov = lre_model.variances_['beta']
re_var = lre_model.variances_['alpha']   # Variance of random effects (sigma_u^2)
sigma_e = lre_model.sigma_               # Residual standard deviation (sigma_e)
re_sd = lre_model.random_effect_sd_      # Dict of group_var -> sigma_u
print(f"\nEstimated Variance of Random Effects (sigma_u^2): {re_var.iloc[0,0]:.3f}")
print(f"Estimated Residual Standard Deviation (sigma_e): {sigma_e:.3f}")
print(f"Random-effect SD per group variable: {re_sd}")
print(f"Log-likelihood: {lre_model.loglike_:.2f}")

# Fit statistics
print(f"\nAIC: {lre_model.aic_:.2f}")
print(f"BIC: {lre_model.bic_:.2f}")

# Summary for fixed effects
# fe_summary = lre_model.summary()
# print("\n--- Fixed Effects Summary ---")
# print(fe_summary)

```

### 3.3. Prediction

Predictions use fixed effects by default. Set `use_re=True` to add
BLUPs for known grouping levels (unknown levels receive zero, matching
lme4's conditional-prediction convention).

```python

# Fixed-effects-only predictions (new data, no group column needed)
new_data_df = pd.DataFrame({
    'Covariate1': np.random.rand(5),
    'Covariate2': np.random.randn(5)
})
predictions_fe_only = lre_model.predict(
    new_data_df,
    x_vars=['Covariate1', 'Covariate2']
)
print(f"\nFirst 5 predictions (fixed effects only): {predictions_fe_only}")

# Predictions including random effects (BLUPs)
predictions_with_re = lre_model.predict(
    data_df,
    x_vars=['Covariate1', 'Covariate2'],
    group_var='ProviderID',
    use_re=True
)
print(f"\nFirst 5 predictions (with BLUPs): {predictions_with_re[:5]}")

```

### 3.4. Standardized Measures Calculation

```python

# Calculate Indirect Standardized Difference vs median random effect
sm_results_lre = lre_model.calculate_standardized_measures(
    stdz='indirect', # Can be 'direct' or ['indirect', 'direct']
    null='median'    # Baseline for random effects: 'median', 'mean', or a float
)
print("\n--- Linear RE Indirect Measures (vs Median Random Effect) ---")
if 'indirect' in sm_results_lre:
    print(sm_results_lre['indirect'].head())

```

### 3.5. Hypothesis Testing for Provider Effects (`test`)

Test provider random effects ($u_i$) against a null value.

```python

# Test providers vs median random effect
test_results_lre = lre_model.test(
    null='median', # Null hypothesis for u_i
    level=0.95,
    alternative='two_sided'
)
print("\n--- Linear RE Provider Test (vs Median Random Effect) ---")
print(test_results_lre.head())

```

### 3.6. Confidence Interval Calculation (`calculate_confidence_intervals`)

Compute CIs for provider random effects ($u_i$) or standardized differences.

```python

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

```

### 3.7. Visualization

Use plotting methods from the `LinearRandomEffectModel` instance. (Examples assume plots are shown interactively or saved).

```python

# Provider random effects (BLUPs) with CIs
lre_model.plot_provider_effects(null='median', level=0.95, use_flags=True)

# Standardized differences caterpillar
lre_model.plot_standardized_measures(stdz='indirect', null='median', level=0.95)

# Funnel plot (differences vs provider size)
lre_model.plot_funnel(stdz='indirect', null='median', target=0.0, alpha=[0.05, 0.01])

# Forest plot of fixed-effect coefficients
lre_model.plot_coefficient_forest()

# Model diagnostics: residuals vs fitted, Q-Q plot
lre_model.plot_residuals()
lre_model.plot_qq()

```

## 4. Discussion

Linear random effect models provide a powerful framework for analyzing clustered data, such as patient outcomes within healthcare providers. They allow for the estimation of overall covariate effects while accounting for provider-specific variability. The prediction of random effects (BLUPs) incorporates shrinkage, which can be beneficial for ranking or comparing providers, especially those with small sample sizes.

**Advantages:**

- **Efficiency:** Can be more efficient than FE models if the random effects assumption holds, particularly with many groups or few observations per group.
- **Borrowing Strength:** BLUPs "borrow strength" from the overall data, leading to more stable estimates for individual group effects.
- **Generalizability:** Allows inferences about the population of groups from which the sample is drawn.
- **Flexibility:** Can model more complex variance structures (though the current implementation focuses on random intercepts).

**Limitations and Assumptions:**

- **Random Effects Assumption:** Assumes random effects are drawn from a specific distribution (typically normal) and are uncorrelated with covariates. Violation of the latter can lead to biased $\boldsymbol\beta$ estimates.
- **Distributional Assumptions:** Relies on normality assumptions for errors and random effects for exact inference, though estimates can be robust.
- **Complexity:** Conceptually and computationally more complex than simple OLS or FE models.
- **Multiple Grouping Factors:** The current implementation supports multiple independent random-intercept terms (crossed grouping factors) via the `group_vars` parameter.

## 5. Conclusion

The `LinearRandomEffectModel`` class offers a comprehensive tool for provider profiling using linear mixed-effects models. It provides estimation of fixed effects and variance components, prediction of provider-specific random effects (BLUPs), and various methods for inference, standardization, and visualization. The pure-Python lme4-style solver supports both REML and ML, optional observation weights and offsets, and multiple crossed random-intercept terms. This approach is valuable when it is reasonable to assume providers are a sample from a population and when interest lies in both overall effects and provider-specific deviations.

## 6. References

```{bibliography} ../references.bib
:list: enumerate
:filter: docname in docnames
:keyprefix: linre-
```
