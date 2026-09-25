(logistic_random_effect_model_stats)=
# Logistic Random Effect Modeling

```{contents}
:local:
:depth: 2
```

## 1. Introduction

When analyzing binary patient outcomes (e.g., mortality, readmission) clustered within healthcare providers, logistic random effect models, also known as Generalized Linear Mixed Models (GLMMs) with a logit link, are a common approach {cite}`Bates2015lme4,McCulloch2001GLMM,Stroup2013GLMM`. These models account for the correlation of outcomes within providers by incorporating provider-specific random effects, typically random intercepts. This allows for provider comparisons while adjusting for patient-level covariates.

The `LogisticRandomEffectModel` class implements such a model using a pure-Python reimplementation of the `lme4` {cite}`Bates2015lme4` fitting algorithm (PIRLS + Laplace approximation). No R or `pymer4` dependency is required.

This document outlines the statistical methodology underpinning the `LogisticRandomEffectModel`, covering:

- The logistic random intercept model formulation.
- Parameter estimation via PIRLS + Laplace approximation (lme4-style), including fixed effects and variance components, and prediction of random effects (BLUPs).
- Calculation of standardized measures (e.g., Standardized Mortality/Morbidity Ratios - SMRs, Standardized Rates) for performance comparison.
- Hypothesis testing procedures for provider random effects.
- Construction of confidence intervals for fixed effects and random effects.
- Visualization tools for interpreting model results.

## 2. Methods

### 2.1. The Logistic Random Intercept Model

Let $Y_{ij}$ be a binary outcome (0 or 1) for subject $j$ ($j = 1, \ldots, n_i$) within provider (group) $i$ ($i = 1, \ldots, m$). Let $\mathbf{X}_{ij}$ be a $p \times 1$ vector of subject-level covariates. The probability of success $P(Y_{ij}=1) = p_{ij}$ is modeled using a logit link function:

$$
\text{logit}(p_{ij}) = \ln\left(\frac{p_{ij}}{1-p_{ij}}\right) = \eta_{ij} = \mathbf{X}_{ij}^\top\boldsymbol\beta + u_i
$$

where:

- $\boldsymbol\beta$ is the $p \times 1$ vector of fixed regression coefficients associated with the covariates. These represent the change in log-odds of the outcome for a one-unit change in the corresponding covariate, holding the provider constant.
- $u_i$ is the random intercept for provider $i$. It represents the deviation of provider $i$'s baseline log-odds from the overall intercept (which is part of $\mathbf{X}_{ij}^\top\boldsymbol\beta$ if an intercept term is included in $\mathbf{X}_{ij}$).
- It is assumed that $u_i \sim N(0, \sigma^2_u)$, where $\sigma^2_u$ is the variance of the provider random effects.
- The random effects $u_i$ are assumed to be independent of the covariates $\mathbf{X}_{ij}$.

The model aims to estimate the fixed effects $\boldsymbol\beta$, the random effects variance $\sigma^2_u$, and to predict the individual random effects $u_i$.

### 2.2. Parameter Estimation and Prediction

The `LogisticRandomEffectModel` fits the GLMM using a pure-Python implementation of the lme4 algorithm: Penalized Iteratively Reweighted Least Squares (PIRLS) for the inner loop, with a two-stage outer optimization (nAGQ=0 then Laplace/nAGQ=1). No R or pymer4 dependency is required.

- **Fixed Effects** ($\hat{\boldsymbol\beta}$): Estimates of the population-average covariate effects. Stored in `coefficients_['beta']` (a `pd.Series`).
- **Random Effects Variance** ($\hat{\sigma}^2_u$): Estimate of the variability between providers. Stored in `variances_['alpha']` (a dict mapping provider_var to $\sigma^2_u$).
- **Random Effects** (BLUPs, $\hat{u}_i$): Predictions of the provider-specific deviations from the overall intercept. These are Best Linear Unbiased Predictors (BLUPs) on the log-odds scale and exhibit shrinkage towards the mean (zero). Stored in `coefficients_['alpha']` (a dict of `pd.Series` per provider_var). Also accessible via `get_random_effects(provider_var)`.
- **Variance-Covariance of Fixed Effects:** Stored in `variances_['beta']`.

The fitted probabilities $\hat{p}_{ij}$ (including random effects) are stored in `fitted_`. The linear predictor from fixed effects only, $\mathbf{X}_{ij}^\top\hat{\boldsymbol\beta}$, is stored in `xbeta_`.

### 2.3. Standardized Measures for Performance Comparison

For logistic random effects models, standardized measures allow for fair comparison of provider performance by adjusting for patient case mix and provider-specific random effects. The `calculate_standardized_measures` method computes both indirect and direct standardized ratios and rates, based on the predicted random effects (BLUPs).

Let $\hat{\boldsymbol{\beta}}$ denote the estimated fixed effects, and $\hat{\alpha}_i$ the estimated random effect (BLUP) for provider $i$. Define a reference or baseline random effect $\alpha_0$ (e.g., the median or mean of $\hat{\alpha}_i$, as specified by the `reference` parameter).

#### 2.3.1. Indirect Standardization

Indirect standardization compares the observed outcome for a provider (as predicted by the full model, including random effects) to the expected outcome for that provider if its random effect were set to the baseline value $\alpha_0$, given its specific patient mix.

- **Observed Outcome for Provider $i$**:
  The sum of fitted probabilities (including both fixed and random effects) for all $n_i$ subjects in provider $i$. This is the `observed` column in the output DataFrame.

$$
O_i = \sum_{j=1}^{n_i} \hat{p}_{ij}^{\text{full}} = \sum_{j=1}^{n_i} \text{logit}^{-1}(\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \hat{\alpha}_i)
$$

- **Expected Outcome for Provider $i$ under Baseline**:
  The sum of expected probabilities for provider $i$'s $n_i$ subjects, if the provider effect was the baseline $\alpha_0$, adjusted for their covariates. This is the `expected` column.

$$
E_i(\alpha_0) = \sum_{j=1}^{n_i} \text{logit}^{-1}(\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \alpha_0)
$$

- **Indirect Standardized Ratio**:
  The ratio of observed to expected outcomes for provider $i$:

$$
\text{ISR}_i = \frac{O_i}{E_i(\alpha_0)}
$$

- **Indirect Standardized Rate**:
  The standardized rate for provider $i$, calculated as the indirect ratio multiplied by the overall population rate (expressed as a percentage):

$$
\text{ISRate}_i = \text{ISR}_i \times \text{Population Rate}
$$

- **Indirect Standardized Difference**:
  The difference between the provider's random effect and the baseline:

$$
\text{ISDiff}_i = \hat{\alpha}_i - \alpha_0
$$

#### 2.3.2. Direct Standardization

Direct standardization compares the expected outcome for the entire population if all subjects experienced provider $k$'s random effect ($\hat{\alpha}_k$) to the expected outcome if all subjects experienced the baseline effect ($\alpha_0$).

- **Expected Outcome under Provider $k$'s Effect**:
  The sum of expected probabilities for all subjects, using provider $k$'s random effect:

$$
E^{(k)} = \sum_{i=1}^m \sum_{j=1}^{n_i} \text{logit}^{-1}(\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \hat{\alpha}_k)
$$

- **Expected Outcome under Baseline Effect**:
  The sum of expected probabilities for all subjects, using the baseline effect:

$$
E^{(0)} = \sum_{i=1}^m \sum_{j=1}^{n_i} \text{logit}^{-1}(\mathbf{X}_{ij}^\top\hat{\boldsymbol{\beta}} + \alpha_0)
$$

- **Direct Standardized Ratio**:
  The ratio of expected outcomes under provider $k$'s effect to the observed total outcome:

$$
\text{DSR}_k = \frac{E^{(k)}}{O_{\text{total}}}
$$

  where $O_{\text{total}}$ is the sum of observed outcomes across all subjects.

- **Direct Standardized Rate**:

$$
\text{DSRate}_k = \text{DSR}_k \times \text{Population Rate}
$$

- **Direct Standardized Difference**:

$$
\text{DSDiff}_k = \hat{\alpha}_k - \alpha_0
$$

The implementation (`LogisticRandomEffectModel.calculate_standardized_measures`) returns a dictionary with DataFrames for each standardization method. Each DataFrame contains the group ID, standardized difference, ratio, rate, and the observed and expected outcomes as described above. These measures allow for meaningful provider comparisons while accounting for both patient case mix and provider-specific random effects.

### 2.4. Hypothesis Testing for Provider Random Effects

To assess if a provider's random effect $u_i$ is significantly different from a null value $u_0$, a Z-test is performed:

$$
Z_i = \frac{\hat{u}_i - u_0}{\widehat{\text{se}}(\hat{u}_i)}
$$

where $\hat{u}_i$ is the BLUP for provider $i$, and $\widehat{\text{se}}(\hat{u}_i)$ is its posterior standard error. These standard errors are derived from the posterior conditional variance of the BLUPs (see `_get_posterior_se` method).

Under the null hypothesis $H_0: u_i = u_0$, the statistic $Z_i$ is assumed to follow a standard normal distribution. P-values are calculated based on this distribution according to the specified `alternative` ('two_sided', 'less', 'greater') in the `test` method.

### 2.5. Confidence Intervals

- **For Fixed Effects** $\boldsymbol\beta$:
  The `summary` method provides confidence intervals for fixed effects, typically based on Z-scores and standard errors from the `lme4` output.

$$
\hat{\beta}_k \pm z_{1-\alpha/2} \times \widehat{\text{se}}(\hat{\beta}_k)
$$

- **For Random Effects** $u_i$ (BLUPs):
  The `calculate_confidence_intervals` method with `option='alpha'` provides approximate confidence intervals for the BLUPs $\hat{u}_i$ on the log-odds scale. These are based on the posterior standard errors of the BLUPs and a normal approximation:

$$
\hat{u}_i \pm z_{1-\alpha/2} \times \widehat{\text{se}}(\hat{u}_i)
$$

- **For Standardized Measures** (Ratios/Rates):
  The `calculate_confidence_intervals` method with `option='SM'` computes approximate CIs for standardized ratios and rates using a Delta-method transformation of the BLUP posterior standard errors on the log scale.

### 2.6. Visualization

The `LogisticRandomEffectModel` class provides several plotting methods via the `RandomEffectPlottingMixin`:

- **Funnel Plot** (`plot_funnel`): Plots the indirect standardized ratio (O/E) against expected count (precision). Control limits use the Poisson approximation $\text{target} \pm z_{1-\alpha/2} / \sqrt{E_i}$, with `target=1.0` for ratios. Providers are flagged using the `test()` method.
- **Caterpillar Plot for Provider Effects** (`plot_provider_effects`): Displays BLUPs $\hat{u}_i$ on the log-odds scale with confidence intervals from the posterior standard errors. Optionally colour-codes providers using significance flags.
- **Caterpillar Plot for Standardized Measures** (`plot_standardized_measures`): Displays indirect or direct standardized ratios or rates with CIs from `calculate_confidence_intervals(option='SM')`.
- **Coefficient Forest Plot** (`plot_coefficient_forest`): Displays fixed-effect estimates $\hat{\boldsymbol{\beta}}$ and z-based Wald confidence intervals.

## 3. Implementation and Usage

The `LogisticRandomEffectModel` class in `pprof_py` implements these methods.

### 3.1. Initialization and Fitting

```python
import pandas as pd
import numpy as np
from pprof_py import LogisticRandomEffectModel

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
logit_re_model.fit(X=data, y_var='Y', x_vars=['X1', 'X2'], provider_var='GroupID')
```

### 3.2. Accessing Results

```python
# Fixed Effects Coefficients (log-odds scale)
fixed_effects = logit_re_model.coefficients_['beta']
print("Estimated Fixed Effects (Betas):")
print(fixed_effects)

# Predicted Random Effects (BLUPs, log-odds scale)
random_effects_blups = logit_re_model.get_random_effects()
print("\nPredicted Random Effects (BLUPs, first 5):")
print(random_effects_blups.head())

# Variance Components
fe_var_cov = logit_re_model.variances_['beta']
re_var = logit_re_model.variances_['alpha']  # {provider_var: sigma_u^2}
sigma_u2 = list(re_var.values())[0]
print(f"\nEstimated Variance of Random Effects (sigma_u^2): {sigma_u2:.3f}")

# Fit statistics
print(f"\nAIC: {logit_re_model.aic_:.2f}")
print(f"BIC: {logit_re_model.bic_:.2f}")
print(f"Log-likelihood: {logit_re_model.loglike_:.2f}")
```

### 3.3. Prediction (Fixed Effects Only)

```python
# Predict probabilities for new data using only fixed effects
new_data_df = pd.DataFrame({
    'X1': np.random.randn(5),
    'X2': np.random.binomial(1, 0.5, 5)
})

predictions_fe_only = logit_re_model.predict(
    X=new_data_df,
    x_vars=['X1', 'X2']
)
print(f"\nPredicted probabilities (fixed effects only): {predictions_fe_only}")
```

### 3.4. Standardized Measures Calculation

```python
# Calculate Indirect Standardized Ratios and Rates vs median random effect
sm_results = logit_re_model.calculate_standardized_measures(
    stdz='indirect',
    reference='median'
)
print("\n--- Logistic RE Indirect Measures (vs Median Random Effect) ---")
if 'indirect' in sm_results:
    print(sm_results['indirect'].head())
```

### 3.5. Hypothesis Testing for Provider Random Effects (`.test()`)

Test provider random effects ($u_i$, on log-odds scale) against a null value.

```python
test_results = logit_re_model.test(
    reference=0.0,    # the default: the random-effect mean
    level=0.95,
    alternative='two_sided'
)
print("\n--- Logistic RE Provider Test (vs Null RE of 0.0) ---")
print(test_results.head())
```

`test_method='poibin_exact'` and `'resampling'` test each provider's event count instead of its
BLUP; every method returns the columns described in {ref}`ll_ref_measures`, with `flag` = 1 for
providers above the reference and -1 below.

### 3.6. Confidence Interval Calculation (`.calculate_confidence_intervals()`)

Compute CIs for provider random effects ($u_i$, on log-odds scale).

```python
# Get 95% CIs for random effects (BLUPs)
alpha_cis = logit_re_model.calculate_confidence_intervals(
    option='alpha',
    level=0.95
)
print("\n--- Logistic RE Random Effect (u_i) CIs (Log-Odds Scale) ---")
if 'alpha_ci' in alpha_cis:
    print(alpha_cis['alpha_ci'].head())

# CIs for standardized measures (ratio and rate)
sm_cis = logit_re_model.calculate_confidence_intervals(
    option='SM',
    stdz='indirect',
    reference='median',
    measure=['ratio', 'rate'],
    level=0.95
)
if 'indirect_ratio' in sm_cis:
    print(sm_cis['indirect_ratio'].head())
```

### 3.7. Visualization

Use plotting methods from the `LogisticRandomEffectModel` instance.

```python
# Funnel plot: indirect standardized ratio (O/E) vs expected count
logit_re_model.plot_funnel(
    test_method='wald', reference='median', target=1.0, alpha=[0.05, 0.01]
)

# Provider effects caterpillar (BLUPs on log-odds scale)
logit_re_model.plot_provider_effects(
    reference='median', level=0.95, use_flags=True
)

# Standardized measures caterpillar (indirect ratio with CIs)
logit_re_model.plot_standardized_measures(
    stdz='indirect', measure='ratio', level=0.95, use_flags=True
)

# Forest plot of fixed-effect coefficients (z-based Wald CIs)
logit_re_model.plot_coefficient_forest()
```

## 4. Discussion

Logistic random effect models, as implemented in `pprof_py` using lme4-style PIRLS + Laplace approximation, provide a robust framework for analyzing binary outcomes clustered within providers. They account for provider-level heterogeneity through random intercepts, allowing for more nuanced comparisons than models ignoring such clustering.

**Advantages:**

- Appropriately models binary outcomes and accounts for data clustering.
- Uses a pure-Python reimplementation of the `lme4` fitting algorithm, eliminating R dependencies.
- Provides estimates of fixed effects (population average) and predictions of random effects (provider-specific deviations, BLUPs).
- BLUPs incorporate shrinkage, which can be beneficial for ranking or comparing providers, especially those with small sample sizes or extreme raw rates.

**Limitations and Assumptions:**

- **Software Dependencies:** Requires `scipy` and `numpy` (standard Python scientific stack). No R, `lme4`, or `pymer4` is required.
- **GLMM Assumptions:** Relies on standard GLMM assumptions, including linearity on the logit scale, correct specification of the random effects distribution (typically normal), and independence of random effects from covariates.
- **Computational Complexity:** Fitting GLMMs can be computationally intensive, especially with large datasets or complex random effects structures.
- **Interpretation of Standardized Measures:** Standardized ratios/rates derived from logistic models can be complex to interpret, and their confidence intervals are challenging to compute accurately without advanced statistical methods (e.g., bootstrapping). The current implementation provides point estimates for these measures.

## 5. Conclusion

The `LogisticRandomEffectModel` offers a valuable tool for provider profiling with binary outcomes. By incorporating random effects, it provides a statistically sound approach to adjust for case mix and account for provider-level variability. While the interpretation and inference for standardized measures derived from these models require care, the model's ability to estimate provider-specific effects (BLUPs) on the log-odds scale, along with their approximate confidence intervals, is a key strength for performance evaluation.


## References

```{bibliography} ../references.bib
:list: enumerated
:filter: docname in docnames
:keyprefix: logre-
```
