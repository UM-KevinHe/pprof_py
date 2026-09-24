(linear_random_effect_tutorial)=
# Tutorial: Linear Random Effect Model

```{contents}
:local:
:depth: 2
```

## 1. What this tutorial covers

This tutorial profiles **length of stay (LOS)** across 25 hospitals
using `LinearRandomEffectModel` — a linear mixed-effects model that
treats provider intercepts as random draws from $N(0, \sigma_u^2)$. You
will:

1. Build a synthetic cohort with known parameters.
2. Fit the model (REML) and inspect fixed effects and variance components.
3. Extract BLUPs and understand **shrinkage**.
4. Compute standardized differences.
5. Run hypothesis tests.
6. Construct confidence intervals for BLUPs and standardized measures.
7. Visualize results.

Every number is real output from `pprof_py` with `seed=789`. For the
statistical derivations, see the companion
[expert reference](linear_random_effect_model_stats). For the
fixed-effect counterpart, see the
[linear FE tutorial](linear_fixed_effect_tutorial).

## 2. Build the synthetic cohort

```python
import numpy as np
import pandas as pd
from pprof_py import LinearRandomEffectModel

np.random.seed(789)
n_hospitals = 25
sizes = np.random.randint(30, 100, n_hospitals)
N = sizes.sum()
hospital_ids = np.repeat(
    [f'Hospital_{i+1}' for i in range(n_hospitals)], sizes
)

age = np.random.normal(65, 10, N)
severity = np.random.uniform(1, 5, N)
comorbidity = np.random.binomial(1, 0.30, N)

true_beta = np.array([0.12, 2.0, 4.0])
true_intercept = 8.0
true_sigma_u = 1.5
true_sigma_e = 3.5
true_u = np.random.normal(0, true_sigma_u, n_hospitals)
u_map = {f'Hospital_{i+1}': true_u[i] for i in range(n_hospitals)}
u_obs = np.array([u_map[h] for h in hospital_ids])

y = (true_intercept + u_obs
     + age * true_beta[0]
     + severity * true_beta[1]
     + comorbidity * true_beta[2]
     + np.random.normal(0, true_sigma_e, N))

data = pd.DataFrame({
    'los': y, 'age': age, 'severity': severity,
    'comorbidity': comorbidity, 'hospital': hospital_ids,
})
```

The cohort has **1,660 patients** across **25 hospitals** (sizes
31–98, median 63). Mean LOS is 23.43 days with SD 4.93. The true
provider SD is $\sigma_u = 1.5$ and the residual SD is
$\sigma_e = 3.5$.

## 3. Fit the model

```python
model = LinearRandomEffectModel(verbose=False)
model.fit(
    data,
    y_var='los',
    x_vars=['age', 'severity', 'comorbidity'],
    group_var='hospital',
    reml=True,
)
```

REML (Restricted Maximum Likelihood) is the default and preferred method
for variance-component estimation — it accounts for the degrees of
freedom consumed by the fixed effects.

## 4. Fixed effects and variance components

```python
print(model.summary())
```

```
             estimate  std_error       stat  p_value  ci_lower   ci_upper
(Intercept)  8.829901   0.664163  13.294782      0.0  7.527199  10.132603
age          0.114446   0.008715  13.131799      0.0  0.097352   0.131540
severity     1.913312   0.074468  25.693172      0.0  1.767250   2.059374
comorbidity  4.246697   0.191682  22.154861      0.0  3.870727   4.622666
```

All covariates are highly significant:

- **Age:** 0.114 days per year (true = 0.12).
- **Severity:** 1.91 days per unit (true = 2.0).
- **Comorbidity:** 4.25 days (true = 4.0).
- **Intercept:** 8.83 (true = 8.0).

Variance components:

```
sigma_e (residual SD):      3.5234   (true = 3.5)
sigma_u (random-effect SD): 1.1704   (true = 1.5)
sigma_u^2:                  1.3698
```

AIC = 8965.44, BIC = 8997.93, Log-likelihood = −4476.72.

The estimated $\hat{\sigma}_u = 1.17$ is moderately shrunk from the
true 1.5. This is common with 25 groups and moderate group sizes —
the REML estimator is conservative. The residual SD is recovered almost
exactly.

## 5. BLUPs and shrinkage

The Best Linear Unbiased Predictors (BLUPs) $\hat{u}_i$ estimate each
hospital's deviation from the overall intercept, *pulled toward zero* by
a shrinkage factor:

$$
\hat{u}_i = \frac{n_i \hat{\sigma}_u^2}{n_i \hat{\sigma}_u^2 + \hat{\sigma}_e^2}
(\bar{Y}_i - \bar{\mathbf{X}}_i^\top\hat{\boldsymbol\beta})
$$

For a hospital with $n_i = 63$ (the median), the shrinkage factor is
0.87 — the BLUP retains 87 % of the raw residual. Smaller hospitals
are pulled more toward zero.

```python
blups = model.get_random_effects()
print(blups.head(10))
```

```
Hospital_1     0.213653
Hospital_10    0.488747
Hospital_11   -0.639378
Hospital_12   -1.003254
Hospital_13   -0.486697
Hospital_14   -1.581776
Hospital_15   -1.391879
Hospital_16   -0.087237
Hospital_17    0.428253
Hospital_18   -2.540525
```

The BLUP range is $[-2.54, 2.33]$ with SD 1.09. Hospital 18 has the
most negative BLUP ($-2.54$), meaning its patients stay about 2.5 days
*less* than the overall average after case-mix adjustment. Hospital 2
has the highest BLUP.

## 6. Standardized differences

The **Indirect Standardized Difference** (ISDiff) is:

$$
\text{ISDiff}_i = \hat{u}_i - u_0
$$

where $u_0$ is the baseline random effect (here, the median BLUP =
$-0.054$). For the linear model, ISDiff reduces to the BLUP minus the
baseline — on the original outcome scale (days).

```python
sm = model.calculate_standardized_measures(stdz='indirect', null='median')
print(sm['indirect'].head(10))
```

```
      group_id  indirect_difference     observed     expected
0   Hospital_1             0.213653  1955.235135  1937.929217
1  Hospital_10             0.488747  2216.366011  2170.912568
2  Hospital_11            -0.639378  2133.990591  2194.731545
3  Hospital_12            -1.003254  1215.984027  1270.159719
4  Hospital_13            -0.486697  2033.396048  2075.738673
5  Hospital_14            -1.581776   736.970909   790.751280
6  Hospital_15            -1.391879  1371.589998  1459.278352
7  Hospital_16            -0.087237  2280.709964  2289.259209
8  Hospital_17             0.428253  2271.276013  2229.735491
9  Hospital_18            -2.540525   624.726896   703.483178
```

`observed` is the sum of fitted values (including BLUPs) for that
hospital; `expected` is the sum under the baseline effect. Hospital 18's
patients total 625 vs. an expected 703 — consistent with its strongly
negative BLUP.

## 7. Hypothesis testing

The Z-test divides each BLUP by its posterior standard error:

$$
Z_i = \frac{\hat{u}_i - u_0}{\widehat{\text{se}}(\hat{u}_i)}
$$

```python
median_blup = blups.median()  # -0.054184, used again below
test_results = model.test(reference='median', level=0.95, alternative='two_sided')   # 'median' is median_blup
print(test_results[['estimate', 'se', 'z_raw', 'p_value', 'flag', 'ci_lower', 'ci_upper']].head(10))
```

```
             estimate        se     z_raw   p_value  flag  ci_lower  ci_upper
provider
Hospital_1   0.213653  0.371274  0.721401  0.470663     0 -0.514031  0.941337
Hospital_10  0.488747  0.348766  1.556721  0.119537     0 -0.194821  1.172315
Hospital_11 -0.639378  0.345398 -1.694261  0.090216     0 -1.316346  0.037589
Hospital_12 -1.003254  0.443691 -2.139032  0.032433    -1 -1.872872 -0.133635
Hospital_13 -0.486697  0.359493 -1.203120  0.228930     0 -1.191289  0.217896
Hospital_14 -1.581776  0.536928 -2.845058  0.004440    -1 -2.634135 -0.529416
Hospital_15 -1.391879  0.415061 -3.222887  0.001269    -1 -2.205383 -0.578374
Hospital_16 -0.087237  0.340524 -0.097065  0.922675     0 -0.754653  0.580178
Hospital_17  0.428253  0.342126  1.410116  0.158506     0 -0.242302  1.098807
Hospital_18 -2.540525  0.556668 -4.466468  0.000008    -1 -3.631575 -1.449476
```

`reference='median'` uses the median BLUP directly; without it the
reference is 0, the random-effect mean.

**11 of 25** hospitals are flagged at the 5 % level (5 high, 6 low).
Hospital 2 has the strongest positive signal ($Z = 4.93$, $p < 10^{-6}$)
and Hospital 18 the strongest negative ($Z = -4.47$, $p < 10^{-5}$).

Compared to a fixed-effect analysis of the same data, the RE model
flags *fewer* providers because shrinkage pulls extreme BLUPs toward
zero, and posterior standard errors account for the additional
uncertainty.

## 8. Confidence intervals

### CIs for random effects ($u_i$)

```python
alpha_ci = model.calculate_confidence_intervals(option='alpha', level=0.95)
print(alpha_ci['alpha_ci'].head(10))
```

```
                group_id     alpha  alpha_lower  alpha_upper
Hospital_1    Hospital_1  0.213653    -0.514031     0.941337
Hospital_10  Hospital_10  0.488747    -0.194821     1.172315
Hospital_11  Hospital_11 -0.639378    -1.316346     0.037589
Hospital_12  Hospital_12 -1.003254    -1.872872    -0.133635
Hospital_13  Hospital_13 -0.486697    -1.191289     0.217896
Hospital_14  Hospital_14 -1.581776    -2.634135    -0.529416
Hospital_15  Hospital_15 -1.391879    -2.205383    -0.578374
Hospital_16  Hospital_16 -0.087237    -0.754653     0.580178
Hospital_17  Hospital_17  0.428253    -0.242302     1.098807
Hospital_18  Hospital_18 -2.540525    -3.631575    -1.449475
```

Hospital 18's CI is $[-3.63, -1.45]$ — entirely below zero, confirming
a significantly shorter LOS. Hospital 16's CI $[-0.75, 0.58]$ spans
zero, consistent with it not being flagged.

### CIs for standardized differences

```python
sm_ci = model.calculate_confidence_intervals(
    option='SM', stdz='indirect', null=median_blup,
    level=0.95, alternative='two_sided',
)
print(sm_ci['indirect_ci'].head(10))
```

```
      group_id  indirect_difference     observed     expected     lower     upper
0   Hospital_1             0.213653  1955.235135  1937.929217 -0.514031  0.941337
1  Hospital_10             0.488747  2216.366011  2170.912568 -0.194821  1.172315
2  Hospital_11            -0.639378  2133.990591  2194.731545 -1.316922  0.037589
3  Hospital_12            -1.003254  1215.984027  1270.159719 -1.872872 -0.133635
4  Hospital_13            -0.486697  2033.396048  2075.738673 -1.191289  0.217896
5  Hospital_14            -1.581776   736.970909   790.751280 -2.634135 -0.529416
6  Hospital_15            -1.391879  1371.589998  1459.278352 -2.205383 -0.578374
7  Hospital_16            -0.087237  2280.709964  2289.259209 -0.754653  0.580178
8  Hospital_17             0.428253  2271.276013  2229.735491 -0.242302  1.098807
9  Hospital_18            -2.540525   624.726896   703.483178 -3.631575 -1.449475
```

## 9. Visualizing the results

### Funnel plot

```python
model.plot_funnel(
    null=median_blup,
    target=0.0,
    alpha=[0.05, 0.01],
    plot_title="Funnel Plot: Indirect Standardized Difference (LOS)",
)
```

### Caterpillar plot — BLUPs

```python
model.plot_provider_effects(
    null=median_blup,
    level=0.95,
    use_flags=True,
    plot_title="Provider Random Effects (BLUPs, LOS days)",
)
```

### Caterpillar plot — standardized differences

```python
model.plot_standardized_measures(
    stdz='indirect',
    null=median_blup,
    level=0.95,
    use_flags=True,
    plot_title="Indirect Standardized Difference (LOS days)",
)
```

### Forest plot — fixed effects

```python
model.plot_coefficient_forest(
    plot_title="Forest Plot of Covariate Coefficients",
)
```

### Diagnostic plots

```python
model.plot_residuals()
model.plot_qq()
```

## 10. Quick interpretation guide

| Quantity | Scale | This cohort |
|---|---|---|
| $\hat{\beta}_{\text{age}}$ | days / year | 0.114 |
| $\hat{\beta}_{\text{severity}}$ | days / unit | 1.913 |
| $\hat{\beta}_{\text{comorbidity}}$ | days (binary) | 4.247 |
| $\hat{\sigma}_u$ | days (provider SD) | 1.170 (true = 1.5) |
| $\hat{\sigma}_e$ | days (residual SD) | 3.523 (true = 3.5) |
| BLUP $\hat{u}_i$ | days from mean | range $[-2.54, +2.33]$ |
| ISDiff | days above/below median | same as BLUP minus baseline |
| Flagged | — | 11 of 25 at $\alpha = 0.05$ |

### Shrinkage in action

The key difference from the
[fixed-effect model](linear_fixed_effect_tutorial) is **shrinkage**: each
hospital's BLUP is pulled toward zero by a factor that depends on its
sample size and the ratio $\sigma_u^2 / \sigma_e^2$. Small hospitals
are pulled more aggressively. This means:

- Fewer hospitals are flagged (11/25 here vs. potentially more with FE).
- Rankings are more stable, especially for small hospitals.
- Estimates are biased toward zero but have lower mean-squared error.

Use the random-effect model when providers are viewed as a sample from a
larger population, when small-sample providers need stabilization, or
when the focus is on ranking rather than unbiased point estimates for
each individual provider. The
[Fixed vs. Random Effects](fixed_vs_random_effects) page discusses the
trade-off in detail.
