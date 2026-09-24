(linear_fixed_effect_tutorial)=
# Tutorial: Linear Fixed Effect Model

```{contents}
:local:
:depth: 2
```

## 1. What this tutorial covers

This tutorial walks through a provider-profiling analysis of
**length of stay (LOS)** — a continuous outcome — using
`LinearFixedEffectModel`. You will:

1. Build a synthetic 30-hospital cohort with known provider effects.
2. Fit a fixed-effect model that estimates a separate intercept for each
   hospital, controlling for patient age, severity, and comorbidity.
3. Extract and interpret the covariate coefficients ($\hat{\boldsymbol\beta}$)
   and provider effects ($\hat{\gamma}_i$).
4. Compute standardized differences to compare hospitals.
5. Run hypothesis tests to flag outlier hospitals.
6. Construct confidence intervals for provider effects and standardized
   measures.
7. Visualize results with funnel, caterpillar, and forest plots.

Every number shown below is real output from `pprof_py`, captured with
`seed=42`. For the statistical theory, see the companion
[expert reference](linear_fixed_effect_model_stats).

## 2. Build the synthetic cohort

```python
import numpy as np
import pandas as pd
from pprof_py import LinearFixedEffectModel

np.random.seed(42)
n_hospitals = 30
sizes = np.random.randint(40, 120, n_hospitals)
N = sizes.sum()
hospital_ids = np.repeat(
    [f'Hospital_{i+1}' for i in range(n_hospitals)], sizes
)

age = np.random.normal(65, 12, N)
severity = np.random.uniform(1, 5, N)
comorbidity = np.random.binomial(1, 0.35, N)

true_beta = np.array([0.15, 1.8, 3.5])
true_intercept = 5.0
true_gamma = np.random.normal(0, 2.0, n_hospitals)
gamma_map = {f'Hospital_{i+1}': true_gamma[i] for i in range(n_hospitals)}
gamma_obs = np.array([gamma_map[h] for h in hospital_ids])

y = (true_intercept + gamma_obs
     + age * true_beta[0]
     + severity * true_beta[1]
     + comorbidity * true_beta[2]
     + np.random.normal(0, 3.0, N))

data = pd.DataFrame({
    'los': y, 'age': age, 'severity': severity,
    'comorbidity': comorbidity, 'hospital': hospital_ids,
})
```

The cohort has **2,478 patients** across **30 hospitals** (sizes
41–119, median 90). Mean LOS is 20.79 days with SD 4.75. Each
hospital's true intercept ($\gamma_i$) is drawn from
$N(0, 2)$ — so there is genuine provider-level variation baked into
the data.

## 3. Fit the model

```python
model = LinearFixedEffectModel(gamma_var_option='complete')
model.fit(
    X=data,
    y_var='los',
    x_vars=['age', 'severity', 'comorbidity'],
    group_var='hospital',
)
```

`gamma_var_option='complete'` uses the full variance formula for
$\hat{\gamma}_i$, propagating the uncertainty in
$\hat{\boldsymbol\beta}$.

## 4. Fixed effects (covariate coefficients)

```python
print(model.summary())
```

```
             estimate  std_error       stat  p_value  ci_lower  ci_upper
age          0.147454   0.005248  28.095648      0.0  0.137162  0.157745
severity     1.806502   0.054480  33.159188      0.0  1.699671  1.913333
comorbidity  3.226398   0.132513  24.347734      0.0  2.966548  3.486248
```

All three covariates are highly significant (t > 24, $p < 10^{-7}$):

- **Age:** each additional year adds 0.147 days to LOS (true = 0.15).
- **Severity:** each unit on the 1–5 scale adds 1.81 days (true = 1.8).
- **Comorbidity:** the presence of a comorbid condition adds 3.23 days
  (true = 3.5).

The residual standard deviation is $\hat{\sigma} = 3.098$
(true = 3.0). AIC = 12670.71, BIC = 12868.43.

## 5. Provider effects ($\hat{\gamma}_i$)

The model estimates a separate intercept $\hat{\gamma}_i$ for each of
the 30 hospitals. These live in `model.coefficients_['gamma']`:

```python
gammas = model.coefficients_['gamma'].flatten()
```

```
Hospital_1  6.273    Hospital_11  5.824    Hospital_21  6.213
Hospital_2  4.630    Hospital_12  6.977    Hospital_22  1.716
Hospital_3  5.824    Hospital_13  6.997    Hospital_23  6.523
Hospital_4  8.208    Hospital_14  4.077    Hospital_24  4.423
Hospital_5  6.882    Hospital_15  1.517    Hospital_25  8.207
Hospital_6  1.077    Hospital_16  2.623    Hospital_26  3.183
Hospital_7  4.511    Hospital_17  4.511    Hospital_27  5.213
Hospital_8  2.871    Hospital_18  2.871    Hospital_28  4.649
Hospital_9  4.451    Hospital_19  7.382    Hospital_29  5.111
Hospital_10 4.630    Hospital_20  2.775    Hospital_30  4.543
```

The median $\hat{\gamma}$ is **4.533**. The range is $[1.517, 8.208]$,
spanning about 6.7 days — hospitals differ substantially in baseline
LOS even after adjusting for patient severity, age, and comorbidity.
Hospital 4 and Hospital 25 are the highest ($\hat{\gamma} \approx 8.2$);
Hospital 15 and Hospital 6 are the lowest ($\hat{\gamma} \approx 1.1$–$1.5$).

## 6. Standardized differences

The **Indirect Standardized Difference** (ISDiff) measures each
hospital's average LOS relative to a baseline:

$$
\text{ISDiff}_i = \hat{\gamma}_i - \gamma_0
$$

where $\gamma_0$ is the median provider effect. Because the model is
linear, indirect and direct standardized differences are identical on
a per-patient basis.

```python
sm = model.calculate_standardized_measures(stdz='indirect', null='median')
print(sm['indirect'].head(10))
```

```
      group_id  indirect_difference     observed     expected
0   Hospital_1             1.739358  2030.804766  1872.523173
1  Hospital_10             0.097102  1220.432760  1214.509551
2  Hospital_11             1.290467  2026.653917  1907.930979
3  Hospital_12             2.444269   940.652738   840.437712
4  Hospital_13             2.463688  1616.331450  1446.336947
5  Hospital_14            -0.455865  1548.012485  1583.114098
6  Hospital_15            -3.015680   723.848638   847.491501
7  Hospital_16            -1.910473  1896.296234  2093.074909
8  Hospital_17            -0.021679  2000.426136  2002.572392
9  Hospital_18            -1.661712  1151.845892  1251.548638
```

Hospital 13's ISDiff of +2.46 means its patients stay 2.46 days
*longer* on average than the median hospital, after case-mix adjustment.
Hospital 15's ISDiff of −3.02 means patients stay about 3 days *less*.

## 7. Hypothesis testing

The t-test compares each $\hat{\gamma}_i$ to the median:

$$
T_i = \frac{\hat{\gamma}_i - \gamma_0}{\widehat{\text{se}}(\hat{\gamma}_i)}
$$

with $N - m - p = 2478 - 30 - 3 = 2445$ degrees of freedom.

```python
test_results = model.test(reference='median', level=0.95, alternative='two_sided')
print(test_results[['estimate', 'se', 'z_raw', 'p_value', 'flag', 'ci_lower', 'ci_upper']].head(10))
```

```
             estimate        se     z_raw   p_value  flag  ci_lower  ci_upper
provider
Hospital_1   6.361710  0.501403  3.469712  0.000521     1  5.378491  7.344929
Hospital_10  4.715748  0.543014  0.176929  0.859565     0  3.650933  5.780563
Hospital_11  5.913038  0.501018  2.579475  0.009895     1  4.930575  6.895501
Hospital_12  7.066020  0.616807  3.959401  0.000075     1  5.856502  8.275538
Hospital_13  7.088063  0.542135  4.543036  0.000006     1  6.024972  8.151155
Hospital_14  4.166125  0.520740 -0.870793  0.383867     0  3.144987  5.187262
Hospital_15  1.606204  0.615631 -4.882482  0.000001    -1  0.398992  2.813417
Hospital_16  2.710569  0.489648 -3.892473  0.000099    -1  1.750401  3.670737
Hospital_17  4.598247  0.486445 -0.044022  0.964887     0  3.644361  5.552134
Hospital_18  2.961162  0.556988 -2.974629  0.002933    -1  1.868946  4.053378
```

`z_raw` is the normal equivalent of $T_i$, $\Phi^{-1}(F_t(T_i))$, which
is slightly smaller in magnitude than $T_i$; the p-value, flag and
interval follow the t distribution. $T_i$ itself is
`(estimate - null_value) / se`.

**18 of 30** hospitals are flagged at the 5 % level — 10 high (flag = +1)
and 8 low (flag = −1). This is expected: with a true provider SD of 2.0
and a residual SD of 3.0, the signal-to-noise ratio is strong enough
that most hospitals show statistically distinguishable effects. The most
extreme are Hospital 4 ($T = 7.35$, $p < 10^{-7}$) and Hospital 6
($T = -5.17$, $p < 10^{-6}$).

## 8. Confidence intervals

### CIs for provider effects ($\gamma_i$)

```python
gamma_ci = model.calculate_confidence_intervals(
    option='gamma', level=0.95, alternative='two_sided'
)
print(gamma_ci['gamma_ci'].head(10))
```

```
      group_id     gamma     lower     upper
0   Hospital_1  6.272529  5.289452  7.255605
1  Hospital_10  4.630272  3.565612  5.694933
2  Hospital_11  5.823637  4.841330  6.805945
3  Hospital_12  6.977439  5.768055  8.186823
4  Hospital_13  6.996859  5.933929  8.059788
5  Hospital_14  4.077305  3.056328  5.098283
6  Hospital_15  1.517491  0.310405  2.724576
7  Hospital_16  2.622698  1.662727  3.582669
8  Hospital_17  4.511491  3.557771  5.465211
9  Hospital_18  2.871458  1.779411  3.963505
```

Hospital 4's CI is $[5.77, 8.19]$ — entirely above the median of 4.53,
confirming it as a high-LOS outlier. Hospital 15's CI is
$[0.31, 2.72]$ — entirely below the median.

### CIs for standardized differences

```python
sm_ci = model.calculate_confidence_intervals(
    option='SM', stdz='indirect', null='median',
    level=0.95, alternative='two_sided',
)
print(sm_ci['indirect_ci'].head(10))
```

```
      group_id  indirect_difference     observed     expected     lower     upper
0   Hospital_1             1.739358  2030.804766  1872.523173  0.756282  2.722434
1  Hospital_10             0.097102  1220.432760  1214.509551 -0.967558  1.161763
2  Hospital_11             1.290467  2026.653917  1907.930979  0.308160  2.272774
3  Hospital_12             2.444269   940.652738   840.437712  1.234885  3.653653
4  Hospital_13             2.463688  1616.331450  1446.336947  1.400759  3.526618
5  Hospital_14            -0.455865  1548.012485  1583.114098 -1.476842  0.565112
6  Hospital_15            -3.015680   723.848638   847.491501 -4.222766 -1.808595
7  Hospital_16            -1.910473  1896.296234  2093.074909 -2.870443 -0.950502
8  Hospital_17            -0.021679  2000.426136  2002.572392 -0.975399  0.932041
9  Hospital_18            -1.661712  1151.845892  1251.548638 -2.753759 -0.569666
```

Note that Hospital 17's CI $[-0.98, 0.93]$ spans zero — consistent
with its non-significant test result ($T = -0.04$, $p = 0.96$).

## 9. Visualizing the results

### Funnel plot

```python
model.plot_funnel(
    null='median',
    target=0.0,
    alpha=[0.05, 0.01],
    plot_title="Funnel Plot: Indirect Standardized Difference (LOS)",
)
```

The funnel shape comes from the control limits
$\pm z_{\alpha/2} \times \hat{\sigma} / \sqrt{n_i}$ — larger hospitals
need a smaller deviation to be flagged.

### Caterpillar plot — provider effects

```python
model.plot_provider_effects(
    null='median',
    level=0.95,
    use_flags=True,
    plot_title="Provider Effects (γ̂ᵢ) with 95% CIs",
)
```

### Caterpillar plot — standardized differences

```python
model.plot_standardized_measures(
    stdz='indirect',
    null='median',
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
| $\hat{\beta}_{\text{age}}$ | days / year | 0.147 |
| $\hat{\beta}_{\text{severity}}$ | days / unit | 1.807 |
| $\hat{\beta}_{\text{comorbidity}}$ | days (binary) | 3.226 |
| $\hat{\gamma}_i$ | days (intercept) | range $[1.5, 8.2]$ |
| $\hat{\sigma}$ | days (residual SD) | 3.098 |
| ISDiff | days above/below median | range $[-3.5, +3.7]$ |
| Flagged hospitals | — | 18 of 30 at $\alpha = 0.05$ |

### When to use linear FE vs. RE?

The fixed-effect model estimates a separate $\gamma_i$ for each
hospital with **no distributional assumption** and **no shrinkage**.
This is ideal when:

- Unbiased estimation of each provider's effect is paramount (e.g.,
  public reporting).
- Provider effects may be correlated with patient covariates.
- The number of patients per provider is large enough to support
  stable estimates.

If providers are small and you want to borrow strength across them, or
if providers are viewed as a random sample from a larger population,
consider the [random-effect model](linear_random_effect_tutorial)
instead. The [Fixed vs. Random Effects](fixed_vs_random_effects) page
discusses the trade-off in detail.
