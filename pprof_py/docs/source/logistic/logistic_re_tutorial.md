(logistic_random_effect_tutorial)=
# Tutorial: Provider Profiling with Logistic Random Effects

This tutorial walks through a complete provider profiling analysis
using `LogisticRandomEffectModel` — a Generalized Linear Mixed Model
(GLMM) with a logit link and random provider intercepts. If you have
already worked through the
[fixed-effect logistic tutorial](logistic_fixed_effect_tutorial), the
workflow will feel familiar; the key difference is that the random-effect
model **shrinks** each provider's estimate toward the grand mean,
borrowing strength across providers and producing more stable estimates
for small hospitals.

By the end you will know how to:

- fit a logistic random-intercept model and interpret both the fixed
  effects and the provider-specific BLUPs,
- compute indirect standardized ratios using the BLUPs,
- test each provider against a benchmark using a Z-test on the
  posterior standard errors, and
- produce caterpillar, funnel, and forest plots.

The statistical methodology is described in full on the
[Logistic Random Effect Modeling](logistic_random_effect_model_stats)
reference page.

---

## 1. Building the cohort

We simulate 25 hospitals, each with 40–120 patients, a binary
complication outcome, and three risk adjusters. The true provider effects
are drawn from $N(0, 0.6^2)$.

```python
import pandas as pd
import numpy as np
from pprof_py import LogisticRandomEffectModel

np.random.seed(789)
n_providers = 25
n_per = np.random.randint(40, 120, n_providers)
n_total = n_per.sum()

provider_ids = []
for i, count in enumerate(n_per):
    provider_ids.extend([f"Hospital_{i+1}"] * count)

data = pd.DataFrame({
    'patient_id': range(n_total),
    'provider_id': provider_ids,
    'age': np.random.normal(60, 10, n_total),
    'severity': np.random.gamma(2.5, 1.2, n_total),
    'urgent': np.random.choice([0, 1], n_total, p=[0.7, 0.3]),
})

true_re_sd = 0.6
true_effects = {f"Hospital_{i+1}": np.random.normal(0, true_re_sd)
                for i in range(n_providers)}
data['true_re'] = data['provider_id'].map(true_effects)

log_odds = (
    -2.0
    + 0.03 * (data['age'] - 60)
    + 0.25 * data['severity']
    + 0.5 * data['urgent']
    + data['true_re']
)
data['complication'] = (np.random.rand(n_total) < 1/(1+np.exp(-log_odds))).astype(int)
```

That gives us 2,002 patients across 25 hospitals, with an overall
complication rate of 24.0 %. Hospital sizes range from 41 to 119
patients (median 78).

## 2. Fitting the model

`LogisticRandomEffectModel` fits a GLMM of the form

$$
\text{logit}(p_{ij}) = \mathbf{X}_{ij}^\top \boldsymbol{\beta} + u_i, \qquad u_i \sim N(0, \sigma_u^2)
$$

using Penalized Iteratively Reweighted Least Squares (PIRLS) with
Laplace approximation — a pure-Python reimplementation of R's `lme4`
algorithm. No R or `pymer4` dependency is required.

```python
model = LogisticRandomEffectModel()

model.fit(
    X=data,
    y_var='complication',
    x_vars=['age', 'severity', 'urgent'],
    provider_var='provider_id',
)
```

## 3. Reading the fixed effects

```python
summary = model.summary()
print(summary)
```

```
             Estimate  Std.Error    z value      Pr(>|z|)
(Intercept) -4.744732   0.384863 -12.328360  6.373025e-35
age          0.042402   0.005639   7.520125  5.472391e-14
severity     0.248302   0.027888   8.903560  5.408364e-19
urgent       0.504732   0.117978   4.278187  1.884214e-05
```

The columns mirror what `lme4::summary()` shows in R:

- **Estimate** — the $\hat{\beta}$ on the log-odds scale.
- **Std.Error** — Wald standard error.
- **z value** and **Pr(>|z|)** — Wald test and two-sided p-value.

All three risk adjusters are highly significant. Comparing to the true
data-generating values:

| Covariate | True $\beta$ | Estimated $\hat{\beta}$ | Odds Ratio $e^{\hat{\beta}}$ |
|---|---|---|---|
| age (per year) | 0.03 | 0.042 | 1.043 |
| severity | 0.25 | 0.248 | 1.282 |
| urgent | 0.50 | 0.505 | 1.656 |

Each unit of severity score multiplies the odds of complication by
about 1.28. An urgent case has 66 % higher odds ($e^{0.505} \approx 1.66$)
than a non-urgent one, holding age and severity fixed.

The model also reports:

```python
print(f"AIC: {model.aic_:.2f}")   # 2047.79
print(f"BIC: {model.bic_:.2f}")   # 2075.80
print(f"Log-likelihood: {model.loglike_:.2f}")  # -1018.89
```

## 4. The random effects (BLUPs) and shrinkage

The estimated variance component is $\hat{\sigma}_u^2 = 0.1249$
($\hat{\sigma}_u = 0.353$), compared to the true $\sigma_u = 0.6$.
Some shrinkage of the variance estimate toward zero is expected with
only 25 groups.

```python
re = model.get_random_effects()
print(re.head(10))
```

```
Hospital_1    -0.143147
Hospital_10    0.355628
Hospital_11    0.151774
Hospital_12   -0.112255
Hospital_13   -0.163386
Hospital_14    0.066239
Hospital_15    0.359097
Hospital_16    0.738506
Hospital_17    0.079881
Hospital_18    0.051852
```

The BLUPs range from $-0.641$ to $+0.739$ (SD = 0.279). This is
narrower than the raw provider-level log-odds would be — that's
**shrinkage** at work. Hospitals with few patients or moderate raw
rates are pulled toward the overall mean, stabilizing the estimates.
Only hospitals with large sample sizes and genuinely extreme rates
retain large BLUPs.

### Predictions: fixed-effects-only vs. full model

```python
preds_fe = model.predict(X=data, x_vars=['age', 'severity', 'urgent'])
fitted_full = model.fitted_

print(f"FE only:  min={preds_fe.min():.4f}, mean={preds_fe.mean():.4f}, max={preds_fe.max():.4f}")
print(f"FE + RE:  min={fitted_full.min():.4f}, mean={fitted_full.mean():.4f}, max={fitted_full.max():.4f}")
```

```
FE only:  min=0.0419, mean=0.2336, max=0.9310
FE + RE:  min=0.0297, mean=0.2396, max=0.9401
```

`predict()` returns probabilities from fixed effects alone (useful for
scoring new patients at an unknown hospital). `fitted_` includes the
random effects — giving the personalized probability for each patient
at their actual hospital.

## 5. Standardized measures

The **Indirect Standardized Ratio** (ISR) compares each hospital's
predicted complication burden (using its BLUP) to what would be expected
if that hospital performed at the median level:

```python
sm = model.calculate_standardized_measures(stdz='indirect', reference='median')
sm_df = sm['indirect']
print(sm_df.head(10))
```

```
      provider_id  indirect_ratio  indirect_rate  observed   expected
0   Hospital_1        0.801878      19.265898      17.0  21.200234
1  Hospital_10        1.432589      34.419342      19.0  13.262703
2  Hospital_11        1.102326      26.484456      30.0  27.215180
3  Hospital_12        0.851397      20.455654      23.0  27.014409
4  Hospital_13        0.800663      19.236709      21.0  26.228262
5  Hospital_14        1.034282      24.849631      16.0  15.469670
6  Hospital_15        1.331996      32.002506      32.0  24.024093
7  Hospital_16        1.721810      41.368153      51.0  29.620000
8  Hospital_17        1.065650      25.603279      12.0  11.260733
9  Hospital_18        1.014591      24.376529      17.0  16.755526
```

Note that `observed` here is the sum of the full model's *fitted
probabilities* (not the raw binary outcomes) — this is the convention
for RE models, where the BLUP-adjusted probability is the best estimate
of each patient's true risk at their hospital. `expected` uses the
median BLUP as the reference.

Hospital 16 stands out: ISR = 1.72, meaning 72 % more complications
than expected at the median level. Hospital 25 is at the other extreme
with ISR = 0.40. The full ISR range is $[0.402, 1.722]$.

## 6. Hypothesis testing

The Z-test divides each BLUP by its posterior standard error and
compares to a standard normal:

$$
Z_i = \frac{\hat{u}_i - u_0}{\widehat{\text{se}}(\hat{u}_i)}
$$

```python
test_results = model.test(reference='median', level=0.95, alternative='two_sided')
print(test_results[['estimate', 'se', 'z_raw', 'p_value', 'flag', 'ci_lower', 'ci_upper']].head(10))
```

```
             estimate        se     z_raw   p_value  flag  ci_lower  ci_upper
provider
Hospital_1  -0.143147  0.214148 -0.977764  0.328191     0 -0.562868  0.276575
Hospital_10  0.355628  0.231867  1.248078  0.212003     0 -0.098824  0.810079
Hospital_11  0.151774  0.193643  0.441713  0.658697     0 -0.227760  0.531308
Hospital_12 -0.112255  0.200886 -0.888534  0.374254     0 -0.505985  0.281475
Hospital_13 -0.163386  0.203617 -1.127735  0.259432     0 -0.562468  0.235695
Hospital_14  0.066239  0.230360  0.000000  1.000000     0 -0.385258  0.517736
Hospital_15  0.359097  0.194503  1.505670  0.132152     0 -0.022122  0.740316
Hospital_16  0.738506  0.173290  3.879433  0.000105     1  0.398864  1.078148
Hospital_17  0.079881  0.255352  0.053424  0.957394     0 -0.420600  0.580363
Hospital_18  0.051852  0.224577 -0.064064  0.948919     0 -0.388310  0.492014
```

`reference='median'` compares each BLUP with the median BLUP; without
it the reference is 0, the random-effect mean. The Wald test also
reports a 95 % interval for each BLUP (`ci_lower`, `ci_upper`), which
excludes the reference exactly when the hospital is flagged.

Only **2 of 25** hospitals are flagged at the 5 % level: Hospital 16
(flag = +1, $p = 0.0001$, $Z = 3.88$) and one flagged low. With
25 providers and BLUPs already shrunk toward the mean, the test is
conservative — only the strongest deviations survive. This is by
design: RE models trade power for stability.

## 7. Confidence intervals

### CIs for random effects ($u_i$)

```python
alpha_ci = model.calculate_confidence_intervals(option='alpha', level=0.95)
alpha_ci_df = alpha_ci['alpha_ci']
print(alpha_ci_df.head(10))
```

```
      provider_id     alpha  alpha_lower  alpha_upper
0   Hospital_1 -0.143147    -0.562868     0.276575
1  Hospital_10  0.355628    -0.098824     0.810079
2  Hospital_11  0.151774    -0.227760     0.531308
3  Hospital_12 -0.112255    -0.505985     0.281475
4  Hospital_13 -0.163386    -0.562468     0.235695
5  Hospital_14  0.066239    -0.385258     0.517736
6  Hospital_15  0.359097    -0.022122     0.740316
7  Hospital_16  0.738506     0.398864     1.078148
8  Hospital_17  0.079881    -0.420600     0.580363
9  Hospital_18  0.051852    -0.388310     0.492014
```

Hospital 16's CI is $[0.399, 1.078]$ — entirely above zero,
confirming a significantly elevated complication rate. Most other
hospitals' CIs span zero, consistent with not being flagged.

### CIs for standardized measures

```python
sm_ci = model.calculate_confidence_intervals(
    option='SM', stdz='indirect', reference='median',
    measure=['ratio'], level=0.95,
)
print(sm_ci['indirect_ratio'].head(10))
```

```
      provider_id  indirect_ratio     lower     upper
0   Hospital_1        0.801878  0.527018  1.220088
1  Hospital_10        1.432589  0.909402  2.256771
2  Hospital_11        1.102326  0.754190  1.611163
3  Hospital_12        0.851397  0.574299  1.262197
4  Hospital_13        0.800663  0.537194  1.193352
5  Hospital_14        1.034282  0.658501  1.624507
6  Hospital_15        1.331996  0.909791  1.950133
7  Hospital_16        1.721810  1.225972  2.418186
8  Hospital_17        1.065650  0.646038  1.757806
9  Hospital_18        1.014591  0.653327  1.575618
```

Hospital 16's ISR CI is $[1.226, 2.418]$ — entirely above 1.0. This
is consistent with the Z-test: even after shrinkage, its complication
rate is significantly elevated.

## 8. Visualizing the results

### Funnel plot

The funnel plot places ISR on the y-axis and expected count on the
x-axis. Poisson-based control limits tighten as the expected count
grows — small hospitals need an extreme ISR to be flagged.

```python
model.plot_funnel(
    test_method='wald',
    reference='median',
    target=1.0,
    alpha=[0.05, 0.01],
    plot_title="Funnel Plot: Indirect Standardized Ratio (O/E)",
)
```

### Caterpillar plot — BLUPs

```python
model.plot_provider_effects(
    level=0.95,
    use_flags=True,
    reference='median',
    plot_title="Provider Random Effects (BLUPs, Log-Odds Scale)",
)
```

The shrinkage is visible here: the BLUPs are clustered near zero, with
CIs much wider for small hospitals. Hospital 16 is the only provider
whose CI clearly separates from the rest on the high side.

### Caterpillar plot — ISR

```python
model.plot_standardized_measures(
    stdz='indirect',
    measure='ratio',
    level=0.95,
    use_flags=True,
    reference='median',
    plot_title="Indirect Standardized Ratio (O/E) with CIs",
)
```

### Forest plot — fixed effects

```python
model.plot_coefficient_forest(
    plot_title="Forest Plot of Covariate Coefficients (Log-Odds)",
)
```

## 9. Quick interpretation guide

| Quantity | Scale | Interpretation |
|---|---|---|
| $\hat{\beta}$ | log-odds | Population-average covariate effect |
| $e^{\hat{\beta}}$ | odds ratio | 1.28 per unit of severity, 1.66 for urgent |
| $\hat{u}_i$ (BLUP) | log-odds | Provider deviation from mean — shrunk toward zero |
| $\sigma_u^2$ | variance | 0.125 — moderate provider-level heterogeneity |
| ISR (O/E) | ratio | < 1 = better than median; > 1 = worse |
| flag | −1/0/+1 | 2 of 25 flagged in this cohort |

### Fixed effects vs. random effects — when to use which?

The [fixed-effect model](logistic_fixed_effect_tutorial) estimates a
separate intercept $\gamma_i$ for every provider without any
distributional assumption. The random-effect model assumes $u_i \sim
N(0, \sigma_u^2)$, which introduces **shrinkage**: small or moderate
providers are pulled toward the mean, producing more stable but
potentially biased estimates. The
[Fixed vs. Random Effects](fixed_vs_random_effects) page discusses the
trade-off in detail.

As a rule of thumb: use fixed effects when unbiased estimation of each
individual provider is the primary goal (e.g., public reporting); use
random effects when you have many small providers and want to stabilize
rankings or when providers are a sample from a larger population.
