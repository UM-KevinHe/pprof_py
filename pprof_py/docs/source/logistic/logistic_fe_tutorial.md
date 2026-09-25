(logistic_fixed_effect_tutorial)=
# Tutorial: Provider Profiling with Logistic Fixed Effects

This tutorial walks through a complete provider profiling analysis
using `LogisticFixedEffectModel` — from generating a synthetic cohort,
through fitting the model and reading every number in its output, to
identifying outlier providers and visualizing the results.

By the end you will know how to:

- prepare a patient-level DataFrame for provider profiling,
- fit a logistic fixed-effect model and interpret the covariate and
  provider-effect estimates,
- compute indirect standardized ratios (ISRs) to compare providers
  on a common scale,
- test each provider against a benchmark using the exact
  Poisson-Binomial test, and
- produce caterpillar and funnel plots that make the results readable
  at a glance.

The model we are fitting is the one described in detail on the
[Logistic Fixed Effect Modeling](logistic_fixed_effect_model.md) reference
page. Here we focus on *using* it rather than deriving it.

---

## 1. Building the cohort

Real provider profiling data — CMS claims, hospital discharge records —
is patient-level, with each row representing one patient's episode of
care at a single facility. We simulate that structure here: 100
clinics, each with 50–150 patients, a 30-day readmission outcome, and
three risk factors.

```python
import pandas as pd
import numpy as np
from pprof_py import LogisticFixedEffectModel

np.random.seed(420)
n_providers = 100
n_per = np.random.randint(50, 150, n_providers)
n_total = n_per.sum()

provider_ids = []
for i, count in enumerate(n_per):
    provider_ids.extend([f"Clinic_{i+1}"] * count)

data = pd.DataFrame({
    'patient_id': range(n_total),
    'provider_id': provider_ids,
    'age': np.random.normal(65, 8, n_total),
    'chronic_conditions': np.random.randint(0, 4, n_total),
    'prior_admission': np.random.choice([0, 1], n_total, p=[0.8, 0.2]),
})

# True data-generating process
true_effects = {f"Clinic_{i+1}": np.random.normal(0, 0.5)
                for i in range(n_providers)}
data['true_effect'] = data['provider_id'].map(true_effects)

log_odds = (
    -2.5
    + 0.02 * (data['age'] - 65)
    + 0.3 * data['chronic_conditions']
    + 0.6 * data['prior_admission']
    + data['true_effect']
)
data['readmitted'] = (np.random.rand(n_total) < 1/(1+np.exp(-log_odds))).astype(int)
```

That gives us 9,888 patients across 100 clinics, with an overall
readmission rate of 13.8 %. Clinic sizes range from 50 to 149
patients (median 98). The first few rows:

```
provider_id       age  chronic_conditions  prior_admission  readmitted
   Clinic_1 59.843844                   3                0           1
   Clinic_1 69.650745                   3                1           0
   Clinic_1 64.256096                   1                1           0
   Clinic_1 61.723780                   3                0           1
   Clinic_1 59.242851                   3                1           0
```

Three risk adjusters go into the model:

- **age** — continuous, centered on 65 in the DGP. Older patients are
  at higher risk.
- **chronic_conditions** — integer 0–3, representing comorbidity burden.
- **prior_admission** — binary (0/1), capturing recent utilization
  history.

The true data-generating coefficients are $\beta_{\text{age}} = 0.02$,
$\beta_{\text{chronic}} = 0.3$, $\beta_{\text{prior}} = 0.6$, and each
clinic's baseline log-odds is drawn from $N(0, 0.5^2)$ around a global
intercept of $-2.5$.

## 2. Fitting the model

`LogisticFixedEffectModel` estimates a separate intercept $\gamma_i$
for every clinic while sharing the covariate coefficients $\beta$ across
the entire population — see Section 2.1 of the
[methodology page](logistic_fixed_effect_model.md) for the full formulation.

```python
model = LogisticFixedEffectModel(
    algorithm='Serbin',
    screen_providers=True,
    cutoff=10,
)

model.fit(
    X=data,
    y_var='readmitted',
    x_vars=['age', 'chronic_conditions', 'prior_admission'],
    provider_var='provider_id',
    max_iter=1000,
    tol=1e-6,
)
```

`algorithm='Serbin'` uses the block-diagonal Schur-complement trick
described in {cite}`logfe-Wu2022Improving`, which keeps the per-iteration
cost at $O(mp^2)$ instead of $O((m+p)^3)$. `cutoff=10` drops any
clinic with 10 or fewer patients — none are dropped here because the
smallest clinic has 50. All 100 providers are retained.

## 3. Reading the covariate coefficients

```python
summary = model.summary(test_method='wald')
print(summary)
```

```
                    estimate  std_error       stat   p_value  ci_lower  ci_upper
age                 0.014765   0.003804   3.881309  0.000104  0.007309  0.022221
chronic_conditions  0.311201   0.027841  11.177712  0.000000  0.256633  0.365769
prior_admission     0.694115   0.068312  10.160995  0.000000  0.560226  0.828003
```

Every column, in plain language:

- **estimate** — the $\hat{\beta}$ coefficients. These are on the
  log-odds scale: a one-unit increase in the covariate shifts the
  log-odds of readmission by this amount, holding the other covariates
  and the provider constant.
- **std_error** — how precisely each coefficient is estimated. Smaller
  means more information.
- **stat** and **p_value** — the Wald $z = \hat{\beta}/\text{se}$ and
  its two-sided p-value. All three covariates are highly significant.
- **ci_lower**, **ci_upper** — 95 % Wald confidence intervals.

The estimates recover the true DGP well:

| Covariate | True $\beta$ | Estimated $\hat{\beta}$ | Odds Ratio $e^{\hat{\beta}}$ |
|---|---|---|---|
| age | 0.02 | 0.0148 | 1.015 |
| chronic_conditions | 0.30 | 0.3112 | 1.365 |
| prior_admission | 0.60 | 0.6941 | 2.002 |

In clinical terms: each additional chronic condition multiplies the
odds of readmission by about 1.37, and a prior admission roughly
doubles them ($e^{0.694} \approx 2.00$). The age effect is modest —
each extra year adds about 1.5 % to the odds.

The model also reports overall fit statistics:

```python
print(f"AIC: {model.aic_:.2f}")   # 7579.92
print(f"BIC: {model.bic_:.2f}")   # 8321.43
print(f"AUC: {model.auc_:.4f}")   # 0.6888
```

An AUC of 0.69 is typical for administrative-claims–based risk models:
the three risk adjusters capture a meaningful fraction of the variation,
but a substantial share of the readmission risk remains unexplained
(as is realistic with only age, comorbidity count, and prior admission).

## 4. The provider effects

The other half of the model's output is the vector of clinic-specific
intercepts $\hat{\gamma}_i$. Each one captures how much higher or lower
the log-odds of readmission are at that clinic, after adjusting for its
patient mix.

```python
gamma = model.coefficients_['gamma']   # length-100 array
groups = model.provider_ids_            # provider labels

print(f"Gamma range: [{gamma.min():.4f}, {gamma.max():.4f}]")
print(f"Gamma median: {np.median(gamma):.4f}")
```

```
Gamma range: [-5.7303, -2.3693]
Gamma median: -3.5285
```

All gammas are negative because the baseline readmission rate is low
(13.8 %) — on the logit scale, $\text{logit}(0.138) \approx -1.83$,
and the clinic intercepts sit in a range around that once the covariate
contributions are netted out. What matters is relative position: a
clinic at $-2.37$ has much higher baseline risk than one at $-5.73$.

Standard errors for the gammas come from the inverse Fisher information
matrix (see Section 2.2 of the
[methodology page](logistic_fixed_effect_model.md)):

```python
se_gamma = np.sqrt(model.variances_['gamma'])
print(f"SE range: [{se_gamma.min():.4f}, {se_gamma.max():.4f}]")
```

```
SE range: [0.2924, 0.6389]
```

Smaller clinics have larger standard errors — more uncertainty about
their true performance — which is exactly the signal the funnel plot
(Section 8) is designed to visualize.

## 5. Predictions

```python
preds = model.predict(
    X=data,
    x_vars=['age', 'chronic_conditions', 'prior_admission'],
    provider_var='provider_id',
)
print(f"Predicted probabilities: min={preds.min():.4f}, mean={preds.mean():.4f}, max={preds.max():.4f}")
```

```
Predicted probabilities: min=0.0070, mean=0.1380, max=0.6028
```

The mean predicted probability (0.138) matches the observed readmission
rate exactly — as it should for maximum likelihood. Individual
predictions range from under 1 % (young, healthy patients at
low-readmission clinics) to 60 % (older, multi-morbid patients at
high-readmission clinics).

## 6. Standardized measures: comparing providers on a common scale

Raw gammas are hard to communicate to non-statisticians. The
**Indirect Standardized Ratio** (ISR) solves this — it answers:
"given this clinic's actual patient mix, how many readmissions did
it observe compared to how many would be *expected* if it performed
at the median clinic's level?"

```python
sm = model.calculate_standardized_measures(stdz='indirect', reference='median')
sm_df = sm['indirect']
print(sm_df.head(10))
```

```
     provider_id  indirect_ratio  indirect_rate  observed   expected
0    Clinic_1        1.257097      17.353737        17  13.523219
1   Clinic_10        0.895757      12.365584        14  15.629231
2  Clinic_100        0.579154       7.995002        11  18.993208
3   Clinic_11        1.218206      16.816860        12   9.850551
4   Clinic_12        0.452771       6.250332         4   8.834483
5   Clinic_13        0.815204      11.253577        14  17.173612
6   Clinic_14        2.163275      29.863168        24  11.094291
7   Clinic_15        1.152829      15.914350        12  10.409181
8   Clinic_16        1.803180      24.892197        14   7.764062
9   Clinic_17        0.596342       8.232273        10  16.768895
```

Reading the columns:

- **observed** — the actual number of readmissions at that clinic.
- **expected** — the number expected if the clinic performed at the
  median level, given *its* patients' risk profiles.
- **indirect_ratio** — $O_i / E_i$. A ratio of 1.0 means "as expected."
  Clinic 14 has a ratio of 2.16: it observed 24 readmissions where only
  11 were expected — more than double the benchmark rate.
- **indirect_rate** — the ISR scaled by the overall population rate
  (13.8 %). Clinic 14's adjusted rate is 29.9 %, versus the 13.8 %
  population average.

The five best and worst performers by ISR:

| | Clinic | ISR | Observed | Expected |
|---|---|---|---|---|
| Best | Clinic_54 | 0.131 | 1 | 7.66 |
| | Clinic_85 | 0.259 | 3 | 11.57 |
| | Clinic_30 | 0.352 | 5 | 14.21 |
| Worst | Clinic_63 | 2.354 | 20 | 8.50 |
| | Clinic_93 | 2.379 | 45 | 18.92 |
| | Clinic_29 | 2.443 | 16 | 6.55 |

Clinic 54's single readmission against 7.66 expected is striking, but
with so few events the estimate is noisy. That is exactly why we need
hypothesis testing — to separate real performance differences from
sampling noise.

## 7. Hypothesis testing: which providers are outliers?

The exact Poisson-Binomial test (Section 2.4 of the
[methodology page](logistic_fixed_effect_model.md)) is the recommended
default. Under the null $H_0: \gamma_i = \gamma_0$ (the median gamma),
each patient's readmission is an independent Bernoulli trial with a
known probability, so the total observed count $O_i$ follows a
Poisson-Binomial distribution. The test computes an exact p-value —
no normal approximation required.

```python
test_results = model.test(
    reference='median',
    level=0.95,
    test_method='poibin_exact',
    alternative='two_sided',
)
print(test_results[['estimate', 'z_raw', 'p_value', 'flag']].head(10))
```

```
             estimate     z_raw   p_value  flag
provider_id
Clinic_1    -3.249995  1.021081  0.307216     0
Clinic_10   -3.656766 -0.416515  0.677033     0
Clinic_100  -4.153595 -2.092793  0.036368    -1
Clinic_11   -3.289742  0.753502  0.451149     0
Clinic_12   -4.428621 -1.869366  0.061572     0
Clinic_13   -3.765100 -0.814136  0.415567     0
Clinic_14   -2.531061  3.750076  0.000177     1
Clinic_15   -3.355796  0.558359  0.576599     0
Clinic_16   -2.757958  2.273616  0.022989     1
Clinic_17   -4.113990 -1.858888  0.063043     0
```

Reading the output:

- **flag** — `+1` means the provider has significantly *more*
  readmissions than expected; `-1` means significantly *fewer*; `0`
  means not significantly different from the median at the 5 % level
  (`NA` would mean the provider could not be tested).
- **p_value** — the exact two-sided (mid-p) Poisson-Binomial p-value.
- **z_raw** — the test statistic on the z scale; for the exact test it
  is the normal quantile whose tail reproduces the exact p-value (for
  reference; the p-value is the inferential quantity).
- **estimate** — the provider's fitted effect $\hat{\gamma}_i$; the
  median it is compared with is in the `null_value` column.

The full result has more columns, described in the
[reference page](../reference/measures_tests_plots); the exact test has
no standard error or interval, so those columns are `NaN` here.

Clinic 14 (p = 0.0002, flag = +1) is the strongest outlier on the high
side. Clinic 100 (p = 0.036, flag = -1) is flagged on the low side.
Clinic 12 (p = 0.062) narrowly misses the 5 % threshold.

Overall, 29 of 100 providers are flagged: 19 higher than expected and
10 lower. In a simulation where the true provider effects are drawn
from $N(0, 0.5^2)$, this proportion is realistic — some facilities
genuinely deviate from the median.

## 8. Confidence intervals

### CIs for provider effects ($\gamma_i$)

Wald confidence intervals use the asymptotic formula:

$$
\hat{\gamma}_i \pm z_{0.975} \times \widehat{\text{se}}(\hat{\gamma}_i)
$$

```python
gamma_ci = model.calculate_confidence_intervals(
    option='gamma',
    level=0.95,
    test_method='wald',
    alternative='two_sided',
)
print(gamma_ci['gamma_ci'].head(10))
```

```
     provider_id     gamma  gamma_lower  gamma_upper
0    Clinic_1 -3.249995    -3.978365    -2.521626
1   Clinic_10 -3.656766    -4.406950    -2.906583
2  Clinic_100 -4.153595    -4.953717    -3.353474
3   Clinic_11 -3.289742    -4.103444    -2.476040
4   Clinic_12 -4.428621    -5.573762    -3.283481
5   Clinic_13 -3.765100    -4.512238    -3.017962
6   Clinic_14 -2.531061    -3.220155    -1.841967
7   Clinic_15 -3.355796    -4.162740    -2.548852
8   Clinic_16 -2.757958    -3.563606    -1.952311
9   Clinic_17 -4.113990    -4.934460    -3.293520
```

For Clinic 14: $\hat{\gamma} = -2.531$, SE = 0.352, 95 % CI =
$[-3.220, -1.842]$. The interval sits well above the median gamma
($-3.529$), consistent with the significant test result.

For Clinic 100: $\hat{\gamma} = -4.154$, SE = 0.408, 95 % CI =
$[-4.954, -3.354]$. The interval sits below the median, also
consistent with its flag = $-1$.

Score and exact Poisson-Binomial CIs are also available via
`test_method='score'` and `test_method='exact'`; see the
[methodology page](logistic_fixed_effect_model.md) Section 2.5 for the
theory behind each.

### CIs for standardized measures

Confidence intervals for ISR and DSR are obtained by transforming the
gamma CIs through the expected-count function (Section 2.5.2):

```python
sm_ci = model.calculate_confidence_intervals(
    option='SM',
    stdz='indirect',
    measure=['ratio', 'rate'],
    reference='median',
    level=0.95,
    test_method='wald',
    alternative='two_sided',
)
print(sm_ci['indirect_ratio'].head(10))
```

```
     provider_id  indirect_ratio  indirect_rate  observed   expected  ci_ratio_lower  ci_ratio_upper
0    Clinic_1        1.257097      17.353737        17  13.523219        0.677407        2.157349
1   Clinic_10        0.895757      12.365584        14  15.629231        0.455720        1.653704
2  Clinic_100        0.579154       7.995002        11  18.993208        0.274811        1.155800
3   Clinic_11        1.218206      16.816860        12   9.850551        0.604994        2.230614
4   Clinic_12        0.452771       6.250332         4   8.834483        0.152526        1.221272
5   Clinic_13        0.815204      11.253577        14  17.173612        0.413358        1.516942
6   Clinic_14        2.163275      29.863168        24  11.094291        1.290575        3.307138
7   Clinic_15        1.152829      15.914350        12  10.409181        0.575536        2.103350
8   Clinic_16        1.803180      24.892197        14   7.764062        0.971456        2.971002
9   Clinic_17        0.596342       8.232273        10  16.768895        0.276670        1.217648
```

Clinic 14's ISR CI is $[1.291, 3.307]$ — the lower bound exceeds 1.0,
confirming that its elevated readmission rate is statistically
significant. Clinic 100's ISR CI is $[0.275, 1.156]$, spanning 1.0,
so while flagged at $\alpha = 0.05$ by the Poisson-Binomial test, the
Wald CI for its ISR just includes 1.0 — a reminder that Wald and exact
methods don't always agree at the boundary.

## 9. Visualizing the results

### Caterpillar plot — provider effects

The caterpillar plot sorts all 100 clinics by their estimated
$\hat{\gamma}_i$ and draws a point-and-interval for each. Providers
whose confidence interval excludes the median (the dashed reference
line) are colored; others are gray.

```python
model.plot_provider_effects(
    level=0.95,
    test_method='wald',
    use_flags=True,
    reference='median',
    title="Provider Effects: Adjusted Log-Odds (Gamma)",
    figure_size=(10, 6),
)
```

Clinics cluster in a band around $-3.5$, with a few strong outliers
on each side — exactly the pattern the Poisson-Binomial test
identified in Section 7.

### Caterpillar plot — indirect standardized ratios

The same layout, but now on the ISR scale (O/E). A horizontal
reference line at 1.0 marks "as expected."

```python
model.plot_standardized_measures(
    stdz='indirect',
    measure='ratio',
    level=0.95,
    test_method='wald',
    use_flags=True,
    reference='median',
    title="Provider Standardized Ratios (Indirect O/E)",
    figure_size=(10, 6),
)
```

### Funnel plot

The funnel plot adds a precision dimension: the x-axis shows the
expected count $E_i$ (a proxy for clinic size and patient-risk
intensity), and the curved control limits tighten as $E_i$ grows.
Providers outside the funnel are the ones whose ISR is too far from
1.0 to be explained by chance, given their volume.

```python
model.plot_funnel(
    test_method='poibin_exact',
    reference='median',
    target=1.0,
    alpha=[0.05, 0.01],
    plot_title="Funnel Plot: Indirect Standardized Ratios (O/E)",
    ylab="Indirect Standardized Ratio (O/E)",
)
```

Small clinics (left side, wider funnel) need a very extreme ISR to
breach the limits; large clinics (right side, narrow funnel) are held
to a tighter standard. This is the right behavior: it takes more
evidence to confidently call a small clinic an outlier.

### Forest plot — covariate coefficients

Finally, the forest plot shows the three $\hat{\beta}$ estimates with
their 95 % Wald CIs. All three exclude zero, confirming that age,
chronic conditions, and prior admission are each independently
associated with readmission risk.

```python
model.plot_coefficient_forest(
    plot_title="Forest Plot of Covariate Log-Odds Ratios (Beta)",
)
```

## 10. Quick interpretation guide

| Quantity | Scale | "Good" direction | Example from this tutorial |
|---|---|---|---|
| $\hat{\beta}$ | log-odds | Depends on context | $\hat{\beta}_{\text{chronic}} = 0.311$ (log-odds per condition) |
| $e^{\hat{\beta}}$ | odds ratio | Depends on context | $e^{0.311} = 1.365$ (37 % higher odds per condition) |
| $\hat{\gamma}_i$ | log-odds | Lower = fewer readmissions | Clinic 54: $-5.73$; Clinic 29: $-2.37$ |
| ISR (O/E) | ratio | < 1 = fewer events than expected | Clinic 54: 0.131; Clinic 29: 2.443 |
| flag | −1/0/+1 | −1 or 0 | 19 flagged high, 10 flagged low, 71 not flagged |

