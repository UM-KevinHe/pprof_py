# Chapter 11 — Group Lasso for Cox Models

## 11.1 The same problem, on the hazard scale

[Chapter 8](08_penalized_regression) penalizes the Cox partial
likelihood one coefficient at a time. The same categorical-covariate
problem the
[group lasso logistic chapter](../logistic/group_lasso_logistic)
opens with shows up here too: a one-hot-encoded dialysis access type
(fistula, graft, catheter) becomes two dummy columns, and elastic net
has no way to know they're one clinical decision rather than two
independent ones — it can happily keep the catheter effect while
dropping the graft effect, which answers a question ("does access
type matter?") with an answer that doesn't actually correspond to any
real clinical distinction. `GroupLassoCoxPH` extends `PenalizedCoxPH`
with exactly the group penalty [the group lasso logistic
chapter's Section 2](../logistic/group_lasso_logistic) derives —
$\lambda\left[(1-\alpha)\sum_g m_g\|\beta_g\|_2 + \alpha\sum_j
\text{pf}_j|\beta_j|\right]$ in place of the elastic net term — applied
to the partial-likelihood score instead of the binomial or Gaussian
one. That derivation isn't repeated here.

Architecturally, this chapter's classes are also different from the
logistic and linear ones you may have just read: `GroupLassoCoxPH` and
`GroupLassoCoxPHCV` are built as extensions of `PenalizedCoxPH` and
`PenalizedCoxPHCV` (sharing input validation, standardization, and
path-storage code with them, via two internal base classes), not as
standalone implementations. In practice this means the Cox family's
group lasso classes are noticeably more complete than their logistic
and linear counterparts — better input validation, a `nonzero_features()`
convenience method, and an explicit, helpful error when you call
`predict()` on a multi-lambda fit without saying which lambda you
mean.

## 11.2 Extending the cohort with a categorical group

The running ESRD cohort from [Chapter 0](00_start_here) gets one
addition here: a categorical dialysis access type, one-hot encoded,
plus a two-column noise lab panel — the same "real group, real group,
noise group" structure the logistic and linear group lasso chapters
use, so the three chapters' results are directly comparable.

```python
import numpy as np
import pandas as pd

def make_esrd_cohort_grouped(n_patients=4000, n_facilities=40, seed=0):
    rng = np.random.default_rng(seed)
    facility_id = rng.integers(0, n_facilities, n_patients)
    facility_quality = rng.normal(0, 0.25, n_facilities)

    age = rng.normal(62, 14, n_patients).clip(18, 95)
    sex = rng.binomial(1, 0.45, n_patients)
    diabetes = rng.binomial(1, 0.45, n_patients)
    comorbidity_count = rng.poisson(1.3, n_patients)
    vintage_years = rng.exponential(2.0, n_patients)

    # Categorical group: dialysis access type ("Fistula" is the reference level)
    access = rng.choice(["Fistula", "Graft", "Catheter"], size=n_patients, p=[0.55, 0.25, 0.20])
    access_graft = (access == "Graft").astype(int)
    access_catheter = (access == "Catheter").astype(int)

    lab_g = rng.normal(0, 1, n_patients)   # noise
    lab_h = rng.normal(0, 1, n_patients)   # noise

    log_hazard = (
        -4.2 + 0.045 * (age - 62) + 0.35 * diabetes + 0.20 * comorbidity_count
        + 0.30 * access_graft + 0.55 * access_catheter + facility_quality[facility_id]
    )
    death_time = rng.exponential(1.0 / np.exp(log_hazard))
    censor_time = rng.uniform(0.5, 4.0, n_patients)
    time = np.minimum(death_time, censor_time)
    death = (death_time <= censor_time).astype(int)

    return pd.DataFrame({
        "patient_id": np.arange(n_patients), "facility_id": facility_id,
        "age": age.round(1), "sex": sex, "diabetes": diabetes,
        "comorbidity_count": comorbidity_count, "vintage_years": vintage_years.round(2),
        "access_graft": access_graft, "access_catheter": access_catheter,
        "lab_g": lab_g.round(3), "lab_h": lab_h.round(3),
        "time": time.round(3), "death": death,
    })

cohort = make_esrd_cohort_grouped()
candidates = ["age", "sex", "diabetes", "comorbidity_count", "vintage_years",
              "access_graft", "access_catheter", "lab_g", "lab_h"]
X = cohort[candidates]
time, death = cohort["time"], cohort["death"]

# 0 = unpenalized; singleton groups (4, 5) behave exactly like an
# ordinary lasso term for that one column.
groups = np.array([0, 4, 2, 2, 5, 1, 1, 3, 3])
```

4,000 patients, 40 facilities, 7.6% event rate — deliberately in the
same range as Chapter 8's own cohort, since Section 11.4 draws a
direct comparison to what Chapter 8 found there. `sex` and
`vintage_years` are included as their own singleton groups (4 and 5):
neither actually drives the hazard in this cohort's data-generating
process, alongside the explicit noise panel (`lab_g`, `lab_h`, group
3) — a slightly more honest test of variable selection than giving
every non-noise column a real effect.

## 11.3 Fitting the path

```python
from pprof_py import GroupLassoCoxPH

m = GroupLassoCoxPH(groups=groups, alpha=0.0)
m.fit(X, duration=time, event=death)

m.lambda_max_     # 0.018695372653239166
m.n_groups_        # 5 -- the four penalized groups plus the two singletons
m.group_sizes_     # array([2, 2, 2, 1, 1])
```

Cox has no intercept, so `coef_at()`, `predict_linear()`, and
`predict_partial_hazard()` work without any intercept-interpolation
concern, and this was confirmed directly:

```python
lam = m.lambda_path_[60]
m.predict_partial_hazard(X.values[:3], lambda_value=lam)
# [31.805, 6.502, 30.534], confirmed equal to exp(X @ coef_at(lam))
# computed independently, to full floating-point precision.
```

Group entry order tells you which groups this cohort's data actually
supports:

```python
for g in range(1, m.n_groups_ + 1):
    active = np.array([ag[g - 1] for ag in m.active_groups_])
    first = np.argmax(active) if active.any() else None
    print(g, first, m.lambda_path_[first] if first is not None else None)
```
```
group 1 (access type)          first active at index  8, lambda=0.00889
group 2 (diabetes + comorbidity) first active at index  1, lambda=0.01704
group 3 (noise lab panel)      first active at index 25, lambda=0.00183
group 4 (sex, singleton)       first active at index 36, lambda=0.00066
group 5 (vintage_years, singleton) first active at index 21, lambda=0.00265
```

The comorbidity group enters first, access type not far behind — both
carry real signal in the data-generating process. The noise lab panel
enters *before* the `sex` singleton here, which is a fair result, not
a discouraging one: `sex` and `vintage_years` are exactly as
uninformative as `lab_g`/`lab_h` in this cohort (neither appears in
`log_hazard` above), so there's no reason to expect the noise panel to
be *last*, only for all four to enter well after the two groups that
actually matter — which they do.

## 11.4 Cross-validation with `GroupLassoCoxPHCV`

`GroupLassoCoxPHCV` inherits three refinements over the plain
`PenalizedCoxPHCV` you saw in Chapter 8, all from `grplasso`'s own
methodology: **event-stratified folds** (so every fold gets a
comparable share of the limited number of deaths — the same concern
[Section 7 of the penalized logistic chapter](../logistic/penalized_logistic)
raises for a rare binary outcome, here for a rare *event*, which is
the more usual case for survival data), **saturated-lambda
elimination** (lambda values where any fold's held-out deviance came
back non-finite — numerically degenerate, typically from an
extremely small lambda pushing a fold's partial likelihood to
saturation — are dropped from consideration entirely, rather than
distorting the mean), and a **strata coverage check** that warns if
any stratum is entirely missing from a training fold.

```python
from pprof_py import GroupLassoCoxPHCV

cv = GroupLassoCoxPHCV(groups=groups, alpha=0.0, n_lambda=50, n_folds=5, random_state=0)
cv.fit(X, duration=time, event=death)   # se_rule='min' by default

cv.lambda_min_   # 0.002365958818367196
cv.lambda_1se_   # 0.01870537265323917
cv.coef_          # at lambda_min_, the default
```
```
age                  0.0412
sex                  0.0000
diabetes             0.2435
comorbidity_count    0.2511
vintage_years       -0.0021
access_graft         0.1482
access_catheter      0.4426
lab_g                0.0000
lab_h                0.0000
```

At `lambda_min_`, both real groups are fully in, both noise-adjacent
singletons (`sex`, `vintage_years`) are at or near zero, and the noise
panel is zeroed as a unit — group lasso's structured selection working
correctly.


## 11.5 Selecting `lambda_1se`

`GroupLassoCoxPHCV` takes the lambda-selection rule as `se_rule`
(`"min"` or `"1se"`), like every CV class, but defaults to `"min"`
(the Cox CV classes' default; the others default to `"1se"`).

```python
cv_1se = GroupLassoCoxPHCV(groups=groups, alpha=0.0, n_lambda=50, n_folds=5,
                            random_state=0, se_rule="1se")
cv_1se.fit(X, duration=time, event=death)
cv_1se.coef_
```
```
age                  0.0412
sex                  0.0000
diabetes             0.0000
comorbidity_count    0.0000
vintage_years        0.0000
access_graft         0.0000
access_catheter      0.0000
lab_g                0.0000
lab_h                0.0000
```

Every penalized group is zeroed at `lambda_1se_` — only the
unpenalized `age` coefficient survives. This is a striking, exact
echo of what
[Chapter 8 §8.5](08_penalized_regression) found for plain
element-wise `PenalizedCoxPHCV` on a similarly-sized cohort:
*"Switching to the more conservative `se_rule='1se'` on this
same data pushes the penalty hard enough to zero out every covariate
except age."* With roughly 300 events spread across a 4,000-patient
cohort, the same event-count scarcity Chapter 8 and this guide's
logistic chapter both point to shows up a third time here, for a third
penalty shape — `lambda_1se_`'s one-standard-error margin is
genuinely wide when there isn't much data to estimate the standard
error from in the first place, and it prefers the simplest possible
model whenever the extra complexity can't clear that margin.

`GroupLassoCoxPHCV` also correctly raises this package's canonical
`NotFittedError` — not a generic `AttributeError` — if you call
`predict()` before `fit()`, inherited from the same base class
`PenalizedCoxPHCV` uses:

```python
from pprof_py import GroupLassoCoxPHCV, NotFittedError
try:
    GroupLassoCoxPHCV(groups=groups).predict(X.values[:3])
except NotFittedError as e:
    print(e)   # "This GroupLassoCoxPHCV instance is not fitted yet. Call `fit` first."
```

This is consistent with every other CV class in the package —
all raise `NotFittedError` for this condition.

## 11.6 What's next

Chapter 12 covers `ProviderPenalizedCoxPH` — the two-layer model that
jointly estimates provider effects and penalized covariate selection,
this package's central methodological contribution, extending the
plain `PenalizedCoxPH` machinery from Chapter 8 the way this chapter
extended it for groups.
