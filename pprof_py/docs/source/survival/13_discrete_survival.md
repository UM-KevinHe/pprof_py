# Chapter 13 — Discrete-Time Survival Models

## 13.1 When "continuous time" is the wrong assumption

Every Cox model in this guide assumes, at least in principle, that an
event could occur at any real-valued moment — ties are handled
(Section 2.4) but treated as an approximation to an underlying
continuous process. Plenty of real follow-up data isn't like that at
all. A dialysis facility's chart review happens annually: a patient is
known to have died sometime in "year 2," not at a specific day within
it. A claims database resolves hospitalization to the *month* it was
billed in. An interval-censored study only confirms a patient's status
at scheduled visits. In all of these, the event time is genuinely,
structurally discrete and coarse — not a continuous time rounded for
convenience — and a handful of covariate patterns share dozens or
hundreds of exactly tied event times, not by coincidence but by
construction. `DiscreteSurvival` models this directly instead of
treating it as a Cox tie-handling problem: the outcome is "did the
event happen in period $k$," one binary question per patient per
period, with its own baseline rate that's free to take any shape
across periods.

## 13.2 The model: one logistic regression per timepoint, tied together

$$
\text{logit}\big(h(t_k \mid Z_i)\big) = \alpha_k + Z_i^\top\beta,
\qquad k = 1, \dots, K
$$

$h(t_k \mid Z_i)$ is patient $i$'s **conditional hazard** at period
$k$ — the probability they experience the event *during* period $k$,
*given* they were still at risk at the start of it. $\alpha_k$ is a
free baseline parameter per discrete period (unpenalized — there are
usually few enough periods that they don't need shrinking, and
shrinking them would bias the baseline rate the same way shrinking a
Cox baseline hazard would), and $\beta$ is the usual penalized
covariate effect, shared across all periods (a proportional-odds-style
assumption: a covariate's effect on the hazard is the same multiplier
at every timepoint). The R equivalent this class's own docstring names
is closest to `glmnet` fit on **person-period expanded data** with a
complementary log-log or logit link — which is also the cleanest way
to understand what `DiscreteSurvival` does internally.

**Person-period expansion** turns one row per patient into one row per
(patient, period-at-risk) pair: a patient who survived to period 3
event-free contributes 3 rows (period 1: no event, period 2: no event,
period 3: no event); a patient who died in period 2 contributes 2 rows
(period 1: no event, period 2: event). Fitting the model above is then
exactly a penalized logistic regression on this expanded data, with
one dummy-coded intercept per period standing in for $\alpha_k$ — the
same lasso machinery [Chapter 1](../logistic/penalized_logistic)
describes, applied to data reshaped this specific way, rather than a new
optimization method.

## 13.3 The running example: annual chart review

An interval-censored version of the ESRD cohort: [Chapter 0's](00_start_here)
`cohort`, with follow-up known only to the year (`time_year`
$\in \{1, 2, 3, 4\}$), the way an annual chart-review protocol would
report it. A death or censoring during year $k$ is recorded as year $k$,
so `time_year` is the continuous follow-up rounded up:

```python
import numpy as np
from pprof_py import DiscreteSurvival, DiscreteSurvivalCV

cohort["time_year"] = np.ceil(cohort["time"]).astype(int)   # 1-4: the year of death or censoring

X = cohort[["age", "sex", "diabetes", "comorbidity_count"]]
time = cohort["time_year"]
event = cohort["death"]
print(time.value_counts().sort_index().to_dict(), f"event rate {event.mean():.2%}")

model = DiscreteSurvival(penalty_type="lasso")
model.fit(X, time, event)

print(f"lambda_max = {model.lambda_max_:.6f}")
print("timepoints:", model.timepoint_map_)
print("coef_path_:", model.coef_path_.shape, " alpha_path_:", model.alpha_path_.shape)
```
```
{1: 700, 2: 1115, 3: 1112, 4: 1073} event rate 6.48%
lambda_max = 0.040016
timepoints: [1. 2. 3. 4.]
coef_path_: (100, 4)  alpha_path_: (100, 4)
```

4,000 patients in 40 facilities; `alpha_path_` holds one baseline
parameter per year at each of the 100 lambdas. `penalty_type` is `'lasso'`, the
only penalty here: group penalties are not implemented for discrete-time
survival (R's `DiscSurv` is lasso-only too), and `'group_lasso'` or
`'sparse_group_lasso'` raise an error. Per-feature `penalty_factor`
values work as in [Chapter 1](../logistic/penalized_logistic).

`coef_at(lambda_value)` interpolates continuously, the same way
[Chapter 1's](../logistic/penalized_logistic) `coef_at()` does.

## 13.4 Reading the path

```python
print([int((model.coef_path_[i] != 0).sum()) for i in [0, 20, 40, 60, 80, 99]])
print(pd.Series(model.coef_path_[60], index=X.columns).round(4).to_string())
print("converged at", int(model.converged_.sum()), "of", model.converged_.size, "lambdas")
```
```
[0, 3, 4, 4, 4, 4]
age                  0.0476
sex                 -0.0431
diabetes             0.1991
comorbidity_count    0.1858
converged at 100 of 100 lambdas
```

`converged_` records each lambda separately.
This class's default `tol` (`1e-4`) is considerably looser than the
`1e-9`/`1e-7` used elsewhere in this guide, which makes clean
convergence easier to reach in practice.

`predict()` supports `type='link'` (default), `type='hazard'`, and
`type='survival'`.  The latter two require a `time=` argument and
delegate to `predict_hazard()`/`predict_survival()` internally
(Section 13.5).

## 13.5 `predict_hazard()`/`predict_survival()`: person-period or one column per time

Unlike `CoxPH.predict_partial_hazard()` (a Cox partial hazard is a single,
time-invariant relative-risk number), a discrete-time hazard is one number
*per patient per period*. With `time=`, both methods return the person-period
form, one value per period up to each patient's time:

```python
X3, time3 = X.values[:3], time.values[:3]
print("time3:", time3)
print("hazard:  ", model.predict_hazard(X3, time3, which=60).round(4))    # one per person-period row
print("survival:", model.predict_survival(X3, time3, which=60).round(4))  # one per patient, at its time
```
```
time3: [4 2 2]
hazard:   [0.0221 0.019  0.017  0.0133 0.0068 0.0058 0.0273 0.0234]
survival: [0.9304 0.9874 0.9499]
```

The hazard has `4 + 2 + 2 = 8` entries, one per person-period row; the
survival probability is one number per patient, at its own `time`.

Internally, both methods map `time` into the integer codes that
`self.timepoint_map_` established during `fit()` via
`np.searchsorted`, selecting the correct column of `alpha_path_`
for each query point.

Without `time=`, they return an `(n, K)` array, one column per time point.
The signature is the one `ProviderPenalizedDiscreteSurvival`
([Chapter 14](14_provider_discrete_survival)) uses after its `provider_id`:
`(X, time=None, lambda_value=None, which=None)`, where `which` picks a path
point by index and `lambda_value` a lambda (by default, the last).

## 13.6 Cross-validation with `DiscreteSurvivalCV`

```python
cv = DiscreteSurvivalCV(n_folds=5, random_state=0, penalty_type="lasso")
cv.fit(X, time, event)
print(f"lambda_min = {cv.lambda_min_:.6f}, lambda_1se = {cv.lambda_1se_:.6f}")
```
```
lambda_min = 0.001693, lambda_1se = 0.019015
```

Folds here are **event-stratified with an added timepoint-coverage
retry**: `_assign_folds()` doesn't just balance events across folds
the way `PenalizedLogisticCV` does — it checks that *every training
fold* still contains at least one observation at *every* discrete
timepoint (otherwise a baseline hazard parameter $\alpha_k$ for a
timepoint absent from a training fold would be inestimable), retrying
the random assignment up to `max_fold_retries` (default 100) times
before raising a clear `RuntimeError` if no valid split is found.

The full-data fit is `cv.model_`:

```python
print(pd.DataFrame({"lambda_min": cv.model_.coef_at(cv.lambda_min_),
                    "lambda_1se": cv.model_.coef_at(cv.lambda_1se_)}, index=X.columns).round(4))
```
```
                   lambda_min  lambda_1se
age                    0.0457      0.0246
sex                    0.0000      0.0000
diabetes               0.1518      0.0000
comorbidity_count      0.1667      0.0000
```

`sex` — which carries no real effect in this cohort's data-generating
process — is zero at both. The one-standard-error rule is much sparser
here: it keeps only `age`, and zeroes `diabetes` and `comorbidity_count`,
although both have real effects in the data-generating process (0.35
and 0.20 on the log-hazard scale): the rule picks the sparsest model
whose cross-validated deviance is within one standard error of the
minimum, and with 259 deaths that standard error is wide enough to
include it. `predict()` and `summary()` both default to `rule='1se'`.
The rule is fixed at construction with `se_rule`, the parameter every CV class takes:

```python
print({k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in cv.summary().attrs.items()})
```
```
{'lambda': 0.019, 'rule': '1se', 'cv_mean_deviance': 0.2231, 'cv_se_deviance': 0.001}
```

`cv_mean_deviance`/`cv_se_deviance` here are **person-period binary cross-entropy
deviance** — the same loss `person_period_expand()` and
`predict_discrete_hazard()` operate on internally, averaged over every
expanded (subject, period) row in each held-out fold, not a per-subject
quantity.

## 13.7 What's next

[Chapter 14](14_provider_discrete_survival) adds provider effects to
this same discrete-time model — the three-layer architecture (provider
+ baseline hazard + covariates) that gives
`ProviderPenalizedDiscreteSurvival` its name.
