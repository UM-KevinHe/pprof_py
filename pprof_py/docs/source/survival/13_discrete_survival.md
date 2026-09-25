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
same elastic-net/group-lasso machinery
[Chapter 1](../logistic/penalized_logistic) and
[the group lasso chapter](../logistic/group_lasso_logistic) describe,
applied to data reshaped this specific way, rather than a new
optimization method.

## 13.3 The running example: annual chart review

An interval-censored version of the ESRD cohort — the same
data-generating process as [Chapter 0's](00_start_here) cohort, but
follow-up is only known to the nearest year (`time_year` $\in \{1, 2,
3, 4\}$), the way an annual chart-review protocol would actually
report it.

```python
from pprof_py import DiscreteSurvival, DiscreteSurvivalCV

X = cohort[["age", "sex", "diabetes", "comorbidity_count"]]
time = cohort["time_year"]   # integer years 1-4, not continuous days
event = cohort["death"]

model = DiscreteSurvival(penalty_type="lasso")
model.fit(X, time, event)

model.lambda_max_       # 0.04667363567097111
model.n_timepoints_     # 4
model.timepoint_map_    # array([1., 2., 3., 4.])
model.coef_path_.shape  # (100, 4)
model.alpha_path_.shape # (100, 4)  -- one baseline parameter per timepoint, per lambda
```

4,000 patients, 40 facilities, 7.8% event rate, spread `711 / 1174 /
1080 / 1035` across years 1 through 4. `penalty_type` is `'lasso'`,
`'group_lasso'`, or `'sparse_group_lasso'` — one class covers all
three, unlike the logistic and linear families, which use separate
`Penalized*`/`GroupLasso*` classes for the same distinction; pass
`groups=` when using either group option, exactly as
[the group lasso chapter](../logistic/group_lasso_logistic) describes.

`coef_at(lambda_value)` interpolates continuously, the same way
[Chapter 1's](../logistic/penalized_logistic) `coef_at()` does.

## 13.4 Reading the path

```python
[int((model.coef_path_[i] != 0).sum()) for i in [0, 20, 40, 60, 80, 99]]
# [0, 3, 3, 4, 4, 4]
```

```python
model.coef_path_[60]
# age: 0.0470, sex: 0.0262, comorbidity_count: 0.234, diabetes: 0.380
```

`converged_` is `True` at all 100 lambdas on this cohort.
This class's default `tol` (`1e-4`) is considerably looser than the
`1e-9`/`1e-7` used elsewhere in this guide, which makes clean
convergence easier to reach in practice.

`predict()` supports `type='link'` (default), `type='hazard'`, and
`type='survival'`.  The latter two require a `time=` argument and
delegate to `predict_hazard()`/`predict_survival()` internally
(Section 13.5).

## 13.5 `predict_hazard()`/`predict_survival()` need `time=`

Unlike `CoxPH.predict_partial_hazard()`, which needs no `time=`
argument at all (a Cox partial hazard is a single, time-invariant
relative-risk number), a discrete-time hazard is timepoint-specific by
construction — "the hazard" isn't one number per patient, it's one
number *per patient per period*, so both methods require `time=`
explicitly:

```python
import numpy as np

X3, time3 = X.values[:3], time.values[:3]   # time3 = [2, 2, 3]
model.predict_hazard(X3, time3, which=60)
# shape (7,) -- one hazard per person-period row (sum of discretised times)

model.predict_survival(X3, time3, which=60)
# same convention
```

Internally, both methods map `time` into the integer codes that
`self.timepoint_map_` established during `fit()` via
`np.searchsorted`, selecting the correct column of `alpha_path_`
for each query point.

Note that `ProviderPenalizedDiscreteSurvival`'s
`predict_hazard()` ([Chapter 14](14_provider_discrete_survival))
follows a different convention, returning a 2-D `(n, K)` wide-format
array instead of a 1-D person-period vector — same method name,
different return shape.

## 13.6 Cross-validation with `DiscreteSurvivalCV`

```python
cv = DiscreteSurvivalCV(n_folds=5, random_state=0, penalty_type="lasso")
cv.fit(X, time, event)

cv.lambda_min_    # 0.001493414638159624
cv.lambda_1se_    # 0.013928358245089619
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
cv.model_.coef_at(cv.lambda_min_)
# age: 0.0455, sex: 0.0, diabetes: 0.3451, comorbidity_count: 0.2206
cv.model_.coef_at(cv.lambda_1se_)
# age: 0.0327, sex: 0.0, diabetes: 0.0214, comorbidity_count: 0.0910
```

`sex` — which carries no real effect in this cohort's data-generating
process — is correctly zero at both. `predict()` and `summary()` both default to `rule='1se'`.
The rule is fixed at construction with `se_rule`, the parameter every CV class takes:

```python
cv.summary().attrs
# {'lambda': 0.0139, 'rule': '1se', 'cv_mean': 0.2553, 'cv_se': 0.0019}
```

`cv_mean`/`cv_se` here are **person-period binary cross-entropy
deviance** — the same loss `person_period_expand()` and
`predict_discrete_hazard()` operate on internally, averaged over every
expanded (subject, period) row in each held-out fold, not a per-subject
quantity.

## 13.7 What's next

[Chapter 14](14_provider_discrete_survival) adds provider effects to
this same discrete-time model — the three-layer architecture (provider
+ baseline hazard + covariates) that gives
`ProviderPenalizedDiscreteSurvival` its name.
