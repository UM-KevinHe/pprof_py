(provider_discrete_survival)=
# Chapter 14 — Provider-Penalized Discrete-Time Survival Models

## 14.1 Three layers, not two

[Chapter 12](12_provider_penalized_cox) and
[the provider-penalized logistic chapter](../logistic/provider_penalized_logistic)
both alternate **two** parameter blocks: provider effects and
covariates. `ProviderPenalizedDiscreteSurvival` — R's `pp.DiscSurv` in
the same `grplasso` lineage — alternates **three**, because
[Chapter 13](13_discrete_survival) already introduced a second
unpenalized block discrete-time models need that Cox and logistic
don't: the per-timepoint baseline hazard $\alpha_k$.

$$
\text{logit}\big(h(t_k \mid Z_i)\big) = \alpha_k + \gamma_{p(i)} + Z_i^\top\beta
$$

Each lambda now alternates three Newton/CD steps instead of two, up to
`max_outer_iter` times:

1. **Provider step**: Newton update for $\gamma_k$ (median-clamp
   bounded by `provider_bound`, same mechanism as
   [the logistic chapter's](../logistic/provider_penalized_logistic)
   Section 3), up to `provider_max_iter` (default 10) rounds.
2. **Baseline hazard step**: Newton update for $\alpha_t$ via
   `baseline_hazard_update()`, up to `baseline_max_iter` (default 10)
   rounds — a third iteration cap, alongside `provider_max_iter` and
   the covariate solver's own `max_inner_iter`, that has no equivalent
   in the two-layer models.
3. **Covariate step**: penalized coordinate descent for $\beta$ on the
   person-period working response
   ([Chapter 13's](13_discrete_survival) expansion).

Convergence is checked once per outer iteration, on the largest change
in any of the three blocks: `max(max|Δβ|, max|Δγ|, max|Δα|) < outer_tol`
(default `1e-7`), that is, every block's largest change must be below
the tolerance.

## 14.2 `alpha`: elastic-net mixing

`alpha` is the elastic-net mixing parameter (1 = lasso, 0 = ridge),
as in the other penalized classes.
[`DiscreteSurvival`](13_discrete_survival) has no `alpha`: it fits the
lasso only.

## 14.3 `standardize=False` by default — deliberately

Every other penalized class in this documentation defaults to
`standardize=True`. This one defaults to **`False`**, and says why
directly in its own docstring: R's `pp.DiscSurv` "comments:
`standardize = TRUE` may cause problems in transforming gamma and
alpha back." With three unpenalized parameter blocks ($\gamma$,
$\alpha$, and the intercept-like role they jointly play) all
interacting with a standardized-then-unstandardized $\beta$, the
un-transformation has more moving parts than the two-layer models'
single provider term, and the package follows R's own documented
caution here rather than overriding it. Pass `standardize=True`
explicitly if you want it; nothing prevents it, but it isn't the
default the way it is everywhere else.

## 14.4 The running example, extended with facilities

[Chapter 13's](13_discrete_survival) annual chart-review cohort, with
`time_year` built the same way; `facility_id` was already present, just
unused there.

```python
import numpy as np
from pprof_py import ProviderPenalizedDiscreteSurvival, ProviderPenalizedDiscreteSurvivalCV

cohort["time_year"] = np.ceil(cohort["time"]).astype(int)   # as in Chapter 13

X = cohort[["age", "sex", "diabetes", "comorbidity_count"]]
time, event = cohort["time_year"], cohort["death"]
provider_id = cohort["facility_id"]

model = ProviderPenalizedDiscreteSurvival(alpha=1.0)
model.fit(X, time, event, provider_id)

print(model.n_providers_, "providers,", model.n_timepoints_, "timepoints")
print("coef_path_:", model.coef_path_.shape,
      " baseline_hazard_path_:", model.baseline_hazard_path_.shape,   # alpha_t per lambda
      " gamma_path_:", model.gamma_path_.shape)                       # one column per facility
```
```
40 providers, 4 timepoints
coef_path_: (100, 4)  baseline_hazard_path_: (100, 4)  gamma_path_: (100, 40)
```

## 14.5 `predict_hazard()` and `predict_survival()`

```python
hz = model.predict_hazard(X.values[:3], provider_id.values[:3], which=50)
sv = model.predict_survival(X.values[:3], provider_id.values[:3], which=50)
print(hz.shape, sv.shape)   # one row per subject, one column per discrete timepoint
print(bool(np.allclose(sv, np.cumprod(1 - hz, axis=1))))
```
```
(3, 4) (3, 4)
True
```

The signature is [Chapter 13's](13_discrete_survival) with `provider_id`
second: `predict_hazard(X, provider_id=None, time=None, lambda_value=None,
which=None)`. Without `time=`, as here, the result has one column per time
point; with it, the person-period form of Chapter 13, one value per period up
to each subject's time. `which` picks a path point by index (by default, the
last) and `lambda_value` the path point nearest a lambda; rows of providers the
model has not seen get no provider effect.

`predict_survival()` here is exactly the cumulative product of
`1 - predict_hazard()` across the timepoint axis
(`np.cumprod(1 - hazard, axis=1)`) — the discrete-time analogue of a
Cox survival curve, one row per subject, cumulative down the columns.

## 14.6 Provider effects and cross-validation

```python
cv = ProviderPenalizedDiscreteSurvivalCV(n_folds=5, random_state=0, alpha=1.0)
cv.fit(X, time, event, provider_id)

print(f"lambda_min = {cv.lambda_min_:.6f}, lambda_1se = {cv.lambda_1se_:.6f}")
print(pd.Series(cv.coef_, index=X.columns).round(4).to_string())   # at the selected lambda (se_rule="1se")
```
```
lambda_min = 0.000105, lambda_1se = 0.259308
age                  0.0246
sex                  0.0000
diabetes             0.0000
comorbidity_count    0.0000
```

Only `age` survives at the default `lambda_1se_`, as in
[Chapter 13](13_discrete_survival) without provider effects: with 259
deaths spread across 40 facilities and four discrete timepoints, the
one-standard-error margin prefers the simplest model. This class's CV parameter is `se_rule` (default `"1se"`), as in
every CV class in the package, and the full-data fit is `cv.model_`.

## 14.7 What's next

This is the last chapter of the survival guide. For provider tests and
confidence intervals on standardized ratios (SMRs), return to
[Chapter 4](04_indirect_standardization_smr_shr); the
[empirical-null guide](empirical-null-guide) covers calibrating those
tests.
