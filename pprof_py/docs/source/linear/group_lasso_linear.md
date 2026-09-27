(group_lasso_linear)=
# Group Lasso Linear Regression: Penalizing Covariates in Blocks

`GroupLassoLinear` is the continuous-outcome sibling of
`GroupLassoLogistic` — this chapter assumes you've read
[Group Lasso Logistic Regression](group_lasso_logistic) for the "why
groups" motivation, the penalty formula, and `alpha`'s group-lasso
meaning (the group/element-wise mix, not elastic net's ridge/lasso
mix), none of which is repeated here. Two things are genuinely
different for this class, and both matter more than usual: it has no
cross-validated wrapper at all.

## 1. The running example: cost, with the same group structure

Same cohort shape as the [group lasso logistic chapter](group_lasso_logistic)
— admission source (one-hot, 3 columns), a comorbidity panel (3
columns), a noise lab panel (2 columns), age left unpenalized — but
with 30-day cost as a continuous outcome instead of a mortality flag.

```python
import numpy as np
import pandas as pd

def make_grouped_cost_cohort(n_patients=6000, n_facilities=60, seed=1):
    rng = np.random.default_rng(seed)
    facility_id = rng.integers(0, n_facilities, n_patients)
    facility_efficiency = rng.normal(0, 1.5, n_facilities)
    age = rng.normal(64, 13, n_patients).clip(18, 96)

    source = rng.choice(["Elective", "Emergency", "Transfer", "Urgent"],
                         size=n_patients, p=[0.40, 0.35, 0.10, 0.15])
    src_emergency = (source == "Emergency").astype(int)
    src_transfer = (source == "Transfer").astype(int)
    src_urgent = (source == "Urgent").astype(int)

    diabetes = rng.binomial(1, 0.42, n_patients)
    chf = rng.binomial(1, 0.28, n_patients)
    ckd = rng.binomial(1, 0.33, n_patients)
    lab_e = rng.normal(0, 1, n_patients)   # noise
    lab_f = rng.normal(0, 1, n_patients)   # noise

    cost = (
        18.0 + 0.12 * (age - 64)
        + 9.0 * src_emergency + 11.0 * src_transfer + 5.5 * src_urgent
        + 4.5 * diabetes + 7.8 * chf + 5.0 * ckd
        + facility_efficiency[facility_id] + rng.normal(0, 5.0, n_patients)
    )
    return pd.DataFrame({
        "patient_id": np.arange(n_patients), "facility_id": facility_id, "age": age.round(1),
        "src_emergency": src_emergency, "src_transfer": src_transfer, "src_urgent": src_urgent,
        "diabetes": diabetes, "chf": chf, "ckd": ckd,
        "lab_e": lab_e.round(3), "lab_f": lab_f.round(3),
        "cost_30d": np.clip(cost, 1.0, None).round(2),
    })

cohort = make_grouped_cost_cohort()
candidates = ["age", "src_emergency", "src_transfer", "src_urgent",
              "diabetes", "chf", "ckd", "lab_e", "lab_f"]
X = cohort[candidates]
y = cohort["cost_30d"].values
groups = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3])
```

## 2. Fitting the path: a clean group-entry story

```python
from pprof_py import GroupLassoLinear

m = GroupLassoLinear(groups=groups, alpha=0.0)
m.fit(X, y)
print(f"lambda_max = {m.lambda_max_:.6f}")
```
```
lambda_max = 2.727163
```

`age` is group 0: it carries no penalty, so it is fitted at every point
of the path, including the null point at $\lambda_{\max}$, the smallest
$\lambda$ at which every penalized group is zero.

At $\lambda_{\max}$ every penalized group is exactly zero
(`active_groups_path_[0]` is `[False, False, False]`), as in the
[logistic chapter](group_lasso_logistic), and the group-level selection
story comes through directly:

```python
names = ["admission source", "comorbidity panel", "noise lab panel"]
for g in range(3):
    first = int(np.argmax(m.active_groups_path_[:, g]))
    print(f"group {g + 1} ({names[g]}): first active at index {first}, lambda={m.lambda_path_[first]:.4f}")
```
```
group 1 (admission source): first active at index 1, lambda=2.4849
group 2 (comorbidity panel): first active at index 1, lambda=2.4849
group 3 (noise lab panel): first active at index 42, lambda=0.0548
```

Both real-signal groups enter at the first point below $\lambda_{\max}$;
the noise group doesn't enter until 41 steps later, at a lambda about
45 times smaller. This is the group-lasso
selection property working as intended: it isn't just shrinking two
individually-noisy coordinates independently (the way element-wise
lasso would, potentially at two different, unrelated lambdas) — it's
recognizing that `lab_e` and `lab_f` carry no *collective* signal as a
unit and excluding the whole unit together, over a wide stretch of the
path, before either one is allowed to enter.

`coef_` doesn't exist for this fit (`hasattr(m, "coef_")` is `False`,
same single-lambda-only condition as every other path-fitting class in
this package); use `coef_at()` for an arbitrary lambda:

```python
print(pd.Series(m.coef_at(m.lambda_path_[50]), index=candidates).round(4).to_string())
```
```
age               0.1213
src_emergency     9.2163
src_transfer     10.9857
src_urgent        5.3311
diabetes          4.5784
chf               7.7336
ckd               4.7874
lab_e             0.0026
lab_f             0.0437
```

Both real groups already carry substantial coefficients by this point
on the path; the noise group's two coordinates are still two orders of
magnitude smaller, consistent with it having only just become active
eight steps earlier. `age` (true effect 0.12) is unpenalized and fitted
throughout.

## 3. Sparse group lasso

Same behavior as the logistic case: `alpha > 0` allows individual
coordinates within an active group to be shrunk to zero, while the
group itself is still selected as a unit at the group-norm level.

```python
m_sparse = GroupLassoLinear(groups=groups, alpha=0.7)
m_sparse.fit(X, y)
print(pd.Series(m_sparse.coef_at(m_sparse.lambda_path_[40]), index=candidates).round(4).to_string())
```
```
age               0.1212
src_emergency     9.0322
src_transfer     10.7437
src_urgent        5.3909
diabetes          4.5667
chf               7.5993
ckd               4.5951
lab_e             0.0000
lab_f             0.0000
```

At `alpha=0.7` the noise group is still entirely zero at this point
(index 40, $\lambda = 0.0880$) and enters at index 45, both coordinates
at once. The sparse group lasso *can* zero a coordinate inside an active
group, because its L1 term thresholds each coordinate separately, but on
this cohort it has nothing to separate: `lab_e` and `lab_f` are
exchangeable pure noise. Pure group lasso (`alpha=0.0`) cannot do it at
all, since it moves a whole group's coordinates by one shared factor.

## 4. There is no `GroupLassoLinearCV`

Unlike its logistic and Cox counterparts, the linear family's group
lasso has no built-in cross-validated wrapper — `pprof_py.__all__`
exports `GroupLassoLinear` alone. This isn't a bug to work around, just
a real, current gap in the public API: if you need a cross-validated
lambda for `GroupLassoLinear`, you write the k-fold loop yourself. It
is a short loop, because it's the same one every other CV class in
this package runs internally:

```python
rng = np.random.RandomState(0)
n_folds = 10
fold_id = rng.randint(0, n_folds, size=len(y))
X_arr = X.values

full = GroupLassoLinear(groups=groups, alpha=0.0).fit(X, y)
lambda_path = full.lambda_path_
cv_dev = np.full((n_folds, len(lambda_path)), np.nan)

for k in range(n_folds):
    train, val = fold_id != k, fold_id == k
    fold_model = GroupLassoLinear(
        groups=groups, alpha=0.0, lambda_path=lambda_path,
    ).fit(X_arr[train], y[train])
    for j in range(len(lambda_path)):
        pred = X_arr[val] @ fold_model.coef_path_[j] + fold_model.intercept_path_[j]
        cv_dev[k, j] = np.sum((y[val] - pred) ** 2)

cv_mean = np.nanmean(cv_dev, axis=0)
cv_se = np.nanstd(cv_dev, axis=0, ddof=1) / np.sqrt(n_folds)
idx_min = int(np.nanargmin(cv_mean))
threshold = cv_mean[idx_min] + cv_se[idx_min]
idx_1se = int(np.where(cv_mean <= threshold)[0][0])

lambda_min, lambda_1se = lambda_path[idx_min], lambda_path[idx_1se]
print(f"lambda_min = {lambda_min:.5f}, lambda_1se = {lambda_1se:.4f}")
print(pd.Series(full.coef_at(lambda_1se), index=candidates).round(4).to_string())
```
```
lambda_min = 0.01490, lambda_1se = 0.2924
age              0.1209
src_emergency    8.2755
src_transfer     9.8468
src_urgent       4.8287
diabetes         4.1310
chf              6.9691
ckd              4.3117
lab_e            0.0000
lab_f            0.0000
```

Notice `lab_e`
and `lab_f` land on exactly the same value (`0.0000`) at `lambda_1se`
— they're zeroed *together*, as a group, which is the property this
whole chapter is about. This loop follows exactly the same fold-then-
average-then-1SE pattern `PenalizedLinearCV` uses internally (per-fold
weighted sum of squared residuals, not divided by fold size — see
the penalized linear chapter if you want a true
per-observation scale instead of this relative comparison), so it's a
faithful stand-in, not an approximation of one.

## 5. What's next

A [group lasso Cox chapter](../survival/11_group_lasso_cox) covers the
same penalty for time-to-event outcomes, where it's built on the
survival guide's existing `PenalizedCoxPH` infrastructure rather than
standing alone the way this chapter and the logistic one do — and
where `GroupLassoCoxPHCV` **does** exist, with a properly analytical
standard error and a few methodological refinements neither
`GroupLassoLogisticCV` nor this chapter's manual loop has.
