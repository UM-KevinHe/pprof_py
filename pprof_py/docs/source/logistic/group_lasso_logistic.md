(group_lasso_logistic)=
# Group Lasso Logistic Regression: Penalizing Covariates in Blocks

In R, this is closest to `grplasso::grp.lasso()` (pure group lasso) or
`gglasso`/`grpreg` (sparse group lasso) for a binomial outcome.
`GroupLassoLogistic` is R's `grplasso` for binomial family and
`GroupLassoLogisticCV` cross-validates it. This chapter assumes you've
read [Penalized Logistic Regression](penalized_logistic.md) — the lambda
path, standardization, and cross-validation machinery are all shared,
so this chapter only covers what's different when the penalty acts on
*groups* of coefficients instead of one coefficient at a time. One of
those differences is a real bug (Section 5), so read that section even
if you skim the rest.

## 1. The problem elastic net doesn't quite solve

[Penalized Logistic Regression's Section 1](penalized_logistic.md) makes
the case for shrinking coefficients when you have more candidate risk
adjusters than you're confident belong in the model. Elastic net
shrinks and selects one *coefficient* at a time — but a lot of real
covariates don't come as single coefficients. A categorical admission
source (elective, emergency, transfer, urgent) becomes three dummy
columns once one-hot encoded, and those three columns are really one
clinical concept. Elastic net has no way to know that: it can zero out
`src_transfer` while keeping `src_emergency` and `src_urgent`, which
turns "how did this patient arrive?" into a partial, hard-to-interpret
answer — the model effectively decides admission source matters for
two of its four levels and not the other two, a distinction with no
clinical meaning. The same problem shows up for a panel of related lab
values you'd only ever want to include or exclude together, or for a
spline basis representing one continuous covariate as several columns.

Group lasso fixes this by penalizing the *norm* of a whole block of
coefficients together, so the elastic net's binary in/out decision
happens at the group level: either the whole admission-source block
enters the model, or none of it does.

## 2. The group penalty

Where elastic net penalizes $\alpha\|\beta\|_1 +
(1-\alpha)\tfrac12\|\beta\|_2^2$, the sparse group lasso penalty
(covered in general terms, for the shared solver, in
[survival Chapter 8](../survival/08_penalized_regression)) replaces
the ridge term with a sum of per-group $\ell_2$ norms:

$$
\lambda\left[(1-\alpha)\sum_{g} m_g\|\beta_g\|_2 \;+\; \alpha\sum_j \text{pf}_j|\beta_j|\right]
$$

In words: for each group $g$ of coefficients, take the length of that
group's coefficient vector ($\|\beta_g\|_2 = \sqrt{\sum_{j \in g}
\beta_j^2}$) and penalize it, the same way LASSO penalizes the length
of a single coefficient. $m_g$ is a per-group multiplier — by default
$\sqrt{|g|}$, the square root of the group's size, which compensates
for larger groups otherwise attracting a larger penalty just by having
more coordinates — and $\alpha$ now means something different from
`alpha` in the penalized-regression chapters: **here it's the mixing
weight between the group penalty and an *additional* element-wise
lasso term**, not between ridge and lasso. `alpha=0.0` (this class's
default) is pure group lasso: a group is either entirely zero or
entirely nonzero. `alpha=1.0` reduces to ordinary element-wise lasso
and ignores the group structure completely. Values in between are
*sparse* group lasso (Section 6): a group can be selected in as a
whole and still have some of its own coordinates shrunk to zero.

## 3. The running example: admission source and a comorbidity panel

Same shape of mortality cohort as the
[penalized logistic chapter](penalized_logistic.md), but built around
three genuine groups instead of ten individual covariates: a
categorical admission source (one-hot into three dummy columns), a
three-item comorbidity panel, and a two-item noise lab panel — plus an
age term left deliberately **unpenalized**, which is where this
chapter's main finding shows up.

```python
import numpy as np
import pandas as pd

def make_grouped_mortality_cohort(n_patients=6000, n_facilities=60, seed=0):
    rng = np.random.default_rng(seed)
    facility_id = rng.integers(0, n_facilities, n_patients)
    facility_quality = rng.normal(0, 0.2, n_facilities)
    age = rng.normal(64, 13, n_patients).clip(18, 96)

    # Group 1: admission source (categorical, "Elective" is the reference level)
    source = rng.choice(["Elective", "Emergency", "Transfer", "Urgent"],
                         size=n_patients, p=[0.40, 0.35, 0.10, 0.15])
    src_emergency = (source == "Emergency").astype(int)
    src_transfer = (source == "Transfer").astype(int)
    src_urgent = (source == "Urgent").astype(int)

    # Group 2: comorbidity panel
    diabetes = rng.binomial(1, 0.42, n_patients)
    chf = rng.binomial(1, 0.28, n_patients)
    ckd = rng.binomial(1, 0.33, n_patients)

    # Group 3: noise lab panel -- no effect on the outcome
    lab_e = rng.normal(0, 1, n_patients)
    lab_f = rng.normal(0, 1, n_patients)

    log_odds = (
        -3.6 + 0.038 * (age - 64)
        + 0.70 * src_emergency + 0.85 * src_transfer + 0.40 * src_urgent
        + 0.30 * diabetes + 0.55 * chf + 0.35 * ckd
        + facility_quality[facility_id]
    )
    death_30d = rng.binomial(1, 1.0 / (1.0 + np.exp(-log_odds)))
    return pd.DataFrame({
        "patient_id": np.arange(n_patients), "facility_id": facility_id, "age": age.round(1),
        "src_emergency": src_emergency, "src_transfer": src_transfer, "src_urgent": src_urgent,
        "diabetes": diabetes, "chf": chf, "ckd": ckd,
        "lab_e": lab_e.round(3), "lab_f": lab_f.round(3), "death_30d": death_30d,
    })

cohort = make_grouped_mortality_cohort()
candidates = ["age", "src_emergency", "src_transfer", "src_urgent",
              "diabetes", "chf", "ckd", "lab_e", "lab_f"]
X = cohort[candidates]
y = cohort["death_30d"].values

# group_labels: 0 = unpenalized; 1, 2, 3 = penalized groups.
# Columns sharing a label must be contiguous, matching X's column order.
groups = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3])
```

6,000 patients, 7.1% 30-day mortality. `groups` maps each of the nine
columns in `X` to a group label, positionally — `groups[j]` is the
group for `X`'s $j$-th column, so `groups` and `candidates` must stay
in the same order. `0` is reserved to mean "not penalized"; positive
integers are penalized groups, and the columns sharing a label must be
contiguous in `X` (which they are here, since `candidates` was written
group-by-group on purpose — `validate_groups` raises a clear error if
they aren't).

## 4. Fitting the path

```python
from pprof_py import GroupLassoLogistic

m = GroupLassoLogistic(groups=groups, alpha=0.0)   # pure group lasso
m.fit(X, y)

print(f"lambda_max = {m.lambda_max_:.6f}")
print(m.n_groups_, m.group_sizes_, m.group_weights_.round(4))   # penalized groups only; weights sqrt(size)
```
```
lambda_max = 0.013751
3 [3 3 2] [1.7321 1.7321 1.4142]
```

`age` is group 0: unpenalized, and fitted at every point of the path,
including the null point at $\lambda_{\max}$, the smallest $\lambda$ at
which every penalized group is zero.

The mechanics you already know from `PenalizedLogistic` carry over
directly: `coef_` doesn't exist for this full, 100-point path fit
(only for a single-lambda fit); pull coefficients out via `coef_path_`
by position or `coef_at()` by an arbitrary lambda. What's new is a
per-group view of the path. `active_groups_path_` (shape `(100, 3)`)
and `active_group_labels(which=)` report which groups have a nonzero
norm at a given path index; `group_norms_path_` (shape `(100, 3)`)
gives the actual $\|\beta_g\|_2$ trajectory (Section 6):

```python
lam50 = m.lambda_path_[50]
print(m.active_group_labels(which=50))   # all three active by here
print(pd.Series(m.coef_at(lam50), index=candidates).round(4).to_string())
```
```
[1 2 3]
age              0.0339
src_emergency    0.7061
src_transfer     0.9292
src_urgent       0.3358
diabetes         0.1365
chf              0.4532
ckd              0.5889
lab_e            0.0145
lab_f           -0.0014
```

`age` (true effect 0.038) is unpenalized, so it carries its fitted
value throughout.

## 5. Sparse group lasso: selection within a selected group

At `alpha=0.0`, a selected group's coordinates all move together,
scaled by the same shrinkage factor — the categorical dummy for
`src_transfer` can't be individually zeroed while `src_emergency` and
`src_urgent` stay in. Setting `alpha` above zero adds the element-wise
lasso term back in *within* the group-selection structure: a group can
still enter or leave as a block, but once it's in, its own coordinates
are independently shrunk and can be individually zeroed:

```python
m_sparse = GroupLassoLogistic(groups=groups, alpha=0.5)
m_sparse.fit(X, y)
print(pd.Series(m_sparse.coef_at(m_sparse.lambda_path_[50]), index=candidates).round(4).to_string())
```
```
age              0.0339
src_emergency    0.7061
src_transfer     0.9293
src_urgent       0.3385
diabetes         0.1355
chf              0.4518
ckd              0.5899
lab_e            0.0142
lab_f           -0.0015
```

At this particular lambda the two are nearly identical, because none
of the individually-weak coordinates within an active group have
crossed their own element-wise threshold yet — sparse group lasso is a
strict refinement of pure group lasso, not a different answer, until
a coordinate's *own* signal is weak enough for the element-wise term
to matter. Use `alpha=0.0` when you trust the groups themselves as the
unit of selection (e.g. "does admission source matter at all") and
have no reason to drop part of a group; use a modest `alpha > 0` when
you also suspect some coordinates *within* an included group are
individually uninformative.

## 6. Group norms along the path

`group_norms_path_` shows when each group enters and how fast it grows:

```python
rows = [0, 1, 2, 3, 5, 8, 31]
norms = pd.DataFrame(m.group_norms_path_[rows], index=rows,
                     columns=["group 1 (source)", "group 2 (comorbidity)", "group 3 (noise)"])
norms.insert(0, "lambda", m.lambda_path_[rows])
print(norms.round(6).to_string())
```
```
      lambda  group 1 (source)  group 2 (comorbidity)  group 3 (noise)
0   0.013751          0.000000               0.000000          0.00000
1   0.012529          0.008804               0.032652          0.00000
2   0.011416          0.038814               0.062346          0.00000
3   0.010402          0.066097               0.089109          0.00000
5   0.008636          0.113631               0.135156          0.00000
8   0.006533          0.170687               0.189297          0.00000
31  0.000769          0.335217               0.336155          0.00132
```

At `lambda_max_` (row 0) every penalized group is exactly zero. The two
real groups enter together at the next point, while the noise panel
stays at zero until row 31, where it first becomes active, at a lambda
about one eighteenth of `lambda_max_`: group lasso holds the panel out as a
unit until the penalty has relaxed a long way.

## 7. Cross-validation with `GroupLassoLogisticCV`

```python
from pprof_py import GroupLassoLogisticCV

cv = GroupLassoLogisticCV(groups=groups, alpha=0.0, n_folds=10, random_state=0)
cv.fit(X, y)

print(f"lambda_min = {cv.lambda_min_:.6f}, lambda_1se = {cv.lambda_1se_:.6f}")   # lambda_ is lambda_1se_
```
```
lambda_min = 0.001618, lambda_1se = 0.006533
```

Same parameter (`se_rule`), same default, same `cv_mean_deviance_` /
`cv_se_deviance_` naming as `PenalizedLogisticCV` — group lasso's CV
class was built consistently with its element-wise sibling here, even
though (as the [next chapter](../linear/group_lasso_linear) and the
[group lasso Cox chapter](../survival/11_group_lasso_cox) will show)
that consistency doesn't hold everywhere in the package.

```python
print(pd.Series(cv.coef_, index=candidates).round(4).to_string())   # at lambda_1se_
```
```
age              0.0329
src_emergency    0.3286
src_transfer     0.4622
src_urgent       0.1334
diabetes         0.0709
chf              0.2409
ckd              0.3190
lab_e            0.0000
lab_f            0.0000
```

At `lambda_1se_` both real groups are in, with every coefficient
shrunk toward zero and positive like its true effect, and the noise
panel is zeroed as a unit. At `lambda_min_`
(`cv.model_.coef_at(cv.lambda_min_)`; `cv.model_` holds the full path)
the real coefficients are larger and the noise panel is still zero.

## 8. Choosing groups for healthcare data

A few patterns come up often enough in provider-profiling work to call
out directly:

- **One-hot-encoded categoricals** (admission source, race/ethnicity,
  payer type): the natural, canonical use case — the columns are
  already a single clinical concept split across several coordinates
  by the encoding, not several independent decisions.
- **A lab or vital-signs panel measuring one underlying process**
  (say, three markers of kidney function): group them if the
  scientific question is "does renal function matter" rather than
  "which specific marker matters," which also stabilizes the fit
  against the multicollinearity such panels usually carry.
- **Interaction terms with their main effects**: group an interaction
  column with the main-effect column(s) it modifies, so the model
  can't retain an interaction while dropping the main effect it
  interacts with — a combination that's hard to interpret on its own.
- **When in doubt, use singleton groups.** A group of size 1 makes the
  group penalty behave exactly like an ordinary lasso term for that
  coordinate ($\|\beta_g\|_2 = |\beta_j|$), so mixing genuine multi-
  column groups with singleton groups for covariates you want treated
  individually is a normal, supported pattern — every column needs a
  group label, but not every label needs more than one column.

## 9. `predict_proba()`

`predict_proba()` at an arbitrary lambda interpolates both the
coefficients and the intercept in $\log\lambda$, as `coef_at()` does:

```python
lam = m.lambda_path_[60]
p = m.predict_proba(X.values[:5], lambda_value=lam)
intercept = np.interp(np.log(lam), np.log(m.lambda_path_[::-1]), m.intercept_path_[::-1])
manual = 1 / (1 + np.exp(-(X.values[:5] @ m.coef_at(lam) + intercept)))
print(p.round(4), bool(np.allclose(p, manual, rtol=0, atol=1e-12)))
```
```
[0.0287 0.0222 0.057  0.0427 0.0663] True
```

## 10. What's next

The [group lasso linear chapter](../linear/group_lasso_linear) covers
the same penalty for continuous outcomes, and the
[group lasso Cox chapter](../survival/11_group_lasso_cox) covers it for
time-to-event outcomes.
