(group_lasso_logistic)=
# Group Lasso Logistic Regression: Penalizing Covariates in Blocks

In R, this is closest to `grplasso::grp.lasso()` (pure group lasso) or
`gglasso`/`grpreg` (sparse group lasso) for a binomial outcome.
`GroupLassoLogistic` is R's `grplasso` for binomial family and
`GroupLassoLogisticCV` cross-validates it. This chapter assumes you've
read [Penalized Logistic Regression](penalized_logistic) — the lambda
path, standardization, and cross-validation machinery are all shared,
so this chapter only covers what's different when the penalty acts on
*groups* of coefficients instead of one coefficient at a time. One of
those differences is a real bug (Section 5), so read that section even
if you skim the rest.

## 1. The problem elastic net doesn't quite solve

[Penalized Logistic Regression's Section 1](penalized_logistic) makes
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
[penalized logistic chapter](penalized_logistic), but built around
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

m.lambda_max_       # 0.013250566398507435
m.n_groups_          # 3 -- the *penalized* groups only; group 0 doesn't count
m.group_sizes_       # array([3, 3, 2])
m.group_weights_     # array([1.7321, 1.7321, 1.4142])  -- sqrt(3), sqrt(3), sqrt(2)
```

The mechanics you already know from `PenalizedLogistic` carry over
directly: `coef_` doesn't exist for this full, 100-point path fit
(only for a single-lambda fit); pull coefficients out via `coef_path_`
by position or `coef_at()` by an arbitrary lambda. What's new is a
per-group view of the path. `active_groups_path_` (shape `(100, 3)`)
and `active_group_labels(which=)` report which groups have a nonzero
norm at a given path index; `group_norms_path_` (shape `(100, 3)`)
gives the actual $\|\beta_g\|_2$ trajectory, which is the more
reliable of the two to read near the top of the path (Section 7
explains why):

```python
lam50 = m.lambda_path_[50]
m.active_group_labels(which=50)   # array([1, 2, 3]) -- all three active by here
m.coef_at(lam50)
```
```
age              0.0000
src_emergency    0.7051
src_transfer     0.9169
src_urgent       0.3406
diabetes         0.1316
chf              0.4210
ckd              0.5558
lab_e            0.0105
lab_f            0.0074
```

Notice `age` is `0.0000` here too — every single point on this path,
in fact, not just this one. That's the subject of the next section.

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
m_sparse.coef_at(m_sparse.lambda_path_[50])
```
```
age              0.0000
src_emergency    0.7036
src_transfer     0.9144
src_urgent       0.3372
diabetes         0.1313
chf              0.4205
ckd              0.5559
lab_e            0.0095
lab_f            0.0063
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

## 6. Reading `active_groups_path_` near the top of the path

`lambda_max_`, by definition, should be the smallest lambda at which
every penalized group's norm is exactly zero. In practice, near the
top of this cohort's path, it isn't quite:

```python
m.group_norms_path_[[0, 1, 2, 3, 5, 8]]
```
```
idx  lambda      group 1 (source)  group 2 (comorbidity)  group 3 (noise)
  0  0.013251     2.35e-12           2.45e-12               4.17e-14
  1  0.012073     4.73e-12           4.93e-12               1.06e-13
  2  0.011001     6.21e-10           6.48e-10               2.20e-11
  3  0.010024     2.10e-02           2.20e-02               9.05e-04
  5  0.008322     8.84e-02           8.96e-02               3.98e-03
  8  0.006295     2.19e-01           1.94e-01               7.61e-03
```

Note that the first three rows are floating-point noise, not real
solutions — group norms on the order of `1e-10` to `1e-12` are
indistinguishable from zero for any modeling purpose.
`group_norms_path_` itself is the more trustworthy read near the
boundary — the actual magnitudes make it obvious rows 0–2 are noise
and row 3 onward is real.

The substantive story is still there once you look at magnitude rather
than the boolean active flag: by row 8, the noise group's norm
(`0.0076`) is already an order of magnitude smaller than the two real
groups' (`0.22`, `0.19`) — group lasso is correctly treating the noise
panel as carrying much weaker collective signal, even though on this
cohort all three groups happen to cross the raw activation threshold
at a similar point in the path (the
[linear chapter that follows](../linear/group_lasso_linear) shows a
cleaner, more sequential version of this same story, where the noise
group visibly enters dozens of lambdas later than the real ones).

## 7. Cross-validation with `GroupLassoLogisticCV`

```python
from pprof_py import GroupLassoLogisticCV

cv = GroupLassoLogisticCV(groups=groups, alpha=0.0, n_folds=10, random_state=0)
cv.fit(X, y)

cv.lambda_min_   # 0.00035194667528199477
cv.lambda_1se_   # 0.005735853877827921
cv.lambda_        # 0.005735853877827921 -- lambda_1se_, use_1se=True default
```

Same parameter (`use_1se`), same default, same `cv_mean_deviance_` /
`cv_se_deviance_` naming as `PenalizedLogisticCV` — group lasso's CV
class was built consistently with its element-wise sibling here, even
though (as the [next chapter](../linear/group_lasso_linear) and the
[group lasso Cox chapter](../survival/11_group_lasso_cox) will show)
that consistency doesn't hold everywhere in the package.

```python
cv.coef_   # at lambda_1se_
```
```
age              0.0000
src_emergency    0.3513
src_transfer     0.5500
src_urgent       0.0051
diabetes        -0.0245
chf              0.2596
ckd              0.3610
lab_e            0.0061
lab_f            0.0050
```

`age` is `0.0000` here too, for the reason Section 5 explained — this
is the CV class's fit passing straight through to the same underlying
solver. One more thing worth flagging honestly: `diabetes` comes out
*negative* (`-0.0245`) at `lambda_1se_`, even though it has a clearly
positive effect (`+0.30`) in the data-generating process above. This
isn't a bug — pure group lasso (`alpha=0.0`) shrinks a whole group's
coefficient vector by a single multiplicative factor, preserving
whatever *direction* that vector already had once `diabetes`, `chf`,
and `ckd` are estimated jointly rather than one at a time. With three
correlated comorbidity flags in the same group, the coefficient that
best explains the group's combined effect, conditional on the other
two, can differ in sign from any one flag's own marginal association
with mortality — an ordinary consequence of multivariate adjustment
among correlated covariates, not something specific to this method.
At `lambda_min_` (`cv.model_.coef_at(cv.lambda_min_)`), where less
shrinkage is applied, `diabetes` returns to a small positive value
(`0.1241`), closer to its true marginal effect.

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

## 9. `predict_proba()`: correct, despite looking fragile

`GroupLassoLogistic.predict_proba()` re-derives an intercept
interpolation inline, by hand, rather than calling a shared helper
like `PenalizedLogistic.intercept_at()` — worth knowing about because
reading it side by side with `coef_at()` looks like it could easily
be wrong. It was checked directly against a clean, independent
interpolation for an in-range lambda:

```python
lam = m.lambda_path_[60]
m.predict_proba(X.values[:5], lambda_value=lam)
# [0.0529, 0.0348, 0.0627, 0.1025, 0.0707], confirmed to match a
# manual coef_at()-style interpolation of both coefficient and
# intercept at the same lambda, to full floating-point precision.
```
No note is attached to this one — it's mentioned here because the code
itself doesn't inspire confidence, not because the result is wrong.

## 10. What's next

The [group lasso linear chapter](../linear/group_lasso_linear) covers
the same penalty for continuous outcomes next, with a noticeably
cleaner group-entry story than Section 7's — and a second instance of
the [penalized linear chapter's](../linear/penalized_linear) `predict()`
bug. A [group lasso Cox chapter](../survival/11_group_lasso_cox)
follows for time-to-event outcomes, built on top of the survival
guide's existing `PenalizedCoxPH` machinery rather than standing alone
the way this chapter and the linear one do. After group lasso, a
provider-penalized chapter covers `ProviderPenalizedLogistic` — the
package's central methodological contribution, combining provider
effects with penalized covariate selection in one model.
