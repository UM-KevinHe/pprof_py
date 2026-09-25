(penalized_linear)=
# Penalized Linear Regression: Elastic Net for Continuous Outcomes

In R, `PenalizedLinear` is `glmnet(family="gaussian")`, and
`PenalizedLinearCV` is `cv.glmnet(family="gaussian")`. This chapter
assumes you've read the
[penalized logistic chapter](../logistic/penalized_logistic) that
precedes it — the elastic net penalty, the lambda path, and
cross-validated selection between `lambda_min_` and `lambda_1se_` are
all identical in shape here, so this chapter only re-explains what's
actually different for a continuous outcome. Two things are: the
likelihood and the natural-scale interpretation of the coefficients.

## 1. The problem, restated for a continuous outcome

The [logistic chapter's Section 1](../logistic/penalized_logistic)
frames the too-many-covariates problem around event count, because a
binomial likelihood's information accumulates on the rarer outcome
class. An ordinary linear model doesn't have that particular failure
mode — every row contributes information regardless of the outcome's
value — but the underlying problem is the same one in a more familiar
form: with enough candidate covariates relative to $n$, ordinary least
squares coefficients become unstable and overfit, and including
several correlated lab values or comorbidity flags inflates their
individual coefficients' variance even when the *combined* signal
they carry is perfectly well estimated. Elastic net regularization is
the same fix for the same reason — shrink coefficients toward zero,
trading a little bias for a real reduction in variance, and optionally
zero out the covariates that don't earn their place. The penalty
itself is explained in
[survival Chapter 8](../survival/08_penalized_regression) and is not
repeated here.

## 2. From deviance to residual sum of squares

Swap the binomial log-likelihood for the Gaussian one and the same
penalized objective from the logistic chapter applies:

$$
\ell(\beta, \beta_0) - \lambda\left[(1-\alpha)\tfrac{1}{2}\|\beta\|_2^2 + \alpha\|\beta\|_1\right],
\qquad
\ell(\beta, \beta_0) = -\tfrac{1}{2}\sum_i w_i\left(y_i - \eta_i\right)^2,
\quad \eta_i = \mathbf{x}_i^\top\beta + \beta_0
$$

Maximizing $\ell$ alone is exactly weighted ordinary least squares —
maximizing the Gaussian log-likelihood is the same as minimizing the
weighted residual sum of squares. "Deviance" for this family (what
`deviance_path_`, `null_deviance_`, and `deviance_ratio_path_` report)
*is* that weighted RSS, $\sum_i w_i(y_i - \eta_i)^2$ — there's no
extra transformation the way there is between log-odds and probability
for logistic. That equivalence matters later in Section 8, where a
cross-validation attribute turns out to be this exact quantity under a
misleading name.

## 3. The running example: a 30-day cost cohort

Same shape of cohort as the logistic chapter — patients at 60
facilities, ten candidate risk adjusters, two of them (`lab_c`,
`lab_d`) pure noise — but with a continuous outcome: total cost of
care in the 30 days following an index admission, in thousands of
dollars.

```python
import numpy as np
import pandas as pd

def make_cost_cohort(n_patients=6000, n_facilities=60, seed=1):
    """Synthetic 30-day total cost of care (USD, thousands). lab_c and
    lab_d have no effect on cost by construction."""
    rng = np.random.default_rng(seed)

    facility_id = rng.integers(0, n_facilities, n_patients)
    facility_efficiency = rng.normal(0, 1.5, n_facilities)

    age = rng.normal(64, 13, n_patients).clip(18, 96)
    sex = rng.binomial(1, 0.46, n_patients)
    diabetes = rng.binomial(1, 0.42, n_patients)
    chf = rng.binomial(1, 0.28, n_patients)
    comorbidity_count = rng.poisson(1.4, n_patients)
    prior_admissions = rng.poisson(0.6, n_patients)
    albumin = rng.normal(3.8, 0.5, n_patients)
    bmi = rng.normal(27, 5, n_patients).clip(14, 55)
    lab_c = rng.normal(0, 1, n_patients)   # pure noise
    lab_d = rng.normal(0, 1, n_patients)   # pure noise

    cost = (
        18.0
        + 0.12 * (age - 64)
        + 4.5 * diabetes
        + 7.8 * chf
        + 3.1 * comorbidity_count
        + 2.6 * prior_admissions
        - 2.0 * (albumin - 3.8)
        + 0.05 * (bmi - 27)
        + facility_efficiency[facility_id]
        + rng.normal(0, 5.0, n_patients)
    )
    cost = np.clip(cost, 1.0, None)

    return pd.DataFrame({
        "patient_id": np.arange(n_patients), "facility_id": facility_id,
        "age": age.round(1), "sex": sex, "diabetes": diabetes, "chf": chf,
        "comorbidity_count": comorbidity_count, "prior_admissions": prior_admissions,
        "albumin": albumin.round(2), "bmi": bmi.round(1),
        "lab_c": lab_c.round(3), "lab_d": lab_d.round(3), "cost_30d": cost.round(2),
    })

cohort = make_cost_cohort()
candidates = ["age", "sex", "diabetes", "chf", "comorbidity_count",
              "prior_admissions", "albumin", "bmi", "lab_c", "lab_d"]
X = cohort[candidates]
y = cohort["cost_30d"].values
```

```
   patient_id  facility_id   age  sex  diabetes  chf  comorbidity_count  prior_admissions  albumin   bmi  lab_c  lab_d  cost_30d
0           0           28  96.0    1         0    0                  2                 0     4.25  22.0 -0.059 -0.909     33.89
1           1           30  54.9    1         0    0                  0                 1     3.84  30.5 -1.304  0.324     16.46
2           2           45  70.7    0         1    0                  1                 0     3.21  24.4  1.686 -0.775     32.66
3           3           57  69.4    0         0    1                  1                 2     4.26  22.6 -0.074 -0.092     35.56
4           4            2  78.4    1         1    1                  1                 0     4.69  28.3  0.551 -0.203     33.86
```

6,000 patients, mean cost \$28.18k (std \$8.09k, range \$1.92k–\$59.74k).

## 4. Fitting the full path

```python
from pprof_py import PenalizedLinear

lasso = PenalizedLinear(alpha=1.0)
lasso.fit(X, y)

lasso.lambda_max_          # 3.6976337628046547
lasso.n_nonzero_path_[[0, 10, 30, 50, 70, 99]]
# array([ 0,  4,  7,  7,  9, 10])
```

The mechanics are identical to the logistic case: `coef_` doesn't
exist after this call (`hasattr(lasso, "coef_")` is `False` — only set
for a single-lambda fit), so pull coefficients from `coef_path_` by
position or from `coef_at()` by an arbitrary lambda:

```python
lam = lasso.lambda_path_[70]   # 0.005490868783205466
lasso.coef_at(lam)
```
```
age                  0.1072
sex                 -0.0593
diabetes             4.4017
chf                  8.0040
comorbidity_count    3.1146
prior_admissions     2.5454
albumin             -2.1096
bmi                  0.0619
lab_c                0.0061
lab_d                0.0000
```

These are already on the natural, dollar scale — a coefficient of
4.40 means \$4,400 more cost associated with diabetes, holding the
other retained covariates fixed. No exponentiation step, unlike the
odds ratios the logistic chapter needed.

```{note}
As with `PenalizedLogistic`, the attributes actually set by `fit()`
run well beyond what either class's docstring documents (in fact
`PenalizedLinear` and `PenalizedLinearCV` have no "Attributes" section
in their docstrings at all). Everything shown in this chapter —
`column_scale_`, `converged_path_`, `deviance_path_`,
`lambda_min_ratio_`, `log_likelihood_path_`, `n_nonzero_path_`,
`null_deviance_`, `penalty_factor_`, alongside the documented
`coef_path_`/`intercept_path_`/`lambda_path_`/`lambda_max_`/
`deviance_ratio_path_` — is real and was confirmed on a fitted
instance.
```

## 5. Two parameters `PenalizedLogistic` has that this class doesn't

Two gaps are worth flagging explicitly rather than letting you
discover them by a `TypeError` or a silent difference in behavior:

- **`use_active_set`.** `PenalizedLogistic` accepts
  `use_active_set=True` to accelerate coordinate descent by skipping
  variables that are already zero and satisfy the KKT condition.
  `PenalizedLinear` has no such parameter — every fit runs the full
  (non-active-set) coordinate descent, regardless of how sparse the
  solution ends up being.
- **`n_iter_path_`.** `PenalizedLogistic` records outer-loop iteration
  counts per lambda; `PenalizedLinear` does not store this at all
  (`hasattr(lasso, "n_iter_path_")` is `False`). `converged_path_`
  (a per-lambda boolean) is still available on both classes.

Both gaps reflect wiring differences between the two classes rather
than a fundamental limitation of the underlying solver.

## 6. The outer loop: usually one step, with one wrinkle

Section 2 noted that maximizing the Gaussian log-likelihood is exactly
weighted least squares. Unlike the logistic case, the curvature here —
`linear_information(X, weight) = X.T @ diag(weight) @ X` — does not
depend on $\beta$ at all, so the quadratic model the outer loop builds
is *exact*, not a local approximation. Calling the package's own
solver directly, bypassing the class for a moment, confirms this: with
a plain (non-intercept-tracking) Gaussian objective, every one of 100
lambdas converges in exactly one outer iteration:

```python
from pprof_py.algorithms.coordinate_descent import fit_regularization_path
from pprof_py.algorithms.linear.likelihood import build_linear_objective
# ... (X_fit, pf, lambda sequence set up as in Section 4) ...
results = fit_regularization_path(
    build_linear_objective(X_fit, y, weight), p, c, alpha=1.0,
    lambda_sequence=lam_seq, penalty_factor=pf,
)
[r.n_outer_iter for r in results][:5]   # [1, 1, 1, 1, 1]
[r.converged for r in results].count(True)   # 100
```

`PenalizedLinear.fit()` itself doesn't call the solver quite this
plainly, though — it wraps the objective in a closure that also
updates the (unpenalized) intercept by its own Newton step every time
the objective is evaluated, the same design
[the logistic chapter's Section 5](../logistic/penalized_logistic)
described. The same consequence follows here: `converged_path_` on the
fit from Section 4 is `True` for only 11 of the 100 lambdas, even
though refitting with `fit_intercept=False` converges cleanly at every
one (100/100). This intercept-tracking wrinkle affects
the convergence *flag* but not the actual coefficients — the
coefficients this chapter reports are stable regardless of the flag.

## 7. Standardization

`standardize=True` behaves exactly as
[described for the logistic model](../logistic/penalized_logistic):
divide each column by its weighted population standard deviation,
leave centering to the (unpenalized) intercept, and transform
`coef_path_` back to the original scale before storing it — that
mechanism is family-agnostic and isn't repeated here. One difference
worth knowing about: if every candidate column happens to be
degenerate (zero weighted variance — e.g. a constant column), the
error `PenalizedLinear` raises is `"penalty_factor cannot be all-zero
(nothing would be penalized)"`, which is technically true but blames
the wrong thing — `PenalizedLogistic` catches the same situation
earlier and reports the actual cause, `"All predictors have zero
weighted variance"`. This won't come up with real covariates that vary at all, so
it doesn't affect the cohort in this chapter.

## 8. Cross-validation with `PenalizedLinearCV`

```python
cv = PenalizedLinearCV(alpha=1.0, n_folds=10, random_state=0)
cv.fit(X, y)

cv.lambda_min_   # 0.03216013277245188
cv.lambda_1se_   # 0.29992647459797356
cv.lambda_       # 0.29992647459797356 -- lambda_1se_, se_rule="1se" default
```

`cv.coef_` (at `lambda_1se_`):

```
age                  0.0843
sex                  0.0000
diabetes             3.7647
chf                  7.3234
comorbidity_count    2.8734
prior_admissions     2.1690
albumin             -1.5346
bmi                  0.0000
lab_c                0.0000
lab_d                0.0000
```

Both `lab_c` and `lab_d` are correctly zeroed, along with `sex` and
`bmi` — five covariates survive, the same variable-selection story
Section 7 of the logistic chapter told, for the same reason (LASSO at
`lambda_1se_` prefers the simpler model whenever cross-validated
performance can't statistically tell the difference). At
`lambda_min_` (via `cv.model_.coef_at(cv.lambda_min_)`), `lab_c` and
`lab_d` are still correctly zero here, though `sex` picks up a small
nonzero coefficient (-0.0049) that isn't really there in the
data-generating process — the same finite-sample noise Chapter 8 warns
about, just landing on a different covariate than it did for the
logistic fit on this cohort's own noise.

## 9. Ridge, LASSO, and elastic net on the same cohort

Each read at its own cross-validated `lambda_min_`, as in the logistic
chapter:

| | ridge ($\alpha{=}0$) | elastic net ($\alpha{=}0.5$) | lasso ($\alpha{=}1$) |
|---|---:|---:|---:|
| age | 0.0790 | 0.1065 | 0.1051 |
| sex | -0.0275 | -0.0571 | -0.0049 |
| diabetes | 3.1647 | 4.3724 | 4.3431 |
| chf | 5.8147 | 7.9536 | 7.9405 |
| comorbidity_count | 2.2784 | 3.0955 | 3.0927 |
| prior_admissions | 1.8647 | 2.5295 | 2.5117 |
| albumin | -1.5987 | -2.0978 | -2.0596 |
| bmi | 0.0418 | 0.0613 | 0.0561 |
| lab_c | 0.0017 | 0.0053 | 0.0000 |
| lab_d | -0.0112 | 0.0000 | 0.0000 |

The pattern is the same one the logistic chapter found: ridge shrinks
every coefficient furthest toward zero without ever reaching it
exactly; LASSO and the elastic net sit closer to the unpenalized
magnitude at their own (smaller) cross-validated lambdas, and mostly —
not perfectly — separate the two noise covariates from the eight real
ones. `lab_c` picks up a small nonzero coefficient under ridge and
elastic net, and ridge alone gives `lab_d` a small nonzero coefficient
too; only LASSO zeroes both at `lambda_min_` on this particular draw of
the data. As before, this is a matter of cross-validated noise at the
level of individual coefficients, not a general property of one
penalty being more "correct" than another — see the logistic chapter's
Section 8 for the fuller discussion, which applies here without
modification.

### When to reach for a penalized linear model

The calculus is the same as
[the logistic chapter's guidance](../logistic/penalized_logistic):
reach for `PenalizedLinear`/`PenalizedLinearCV` when you have more
candidate cost or outcome drivers than you're confident belong in the
model, want automatic selection or prediction accuracy over a specific
reportable coefficient, or are working with correlated candidates.
Stay with the unpenalized `LinearFixedEffectModel` or
`LinearRandomEffectModel` when you need valid standard errors for a
small, pre-specified covariate set — and remember penalized
coefficients here carry the same no-inference caveat survival Chapter
8 §8.6 states for Cox and this family's logistic sibling repeats in
its own Section 8: shrinkage is the mechanism, not a side effect, so
`PenalizedLinear` reports no standard errors for its coefficients and
none should be inferred from them.

## 10. Comparison with R's `glmnet`

Everything Section 11 of the logistic chapter says about matching
`glmnet` 4.1-8's conventions — the lambda sequence, standardization
without centering, penalty-factor rescaling — applies identically to
the Gaussian family; both classes share the same lambda-sequence and
standardization code (`algorithms/coordinate_descent.py`,
`algorithms/penalty.py`). As with the logistic model, this reflects
the implementation's documented design target rather than a
numerically-verified comparison against live R output — no
`R_COMPATIBILITY.md`-style validation report exists yet for either
penalized GLM family.

## 11. What's next

A group lasso chapter extends this chapter's element-wise penalty to
handle groups of related covariates as a single structured unit — the
same extension the [penalized logistic chapter](../logistic/penalized_logistic)
points to next, now for continuous outcomes.
