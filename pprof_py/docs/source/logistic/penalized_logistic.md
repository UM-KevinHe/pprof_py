(penalized_logistic)=
# Penalized Logistic Regression: Elastic Net for Binary Outcomes

In R, the model in this chapter is `glmnet(family="binomial")` for
`PenalizedLogistic`, and `cv.glmnet(family="binomial")` for
`PenalizedLogisticCV`. If you've used `glmnet` before, the lambda path,
the `alpha` mixing parameter, and the `lambda.min`/`lambda.1se`
distinction will all be familiar — this chapter is as much about where
`pprof_py`'s implementation matches that convention as it is about the
statistics.

## 1. The problem: many risk adjusters, few deaths

The logistic fixed- and random-effect chapters elsewhere in these docs
both work with a handful of deliberately chosen covariates — age, a
comorbidity count, one or two flags. Real risk-adjustment work rarely
starts that clean. A hospital claims extract might hand you thirty
comorbidity flags, a dozen lab values, admission-source indicators, and
every interaction someone on the clinical team has ever suspected
matters. Throwing all of them into an ordinary (unpenalized) logistic
fit runs into the same failure mode
[survival Chapter 8](../survival/08_penalized_regression) describes for
Cox models — and for the same underlying reason. There, what mattered
was the number of *events*, not the number of rows, because the partial
likelihood accumulates information on deaths. Here, it's the number of
outcomes in the *rarer* class. A 30-day post-discharge mortality outcome
is typically single digits to low teens as a percentage — in the cohort
built below, it's 6.05%. If you have 6,000 patients but only 363 deaths,
you effectively have 363 "events" to spend across an intercept and
every covariate you include, not 6,000. Push that ratio too far and
coefficients inflate, standard errors balloon, and the fit becomes
sensitive to which handful of patients happened to die in your
particular sample — precisely the instability that motivates
penalization.

The fix is the same one: shrink coefficients toward zero by adding a
penalty to what the model maximizes. The penalty itself — ridge,
LASSO, elastic net, the `alpha` and `lambda` parameters, the
coefficient path, cross-validated selection between `lambda_min_` and
`lambda_1se_` — is explained in full in
[survival Chapter 8](../survival/08_penalized_regression); nothing
about the *shape* of the penalty changes here, so this chapter doesn't
re-derive it. What changes is the likelihood the penalty is attached
to.

## 2. From partial likelihood to binomial deviance

Chapter 8 penalizes the Cox partial likelihood. `PenalizedLogistic`
penalizes the ordinary binomial log-likelihood under the logit link —
the same likelihood `LogisticFixedEffectModel` maximizes, just without
provider effects, and now with a penalty term:

$$
\ell(\beta, \beta_0) - \lambda\left[(1-\alpha)\tfrac{1}{2}\|\beta\|_2^2 + \alpha\|\beta\|_1\right],
\qquad
\ell(\beta, \beta_0) = \sum_i w_i\Big[y_i\eta_i - \log(1 + e^{\eta_i})\Big],
\quad \eta_i = \mathbf{x}_i^\top\beta + \beta_0
$$

In words: find the coefficients $\beta$ that make the observed 0/1
outcomes as likely as possible under the logistic curve, minus a cost
that grows with the size of $\beta$. $\lambda$ and $\alpha$ do exactly
what Chapter 8 §8.1–§8.2 describes — $\lambda$ is overall strength,
$\alpha$ is the ridge/LASSO mix — and the practical consequences carry
over unchanged: LASSO ($\alpha=1$) can zero a coefficient out entirely
and so performs automatic variable selection; ridge ($\alpha=0$)
shrinks everything a little without zeroing anything; elastic net sits
between the two and handles correlated covariates more gracefully than
LASSO alone. The intercept $\beta_0$ is never penalized in any of
these models — it is fit by its own Newton update at every step, not
by coordinate descent, so that the model's overall calibration
(average predicted risk matching the average observed rate) isn't
traded away for sparsity elsewhere.

## 3. The running example: a 30-day mortality cohort

Following this guide's convention of grounding every chapter in
provider-profiling data, the example below is a synthetic cohort of
patients at 60 dialysis facilities, each with a 30-day post-discharge
mortality outcome and ten candidate risk adjusters — eight with a real
effect on mortality, and two (`lab_c`, `lab_d`) that are pure noise,
included on purpose so the variable-selection story later in this
chapter has something honest to show.

```python
import numpy as np
import pandas as pd

def make_mortality_cohort(n_patients=6000, n_facilities=60, seed=0):
    """Synthetic 30-day post-discharge mortality cohort. lab_c and
    lab_d have no effect on the outcome by construction."""
    rng = np.random.default_rng(seed)

    facility_id = rng.integers(0, n_facilities, n_patients)
    facility_quality = rng.normal(0, 0.2, n_facilities)

    age = rng.normal(64, 13, n_patients).clip(18, 96)
    sex = rng.binomial(1, 0.46, n_patients)
    diabetes = rng.binomial(1, 0.42, n_patients)
    chf = rng.binomial(1, 0.28, n_patients)
    comorbidity_count = rng.poisson(1.4, n_patients)
    prior_admissions = rng.poisson(0.6, n_patients)
    albumin = rng.normal(3.8, 0.5, n_patients)       # g/dL, lower = sicker
    bmi = rng.normal(27, 5, n_patients).clip(14, 55)
    lab_c = rng.normal(0, 1, n_patients)             # pure noise
    lab_d = rng.normal(0, 1, n_patients)             # pure noise

    log_odds = (
        -3.6
        + 0.038 * (age - 64)
        + 0.30 * diabetes
        + 0.55 * chf
        + 0.22 * comorbidity_count
        + 0.15 * prior_admissions
        - 0.45 * (albumin - 3.8)
        + 0.01 * (bmi - 27)
        + facility_quality[facility_id]
    )
    p = 1.0 / (1.0 + np.exp(-log_odds))
    death_30d = rng.binomial(1, p)

    return pd.DataFrame({
        "patient_id": np.arange(n_patients), "facility_id": facility_id,
        "age": age.round(1), "sex": sex, "diabetes": diabetes, "chf": chf,
        "comorbidity_count": comorbidity_count, "prior_admissions": prior_admissions,
        "albumin": albumin.round(2), "bmi": bmi.round(1),
        "lab_c": lab_c.round(3), "lab_d": lab_d.round(3), "death_30d": death_30d,
    })

cohort = make_mortality_cohort()
candidates = ["age", "sex", "diabetes", "chf", "comorbidity_count",
              "prior_admissions", "albumin", "bmi", "lab_c", "lab_d"]
X = cohort[candidates]
y = cohort["death_30d"].values
```

```
   patient_id  facility_id   age  sex  diabetes  chf  comorbidity_count  prior_admissions  albumin   bmi  lab_c  lab_d  death_30d
0           0           51  47.9    0         0    0                  2                 0     3.96  27.7 -0.637 -0.867          0
1           1           38  53.5    0         1    0                  2                 0     3.75  29.4  0.537  0.248          0
2           2           30  64.7    1         0    0                  1                 1     3.80  24.3 -0.816 -1.256          0
3           3           16  38.9    1         0    0                  1                 1     3.67  31.1  1.296  0.616          0
4           4           18  65.3    1         1    0                  0                 0     3.64  31.1  1.110 -0.442          0
```

6,000 patients, 60 facilities, 363 deaths (6.05%). `facility_id` is
included for realism and for later chapters — this chapter's models
don't take a `provider_id=` argument at all; provider effects and
penalization meet in a later chapter ({ref}`penalized-vs-unpenalized`
below says more about that division of labor).

## 4. Fitting the full path

```python
from pprof_py import PenalizedLogistic

lasso = PenalizedLogistic(alpha=1.0)   # n_lambda=100, standardize=True,
lasso.fit(X, y)                        # fit_intercept=True are all defaults
```

As in Chapter 8, this fits an entire *path* of 100 models, not one:

```python
lasso.lambda_max_          # 0.021890177718538...
lasso.lambda_path_.shape   # (100,)
lasso.lambda_path_[:3]     # [0.02189018, 0.01994551, 0.01817361]
lasso.lambda_path_[-3:]    # [2.63667966e-06, 2.40244430e-06, 2.18901777e-06]
```

`lambda_max_` is the smallest $\lambda$ at which every penalized
coefficient is exactly zero — the left edge of the path. The 100
values in `lambda_path_` step down geometrically from there by the
default `lambda_min_ratio` (here $10^{-4}$, since $n=6{,}000 >
p=10$), matching `glmnet`'s own
`exp(seq(log(lambda_max), log(lambda_max*lambda.min.ratio), length.out=nlambda))`
construction. `n_nonzero_path_` shows coefficients entering the model
as $\lambda$ falls:

```python
lasso.n_nonzero_path_[[0, 10, 30, 50, 70, 99]]
# array([ 0,  5, 10, 10, 10, 10])
```

By the 30th lambda, all ten candidates already have nonzero
coefficients at this level of penalization — with only two genuinely
irrelevant covariates among ten reasonably-informative ones, LASSO
doesn't need to travel far down the path before including everything.
Section 7 revisits this with cross-validation, where the picture looks
rather different.

### `coef_` only exists for a single-lambda fit

This is worth stating plainly because it's easy to trip over:
`lasso.coef_` does not exist after the call above.

```python
hasattr(lasso, "coef_")   # False
```

`coef_`, `intercept_`, and `lambda_` are only set when `fit()` produces
a path of length one — i.e. when you pass a scalar `lambda_path=`
instead of letting `n_lambda` generate the full grid. For the ordinary
full-path fit, pull coefficients out of the path explicitly, either by
integer position (`coef_path_[i]`, shape `(100, 10)`) or by an
arbitrary $\lambda$ value via `coef_at()`, which linearly interpolates
in $\log\lambda$ between the two bracketing path points — and its
counterpart `intercept_at()`:

```python
lam = lasso.lambda_path_[70]          # 3.2506e-05, near the unpenalized end
lasso.coef_at(lam)
```
```
age                  0.0316
sex                  0.1036
diabetes             0.4412
chf                  0.4177
comorbidity_count    0.2293
prior_admissions     0.0552
albumin             -0.5476
bmi                  0.0107
lab_c                0.1154
lab_d               -0.0866
```
```python
lasso.intercept_at(lam)   # -3.8759149577528182
```

Coefficients come back on the *original* covariate scale — a raw log-odds
per year of age, per g/dL of albumin, and so on — even though fitting
happens internally on standardized columns (Section 6). There's no
manual rescaling step for you to get wrong.

```{note}
The full attribute surface after `fit()` is broader than the class
docstring lists: alongside `coef_path_`, `intercept_path_`,
`lambda_path_`, `lambda_max_`, and `deviance_ratio_path_`, a fitted
`PenalizedLogistic` also carries `column_scale_`, `converged_path_`,
`deviance_path_`, `feature_names_in_`, `lambda_min_ratio_`,
`log_likelihood_path_`, `n_iter_path_`, `n_nonzero_path_`,
`null_deviance_`, and `penalty_factor_` — all used in this chapter.
```

## 5. Why fitting takes so many iterations here

`PenalizedLogistic` and `PenalizedCoxPH` share the same outer-loop
solver (proximal Newton, cycling coordinate descent on the penalized
quadratic at each step), but they need different numbers of outer
iterations for a structural reason. The binomial log-likelihood isn't
quadratic in $\beta$ — its curvature, $p_i(1-p_i)$, depends on the
current fitted probabilities, which depend on $\beta$. Each outer step
re-linearizes around the current coefficients (this *is* IRLS —
iteratively reweighted least squares), solves that local quadratic
exactly by coordinate descent, then has to re-evaluate the curvature
before it can trust the next step. `n_iter_path_` records how many
outer steps each lambda actually took:

```python
lasso.n_iter_path_.min(), lasso.n_iter_path_.max(), lasso.n_iter_path_.mean()
# (1, 100, 99.01)
lasso.n_iter_path_[:5]
# array([  1, 100, 100, 100, 100])
```

```{note}
Almost every lambda in the default path runs all the way to
`max_outer_iter` (100) without the solver's own convergence flag
(`converged_path_`) ever firing — only the trivial, all-zero first
lambda formally converges. This traces to how the (unpenalized)
intercept is updated: it's folded into the same objective evaluation
the outer loop uses for its convergence check, as a side effect, which
keeps nudging that check just enough to prevent it from settling under
the default `outer_tol=1e-9`. Refitting with `fit_intercept=False`
converges cleanly at every lambda (100/100, mean 2.4 iterations), which
isolates the cause. The **coefficients themselves are stable** —
loosening `outer_tol` to `1e-6` changes `coef_path_` by at most 2e-4
while needing far fewer iterations — so treat `n_iter_path_` sitting at
100 as a (currently unresolved) reporting quirk rather than evidence
the fit failed.
```

The [penalized linear chapter](../linear/penalized_linear) that
follows this one shows the other end of this contrast: a continuous
outcome under Gaussian errors has *constant* curvature, so its outer
loop is structurally simpler — when the same intercept-update wrinkle
isn't in play, one step suffices.

## 6. Standardization and the intercept

`standardize=True` (the default) puts every candidate covariate on a
common scale before penalizing, by dividing by its weighted population
standard deviation — `column_scale_` after the fit above:

```
age                  12.980
sex                   0.499
diabetes              0.495
chf                   0.447
comorbidity_count     1.204
prior_admissions      0.771
albumin               0.494
bmi                   4.950
lab_c                 1.001
lab_d                 1.004
```

This matters because the elastic net penalty is not scale-invariant:
without standardizing first, `age` (spread over roughly 13 years of
standard deviation) would need a much larger raw coefficient than a
0/1 flag like `diabetes` to represent a comparable effect, so an
unstandardized fit would penalize `age` far more heavily than it
penalizes `diabetes` for no substantive reason. Standardizing first
means every covariate starts the path competing on equal footing;
`coef_path_` is then transformed back to the original scale (dividing
by `column_scale_`) before it's stored, which is exactly what produced
the directly-interpretable log-odds coefficients in Section 4. Note
what standardization does *not* do: it doesn't center columns. Under
the hood, centering is unnecessary because the (always-unpenalized)
intercept absorbs it via its own Newton update at every step — this
matches `glmnet`'s convention exactly. Set `standardize=False` only if
your covariates are already on a deliberately comparable scale (all
0/1 flags, say) and you want the penalty applied to raw coefficient
magnitude directly.

`fit_intercept=False` removes $\beta_0$ from the model entirely (every
`intercept_path_` entry is exactly 0.0) rather than merely leaving it
unpenalized-but-present — use it only if you have a specific reason to
force predictions through the origin on the linear-predictor scale,
which is rare for a mortality outcome with a 6% base rate.

## 7. Cross-validation: picking a lambda with `PenalizedLogisticCV`

The path tells you what's *available*; it doesn't tell you which
lambda to actually report or predict with. That's what
`PenalizedLogisticCV` — R's `cv.glmnet(family="binomial")` — is for,
using the same k-fold procedure Chapter 8 §8.4 describes, with
binomial deviance as the held-out scoring criterion instead of
partial-likelihood deviance:

```python
from pprof_py import PenalizedLogisticCV

cv = PenalizedLogisticCV(alpha=1.0, n_folds=10, random_state=0)
cv.fit(X, y)

cv.lambda_min_   # 2.961845624722433e-05
cv.lambda_1se_   # 0.007866921978799356
cv.lambda_       # 0.007866921978799356  -- lambda_1se_, since se_rule="1se" by default
```

Folds are assigned by `_stratified_fold_assignment`, which splits the
death and non-death groups into folds separately before combining
them — with only 363 deaths total, an ordinary unstratified random
split could easily leave some folds with very few events, making the
per-fold deviance unstable in exactly the way Section 1 warned about.

`cv.coef_` — the sparse, default answer — sits at `lambda_1se_`:

```python
cv.coef_   # a pandas Series here for readability; coef_ itself is a plain ndarray
```
```
age                  0.0196
sex                  0.0000
diabetes             0.1450
chf                  0.0926
comorbidity_count    0.1249
prior_admissions     0.0000
albumin             -0.2399
bmi                  0.0000
lab_c                0.0000
lab_d                0.0000
```
```python
cv.intercept_   # -3.3997566856995496
```

Both pure-noise covariates — `lab_c` and `lab_d` — are correctly
zeroed out at `lambda_1se_`, along with `sex`, `prior_admissions`, and
`bmi`. Compare that against `lambda_min_`, reached via
`cv.model_.coef_at(cv.lambda_min_)` (there's no separate `.fit()` needed
— `cv.model_` is the same full-path `PenalizedLogistic` object the CV
loop already fit on all the data):

```
age                  0.0316
sex                  0.1037
diabetes             0.4414
chf                  0.4179
comorbidity_count    0.2293
prior_admissions     0.0552
albumin             -0.5477
bmi                  0.0107
lab_c                0.1154
lab_d               -0.0866
```

At `lambda_min_`, nothing is zero — both noise variables carry small
but nonzero coefficients. This is the same honest lesson survival
Chapter 8 §8.5 draws from its own noise covariates: `lambda_min_`
optimizes a *point estimate* of cross-validated deviance, which is
itself noisy, and with only 363 events that noise is enough to let a
couple of irrelevant covariates sneak through. `lambda_1se_` — the
largest $\lambda$ within one standard error of the best observed
deviance — trades a small amount of that point-estimate performance
for a model that's actively sparse, which is why it's this class's
default (`se_rule="1se"`):

```python
cv.cv_mean_deviance_[cv.lambda_min_idx_], cv.cv_mean_deviance_[cv.lambda_1se_idx_]
# (262.31096721525..., 265.45426807974...)
cv.cv_se_deviance_[cv.lambda_min_idx_]
# 3.626404928283607
```

265.45 sits well inside one standard error (3.63) of 262.31 — the
1se rule's entire premise is that this difference in cross-validated
deviance is not distinguishable from noise, so there's no real cost to
preferring the simpler, five-covariate model.

```{note}
`cv_se_deviance_` — the quantity `lambda_1se_` is actually computed
from — isn't listed in `PenalizedLogisticCV`'s docstring, which
mentions only `cv_mean_deviance_` and `cv_std_deviance_`. It exists and
behaves as shown here.
```

## 8. Ridge, LASSO, and elastic net on the same cohort

Cross-validating all three at `alpha=0` (ridge), `alpha=0.5` (elastic
net), and `alpha=1` (LASSO), each read at its own `lambda_min_`:

```python
for alpha, name in [(0.0, "ridge"), (0.5, "elastic_net_0.5"), (1.0, "lasso")]:
    cv_a = PenalizedLogisticCV(alpha=alpha, n_folds=10, random_state=0)
    cv_a.fit(X, y)
    print(name, cv_a.model_.coef_at(cv_a.lambda_min_))
```

| | ridge ($\alpha{=}0$) | elastic net ($\alpha{=}0.5$) | lasso ($\alpha{=}1$) |
|---|---:|---:|---:|
| age | 0.0290 | 0.0313 | 0.0316 |
| sex | 0.0968 | 0.0988 | 0.1037 |
| diabetes | 0.4062 | 0.4348 | 0.4414 |
| chf | 0.3844 | 0.4111 | 0.4179 |
| comorbidity_count | 0.2139 | 0.2270 | 0.2293 |
| prior_admissions | 0.0509 | 0.0521 | 0.0552 |
| albumin | -0.5032 | -0.5410 | -0.5477 |
| bmi | 0.0099 | 0.0101 | 0.0107 |
| lab_c | 0.1058 | 0.1126 | 0.1154 |
| lab_d | -0.0794 | -0.0839 | -0.0866 |

At `lambda_min_`, ridge never zeroes anything by construction, and — on
this particular cohort — neither does LASSO at its own `lambda_min_`,
so the three columns end up looking more like a smooth shrinkage
gradient (ridge pulling hardest toward zero, LASSO least) than three
qualitatively different variable-selection outcomes. That's a real,
useful thing to notice: the *sparsity* story from Section 7 comes from
moving along LASSO's own path to `lambda_1se_`, not from switching
`alpha` at a fixed point in cross-validated deviance. If your goal is
selection specifically, cross-validate LASSO (or an elastic net close
to it) and read `lambda_1se_`, not `lambda_min_`.

(penalized-vs-unpenalized)=
### When to reach for a penalized model at all

Nothing in `pprof_py`'s logistic family fits a plain, unpenalized
model with an arbitrary covariate set and no provider structure —
the two unpenalized options are `LogisticFixedEffectModel` and
`LogisticRandomEffectModel`, both of which model provider effects
directly. `PenalizedLogistic`/`PenalizedLogisticCV` sit at a different
point in the design space entirely: no provider effects, but principled
shrinkage and selection over however many risk adjusters you're willing
to propose as candidates. Reach for penalization when:

- you have more candidate covariates than you're confident belong in
  the model, and want the data to help decide which survive (variable
  selection via LASSO or an elastic net close to it), or
- your goal is predictive accuracy rather than a specific, reportable
  covariate effect, or
- you have collinear or redundant candidates (several correlated lab
  values measuring similar underlying physiology) where an
  unpenalized fit would produce unstable, sign-flipping coefficients.

Stay unpenalized when you have a small, pre-specified, clinically
justified covariate set and need valid standard errors, confidence
intervals, or p-values to report — survival Chapter 8 §8.6's caution
applies here without modification: penalized coefficients are biased
toward zero *by design*, so `PenalizedLogistic` reports no standard
errors or p-values for them, and none should be inferred. The standard
pattern is to use penalization to find which covariates matter, then
refit only those with an unpenalized model for inference — exactly
what the selection-then-refit workflow in
[survival Chapter 9](../survival/09_variable_selection) does for Cox
models.

If you need *both* provider effects and covariate selection at once —
the actual provider-profiling problem, not a simplification of it —
that combination is its own model family, covered in a later chapter
(`ProviderPenalizedLogistic`), built directly on the machinery in this
chapter.

## 9. Reading coefficients as odds ratios

Logistic coefficients live on the log-odds scale, so interpreting them
means exponentiating. Using the sparse `lambda_1se_` coefficients from
Section 7:

```python
import numpy as np
np.exp(cv.coef_[candidates.index("diabetes")])   # 1.156
```

- **diabetes**: coefficient 0.1450 → odds ratio $e^{0.1450} = 1.156$ —
  patients with diabetes have 15.6% higher odds of 30-day death,
  holding the other retained covariates fixed.
- **chf**: coefficient 0.0926 → OR 1.097, a 9.7% higher odds.
- **comorbidity_count**: coefficient 0.1249 → OR 1.133 *per additional
  comorbidity* — each extra comorbidity multiplies the odds by 1.133,
  so three additional comorbidities compound to roughly
  $1.133^3 \approx 1.45\times$.
- **albumin**: coefficient -0.2399 → OR 0.787 per +1 g/dL — each
  additional gram per deciliter of albumin (a marker of better
  nutritional status) is associated with 21.3% *lower* odds of death.

These are the same shrunk, no-inference coefficients Section 8
discussed — useful for ranking risk factors and for prediction, not
for a reportable confidence interval.

## 10. Predicting for new patients

```python
new_patients = pd.DataFrame({
    "age": [58.0, 81.0], "sex": [1, 0], "diabetes": [0, 1], "chf": [0, 1],
    "comorbidity_count": [1, 4], "prior_admissions": [0, 2],
    "albumin": [4.0, 3.1], "bmi": [26.0, 24.0], "lab_c": [0.0, 0.0], "lab_d": [0.0, 0.0],
})
cv.predict_proba(new_patients)
# array([0.0431, 0.1391])
```

The first patient — 58, no diabetes or CHF, one comorbidity, normal
albumin — gets a 4.3% predicted 30-day mortality risk; the second — 81,
diabetic with CHF, four comorbidities, low albumin — gets 13.9%,
roughly a 3.2-fold difference driven entirely by the five covariates
`lambda_1se_` kept. `cv.predict_proba()` and `cv.predict()` (a
thresholded 0/1 call, default cutoff 0.5) both default to `lambda_`
(the CV-selected value) when `lambda_value=` is omitted, and both
correctly interpolate coefficient *and* intercept at whatever lambda
is requested — `PenalizedLogistic.intercept_at()` exists and is used
here, unlike its linear-model counterpart (the
[next chapter](../linear/penalized_linear)'s Section 8 covers that
distinction in detail).

For a lambda-indexed view of a single fitted point rather than a
prediction, `summary()` gives a small DataFrame plus metadata in
`.attrs`:

```python
lasso.summary(which=cv.lambda_1se_idx_)
```
```
             feature      coef  nonzero
0                age  0.019552     True
1                sex  0.000000    False
2           diabetes  0.144984     True
3                chf  0.092598     True
4  comorbidity_count  0.124900     True
5   prior_admissions  0.000000    False
6            albumin -0.239915     True
7                bmi  0.000000    False
8              lab_c  0.000000    False
9              lab_d  0.000000    False
```
```python
# .attrs: {'lambda': 0.0079, 'deviance_ratio': 0.035, 'n_nonzero': 5}
```

`deviance_ratio` (0.035 here) is the fraction of null deviance this
sparse, heavily-penalized model explains — modest, as expected for a
5-covariate model chosen for stability and interpretability rather
than maximum fit; `lasso.deviance_ratio_path_[-1]` (the least-penalized
end of the path) reaches 0.051, the ceiling these ten covariates can
explain at all.

## 11. Comparison with R's `glmnet`

`pprof_py`'s penalized logistic solver documents itself, at the module
level, as matching `glmnet` 4.1-8's actual source rather than a
textbook description of elastic net — specifically the lambda
sequence construction (Section 4), the weighted-population-standard-deviation
standardization with the intercept absorbing centering (Section 6),
and the convention that user-supplied `penalty_factor` values are
internally rescaled to sum to the number of penalized features. Those
are the claims made in the source and relied on throughout this
chapter; unlike the survival guide's
[R compatibility notes](../survival/R_COMPATIBILITY), which report
coefficient-level agreement checked against live R output during
development, no equivalent side-by-side numerical validation report
exists yet for the logistic (or linear) penalized models — treat the
conventions above as the implementation's documented design target,
not as independently re-verified here.

## 12. What's next

The next chapter, [Penalized Linear Regression](../linear/penalized_linear),
covers the same elastic net machinery for continuous outcomes — cost,
length of stay — where the Gaussian likelihood's constant curvature
simplifies the outer loop, and where one of this chapter's helper
methods (`intercept_at()`) turns out to be missing in a way that
matters for prediction. After that, a group lasso chapter extends the
penalty in this chapter to handle groups of related covariates —
dummy-coded categorical variables, or a lab panel that should enter or
leave the model together — as a single structured unit.
