(provider_penalized_logistic)=
# Provider-Penalized Logistic Regression: Selection and Profiling Together

There's no single R function this chapter maps onto — `pprof_py`'s
own module docstring calls this "R `grplasso::pp.lasso()` for binomial
family," but `pp.lasso` is itself part of the same research lineage
this package grew out of, not a general-purpose CRAN package you'd
already know. This is the model this whole documentation task keeps
gesturing toward: the one that doesn't make you choose between
*estimating provider quality* and *selecting which covariates belong
in the risk adjustment*. It assumes you've read
[Penalized Logistic Regression](penalized_logistic.md) (the lambda path,
`alpha`, cross-validation) and the
[logistic fixed-effect reference](logistic_fixed_effect_model.md) (what a
provider effect $\gamma_k$ is, and why fixed effects — not random —
are the right tool when you don't want to assume providers are
uncorrelated with their own patient mix).

## 1. Why one model, not two

[Survival Chapter 4](../survival/04_indirect_standardization_smr_shr)
opens with the provider-profiling problem in its purest form: a raw
comparison of outcome rates across facilities conflates case mix with
quality, and fixing that means estimating a provider effect *adjusted
for* patient risk factors. Every model in this documentation so far
handles one half of that or the other, never both. The logistic
fixed-effect model (`LogisticFixedEffectModel`) estimates provider
effects $\gamma_k$ jointly with covariates $\beta$, but $\beta$ is
never penalized — you supply a small, pre-chosen covariate set and get
back an unbiased, unpenalized $\gamma_k$ for every provider.
`PenalizedLogistic` penalizes $\beta$ for exactly the reasons
[Chapter 1](penalized_logistic.md) describes, but has no notion of a
provider at all. Neither gives you both.

The practical problem this creates: real risk-adjustment covariate
sets are large enough that you *want* penalization's variable
selection — but you can't just run `PenalizedLogistic` first to pick
covariates, then hand the survivors to `LogisticFixedEffectModel` for
provider effects, treating this as two separate steps. The covariates
you'd select depend on which ones best explain outcomes *net of*
provider quality, which you don't know until you've also estimated
provider effects — and the provider effects you'd estimate depend on
which covariates you've adjusted for. Sequential two-step approaches
solve this by fixing one side and hoping the other doesn't move much;
`ProviderPenalizedLogistic` solves it by not fixing either side —
estimating both simultaneously, each iteration informed by the other's
current estimate.

## 2. The two-layer objective

$$
J(\gamma, \beta) = -\ell(\gamma, \beta) + \lambda \cdot \text{penalty}(\beta), \qquad
\eta_{ij} = \gamma_{k(ij)} + \mathbf{x}_{ij}^\top\beta
$$

$\ell$ is the ordinary binomial log-likelihood
([Penalized Logistic Regression's Section 2](penalized_logistic.md)) —
the same one `LogisticFixedEffectModel` maximizes — with the linear
predictor split into a provider term $\gamma_{k(ij)}$ (patient $j$'s
provider $k$'s effect — unpenalized, one free parameter per provider,
exactly like the fixed-effect model) and a penalized covariate term
$\mathbf{x}_{ij}^\top\beta$ (elastic net, group lasso, or sparse group
lasso, selected via `penalty_type=`). The penalty applies to $\beta$
only — provider effects are never shrunk toward each other, which
matters: shrinking $\gamma_k$ would bias exactly the quantity
([Chapter 4's](../survival/04_indirect_standardization_smr_shr)
entire point) you need unbiased to fairly flag an outlier facility.
Instead, provider effects are stabilized a different way — bounded,
not shrunk (Section 3).

Fitting alternates two updates at every lambda, in a hand-written outer
loop rather than the shared `fit_regularization_path` engine the
element-wise and group lasso classes use (that engine has no notion of
a second parameter block to alternate against):

1. **Update $\gamma$ given the current $\beta$.** A Newton step per
   provider — provider $k$'s score and information come only from
   patients at provider $k$, so this is $K$ independent 1-dimensional
   Newton updates, not a $K$-dimensional joint solve — followed by
   median-clamp bounding (Section 3). Repeated up to `provider_max_iter`
   times (default 10) or until the largest single-provider change
   drops below `outer_tol`.
2. **Update $\beta$ given the current $\gamma$.** With $\gamma$ held
   fixed, this is exactly a penalized weighted least-squares step —
   the same proximal-Newton-plus-coordinate-descent machinery
   [Chapter 1](penalized_logistic.md) describes, called directly rather
   than through the path-fitting wrapper.

The model is fitted as `ProviderPenalizedLogistic(...).fit(X, y,
provider_id)` (Section 4 runs it). `fit()`'s third argument is **positional and named `provider_id`** —
not a keyword, and not `provider`. Keep that in mind if you've just
come from survival Chapter 4, which never names this argument at all
(it works with `strata=`), or if you go on to read the
[provider-penalized Cox chapter](../survival/12_provider_penalized_cox)
next, which takes the same `provider_id=` argument.

## 3. Bounding, not shrinking: `provider_bound`

A provider with very few patients has a poorly identified $\gamma_k$
— in the extreme, a provider with zero deaths in a small sample would
otherwise drive its Newton update toward $-\infty$, chasing a perfect
fit to a tiny amount of data. `provider_bound` (default `10.0`) prevents
this by clamping every provider effect to within `provider_bound` of the
*median* provider effect after each Newton step, rather than shrinking
provider effects toward each other (which would bias exactly the
quantity you're trying to estimate fairly). This is the same
median-clamp mechanism `LogisticFixedEffectModel` uses for its own
$\gamma_k$ bounding — provider-penalized regression inherits it
unchanged rather than reinventing it.

## 4. The running example: mortality with real facility variation

Reusing [Chapter 1's](penalized_logistic.md) mortality cohort exactly —
same ten candidate covariates, two of them (`lab_c`, `lab_d`) pure
noise — since it already has realistic, seeded facility-level quality
variation baked into `facility_id` that this chapter can now recover
and check against.

```python
from pprof_py import ProviderPenalizedLogistic, ProviderPenalizedLogisticCV

X = cohort[candidates]   # the same 10 covariates as Chapter 1
y = cohort["death_30d"].values
provider_id = cohort["facility_id"].values   # 60 facilities

model = ProviderPenalizedLogistic(alpha=1.0)   # elastic_net, alpha=1.0 -> lasso
model.fit(X, y, provider_id)

print(f"lambda_max = {model.lambda_max_:.6f}")   # taken with the facility effects fitted
print(model.n_providers_, "providers; gamma_path_:", model.gamma_path_.shape)
```
```
lambda_max = 0.021897
60 providers; gamma_path_: (100, 60)
```

[Chapter 1's](penalized_logistic.md) `lambda_max_` is 0.021890; this
model's is slightly larger because it is computed with the 60
facility effects fitted, at the null point where every covariate is
zero.

`coef_path_`, `gamma_path_`, and `lambda_path_` follow the same
path-fitting shape as every other class in this documentation, and
`coef_at(lambda_value)` and `intercept_at(lambda_value)` interpolate the
covariate coefficients and the intercept in $\log\lambda$ between path
points, as they do for [Chapter 1's](penalized_logistic.md) class.
Predictions and provider effects, by contrast, use a single path point:

`predict_proba()` and `predict()` accept both a `which=` integer
path index and a `lambda_value=` continuous lambda.  When
`lambda_value` is given and `which` is left at its default (`-1`),
the method resolves to the nearest lambda on the grid:

```python
new_X, new_prov = X.values[:3], provider_id[:3]
p1 = model.predict_proba(new_X, new_prov, which=50)
p2 = model.predict_proba(new_X, new_prov, lambda_value=model.lambda_path_[50])
print(p1.round(4), bool(np.array_equal(p1, p2)))
```
```
[0.0109 0.0652 0.049 ] True
```

## 5. Reading the path, and cross-validating it

```python
print(model.n_nonzero_path_[[0, 10, 20, 30, 40, 50, 70, 99]])
```
```
[ 0  5  7 10 10 10 10 10]
```

Covariates enter the model as lambda falls, the same story
[Chapter 1](penalized_logistic.md) told without provider effects in the
picture. At `which=40`, well into the path:

```python
print(pd.Series(model.coef_path_[40], index=candidates).round(4).to_string())
```
```
age                  0.0315
sex                  0.0742
diabetes             0.4278
chf                  0.3719
comorbidity_count    0.2271
prior_admissions     0.0511
albumin             -0.5344
bmi                  0.0097
lab_c                0.1061
lab_d               -0.0872
```

Close to [Chapter 1's](penalized_logistic.md) own near-unpenalized
values — sensible, since 60 facilities' worth of unpenalized $\gamma$
doesn't change what an *individual patient's* covariates predict about
their own risk, only how facility-level baseline rates get absorbed.

Cross-validating the path:

```python
cv = ProviderPenalizedLogisticCV(alpha=1.0, n_folds=5, random_state=0)
cv.fit(X, y, provider_id)
print(f"lambda_min = {cv.lambda_min_:.6f} (index {cv.lambda_min_idx_}), "
      f"lambda_1se = {cv.lambda_1se_:.6f} (index {cv.lambda_1se_idx_})")
print(pd.Series(cv.coef_, index=candidates).round(4).to_string())   # at lambda_1se_ (se_rule="1se")
```
```
lambda_min = 0.001949 (index 26), lambda_1se = 0.007170 (index 12)
age                  0.0209
sex                  0.0000
diabetes             0.1705
chf                  0.0899
comorbidity_count    0.1358
prior_admissions     0.0000
albumin             -0.2674
bmi                  0.0000
lab_c                0.0000
lab_d                0.0000
```

Cross-validated deviance falls from 547.6 at `lambda_max_` to its
minimum at index 26 and then rises slightly to 527.6 at the end of
the path. The one-standard-error rule picks a sparser point: at
`lambda_1se_` five covariates are in (`age`, `diabetes`, `chf`,
`comorbidity_count`, `albumin`), and the two noise labs, `sex`, `bmi`
and `prior_admissions` are zero. `bmi` and `prior_admissions` have
real but small effects in the data-generating process (0.01 per
kg/m² and 0.15 per admission), which the one-standard-error margin
trades for sparsity. Every one of the 100 path points is converged
(`converged_path_`), as with `ProviderPenalizedCoxPH` in the
[Cox chapter](../survival/12_provider_penalized_cox).

## 6. Provider effects: recovering facility quality

```python
gamma_df = cv.predict_provider_effect()   # DataFrame: provider_id, gamma
print(gamma_df.sort_values("gamma").head(3).to_string(index=False))                   # 3 lowest
print(gamma_df.sort_values("gamma", ascending=False).head(3).to_string(index=False))  # 3 highest

rng = np.random.default_rng(0)            # the cohort's seeded facility quality (make_mortality_cohort)
_ = rng.integers(0, 60, 6000)
true_quality = rng.normal(0, 0.2, 60)
labels = np.unique(provider_id)
print(f"correlation with true quality: lambda_1se {np.corrcoef(gamma_df['gamma'], true_quality[gamma_df['provider_id']])[0, 1]:.3f}, "
      f"path point 40 {np.corrcoef(model.gamma_path_[40], true_quality[labels])[0, 1]:.3f}, "
      f"lambda_max {np.corrcoef(model.gamma_path_[0], true_quality[labels])[0, 1]:.3f}")
```
```
 provider_id     gamma
          37 -1.139789
           6 -1.012925
          10 -0.896544
 provider_id    gamma
          12 0.822833
          27 0.537943
          34 0.495046
correlation with true quality: lambda_1se 0.200, path point 40 0.187, lambda_max 0.217
```

Because this cohort's facility quality is a known, seeded quantity
(not something you'd have in real data, but exactly what makes a
teaching cohort useful), the estimated $\hat\gamma$ can be checked
directly against the truth. At `lambda_1se_` the correlation between
estimated and true facility quality is `0.200`. Checking whether
covariate adjustment actually improves that — comparing
`gamma_path_[40]` (all ten covariates active, Section 5) against
`gamma_path_[0]` (`lambda_max_`, where $\beta$ is exactly zero and
$\gamma$ alone carries the entire fit) — gives `0.187` and `0.217`
respectively: no real improvement from adjustment here, if anything
slightly worse. That's a genuinely honest
result worth sitting with rather than explaining away: this cohort
assigns patients to facilities independently of their covariates (no
built-in case-mix confounding for adjustment to correct), so there is
no reason *within this particular synthetic cohort* to expect risk
adjustment to sharpen the facility-quality signal — the two
differences (`0.187`, `0.200`, `0.217`) are well within what 60 facilities'
worth of correlation noise could produce either way. The place risk
adjustment *would* visibly matter is a cohort where facility
assignment correlates with patient risk — sicker patients
disproportionately at certain facilities — which is precisely the
scenario [Chapter 4](../survival/04_indirect_standardization_smr_shr)
motivates the whole exercise with, and precisely what this synthetic
cohort, by construction, does not have.

```{note}
Section 4.10 of the survival guide's SMR chapter describes, in general
terms, "a more recent line of methodology" that reformulates provider
profiling as one alternating model instead of a two-stage pipeline —
and states plainly that `coxph` "does not currently provide a
ready-made estimator for this one-stage \[...\] model." This chapter
and the [next one](../survival/12_provider_penalized_cox) are that
estimator, for the logistic and Cox cases respectively — not identical
to the specific fixed-point algorithm Chapter 4 sketches (this class
adds penalization on $\beta$, which that sketch doesn't, and bounds
$\gamma$ by median-clamping rather than a population-SMR-sum
constraint), but the same lineage: provider effects and covariate
effects, estimated jointly, in one alternating fit.
```

## 7. What's next

The [provider-penalized Cox chapter](../survival/12_provider_penalized_cox)
covers `ProviderPenalizedCoxPH`, the same model for time-to-event
outcomes, and the fulfillment of the gap Section 6's note describes.
It has no cross-validated wrapper; that chapter builds the fold loop
by hand.
