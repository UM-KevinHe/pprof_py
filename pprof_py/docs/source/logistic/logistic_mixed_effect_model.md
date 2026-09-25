(logistic_mixed_effect_model)=
# Logistic Mixed-Effect Regression: Stage 3 of the Staged SRR Approach

`LogisticMixedEffectModel` is not a self-sufficient, hand-it-raw-data
model the way every other class in this documentation is. It is
**Stage 3** of a three-stage estimation approach for facility-level
Standardized Readmission Ratios (SRRs) adjusted for discharging
hospitals, originating in He et al. (2013), *"Evaluating Hospital
Readmission Rates in Dialysis Facilities; Adjusting for Hospital
Effects,"* and refined since into the cleaner three-stage form this
chapter documents. Reading this class in isolation — as this
documentation originally did — makes its central design choice look
like a bug: covariate effects ($\beta$) go in and come back out
completely unchanged, no matter how long `fit()` runs. That's not an
oversight. It's Stage 3 correctly doing the one job that belongs to
it, and no more.

## 1. The three stages

$$
\text{logit}\,P(Y_{ij} = 1) = \gamma_{p(ij)} + \alpha_{h(ij)} + X_{ij}^\top\beta,
\qquad \alpha_h \sim N(0, \sigma^2)
$$

This is He et al.'s own "Model 2: mixed effects model" — $\gamma_p$ a
**fixed** effect per provider (the entity being profiled), $\alpha_h$
a **random** effect per hospital (a confounder to adjust for, not
itself of interest), $\beta$ fixed covariate effects. Jointly
estimating all three from scratch is exactly what the 2013 paper's
own experiments found becomes numerically unreliable once the
provider/hospital structure gets sparse — most patients at a facility
discharged from only one or two hospitals, most facility–hospital
cells empty. The paper's original fix staged the fitting: pre-estimate
the hospital variance from an auxiliary model, then fit the main model
with that variance held fixed. The refinement this package implements
carries that idea one step further, splitting the work into three
purpose-built stages instead of two, each handled by its own tested
class:

| Stage | Estimates | Tool |
|---|---|---|
| 1 — fixed-effects model | $\beta$ | [`LogisticFixedEffectModel`](logistic_fixed_effect_model) |
| 2 — random-effects model | hospital variance $\sigma^2$ | [`LogisticRandomEffectModel`](logistic_random_effect_model_stats) |
| 3 — mixed-effects model | provider effects $\gamma$ | `LogisticMixedEffectModel` (this class) |

Each stage takes the previous stages' output as a fixed input, not
something it re-derives. By the time Stage 3 runs, $\beta$ and
$\sigma$ are already decided — its Newton-Raphson loop has exactly one
job, estimating $\gamma$, while integrating $\alpha_h$ out of the
likelihood via Gauss-Hermite quadrature using the $\sigma$ Stage 2
already found. This is a deliberately narrower, more modular
decomposition than fitting everything jointly: each stage reuses an
existing, independently-tested estimator (Stage 1 is exactly
[`LogisticFixedEffectModel`](logistic_fixed_effect_model), which the
paper itself argues gives less biased, lower-MSE effect estimates than
any random-effects alternative — "the FEM yields estimates of extreme
values of facility effects that are less biased and have smaller mean
squared error than the REM"), and Stage 3's own solve is smaller and
more scalable for it, needing only one Newton-Raphson block instead of
two.

## 2. The running example: facilities nested in hospitals

The same nested cohort as before — 80 facilities, 10 per cluster
(standing in for "hospital" here), 8 clusters, each facility with its
own true quality effect and each cluster its own true random effect on
top:

```python
import numpy as np
import pandas as pd

def make_nested_mortality_cohort(n_patients=8000, n_facilities=80, n_clusters=8, seed=0):
    rng = np.random.default_rng(seed)
    facility_id = rng.integers(0, n_facilities, n_patients)
    cluster_of_facility = np.arange(n_facilities) // (n_facilities // n_clusters)
    facility_quality = rng.normal(0, 0.3, n_facilities)      # true fixed provider effect
    cluster_effect_true = rng.normal(0, 0.25, n_clusters)     # true random hospital effect

    age = rng.normal(64, 13, n_patients).clip(18, 96)
    diabetes = rng.binomial(1, 0.42, n_patients)
    chf = rng.binomial(1, 0.28, n_patients)
    comorbidity_count = rng.poisson(1.4, n_patients)

    log_odds = (
        -3.6 + 0.030 * (age - 64) + 0.35 * diabetes + 0.55 * chf
        + 0.20 * comorbidity_count
        + facility_quality[facility_id]
        + cluster_effect_true[cluster_of_facility[facility_id]]
    )
    death_30d = rng.binomial(1, 1.0 / (1.0 + np.exp(-log_odds)))
    return pd.DataFrame({
        "patient_id": np.arange(n_patients), "facility_id": facility_id,
        "cluster_id": cluster_of_facility[facility_id],
        "age": age.round(1), "diabetes": diabetes, "chf": chf,
        "comorbidity_count": comorbidity_count, "death_30d": death_30d,
    })

df = make_nested_mortality_cohort()
candidates = ["age", "diabetes", "chf", "comorbidity_count"]
```

8,000 patients, 5.8% mortality, exactly 10 facilities per cluster.

## 3. Stage 1: `LogisticFixedEffectModel` → $\beta$

```python
from pprof_py import LogisticFixedEffectModel

fe = LogisticFixedEffectModel(algorithm="Serbin", screen_providers=False)
fe.fit(X=df, y_var="death_30d", x_vars=candidates, group_var="facility_id",
       max_iter=200, tol=1e-6)
beta_stage1 = fe.coefficients_["beta"]
# age: 0.0348, diabetes: 0.4053, chf: 0.3633, comorbidity_count: 0.2304
```

Nothing specific to the mixed-effect workflow here — this is an
ordinary [Stage-1-style fixed-effects fit](logistic_fixed_effect_model),
with `facility_id` as the grouping variable, exactly as it would be
used standalone. $\beta$ from this fit is what Stage 3 will treat as
given.

## 4. Stage 2: `LogisticRandomEffectModel` → hospital variance $\sigma^2$

```python
from pprof_py import LogisticRandomEffectModel

df["xbeta_offset"] = df[candidates].values @ beta_stage1
df["cluster_id"] = df["cluster_id"].astype("category")

re = LogisticRandomEffectModel()
re.fit(df, y_var="death_30d", group_var="cluster_id",
       offset_var="xbeta_offset", x_vars=None)
sigma_stage2 = re.sigma_["cluster_id"]
# 0.1491  (true cluster-effect sd used to generate this cohort: 0.187)
```

`offset_var` is what makes `LogisticRandomEffectModel` a clean fit for
this role: passing Stage 1's $X\beta$ in as a fixed offset (rather
than `x_vars=candidates`, which would re-estimate $\beta$ a second
time inside Stage 2) means this fit's only remaining job is the
random-intercept variance for `cluster_id` — precisely Stage 2's
scope, no more. `sigma_` is a dict keyed by grouping variable, since
this class supports more than one random-intercept term at once; with
a single `group_var` here, `sigma_["cluster_id"]` is the one value
Stage 3 needs.

## 5. Stage 3: `LogisticMixedEffectModel` → provider effects $\gamma$

```python
from pprof_py import LogisticMixedEffectModel

df["facility_id"] = df["facility_id"].astype("category")
n_providers = df["facility_id"].nunique()

model = LogisticMixedEffectModel(n_nodes=15, max_iter=200, tol=1e-6)
model.fit(
    df, y_var="death_30d", x_vars=candidates,
    provider_var="facility_id", cluster_var="cluster_id",
    gamma_init=np.zeros(n_providers),
    beta_init=beta_stage1,        # from Stage 1 -- held fixed throughout fit()
    sigma_init=sigma_stage2,      # from Stage 2 -- held fixed
    stage1_model=fe,              # the Stage 1 model, for summary()
    verbose=False,
)
model.converged_     # True
model.iterations_    # well below max_iter
```

$\beta$ and $\sigma$ enter through `beta_init`/`sigma_init` — names
that describe their role in *this stage's* iteration (they seed the
Newton-Raphson/GH-quadrature loop) more than where they come from, so
it's easy to read them as values this call will go on to refine, the
way `gamma_init` clearly is. It won't: `beta`/`xbeta` are set once
before the loop and never reassigned inside it — only `gamma` (Newton-
Raphson) and the cluster posterior moments (GH quadrature) actually
update each iteration. That is Stage 3 correctly treating Stage 1 and
2's output as fixed, not a gap in the loop. Skipping Stages 1–2 and
passing arbitrary `beta_init`/`sigma_init` values will still run
without error — `fit()` has no way to know they weren't properly
estimated — and will fit `gamma_` around whatever nonsense it was
given.

The class docstring documents this intentional design: `beta_init`
and `sigma_init` are held fixed throughout `fit()`, consistent with
the three-stage decomposition of He et al. (2013).

## 6. Reading the fitted effects

```python
model.alpha_mean_cluster_    # posterior mean cluster (hospital) effect, shape (8,)
# array([0.0001, 0.0, 0.0001, -0.0024, -0.0020, 0.0, 0.0011, 0.0001])
```

These are **posterior means**, not point estimates with their own
standard errors the way `gamma_` has — the random effect's job is to
be shrunk toward zero relative to what a fixed effect would show,
borrowing strength across the (here, only 8) clusters rather than
trusting each one's own limited data alone. Correlation with this
cohort's true, seeded cluster effects is `0.26` in this run — modest,
but a real improvement over feeding Stage 3 an arbitrary guessed
$\sigma$ instead of Stage 2's actual estimate, and an honest reminder
that 8 is very few units to estimate a variance component and its
posterior means from.

```python
model.gamma_    # fixed provider effects, shape (80,)
```
Correlation with the cohort's true, seeded facility quality: `0.40` —
in the same range as the
[provider-penalized logistic chapter's](../logistic/provider_penalized_logistic)
own recovery numbers on comparably-sized synthetic data, for the same
reason: dozens of patients per facility says something, not enough to
say it with great precision. At least one facility's `gamma_` sits
exactly at the lower bound, `median(gamma_) - bound` with the default
`bound_mode="relative"` (`-bound` with `bound_mode="absolute"`, as in
R's `glmm.fac.hosp`) — an ordinary, expected occurrence for a small or
extreme facility, the same median-clamp mechanism
[the logistic provider chapter's](../logistic/provider_penalized_logistic)
Section 3 describes, not a sign anything is wrong.

`summary()` reports inference for $\beta$ from Stage 1: $\beta$ is
estimated there and held fixed here, so its standard errors are Stage 1's,
whose Wald variance accounts for the estimated provider effects. It returns
the Stage 1 model's Wald table, the same as `fe.summary(test_method="wald")`
from Section 3, and needs that model, given to `fit(stage1_model=...)` or to
`summary(stage1_model=...)`; it raises if the model's $\beta$ is not the one
passed as `beta_init`. (R's `summary.glmm.covar` also reports the Stage 1 fit.
Earlier versions computed an information matrix from the Stage 3 fit at fixed
$\beta$ and $\gamma$, which understated the standard errors.)

```python
model.summary()    # the Stage 1 Wald table: estimate, std_error, stat, p_value, ci_lower, ci_upper
```

## 7. Two things worth checking before trusting Stage 3's output

**`n_nodes` (quadrature points for the hospital-effect integral)
matters more than its default suggests — check convergence, not just
the final numbers.** At `n_nodes=5` on this cohort, `fit()` never
converges: `iterations_` runs out the full `max_iter` (201, one past
the 200 cap) rather than stopping early, and `gamma_` collapses to the
lower bound for most facilities rather than taking the spread-out
values a converged fit gives. `n_nodes=10` also exhausts `max_iter` without formally
converging, though its final iterate happens to already resemble the
converged answer. `n_nodes=15` and `n_nodes=20` (the default) both
converge cleanly in well under `max_iter` (14 and 10 iterations here)
and agree with each other closely. The practical check: after fitting,
confirm `model.iterations_` is comfortably below `max_iter` — if it
isn't, the fit didn't converge, and dropping straight to `n_nodes=5`
or similar to save time is a real way to get confidently wrong,
bound-saturated `gamma_` values rather than merely noisier ones.

$\sigma$ stays at Stage 2's value. Earlier versions offered
`update_sigma=True`, which re-estimated $\sigma$ from the posterior cluster
moments each iteration; that update drove $\sigma$ toward zero (R's
`glmm.fac.hosp` does the same), so it was removed. The convergence rule is
also a choice: `convergence_criterion="relative"` (the default, as in R)
stops on the change in the objective relative to its change since the first
iteration, which can stop well short of the solution on some data sets;
`"max_delta_gamma"` stops when no $\gamma$ moves by more than `tol`.

## 8. Standardized measures and provider testing

`calculate_standardized_measures()` and `test()` — mixed into this
class from `MixedEffectMeasuresMixin` — follow the same conventions as
`LogisticFixedEffectModel` and
[`LogisticRandomEffectModel`](logistic_random_effect_model_stats),
with the cluster random effect's posterior mean folded into each
provider's expected count the same way $X\beta$ is:

```python
model.calculate_standardized_measures(stdz="indirect")   # {'indirect': <DataFrame>}
result = model.test(test_method="poibin_exact")
print(result[["estimate", "null_value", "z_raw", "p_value", "flag"]].head(3).round(4))
```
```
          estimate  null_value   z_raw  p_value  flag
provider
0          -5.8522     -5.8377  0.0352   0.9719     0
1          -6.3107     -5.8377 -0.8941   0.3713     0
2          -6.0450     -5.8377 -0.3817   0.7027     0
```

All three test methods compare each provider's event count with its
distribution when its effect is the reference $\gamma_0$
(`reference="median"` by default) and the cluster effects follow their
posterior; they differ in how the cluster effects enter:

- `"exact"` (the default): each hospital's effect is drawn once and shared by
  the facility's patients there (He et al. 2013, Section 3.3, step (ii)), and
  the count's distribution is computed exactly.
- `"poibin_exact"`: exact Poisson-binomial test with the cluster effects fixed
  at their posterior means.
- `"resampling"`: Monte Carlo with a separate cluster-effect draw for every
  patient, as in R's `summary.glmm.fac`. Providers whose simulated tail falls
  to the resolution floor (`0.5 / n_resample`) get the exact tails of the same
  null, with a warning.

The result has the columns described in {ref}`ll_ref_measures`, with `flag` =
1 for providers above the reference and -1 below. For `"exact"` and
`"poibin_exact"`, `ci_lower`/`ci_upper` invert the (calibrated) test, so a
limit excludes $\gamma_0$ exactly when the provider is flagged; a facility
with no events has lower limit $-\infty$. `calculate_confidence_intervals()`
returns the same limits (`option="gamma"`) or maps them to standardized
ratios and rates (`option="SM"`), as for the fixed-effect model.

The default null is the theoretical N(0, 1). R's `summary.glmm.fac`
calibrates against an empirical null fitted with `MASS::rlm` defaults within
quartiles of a facility-size variable (R sets missing sizes to 0); pass that,
or any other null model, through `null_model`:

```python
from pprof_py.inference import EmpiricalNull, HUBER_RLM

facility_size = ...   # one size per provider, in model.provider_ids_ order, no missing values
result_en = model.test(
    null_model=EmpiricalNull.fitter(size=facility_size, n_groups=4, grouping="quantile",
                                    estimator=HUBER_RLM),
)
```

Earlier versions of pprof_py calibrated by default with equal-count groups of
discharge counts (`grouping="rank"`, `size` = each facility's number of
patients, `small_group="theoretical"`), which is not R's configuration.

[The empirical null guide](../reference/empirical_null) covers the
options.

## 9. When this three-stage approach is the right tool

Reach for the full three-stage pipeline — not
[`LogisticFixedEffectModel`](logistic_fixed_effect_model) alone, not
[`LogisticRandomEffectModel`](logistic_random_effect_model_stats)
alone — when your providers have a genuine two-level structure: you
need individually unbiased, publicly reportable effects at the
provider level (ruling out a pure random-intercept model, which
shrinks and therefore biases individual estimates toward the mean by
design), but those providers are naturally grouped into coarser
clusters whose own variation you want to account for rather than
either ignore or treat as more fixed effects you don't have the data
(or the interest) to report individually. Facilities within corporate
dialysis chains, discharging hospitals for dialysis facilities (He et
al.'s own motivating example), or clinics within a state's regions are
all natural fits. If there's no meaningful second level of grouping,
[`LogisticFixedEffectModel`](logistic_fixed_effect_model) alone is
simpler and sufficient. If you don't need individually unbiased
provider effects at all, a plain random intercept via
[`LogisticRandomEffectModel`](logistic_random_effect_model_stats) —
which jointly estimates everything it reports in one self-sufficient
call, unlike Stage 3 of this pipeline — is both simpler and closer to
what "mixed-effects model" suggests outside this specific staged
context.
