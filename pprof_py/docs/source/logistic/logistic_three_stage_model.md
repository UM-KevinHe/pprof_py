(logistic_three_stage_model)=
(logistic_mixed_effect_model)=
# The Three-Stage Logistic Model: Facility Effects Adjusted for Hospitals

A dialysis facility's readmission ratio should not blame the facility for the hospitals its patients
are discharged from. He et al. (2013), *"Evaluating Hospital Readmission Rates in Dialysis Facilities;
Adjusting for Hospital Effects,"* adjust for discharging hospitals with a model that has a fixed effect
per facility and a random effect per hospital, fitted in three stages. `LogisticThreeStageModel` runs
that pipeline from raw discharges, as R's `glmm.fac.hosp` does. `LogisticFERandomClusterModel` is its
third stage, and can be used on its own when the first two stages are fitted separately.

## 1. The model and its three stages

$$
\text{logit}\,P(Y_{ij} = 1) = \gamma_{p(ij)} + \alpha_{h(ij)} + X_{ij}^\top\beta,
\qquad \alpha_h \sim N(0, \sigma^2)
$$

$\gamma_p$ is a **fixed** effect per provider (the facility being profiled), $\alpha_h$ a **random**
effect per hospital (a confounder to adjust for, not itself of interest), and $\beta$ the covariate
effects. Estimating all three jointly becomes numerically unreliable when the facility–hospital
structure is sparse (most facilities discharge to one or two hospitals, and most facility–hospital
cells are empty), so the model is fitted in stages, each taking the previous stages' output as fixed:

| Stage | Model | Estimates | Class |
|---|---|---|---|
| 1 | a fixed effect per facility × hospital cell, on the cells with more than `cutoff` discharges and the raw outcome | $\beta$ | [`LogisticFixedEffectModel`](logistic_fixed_effect_model.md) |
| 2 | crossed random intercepts for facilities and hospitals, with the offset $X\beta$, on the adjusted outcome | $\sigma$, the facilities' BLUPs, the intercept | [`LogisticRandomEffectModel`](logistic_random_effect_model_stats) |
| 3 | fixed facility effects and random hospital effects, with $\beta$ and $\sigma$ held fixed | $\gamma$ | `LogisticFERandomClusterModel` |

Stage 1's cell effects absorb both facility and hospital differences, so $\beta$ is confounded by
neither. Stage 2 separates the hospital variance from facility variation. Stage 3 estimates each
facility's effect with the hospital effects integrated out, starting from Stage 2's facility effects
(the BLUP plus the intercept).

## 2. The running example: facilities that discharge to several hospitals

The adjustment matters when facilities share hospitals. This cohort has 120 facilities discharging to
20 hospitals; each facility uses one to four of them:

```python
import numpy as np
import pandas as pd

def make_crossed_cohort(n_facilities=120, n_hospitals=20, seed=2026):
    rng = np.random.default_rng(seed)
    facility_quality = rng.normal(0, 0.3, n_facilities)     # true fixed facility effects
    hospital_effect = rng.normal(0, 0.3, n_hospitals)       # true random hospital effects
    rows = []
    for f in range(n_facilities):
        k = rng.integers(1, 5)                              # hospitals this facility discharges to
        hospitals = rng.choice(n_hospitals, k, replace=False)
        n = rng.integers(5, 250)
        rows.append(pd.DataFrame({"facility_id": f + 1,
                                  "hospital_id": rng.choice(hospitals, n, p=rng.dirichlet(np.ones(k))) + 1}))
    df = pd.concat(rows, ignore_index=True)
    n = len(df)
    df["age"] = rng.normal(64, 13, n).clip(18, 96).round(1)
    df["diabetes"] = rng.binomial(1, 0.42, n)
    df["chf"] = rng.binomial(1, 0.28, n)
    df["comorbidity_count"] = rng.poisson(1.4, n)
    log_odds = (-1.6 + 0.02 * (df["age"] - 64) + 0.30 * df["diabetes"] + 0.45 * df["chf"]
                + 0.15 * df["comorbidity_count"]
                + facility_quality[df["facility_id"] - 1] + hospital_effect[df["hospital_id"] - 1])
    df["readmit"] = rng.binomial(1, 1 / (1 + np.exp(-log_odds)))
    return df

df = make_crossed_cohort()
covariates = ["age", "diabetes", "chf", "comorbidity_count"]
```

15,365 discharges, 25.6% readmitted; 34 facilities use one hospital, 30 two, 32 three and 24 four.

## 3. Preparing the data

```python
from pprof_py.data import glmm_data_prep

prep = glmm_data_prep(df, y_var="readmit", provider_var="facility_id", cluster_var="hospital_id", cutoff=10)
d = prep.data
print(d[["facility_id", "hospital_id", "readmit", "y_adj", "provider_size", "cell_id", "included"]].head(4))
```
```
  facility_id hospital_id  readmit  y_adj  provider_size  cell_id  included
0           2           1        1    1.0            214        1         1
1           2           1        0    0.0            214        1         1
2           2           1        0    0.0            214        1         1
3           2           1        0    0.0            214        1         1
```

`glmm_data_prep` reproduces R's `glmm.data.prep`. It keeps facilities with more than `cutoff`
discharges (115 of the 120 here); sets `y_adj`, the outcome raised by `0.01 / provider_size` for a
facility with no events and lowered by that amount for one with all events, so that every facility's
effect is finite (no facility needs it here); sorts by hospital and then facility; and numbers the
facility × hospital cells (`cell_id`), marking those with more than `cutoff` discharges (`included`).
Of the 277 cells, 216 are included, holding 97.7% of the discharges; Stage 1 fits those.
{ref}`The data reference <ll_ref_data>` lists the matching R names.

## 4. The pipeline in one call

```python
from pprof_py import LogisticThreeStageModel

model = LogisticThreeStageModel(cutoff=10).fit(df, y_var="readmit", x_vars=covariates,
                                               provider_var="facility_id", cluster_var="hospital_id")
stage3 = model.stage3_
print({k: round(float(v), 4) for k, v in zip(covariates, stage3.beta_)})
print(round(stage3.sigma_, 4), stage3.converged_, stage3.iterations_)
```
```
{'age': 0.0207, 'diabetes': 0.3742, 'chf': 0.4402, 'comorbidity_count': 0.1672}
0.4244 True 25
```

The fitted model keeps every stage: `prep_` (the prepared data), `data_` (with the Stage 1 offset
column `stage1_offset`), and `stage1_`, `stage2_` and `stage3_`. Its `test()`,
`calculate_standardized_measures()`, `calculate_confidence_intervals()` and `summary()` are Stage 3's.
The hospital SD, 0.42, is estimated from 20 hospitals (their true SD is 0.3).

## 5. The stages by hand

The class runs these steps, which can also be taken one at a time:

```python
from pprof_py import LogisticFixedEffectModel, LogisticRandomEffectModel, LogisticFERandomClusterModel

stage1 = LogisticFixedEffectModel(use_dataprep=False, screen_providers=False)
stage1.fit(X=d[d["included"] == 1], y_var="readmit", x_vars=covariates, provider_var="cell_id")
beta = stage1.coefficients_["beta"].ravel()

d2 = d.assign(stage1_offset=d[covariates].to_numpy() @ beta)
stage2 = LogisticRandomEffectModel(verbose=False)
stage2.fit(d2, y_var="y_adj", x_vars=None, provider_var="facility_id", cluster_vars=["hospital_id"],
           offset_var="stage1_offset", verbose=False)

stage3_by_hand = LogisticFERandomClusterModel()
stage3_by_hand.fit(d2, "y_adj", covariates, "facility_id", "hospital_id", stage1=stage1, stage2=stage2,
                   obs_var="readmit", verbose=False)
print(np.array_equal(stage3_by_hand.gamma_, model.stage3_.gamma_))
```
```
True
```

Given the fitted stages, Stage 3's `fit()` takes $\beta$ from Stage 1 by covariate name, $\sigma$ from
Stage 2 by name (`stage2.sigma_["hospital_id"]`; the facility SD is `sigma_["facility_id"]`), and
starts from Stage 2's facility effects plus its intercept, matched to its own facilities by ID. Taking
$\sigma$ by name matters: R's `glmm.fac.hosp` takes the second variance component, which is the
facility SD when hospitals outnumber facilities. Matching by ID matters too, because text IDs sort
`'1', '10', '100', …` in one model and possibly numerically in another. Values computed elsewhere
(for example in R) can be passed instead, as `beta=`, `sigma=` and `gamma_init=`; `obs_var` names the
true 0/1 outcome, which the tests and observed counts use.

## 6. Provider tests, measures and intervals

```python
result = model.test()
print(result[["estimate", "null_value", "z_raw", "p_value", "flag", "ci_lower", "ci_upper"]].head(3).round(4))
```
```
             estimate  null_value   z_raw  p_value  flag  ci_lower  ci_upper
provider_id
1             -4.2868     -3.0599 -1.7461   0.0808     0   -6.1618   -2.9362
2             -3.2135     -3.0599 -0.9725   0.3308     0   -3.5283   -2.9086
3             -3.6813     -3.0599 -2.4522   0.0142    -1   -4.2259   -3.1780
```

Every test compares a facility's event count with its distribution when its effect is the reference
$\gamma_0$ (`reference="median"`) and the hospital effects follow their posterior. They differ in how
the hospital effects enter:

- `"exact"` (the default): each hospital's effect is drawn once and shared by the facility's patients
  there (He et al. 2013, Section 3.3, step (ii)), and the count's distribution is computed exactly.
- `"poibin_exact"`: an exact Poisson-binomial test with the hospital effects at their posterior means.
- `"resampling"`: Monte Carlo with a separate hospital-effect draw for every patient, as in R's
  `summary.glmm.fac`. Facilities whose simulated tail reaches the resolution floor (`0.5 / n_resample`)
  get the exact tails of the same null, with a warning.

`flag` is 1 for a facility above the reference (for readmissions, worse) and -1 below, with the columns
described in {ref}`ll_ref_measures`. For `"exact"` and `"poibin_exact"`, `ci_lower` and `ci_upper`
invert the test, so a limit excludes $\gamma_0$ exactly when the facility is flagged; a facility with no
events has the lower limit $-\infty$. `calculate_confidence_intervals()` returns the same limits
(`option="gamma"`) or maps them to standardized ratios and rates (`option="SM"`).

The default null is the theoretical N(0, 1). R's `summary.glmm.fac` calibrates against an empirical
null fitted with `MASS::rlm` defaults within quartiles of facility size; the prepared data carry that
size:

```python
from pprof_py.inference import EmpiricalNull, HUBER_RLM

size = d.groupby("facility_id", observed=True)["provider_size"].first().reindex(stage3.provider_ids_)
result_en = model.test(null_model=EmpiricalNull.fitter(size=size.to_numpy(float), n_groups=4,
                                                       grouping="quantile", estimator=HUBER_RLM))
```

| test | flagged, theoretical null (above the reference) | flagged, empirical null |
|---|---|---|
| `"exact"` | 28 (17) | 9 |
| `"poibin_exact"` | 29 (18) | 8 |
| `"resampling"` (`seed=1`) | 29 (18) | 9 |

[The empirical null guide](../reference/empirical_null) covers the options. Standardized readmission
ratios come from `calculate_standardized_measures()`:

```python
print(model.calculate_standardized_measures(stdz="indirect")["indirect"].head(3).round(4))
```
```
   provider_id  indirect_ratio  indirect_rate  observed  expected
0            1          0.3805         9.7219       2.0    5.2558
1            2          0.9002        22.9979      58.0   64.4313
2            3          0.6308        16.1165      18.0   28.5338
```

## 7. Two estimators

Stage 3 has two estimators of $\gamma$:

- `estimator="he2013"` (the default) is the iteration of He et al. (2013), as in R's `glmm.fac.hosp`.
  Its fixed point depends on the starting value and is not the maximum likelihood estimate.
- `estimator="marginal"` maximizes the marginal likelihood in $\gamma$ with $\beta$ and $\sigma$ fixed.
  That likelihood is concave in $\gamma$, so the estimate is unique and does not depend on the start.
  It is computed by adaptive Gauss–Hermite quadrature (each hospital's nodes centered and scaled at its
  posterior) and a projected Newton iteration.

```python
marginal = LogisticThreeStageModel(cutoff=10, estimator="marginal").fit(
    df, "readmit", covariates, "facility_id", "hospital_id")
print(round(model.stage3_.loglik_, 4), round(marginal.stage3_.loglik_, 4))
```
```
-8119.6908 -8117.9024
```

`loglik_` is the marginal log-likelihood under either estimator. Here the marginal estimate is 1.79
higher, in 4 Newton steps; the two estimates of $\gamma$ differ by up to 0.22 (median 0.079), and three
facilities' `"exact"`-test flags change (28 flagged against 27). Starting the marginal estimator from
$\gamma = 0$ instead reaches the same estimate.

Most of that difference comes from `"he2013"`'s fixed quadrature nodes, not from its iteration. Its
`n_nodes` nodes (20 by default) are fixed, and a hospital with hundreds of discharges has a posterior
narrower than their spacing. This cohort's hospitals have a median of 816 discharges: with 60 fixed
nodes the two estimates differ by at most 0.079, and with 100 by at most 0.004. For output comparable
with R, keep `"he2013"` with R's settings (Section 10); otherwise `"marginal"` is the accurate choice,
and costs about as much.

## 8. Checks before trusting Stage 3

- **Convergence.** Confirm `converged_` is `True` and `iterations_` is well below `max_iter`.
  `convergence_criterion="max_delta_gamma"` (the default) stops when no $\gamma$ moves by more than `tol`;
  `"relative"`, R's rule, stops on the change in the objective relative to its change since the first
  iteration, which can stop short of the solution on some data sets. Under `"marginal"`, `tol` bounds the
  largest score.
- **Facilities at the bound.** $\gamma$ is clipped to `median(gamma_) ± bound` (`bound_mode="relative"`,
  the default) or `±bound` (`"absolute"`, as in R). A facility there has an outcome that the adjustment
  in `y_adj` did not make finite (none does here).
- **Quadrature under `"he2013"`.** With large hospitals, check the estimate against `"marginal"` or a
  larger `n_nodes` (Section 7).
- **$\sigma$ is Stage 2's.** Earlier versions offered `update_sigma=True`, which re-estimated $\sigma$ from
  the posterior hospital effects each iteration and drove it toward zero (R's `glmm.fac.hosp` does the
  same); it was removed.

## 9. Covariate inference

`summary()` reports inference for $\beta$ from Stage 1: $\beta$ is estimated there and held fixed here,
so its standard errors are Stage 1's, whose Wald variance accounts for the estimated cell effects (R's
`summary.glmm.covar` also reports the Stage 1 fit).

```python
print(model.summary().round(4))
```
```
                   estimate  std_error     stat  p_value  ci_lower  ci_upper
age                  0.0207     0.0016  13.2998      0.0    0.0176    0.0237
diabetes             0.3742     0.0398   9.3963      0.0    0.2961    0.4522
chf                  0.4402     0.0426  10.3390      0.0    0.3567    0.5236
comorbidity_count    0.1672     0.0165  10.1643      0.0    0.1350    0.1995
```

`LogisticFERandomClusterModel.summary()` needs the Stage 1 model: `fit(stage1=..., stage2=...)` stores
it, or pass `summary(stage1=...)`. It raises if that model's $\beta$ is not the one the fit used.

## 10. Matching R's `glmm.fac.hosp`

For output comparable with R, use `LogisticThreeStageModel(bound_mode="absolute", convergence_criterion="relative")`
with the other defaults (`n_nodes=20`, `tol=1e-5`, `estimator="he2013"`), and for flags
comparable with R's `summary.glmm.fac`, `test_method="resampling"` with the empirical null of Section 6.
On a crossed synthetic cohort (40 facilities, 12 hospitals), given the same $\beta$, Stage 2 matches R's
`glmer` ($\sigma$ to $7\times10^{-6}$, the starting $\gamma$ to $10^{-5}$) and Stage 3 matches
`glmm.fac.hosp` ($\gamma$ to $2\times10^{-5}$, the SRRs to $10^{-5}$); R's Stage 1 (`pprof::logis_fe`)
could not be run.

pprof_py departs from R deliberately in a few places: `σ` is taken by name (see Section 5); in
`"resampling"`, the hospital-effect draws have SD $\sqrt{\nu}$ (R passes the variance $\nu$ as the SD), the
sign of z comes from the tails, the random-number streams are independent, and the stopping objective is
evaluated after the $\gamma$ update. Comparisons with R should use exact references for these. The flag
convention is `+1` above the reference (worse, for readmissions); older pprof_py versions, and possibly an
R pipeline, used the opposite.

## 11. When this approach is the right tool

Reach for the three-stage model, rather than [`LogisticFixedEffectModel`](logistic_fixed_effect_model.md)
or [`LogisticRandomEffectModel`](logistic_random_effect_model_stats) alone, when providers have a
genuine two-level structure: you need unbiased, reportable effects for each provider (which rules out a
pure random-intercept model, which shrinks individual estimates toward the mean by design), but the
providers' patients pass through coarser units whose own variation you want to account for rather than
ignore or estimate as more fixed effects. Discharging hospitals for dialysis facilities (He et al.'s
example) are the natural case; clinics within regions or facilities within chains are others. Without a
meaningful second level, [`LogisticFixedEffectModel`](logistic_fixed_effect_model.md) alone is simpler and
sufficient; without the need for unbiased individual effects, a random intercept via
[`LogisticRandomEffectModel`](logistic_random_effect_model_stats) is simpler still.
