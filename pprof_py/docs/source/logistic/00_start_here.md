(logistic_start)=
# Logistic models in `pprof_py`: start here

*Binary and binomial outcomes, from fixed effects to the three-stage model*

---

## What this family is for

The logistic family profiles providers on a **binary outcome** — death,
readmission, a complication, a fistula in use, a patient on the
transplant waitlist — or on a **binomial** one, where each record
carries a number of trials and a count of events. Each model adjusts
for patient covariates and estimates a provider effect $\gamma$ on the
logit scale; from it come the indirect standardized ratio (observed
over expected events) and the direct standardized rate, with Wald,
score, exact Poisson-binomial and bootstrap tests and the matching
confidence intervals.

## Which measures it serves

| Outcome | Example measures | Model |
|---|---|---|
| One 0/1 outcome per patient (PPPW- and SFR-style measures: is the patient waitlisted, is a fistula in use) | Indirect standardized ratio, direct standardized rate | `LogisticFixedEffectModel`; `LogisticRandomEffectModel` for shrunken estimates |
| A count of events out of a number of trials per record (AOH-style measures) | The same, with `n_var` naming the trials column | `LogisticFixedEffectModel` with `n_var` |
| Readmission after a hospital discharge, adjusted for the discharging hospital (SRR) | Standardized readmission ratio from fixed facility effects with random hospital effects | `LogisticThreeStageModel`; its last stage is `LogisticFERandomClusterModel` |
| Any of the above with many candidate risk adjusters | A risk model chosen by elastic net or group lasso, or jointly with the provider effects | `PenalizedLogisticCV`, `GroupLassoLogisticCV`, `ProviderPenalizedLogisticCV` |

Continuous outcomes belong to the [linear family](../linear/00_start_here.md);
time-to-event outcomes (SMR, SHR) to the
[survival family](../survival/00_start_here.md).

## The classes

All ten are importable from the package root.

| Class | What it does |
|---|---|
| `LogisticFixedEffectModel` | One fixed effect per provider, fitted with a SerBIN-style block algorithm so that thousands of providers are feasible; the default profiling model, with the exact Poisson-binomial test. |
| `LogisticRandomEffectModel` | Random provider intercepts (a logit GLMM) fitted like `lme4::glmer`, by penalized IRLS with a Laplace approximation; shrinks small providers toward the mean. |
| `LogisticFERandomClusterModel` | Fixed provider effects plus a random effect for a second, crossed grouping (the discharging hospital), with $\beta$ and $\sigma$ held fixed. Stage 3 of the three-stage model; usable on its own. |
| `LogisticThreeStageModel` | Runs the three stages of He et al. (2013) from raw discharges, as R's `glmm.fac.hosp` does, and returns facility effects adjusted for hospital effects. |
| `PenalizedLogistic` / `PenalizedLogisticCV` | Elastic net over a lambda path, as `glmnet(family="binomial")` and `cv.glmnet`. |
| `GroupLassoLogistic` / `GroupLassoLogisticCV` | Group lasso and sparse group lasso over blocks of coefficients, with cross-validation. |
| `ProviderPenalizedLogistic` / `ProviderPenalizedLogisticCV` | Unpenalized provider effects (bounded by `provider_bound`) and penalized covariate effects in one fit: covariate selection and profiling together. |

## Which class do I need?

| Situation | Use |
|---|---|
| Profile providers on a 0/1 or binomial outcome; providers may differ systematically in case mix | `LogisticFixedEffectModel` |
| Many small providers, and you want stable (shrunken) estimates or inference about the population of providers | `LogisticRandomEffectModel` |
| The outcome is confounded by a second grouping, such as readmissions and the discharging hospital | `LogisticThreeStageModel` |
| Stages 1 and 2 already fitted; only the last stage is needed | `LogisticFERandomClusterModel` |
| Many candidate risk adjusters, few of them known to matter | `PenalizedLogisticCV` |
| Risk adjusters that come in blocks (categorical dummies, comorbidity panels) | `GroupLassoLogisticCV` |
| Select the risk adjusters and estimate the provider effects in one model | `ProviderPenalizedLogisticCV` |
| Not sure whether fixed or random effects fit the question | [Fixed vs Random Effects](../fixed_vs_random_effects.md) |

## Recommended reading order

1. [Logistic fixed-effect modeling](logistic_fixed_effect_model.md) —
   the model, its estimation, the standardized measures, the four tests
   and their intervals. The methods reference for the family.
2. The [provider profiling with logistic fixed effects](logistic_fe_tutorial.md) tutorial
   — the same workflow run end to end on a synthetic cohort.
3. [Logistic random-effect modeling](logistic_random_effect_model.md)
   and its [tutorial](logistic_re_tutorial.md) — the GLMM, BLUPs and
   shrinkage.
4. [The three-stage logistic model](logistic_three_stage_model.md) —
   facility effects adjusted for hospitals; also the chapter for
   `LogisticFERandomClusterModel`.
5. [Penalized logistic regression](penalized_logistic.md) — the lambda
   path and cross-validation, which the next two chapters and the
   penalized linear chapter assume.
6. [Group lasso logistic regression](group_lasso_logistic.md).
7. [Provider-penalized logistic regression](provider_penalized_logistic.md).

The concepts shared with the other families are in
[Direct vs Indirect Standardization](../direct_vs_indirect_standardization.md),
[Inference for Provider Effects](../inference_provider_effects_theory.md)
and [Empirical Null Calibration](../empirical_null_theory.md). The
argument and attribute lists are in {ref}`ll_ref_logistic` and
{ref}`ll_ref_penalized`; the measure, test, interval and plot methods in
{ref}`ll_ref_measures`.

## Importing

```python
from pprof_py import LogisticFixedEffectModel, LogisticRandomEffectModel
from pprof_py import LogisticFERandomClusterModel, LogisticThreeStageModel
from pprof_py import PenalizedLogistic, PenalizedLogisticCV
from pprof_py import GroupLassoLogistic, GroupLassoLogisticCV
from pprof_py import ProviderPenalizedLogistic, ProviderPenalizedLogisticCV
```

Every model is configured in its constructor and fitted with `fit()`;
fitted attributes end in an underscore (`coefficients_`, `variances_`,
`fitted_`), as described in [Model Architecture](../model_architecture.md).
