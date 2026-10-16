(linear_start)=
# Linear models in `pprof_py`: start here

*Continuous outcomes, from fixed effects to penalized regression*

---

## What this family is for

The linear family profiles providers on a **continuous outcome**: length
of stay, cost of care, a clinical measurement such as blood pressure or
eGFR. Each model adjusts the outcome for patient covariates and estimates
one effect per provider; that effect is then standardized, tested
against a reference and plotted, in the same workflow as the logistic
and survival families, with ordinary least squares and linear
mixed-model arithmetic in place of a likelihood on the logit or hazard
scale.

Because the outcome keeps its natural units, the standardized measures
are **differences** (observed minus expected, in the units of the
outcome) rather than ratios, and the provider tests are t- or z-tests on
those differences. Section 5 of
[Inference for Standardized Measures](../inference_standardized_measures_theory.md)
places the linear case within the general theory.

## Which measures it serves

| Outcome | Example measures | Model |
|---|---|---|
| A quantitative outcome per patient or per record | Risk-adjusted mean length of stay, cost or lab value; indirect and direct standardized differences between a provider and the reference | `LinearFixedEffectModel`, `LinearRandomEffectModel` |
| The same, with many candidate risk adjusters | A risk-adjustment model selected by elastic net or group lasso before profiling | `PenalizedLinear`, `PenalizedLinearCV`, `GroupLassoLinear` |

Binary and binomial outcomes (PPPW- and SFR-style measures, readmission
ratios) belong to the [logistic family](../logistic/00_start_here.md);
time-to-event outcomes (SMR, SHR) to the
[survival family](../survival/00_start_here.md).

## The classes

All five are importable from the package root.

| Class | What it does |
|---|---|
| `LinearFixedEffectModel` | One intercept per provider, estimated by within-provider OLS; robust to provider effects that are correlated with case mix. The default profiling model. |
| `LinearRandomEffectModel` | Provider intercepts as random draws from $N(0, \sigma_u^2)$, fitted by (restricted) maximum likelihood with an `lme4`-style solver; shrinks small providers toward the mean. |
| `PenalizedLinear` | Elastic net (ridge, LASSO or a mix) over a lambda path for a continuous outcome, as R's `glmnet(family="gaussian")`. |
| `PenalizedLinearCV` | `PenalizedLinear` with K-fold cross-validation to choose `lambda_min_` or `lambda_1se_`, as `cv.glmnet`. |
| `GroupLassoLinear` | Group lasso and sparse group lasso: penalizes blocks of coefficients (a one-hot factor, a panel of lab values) together. It has no cross-validated wrapper. |

## Which class do I need?

| Situation | Use |
|---|---|
| Profile providers on a continuous outcome; providers may differ systematically in case mix | `LinearFixedEffectModel` |
| Many small providers, and you want stable (shrunken) estimates or inference about the population of providers | `LinearRandomEffectModel` |
| Many candidate risk adjusters, few of them known to matter | `PenalizedLinearCV`, then refit the chosen covariates with a fixed- or random-effect model |
| Risk adjusters that come in blocks (categorical dummies, panels of related measurements) | `GroupLassoLinear` |
| Not sure whether fixed or random effects fit the question | [Fixed vs Random Effects](../fixed_vs_random_effects.md) |

## Recommended reading order

1. [Linear fixed-effect modeling](linear_fixed_effect_model.md) — the
   model, within-group OLS, standardized differences, t-tests and
   confidence intervals. The methods reference for the family.
2. The [linear fixed-effect model tutorial](linear_fe_tutorial.md) — the
   same workflow run end to end on a synthetic 30-hospital cohort.
3. [Linear random-effect modeling](linear_random_effect_model.md) and
   its [tutorial](linear_re_tutorial.md) — REML, BLUPs and shrinkage.
4. [Penalized linear regression](penalized_linear.md) — assumes the
   [penalized logistic chapter](../logistic/penalized_logistic.md),
   which introduces the lambda path and cross-validation.
5. [Group lasso linear regression](group_lasso_linear.md) — assumes the
   [group lasso logistic chapter](../logistic/group_lasso_logistic.md).

The argument and attribute lists are in {ref}`ll_ref_linear` and
{ref}`ll_ref_penalized`; the measure, test, interval and plot methods
shared by the profiling models are in {ref}`ll_ref_measures`.

## Importing

```python
from pprof_py import LinearFixedEffectModel, LinearRandomEffectModel
from pprof_py import PenalizedLinear, PenalizedLinearCV, GroupLassoLinear
```

Every model is configured in its constructor and fitted with `fit()`;
fitted attributes end in an underscore (`coefficients_`, `variances_`,
`fitted_`), as described in [Model Architecture](../model_architecture.md).
