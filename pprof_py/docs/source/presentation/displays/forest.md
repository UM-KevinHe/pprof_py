# Coefficient forest

```{include} ../_figures/forest.md
```

**Answers:** How large are the covariate effects, with their intervals?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | How large are the covariate effects, with their intervals? |
| Quantity and source | Each term's estimate and interval from the model's `summary()` (`CoefficientProfile`). Odds and hazard ratios come from `CoefficientProfile.exponentiate()`, a named, tested conversion of the estimates and bounds. |
| Uncertainty | The intervals of `summary()` at `level` (95% only for CoxPH, whose summary reports no other). |
| Denominator | Not applicable: the rows are covariates, not providers. |
| Reference | A line at no association: 0, or 1 on a ratio axis. |
| Misreadings and mitigations | The effects are associations adjusted for the other covariates and the provider effects, not causal effects, and covariates measured in different units cannot be compared by the size of their effects; the footnote says both. The intercept is left out by default. |
| Static or interactive | Static. |
| From 10 to 50,000 providers | Not affected: one row per term. |

## How to read it

Each row is a covariate: its estimate, its interval, and the same as text on the right. An interval that excludes the reference line excludes no association at the stated level.

## How it can mislead

A larger odds ratio is not a more important covariate when the covariates are in different units: an odds ratio per year of age and one for a binary diagnosis answer different questions.

## Call

```{code-block} python
from pprof_py.presentation import coefficient_table, forest

forest(model)
coefficient_table(model).to_markdown()
```
