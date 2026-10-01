# Shrinkage

```{include} ../_figures/shrinkage.md
```

**Answers:** How far does pooling move each provider?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | How far does pooling move each provider? |
| Quantity and source | Each provider's fixed-effect estimate minus its test's null value (with `reference="mean"` by default, the size-weighted mean) against its BLUP minus 0, each from its own test; volumes from the fixed-effect fit. |
| Uncertainty | Not shown per provider; the interval plot of each fit shows it. |
| Denominator | Point area is proportional to records. |
| Reference | The diagonal (no shrinkage) and the horizontal line at 0 (complete pooling). |
| Misreadings and mitigations | Shrunken estimates are not the truth: how far each moves depends on the assumed normal random-effect distribution, and the footnote gives σ. Providers without a finite fixed-effect estimate are drawn at the axis edge and counted. |
| Static or interactive | Static; `shrinkage_table()` lists both estimates and the change. |
| From 10 to 50,000 providers | Points are rasterized above 2,000 providers. |

## How to read it

A provider on the diagonal is not moved by pooling; the closer it lies to the horizontal line, the further it is pulled toward the average. Small providers (small points) move most.

## How it can mislead

Shrinkage is a modelling choice: with another assumed distribution the BLUPs would move differently. Compare the fixed and random effects, but do not read either as the true effect.

## Call

```{code-block} python
from pprof_py.presentation import shrinkage, shrinkage_table

shrinkage(model, random_model)
```
