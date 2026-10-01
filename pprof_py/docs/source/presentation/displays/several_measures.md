# Several measures

```{include} ../_figures/multi_measure.md
```

```{include} ../_figures/measure_agreement.md
```

**Answers:** How do providers compare across measures, and do the measures agree?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | How do providers compare across measures, and do the measures agree? |
| Quantity and source | A `ProfileCollection` of measures, each from its own `test()`: estimates, intervals and flags. The agreement plot counts the two tests' joint flags (same direction, opposite, one measure only, neither). |
| Uncertainty | Each measure's intervals: segments in the small multiples, crosses in the agreement plot. |
| Denominator | Not shown; the multi-measure table and each measure's provider table give volumes. |
| Reference | Each measure's own reference line. |
| Misreadings and mitigations | Panels are not a composite score, and each has its own scale. Estimation noise attenuates the observed agreement, so the cloud understates how closely true provider effects agree, and a flag on one measure says nothing by itself about the other; the footnotes say so. Providers missing from a measure read n/a, and providers without a finite estimate are not placed in the agreement plot. |
| Static or interactive | Static; `multi_measure_table()` puts each measure under its own grouped header. |
| From 10 to 50,000 providers | Small multiples are labelled up to 60 rows and dense beyond; the agreement plot is rasterized above 2,000 providers. |

## How to read it

In the small multiples a provider keeps its row across panels, so its position can be followed from measure to measure. In the agreement plot each point is a provider in both measures, with its two intervals as a cross.

## How it can mislead

Averaging or ranking across the panels creates a composite that no test supports. Compare measures provider by provider, with their intervals.

## Call

```{code-block} python
from pprof_py.presentation import ProfileCollection, measure_agreement, multi_measure, multi_measure_table

measures = ProfileCollection({"Readmission": readmission_model, "Mortality": mortality_model})
multi_measure(measures)
measure_agreement(measures)
```
