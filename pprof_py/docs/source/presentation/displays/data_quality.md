# Data quality

```{include} ../_figures/data_quality.md
```

**Answers:** Who is missing or unreliable, and why?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | Who is missing or unreliable, and why? |
| Quantity and source | The profile's statuses and attributes, the model's record of providers excluded by data preparation (`excluded_providers_`), and the denominators. |
| Uncertainty | Not applicable: the panel counts providers. |
| Denominator | The volume panel's axis, in its unit (records, trials or expected events); excluded providers are placed on it only when its unit is records. |
| Reference | None: this is an accounting display. |
| Misreadings and mitigations | Silent exclusions: every provider is counted, and a count the source does not record reads "not recorded", never zero. Volumes in different units are never mixed. |
| Static or interactive | Static; `data_quality_table(..., details=True)` lists every affected provider. |
| From 10 to 50,000 providers | The counts are exact at any size; the volume strips are rasterized above 2,000 points. |

## How to read it

The bars account for every provider: in the data, excluded by data preparation, analysed by flag, not tested and suppressed, then the attributes (no finite estimate, zero events, no interval). The strips show each group's volumes, so it is clear whether the problem providers are the small ones.

## How it can mislead

A table of flagged providers alone hides who was never assessed. Read the flags with this accounting: providers excluded by data preparation or without a finite estimate are not "not different".

## Call

```{code-block} python
from pprof_py.presentation import data_quality, data_quality_table

data_quality(profile)
data_quality_table(profile, details=True).to_text()
```
