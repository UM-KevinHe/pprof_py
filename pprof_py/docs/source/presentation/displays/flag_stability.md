# Flag stability

```{include} ../_figures/flag_stability.md
```

**Answers:** How fragile are the flags?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | How fragile are the flags? |
| Quantity and source | The flags of one `test()` call per scenario: by default an alternative reference (where the test takes one), the other null, and for three-stage models the flags at the bounds of σ's interval (`sigma_sensitivity()`). The display decides no flag. |
| Uncertainty | Each scenario's flags summarize its own test; the footnote states each scenario's settings. |
| Denominator | Not shown; the data-quality display and the provider table give volumes. |
| Reference | Each scenario's own reference. |
| Misreadings and mitigations | Agreement across a few scenarios does not make a flag robust: risk adjustment, data preparation and the model itself are not varied, and the footnote says so. Rows are ordered by the base estimate for legibility, not as a ranking. |
| Static or interactive | Static; `flag_stability_table(..., details=True)` lists the changing providers. |
| From 10 to 50,000 providers | Rows are the providers flagged in at least one scenario; labelled up to 60 rows, unlabelled and rasterized beyond. |

## How to read it

Each row is a provider flagged in at least one scenario and each column a scenario; a diamond marks the providers whose status changes. Changes under the empirical null usually mean the theoretical null overstated the evidence.

## How it can mislead

A provider flagged in all three default scenarios may still change with another risk adjustment or data preparation; the matrix only covers the choices it shows.

## Call

```{code-block} python
from pprof_py.presentation import flag_stability, flag_stability_table

flag_stability(model, scenarios={"Wald": {"test_method": "wald"}})
```
