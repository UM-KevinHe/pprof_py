# Tables

Every table is a `TableResult` built from a library-independent specification and exported to HTML (semantic markup,
`<caption>`, `<th scope>`), Markdown, LaTeX (booktabs; `longtable` above 40 rows), plain text, Excel (numeric cells
with number formats; the optional `excel` extra) or a DataFrame (`to_frame()`, with provenance). Footnotes are
generated from the source's settings. Markdown and text, which cannot span columns, write a grouped header as a prefix
(`Readmission: Estimate (CI)`).

| Function | Lists |
|:--|:--|
| `provider_table(source)` | Each provider's volume, observed and expected events, estimate with interval, flag and p-value, in source order |
| `coefficient_table(model)` | Covariate effects from `summary()` with intervals and p-values; odds or hazard ratios for logistic and Cox models |
| `multi_measure_table(measures)` | Each measure's estimate and flag under a grouped header |
| `data_quality_table(source)` | The provider accounting with definitions; `details=True` lists the affected providers |
| `null_calibration_table(model, ...)` | Per null group: the fitted null and the flags under the theoretical and fitted nulls |
| `reliability_table(iur)` | The overall IUR, its variance decomposition and reliability by size decile |
| `provider_variation_table(model)` | The random-effect SD with its interval and the range of true effects it implies |
| `shrinkage_table(fixed, random)` | Each provider's fixed-effect and random-effect estimate and the change |
| `flag_stability_table(model, ...)` | Flags per scenario, or with `details=True` the providers whose flag changes |

## Symbols and formatting

| Symbol | Meaning |
|:--|:--|
| ▲ / ▼ | Flagged above / below the reference |
| ● | Not different from the reference |
| NT | Not tested |
| NE | No finite estimate |
| NI | No interval |
| S | Suppressed (below the minimum volume) |
| — | Not applicable (for example, a measure that does not include the provider) |

Intervals read `1.12 (0.95–1.31)`, or `−0.40 (−0.62 to −0.18)` when a bound is negative, with a true minus
sign. P-values below 0.001 read `<0.001` and never `0`. Rounding never changes meaning: if a flagged provider's
rounded interval would touch the rounded reference, the table shows more digits.
