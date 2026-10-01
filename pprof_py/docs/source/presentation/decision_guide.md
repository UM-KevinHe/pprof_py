# Which display answers my question?

Start from the question. Every display below takes a fitted model (tested inside the call) or a
`ProviderProfile`, and each has a page under {doc}`displays/index` with its spec sheet and the ways it can mislead.

| Question | Display | Table |
|:--|:--|:--|
| Which providers look unusually high or low? | {doc}`displays/funnel` (departures beyond the test's sampling variation, by precision) | `provider_table` |
| How uncertain are the estimates? | {doc}`displays/caterpillar` (each provider's interval) | `provider_table` |
| How do providers compare with the benchmark? | {doc}`displays/funnel`, {doc}`displays/caterpillar` (the reference line; flags above or below it) | `provider_table` |
| Are the extreme providers small or large? | {doc}`displays/funnel` (precision axis), {doc}`displays/observed_expected` (expected events), {doc}`displays/caterpillar` (volume panel) | `provider_table` |
| How much variation exists across providers? | {doc}`displays/provider_variation` (random-effect models), {doc}`displays/reliability` | `provider_variation_table`, `reliability_table` |
| How stable are the rankings? | The package does not rank providers; {doc}`displays/flag_stability` shows how fragile the flags are, and {doc}`displays/shrinkage` how far pooling moves each provider | `flag_stability_table`, `shrinkage_table` |
| How do providers compare across measures? | {doc}`displays/several_measures` | `multi_measure_table` |
| How should one estimate be read, given its uncertainty? | {doc}`displays/caterpillar` (its interval and volume), {doc}`displays/reliability` (reliability at its size) | `provider_table` |
| How large are the covariate effects? | {doc}`displays/forest` | `coefficient_table` |
| Is the theoretical null credible? | {doc}`displays/null_calibration` | `null_calibration_table` |
| Who is missing or unreliable, and why? | {doc}`displays/data_quality` | `data_quality_table` |

## Which models support which display

✓ supported; — not offered (the call raises an error that says what to use instead). Every cell is exercised on
fitted models by the package's tests (`tests/presentation/test_capability_matrix.py`).

| Display | Logistic FE | Logistic RE | Linear FE | Linear RE | Three-stage | CoxPH |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Funnel | ✓ | count tests only¹ | ✓ (Wald) | —² | ✓ | ✓ |
| Interval plot | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Observed versus expected | ✓ | count tests only¹ | — | — | ✓ | ✓ |
| Coefficient forest | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Data quality | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Null calibration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Between-provider variation | — | ✓ | — | ✓ | — | — |
| Shrinkage | with logistic RE | with logistic FE | with linear RE | with linear FE | — | — |
| Flag stability | ✓ | ✓ | ✓ | ✓ | ✓ (adds σ's bounds) | ✓ (no reference scenario) |
| Several measures | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

¹ These displays default to `test_method="poibin_exact"` (`"exact"` also works), which tests the plotted observed
counts. The model's default Wald test of shrunken estimates would contradict any O/E funnel, so `test_method="wald"`
is refused.
² Linear random-effect models flag with a test of shrunken estimates; no funnel can agree with those flags. Use the
interval plot.

Reliability takes an IUR object (`BootstrapIUR`, `DirectIUR`, `SplitHalfIUR`) built from any family's observed and
expected outcomes; its figure needs `BootstrapIUR`.
