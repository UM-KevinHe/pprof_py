# Observed versus expected

```{include} ../_figures/observed_expected.md
```

**Answers:** Where do observed counts depart from expected, and at what volume?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | Where do observed counts depart from expected, and at what volume? |
| Quantity and source | Each provider's observed and expected events from the same test (the funnel limits of a count or score test, or the CoxPH test's columns). The limits are the funnel limits of that test multiplied by the provider's expected count, a named conversion to events. |
| Uncertainty | The test's limits in events: each provider's own limits as marks, and exact count curves for CoxPH. Poisson reference curves of exact count tests are not drawn, because they are not that test's limits. |
| Denominator | The x axis: expected events. |
| Reference | The line O = E, with fainter guides at O/E = 0.5 and 2. |
| Misreadings and mitigations | Differences and ratios mean different things at different volumes. On square-root axes Poisson noise has roughly constant spread, so distances from the line are comparable across volumes, and the same ratio visibly means more excess events at a larger volume. For count tests the limits are half-integer counts, so no provider lies on a limit, and flagged providers are exactly those outside. |
| Static or interactive | Static. |
| From 10 to 50,000 providers | As the funnel: vectors up to 2,000 providers, rasterized not-different layer above that, flagged providers on top. |

## How to read it

A provider above the line had more events than expected and one below fewer. Its limit marks show how far from the line its count may fall at its volume before the test flags it.

## How it can mislead

An O/E of 1.5 is 5 extra events at E = 10 and 50 at E = 100: the same ratio is a much larger and much more certain departure at the larger provider. Read departures in events and in ratios together, with the volume.

## Call

```{code-block} python
from pprof_py.presentation import observed_expected

observed_expected(model, test_method="poibin_exact")
```
