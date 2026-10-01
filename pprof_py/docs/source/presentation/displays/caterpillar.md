# Interval plot with volume panel

```{include} ../_figures/caterpillar.md
```

**Answers:** How large and how uncertain are the provider estimates, relative to the reference?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | How large and how uncertain are the provider estimates, relative to the reference? |
| Quantity and source | One `test()` call: `estimate`, `ci_lower`, `ci_upper`, `flag` and `null_value`, on the test's scale (the provider effect, or a standardized measure through `test_standardized()`). Volumes come from the model's data. |
| Uncertainty | Each provider's interval from inverting the same test, so an interval excludes the reference exactly when the provider is flagged. Intervals are drawn from lower to upper bound, so shifted empirical-null intervals are drawn correctly. |
| Denominator | A volume panel beside the intervals, labelled with its kind: records or patients, expected events, or person-time. |
| Reference | A vertical line at the test's null value, defined in the footnote. |
| Misreadings and mitigations | The order is not a rank: the axis says the providers are ordered by estimate for legibility. Non-overlapping intervals are not a test of a difference between two providers. Small providers have wide intervals, which the volume panel makes visible. Providers without a finite estimate are marked at the axis edge with their one-sided interval, and the solver's bound is never drawn as an estimate. The footnote states whether the estimates are unshrunken fixed effects or shrunken BLUPs. |
| Static or interactive | Static. |
| From 10 to 50,000 providers | Rows are labelled up to 60 providers; above that they are unlabelled and `highlight=` providers are annotated. Above 2,000 providers the interval layer is rasterized. Legend and alt text always report the full count. |

## How to read it

Each row is a provider: the segment is its interval, the marker its estimate and status, and the bar beside it its volume. Rows whose interval crosses the reference line are not different from it at the test's level.

## How it can mislead

The ends of an ordering by estimate are dominated by the least certain providers. Below, the four lowest estimates belong to providers without events: their estimates are the solver's bound, not measurements, and their intervals are one-sided. Three of the five highest belong to providers with at most 20 records, whose intervals are wide. The interval plot marks the first at the axis edge and shows the second with their volumes.

```python
from pprof_py import LogisticFixedEffectModel
from pprof_py.presentation import ProviderProfile
from pprof_py.presentation._synthetic import provider_data

data = provider_data(200)
model = LogisticFixedEffectModel()
model.fit(data, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
import pandas as pd

profile = ProviderProfile.from_model(model, test_method="poibin_exact")
ordered = profile.data.sort_values("estimate")
ends = pd.concat([ordered.head(5), ordered.tail(5)])
print(ends[["estimate", "ci_lower", "ci_upper", "denominator", "finite_estimate", "status"]].round(2).to_string())
```
```text
             estimate  ci_lower  ci_upper  denominator  finite_estimate status
provider_id
F092            -9.88      -inf     -1.62         13.0            False  below
F175            -9.82      -inf     -1.56         13.0            False  below
F161            -9.68      -inf     -1.59         15.0            False  below
F040            -9.67      -inf     -2.15         25.0            False  below
F177            -3.56     -6.60     -1.90         32.0             True  below
F192            -0.31     -1.37      0.72         16.0             True  above
F018            -0.25     -0.72      0.21         78.0             True  above
F141            -0.23     -0.67      0.20         88.0             True  above
F111            -0.10     -1.01      0.84         20.0             True  above
F190            -0.04     -1.05      0.99         17.0             True  above
```

## Call

```{code-block} python
from pprof_py.presentation import caterpillar

caterpillar(profile, highlight=["F023"]).save("intervals.pdf")
```
