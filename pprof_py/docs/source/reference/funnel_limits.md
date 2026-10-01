(funnel-limits-guide)=
# Funnel limits

A funnel plot shows each provider's estimate against its precision, with control limits that narrow as precision
grows. A funnel only helps when its limits agree with the flags: a provider drawn outside the limits must be a
flagged provider, and a flagged provider must be drawn outside them.
{func}`~pprof_py.inference.funnel_limits` guarantees this by building the limits from the same test as the flags.
It runs the model's `test()` once and reuses that test's null distributions, its calibration (`null_mean`,
`null_sd`) and its decision rule, then checks the result before returning it.

## Quick start

The examples use a simulated logistic fixed-effect fit with 80 providers. One provider has 8 records, which data
preparation removes, and one has no events.

```python
import numpy as np
import pandas as pd
from pprof_py import LogisticFixedEffectModel
from pprof_py.inference import EmpiricalNull, funnel_limits

rng = np.random.default_rng(2026)
n = 80
size = rng.integers(20, 200, n)
size[0] = 8                                    # at most 10 records: removed by data preparation
gamma = rng.normal(-1.5, 0.35, n)
gamma[1:5] += [0.9, -0.9, 0.7, -0.7]           # two providers well above and two well below the others
pid = np.repeat(np.arange(n), size)
x = rng.normal(size=pid.size)
y = rng.binomial(1, 1 / (1 + np.exp(-(gamma[pid] + 0.5 * x))))
y[pid == 5] = 0                                # a provider with no events
data = pd.DataFrame({"y": y, "x": x, "provider": [f"H{j:02d}" for j in pid]})

model = LogisticFixedEffectModel()
model.fit(data, y_var="y", x_vars=["x"], provider_var="provider")
```

`funnel_limits(model)` calls `model.funnel_limits()`. For logistic fixed-effect models the funnel uses the score
test by default, whose limits are smooth curves. The other keywords are those of `test()`.

```python
fl = funnel_limits(model)
print(fl.attrs["test_method"], "|", fl.attrs["estimate_kind"], "|", fl.attrs["precision_kind"])
cols = ["observed", "expected", "estimate", "precision", "lower", "upper", "flag"]
print(fl.providers[cols].head(6).round(3).to_string())
```
```text
score | ratio | inverse_null_variance
             observed  expected  estimate  precision  lower  upper  flag
provider_id
H01              19.0    10.368     1.833     13.836  0.473  1.527     1
H02               1.0     4.728     0.212      6.073  0.205  1.795     0
H03              39.0    25.470     1.531     32.780  0.658  1.342     1
H04               4.0    16.411     0.244     21.191  0.574  1.426    -1
H05               0.0    19.804     0.000     25.746  0.614  1.386    -1
H06               6.0     5.854     1.025      7.362  0.278  1.722     0
```

The result is a {class}`~pprof_py.inference.FunnelLimits`:

* `test`: the `test()` result the limits come from, unchanged, so a display can use it rather than test again;
* `providers`: each provider's coordinates and its own limits at the test's level, with the test's `flag`;
* `curves`: limit curves over a precision grid, in long form, one set per null group and level;
* `attrs`: the test's settings plus the construction (`estimate_kind`, `precision_kind`, `limit_rule`,
  `curve_kind`, `levels`).

The guarantee, checked on the result:

```python
p = fl.providers
outside = (p["estimate"] > p["upper"]) | (p["estimate"] < p["lower"])
print(pd.crosstab(outside.rename("outside the limits"), p["flag"]))
```
```text
flag                -1  0   1
outside the limits
False                0  55   0
True                 8   0  16
```

`funnel_limits` raises an error instead of returning limits that disagree with the flags.

## How the limits are built

| Test | Estimate | Precision | Limits |
|---|---|---|---|
| Count tests (`poibin_exact`, random-effect `exact` and `poibin_exact`, three-stage) | O/E | expected count E | for each provider, the smallest count the test flags high and the largest it flags low; limits half-way between counts, `(o_hi − ½)/E` and `(o_lo + ½)/E` |
| CoxPH `midp` and `exact` | O/E | E | the same search under the test's Poisson null |
| Logistic score test | O/E | E²/V₀ | `1 + (null_mean ± c·null_sd)·√V₀/E` |
| Wald tests (logistic `wald`, linear fixed effects) | effect | 1/SE² | `reference + q(null_mean ± c·null_sd)·SE`, where `q` converts to Student-t when the test uses it |

Here `c` is the critical value: `critical` when given, otherwise the normal quantile for `level` and
`alternative`, exactly as in {func}`~pprof_py.inference.flags`. One-sided alternatives have one limit; the other is
`-inf` or `inf`. A side that no count can reach is also `inf` or `-inf`.

## Exact count tests

For an exact Poisson-binomial test, the limit at a given expected count also depends on how the provider's
patients' probabilities are spread, so the limits are per provider. They sit half-way between counts, and no
provider lies on a line:

```python
exact = model.funnel_limits(test_method="poibin_exact")
e = exact.providers
limits = e[["lower", "upper"]].to_numpy() * e[["expected"]].to_numpy()
print(exact.attrs["curve_kind"], np.unique(np.round(limits[np.isfinite(limits)] % 1, 9)))
print(pd.crosstab(fl.providers["flag"].rename("score"), e["flag"].rename("poibin_exact")))
```
```text
poisson_reference [0.5]
poibin_exact  -1  0   1
score
-1             8   0   0
0              1  54   0
1              0   0  16
```

The two tests flag the providers slightly differently, and each funnel agrees with its own test. The curves of an
exact count test are a reference: Poisson limits for the same calibration (`curve_kind` `"poisson_reference"`), not
the test itself. Score, Wald and CoxPH curves are exact functions of the precision (`curve_kind` `"exact"`).

## Levels

Flags exist only at the test's level. Further levels, for example 99.8%, give reference curves; the curve rows of
the test's own decision have `test_level` True:

```python
c = model.funnel_limits(levels=(0.95, 0.998)).curves
ends = c[c["precision"].isin([c["precision"].min(), c["precision"].max()])]
print(ends[["level", "test_level", "precision", "lower", "upper"]].round(3).to_string(index=False))
```
```text
 level  test_level  precision  lower  upper
 0.950        True      5.042  0.127  1.873
 0.950        True     50.032  0.723  1.277
 0.998       False      5.042 -0.376  2.376
 0.998       False     50.032  0.563  1.437
```

## Empirical nulls

Under an empirical null fitted by groups, each group has its own calibration, so the curves come in one set per
`null_group`:

```python
emp = model.funnel_limits(null_model=EmpiricalNull.fitter(size=model.provider_sizes_, n_groups=2))
print(emp.test.groupby("null_group")[["null_mean", "null_sd"]].first().round(3).to_string())
print(emp.curves.groupby("null_group").size().to_string())
```
```text
            null_mean  null_sd
null_group
1               0.209    1.753
2              -0.004    2.056
null_group
1    200
2    200
```

When the calibration differs between providers of one group, no curve can represent it: `curves` is empty and
`curve_kind` is `"unavailable"`, while the per-provider limits remain exact.

## Models without a funnel

A funnel needs a test of the plotted measure (ADR-004 in the design notes). Random-effect models flag providers with
a Wald test of their shrunken estimates by default; a funnel of observed over expected events cannot agree with
those flags. Logistic random-effect models therefore offer funnels for their count tests only, and linear
random-effect models offer none. Monte Carlo tests have no funnel limits either.

```python
from pprof_py import LinearRandomEffectModel, LogisticRandomEffectModel

for call in (lambda: funnel_limits(LinearRandomEffectModel()),
             lambda: LogisticRandomEffectModel().funnel_limits(test_method="wald"),
             lambda: model.funnel_limits(test_method="bootstrap_exact")):
    try:
        call()
    except (TypeError, ValueError) as err:
        print(f"{type(err).__name__}: {err}")
```
```text
TypeError: LinearRandomEffectModel flags providers with a Wald test of their shrunken estimates (BLUPs); funnel limits could not agree with those flags (ADR-004). Use an interval plot, or fit a LinearFixedEffectModel for a funnel.
ValueError: test_method='wald' tests the shrunken estimates (BLUPs), so no funnel of observed versus expected events can agree with its flags (ADR-004). Use test_method='poibin_exact' or 'exact', or an interval plot.
ValueError: test_method='bootstrap_exact' is a Monte Carlo test and has no funnel limits; use 'score', 'poibin_exact' or 'wald'.
```

## Zero-event providers and exclusions

A provider with no events, or only events, has no finite fixed-effect estimate. Its score and exact tests are
still valid, and its flag is never changed. {func}`~pprof_py.inference.degenerate_providers` reports these
providers for every binary-outcome model, so displays can mark them instead of drawing the solver's clamp:

```python
from pprof_py.inference import at_bound, degenerate_providers

deg = degenerate_providers(model)
print(deg[~deg["finite_estimate"]].to_string())
print(model.excluded_providers_.to_string())
```
```text
             events  trials  zero_events  all_events  finite_estimate
provider_id
H05             0.0   104.0         True       False            False
             n_records              reason
provider_id
H00                  8  at most 10 records
```

`excluded_providers_` lists the providers that data preparation removed. It is `None` when the model did not
prepare the data itself. {func}`~pprof_py.inference.at_bound` applies to fixed provider effects on a binary outcome
and says so elsewhere:

```python
try:
    at_bound(LogisticRandomEffectModel())
except TypeError as err:
    print(err)
```
```text
at_bound() needs a fitted model with fixed provider effects (coefficients_['gamma']); LogisticRandomEffectModel has none. degenerate_providers(model) reports providers with no events or only events for every binary-outcome model.
```

## Cost

`funnel_limits` costs one `test()` call plus the construction of the limits. For score and Wald tests the
construction is negligible. For exact count tests it is about half a second for 1,000 providers on one CPU, while
`test(test_method="poibin_exact")` itself takes about five seconds there, almost all of it inverting the exact test
for each provider's confidence interval. A display that uses `FunnelLimits.test` pays for the test once.
