(plotting-guide)=
# Plotting: Caterpillar and Funnel Plots

```{note}
The model plot methods now delegate to the presentation layer (the funnel and interval plots of
`pprof_py.presentation`); some keywords described below no longer have an effect and warn until 0.7.0. See
{ref}`presentation-guide` for the displays and the migration table.
```

Two chart types cover most provider-profiling visualization needs:
one estimate per provider with an uncertainty interval, sorted
(**caterpillar**), and one estimate per provider plotted against its
own precision, with statistical control limits instead of per-point
intervals (**funnel**). Both take a plain DataFrame, not a fitted
model object, so they work identically regardless of which model
class produced the numbers — `LogisticFixedEffectModel`,
`LogisticFERandomClusterModel`
([Chapter 5](../logistic/logistic_three_stage_model)),
`ProviderPenalizedLogistic`
([provider-penalized chapter](../logistic/provider_penalized_logistic)),
or `ProviderPenalizedCoxPH`
([Chapter 12](../survival/12_provider_penalized_cox)).

## `plot_caterpillar`: sorted estimates with intervals

The examples use 30 synthetic providers, with Poisson counts observed
against their expected values, so that they run on their own:

```python
import numpy as np
import pandas as pd
from scipy.stats import chi2
from pprof_py import plot_caterpillar

rng = np.random.default_rng(0)
expected = rng.uniform(5, 60, 30)
observed = rng.poisson(expected * np.exp(rng.normal(0, 0.2, 30)))
est = observed / expected                                              # SMR-like ratio
ci_lo = chi2.ppf(0.025, 2 * observed) / 2 / expected                   # exact Poisson limits
ci_hi = chi2.ppf(0.975, 2 * (observed + 1)) / 2 / expected
flags = np.where(ci_lo > 1, 1, np.where(ci_hi < 1, -1, 0))

results = pd.DataFrame({
    "provider": [f"Facility_{i}" for i in range(30)],
    "estimate": est,        # e.g. gamma_, or an SRR/SMR
    "lower": ci_lo,
    "upper": ci_hi,
    "flag": flags,          # -1 / 0 / 1, optional
})

plot_caterpillar(
    results, estimate_col="estimate", ci_lower_col="lower",
    ci_upper_col="upper", group_col="provider", flag_col="flag",
    sort_by_estimate=True, refline_value=1.0,
)
```

Sorts providers by `estimate_col` (`sort_by_estimate=True`, the
default) and draws each with its `ci_lower_col`/`ci_upper_col`
interval; `flag_col`, if given, color-codes points by significance
(matching the `flag` convention every `.test()` method returns: `1`
above the reference, `-1` below, `0` not significant, and `NA` for a
provider the test could not evaluate).
`refline_value` draws a reference line (default `0.0` — the
right choice for a log-scale effect like `gamma_`; pass `1.0` instead
for a ratio measure like an SRR/SMR). Dozens of further keyword
arguments (`figure_size`, `point_color_default`, `orientation`,
`errorbar_alpha`, and more) control styling without changing the
plot's structure — the function signature groups them clearly by
purpose (point styling, reference line, typography, grid); check it
directly for the full list rather than this page enumerating every
one. `save_path=` writes the figure to a file; the function returns
`None`.

## `plot_funnel`: estimates against precision, with control limits

```python
from pprof_py.plotting import plot_funnel

results["precision"] = expected                                        # the funnel's x-axis
grid = np.linspace(expected.min(), expected.max(), 200)
limits_df = pd.concat([
    pd.DataFrame({"precision": grid, "alpha": a,
                  "control_lower": chi2.ppf(a / 2, 2 * grid) / 2 / grid,
                  "control_upper": chi2.ppf(1 - a / 2, 2 * (grid + 1)) / 2 / grid})
    for a in (0.05, 0.002)
])
fig, ax = plot_funnel(results, limits_df, estimate_col="estimate",
                      precision_col="precision", flag_col="flag", target=1.0)
```

`plot_funnel` is not exported at the `pprof_py` package root; import it
from `pprof_py.plotting`. It returns the figure and axes. `limits_df`
needs the columns `precision`, `control_lower`, `control_upper` and
`alpha`, one set of curves per `alpha`.

A funnel plot's whole point is different from a caterpillar plot's:
rather than an interval per provider, every provider is one point at
`(precision_col, estimate_col)`, and `limits_df` supplies **control
limit curves** — pre-computed boundaries (typically from the same
Poisson/exact machinery
[survival Chapter 4 §4.8](../survival/04_indirect_standardization_smr_shr)
walks through) — as continuous lines across the precision range, so a
provider's distance from the target line relative to its own precision
is visible directly, without needing to read dozens of individual
error bars. `target` (default `1.0`) is the reference value a provider
"as expected" would sit on — `1.0` for a ratio measure (SRR/SMR),
`0.0` for a difference-scale one. `alpha_levels` controls how many
control-limit curves are drawn (e.g., 95% and 99.8%, following the
common two-tier convention in this literature) — omit it to use the
function's own default tiers.

## Building the inputs

Neither function fits a model or computes intervals — both are pure
plotting layers over a DataFrame you construct from whatever fitted
model you're profiling. The `predict_provider_effect()` /
`calculate_standardized_measures()` / `test()` methods this
documentation covers throughout (most recently in
[Chapter 5's](../logistic/logistic_three_stage_model) Section 8 and
the [provider-penalized logistic chapter's](../logistic/provider_penalized_logistic)
Section 6) are the usual source for `estimate_col`, interval columns,
and `flag_col` — assemble their output into one DataFrame with the
column names above (or pass different names via the `*_col`
parameters) and either plotting function takes it directly.
