(plotting-guide)=
# Plotting: Caterpillar and Funnel Plots

Two chart types cover most provider-profiling visualization needs:
one estimate per provider with an uncertainty interval, sorted
(**caterpillar**), and one estimate per provider plotted against its
own precision, with statistical control limits instead of per-point
intervals (**funnel**). Both take a plain DataFrame, not a fitted
model object, so they work identically regardless of which model
class produced the numbers — `LogisticFixedEffectModel`,
`LogisticMixedEffectModel`
([Chapter 5](../logistic/logistic_mixed_effect_model)),
`ProviderPenalizedLogistic`
([provider-penalized chapter](../logistic/provider_penalized_logistic)),
or `ProviderPenalizedCoxPH`
([Chapter 12](../survival/12_provider_penalized_cox)).

## `plot_caterpillar`: sorted estimates with intervals

```python
from pprof_py import plot_caterpillar
import pandas as pd

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
    sort_by_estimate=True, refline_value=0.0,
)
```

Sorts providers by `estimate_col` (`sort_by_estimate=True`, the
default) and draws each with its `ci_lower_col`/`ci_upper_col`
interval; `flag_col`, if given, color-codes points by significance
(matching the `-1`/`0`/`1` convention
[the empirical null guide's](empirical-null-guide) `assign_flags`
produces and every `.test()` method in this documentation returns).
`refline_value` draws a horizontal reference (default `0.0` — the
right choice for a log-scale effect like `gamma_`; pass `1.0` instead
for a ratio measure like an SRR/SMR). Dozens of further keyword
arguments (`figure_size`, `point_color_default`, `orientation`,
`errorbar_alpha`, and more) control styling without changing the
plot's structure — the function signature groups them clearly by
purpose (point styling, reference line, typography, grid); check it
directly for the full list rather than this page enumerating every
one. `save_path=` writes directly to a file instead of (or alongside)
displaying the figure; the function itself returns `None`.

## `plot_funnel`: estimates against precision, with control limits

```python
from pprof_py import plot_funnel

plot_funnel(
    results_df, limits_df, estimate_col="estimate",
    precision_col="precision", flag_col="flag", target=1.0,
)
```

```{note}
`plot_funnel` is **not** re-exported at the `pprof_py` package root —
`from pprof_py import plot_funnel` raises `ImportError`. Import it from
`pprof_py.plotting` instead: `from pprof_py.plotting import
plot_funnel`. `plot_caterpillar` is the only plotting function in
`pprof_py.__all__`; this is a real, confirmed asymmetry, not an
oversight in this page.
```

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
[Chapter 5's](../logistic/logistic_mixed_effect_model) Section 8 and
the [provider-penalized logistic chapter's](../logistic/provider_penalized_logistic)
Section 6) are the usual source for `estimate_col`, interval columns,
and `flag_col` — assemble their output into one DataFrame with the
column names above (or pass different names via the `*_col`
parameters) and either plotting function takes it directly.
