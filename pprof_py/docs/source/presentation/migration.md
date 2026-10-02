# Migrating to the presentation layer

The earlier plotting calls keep working. Most now draw through the presentation layer and return a `FigureResult`;
calls whose options or meaning have no test-consistent replacement keep their earlier drawing with a
`DeprecationWarning`. Deprecated behaviour is removed in 0.7.0.

## Earlier call and replacement

| Earlier call | Now |
|---|---|
| `model.plot_funnel(...)` (logistic and linear fixed effects) | Delegates to `funnel(model, ...)` and returns a `FigureResult`; `fig, ax = model.plot_funnel()` still works. `alpha` sets the curve levels. |
| `model.plot_funnel()` (logistic random effects) | The funnel of the exact count test (`test_method="poibin_exact"`); the default `"wald"` warns, is replaced, and raises in 0.7.0. |
| `model.plot_funnel()` (linear random effects) | Deprecated: no funnel can agree with the test of shrunken estimates. Use `caterpillar(model)`. |
| `model.plot_provider_effects(...)` | Delegates to `caterpillar(model, ...)`. |
| `model.plot_standardized_measures(...)` (logistic fixed effects) | Delegates to `caterpillar` with the profile of `model.test_standardized(...)`. |
| `model.plot_standardized_measures(...)` (other models) | Deprecated: their measure-scale intervals do not come from the flagging test. Use `caterpillar(model)`. |
| Styling keywords (`point_colors`, `labels`, `figsize`, ...), `target=`, `use_flags=False` | No effect; `DeprecationWarning` now, `TypeError` in 0.7.0. Use `theme=`. |
| `save_path=` | Still works; or `result.save(path)`. |
| `plt.show()` after a plot call | Not needed in notebooks; in scripts, save the result. |
| `plot_caterpillar(df)` with `lower`/`upper` columns | The defaults are now `ci_lower`/`ci_upper`, the columns of `test()`; `lower`/`upper` are used with a warning until 0.7.0. |
| A hand-built `limits_df` for `pprof_py.plotting.plot_funnel` | `pprof_py.inference.funnel_limits(model)`, or `funnel(model)`. |
| `pprof_py.plotting.plot_funnel(df, limits_df, ...)` | Still works and now draws through `funnel()`: the supplied curves as given, each provider's limits read off the curve of the largest `alpha`, and a warning when flags contradict them. Returns a `FigureResult` (`fig, ax = ...` still works). Styling keywords are deprecated and ignored; `ax=` keeps the earlier drawing until 0.7.0. |
| `pprof_py.plot_caterpillar(df, ...)` | Still works and now draws through `caterpillar()`, returning a `FigureResult` instead of `None` (no `plt.show()`). Styling keywords are deprecated and ignored; no intervals, `refline_value=None`, `sort_by_estimate=False` and `orientation='horizontal'` keep the earlier drawing until 0.7.0. For model results, `caterpillar(model)` also marks providers without a finite estimate. |
| (new) | `provider_table(model)`: HTML, Markdown, LaTeX, text, DataFrame and Excel. |
| `model.plot_coefficient_forest(...)` | `forest(model)` and `coefficient_table(model)` (the method itself is unchanged for now). |


## What looks different

- Calls return a `FigureResult` (which still unpacks as `fig, ax`) instead of `None`, and never call `plt.show()`;
  results render in notebooks, and `fig.save(path)` writes files.
- Flags and intervals come from one `test()` call, so an interval excludes the reference exactly when the provider is
  flagged, and a funnel's limits come from the same test, so the providers outside them are the flagged ones.
- Statuses differ in shape, fill and label as well as colour; providers with a missing flag are drawn as "Not tested",
  and providers without a finite estimate at the axis edge.
- Every figure carries a footnote with its level, test, null model, reference and estimator.
- Exports are byte-identical for the same input and environment.

## The 0.6.0 look

From 0.7.0 the presets carry the package's visual identity (IBM Plex Sans, a copper and petrol status pair, a grid
instead of a frame). `theme="classic"` reproduces 0.6.0 figures byte for byte in the same environment;
`Theme.classic("notebook")` and `Theme.classic("report")` do the same for the other two presets.

## Styling keywords and themes

Styling keywords (`font_size`, `point_size`, `flag_colors`, `figure_size`, ...) are ignored with a warning. Derive a
theme instead and pass it with `theme=`:

```{code-block} python
from pprof_py.presentation import Theme, funnel

house = Theme.publication().derive(typography={"tick": 8.0}, status={"above": {"color": "#8C510A"}})
funnel(model, theme=house, size=120)          # width in mm
```

A derived theme is checked like the presets with `house.accessibility_report()`; see
{doc}`theme_export_accessibility`.
