# Migrating to the presentation layer

The earlier plotting calls keep working. Most now draw through the presentation layer and return a `FigureResult`;
calls whose options or meaning had no test-consistent replacement were deprecated in 0.6.0 and removed in 0.7.0.

## Earlier call and replacement

| Earlier call | Now |
|---|---|
| `model.plot_funnel(...)` (logistic and linear fixed effects) | Delegates to `funnel(model, ...)` and returns a `FigureResult`; `fig, ax = model.plot_funnel()` still works. `alpha` sets the curve levels. |
| `model.plot_funnel()` (logistic random effects) | The funnel of the exact count test, `test_method="poibin_exact"`, now the default; `"wald"` raises a `ValueError` (0.7.0). |
| `model.plot_funnel()` (linear random effects) | Removed in 0.7.0: no funnel can agree with the test of shrunken estimates. Use `caterpillar(model)`. |
| `model.plot_provider_effects(...)` | Delegates to `caterpillar(model, ...)`. |
| `model.plot_standardized_measures(...)` (logistic fixed effects) | Delegates to `caterpillar` with the profile of `model.test_standardized(...)`. |
| `model.plot_standardized_measures(...)` (other models) | Removed in 0.7.0: their measure-scale intervals do not come from the flagging test. Use `caterpillar(model)`. |
| Styling keywords (`point_colors`, `labels`, `figsize`, ...), `target=`, `use_flags=False`, `stdz=` (linear funnel) | Removed in 0.7.0: they raise a `TypeError`. Use `theme=`; `title=` sets the title. |
| `save_path=` | Still works; or `result.save(path)`. |
| `plt.show()` after a plot call | Not needed in notebooks; in scripts, save the result. |
| `plot_caterpillar(df)` with `lower`/`upper` columns | Pass `ci_lower_col='lower', ci_upper_col='upper'`; the defaults are `ci_lower`/`ci_upper`, the columns of `test()`, and the fallback was removed in 0.7.0. |
| A hand-built `limits_df` for `pprof_py.plotting.plot_funnel` | `pprof_py.inference.funnel_limits(model)`, or `funnel(model)`. |
| `pprof_py.plotting.plot_funnel(df, limits_df, ...)` | Draws through `funnel()`: the supplied curves as given, each provider's limits read off the curve of the largest `alpha`, and a warning when flags contradict them. Returns a `FigureResult`; the styling keywords and `ax=` were removed in 0.7.0. |
| `pprof_py.plot_caterpillar(df, ...)` | Draws through `caterpillar()` and returns a `FigureResult` instead of `None` (no `plt.show()`). It needs intervals and `refline_value`; the styling keywords, `sort_by_estimate` and `orientation` were removed in 0.7.0. |
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

From 0.7.0 the presets carry the package's visual identity. Every value, flag, limit and interval is unchanged; only
the drawing is. `theme="classic"` reproduces 0.6.0 figures byte for byte in the same environment, and 0.6.0 HTML
tables and reports apart from the package version they record in their source line and footer;
`Theme.classic("notebook")` and `Theme.classic("report")` do the same for the other two presets.

To keep one 0.6.0 behaviour and take the rest, derive a theme and set its token, for example
`Theme().derive(footnote="full", key_position="bottom")`:

| What changed in 0.7.0 | Token | 0.6.0 value |
|:--|:--|:--|
| Text in IBM Plex Sans (bundled), titles semibold, axis labels medium | `typography={"family": ..., "title_weight": ..., "label_weight": ...}` | `"DejaVu Sans"`, `"normal"`, `"normal"` |
| Copper and petrol status pair, a cool grey for not different | `status={...}` | `#B35806`, `#542788`, `#8C8C8C` |
| A light grid and muted tick labels instead of an open frame | `grid`, `spines`, `tick_length`, `tick_label_color` | `False`, `("left", "bottom")`, `3.0`, `None` |
| The 95% acceptance region shaded in funnels | `corridor` | `None` |
| White halos on filled marks | `halo` | `0.0` |
| The key above the plot, under a title row | `key_position` | `"bottom"` |
| A one-line footnote in publication figures (the full text is `FigureResult.caption`) | `footnote` | `"full"` |
| Intervals as bars ending exactly on their bounds, marks on the bars a shade darker, thin volume bars | `interval_bars`, `bar_marks`, `volume_half` | `False`, `None`, `(0.35, 0.5)` |
| A ring, band and semibold label for `highlight=` providers | `highlight_ring`, `highlight_wash` | `None`, `None` |
| HTML tables and reports styled from the theme, status glyphs named | `table_style` | `"classic"` |

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
