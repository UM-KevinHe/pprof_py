# Gallery

Every figure the presentation layer draws, from 200 synthetic providers (`pprof_py.presentation._synthetic`, with
planted outliers, overdispersion and zero-event providers). The images are generated when the documentation is built,
with the same deterministic renderers you call; each shows its own footnote with the settings it depends on.
All of them are drawn at double-column width (`size="double"`, 175 mm, the interval plot's default), so they share
one scale and one type size. Called without `size`, the interval plot, data quality, null calibration and several
measures default to double width and the other displays to single-column width (`size="single"`, 85 mm).
Every display also has an interactive version on its own page (hover for values, a legend, a linked sortable table
and a table-only view); the *Interactive version* link beside each item opens it.

```{include} _gallery_items.md
```
