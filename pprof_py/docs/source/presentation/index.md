# Presentation

The presentation layer turns validated `pprof_py` results into figures, tables and reports that communicate them
honestly. Every display reads its quantities from the statistical layer and never recomputes them: estimates,
intervals and flags come from one `test()` call, and funnel limits from `funnel_limits()`, so a provider outside the
limits is exactly a flagged provider. Uncertainty and volume are always shown, missing or untested providers are
labelled rather than dropped, providers are ordered for legibility rather than ranked, and every output carries the
settings it depends on (level, reference, null model, test method, estimator).

All displays share one theme, one formatter and one table specification, render without pyplot, and export
byte-identical SVG, PDF and PNG; tables export to HTML, Markdown, LaTeX, text and Excel.

## In this section

- [Which display answers my question?](decision_guide.md) — from the question you want to answer to the display and table that answer it.
- [Gallery](gallery.md) — every figure the layer draws, from 200 synthetic providers, each with its settings footnote and a link to its reading page.
- [Reading the displays](displays/index.md) — one page per display: what it shows, where the numbers come from, and how it can mislead.
- [Tables](tables.md) — every table as a `TableResult`, exported to HTML, Markdown, LaTeX, text, Excel or a DataFrame.
- [Theme, export and accessibility](theme_export_accessibility.md) — the `Theme` presets, the export formats and the accessibility choices behind the displays.
- [Migrating to the presentation layer](migration.md) — each earlier plotting call and its replacement.

The worked examples are on {doc}`../reference/presentation`; the migration table is on {doc}`migration`.
