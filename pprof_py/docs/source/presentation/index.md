# Presentation

The presentation layer turns validated `pprof_py` results into figures, tables and reports that communicate them
honestly. Every display reads its quantities from the statistical layer and never recomputes them: estimates,
intervals and flags come from one `test()` call, and funnel limits from `funnel_limits()`, so a provider outside the
limits is exactly a flagged provider. Uncertainty and volume are always shown, missing or untested providers are
labelled rather than dropped, providers are ordered for legibility rather than ranked, and every output carries the
settings it depends on (level, reference, null model, test method, estimator).

All displays share one theme, one formatter and one table specification, render without pyplot, and export
byte-identical SVG, PDF and PNG; tables export to HTML, Markdown, LaTeX, text and Excel.

```{toctree}
:maxdepth: 1

gallery
displays/index
tables
```

The worked examples and the migration table are on {doc}`../reference/presentation`.
