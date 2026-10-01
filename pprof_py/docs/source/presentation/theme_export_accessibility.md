# Theme, export and accessibility

## Themes

Every figure and table reads its typography, line widths, colours, marker encodings and sizes from one immutable
`Theme`. Three presets cover the usual outlets: `publication` (the default, for print at final size), `notebook` and
`report`. To change the look, derive a theme rather than passing styling keywords:

```python
from pprof_py.presentation import Theme

for name in ("publication", "notebook", "report"):
    t = getattr(Theme, name)()
    print(f"{name}: title {t.typography.title} pt, axis label {t.typography.label} pt, tick {t.typography.tick} pt")
print("named widths (mm):", Theme.publication().widths_mm)

house = Theme.publication().derive(typography={"tick": 7.5})
check = house.accessibility_report()
print("house theme: minimum text", check["min_font_pt"], "pt; statuses distinct in shape and fill:", check["unique_encodings"])
print("status contrast against white:", {k: round(v, 2) for k, v in check["status_contrast"].items()})

pale = house.derive(status={"above": {"color": "#F5D76E"}})
print("a pale 'above' colour has contrast", round(pale.accessibility_report()["status_contrast"]["above"], 2), "(3:1 required)")
```
```text
publication: title 8.5 pt, axis label 7.5 pt, tick 7.0 pt
notebook: title 12.0 pt, axis label 11.0 pt, tick 10.0 pt
report: title 13.0 pt, axis label 11.0 pt, tick 10.0 pt
named widths (mm): {'single': 85.0, 'double': 175.0}
house theme: minimum text 7.0 pt; statuses distinct in shape and fill: True
status contrast against white: {'above': 4.87, 'below': 10.38, 'not_different': 3.36, 'not_tested': 8.45, 'no_finite_estimate': 21.0}
a pale 'above' colour has contrast 1.42 (3:1 required)
```

Pass a theme to any display with `theme=` (a `Theme` or a preset name), and a width with `size=` (`"single"`,
`"double"` or a width in millimetres).

## Export

`FigureResult.save(path)` writes SVG, PDF or PNG by the file's extension (`dpi=` for PNG, 300 by default), and
`to_bytes(format)` returns the bytes. Exports are deterministic: the same input, theme and environment give
byte-identical files, whatever was drawn before in the same process, because layouts are frozen and quantised and
SVG ids are canonical. Different Matplotlib or FreeType versions can change glyph outlines, so byte identity holds
within one environment. Text uses DejaVu Sans, which ships with Matplotlib, so no system font is needed; nothing is
loaded from the network.

Tables export with `to_html()`, `to_markdown()`, `to_latex()`, `to_text()`, `to_excel()` (the optional `excel` extra)
and `to_frame()`. A `Report` writes figures and tables into one self-contained HTML file.

## Accessibility

- **Status encodings.** Above, below, not different, not tested and no finite estimate differ in colour, marker
  shape, fill and label, never in colour alone. The above and below colours are a dark orange and purple pair that
  stays distinct in grayscale and under simulated protan, deutan and tritan vision.
- **Contrast.** Status markers and meaningful lines have at least 3:1 contrast against the background and text at
  least 4.5:1; `Theme.accessibility_report()` measures a theme, as in the example above, and the package's tests
  hold the presets to these thresholds.
- **Text size.** At least 7 pt at final print size in figures; at least 14 px body text in HTML tables and reports.
- **Alt text.** Every figure has generated alt text and a long description (`fig.alt_text`,
  `fig.long_description`), built from the data it shows; SVG exports carry them as `<title>` and `<desc>`, and
  reports as `alt` and `aria-describedby`.
- **Tables.** HTML tables use `<caption>`, `<th scope>` and footnotes in `<tfoot>`; nothing is conveyed by colour.
- **Disclosure.** Figures carry only what they draw; tables, and reports that contain them, list provider-level
  values, and reports say so.
