# ADR-002 — Dependencies for figures, tables and Excel

**Status:** accepted; `excel` extra approved (D14), interactive HTML deferred (D14) · 2026-09-30 · Evidence: `spikes/dep_eval.json`, `spikes/out/tables/`

## Evidence (PyPI, 2026-09-30; GitHub API returned HTTP 403 from this network, so maintenance is judged on release cadence)

| Package | Latest (date) | Releases, 12 mo | Requires-Python (resolves on 3.9 to) | Wheel | Adds beyond core (py3.12) | Finding |
|---|---|---:|---|---:|---|---|
| matplotlib | 3.11.2 (2026-09-11) | 8 | ≥3.11 (3.9.4) | 9.9 MB | core already | incumbent |
| plotly | 7.1.0 (2026-09-15) | 12 | ≥3.8 (7.1.0) | 9.7 MB | narwhals | static export needs Kaleido |
| kaleido | 1.4.0 (2026-08-31) | 3 | ≥3.8 | 0.1 MB | 5 packages | requires Chrome/Chromium (red flag for CI/HPC/air-gapped) |
| altair | 6.3.0 (2026-09-15) | 25 | ≥3.11 (6.0.0) | 0.8 MB | 9 packages (jsonschema, rpds-py, …) | static export needs vl-convert |
| vl-convert-python | 1.9.0.post1 (2026-09-14) | 8 | ≥3.7 | 33.5 MB | — | bundled JS engine |
| great-tables | 1.0.0 (2026-09-25) | 7 | ≥3.10 (0.21.0) | 1.6 MB | 11 packages, 12.1 MB | HTML differs between runs (random ids) |
| itables | 2.9.1 (2026-07-22) | 14 | ≥3.9 | 1.6 MB | — | interactive tables only |
| jinja2 (for `pandas.Styler`) | 3.1.6 (2025-03-05) | 0 | ≥3.7 | 0.1 MB | markupsafe | Styler HTML differs between runs (uuid ids) |
| openpyxl | 3.1.5 (2024-06-28) | 0 | ≥3.8 | 0.3 MB | et-xmlfile | output differs between runs even with fixed document dates |
| xlsxwriter | 3.2.9 (2025-09-16) | 0 | ≥3.8 | 0.2 MB | none | byte-identical with a fixed `created` date |

Spike (`table_spike.py`, two processes): hand-written Markdown/HTML/LaTeX identical; Great Tables HTML differs, its LaTeX identical; Styler HTML differs, LaTeX identical; XlsxWriter identical; openpyxl differs.

## Scored matrix (1 = poor, 5 = excellent)

| Criterion | Matplotlib | Plotly + Kaleido | Altair + vl-convert | Great Tables | Styler | itables | Hand-written behind spec | XlsxWriter | openpyxl |
|---|---|---|---|---|---|---|---|---|---|
| Fit for statistical graphics | 5 | 4 | 4 | — | — | — | — | — | — |
| Static export quality and determinism | 5 (verified, ADR-006) | 2 (headless browser) | 3 (33.5 MB engine) | 3 (HTML ids) | 3 (HTML ids) | 1 | 5 (verified) | 5 (verified) | 2 (verified non-reproducible) |
| Interactive / offline HTML | 1 | 5 | 4 | 3 | 2 | 4 | 3 (vendored JS later) | — | — |
| Table capabilities (spanners, notes, LaTeX, Excel) | — | — | — | 5 | 3 | 2 | 4 (we build what we need) | 4 | 4 |
| Python 3.9 floor and maintenance | 4 (3.9 → mpl 3.9.4) | 5 | 3 (latest ≥3.11) | 2 (latest ≥3.10) | 4 | 5 | 5 | 4 (stable, quiet) | 2 (no release in 2 years) |
| Footprint (added packages) | 5 (none) | 3 | 2 | 2 | 4 | 4 | 5 | 5 | 4 |
| Testability (assert on structure) | 5 | 3 | 3 | 3 | 3 | 2 | 5 | 4 | 4 |
| No network at render/view time | 5 | 3 (CDN unless inlined) | 3 | 4 | 5 | 3 | 5 | 5 | 5 |

## Decision
* Static figures: Matplotlib only (already core).
* Tables: library-independent `TableSpec` with hand-written HTML, Markdown, LaTeX and text renderers; no new dependency.
* Excel: XlsxWriter behind an optional extra `excel` (**needs approval**); openpyxl rejected.
* Interactive HTML: not in the MVP; a later ADR chooses between vendored JS over our own SVG and Plotly as an optional extra. Kaleido is excluded either way (browser requirement).

## Consequences
The MVP adds no runtime dependency. `import pprof_py` and all statistical APIs are unaffected. Great Tables can still be offered later as an optional renderer behind the same spec if its HTML ids become controllable.
