# ADR-001 — Layering and package layout

**Status:** accepted (Claude, module structure) · 2026-09-30

## Context
`plotting/` today mixes a style module, two standalone renderers and four model mixins that are base classes of the models, so `import pprof_py` imports `matplotlib.pyplot` (audit R4). Tables and reports have no home. The brief requires one home per concern, no duplicated style constants, old import paths that keep working, and a one-way import direction enforced by a test (§8.1–8.2).

## Options
A. Evolve `plotting/` into the home (add `plotting/tables/`, `plotting/reports/`). Fewest moves, but "plotting" would hold tables and reports, and the model mixins stay entangled.
B. **New `pprof_py/presentation/` package** (`data/`, `formatting/`, `theme/`, `figures/`, `tables/`, `reports/`), with `pprof_py.plotting` kept as a façade that re-exports the old functions and `style` constants from the new home.
C. A separate distribution (`pprof_py-viz`). Cleanest dependency story, but premature and splits docs and CI.

## Decision
B. Model mixins become thin delegates whose module-level imports contain no Matplotlib; renderers are imported inside the method bodies.

## Consequences
* `import pprof_py` stops importing pyplot; a test asserts `"matplotlib.pyplot" not in sys.modules` after import.
* A second test parses imports of `models/`, `algorithms/`, `inference/`, `measures/` and fails on any module-level import of `presentation` or `plotting`.
* `pprof_py.plotting.style` becomes a re-export of `presentation.theme` tokens; no second copy of any constant.
