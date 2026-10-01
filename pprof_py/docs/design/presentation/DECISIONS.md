# Decision log — pprof_py presentation layer

Newest last. "Maintainer" decisions were given explicitly; "Claude" decisions fall inside the brief's §3.2 "decide yourself" scope and can be overturned at any gate.

| ID | Date | Decision | By | Implication | Record |
|---|---|---|---|---|---|
| D1 | 2026-09-30 | Funnel limits come from an accessor in the statistical/result layer | Maintainer | New `funnel_limits` accessor (API to confirm, spec §7); presentation never computes limits | ADR-003 |
| D2 | 2026-09-30 | No generic funnel for random-effect models; only methodologically justified cases | Maintainer | Funnels only where the flagging test is a test of the plotted count; Wald-on-BLUP fits get a capability error and the interval plot | ADR-004 |
| D3 | 2026-09-30 | Establish test CI early | Maintainer | `round_ci_tests.diff` (tests workflow) delivered with this phase | ADR-007 |
| D4 | 2026-09-30 | Zero-event providers are shown explicitly; no silent continuity correction | Maintainer | Explicit status and encoding; never draw the solver's clamp as an estimate; flags unchanged | ADR-005 |
| D5 | 2026-09-30 | New `pprof_py.presentation` package; `pprof_py.plotting` kept as a backward-compatible façade | Claude | One home for figures, tables, reports; old imports keep working | ADR-001 |
| D6 | 2026-09-30 | Renderers build `matplotlib.figure.Figure` directly (no pyplot); the theme's `rc_context` wraps build *and* save | Claude | Nothing left open; byte-identical exports without global state (verified) | ADR-006 |
| D7 | 2026-09-30 | Status encodings: hue + shape + fill + label, dark PuOr pair for above/below | Claude | Thresholds become accessibility tests | ADR-008 |
| D8 | 2026-09-30 | Tables use hand-written renderers behind a library-independent spec (no new dependency) | Claude | Deterministic HTML/Markdown/LaTeX/text; Excel extra pending approval | ADR-002 |
| D9 | 2026-10-01 | MVP approved: funnel, interval plot with volume panel, provider table, shared infrastructure, in eight rounds | Maintainer | Rounds R1–R8 as in spec §16 | spec §16 |
| D10 | 2026-10-01 | `funnel_limits` as a function in `pprof_py.inference` and as methods on each supported model | Maintainer | Round R3 | ADR-003 |
| D11 | 2026-10-01 | Discrete limits at half-integer boundaries; per-provider limit marks for exact Poisson-binomial tests; score funnel is the logistic FE default | Maintainer | No provider on a line; exact funnels draw marks | ADR-003 |
| D12 | 2026-10-01 | Deprecation: warnings now, removal in `0.7.0`; random-effect `plot_funnel` changed accordingly | Maintainer | Round R8 | ADR-004, spec §13 |
| D13 | 2026-10-01 | MVP statistical-layer additions: zero-event status, excluded-provider records, CoxPH measure/reference metadata. Logistic RE `summary()` intervals later unless the MVP needs them | Maintainer | Round R3 | spec §7, ADR-005 |
| D14 | 2026-10-01 | Excel through XlsxWriter as the optional `excel` extra; interactive HTML deferred until after the MVP | Maintainer | Round R7 | ADR-002 |
| D15 | 2026-10-01 | Minimum Python 3.10 | Maintainer | Round R1: `requires-python`, README, CI matrix 3.10/3.14, docs workflow | ADR-007 |
| D16 | 2026-10-01 | Minimum-version CI job deferred; first raise the numba floor to 0.57 and verify the σ-sensitivity test | Maintainer | Round R1 (numba 0.57); verification reported in STATUS | ADR-007 |
| D17 | 2026-10-01 | No bundled fonts; Matplotlib's DejaVu Sans | Maintainer | Theme default | ADR-006 |
| D18 | 2026-10-01 | No rankings in the MVP | Maintainer | Displays order by estimate for legibility only | brief §5.4 |
| D19 | 2026-10-01 | House style: clean, modern, restrained, publication-quality scientific style; uncertainty, readability, accessibility, consistent typography and spacing, minimal chartjunk, clear provider-volume context | Maintainer | Theme tokens and renderer defaults (R2, R5, R6) | spec §4 |
| D20 | 2026-10-01 | Design documents in `pprof_py/docs/design/presentation/`; branch `test/plotting`; one diff per round | Maintainer | Round R1 adds this folder | — |
| D21 | 2026-10-01 | Tables show "not different" as ● (the figures' marker); ○ means "not tested" in figures and NT in tables | Claude | One symbol set across figures and tables | ADR-008, spec §3.4 |
| D22 | 2026-10-01 | Accessibility tests use CIELAB ΔE (CIE76) with vendored Machado (2009) matrices; thresholds 40 / 25 / ΔL\* 15 | Claude | No test dependency; vendored simulation matches colorspacious to 1e-16 | ADR-008 |
| D23 | 2026-10-01 | `funnel_limits` reuses `test()`'s internals through an inert context-variable recorder (count test, logistic score path, linear Wald df) instead of re-deriving them | Claude | No duplicated argument handling; validated outputs bit-identical (24/24) | ADR-003 |
| D24 | 2026-10-01 | `FunnelLimits` carries the unchanged `test()` result; the count-test budget becomes "≤ 1 s over `test()`" | Claude | Displays test once; `test()`'s own exact-interval cost (≈ 5 s at 1,000 providers) is outside R3 | ADR-003, spec §14 |
| D25 | 2026-10-01 | `excluded_providers_` is `None` when a model did not prepare its data and an empty record when preparation removed nobody | Claude | Displays can tell "unknown" from "none excluded" | ADR-005 |
| D26 | 2026-10-01 | CoxPH `exact` tests get funnel limits too (same Poisson search with the test's own statistic) | Claude | Both CoxPH methods have exact limits and curves | ADR-003 |
| D27 | 2026-10-01 | Profile denominators and counts come only from test-consistent sources: the funnel limits of the same test, the CoxPH test's columns, or the model's own data (`degenerate_providers`, `provider_sizes_`); not from `calculate_standardized_measures()`, whose expected counts use their own reference | Claude | Tables cannot show an E that disagrees with the test | spec §6.2 |
| D28 | 2026-10-01 | `ProfileCollection` waits for the multi-measure displays (Phase 4); no MVP display needs it | Claude | Smaller R4 | spec §6.1 |
| D29 | 2026-10-01 | "No interval" is an attribute (`has_interval`), not a primary status, so a flagged provider without an interval keeps its flag; excluded providers stay in the profile's `excluded` record rather than as rows | Claude | One flag-based status per provider; S6 counts from `status_counts()` | spec §6.1, ADR-005 |
| D30 | 2026-10-01 | Figures stack three sub-figures (data, legend, footnote) sized from measured text, and run constrained layout once, then freeze it | Claude | Legend and footnote never squeeze the data axes; repeated saves are byte-identical (constrained layout is not idempotent across draws) | ADR-006, spec §4 |
| D31 | 2026-10-01 | Continuous limit curves are dashed by level; discrete count curves (Poisson references, CoxPH) are thin solid lines, darker at the test's level; footnotes describe limits without naming dash styles | Claude | Dashes are illegible on step-like limits; levels stay directly labelled and every line meets 3:1 contrast | ADR-008 |
| D32 | 2026-10-01 | Funnel labels: `highlight=` providers always; flagged providers automatically only when at most 10 (the spec suggested 30) and the plot is not dense | Claude | No label collisions at 85 mm width | spec §2.1 |
| D33 | 2026-10-01 | Limit marks are called "exact" only when they come from an exact count test; limits supplied with a frame are labelled as supplied | Claude | No claim the source does not support | spec §2.1 |
| D34 | 2026-10-01 | Interval plots label provider rows up to 60 providers (the spec suggested 150), with row height from the theme's tick size; above 60 the rows are unlabelled and `highlight=` providers are annotated | Claude | 150 labelled rows would make a figure about 45 cm tall; labels never overlap in any preset (tested) | spec §2.2 |
| D35 | 2026-10-01 | Interval plots draw the test's own scale; a standardized-measure scale comes from `test_standardized()` (`ProviderProfile.from_test(...)`), not from exponentiating in the renderer | Claude | No presentation-side transformation in the MVP | spec §2.2, S1 |
| D36 | 2026-10-01 | A provider without a finite estimate gets an interval segment only when exactly one bound lies inside the axis (drawn from the edge to that bound); Wald intervals at the solver bound are not drawn | Claude | ADR-005's exact one-sided intervals are shown; meaningless full-width segments are not | ADR-005 |
| D37 | 2026-10-01 | Themes set `text.hinting: no_hinting` | Claude | Raster hinting widened a footnote by 8 % at 110 dpi; text now has its outline width at every resolution (168.4 ± 0.1 mm from 72 to 300 dpi) | ADR-006 |
