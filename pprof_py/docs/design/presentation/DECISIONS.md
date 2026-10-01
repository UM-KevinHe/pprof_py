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
