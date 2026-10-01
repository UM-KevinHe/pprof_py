# STATUS — pprof_py presentation layer

**Updated:** 2026-10-01 · **Branch:** `test/plotting` (base `d3d92a1` = `v0.5.0`) · **Home:** `pprof_py/docs/design/presentation/` (unpublished, not packaged)

## Phase
3 (MVP) in progress under decisions D9–D37. Rounds apply in strict order; each is one self-contained diff.

| Round | Content | State |
|---|---|---|
| R1 | Test workflow (3.10/3.14), Python ≥ 3.10, numba ≥ 0.57, docs workflow on 3.10 + Node 24 actions, changelog, design docs | delivered |
| R1b (optional) | One-line pandas-1.5 fix in `test_sigma_sensitivity` | delivered; applied by the maintainer |
| R2 | `pprof_py.presentation`: `Theme`, `formatting`, accessibility metrics; layering tests; lazy pyplot imports in `plotting/` | delivered |
| R3 | `pprof_py.inference.funnel_limits` and model methods; `degenerate_providers`; `at_bound` guard; excluded-provider records; CoxPH metadata; reference page *Funnel limits* | delivered |
| R4 | Presentation data: `ProviderProfile` (adapters `from_model`, `from_test`, `from_frame`), statuses, provenance, capabilities and `CapabilityError`, S3/S4 checks on user frames, minimum-volume rule | delivered |
| R5 | Funnel renderer and `FigureResult`: Figure API (no pyplot), theme `rc_context`, frozen layout, deterministic SVG/PDF/PNG, alt text, provenance footnote, dense mode | delivered |
| R6 | Interval plot (`caterpillar`) with volume panel: segments from `ci_lower` to `ci_upper`, legibility ordering, off-scale markers, labelled rows up to 60, dense mode; theme text without hinting | delivered |
| R7 | Tables: `TableSpec`, HTML/Markdown/LaTeX/text renderers, provider table, `excel` extra | next |
| R8 | Delegates, deprecations, acceptance, samples | pending |

## R6 evidence
- Drawn artists equal the profile: points and interval segments (clipped only at infinite bounds), rows in estimate order with ties in source order, volume bars equal the denominators row by row, reference at the test's null value; legend counts equal `status_counts()`.
- Visual QA [viewed]: 30 (notebook preset), 40, 50 and 58 providers (labelled; publication single and double width), 1,000 and 10,000 (unlabelled; rasterized above 2,000) with highlights; effect scale (logistic Wald and exact, linear) and ratio scale (`test_standardized`, CoxPH) with zero-event providers; grayscale. Fixed during QA: footnote overflow at low raster resolution (D37), colliding x labels at single width (two-line volume labels), Wald intervals of providers without a finite estimate drawn across the whole axis (D36), overlapping labels in the notebook preset (row height from the tick size, D34).
- Cost: 10,000 providers in 0.51 s plus 0.41 s for a 300-dpi PNG; SVG 0.43 MB (budgets 3 s and 2 MB).
- 14 new tests pass on Python 3.12 and on the 3.10 floor stack (with R5's, 26 figure tests). Mutations: reversed ordering fails 4 tests; misaligned volume bars fail 1. Full suite: 1 failed (`test_setup_logger`, environmental) / 825 passed / 1 skipped. Docs: 4 warnings (intersphinx).

## R5 evidence
- Drawn artists equal the profile: status points, zero-event markers, limit curves and per-provider marks match `ProviderProfile` (and so `funnel_limits`) exactly; legend counts equal `status_counts()`.
- Visual QA [viewed]: 10, 58, 100 (notebook preset, double width), 1,000 and 10,000 providers; score, exact (marks with Poisson references), grouped empirical null, linear Wald and CoxPH funnels; grayscale and simulated deutan, protan and tritan vision; Matplotlib 3.11.2 and 3.5.0. Fixed during QA: footnote overflow and squeezed data axes (sub-figures, D30), dashes on step-like curves (D31), the reference line crossing its label.
- Determinism: repeated saves and two interpreters give identical SVG, PDF and PNG bytes after freezing the layout (constrained layout drifted between saves, ADR-006); no pyplot import.
- Cost: 10,000 providers render in 0.20 s plus 0.21 s for a 300-dpi PNG; SVG 0.36 MB (budgets 3 s and 2 MB).
- 14 new tests pass on Python 3.12 and on the 3.10 floor stack (Matplotlib 3.5.0). Mutations: points drawn 0.01 % off fail 2 tests; swapped limit marks fail 1. Full suite: 1 failed (`test_setup_logger`, environmental) / 811 passed / 1 skipped. Docs: 4 warnings (intersphinx).

## R4 evidence
- Profiles equal their sources: every `test()` column, every funnel column (`FunnelLimits.providers`, curves), and the counts and denominators (`degenerate_providers`, `provider_sizes_`, CoxPH `observed`/`expected`/`person_time`). 18 of 18 profiles built from `test()` across families and methods (empirical nulls, one-sided, `critical`, both CoxPH tests) show no interval/flag disagreement, direction included.
- 20 new tests pass on Python 3.12 (pandas 3.0.6) and on the 3.10 floor stack (pandas 1.5.0); with R2's, 60 presentation tests on the floor stack. Mutations: estimates rounded to 6 decimals fail the equality contract; swapped above/below fail 4 tests.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 797 passed / 1 skipped. Docs: 4 warnings (intersphinx, as the baseline). Importing `pprof_py.presentation` loads no Matplotlib (tested).

## R3 evidence
- Validated outputs bit-identical to `v0.5.0`: 24 of 24 hashes (`test()` for logistic FE and RE, three-stage, linear FE and RE and CoxPH across methods, empirical nulls, subsets, one-sided tests, `critical`, seeded Monte Carlo; standardized measures; `DataPrep` with and without screening; `glmm_data_prep`). Only `attrs` change: CoxPH `measure` and `reference` (D13).
- S4: 0 disagreements and 0 providers on a limit in every configuration (six families; theoretical and empirical nulls; alternatives; `critical`; levels; subsets). 49 new tests pass on Python 3.12 (pandas 3.0.6) and on the 3.10 floor stack (pandas 1.5.0, numpy 1.23.0, scipy 1.9.0). Mutations: integer limits fail 16 tests; calibration-blind decisions fail the 2 tests that can detect them.
- Full suite: 1 failed (`test_setup_logger`, no tzdata in this container) / 777 passed / 1 skipped. Docs: 4 warnings (intersphinx, as the baseline); the new page's 9 code blocks print exactly their stored outputs, on both stacks.
- Cost at about 1,000 providers, over `test()`: at most +0.59 s for count tests, +0.01 s for score and Wald tests; `test(poibin_exact)` itself takes about 5.3 s (D24).

## R2 evidence
- Legacy outputs bit-identical to `v0.5.0`: 27 of 27 hashes (21 plot calls across four model families, `plot_caterpillar`, four `test()` outputs).
- `import pprof_py`: 1.52–1.54 s with pyplot → 1.14–1.20 s, loading neither pyplot nor Matplotlib.
- 40 new tests (formatting, theme, accessibility, layering). Layering negative control: planted violations fail 4 of 5 tests.
- Accessibility metrics vendored (Machado 2009 matrices reproduce colorspacious to 1e-16); thresholds in ADR-008.

## σ-sensitivity verification (D16)
On the Python 3.10 floor stack (numpy 1.23.0, pandas 1.5.0, scipy 1.9.0, statsmodels 0.13.1, numba 0.57.0) the stage-3 refits at both ends of the σ interval genuinely differ, identically to the latest stack: σ interval 0.189 / 0.319 / 0.567; 4 of 40 providers change flag across it. The test fails there only because its last line calls `DataFrame.to_numpy(float)` on a frame with a nullable `Int64` column containing NA, which pandas 1.5 rejects (`ValueError`); newer pandas maps NA to NaN. R1b passes `na_value=np.nan`.

## Reported, not fixed
- `dev` extra: `statsmodels>=0.13` cannot be installed from wheels on Python 3.10 (0.13.0 has no 3.10 wheel; its source build fails); 0.13.1 is the lowest installable. Raise it before adding a minimum-version job.
- Minimum-version job (D16): the floor stack passes everything except `test_setup_logger` (no tzdata in this container) and the R1b test.
- `test(test_method="poibin_exact")` takes about 5 s at 1,000 providers, almost all of it inverting the exact test for each provider's interval; a vectorised inversion could shorten it but would change validated code (not proposed).
- `test_standardized()` records neither `test_method` nor `reference` in its `attrs` (like CoxPH before D13), so displays say "test method not stated" and "reference not stated" for standardized-measure profiles; adding them is a statistical-layer change (needs approval).
- Statistical-layer observations from audit §9 remain: zero-event providers keep their flags (by design, D4); the score test sets z = 0 when the null variance is below 1e-14 (their funnel limits are infinite); the three-stage `LinAlgError` and `sigma_sensitivity` `ZeroDivisionError` on the audit's harness data were not investigated.

## Next step
R7: tables. A library-independent `TableSpec` (typed columns with roles, spanners, footnote markers, caption, source note, missing-value rules) and hand-written renderers for HTML (self-contained, `<caption>`, `<th scope>`, footnotes in `<tfoot>`), Markdown (GFM), LaTeX (booktabs, `longtable` above 40 rows) and plain text, plus Excel through XlsxWriter as the optional `excel` extra (D14); the provider table built from `ProviderProfile` with generated provenance and missingness footnotes (S2, S6) and the S12 rounding rule; golden-file tests and determinism.
