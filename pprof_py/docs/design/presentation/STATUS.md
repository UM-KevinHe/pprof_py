# STATUS — pprof_py presentation layer

**Updated:** 2026-10-01 · **Branch:** `test/plotting` (base `d3d92a1` = `v0.5.0`) · **Home:** `pprof_py/docs/design/presentation/` (unpublished, not packaged)

## Phase
4 (expand) in progress under decisions D9–D49. Phase 3 (MVP, rounds R1–R8) was approved by the maintainer. Phase 4 rounds follow D49, one self-contained diff each, with a check-in after each round.

| Round | Content | State |
|---|---|---|
| P4-1 | Coefficient forest and coefficient table (`CoefficientProfile`); logistic RE `summary()` intervals (D47) | delivered |
| P4-2 | Data-quality panel and table (`data_quality`, `data_quality_table`) | delivered |
| P4-3 | Null-calibration diagnostics and table (`null_calibration`, `null_calibration_table`) | delivered |
| P4-4 | Observed versus expected (`observed_expected`) | delivered |
| P4-5 | Reliability display and table (`reliability`, `reliability_table`) | delivered |
| P4-6 | Between-provider variation | next |

Phase 3 rounds:

| Round | Content | State |
|---|---|---|
| R1 | Test workflow (3.10/3.14), Python ≥ 3.10, numba ≥ 0.57, docs workflow on 3.10 + Node 24 actions, changelog, design docs | delivered |
| R1b (optional) | One-line pandas-1.5 fix in `test_sigma_sensitivity` | delivered; applied by the maintainer |
| R2 | `pprof_py.presentation`: `Theme`, `formatting`, accessibility metrics; layering tests; lazy pyplot imports in `plotting/` | delivered |
| R3 | `pprof_py.inference.funnel_limits` and model methods; `degenerate_providers`; `at_bound` guard; excluded-provider records; CoxPH metadata; reference page *Funnel limits* | delivered |
| R4 | Presentation data: `ProviderProfile` (adapters `from_model`, `from_test`, `from_frame`), statuses, provenance, capabilities and `CapabilityError`, S3/S4 checks on user frames, minimum-volume rule | delivered |
| R5 | Funnel renderer and `FigureResult`: Figure API (no pyplot), theme `rc_context`, frozen layout, deterministic SVG/PDF/PNG, alt text, provenance footnote, dense mode | delivered |
| R6 | Interval plot (`caterpillar`) with volume panel: segments from `ci_lower` to `ci_upper`, legibility ordering, off-scale markers, labelled rows up to 60, dense mode; theme text without hinting | delivered |
| R7 | `test_standardized()` metadata (D38); tables: `TableSpec`, HTML/Markdown/LaTeX/text/Excel renderers, `provider_table`, `excel` extra | delivered |
| R8 | Model plot methods as delegates (deprecations for 0.7.0), `plot_caterpillar` defaults, `excel` in CI, docs page *Presentation layer (preview)* with migration table, acceptance (Chapter 10, logistic tutorial), demo and samples | delivered |

## P4-5 evidence
- Points equal `BootstrapIUR.group_sizes_` and `iur_groups_`; the curve joins them in size order; the overall line and n' marker equal `iur_` and `n_prime_`; tables equal the attributes, `decile_table()` and `SplitHalfIUR.summary()`; `DirectIUR` gets the table and a capability error for the figure.
- Visual QA [viewed]: 80 providers, reliability 0.14 to 0.76 by size, overall IUR 0.62 at n' = 152.
- 2 new tests pass on Python 3.12 and on the 3.10 floor stack. Mutation: reliabilities misaligned with sizes fail 1 test.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 890 passed / 1 skipped. Docs: 4 warnings (intersphinx); the Presentation page's examples print their stored outputs on both stacks.

## P4-4 evidence
- Points equal the profile's expected and observed counts; count marks equal the funnel limits times E (half-integer counts for exact tests) and agree with the flags in count space (S4); CoxPH count curves equal the funnel curves times their expected counts; Poisson reference curves of exact count tests are not drawn as limits.
- Visual QA [viewed]: logistic FE exact and score tests (58 providers, a zero-event provider at O = 0) and CoxPH (50 providers).
- 4 new tests pass on Python 3.12 and on the 3.10 floor stack. Mutation: lower limits left in ratio units fail 1 test.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 888 passed / 1 skipped. Docs: 4 warnings (intersphinx); the Presentation page's examples print their stored outputs on both stacks.

## P4-3 evidence
- Histograms equal `np.histogram` of each group's `z_raw` over the drawn edges; fitted densities equal `n * width * N(null_mean, null_sd)` exactly and peak at the null mean; flag counts and changes equal an independent cross-tabulation of the two `test()` results; the theoretical-null and profile paths say why there is no comparison.
- Visual QA [viewed]: 598 providers, empirical null in 3 size groups (SD 1.20 to 1.77; 78 flagged under the theoretical null, 23 under the fitted null).
- 5 new tests pass on Python 3.12 and on the 3.10 floor stack. Mutation: the fitted density drawn with SD 1 fails 1 test.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 884 passed / 1 skipped. Docs: 4 warnings (intersphinx); the Presentation page's examples print their stored outputs on both stacks.

## P4-2 evidence
- Accounting equals the profile's statuses and the model's exclusion record (in the data = analysed + excluded); unrecorded exclusions read "not recorded" in figure and table; bars, volume points per group (checked against an independent grouping), excluded providers' records and the minimum-volume line equal their sources; excluded providers stay off an axis in other units.
- Visual QA [viewed]: 58 providers with exclusions, a zero-event provider and suppression; 10,000 providers (the 910 providers without a finite estimate are all small).
- 6 new tests pass on Python 3.12 and on the 3.10 floor stack. Mutations: unrecorded exclusions shown as 0 fails 1 test; no-finite-estimate providers hidden among the analysed fails 1 (after adding the independent grouping check, which the first version lacked).
- Full suite: 1 failed (`test_setup_logger`, environmental) / 879 passed / 1 skipped. Docs: 4 warnings (intersphinx); the Presentation page's examples print their stored outputs on both stacks.

## P4-1 evidence
- `LogisticRandomEffectModel.summary()`: existing columns bit-identical to R8; the new interval equals `Estimate -/+ z * Std.Error` and excludes 0 exactly when `Pr(>|z|) < 1 - level` (levels 0.90, 0.95, 0.99).
- `CoefficientProfile` values equal each family's `summary()` columns (logistic FE and RE, linear FE and RE, CoxPH); exponentiation equals `exp` of the source exactly; forest segments, points, row order, reference line and text column equal the profile. Visual QA [viewed]: logistic odds ratios, linear RE with intercept, CoxPH hazard ratios (narrow range ticks).
- 15 new tests pass on Python 3.12 and on the 3.10 floor stack. Full suite: 1 failed (`test_setup_logger`, environmental) / 873 passed / 1 skipped. Docs: 4 warnings (intersphinx); the Presentation page's new example prints its stored output on both stacks.

## R8 evidence
- Delegates: 12 new tests (delegated methods return `FigureResult`s that unpack to `(fig, ax)`; logistic RE `plot_funnel` re-routed with a warning; legacy displays warn and still draw; styling keywords, `target=`, `use_flags=False` warn; `save_path` works; no pyplot import on delegated paths). `plot_caterpillar` reads `test()` frames by default and falls back to `lower`/`upper` with a warning.
- Docs that call plot methods (11 pages, 90 code blocks): printed output identical to R7 block by block, no exceptions. New page's code prints its stored output on both stacks.
- Acceptance, survival Chapter 10 (run from the chapters' own code): observed identical; expected within 4.3e-4 of the chapter's per-record computation (totals 259.000 vs 259.013); O/E within 8.6e-5; mid-p p-values within 1.1e-4; flags identical except facility 32, where the chapter's report had p = 0.044 with an interval containing 1 (unflagged) and the new layer, from one `CoxPH.test()`, gives p = 0.044 with the test's own interval 0.049-0.974 (flagged below); 0 interval/flag disagreements; funnel outside-the-limits providers = flagged providers. Logistic FE tutorial: its three delegated plot calls produce the new funnel and interval plots (titles kept).
- Demo (`demo_mvp.py`): 20, 1,000 and 10,000 providers in 1.2, 5.2 and 1.6 s for figures and table (the 1,000-provider exact intervals dominate).
- One existing test updated: `test_api_consistency.py::test_untested_providers_are_drawn_as_their_own_category` patched `test()` with a stub and read the legend from `plt.gca()`; it now wraps the real test (the funnel reuses its internals) and reads the returned figure's legend; its intent (an untested provider is drawn as "Not tested (1)") is unchanged.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 858 passed / 1 skipped; the floor stack passes all 121 presentation and metadata tests. Docs: 4 warnings (intersphinx).

## R7 evidence
- `test_standardized()` metadata: 24 of 24 outputs bit-identical to R6 (frames); only the standardized test's `attrs` gain `test_method`, `variance` and `reference`. Its default indirect-ratio statistic equals `test(test_method="score")` (z to 1e-9, identical flags), which the `"score"` label claims.
- Tables: golden HTML, Markdown, LaTeX and text files reviewed and pinned; HTML has `<caption>`, `<th scope>`, notes for every header marker, no scripts or links; LaTeX escapes specials including `<`, `>`, `|` and maps flags, minus and infinity; Excel cells numeric with formats, header frozen, notes sheet, byte-identical across time (XlsxWriter 3.0.1 and 3.2.9).
- 21 new tests pass on Python 3.12 and on the 3.10 floor stack. Mutations: the rounding-collision rule switched off fails 5 tests; Excel numbers written as text fail 1. Full suite: 1 failed (`test_setup_logger`, environmental) / 846 passed / 1 skipped. Docs: 4 warnings (intersphinx).

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
- CI installs `.[dev]`, so the Excel tests skip there (with a reason); adding `excel` to the workflow's install line would run them (CI change; needs approval).
- Statistical-layer observations from audit §9 remain: zero-event providers keep their flags (by design, D4); the score test sets z = 0 when the null variance is below 1e-14 (their funnel limits are infinite); the three-stage `LinAlgError` and `sigma_sensitivity` `ZeroDivisionError` on the audit's harness data were not investigated.

## Next step
STOP. After the maintainer's review of the rendered output: Phase 4 in the agreed order (each tier with the full checklist of brief §1), then Phase 5 (docs, gallery, migration guide) and Phase 6 (quality review).
