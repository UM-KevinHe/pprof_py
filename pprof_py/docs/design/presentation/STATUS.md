# STATUS — pprof_py presentation layer

**Updated:** 2026-10-01 · **Branch:** `test/plotting` (base `d3d92a1` = `v0.5.0`) · **Home:** `pprof_py/docs/design/presentation/` (unpublished, not packaged)

## Phase
6 (quality review) complete under decisions D9–D67. All phases delivered; awaiting the maintainer's review. Phase 3 (MVP, rounds R1–R8) was approved by the maintainer. Phase 4 rounds follow D49, one self-contained diff each, with a check-in after each round.

| Round | Content | State |
|---|---|---|
| P4-1 | Coefficient forest and coefficient table (`CoefficientProfile`); logistic RE `summary()` intervals (D47) | delivered |
| P4-2 | Data-quality panel and table (`data_quality`, `data_quality_table`) | delivered |
| P4-3 | Null-calibration diagnostics and table (`null_calibration`, `null_calibration_table`) | delivered |
| P4-4 | Observed versus expected (`observed_expected`) | delivered |
| P4-5 | Reliability display and table (`reliability`, `reliability_table`) | delivered |
| P4-6 | Between-provider variation (`provider_variation`, `provider_variation_table`); export determinism fix (D55) | delivered |
| P4-7 | Shrinkage display and table (`shrinkage`, `shrinkage_table`) | delivered |
| P4-8 | Flag stability display and table (`flag_stability`, `flag_stability_table`) | delivered |
| P4-9 | Multi-measure displays and table (`ProfileCollection`, `multi_measure`, `measure_agreement`, `multi_measure_table`) | delivered |
| Tier 2 check-in | Maintainer review of the Tier 2 displays | approved (D59) |
| P4-10 | Standalone `plot_funnel` and `plot_caterpillar` as delegates (D45, D60) | delivered |
| P4-11 | HTML report object (`Report`) | delivered |
| P5-1 | Presentation docs section and build-time gallery (`_ext/pprof_gallery.py`, `_synthetic.provider_data`); agreement-plot fix (D63) | delivered |
| P5-2 | Interpretation pages, one per display, and the tables page (D64) | delivered |
| P5-3 | Decision guide (with a tested family matrix); theme, export and accessibility guide; migration guide (D65) | delivered |
| P6-1 | Adversarial review: five hostile datasets, six defects fixed with tests (`03_quality_review.md`, D66) | delivered |
| P6-2 | Eight-area review (all PASS), analyst-question test, final report (`03_quality_review.md`, `04_final_report.md`, D67) | delivered |

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

## P6-2 evidence
- Eight-area review: all PASS (`03_quality_review.md` part 2) — identity battery 24/24; performance budgets met in isolation (interval plot 1.6–2.1 s at 50,000; `funnel_limits` +5.1 s at 10,000; import 1.17 s vs 1.55 s for v0.5.0); API introspection; preset accessibility; pyflakes clean.
- Analyst questions: eight answers in one to three lines each (`test_analyst_questions.py`, both stacks).
- Full suite: 1 failed (`test_setup_logger`, environmental) / 946 passed / 1 skipped. Docs: 4 warnings (intersphinx).

## P6-1 evidence
- 60 calls (every applicable display, the provider table and a report) on five hostile datasets: no failure, no unexpected warning, no wrong value; six layout and degenerate-case defects (A to F in `03_quality_review.md`) fixed, re-checked on contact sheets [viewed], and covered by `test_adversarial.py` (5 tests, both stacks; the decade rule's negative control fails). A first robust bound for null calibration failed the test with 60 providers and was replaced by a MAD-based bound.
- After the fixes: the 60 calls succeed; the executed documentation pages print their stored outputs.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 944 passed / 1 skipped (cold-interpreter determinism included). Docs: 4 warnings (intersphinx); gallery regenerated.

## P5-3 evidence
- Decision guide: every cell of its family matrix exercised on fitted models (supported cells draw, refused cells raise with a message); encoded as `test_capability_matrix.py`. The guide's first footnote was corrected: logistic RE funnels default to `poibin_exact`, and `test_method="wald"` is refused by `funnel_limits()`.
- Theme guide: its executed example (preset sizes, widths, a derived theme's accessibility report, a pale colour caught at 1.42:1) prints its stored output on both stacks. Migration table moved intact (15 rows); the Presentation page's examples unchanged on both stacks.
- Found: `LogisticThreeStageModel.sigma_sensitivity()` raises `ZeroDivisionError` on data with three sub-clusters per provider (audit §9, item 4); flag stability now reports it instead of failing (test added). The statistical-layer cause is not fixed.
- Docs build: exit 0, 4 warnings (the baseline). Full suite: 1 failed (`test_setup_logger`, environmental) / 939 passed / 1 skipped.

## P5-2 evidence
- 11 display pages (spec sheet, reading, misreadings, call) and a tables page; built pages carry 12 figures, all with generated alt text; docs build exit 0 with 4 warnings (the baseline).
- Executed counterexamples (funnel against a league table, ends of the interval-plot ordering, null calibration) print their stored outputs on Python 3.12 and on the 3.10 floor stack; the first funnel claim was false for the data (all ten highest ratios were flagged) and was rewritten to what the output shows.
- New test: every gallery figure has a page with all eight spec-sheet items and both interpretation sections; renaming one item fails it.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 928 passed / 1 skipped. Docs: 4 warnings (intersphinx).

## P5-1 evidence
- Docs build with the gallery: exit 0, 4 warnings (intersphinx, the baseline), 56 s (+6 s); 12 figures and the include file generated; a second build rewrites nothing. Gallery figures checked as a contact sheet [viewed] at 200 providers; the agreement plot's axes were set by a solver-bound interval (fixed, D63).
- 3 new tests (generator determinism and planted structure; gallery smoke) and 1 agreement test pass on Python 3.12 and on the 3.10 floor stack.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 927 passed / 1 skipped. Docs: 4 warnings (intersphinx).

## P4-11 evidence
- Report: each image decodes exactly to its figure's SVG with its alt text and long description; table markup is embedded unchanged with its number; the appendix rows follow the figures and tables with their recorded test and model; the disclosure note appears only with tables; no scripts, links or external URLs; unique ids and resolving ARIA references; text escaped; rebuilds and saved files byte-identical. `render_html` output unchanged (golden files).
- Visual QA: the embedded figures are the renders viewed in earlier rounds; the HTML page itself could not be rendered in this container (no browser), so its layout was checked structurally only.
- 3 new tests pass on Python 3.12 and on the 3.10 floor stack. Mutation: unescaped report text fails 1 test.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 923 passed / 1 skipped. Docs: 4 warnings (intersphinx); the Presentation page's examples print their stored outputs on both stacks.

## P4-10 evidence
- `plot_funnel`: supplied curves drawn exactly as given per level; each provider's marks equal an independent interpolation of the 95% curve; contradicting flags warn (S4); styling keywords warn, unknown ones raise; `ax=` keeps the earlier drawing with a warning; `save_path` writes. `plot_caterpillar`: intervals and reference equal the frame; R8's `lower`/`upper` fallback and its message unchanged; legacy-only options warn. Model mixins still call the `_legacy_*` functions.
- Visual QA [viewed]: hand-built funnel (60 providers, 95% and 99.8% curves) and a `test()` frame interval plot.
- 4 new tests pass on Python 3.12 and on the 3.10 floor stack (Matplotlib 3.5's pyparsing deprecations ignored as library-internal); R8's delegate tests unchanged and passing. Mutation: limits read off the 99.8% curve fail 1 test. Docs pages calling these functions print their stored outputs.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 920 passed / 1 skipped. Docs: 4 warnings (intersphinx).

## P4-9 evidence
- Collection union and intersection; small-multiple points and segments equal each measure's profile in the common row order, with "n/a" for each provider a measure lacks; agreement points equal both estimates for the providers in both, and joint counts equal an independent cross-tabulation of the two tests' flags; table cells, flattened headers and values equal the profiles; golden files of earlier tables unchanged.
- Visual QA [viewed]: two correlated measures of 40 providers (2 missing from one), small multiples and agreement; fixed a dash for missing providers that read as a short interval.
- 4 new tests pass on Python 3.12 and on the 3.10 floor stack. Mutation: same and opposite joint status swapped fails 1 test.
- The cold-interpreter determinism test now covers all 10 figure types (2 added here). Full suite: 1 failed (`test_setup_logger`, environmental) / 916 passed / 1 skipped. Docs: 4 warnings (intersphinx); the Presentation page's examples print their stored outputs on both stacks.

## P4-8 evidence
- Every scenario column equals its own `test()` call (base, alternative reference, empirical null, custom scenarios); three-stage sigma columns equal `sigma_sensitivity()`; CoxPH gets no reference scenario; drawn cells, row order and change markers equal the flag matrix; table counts and changes equal independent computations; change detection treats "not tested" as a status (unit test).
- Visual QA [viewed]: logistic FE (58 providers, 3 scenarios, 6 changing under the empirical null) and the three-stage model (30 providers, 5 scenarios including sigma's bounds).
- 5 new tests pass on Python 3.12 and on the 3.10 floor stack (a pandas 1.5 object-array issue fixed with explicit boolean conversion). Mutation: "not tested" treated as "not different" fails 1 test (after adding the unit test; the first version could not detect it).
- Full suite: 1 failed (`test_setup_logger`, environmental) / 910 passed / 1 skipped. Docs: 4 warnings (intersphinx); the Presentation page's examples print their stored outputs on both stacks.

## P4-7 evidence
- Pairs equal each test's estimate minus its null value (fixed effects with `reference="mean"`; BLUPs against 0); volumes equal the records per provider (checked against the data); drawn points, edge markers, diagonal and equal limits equal the pairs; the table's change equals random minus fixed, with NE where the fixed effect is not finite; mismatched or swapped families raise.
- Visual QA [viewed]: logistic (70 providers, one zero-event provider at the edge) and linear pairs; labels placed in the regions shrinkage leaves empty.
- 3 new tests pass on Python 3.12 and on the 3.10 floor stack. Mutation: fixed effects not centred on their reference fail 1 test.
- Full suite: 1 failed (`test_setup_logger`, environmental) / 905 passed / 1 skipped. Docs: 4 warnings (intersphinx); the Presentation page's examples print their stored outputs on both stacks.

## P4-6 evidence
- Variation: the SD, interval and BLUPs equal `sigma_`/`profile_sigma()` (logistic) and `random_effect_sd_` (linear, not the residual `sigma_`); the histogram equals the BLUPs binned over the drawn edges; fitted and bound densities equal n * width * N(0, s) exactly; the range equals -/+ z sigma and its odds ratios exp of it. Visual QA [viewed]: logistic RE (sigma 0.53, profile interval 0.40-0.71) and linear RE (no interval). Mutation: reading the linear residual SD fails 2 tests.
- Determinism (D55): a figure built after earlier drawing could differ from a cold build (SVG ids; "-0" in PDF), a latent defect since R5. Fixed by quantised frozen layouts and canonical SVG ids; mismatch rates 5/12 (reliability SVG) and 2/3 (observed-versus-expected PDF) fell to 0/12 and 0/10. New test: every figure type cold in its own interpreter, SVG, PDF and PNG.
- 12 new tests (4 variation, 8 cold-interpreter determinism) pass on Python 3.12 and on the 3.10 floor stack; the determinism test passed 4 of 4 repeated runs (32 cold interpreters) and the presentation tests 3 of 3 runs. Full suite: 1 failed (`test_setup_logger`, environmental) / 902 passed / 1 skipped. Docs: 4 warnings (intersphinx); the Presentation page's examples print their stored outputs on both stacks.

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
- `LogisticThreeStageModel.sigma_sensitivity()` raises `ZeroDivisionError` when sigma is estimated at 0 (reproduced in `test_capability_matrix.py`); flag stability omits the sigma scenarios with a warning.
- Statistical-layer observations from audit §9 remain: zero-event providers keep their flags (by design, D4); the score test sets z = 0 when the null variance is below 1e-14 (their funnel limits are infinite); the three-stage `LinAlgError` and `sigma_sensitivity` `ZeroDivisionError` on the audit's harness data were not investigated.

## Next step
None planned: all phases are delivered. The maintainer's review of `04_final_report.md` and the sample report decides what follows.
