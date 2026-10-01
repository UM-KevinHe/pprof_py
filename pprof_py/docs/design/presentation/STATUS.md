# STATUS — pprof_py presentation layer

**Updated:** 2026-10-01 · **Branch:** `test/plotting` (base `d3d92a1` = `v0.5.0`) · **Home:** `pprof_py/docs/design/presentation/` (unpublished, not packaged)

## Phase
3 (MVP) in progress under decisions D9–D29. Rounds apply in strict order; each is one self-contained diff.

| Round | Content | State |
|---|---|---|
| R1 | Test workflow (3.10/3.14), Python ≥ 3.10, numba ≥ 0.57, docs workflow on 3.10 + Node 24 actions, changelog, design docs | delivered |
| R1b (optional) | One-line pandas-1.5 fix in `test_sigma_sensitivity` | delivered; applied by the maintainer |
| R2 | `pprof_py.presentation`: `Theme`, `formatting`, accessibility metrics; layering tests; lazy pyplot imports in `plotting/` | delivered |
| R3 | `pprof_py.inference.funnel_limits` and model methods; `degenerate_providers`; `at_bound` guard; excluded-provider records; CoxPH metadata; reference page *Funnel limits* | delivered |
| R4 | Presentation data: `ProviderProfile` (adapters `from_model`, `from_test`, `from_frame`), statuses, provenance, capabilities and `CapabilityError`, S3/S4 checks on user frames, minimum-volume rule | delivered |
| R5 | Funnel renderer and `FigureResult` (Figure API, theme `rc_context`, deterministic export, alt text, layout rules) | next |
| R6–R8 | Interval plot; tables (+ `excel` extra); delegates, deprecations, acceptance | pending |

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
- Statistical-layer observations from audit §9 remain: zero-event providers keep their flags (by design, D4); the score test sets z = 0 when the null variance is below 1e-14 (their funnel limits are infinite); the three-stage `LinAlgError` and `sigma_sensitivity` `ZeroDivisionError` on the audit's harness data were not investigated.

## Next step
R5: the funnel renderer and `FigureResult`. Matplotlib `Figure` objects built directly inside the theme's `rc_context` (no pyplot); per-provider limit marks for exact count tests and exact curves otherwise, labelled with test, level and null model; reference levels as labelled curves; zero-event providers at O/E = 0 with an outline marker (ADR-005); dense mode above 2,000 providers with flagged providers on top; provenance footnote and generated alt text; deterministic SVG/PDF/PNG; `CapabilityError` for profiles without funnel limits; render, determinism and accessibility tests.
