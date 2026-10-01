# Presentation layer: quality review (Phase 6)

Brief §3.6. Part 1 (round P6-1) is the adversarial review; part 2 (round P6-2) adds the eight-area review with
PASS/FAIL and evidence, the analyst-question check and the final report.

## Part 1: adversarial review

**Method.** Every display that applies, and the provider table and a report, were run on five hostile datasets fitted
with logistic fixed- and random-effect models (`/home/claude/p61/adversarial.py`, 60 calls in all). Each call's
errors and package warnings were recorded, and every figure was inspected on a contact sheet.

| Dataset | Construction | Outcome |
|:--|:--|:--|
| Tiny providers with extreme estimates | 30 providers of 11 to 14 records, some with all events or none, beside 5 of 300 to 600 | Honest: all-event and zero-event providers flagged in both funnels, marked at the interval plot's edge, "NE" in the multi-measure panel, not placed in the agreement plot; data quality accounts for all 35. Defect A. |
| Severe overdispersion | 80 providers, between-provider SD 1.2 | Honest: 55 of 80 flagged under the theoretical null, which the footnote names; null calibration shows a fitted SD of 4.51 and 54 flags falling to 6. Defect A. |
| Heavy-tailed outlier | One provider of 20,000 records with effect +1.5 among 99 | Honest; flag stability shows that the size-weighted mean reference, dominated by the outlier, makes 93 providers "below". Defects A, B, C, D. |
| Ties | 50 providers with identical size and identical event counts | Honest: null calibration reports its SD of 0.01, BLUPs are all 0. Defects D, E, F. |
| Many untested providers | A supplied frame with 37 of 60 flags missing | Honest: untested providers hollow and counted, their own volume strip; exclusions "not recorded"; settings "not stated". No defect. |

No call failed and no package warning was unexpected. No displayed statistical value was wrong; the six defects were
in layout or in degenerate-case handling:

| | Defect | Fix | Test |
|:--|:--|:--|:--|
| A | Observed versus expected: linear ticks crowded at the top of wide square-root axes, without thousands separators | Ticks evenly spaced in square-root space, with separators (`sqrt_ticks`) | `test_square_root_ticks_are_even_and_separated` |
| B | Log axes over many decades: 1-2-5 labels collided (funnel, data quality) | Beyond 2.3 decades only powers of ten are labelled, decided at draw time | `test_log_axes_over_many_decades_label_powers_of_ten` (fails without the rule) |
| C | Null calibration: one extreme statistic (z of about 100) flattened the histogram | The axis spans the median plus six robust SDs (at least 6); providers beyond it are counted at the edge and in the footnote. A first quantile-based bound failed with 60 providers, where the 99th percentile is interpolated from the maximum | `test_null_calibration_keeps_the_bulk_in_view` |
| D | Shrinkage: fixed label positions could land on points | Each label goes to the least crowded of several candidate positions | Visual check (heavy tail, ties) |
| E | Funnel: with all providers near the reference the y-range excluded the test's own upper limits | The y-range includes the test-level limits from the median precision up | `test_funnel_keeps_its_limits_in_view` |
| F | Between-provider variation: sigma estimated at 0 gave an infinite density peak (a count axis of 6.4e7) | No density or range is drawn; the panel and footnote say that no between-provider variation is detected | `test_variation_with_sigma_at_zero` |

After the fixes the 60 calls still succeed, the executed documentation prints its stored outputs, and the full suite
passes (see STATUS).

## Part 2: eight-area review

All evidence below was produced in round P6-2 against the final tree, except where a round is named.

| Area | Result | Evidence |
|:--|:-:|:--|
| Statistical correctness | PASS | Identity battery re-run: 24/24 statistical outputs bit-identical to v0.5.0; attribute changes are only the approved metadata (CoxPH `reference`, D26; `test_method`, `variance`, `reference` of `test_standardized()`, D38). Every display's tests compare drawn values with the test or summary outputs exactly; intervals agree with flags (S3) and funnel limits with flags (S4, also in counts and for hand-built limits); flag stability and null calibration compare `test()` calls without deciding flags (S9). Each round's deliberate bug was caught. |
| Visual quality | PASS | Every display viewed at its round; the gallery's 12 figures at 200 providers (P5-1); five adversarial datasets before and after fixes (P6-1); 10 to 50,000 providers (R5, R6, R8, P6-2). Limitation: the HTML report's page layout could not be viewed in the build container (no browser); its structure is tested. |
| API consistency | PASS | All 12 figure functions take `theme`, `size` and `title` and return a `FigureResult`; all 9 table functions return a `TableResult` with HTML, Markdown, LaTeX, text, Excel and DataFrame exports; one rendering path, including the earlier plotting functions (P4-10); deprecations warn now and are removed in 0.7.0. All 39 public names have docstrings. |
| Accessibility | PASS | Presets: status and line contrast at least 3.36:1 (3:1 required), text 17.4:1 (4.5:1); ΔE76 at least 40 between above and below and 25 against not different, normal and simulated colour vision; lightness gap 20.7 (15); minimum font 7 pt. Every figure has generated alt text and a long description (SVG `<title>`/`<desc>`, report `alt`/`aria-describedby`); tables use `<caption>` and `<th scope>`; statuses differ in shape, fill and label. |
| Performance | PASS | Measured in isolation (P6-2) against spec §14: interval plot render and PNG 0.56 s at 10,000 providers (budget 3 s) and 1.63 to 2.13 s at 50,000 (10 s); funnel 0.77 s at 10,000 (3 s); interval-plot SVG 0.7 MB at 10,000 (2 MB); provider table HTML 0.14 s and 1.75 MB at 10,000 rows (2 s, 5 MB); `funnel_limits()` exact count test 5.09 s over `test()` at 10,000 (10 s); `import pprof_py` 1.17 s against 1.55 s for v0.5.0, without importing pyplot. A first 50,000-provider timing of 13.9 s ran under `tracemalloc` while other jobs shared the CPU; repeated alone it is 1.6 to 2.1 s. |
| Dependency health | PASS | No new required dependency; XlsxWriter is the optional `excel` extra (3.0.1 floor); the presentation tests pass on the floor stack (Python 3.10, Matplotlib 3.5, pandas 1.5, XlsxWriter 3.0.1) and the latest; no pyplot on import; no network access in any output. Pre-existing: statsmodels 0.13 has no wheels for the Python 3.10 floor (reported in R1). |
| Maintainability | PASS | Layered package (`data`, `figures`, `tables`, `reports`, `theme`, `formatting`, shared provenance wording), 5,671 lines in 44 files; layering enforced by tests (no pyplot; tables import no figure code); pyflakes reports nothing; 25 presentation test files; decisions D1 to D67 and the ADRs record every choice; the 0.7.0 removals are listed in the migration guide. |
| Documentation | PASS | Docs build with 4 warnings (the intersphinx baseline), the gallery rendered at build time; executed examples print their stored outputs on both stacks; every display has a page with a complete spec sheet (test); the decision guide's family matrix is checked on fitted models (test); migration and theme guides; changelog entries for every round. |

## Part 3: analyst questions

Each question of brief §3.6 is answered with public API in at most three lines (`test_analyst_questions.py`
checks the answers and the line counts on both stacks):

| Question | Lines | Answer |
|:--|:-:|:--|
| Which providers look unusually high or low? | 2 | `funnel(profile)` and the profile's flagged rows |
| How uncertain are the estimates? | 2 | `caterpillar(profile)` and the intervals |
| How do providers compare with the benchmark? | 2 | The reference value and `status_counts()` |
| Are the extreme providers small or large? | 3 | Small: the ten most extreme O/E ratios have a median of 6.5 expected events against 15.9 overall |
| How much variation exists across providers? | 1 | `provider_variation()` and its table |
| How stable are the rankings? | 1 | No ranking is offered; `flag_stability()` and the changed counts |
| How do providers compare across measures? | 2 | `measure_agreement()` and `multi_measure_table()` |
| How should one estimate be read, given its uncertainty? | 1 | `caterpillar(..., highlight=)` and the provider's table row |
