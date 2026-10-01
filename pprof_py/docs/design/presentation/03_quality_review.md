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
