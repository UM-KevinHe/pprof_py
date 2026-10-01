# Presentation layer: final report

## What was built

`pprof_py.presentation`, a presentation layer that turns validated results into figures, tables and reports without
recomputing them:

- **Figures** (12): funnel, interval plot with volume panel, observed versus expected, coefficient forest, data
  quality, null calibration, reliability, between-provider variation, shrinkage, flag stability, small multiples of
  several measures, and agreement of two measures. Each returns a `FigureResult` with deterministic SVG, PDF and PNG
  export, generated alt text and a footnote of the settings it depends on.
- **Tables** (9): provider results, covariate effects, several measures, data quality, null calibration, reliability,
  between-provider variation, shrinkage and flag sensitivity, exported to HTML, Markdown, LaTeX, text, Excel and
  DataFrames.
- **Reports**: one self-contained HTML file with sections, figures, tables and a provenance appendix.
- **Shared infrastructure**: an immutable `Theme` with three presets and accessibility checks, one formatter, one table
  specification, `ProviderProfile`, `CoefficientProfile` and `ProfileCollection`.
- **Statistical-layer additions** (each approved): `funnel_limits()` on every model, exclusion records, zero-event
  status, test and CoxPH metadata, and Wald intervals in logistic random-effect `summary()`.
- **Earlier plotting calls** draw through the new layer, with deprecations removed in 0.7.0.
- **Documentation**: a Presentation section with a build-time gallery, a spec sheet and interpretation page per
  display, a decision guide, a theme, export and accessibility guide, and a migration guide.

## Rounds

| Phase | Rounds | Diffs |
|:--|:--|:--|
| 3 (MVP) | R1 to R8 | `round_r1_ci_floor.diff` to `round_r8_delegates_acceptance.diff` |
| 4 (expand) | P4-1 to P4-11 | `round_p4_1_forest.diff` to `round_p4_11_report.diff` |
| 5 (docs) | P5-1 to P5-3 | `round_p5_1_gallery.diff` to `round_p5_3_guides.diff` |
| 6 (review) | P6-1, P6-2 | `round_p6_1_adversarial.diff`, `round_p6_2_review.diff` |

Every diff applied silently on a fresh clone of the maintainer's branch and reproduced the verified tree.

## Verification at the end

Full suite: 1 failed (`test_setup_logger`, environmental: the container lacks `tzdata`), 946 passed, 1 skipped; the
presentation tests also pass on the floor stack; the docs build has 4 warnings, the intersphinx baseline; 24/24
statistical outputs are bit-identical to v0.5.0 apart from approved metadata. The eight-area review passes
(`03_quality_review.md`).

## Decisions

D1 to D67 in `DECISIONS.md`: the maintainer approved the scope, the statistical-layer additions, the MVP gate, the
Tier 2 gate and the remaining order (D9 to D22, D38, D46, D47, D59); the others were made within Claude's scope and are
recorded with their rationale.

## Known limitations and open items

1. **Statistical layer, reported, not fixed:** `LogisticThreeStageModel.sigma_sensitivity()` raises
   `ZeroDivisionError` when sigma is estimated at 0 (reproduced in `test_capability_matrix.py`; flag stability now
   omits the sigma scenarios with a warning). The audit's other statistical observations are listed in STATUS.
2. **Environment:** statsmodels 0.13 has no wheels for the Python 3.10 floor (pre-existing).
3. **Not made (D59):** `iur_groups_` on `DirectIUR`; the reference specification in `test()` attributes.
4. **Removed in 0.7.0:** styling keywords of the earlier plotting calls, the `_legacy_*` drawing paths, the
   `lower`/`upper` column fallback of `plot_caterpillar`, and the replacement of a Wald funnel for logistic
   random-effect models.
5. **Not built:** interactive displays (all displays are static by design), Tier 3 displays (D59), PDF reports.
6. **Not verified in the build container:** the HTML report's page layout (no browser). Please open
   `p4_11_sample_report.html`.

## Recommended next steps

Open the sample report; decide on the `sigma_sensitivity()` fix with the statistician; schedule the 0.7.0 removals;
and, if wanted, the two optional statistical-layer additions.
