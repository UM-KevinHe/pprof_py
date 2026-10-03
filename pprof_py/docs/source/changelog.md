(changelog)=
# Changelog

```{note}
This changelog is reconstructed from git history (commit messages,
dates, and the `v0.1-legacy` tag) and from `pprof_py.__version__`.
Treat specific dates as accurate (taken from commit timestamps) and
feature attributions as a best reconstruction. There is no released
`0.3.0`: `pyproject.toml` goes from `0.2.0` directly to `0.4.0`.
```

## Unreleased

The presentation layer takes its visual identity (design approved 2026-10-02). Statistical results are unchanged.

### Changed
- The `publication`, `notebook` and `report` themes set text in IBM Plex Sans, mark status with a copper and petrol
  pair (`#C25E1F`, `#0F4C63`; not different `#7D8793`), draw a light grid with muted tick labels instead of a frame,
  set axis labels in medium and titles in semibold weight, and draw the 95% limits solid and the 99.8% limits dotted.
  Every figure looks different; every value, flag and limit is the same.
- `funnel()` in the new look: where the test's curve reproduces the flags, the region between its limits is shaded
  (with a "Not flagged at 95%" key entry); filled markers have a white halo; the key sits above the plot, under a
  title row when there is a title; `highlight=` providers get a ring and a semibold label; the publication preset
  shows a one-line footnote and keeps the full text in `FigureResult.caption`. `theme="classic"` is unchanged.
- `caterpillar()` in the new look: intervals are bars whose rounded ends sit exactly on their bounds, with each
  estimate's mark on top in a darker shade and a white halo; the volume panel uses thin bars; the key sits above the
  plot, under a title row; highlighted providers get a band across both panels and a semibold label (a ring when
  rows are unlabelled); the publication preset shows a one-line footnote. `theme="classic"` is unchanged.
- The other ten displays (observed against expected, forest, data quality, null calibration, reliability, provider
  variation, shrinkage, flag stability, several measures, measure agreement) take the same look: the key above the
  plot under a title row, white halos on filled marks, a ring and a semibold label for `highlight=` providers (a band
  in several measures), and a one-line publication footnote with the full text in `FigureResult.caption` (data
  quality and null calibration keep their full footnote, which defines what they draw). The forest and
  several-measures plots draw intervals as bars whose rounded ends sit exactly on their bounds. `theme="classic"` is
  unchanged.
- HTML tables take their style from the theme's tokens: IBM Plex Sans named first in the font stack (never embedded),
  hairline rows, muted headers, and each status as a coloured glyph followed by its word. `theme="classic"` gives the
  0.6.0 HTML; Markdown, LaTeX, text and Excel output is unchanged. Reports keep the 0.6.0 table style until their
  own redesign.

### Added
- IBM Plex Sans ships as package data (four unmodified styles, SIL Open Font License 1.1, licence and copyright
  notice in `presentation/theme/fonts/OFL.txt`). Fonts load per text element, without changing Matplotlib's font
  manager or rcParams, and fall back to DejaVu Sans when the files are unavailable or a glyph is missing.
- `Theme.classic(variant)` and `theme="classic"`: the 0.6.0 presets, byte-identical in the same environment.
- Theme tokens `corridor`, `halo`, `spines`, `tick_length`, `tick_label_color`, and `Typography.title_weight` and
  `label_weight`.
- `FigureResult.caption`: the figure's footnote text, for outlets that set captions outside the figure.
- Theme tokens `key_position`, `footnote` and `highlight_ring`.
- `provider_table(..., theme=, intervals=, group_by=)`: the HTML style, an inline interval column in HTML, and rows
  grouped by the test's result under header rows with counts in every format. `TableResult.theme`, the theme token
  `table_style`, `tables.table_css(theme)`, and `TableSpec.groups`.
- `caterpillar(..., group_by="status")`: rows grouped by the test's result under headers with counts, for analyst
  reports. Theme tokens `interval_bars`, `bar_marks`, `volume_half`, `highlight_wash`, `Lines.interval_bar` and
  `Lines.interval_bar_dense`; `accessibility_report()["bar_mark_contrast"]` (3:1 required).
- `Theme.accessibility_report()["corridor_contrast"]`: status marks against the corridor fill (3:1 required).

## 0.6.0 (2026-10-01)

The presentation layer: figures, tables and reports that show validated results without recomputing them, with
funnel limits from the same test as the flags, and documentation on reading each display; plus the provider-test
additions it rests on. Statistical results are unchanged from 0.5.0: estimates, tests and intervals are
bit-identical. This release requires Python 3.10 or newer, and the earlier plotting calls now draw through the
presentation layer; their deprecated options are removed in 0.7.0.

### Migrating from 0.5.0

- Python 3.10 or newer is required, and `numba` 0.57 or newer.
- The plotting methods and the standalone `pprof_py.plot_caterpillar` and `pprof_py.plotting.plot_funnel` return a
  `FigureResult`, which still unpacks as `fig, ax`. `plot_caterpillar` no longer returns `None` or calls
  `plt.show()`: display the result in a notebook, or write it with `.save(path)`.
- Styling keywords of the plotting calls are ignored with a `DeprecationWarning`; pass `theme=` (see
  `Theme.derive`). They raise in 0.7.0.
- `LogisticRandomEffectModel.plot_funnel()` draws the funnel of the exact count test. The linear random-effect
  `plot_funnel()` and `plot_standardized_measures()` of models other than the logistic fixed-effect model keep their
  earlier drawing with a `DeprecationWarning`, and are removed in 0.7.0.
- `plot_caterpillar()` reads `ci_lower` and `ci_upper` by default; `lower` and `upper` are used with a warning until
  0.7.0.
- The full table of earlier calls and their replacements is in the migration guide (Presentation → Migrating to the
  presentation layer).

### Requirements and CI

- **Python 3.10 or newer.** Python 3.9 reached end of life in October 2025, and `fast_poibin` 0.4.2 already requires
  3.10. The `numba` floor is now 0.57, the lowest version `fast_poibin` 0.4 accepts (`numba>=0.56` could not be
  installed together with it).
- **Test workflow.** `.github/workflows/tests.yml` runs the test suite on Python 3.10 and 3.14 for pushes to `main`,
  pull requests and manual runs. The documentation workflow builds on Python 3.10.
- The test workflow also installs the `excel` extra, so the Excel output tests run in CI.

### Provider tests: funnel limits, zero-event status and exclusions

- **Funnel limits that agree with the flags.** `pprof_py.inference.funnel_limits(model, ...)`, and a `funnel_limits`
  method on `LogisticFixedEffectModel`, `LogisticRandomEffectModel`, `LogisticFERandomClusterModel`,
  `LogisticThreeStageModel`, `LinearFixedEffectModel` and `CoxPH`, return a `FunnelLimits`: the `test()` result, each
  provider's funnel coordinates and control limits, and limit curves. The limits come from the same test, null and
  decision rule as the flags, so a provider lies outside its limits exactly when it is flagged; count tests place
  their limits half-way between counts, so no provider lies on a line. Logistic fixed-effect funnels use the score
  test by default. Random-effect models have funnels for their count tests (`poibin_exact`, `exact`) only; linear
  random-effect models and Monte Carlo tests have none. See the new reference page *Funnel limits*.
- **Zero-event status.** `pprof_py.inference.degenerate_providers(model)` reports, for binary-outcome models, each
  provider's events and trials, whether it has no events or only events, and whether its effect has a finite
  estimate.
- **`at_bound()` refuses models it does not apply to.** It raises an informative `TypeError` for models without fixed
  provider effects (random-effect models raised `KeyError: 'gamma'`) and for continuous outcomes (it returned every
  provider of a linear model). Results for logistic fixed-effect models are unchanged.
- **Excluded providers are recorded.** `DataPrep.excluded_providers_`, `GLMMPreparedData.excluded_providers`, and the
  fitted `excluded_providers_` of `LogisticFixedEffectModel`, `LinearFixedEffectModel` and `LogisticThreeStageModel`
  list the providers that data preparation removed, with their record counts and the reason; `None` when the model
  did not prepare the data.
- **CoxPH test metadata.** `CoxPH.test()` records `attrs["measure"] = "indirect_ratio"` and `attrs["reference"] = 1.0`.
- **Random-effect coefficient intervals.** `LogisticRandomEffectModel.summary(level=0.95)` adds the Wald interval
  `ci_lower`, `ci_upper`; the existing columns are unchanged, and an interval excludes 0 exactly when the p-value is
  below `1 - level`.
- **Standardized-test metadata.** `LogisticFixedEffectModel.test_standardized()` records `attrs["test_method"]`
  (`"score"` for indirect measures tested with the null variance on the identity scale, which is the score
  statistic; `"wald"` otherwise), `attrs["variance"]` and `attrs["reference"]` (the reference effect gamma_0, as in
  `test()`).
- No estimate, test result or interval changes: the count-test kernels were split into reusable parts with identical
  arithmetic.

### Presentation layer

- `import pprof_py` no longer imports `matplotlib.pyplot`, or Matplotlib at all: the plotting functions import it
  when they are called, and their output is unchanged. Importing the package is about a quarter faster.
- New provisional namespace `pprof_py.presentation`: `Theme` (immutable design tokens with `publication`,
  `notebook` and `report` presets, varied through `Theme.derive`) and `formatting` (numbers, counts, intervals,
  p-values and flags, with one set of missing-value symbols). The namespace grows with the presentation layer and
  may change until it is complete.
- `pprof_py.presentation.ProviderProfile`: one provider test ready for display, built from a fitted model
  (`from_model`, optionally with the funnel limits of the same test), a `test()` result (`from_test`) or any frame
  with a role map (`from_frame`). It keeps the test's values unchanged and gives each provider one status from its
  flag (above, below, not different, not tested; suppressed only under an explicit minimum-volume rule), with zero
  events and no finite estimate as separate attributes. Denominators, exclusions and the test's settings travel with
  it, and `require()` raises `CapabilityError` when a display needs something the profile lacks. Frames whose
  intervals or funnel limits contradict their flags trigger a warning that names the providers.
- `pprof_py.presentation.funnel(source, ...)`: a funnel plot whose control limits come from the same test as the flags
  (through `funnel_limits`), so a provider lies outside its limits exactly when it is flagged. Score, Wald and CoxPH
  funnels draw exact limit curves; exact count tests draw each provider's own limits, with Poisson reference curves.
  Statuses are encoded by shape, fill and colour, zero-event providers sit at O/E = 0 with an outline marker, and
  above 2,000 providers the not-different points are rasterized while flagged providers stay on top. Each figure
  carries a provenance footnote and alt text, and returns a `FigureResult` whose SVG, PDF and PNG exports are
  byte-identical for the same input and environment; no pyplot state is involved.
- `pprof_py.presentation.caterpillar(source, ...)`: an interval plot of provider estimates with the intervals of their
  own test and a volume panel of denominators. Providers are ordered by estimate for legibility only (the axis says
  so and never shows a rank); each interval is drawn from its lower to its upper bound, so intervals shifted by an
  empirical null render as they are; providers without a finite estimate are marked at the axis edge with any
  one-sided interval drawn from there, and the solver's clamp is never drawn. Up to 60 providers are labelled; above
  2,000 the not-different providers are rasterized and flagged providers stay on top.
- Presentation themes render text without font hinting, so text keeps the same width at every raster resolution and
  footnotes wrap identically in PNG, SVG and PDF output.
- `pprof_py.presentation.provider_table(source, ...)`: the provider summary table, with denominators, observed and
  expected counts, estimates with the intervals of the same test, flags and optional p-values. Its footnotes are
  generated from the test's settings and the providers' statuses (`NE` no finite estimate, `NT` not tested, `NI` no
  interval, `S` suppressed), and an interval gets extra decimals where rounding would blur whether it excludes the
  reference. One table specification renders to self-contained HTML, Markdown, LaTeX (booktabs; `longtable` above 40
  rows), plain text, a tidy DataFrame and Excel; every output is deterministic.
- New optional extra `excel` (`XlsxWriter>=3.0.1`) for `TableResult.to_excel()`.
- Displays on hostile data (quality review): log axes over many decades label only powers of ten; the
  observed-versus-expected axes use ticks even in square-root space; null calibration keeps the bulk of the statistics
  in view and counts extreme ones at the axis edge; the funnel always shows its test-level limits; the variation display
  states when sigma is estimated at 0 instead of drawing a degenerate density; shrinkage labels avoid the points.
- Documentation: **Which display answers my question?** (the analyst's questions mapped to displays and tables, and a
  family-support matrix that the tests check), **Theme, export and accessibility**, and **Migrating to the
  presentation layer** (the table of earlier calls, which moves there from the Presentation reference page).
- `flag_stability()` no longer fails when a three-stage model's `sigma_sensitivity()` raises: it omits the scenarios at
  sigma's bounds, warns, and says so in its footnote and table note.
- Documentation: one page per display under **Presentation → Reading the displays**, each with its spec sheet (what
  it answers, the quantities and their source, uncertainty, denominator, reference, misreadings and mitigations, and
  behaviour from 10 to 50,000 providers), how to read it and how it can mislead, with executed counterexamples on the
  same synthetic data (for example, the funnel against a league table); and a **Tables** page with every table function,
  the symbol set and the formatting rules.
- Documentation: a **Presentation** section with a gallery of every figure, rendered at build time from synthetic data
  by a local Sphinx extension (`docs/source/_ext/pprof_gallery.py`) with the same deterministic renderers; nothing is
  saved by hand. The private generator `pprof_py.presentation._synthetic.provider_data()` (planted outliers,
  overdispersion, zero-event providers) is shared by the docs and the tests.
- `measure_agreement()` no longer places providers without a finite estimate in either measure (their solver-bound
  intervals set the axes); the footnote counts them.
- `pprof_py.presentation.Report`: sections, text, figures and tables composed into one self-contained HTML file with a
  print stylesheet and a generated methods and provenance appendix (`.to_html()`, `.save()`, `.outline()`); no
  scripts, no network, no timestamp unless `date=` is given, and a note when tables carry provider-level values.
- The standalone `pprof_py.plotting.plot_funnel` and `pprof_py.plot_caterpillar` now draw through the presentation
  layer and return a `FigureResult` (`plot_caterpillar` returned `None` and no longer calls `plt.show()`; `fig, ax =`
  still works for both). `plot_funnel` draws the supplied `limits_df` curves as given and warns when flags contradict
  them. Styling keywords are deprecated and ignored; options the new layer does not offer (`ax=`, no intervals,
  `refline_value=None`, `sort_by_estimate=False`, `orientation="horizontal"`) keep the earlier drawing with a
  `DeprecationWarning`; both are removed in 0.7.0.
- `ProviderProfile.from_frame(..., curves=)` accepts funnel-limit curves supplied with the data.
- `pprof_py.presentation.ProfileCollection` (several measures of the same providers), `multi_measure()` (one
  interval panel per measure, common row order), `measure_agreement()` (two measures per provider with interval
  crosses and the joint flag status of their tests) and `multi_measure_table()` (each measure under a grouped header).
  Providers missing from a measure are marked, never dropped.
- Markdown and plain-text tables write a grouped header as a prefix of its columns' headers
  (`Readmission: Estimate (CI)`), since those formats cannot span columns.
- `pprof_py.presentation.flag_stability(model, ...)` and `flag_stability_table(...)`: the flags of each provider
  flagged in at least one scenario, one `test()` call per scenario (by default an alternative reference, the other
  null, and for three-stage models the bounds of sigma's interval from `sigma_sensitivity()`), with the providers
  whose status changes marked; the table counts flags per scenario or lists the changing providers.
- `pprof_py.presentation.shrinkage(fixed, random)` and `shrinkage_table(fixed, random)`: each provider's fixed-effect
  (unshrunken) estimate against its random-effect BLUP, both relative to their own test's reference (the fixed-effect
  test with `reference="mean"` by default), sized by volume, with the lines of no shrinkage and complete pooling;
  the table lists both estimates and the change in source order.
- `pprof_py.presentation.provider_variation(model)` and `provider_variation_table(model)`: for logistic and linear
  random-effect models, the BLUPs against the fitted between-provider distribution, the random-effect SD (with its
  profile-likelihood interval for logistic models) and the range of true effects it implies under normality.
- Figure exports no longer depend on what was drawn earlier in the same process: frozen layouts (axes and sub-figure
  boxes) are quantised with negative zero normalised, and Matplotlib's content-hashed SVG ids are renamed in document
  order. SVG ids of all figures change (`p00001`, `m00001`, ...); drawn content is unchanged.
- `pprof_py.presentation.reliability(iur)` and `reliability_table(iur)`: each provider's reliability at its size and
  the overall IUR from a fitted `BootstrapIUR`, and a table of the overall IUR, its variance decomposition and
  reliability by size decile (also for `DirectIUR`, and the split-half statistics of `SplitHalfIUR`); both state that
  reliability is a property of the measure, not a score for any provider.
- `pprof_py.presentation.observed_expected(source)`: observed against expected events on square-root axes (Poisson
  noise has roughly constant spread there, so departures are comparable across volumes), with the line O = E, guides
  at O/E = 0.5 and 2, and the test's funnel limits converted to events (half-integer counts for count tests).
- `pprof_py.presentation.null_calibration(model, ...)` and `null_calibration_table(...)`: raw z-statistics of each null
  group against the theoretical and the fitted null, with the flags of both nulls compared (the model is tested twice;
  no flag is decided by the display). Profiles now carry `null_mean` and `null_sd`.
- `pprof_py.presentation.data_quality(source)` and `data_quality_table(source, details=False)`: every provider
  accounted for (in the data, excluded by data preparation, analysed by flag, not tested, suppressed; no finite
  estimate, zero events and no interval as attributes), with each group's volumes; counts the source does not record
  are shown as "not recorded", never as zero.
- `pprof_py.presentation.forest(model)` and `coefficient_table(model)`: covariate effects from `summary()` with their
  intervals, as odds or hazard ratios on a log axis for logistic and Cox models (exponentiated for display through
  `CoefficientProfile.exponentiate()`), with a footnote that the associations are adjusted, not causal, and in each
  covariate's own units.
- **Model plot methods delegate to the presentation layer.** `plot_funnel` (logistic and linear fixed effects),
  `plot_provider_effects` (all four models) and `plot_standardized_measures` (logistic fixed effects) now draw with
  `funnel` and `caterpillar` and return a `FigureResult` (`fig, ax = ...` still works); they no longer call
  `plt.show()`. Logistic random-effect `plot_funnel` draws the funnel of the exact count test, and its default
  `test_method="wald"` warns. Linear random-effect `plot_funnel` and the other models' `plot_standardized_measures`
  keep their earlier drawing with a `DeprecationWarning`. Styling keywords, `target=` and `use_flags=False` no longer
  have an effect and warn. Everything deprecated here is removed in 0.7.0; see the migration table on the new page
  *Presentation layer (preview)*.
- `plot_caterpillar` reads intervals from `ci_lower`/`ci_upper` (the columns of `test()`) by default; frames with
  `lower`/`upper` still work, with a `DeprecationWarning`, until 0.7.0.

## 0.5.0 (2026-09-29)

The three-stage model, one inference layer for every provider test, standardized measures and provider tests for Cox
models, a review of the fixed-effect solvers and tests against R, and theory papers on inference for covariate
effects, provider effects and standardized measures, on the empirical null and on inter-unit reliability. This
release renames parts of the API and changes several defaults; no old name is kept as a deprecated alias.

### Migrating from 0.4.1

- **Renamed class.** `LogisticMixedEffectModel` is `LogisticFERandomClusterModel`. Its `fit()` takes the fitted stages
  (`stage1=`, `stage2=`) or explicit `beta`, `sigma` and `gamma_init` (which replace `beta_init` and `sigma_init`), and
  `summary(stage1=...)` replaces `stage1_model=`. `LogisticThreeStageModel` runs the whole pipeline.
- **Arguments.** `provider_var` replaces `group_var`; arrays of provider IDs are `provider_id` (was `groups`, or
  `provider` in `ProviderPenalizedCoxPH` and the provider-penalized logistic CV); the random-effects models take
  `cluster_vars` (was `group_vars`) and `predict(re_vars=...)`; measures, intervals and tests take `reference` for the
  reference value (was `null`).
- **Provider tests.** `test()` takes `providers` first and keyword-only options, and calibrates through `null_model`
  (replacing `empirical_null`, `n_strata` and `strata_var`); in the logistic fixed-effect model `n_resample` and `seed`
  replace `n_bootstrap`, and `score_modified` is removed. `test_standardized()` has a new signature. Every test returns
  one table indexed by `provider_id` with the columns `pprof_py.inference.PROVIDER_TEST_COLUMNS`.
- **Cross-validation.** `se_rule` (`"min"` or `"1se"`) replaces `use_1se` and `select`, and defaults to `"1se"` in
  every class (the Cox CVs defaulted to `"min"`). `model_` replaces `best_model_` and `final_estimator_`; `full_fit_` is
  removed. `provider_bound`, `provider_max_iter`, `alpha` and `lambda_value` replace `gamma_bound`,
  `max_provider_iter`, `alpha_en` and `lambda_val`.
- **Fitted attributes.** `provider_ids_`, `provider_indices_` and `provider_sizes_` replace `groups_`, `group_indices_`
  and `group_sizes_`, in the random-effects models too (with `cluster_ids_`, `cluster_sizes_` and `cluster_indices_`
  dictionaries for their other factors).
- **Removed.** `pprof_py.huber_location_scale` and `pprof_py.estimate_empirical_null` (use `robust_location_scale` and
  `EmpiricalNull` from `pprof_py.inference`); the `pprof_py.statistics` subpackage (its helpers are in
  `pprof_py.utils`); `breslow_baseline_hazard` (use `pprof_py.inference.survival.compute_baseline_hazard`); `update_sigma`. scikit-learn and seaborn
  are no longer dependencies.
- **Changed defaults and results.** `LogisticFERandomClusterModel` and `LogisticThreeStageModel` default to
  `estimator="marginal"` and `convergence_criterion="max_delta_gamma"` (R's recipe is `estimator="he2013"`,
  `convergence_criterion="relative"`, `bound_mode="absolute"`); `DataPrep` keeps providers with more than `cutoff`
  records; the logistic fixed-effect Wald test and model-based standardized-measure standard errors use the variance at
  the average case mix (R's variance stays in `variances_["gamma"]`); `BootstrapIUR.iur_groups_` is the reliability at
  each provider's size; the covariate score test uses the efficient information. The sections below give the reasons
  and the size of each change.

### Three-stage model and structural redesign

**Three-stage model**

- `LogisticThreeStageModel` runs He et al.'s three-stage pipeline from raw data, as R's
  `glmm.fac.hosp`: `glmm_data_prep`, Stage 1 on facility × hospital cells, Stage 2 with crossed
  facility and hospital intercepts and the Stage 1 offset, and Stage 3. See the
  [three-stage chapter](logistic/logistic_three_stage_model), which replaces the mixed-effect chapter.
- `LogisticMixedEffectModel` is renamed `LogisticFERandomClusterModel`. Its `fit()` takes the fitted
  stages (`stage1=`, `stage2=`) and derives β (by covariate name), σ (by name) and the start (Stage 2's
  facility effects matched by ID, plus its intercept); or explicit `beta`, `sigma` and `gamma_init`,
  which replace `beta_init` and `sigma_init`. `summary(stage1=...)` replaces `stage1_model=`.
- `estimator="marginal"` maximizes the marginal likelihood (unique and start-independent) by adaptive
  Gauss–Hermite quadrature, and is the default (see the consistency decisions below); `"he2013"` is R's
  iteration. `loglik_` reports the marginal
  log-likelihood under either estimator.
- `pprof_py.data.glmm_data_prep` reproduces R's `glmm.data.prep`.

**Provider tests**

- One count-test component, `pprof_py.inference.count_tests`, runs every count `test()`.
- `LogisticRandomEffectModel.test(test_method="exact")`: each cluster's effect is drawn once for all of
  a provider's rows in it, and the count's distribution is computed exactly (one cluster factor).
- `"poibin_exact"` returns limits by inverting the test in the fixed-effect and random-effects models;
  the fixed-effect `calculate_confidence_intervals(test_method="exact")` returns those limits.
- The random-effects `"resampling"` gives providers at the Monte Carlo floor the exact tails of the
  same null, with a warning.
- Provider-test results are indexed by `provider_id` (was `provider`), and `predict_provider_effect()`
  returns a `provider_id` column.

**Vocabulary and API**

- `provider_var` replaces `group_var`. The random-effects models take `provider_var` and `cluster_vars`
  (replacing `group_vars`); their accessors take `var`, and `predict()` takes `re_vars`. Arrays of
  provider IDs are `provider_id` (was `groups`, or `provider` in `ProviderPenalizedCoxPH`), and output
  tables have a `provider_id` column (was `group_id`). Standardized measures, intervals and plots take
  `reference` (was `null`).
- The CV selection rule is `se_rule` (`"min"` or `"1se"`) in every CV class, replacing `use_1se` and
  `select`; each class keeps its default. `provider_bound`, `provider_max_iter`, `alpha`, `model_` and
  `lambda_value` replace `gamma_bound`, `max_provider_iter`, `alpha_en`, `best_model_` and `lambda_val`.
  `breslow_baseline_hazard` is removed (use `compute_baseline_hazard`).
- The fixed-effect models' `groups_`, `group_indices_` and `group_sizes_` are `provider_ids_`,
  `provider_indices_` and `provider_sizes_`.

**Data preparation**

- `DataPrep` keeps providers with more than `cutoff` records (was at least `cutoff`), as R does.
- `DataPrep` accepts binomial trials (`n_char`), so binomial fixed-effect fits can use it; the trials
  are now re-read after screening, where they were misaligned.

**Consistency decisions**

- The random-effects models' fitted attributes follow the fixed-effect names: `provider_ids_`, `provider_sizes_` and
  `provider_indices_` for the provider, and `cluster_ids_`, `cluster_sizes_` and `cluster_indices_` (dictionaries keyed
  by cluster column), replacing `groups_`, `group_sizes_` and `group_indices_`.
- `LinearRandomEffectModel`'s measures, intervals, tests and provider plots raise a clear error for a fit with
  `cluster_vars`, which they do not support (they failed obscurely before).
- `LogisticFERandomClusterModel` and `LogisticThreeStageModel` default to `convergence_criterion="max_delta_gamma"`,
  which stops closer to the solution than R's `"relative"` rule; pass `"relative"` for output comparable with R.
- Every CV class defaults to `se_rule="1se"`; the two Cox CV classes defaulted to `"min"`.
- `LogisticFERandomClusterModel` and `LogisticThreeStageModel` default to `estimator="marginal"`; pass
  `estimator="he2013"` (with `convergence_criterion="relative"` and `bound_mode="absolute"`) for output comparable with R.
- Every CV class exposes the same attributes: `lambda_path_`, `cv_mean_deviance_`, `cv_se_deviance_`, `lambda_min_`,
  `lambda_1se_`, `lambda_`, `model_` and `coef_`. The Cox CVs' `final_estimator_` is `model_`; the discrete-survival CVs'
  `cv_mean_`/`cv_se_` are `cv_mean_deviance_`/`cv_se_deviance_`; `DiscreteSurvivalCV` gains `lambda_path_`, `lambda_` and `coef_`.
- `DiscreteSurvival` and `ProviderPenalizedDiscreteSurvival` share one `predict_hazard()`/`predict_survival()` signature,
  `(X, [provider_id,] time=None, lambda_value=None, which=None)`: without `time`, one column per time point; with it, the
  person-period form. Pass `which` by keyword (it was the third positional argument of the provider model's methods).
- Plots draw providers without a test result (`flag` NA) as hollow grey "Not tested" points; they were drawn as not flagged.
- `ProviderPenalizedLogistic.test()`: the fixed-effect model's exact count test at one path point (`which` or
  `lambda_value`), with limits by inversion; `ProviderPenalizedLogisticCV.test()` tests at the selected lambda.
- The cluster-robust (sandwich) covariance of the survival models warns when it uses fewer than 30 clusters.
- The fixed-effect Wald test warns about providers with no events or only events, whose effects have no finite
  estimate.
- The survival R-comparison results (`r_reference/results/`) are committed, so those tests run on a fresh clone.

**Penalized group paths**

- The group block solver is exact for every Hessian block. It was exact only when a group's block is a multiple of the
  identity (orthogonalized groups with the 1/4 majorizer); elsewhere it converged to a point that solves neither the plain
  nor the standardized group lasso. `GroupLassoLogistic` and `GroupLassoLinear` at their defaults are unchanged;
  with `orthogonalize=False` they now fit the plain group lasso exactly.
- `ProviderPenalizedLogistic` fits the standardized group lasso (`orthogonalize=True`, new), as `GroupLassoLogistic` and R's
  `grp.lasso(prov.char=)` do, and matches R on identical data; with one provider its group path is `GroupLassoLogistic`'s.
  Unpenalized (group 0) columns are refitted along the path; they stayed at their null-fit values.
- `ProviderPenalizedLogistic.lambda_max_` is taken with the provider effects fitted, for every penalty type (R's
  `set.lambda.grplasso` does the same). It was taken with them at zero, so on multi-provider data the automatic lambda path
  moves; at or above `lambda_max_` the path returns the null point exactly.
- `ProviderPenalizedLogistic`'s `alpha` defaults to `None` and follows the penalty type: the elastic net's mixing (1 when
  `None`), `0` for `"group_lasso"` (any other value raises; the old default of 1 made it a lasso), and required for
  `"sparse_group_lasso"`. An unknown `penalty_type` raises.
- `group_multiplier` is used by `GroupLassoLogistic`, `GroupLassoLinear` and `ProviderPenalizedLogistic`; it was accepted
  and ignored.
- One group solver serves every family: `GroupLassoCoxPH` and `ProviderPenalizedCoxPH` use the logistic and linear
  classes' solver. The survival copy thresholded multi-member groups on the wrong scale, so the Cox group paths solved no
  objective (KKT residuals 0.2-0.4); it never updated unpenalized (group 0) columns; and it reported convergence without a
  KKT check.
- `GroupLassoCoxPH` and `ProviderPenalizedCoxPH` fit the standardized group lasso (`orthogonalize=True`; it was a
  placeholder in `GroupLassoCoxPH`), with groups orthogonalized on their centered columns, as R's `grplasso::Strat.cox`
  does; `GroupLassoCoxPH` matches `Strat.cox` on identical data. `GroupLassoCoxPH`'s λ path starts at the true
  `lambda_max_` (it started at `lambda_max_ + 1e-5`), and both classes expose `kkt_violation_path_`.
- `ProviderPenalizedCoxPH`: `alpha` follows the penalty type as in `ProviderPenalizedLogistic` (`"group_lasso"` fit a lasso
  at the old default of 1); group-0 columns are fitted at the null point and along the path (they stayed at zero); at or
  above `lambda_max_` the path returns the null point exactly, for the elastic net too. With provider dummies as
  unpenalized columns, `GroupLassoCoxPH` gives the same path.
- A group-penalty null point fits exactly the columns that carry no penalty: group 0, or a column whose group term and L1
  term both vanish. `GroupLassoLogistic`, `GroupLassoLinear` and `ProviderPenalizedLogistic` also fitted columns with a zero
  penalty factor inside a penalized group, which the group term still penalizes. `lambda_max` for a group with a zero
  multiplier under the sparse group lasso divides by `alpha` (it was too small by that factor).
- `DiscreteSurvival` and `DiscreteSurvivalCV` fit the lasso only: `penalty_type="group_lasso"` and `"sparse_group_lasso"`
  raise, and the unused `groups`, `alpha` and `group_multiplier` arguments are removed. The group types were accepted but
  fitted the lasso; R's `DiscSurv` is lasso-only.
- The group paths' `converged_path_` is True wherever the KKT residual meets the tolerance. With the active set on (the
  default of `GroupLassoLogistic` and `GroupLassoLinear`), any group at zero sent the solver to `max_outer_iter` and the
  point was reported as not converged; `n_iter_path_` drops accordingly.

**Fixed-effect fitting and survival references**

- The fixed-effect Newton algorithms (`"Serbin"`, `"Ban"`) accept a step whose predicted gain is below the log-likelihood's
  rounding level instead of backtracking it to zero. Near the optimum the Armijo test compared rounding noise, so the last
  Serbin iteration shrank its step to exactly 0 over about 1,400 log-likelihood evaluations and the zero step read as
  convergence, one Newton step short of the optimum. A Serbin fit now ends at the optimum (the score falls from about 1e-6
  to 1e-11) with a dozen evaluations; fits and covariate LR and score tests run 4-7 times faster. Estimates move by at most
  1e-8 (Serbin) and 2e-6 (Ban, which also stopped short).
- Serbin takes the joint Newton step as it is, as R's `logis_BIN_fe_prov` does; it clipped the provider-effect part to
  ±2·`bound`. With covariates far from 0 (for example a calendar year) the step needs the provider effects to offset
  x̄ᵀΔβ, and clipping them alone made it a descent direction: the line search shrank it to about 1e-16 and the fit stopped
  near the null (log-likelihood −1522 against −1412 on the AOH goldens shifted by +50/−30), and without backtracking it
  diverged. The fit no longer depends on the covariates' origin (β and fitted values equal the centered fit's to 1e-13)
  and equals R's on the shifted goldens (β to 2e-13); fits in which the clip never acted, which include every test and
  documentation fit, are unchanged. Serbin now warns when it reaches `max_iter` or when a line search that shortened the
  step, not a small Newton step, ended the fit.
- `LogisticFixedEffectModel.summary(test_method="score")` refers the score of a covariate to its efficient information,
  with the provider effects as well as the other coefficients partialled out, as R's `summary.logis_fe` does; it
  partialled out the other coefficients only, which overstated the information whenever the covariate is associated with
  the providers (or has a mean far from 0, since the provider effects carry the intercept). The test was conservative
  (5% tests rejected 0.7% of the time under the null in a simulation with provider-dependent covariate means; now 4.7%)
  and nearly powerless for uncentered covariates (an age-like covariate at mean 70: statistic 0.95, now 29.8, with LR
  30.0). The statistics now equal R's to the fitting tolerance.
- The LR and score tests refit the model without the covariate on exactly the fitted rows, with the binomial trials and
  the fit's algorithm and settings. They refitted through the default constructor, which re-applied data preparation
  (a `use_dataprep=False` fit with providers of 10 or fewer records gave a negative LR statistic and a score test that
  raised) and ignored the trials (binomial fits raised). For the default path the LR statistics are unchanged. Testing
  the only covariate raises a clear `ValueError`.
- Ban (`algorithm="Ban"`) alternates its updates in centred covariates and returns the provider effects in the
  original ones. Alternating updates slow down when the covariates' common offset couples them with the provider
  effects: on the AOH goldens it took 19 iterations centred, 705 with the covariates shifted by (+5, −3), and did not
  converge in 10,000 with (+50, −30) (log-likelihood 0.52 short); it now takes 21 at every shift and reaches the maximum.
  At the default tolerance its estimates on the AOH goldens as given move by at most 2.2e-9.
- The model-based Wald test of the provider effects, its limits in `calculate_confidence_intervals`, and the
  `variance="model"` standard errors of `test_standardized` use the variance of the provider effect at the average
  case mix, Var(γ̂_k + x̄ᵀβ̂) = 1/I_k + (x̄_k − x̄)ᵀS⁻¹(x̄_k − x̄), stored as `variances_["gamma_case_mix"]`
  (`variances_["gamma"]` keeps R's Var(γ̂_k)). A test compares γ̂_k with a reference that carries the same error in β̂;
  R's variance measures that error from the covariates' origin, so recording a covariate a few units higher changed
  the standard errors (×1.53 on the AOH goldens at a shift of (+5, −3), ×12.3 at (+50, −30)) and removed every Wald
  flag. The new variance does not depend on the origin and matches the Monte Carlo variance of γ̂_k − median(γ̂)
  (median ratio 1.03, against 0.58 for R's). This departs from R's `logis_fe` Wald test and `confint`; on centred
  covariates the two nearly agree.
- `at_bound()` returns the providers without a finite estimate: those with records and no events or only events,
  and those held at the solver's clamp median ± `bound`. It compared |γ̂| with `bound`, which depended on the
  covariates' origin and missed providers that stopped short of the clamp.
- `calculate_confidence_intervals` weights the expected counts of its standardized-measure limits and its score
  limits by the binomial trials, and scales rates by the crude rate Σy/ΣN, as the estimates do; binomial fits gave
  limits that could exclude the estimate (an indirect ratio of 1.25 with limits 0.05–0.09). Its Wald limits use the
  normal quantile, as R's `confint.logis_fe` and `test()` do, rather than a t quantile on the number of rows, so the
  limits no longer depend on whether the same data are stored as binomial or Bernoulli rows.
- The Fine-Gray R comparison checks `FineGrayPH`'s standard errors against R's cluster-robust SEs, with which they agree to
  1e-14. It compared them with R's model-based `se(coef)`, 5% away; the two failures it reported were this mismatch.

**Survival standardized measures**

- `CoxPH.calculate_standardized_measures` gives indirect and direct standardized ratios (SMR, SHR) per provider, as
  defined in the SMR tutorial: the indirect ratio against the national Breslow baseline (He and Schaubel's two-stage
  estimate for a fit stratified by provider, the pooled model's otherwise), and the direct ratio from each provider's own
  baseline applied to the whole population. It matches R's `survival` to 1e-14, left truncation and tied times included.
  Survival Chapter 4 shows it next to the two-stage computation by hand.

**Uncertainty in the cluster SD**

- `LogisticRandomEffectModel.profile_sigma` gives each random-effect SD's profile-likelihood interval, as lme4's
  `confint(method = "profile")`, with the Laplace deviance evaluated at the conditional mode. On a crossed cohort it
  matches that profile computed in R to 5e-8; lme4's own limits differ by up to 2e-4 because its deviance function
  evaluates the log-determinant away from its mode.
- `LogisticThreeStageModel.sigma_sensitivity` refits Stage 3 at both ends of the cluster SD's interval and reports which
  provider flags change. Stage 2's σ carries more of the pipeline's uncertainty than Stage 1's β (REV-022); a
  hospital-only Stage 2 overstates it, the production two-effect Stage 2 does not.

**Empirical-null operating characteristics**

- The empirical-null guide reports a simulation study of flag rates and power under overdispersion (logistic fixed-effect
  tests and Cox SMRs): the theoretical null's flag rate rises to 0.29 as unexplained provider variation grows, while the
  empirical null grouped by provider size holds about 0.05; one-sided outliers make it conservative, and groups of a dozen
  providers make it liberal. No code changes.

**Provider tests for Cox standardized ratios**

- `CoxPH.test` tests each provider's indirect standardized ratio against 1 with the SMR tutorial's inference: the mid-p
  test calibrated by the theoretical or an empirical null (grouped by person-time), with limits that invert it, or the
  exact Poisson test with Byar and chi-square limits. Survival Chapter 4 shows both.

**Documentation**

- Every chapter's printed output is now what its code prints (in an 80-column terminal; wide tables use
  `to_string()`), checked by running each page. Pages that continue from another say so, and survival Chapters 13
  and 14 build their annual `time_year` cohort. Tutorials, penalized and group-lasso chapters, survival Chapters 1, 3,
  6, 8 and 10–14, and the reference pages were corrected where the text had drifted from the code; survival
  Chapter 10's report no longer flags a facility its own interval does not exclude.
- Signature listings are regenerated from the code, stale module paths and attribute names are updated, and the
  bibliography, cross-reference and docstring markup errors are fixed: the docs build has 4 warnings (intersphinx
  inventories, which need network access), down from 174.
- New theoretical reference, [Empirical Null Calibration of Provider Tests](empirical_null_theory.md): why the
  theoretical null miscalibrates under overdispersion and discreteness, the two-groups model, the robust estimators
  and their behaviour under one-sided outliers and in small groups, and flags and limits under a calibrated null, with
  every derivation checked numerically.
- Two bibliography entries are corrected: He et al. (2013) is the *Lifetime Data Analysis* paper on dialysis
  facilities, and Kalbfleisch and Wolfe (2013) is "On monitoring outcomes of medical providers", *Statistics in
  Biosciences* 5(2), 286–302.

**Three-stage data preparation**

- `glmm_data_prep` numbers the provider x cluster cells from the sorted rows themselves (a new cell wherever the
  pair changes). It repeated the per-cell counts of a `groupby` onto the rows, which assumes the groupby returns
  the cells in the rows' order; on production data it did not, so `cell_id` and `included` were attached to the
  wrong rows while every total stayed the same, and Stage 1's beta moved. `cell_sizes` is counted from the
  category codes (cluster-major, provider-minor, as R's `n.fac.hosp`).

- `LogisticFixedEffectModel.add_providers` reorders every per-provider result when it sorts by provider ID
  (`robust_variances_["gamma_fixed_beta"]` stayed in the pre-sort order, so it no longer matched `provider_ids_`)
  and remaps `provider_indices_`: when added IDs fall between existing ones, the existing providers' positions
  move, and standardized measures, intervals and tests read the wrong records for them.

**Random-effect measures and Cox cross-validation**

- `LinearRandomEffectModel.calculate_standardized_measures` and its `'SM'` intervals use `reference`: the indirect
  difference is the BLUP minus the reference effect (it was the BLUP whatever `reference` was). The default is still
  `"median"`; `reference=0` gives R pprof's measure. `reference="mean"` in `LogisticRandomEffectModel`'s measures is
  the provider-size-weighted mean BLUP, as documented (it was unweighted).
- `PenalizedCoxPHCV.model_` and `GroupLassoCoxPHCV.model_` are the full-data path, as in the other CV classes (they
  were refits at the selected lambda only, so `model_.coef_at(lambda_min_)` under `se_rule="1se"` returned the
  `lambda_1se_` coefficients); `coef_` is the path's point at `lambda_`, and `full_fit_` is removed.
- `GroupLassoCoxPH` returns the null point exactly at lambda >= `lambda_max_` (its penalized coefficients were about
  1e-17), as the provider classes do.
- The cluster-robust variance of `LogisticFixedEffectModel`'s provider effects (`variance="robust"` in `test_standardized`
  and `standardized_measure`) is the full sandwich of the joint (γ, β) fit for the provider effect at the average case
  mix, γ_j + x̄ᵀβ with x̄ the trials-weighted mean covariate row. It was (1/I_j)² times the meat, which treats β as known
  and understates the variance of providers with an unusual case mix; the sandwich of γ_j itself would depend on the
  covariates' origin (γ_j is the effect at x = 0) and overstate the variance the provider test needs by a factor of 50 or
  more with uncentered covariates, whereas the average-case-mix form is origin-invariant. R's `test_aoh` form is
  `variance="robust_fixed_beta"` (it matches `test_aoh` to 2e-15).
- `ProviderPenalizedLogistic` and `ProviderPenalizedDiscreteSurvival` have `coef_at` (and `ProviderPenalizedLogistic` and
  `GroupLassoLogistic` `intercept_at`), and every penalized path class interpolates the same way: linearly in log λ
  between path points, the end points outside the path. `DiscreteSurvival.coef_at` interpolated linearly in λ (values
  between path points move by up to 1e-2; at path points they are unchanged), and `GroupLassoLogistic.predict_proba`
  extrapolated the intercept above the first λ (it now uses the first point, as `coef_at` does).

**Inter-unit reliability**

- `BootstrapIUR.iur_groups_` is each provider's reliability at its own size,
  `s2_between_ / (s2_between_ + s2_within_ / n_k)`, the curve `decile_table()` evaluates. It divided the pooled
  within variance of the measure by `n_k`, which is already that variance at the effective size `n_prime_`, so every
  provider's noise was understated by the factor `n_prime_`: on a simulated cohort of 300 providers the values averaged
  0.993 against true reliabilities of 0.536, and now average 0.496 (mean absolute error 0.040). This departs from the
  internal R function `IUR_bootdata`, whose facility-level `IUR.fac` has the old form; the overall IUR, `s2_between_`,
  `s2_within_` and `n_prime_` equal R's to 1e-13 and are unchanged.
- `SplitHalfIUR` leaves providers with fewer than two records out of the split-half correlations, with a warning, and
  lists them in `excluded_groups_`; it raised `IndexError`. Results without such providers are unchanged.
- New theoretical reference, [Inference for Covariate Effects](inference_covariate_effects_theory.md), Part I of the
  inference theory series: the fixed-effect likelihood and the profile information of the coefficients, the Wald,
  likelihood-ratio and score tests, the incidental-parameter bias of the coefficients with small providers, the
  cluster-robust variance, and the other model families.
- New theoretical reference, [Inference for Provider Effects](inference_provider_effects_theory.md), Part II of the
  inference theory series: what a provider test compares, the Wald test and the variance it needs, the score, exact and
  bootstrap tests at the reference, limits by inversion, the robust variances, and the random-effect and three-stage
  tests.
- New theoretical reference, [Inference for Standardized Measures](inference_standardized_measures_theory.md), Part III
  of the inference theory series: indirect and direct standardized ratios and rates as functions of the provider
  effect, their standard errors and tests, limits by transformation and by inversion, the linear and random-effect
  measures, and the survival SMR.
- New theoretical reference, [Inter-Unit Reliability: Theory](inter_unit_reliability_theory.md): reliability and the
  IUR, the analysis-of-variance and bootstrap estimators as implemented, the reliability curve, the estimate's
  sampling distribution with an approximate interval, and what the split-half variants estimate.

**Internals**

- Models inherit pprof_py's own `ProviderModel` base class; scikit-learn is no longer a dependency.
- Provider tests and intervals moved from `measures/` to `inference/`, and every mixin has a unique name.

### Mixed-effect model inference

Reviewed against R's `glmm.fac.hosp`, `summary.glmm.fac` and
`summary.glmm.covar`. This changes results and breaks parts of the
`LogisticMixedEffectModel` API.

- `test()` defaults to `test_method="exact"`: each cluster's effect is drawn
  once for all of a provider's patients in that cluster (He et al. 2013,
  step (ii)) and the count's distribution is computed exactly, with no Monte
  Carlo error. `"poibin_exact"` is unchanged.
- `test()` returns confidence limits (`ci_lower`/`ci_upper`) for `"exact"`
  and `"poibin_exact"` by inverting the calibrated test; new
  `calculate_confidence_intervals()` maps them to standardized ratios and
  rates, as in the fixed-effect model.
- `test_method="resampling"` (unchanged draws, one per patient) gives
  providers whose simulated tail reaches the resolution floor the exact tails
  of the same null, with a warning. The floor capped |z| at 3.89 (at
  `n_resample=10000`), which under an empirical null could make the most
  extreme providers impossible to flag.
- `summary()` returns the Stage 1 model's Wald table, as R does; it needs the
  Stage 1 model (`fit(stage1_model=...)` or `summary(stage1_model=...)`). It
  previously understated the standard errors.
- `update_sigma` is removed: the update drove σ toward zero.
- New `bound_mode` (default `"relative"`: γ clipped to `median ± bound`;
  `"absolute"` reproduces the old `±bound` and R) and
  `convergence_criterion` (`"relative"` as before, or `"max_delta_gamma"`).
  The relative criterion's 0/0 case, which ended the fit silently with a NaN
  criterion, now counts as converged; a non-finite objective now ends the fit
  as not converged; new `converged_` attribute. The Newton step floors a
  provider's information at 1e-8, with a warning.
- The convergence objective uses the posterior variance unsquared, matching
  the score and information (no effect on the fits checked).
- Documentation: R's empirical-null configuration for this model is quantile
  groups of a facility-size variable; the rank configuration shown earlier was
  pprof_py's previous default.

### Provider testing rebuilt on one inference layer

Provider tests now share one pipeline in `pprof_py.inference`: a
z-statistic per provider, a null model, and one decision layer for
p-values, flags and intervals (see the
[empirical null guide](reference/empirical_null) and the
[tests reference](reference/measures_tests_plots)). This changes
results and breaks the `test()` and `test_standardized()` APIs.

**Corrections**

- `LogisticRandomEffectModel.test()` with `test_method="resampling"` or
  `"poibin_exact"`, and `LogisticMixedEffectModel.test()`, returned
  inverted flags (`-1` for providers above expected). Every test now
  flags `1` above the reference and `-1` below.
- `test_standardized()` no longer divides every z-statistic by a fixed
  `scale=1.81` by default, which made it far more conservative than its
  nominal level; the default null is the theoretical N(0, 1), and
  `null_model=FixedNull(sd=1.81)` reproduces the old behaviour.
- `test_standardized(empirical_null=True)` failed on import; empirical
  nulls are now available to every test through `null_model=`.
- Indirect standardized measures use the variance of the observed count
  under the reference effect (a score-type test) on an identity working
  scale, which keeps the Type I error near nominal when provider sizes
  differ; `indirect_variance="fitted"` with `transform="log"`
  reproduces the previous construction.
- Under an empirical null, intervals are shifted and scaled with the
  null, so an interval excludes the null value exactly when the provider
  is flagged.
- The logistic fixed-effect score, exact and bootstrap tests weight
  binomial outcomes (`n_var`) by their trials.
- The logistic fixed-effect Wald test uses the normal reference, as R
  pprof does, instead of a t distribution.
- p-values are no longer rounded to 7 decimals.

**API changes**

- `test()` in every family takes `providers` first, then keyword-only
  `test_method` (where a family offers several), `reference` (was
  `reference`), `null_model`, `alternative`, `level`, `critical`, `interval`,
  `n_resample` and `seed`. In the logistic fixed-effect model
  `n_resample` and `seed` replace `n_bootstrap` and `random_state`; in
  the random- and mixed-effect models `seed` now defaults to `None`
  (was `1`). `LogisticRandomEffectModel.test()` takes `provider_var` as a
  keyword (it was the first positional parameter). `empirical_null`,
  `n_strata`, `strata_var` and `score_modified` are removed.
- Every test returns one table indexed by `provider`, with the columns
  `pprof_py.inference.PROVIDER_TEST_COLUMNS`; `flag` is a nullable
  integer, `NA` for providers a test could not evaluate.
- `test_standardized(measure, providers, null_value, transform,
  null_model, population, reference, variance, indirect_variance,
  alternative, level, critical, interval, bounds)` replaces the previous
  signature; `reference`, `variance_type`, `empirical_null`, `groupwise`,
  `n_groups`, `remove_outliers`, `scale`, `z_scale`,
  `include_extreme_obs` and `extreme_obs_total_n` are removed.
- `LogisticRandomEffectModel.test()` compares with `reference=0` (the
  random-effect mean, as R pprof) by default. `LogisticMixedEffectModel.test()`
  uses the theoretical null by default; the empirical null it applied
  before is available through `null_model` (see the guide).
- `LinearRandomEffectModel.test()` accepts `"median"` and `"mean"`.
- Removed: `pprof_py.huber_location_scale`, `pprof_py.estimate_empirical_null`,
  the helpers in `pprof_py.inference.empirical_null`
  (`calibrate_empirical_null`, `resample_pvalue`, `poibin_exact_pvalue`,
  `pvalues_to_zscores`, `assign_flags`), and
  `pprof_py.inference.survival.fit_robust_location_scale` and
  `assign_quantile_groups`. Use `robust_location_scale`,
  `EmpiricalNull` and `assign_groups` from `pprof_py.inference`.
- The Monte Carlo tests (`"bootstrap_exact"`, `"resampling"`) draw
  independent streams for each provider from one seeded generator.

**New**

- `pprof_py.inference`: null models (`TheoreticalNull`, `FixedNull`,
  and `EmpiricalNull` with quantile or rank grouping, pooled means,
  fitting subsets and a small-group policy that warns instead of
  falling back silently); Huber, bisquare and MM estimators that
  reproduce `MASS::rlm`; `standardized_measure`, `StandardPopulation`,
  `z_statistic`, `provider_test` and `at_bound`.
- Wald tests report confidence intervals (Student-t intervals for
  `LinearFixedEffectModel`); the `LogisticRandomEffectModel` and
  `LogisticMixedEffectModel` tests accept one-sided alternatives.
- The survival empirical-null functions run on the same layer with R's
  settings (least-squares start; missing sizes left ungrouped).

The test suite checks these against `MASS::rlm`, the EmpiNull R package
and R pprof's conventions to within 1e-12.

## 0.4.1 (2026-09-21)

- **Shared Gamma-frailty Cox model** (`FrailtyCoxPH`) and
  **time-varying-coefficient Cox model** (`TimeVaryingCoxPH`) migrated
  from `coxph_package` into `pprof_py` with full `__init__.py` exports,
  API reference, README, and changelog entries.
- Algorithm modules `algorithms/survival/frailty.py` and
  `algorithms/survival/time_varying.py` added.
- Documentation cleanup: removed all internal review artifacts
  (ISSUE/CODE_ISSUES/K-code references) from docs and Code_review
  files; fixed broken links in README, index, and reference pages.

## 0.4.0 (2026-09-19)

The largest commit in the repository's history, adding the shared
elastic-net/group-lasso coordinate-descent engine
(`algorithms/coordinate_descent.py`, `algorithms/penalty.py`) and
extending all three model families with penalized, group-lasso,
provider-penalized, and discrete-survival estimators:

- **Penalized regression** for logistic and linear outcomes
  (`PenalizedLogistic`, `PenalizedLogisticCV`, `PenalizedLinear`,
  `PenalizedLinearCV`) — see the
  [penalized logistic chapter](logistic/penalized_logistic).
- **Group lasso** across all three model families (`GroupLassoLogistic`,
  `GroupLassoLogisticCV`, `GroupLassoLinear`, `GroupLassoCoxPH`,
  `GroupLassoCoxPHCV`) — see the
  [group lasso chapter](logistic/group_lasso_logistic).
- **Provider-penalized models** (`ProviderPenalizedLogistic`,
  `ProviderPenalizedLogisticCV`, `ProviderPenalizedCoxPH`) — the
  package's own original methodological contribution; see the
  [provider-penalized chapter](logistic/provider_penalized_logistic).
- **Discrete-time survival models**, plain and provider-penalized
  (`DiscreteSurvival`, `DiscreteSurvivalCV`,
  `ProviderPenalizedDiscreteSurvival`,
  `ProviderPenalizedDiscreteSurvivalCV`) — see
  [Chapter 13](survival/13_discrete_survival).
- **`LogisticMixedEffectModel`** — Stage 3 of the He et al. (2013)
  three-stage SRR approach; see the
  [mixed-effect chapter](logistic/logistic_three_stage_model).

## 0.2.0 — the survival/Cox foundation

The "Replace the pprof_oy with refactored code" commit (2026-09-15,
464 files changed) added the entire survival/Cox module: `CoxPH`,
`PenalizedCoxPH`/`PenalizedCoxPHCV`, `CauseSpecificCoxPH`/`FineGrayPH`
(competing risks), `CoxPHSelector`, the indirect-standardization
(SMR/SHR) machinery (see
[Chapter 4](survival/04_indirect_standardization_smr_shr)),
robust/clustered variance, time-dependent-covariate data preparation
(`tmerge`, `build_skeleton`), and the
[diagnostics and R-validation infrastructure](diagnostics-guide).
This is the point where a linear/logistic-only package became a
provider-profiling package with a full survival-analysis arm.

## v0.1-legacy (tagged; last commit 2025-05-17)

The original release: fixed-effect and random-effect models for linear
and logistic outcomes only. Per git history: first commit and README
(2025-03-07), Sphinx scaffolding (2025-04-19),
`LogisticFixedEffectModel` (2025-05-06),
`LinearFixedEffectModel`/`LinearRandomEffectModel` plotting methods
(2025-05-07), `LogisticRandomEffectModel` (2025-05-12), and
documentation fixes through 2025-05-17. No survival/Cox module, no
penalized regression, no `measures.iur`. See the
[logistic FE](logistic/logistic_fixed_effect_model) and
[logistic RE](logistic_random_effect_model_stats) reference pages.

## Undated: `measures.iur`

The [inter-unit reliability](inter-unit-reliability-guide) module
(`BootstrapIUR`, `SplitHalfIUR`, `DirectIUR`, `ratio_measure`) exists
in the `0.4.0` codebase but is not re-exported at the package root
and is not attributable to a specific commit or version from the
available git history.
