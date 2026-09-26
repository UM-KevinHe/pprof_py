(changelog)=
# Changelog

```{note}
This changelog is reconstructed from git history (commit messages,
dates, and the `v0.1-legacy` tag) and from `pprof_py.__version__`.
Treat specific dates as accurate (taken from commit timestamps) and
feature attributions as a best reconstruction. There is no released
`0.3.0`: `pyproject.toml` goes from `0.2.0` directly to `0.4.0`.
```

## Unreleased — three-stage model and structural redesign

This breaks parts of the API; no old name is kept as a deprecated alias.

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
  Gauss–Hermite quadrature; `"he2013"` stays the default. `loglik_` reports the marginal
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

**Internals**

- Models inherit pprof_py's own `ProviderModel` base class; scikit-learn is no longer a dependency.
- Provider tests and intervals moved from `measures/` to `inference/`, and every mixin has a unique name.

## Unreleased — mixed-effect model inference

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

## Unreleased — provider testing rebuilt on one inference layer

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

## 0.4.1 — current (July 2025)

- **Shared Gamma-frailty Cox model** (`FrailtyCoxPH`) and
  **time-varying-coefficient Cox model** (`TimeVaryingCoxPH`) migrated
  from `coxph_package` into `pprof_py` with full `__init__.py` exports,
  API reference, README, and changelog entries.
- Algorithm modules `algorithms/survival/frailty.py` and
  `algorithms/survival/time_varying.py` added.
- Documentation cleanup: removed all internal review artifacts
  (ISSUE/CODE_ISSUES/K-code references) from docs and Code_review
  files; fixed broken links in README, index, and reference pages.

## 0.4.0 (September 2026)

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
