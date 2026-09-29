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
- New theoretical reference, [Inter-Unit Reliability: Theory](inter_unit_reliability_theory.md): reliability and the
  IUR, the analysis-of-variance and bootstrap estimators as implemented, the reliability curve, the estimate's
  sampling distribution with an approximate interval, and what the split-half variants estimate.

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
