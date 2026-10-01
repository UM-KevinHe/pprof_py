# pprof_py presentation layer — Phase 1 audit

**Date:** 2026-09-30 · **Audited commit:** `main` = tag `v0.5.0` = `d3d92a1` (tree-identical to `test/mixed-effect` `394b911`) · **Phase:** 0 (baseline) and 1 (audit) complete · **Next gate:** Phase 2 design spec → stop for approval.

Every claim is labelled **[run]** (reproduced by executing code; the harness and its JSON output are in `harness/` and `evidence/`), **[read]** (file:line in the audited tree), or **[assumed]**. Paths are relative to `pprof_py/` unless they start with `docs/` (= `pprof_py/docs/source/`) or are repo-root files. No package code was changed.

---

## 1. Summary

**The statistical layer is ready to be consumed; the current plotting layer mostly does not consume it.**

1. `test()` returns one fixed schema (15 columns, `PROVIDER_TEST_COLUMNS`) for logistic FE/RE, linear FE/RE and the three-stage model, and a superset (adds `observed`, `expected`, `person_time`) for CoxPH **[run]**. Its own intervals and flags never disagreed: 0 S3 violations in 10 configurations × 2 sizes, under both theoretical and empirical nulls **[run]**. A renderer that draws `test()` output is S3-consistent by construction.
2. The model-attached plots instead recompute statistics in plotting code (funnel limits, forest intervals, reference values) and pair intervals and flags from different sources. At 1,000 providers this produces visible contradictions **[run]**:

   | Display (defaults unless noted) | Contradiction at N = 1,000 |
   |---|---|
   | Logistic RE funnel | 70 of 106 providers outside the 95% limits are **not** flagged |
   | Logistic FE funnel, `test_method="poibin_exact"` | 30 flagged providers sit **exactly on** a limit; 55 unflagged also do |
   | Linear RE funnel | 30 flagged providers drawn strictly **inside** the limits |
   | Logistic RE standardized-measure caterpillar | 219 intervals exclude 1.0 while the provider is not flagged |
   | Logistic FE provider effects, `reference="mean"` | reference line drawn at −2.115, test used −2.020; 53 interval/flag disagreements |
   | Logistic FE standardized-measure caterpillar | 7 zero-event providers: interval excludes 1.0, flag 0 |

   At N = 20 the same displays showed 0–1 disagreements, which is why the problem is easy to miss.
3. **[P0] failures today:** status encoded by hue alone in every caterpillar (one marker shape); all four status colours below 3:1 contrast against white (1.50–1.83 at the caterpillar's drawn alpha); "Lower" and "Expected" differ by ΔL\* 0.8, i.e. identical in grayscale **[run]**. No figure states level, reference, null model, test method or estimator type (S2) **[run]**.
4. **Coverage:** plot methods exist on 4 of 30 exported model classes; none for CoxPH, three-stage, FE-random-cluster or any penalized model **[run]**. The docs contain zero rendered figures **[run]**. There is no test CI **[read]**.
5. **What is already good and should be kept:** a single style module with role-named Okabe–Ito constants; the funnel's ▼●▲ shape redundancy and hollow "Not tested" encoding; flag counts in legend labels; the standalone `plot_funnel` returning `(fig, ax)` and accepting `ax=`; acceptable speed (50,000 providers in ≈2.3 s, PNG) **[run]**; deterministic output is reachable with Matplotlib 3.11 knobs **[run]**.

**Decisions this audit makes likely for Phase 2:** consume `test()` (plus `calculate_standardized_measures()` for denominators) as the primary contract; obtain funnel limits from a new statistical-layer accessor (open question 4); gate capabilities per family because `at_bound()`, interval column names, `summary()` columns and random-effect attributes differ by family (§6).

---

## 2. Phase 0 baseline

| Item | Result |
|---|---|
| Environment | Ubuntu 24.04, 1 CPU, 3 GB. Python 3.12.3 venv; `pip install -e ".[dev,docs]"` exit 0. Resolved: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, matplotlib 3.11.2, numba 0.68.0, llvmlite 0.50.0, fast_poibin 0.4.2, statsmodels 0.15.0, pytest 9.1.1, Sphinx 9.1.0, shibuya 2026.7.12, myst-parser 5.1.0, sphinxcontrib-bibtex 2.7.0 **[run]** |
| Imported tree | From `/tmp`: `pprof_py.__file__` = clone, `__version__` 0.5.0, backend `agg` **[run]** |
| Test suite | `pytest pprof_py/tests`: **690 collected — 1 failed, 688 passed, 1 skipped, 58 warnings, 133.5 s** **[run]** (`evidence/pytest.log`, `evidence/junit.xml`) |
| Failing set | `tests/test_infrastructure.py::TestUtils::test_setup_logger` — `ZoneInfoNotFoundError: 'US/Eastern'`. **Environmental**: negative control passes with `tzdata` on `PYTHONPATH` (venv untouched) and fails without it **[run]** |
| Skipped | `tests/survival/test_robust_variance.py:36` — `lifelines` not installed **[run]** |
| Warnings | 58; 48 from `tests/logistic/test_fe_random_cluster.py`; one from `plotting/coefficients.py:139` ("CI columns ('lower', 'upper') not found") raised by the package's own plotting test **[run]** |
| Docs build | `sphinx-build -E -b html source _build/html`: exit 0, 50.5 s, **4 warnings, all intersphinx inventories unreachable (HTTP 403 via proxy)** for python, pandas, numpy, scipy — matches the reported baseline **[run]** |
| Docs output checks | `chapter_check.py` and siblings are not in the repo (they live in the external handoff kit), so the docs baseline is the Sphinx build **[run]** |
| CI | `.github/workflows/docs_deploy.yml` only: push to `main`, Python 3.9, installs `sphinx_rtd_theme` (unused; `conf.py:62` uses shibuya), builds docs. **No test job** **[read]** |
| README claim | README.md:291 "200 passed, 26 failed, 1 skipped" is stale **[run]** |

---

## 3. Inventory of what exists

**Code** (3,709 lines in `plotting/`) **[read]**

| File | Lines | Contents |
|---|---:|---|
| `plotting/__init__.py` | 11 | exports `plot_caterpillar`, `plot_funnel`; docstring claims all functions return `(fig, ax)` (lines 3–6) |
| `plotting/coefficients.py` | 311 | `plot_caterpillar` (despite the module name; no coefficient code) |
| `plotting/funnel.py` | 266 | `plot_funnel(df, limits_df, ...)` shared renderer |
| `plotting/style.py` | 86 | role-named constants, `remove_top_right_spines` |
| `plotting/logistic/fixed_effect.py` | 737 | mixin for `LogisticFixedEffectModel` |
| `plotting/logistic/random_effect.py` | 628 | mixin for `LogisticRandomEffectModel` |
| `plotting/linear/fixed_effect.py` | 806 | mixin for `LinearFixedEffectModel` |
| `plotting/linear/random_effect.py` | 854 | mixin for `LinearRandomEffectModel` |

There is no caterpillar or forest module: each mixin re-implements its own coefficient forest.

**Public API** **[run]** — root export `pprof_py.plot_caterpillar`; `plot_funnel` only in `pprof_py.plotting`. Plot methods by class:

| Class | Methods |
|---|---|
| `LinearFixedEffectModel`, `LinearRandomEffectModel` | `plot_funnel`, `plot_provider_effects`, `plot_standardized_measures`, `plot_coefficient_forest`, `plot_residuals`, `plot_qq` |
| `LogisticFixedEffectModel` | the same six; `plot_residuals`/`plot_qq` raise `NotImplementedError` (`plotting/logistic/fixed_effect.py:711–737`) |
| `LogisticRandomEffectModel` | first four only (no residual/QQ methods at all) |
| 26 other exported classes (CoxPH and all survival models, three-stage, FE-random-cluster, penalized) | none |

**Tests** **[read]** — `tests/test_infrastructure.py:148–165` (three import smoke tests) and `tests/test_api_consistency.py:80–105` (one "Not tested" legend test). Nothing tests figure structure, intervals, limits, determinism or accessibility.

**Docs** **[run]** — zero images under `docs/`; `matplotlib.sphinxext.plot_directive` is loaded (`conf.py:33`) but never used; plotting calls appear only as code blocks in 12 pages.

---

## 4. Findings

### 4.1 Rendering contract and global state

| # | Finding | Evidence |
|---|---|---|
| R1 | Model `plot_funnel` methods return `(Figure, Axes)` despite `-> None` annotations; every other method returns `None`. Every method creates a **new** figure (never the current one) and leaves it open. The brief's §2.2 ("return None, draw on pyplot's current figure") and §2.4.1 ("the two `plot_funnel` entry points differ in return type") are both refuted | [run] `renders/*/report.json` `return_type`, `figures_left_open`; [read] `plotting/logistic/fixed_effect.py:75,253` and peers |
| R2 | `plot_caterpillar` calls `plt.show()` when no `save_path` (`coefficients.py:309–310`); forests do the same; linear `plot_residuals`/`plot_qq` always call `plt.show()` and cannot save (`plotting/linear/fixed_effect.py:723–724, 805–806`; `plotting/linear/random_effect.py:769–770, 853–854`) | [read] |
| R3 | The standalone funnel's `plt.show()` branch is unreachable (`ax` rebound at `funnel.py:101`, branch at 263–264); `plt.tight_layout()` acts on the current figure (258); every save uses `bbox_inches="tight"`, so exported size depends on content | [read] |
| R4 | `import pprof_py` imports `matplotlib.pyplot` (plotting mixins are model base classes); import takes 1.6–1.8 s in this environment | [run] |
| R5 | Every logistic FE funnel call emits pandas 3 `ChainedAssignmentError` from `df["precision"].replace(..., inplace=True)` (`plotting/logistic/fixed_effect.py:209`); under Copy-on-Write that line no longer modifies `df` | [run] warnings in `report.json`; [read] |
| R6 | Same model, different default references: the logistic RE funnel tests against the median BLUP (`plotting/logistic/random_effect.py:98, 148`), its `plot_provider_effects` against 0 (`random_effect.py:~248`) | [read]; [run] reference lines 0.0 vs median |

### 4.2 Statistical integrity (S1, S3, S4)

**Statistics computed inside plotting code** **[read]**

| Location | What is computed |
|---|---|
| `plotting/logistic/fixed_effect.py:187–250` | null probabilities, expected counts, variances, precision = E²/Var, score limits `target ± z·√Var/E`, exact limits from Poisson-binomial quantiles; ignores binomial trials `N_` (196–203) while `test()` weights by them (`inference/logistic/fixed_effect/provider_tests.py:107–111`) |
| `plotting/logistic/fixed_effect.py:205–209` | providers with null variance ≤ 1e-14 moved to 1.1 × max precision |
| `plotting/logistic/random_effect.py:182–202` | Poisson-approximation limits `target ± z/√E`, unclipped (go below 0) |
| `plotting/linear/fixed_effect.py:166–181`, `plotting/linear/random_effect.py:180–195` | normal-quantile limits `target ± z·σ/√n` |
| `plotting/logistic/fixed_effect.py:596–601`, `plotting/linear/fixed_effect.py:518`, `plotting/linear/random_effect.py:561` | forest intervals with `t.ppf(0.975, n − p − m)`, hard-coded 95% |
| `plotting/logistic/fixed_effect.py:341–349`; `plotting/logistic/random_effect.py:78–86`; `plotting/linear/random_effect.py:149–157, 268, 310, 380` | reference values ("median"/"mean") recomputed |

**S4: points outside the outer limits vs flagged providers** **[run]** (flagged / outside / mismatches; outer α = 0.05 unless stated)

| Funnel | N = 20 | N = 1,000 |
|---|---|---|
| Logistic FE, score (default) | 6 / 6 / 0 | 139 / 139 / 0 |
| Logistic FE, `poibin_exact` | 6 / 6 / 0 | 137 / 107 / **30 flagged exactly on a limit**; 55 unflagged also exactly on a limit; 0 strictly inside |
| Logistic FE, α = (0.05, 0.002), outer 0.002 | 3 / 3 / 0 | 29 / 29 / 0 |
| Logistic FE, no DataPrep (at-bound providers kept) | 5 / 5 / 0 | 137 / 137 / 0 |
| Logistic RE (Poisson limits vs Wald test on BLUPs) | 5 / 5 / 0 | 36 / 106 / **70 outside but not flagged** |
| Linear FE | 9 / 9 / 0 | 422 / 422 / 0 |
| Linear RE | 9 / 9 / 0 | 388 / 358 / **30 flagged strictly inside** |

Interpretation: the score funnel is consistent because its limit formula is the score test inverted. The exact funnel uses count quantiles while flags come from a two-sided mid-p test (`provider_tests.py:51–52`), so discrete boundaries coincide with points. The RE funnels draw limits for a statistic that is not the one being tested.

**S3: interval excludes the null value vs flag, as drawn** **[run]** (disagreements)

| Display | N = 20 | N = 1,000 |
|---|---:|---:|
| Logistic FE provider effects (median, Wald) | 0 | 0 |
| Logistic FE provider effects, `reference="mean"` | 1 | 53 |
| Logistic FE standardized measures (score) | 0 | 7 |
| Logistic RE provider effects | 0 | 0 |
| Logistic RE standardized measures | 1 | 219 (20 providers also have no interval) |
| Linear FE / RE, provider effects and standardized measures | 0 | 0 |
| **`test()` output itself, all families, theoretical and empirical null** | **0** | **0** |

Reference-line check **[run]**: with `reference="mean"` the logistic FE line is drawn at the unweighted mean of γ (−2.1483 at N = 20; −2.1150 at N = 1,000) while the test layer uses the size-weighted mean (−2.1142; −2.0204; `inference/effect_tests.py:46–47`). Median references match.

Coefficient forests vs `summary()` **[run]**: logistic FE differs by up to 7.7e-05 (N = 20) and 1.2e-06 (N = 1,000) because the plot uses a t quantile; linear FE and RE match exactly; logistic RE `summary()` has no interval columns and the plot draws `estimate ± z·se`.

### 4.3 Missingness, extremes and silent handling (S6, S10, S11)

| # | Finding | Evidence |
|---|---|---|
| M1 | **At-bound providers dominate the caterpillar.** In default fits at N = 1,000, 18 zero-event providers survive DataPrep; their Wald intervals (≈ ±100 on the log-odds scale) set the x-axis to ≈ −80…60 and compress the other 980 providers into a vertical line. Nothing marks them | [run] `renders/N1000/logfe_provider_effects.png`, `logfe_atbound_provider_effects.png` |
| M2 | At-bound providers are reported as `flag = 0`, never NA, by `poibin_exact` (estimate −10.13, interval (−∞, −0.74)), `score` (no interval) and `wald` (se 52.2, interval (−112.4, 92.2), with a warning); they are drawn as "Expected" | [run] `report.json` `at_bound` |
| M3 | DataPrep removed the two 6-patient providers at both sizes, announced only through a root-logger warning ("2 out of 20 providers are small and will be filtered out"); neither the fitted model nor `DataPrep` records which providers were excluded | [run] |
| M4 | `at_bound()` is logistic-FE-specific: `KeyError: 'gamma'` on both RE models; on `LinearFixedEffectModel` it returns **all 20 providers** as "no finite estimate" | [run] `evidence/availability.json` |
| M5 | NaN estimates or intervals are not drawn but still counted in legend totals (`coefficients.py:170–176, 245`); logistic RE standardized-measure intervals are NaN for zero-event providers (2 of 20; 20 of 1,000) and disappear silently | [read]; [run] |
| M6 | Injected NA flags render as hollow grey "Not tested (k)" — the code agrees with the changelog, not with `docs/reference/measures_tests_plots.md:262` | [run] |
| M7 | `plot_caterpillar` cannot draw an interval that excludes its own estimate: with intervals from `test(null_model=FixedNull(mean=2.5, sd=1))`, all 18 estimates lie outside their intervals and the call raises `ValueError: 'xerr' must not contain negative values`. Empirical nulls on the harness data had small locations, so no natural case arose | [run] |
| M8 | `plot_caterpillar`'s default interval columns are `lower`/`upper`; a `test()` frame (`ci_lower`/`ci_upper`) silently loses its intervals with a warning — the package's own test triggers this | [run] |
| M9 | The score test returns z = 0 (flag 0) when the null variance is below 1e-14 (`provider_tests.py:113`) | [read] |

### 4.4 Encodings, labels and accessibility (§7)

**Palette** (`plotting/style.py:21–32`; colours confirmed on drawn legend handles) **[run]**

| Status | Hex | Contrast vs white | at α 0.8 (funnel) | at α 0.5 (caterpillar, as drawn) | L\* |
|---|---|---:|---:|---:|---:|
| Lower (−1) | #E69F00 | 2.25 | 1.92 | 1.50 | 70.6 |
| Expected (0) | #56B4E9 | 2.31 | 1.94 | 1.50 | 69.8 |
| Higher (+1) | #009E73 | 3.42 | 2.67 | 1.83 | 57.7 |
| Not tested (hollow edge) | #9E9E9E | 2.68 | 2.13 | 1.57 | 65.1 |

Other elements: reference line 21.0; control-limit lines 3.95; default error bars 1.83; grid 1.50; funnel fill 1.13. Requirement §7.4: ≥ 3:1 for meaningful graphics.

**Pairwise distinctness** (ΔE in CAM02-UCS via `colorspacious` 1.1.2, Machado-2009 simulation at full severity; grayscale as ΔL\*) **[run]**

| Pair | Normal | Protan | Deutan | Tritan | ΔL\* |
|---|---:|---:|---:|---:|---:|
| Lower–Expected | 56.8 | 51.2 | 54.7 | 51.6 | **0.8** |
| Lower–Higher | 42.5 | 20.7 | 30.5 | 50.8 | 12.9 |
| Lower–Not tested | 33.0 | 30.8 | 32.5 | 27.7 | 5.5 |
| Expected–Higher | 31.5 | 33.8 | 31.7 | 13.7 | 12.1 |
| Expected–Not tested | 25.5 | 21.7 | 23.3 | 25.2 | 4.7 |
| Higher–Not tested | 25.6 | 12.2 | 11.3 | 23.7 | 7.4 |

Thresholds are a Phase 2 decision; the visual check (`renders/cvd_sheet.png`) shows Lower and Expected as the same grey and Expected/Higher both teal under tritanopia.

| # | Finding | Evidence |
|---|---|---|
| E1 | Caterpillars use one marker shape for every status → hue-only encoding (fails §7.2 [P0]); funnels use ▼●▲, and "Not tested" reuses ● hollow | [run] `distinct_marker_shapes`; [read] `coefficients.py:206`, `funnel.py:38` |
| E2 | Funnel legends use `int((1−α)·100)`: the 99.8% limits — including the plotting guide's own example — read **"99% CI"**; control limits are labelled "CI" | [run]; [read] `funnel.py:131,146` |
| E3 | Default figure 8 × 6 in with minimum text 10 pt → **4.2 pt** if placed in an 85 mm column (every figure) | [run] |
| E4 | No provenance on any figure; titles are generic ("Funnel Plot (Indirect Standardization)", "Caterpillar Plot"); flag 0 is labelled "Expected"; provider axis is "Group" in the standalone function and "Provider" in methods | [run] |
| E5 | Caterpillars hide provider labels at every N (`coefficients.py:277`); `point_alpha` never reaches drawn points (`coefficients.py:235, 256`) | [run]; [read] |
| E6 | Coefficient forests draw 1.5 pt markers (`point_size=0.05 × 30`); `errorbar_size`/`errorbar_alpha` are accepted but unused; docstring defaults disagree with code (`plotting/logistic/fixed_effect.py:494–660`) | [run]; [read] |
| E7 | Funnel x-axes mean different things per family — E²/Var at the null (logistic FE, "Precision"), expected count (logistic RE), provider size (linear, "Precision (Group Size)") — and limit curves are polylines through providers' own precisions (jagged at N = 20, a zigzag smear for exact limits at N = 1,000) | [run] renders |

### 4.5 Export determinism (§7.3) **[run]**

| Output | Same figure saved twice, defaults | Two processes, defaults | Two processes, `SOURCE_DATE_EPOCH=0` + `svg.hashsalt` |
|---|---|---|---|
| SVG (`plotting.plot_funnel`) | **differs** | **differs** (81 lines: `<dc:date>`, path ids, clip-path ids) | identical |
| PDF | identical | **differs** (creation date) | identical |
| PNG | identical | identical | identical |
| `plot_caterpillar(save_path=…)` SVG / PDF / PNG | — | differs / differs / identical | identical / identical / identical |

Both knobs are global (environment variable, rcParam) in this test. Per-save metadata plus a scoped `rc_context` should achieve the same without global state **[assumed]**; Phase 2 spike to confirm.

### 4.6 Scale (§5.5)

| Display | Providers | Format | Seconds | Size |
|---|---:|---|---:|---:|
| `plot_caterpillar` | 10,000 | PNG | 0.82 | 245 KB |
| `plot_caterpillar` | 10,000 | SVG | 1.15 | 3.4 MB |
| `plot_caterpillar` | 50,000 | PNG | 2.32 | 228 KB |
| `plotting.plot_funnel` | 10,000 | PNG | 0.45 | 356 KB |
| `plotting.plot_funnel` | 10,000 | SVG | 0.27 | 1.1 MB |
| `plotting.plot_funnel` | 50,000 | PNG | 0.80 | 295 KB |

Model methods at N = 1,000: the exact funnel takes 5.5 s (per-provider quantile loop in plotting code); others ≤ 1.4 s **[run]**. Speed is not the problem at scale; legibility is: caterpillars become an unlabeled mesh, at-bound providers wreck the axis, exact limits smear **[run]** (`renders/sheet_N1000.png`).

### 4.7 Documentation drift **[run]/[read]**

1. `docs/reference/measures_tests_plots.md:260` — "return `None` (they draw on the current figure)": funnels return `(Figure, Axes)`; all methods draw on a new figure.
2. `…measures_tests_plots.md:262` — NA flags "drawn as not flagged": drawn hollow grey "Not tested".
3. `…measures_tests_plots.md:273–274` — says the `coefficients` docstring mentions coefficient paths; it no longer does (`coefficients.py:1–6`).
4. `docs/reference/plotting_guide.md:80, 103` — its 99.8% example renders as "99% CI"; lines 104–105 refer to "the function's own default tiers", which do not exist (`funnel.py:106–107`).
5. `README.md:246–267` — lists `statistics/`, `mixed_effect` modules, `measures/logistic/provider_tests` and `algorithms/survival/partial_likelihood`, none of which exist; `README.md:291` test counts are stale.
6. `docs/survival/10_complete_case_study.md` §10.6 (the brief's acceptance target) builds its report by hand: mid-p p-values with non-mid-p exact intervals and string flags. Facility 32 has p = 0.044 and interval 0.036–1.065 (no flag) — an S3 inconsistency in the reference example. Reproducing it with the new layer should consume `CoxPH.test()` instead.

### 4.8 Dependencies and packaging

* Core already depends on Matplotlib; nothing found requires a new dependency to fix the [P0] issues **[run]**.
* The resolved environment is the newest stack on Python 3.12 (pandas 3, numpy 2.5, Matplotlib 3.11) while `requires-python >=3.9` and the only CI job runs Python 3.9. The R5 regression shows the presentation layer must be written for pandas 1.5–3.x semantics (no chained in-place operations) **[run]**.
* The wheel (130 files) contains no `docs/`, `tests/` or `r_reference/` files; `pprof_py/docs/design/presentation/` is outside `docs/source/` and under the `pprof_py.docs*` exclusion, so it is neither published nor packaged **[run]**.

---

## 5. Data-availability matrix (Tier 1 and 2 displays × model families)

● available · ◐ partial · ○ missing. Probed on fitted models **[run]** (`evidence/availability.json`); CoxPH = stratified `CoxPH(ties="efron")`; three-stage fitted with 3 sub-clusters per provider (it raised `LinAlgError` with patient-level clusters on the harness data).

| Display | Logistic FE | Logistic RE | Linear FE | Linear RE | Three-stage | CoxPH |
|---|---|---|---|---|---|---|
| **Funnel, test-consistent limits** | ◐ limits not exposed | ◐ flags test the BLUP, not O/E | ◐ limits not exposed | ◐ as linear FE | ◐ limits not exposed | ◐ O, E, person-time in `test()`; no limits |
| **Caterpillar + volume** (effect scale) | ● `test()` + `attrs["provider_size"]` | ● `test()` + `provider_sizes_` | ● | ● | ● (sizes via stage objects [assumed]) | ● `test()` incl. observed/expected/person_time |
| Caterpillar on measure scale | ● `test_standardized()` | ◐ intervals not from the flagging test | ◐ same | ◐ same | ◐ same | ● (`test()` estimate = O/E, null 1, `test_method="midp"`; but `attrs["measure"]` is `None`) |
| **Provider summary table** | ● | ◐ measure-scale CIs not flag-consistent; direct lacks `n_pop` | ● | ● | ● | ● |
| **Coefficient forest** | ● | ◐ `summary()` has no CIs and no `level` | ● | ● | ● | ● (`exp(coef)`, `lower_95%`, `upper_95%`) |
| Observed vs expected | ● | ● | ● | ● | ● | ● |
| Between-provider variation | ◐ via `DirectIUR` only | ● `sigma_`, `profile_sigma()`, BLUPs | ◐ via `DirectIUR` | ◐ `random_effect_sd_`, BLUPs; no profile interval | ◐ stage-2 σ [assumed] | ○ (`FrailtyCoxPH` not probed) |
| Shrinkage (FE vs RE pair) | ◐ needs both fits; γ (absolute) vs α (deviation) must be aligned in the presentation layer | ◐ | ◐ | ◐ | ○ | ○ |
| Reliability | ● IUR classes from arrays (Bootstrap/SplitHalf: obs, exp, groups; Direct: sizes, estimates, SE); `decile_table()` on Bootstrap and Direct; per-provider `iur_groups_` on Bootstrap | ● | ● | ● | ● | ● |
| Null-calibration diagnostics | ● `z_raw`, `null_mean`, `null_sd`, `null_group`, `z_adjusted` | ● | ● | ● | ● | ● |
| Flag stability | ● re-run `test()` across settings | ● | ● | ● | ◐ `sigma_sensitivity()` raised `ZeroDivisionError` on harness data | ● |
| Multi-measure | ◐ results exist; no container | ◐ | ◐ | ◐ | ◐ | ◐ |
| Data-quality panel | ◐ exclusions ○, `at_bound` ● | ◐ exclusions ○, `at_bound` ○ (KeyError) | ◐ `at_bound` returns all True | ◐ `at_bound` ○ | ◐ | ◐ |

**Field vocabulary by family** (what adapters must map) **[run]**

| Family | `calculate_confidence_intervals(option=)` → interval columns | `summary()` columns | Random-effect SD |
|---|---|---|---|
| Logistic FE | `gamma` → `gamma_lower/upper`; `SM` → `ci_ratio_*`, `ci_rate_*` | `estimate, std_error, stat, p_value, ci_lower, ci_upper` | — |
| Logistic RE | `alpha` → `alpha_lower/upper`; `SM` → `lower/upper` (no observed/expected) | `Estimate, Std.Error, z value, Pr(>\|z\|)` | `sigma_` |
| Linear FE | `gamma` → `lower/upper`; `SM` (key `indirect_ci`) → `lower/upper` | as logistic FE | — |
| Linear RE | `alpha` → `alpha_lower/upper`; `SM` → `lower/upper` | as logistic FE | `random_effect_sd_` |
| Three-stage | default → `ci_ratio_*`, `ci_rate_*` | as logistic FE | — |
| CoxPH | none | `coef, exp(coef), se(coef), z, p, lower_95%, upper_95%` | — |

`test()` attrs: all carry `alternative, bounds, critical, interval, level, measure, null_model, transform` (`inference/decision.py:192–194`) **[read]**; logistic FE/RE, linear FE/RE and three-stage add `test_method` and `reference` (`inference/effect_tests.py:172`); count-test methods add `limits`; logistic FE also adds `provider_size`; CoxPH has `test_method` but no `reference` **[run]**.

---

## 6. Retain, rework, deprecate

**Retain:** `test()`'s schema and `attrs` as the primary input contract; `plotting.style` as the seed of the `Theme` (role names, Okabe–Ito hues, but with darker or outlined variants to meet contrast and grayscale requirements); funnel shape redundancy, hollow "Not tested", legend counts; the standalone funnel's `(fig, ax)` / `ax=` contract; the `limits_df` path as an escape hatch.

**Rework:** all statistics in §4.2 move out of plotting code (funnel limits from the statistical layer; intervals and references from `test()`; forests from `summary()`); `plot_caterpillar` (shapes, labels, defaults matching `PROVIDER_TEST_COLUMNS`, interval drawn as segments rather than errors around the estimate, at-bound handling); funnel limits drawn on a precision grid rather than through providers.

**Deprecate (policy is the maintainer's call, §8.6):** the four per-mixin forest implementations; plot-method kwargs that duplicate theme settings; the default `lower`/`upper` column names. Model methods stay as thin delegates.

---

## 7. Constraints carried into Phase 2

1. Intervals and flags must come from the same `test()` call; `calculate_confidence_intervals()` has four interval schemas and is not flag-consistent for RE measure scales.
2. A funnel needs limits from the statistical layer; no accessor exists. The count-test machinery (`inference/count_tests.py`: per-provider null distributions, mid-p decisions) is the natural source, so discrete limits can be defined to match mid-p flags.
3. Capability gating per family: `at_bound()` (logistic FE only), DataPrep exclusions (not recorded), CoxPH `test()` needing the data again and lacking `reference`, `summary()` vocabularies, `sigma_` vs `random_effect_sd_`.
4. Theme sizes in millimetres; defaults that remain ≥ 7 pt at single-column width.
5. Deterministic export without global side effects (§4.5).
6. Compatibility with pandas 1.5–3.x and Matplotlib 3.5–3.11 (declared floors), on Python ≥ 3.9 unless the floor changes.

## 8. Corrections to the brief's §2 snapshot

| Brief | Verified |
|---|---|
| §2.1 `plotting/` has caterpillar, funnel, coefficient forest, style | `coefficients.py` (caterpillar), `funnel.py`, `style.py`, plus four model mixins; no forest module |
| §2.1 test counts unknown, README 200/26/1 | 1 failed (environmental) / 688 passed / 1 skipped |
| §2.1 "`.github/workflows/` exists; read it" | docs deployment only; no tests in CI |
| §2.2 plot methods return `None`, draw on current figure | funnels return `(Figure, Axes)`; all create a new figure and leave it open |
| §2.2 logistic residual/QQ raise `NotImplementedError` | logistic FE only; logistic RE has no such methods |
| §2.2 IUR has `decile_table()` | `BootstrapIUR` and `DirectIUR`; not `SplitHalfIUR` |
| §2.3 `random_effect_sd_` | linear RE only; logistic RE uses `sigma_` |
| §2.3 `at_bound()` | function `pprof_py.inference.at_bound(model)`; logistic FE only (RE: KeyError; linear: all True) |
| §2.4.1 the two `plot_funnel` entry points differ in return type | refuted: both return `(fig, ax)` at runtime |
| §2.4.5 `plotting.coefficients` docstring mentions coefficient paths | refuted for the code; only the docs page still says so |

## 9. Statistical-layer observations to report (not to fix)

1. Zero-event (at-bound) providers receive `flag = 0` from every test method (§4.3 M2).
2. Score test: z = 0 when null variance < 1e-14 (M9).
3. `at_bound()` semantics outside logistic FE (M4).
4. `LogisticThreeStageModel.fit` raised `LinAlgError` (Cholesky) with patient-level clusters averaging two rows; `sigma_sensitivity()` raised `ZeroDivisionError` with three sub-clusters per provider (possibly an estimated σ of 0). Harness data; not investigated.
5. DataPrep exclusions are logged, not recorded (M3).
6. Logistic RE measure-scale intervals are NaN for zero-event providers (M5).

## 10. Questions for the Phase 2 batch (with recommendations)

1. **Funnel-limit accessor (open question 4).** Recommend yes: a small, tested function in the statistical layer that returns limits on a precision grid from the same null and decision rule as `test()`, including mid-p-consistent discrete limits.
2. **Logistic and linear RE funnels.** Their flags test the BLUP, so no O/E limit can agree with them. Recommend: offer funnels only where the test is a test of the plotted measure (FE, three-stage, CoxPH) and use the caterpillar for RE fits, or define the RE funnel on the scale of the RE test statistic.
3. **Test CI.** Recommend adding a pytest workflow (min and max supported Python) as a separate small PR.
4. **At-bound display semantics.** Recommend the presentation layer show "no finite estimate" from `at_bound()` regardless of `flag`, without changing flags; whether the score test should return NA when the null variance is ~0 is a statistical decision for you.

## 11. Reproduction

```bash
python -m venv venv && . venv/bin/activate && pip install -e ".[dev,docs]"
python -m pytest pprof_py/tests -p no:cacheprovider -q -rfEs --junitxml=junit.xml
(cd pprof_py/docs && sphinx-build -E -b html source _build/html -w sphinx_warnings.log)
cd /tmp   # neutral directory; each harness asserts which tree it imported
python harness/harness_render.py 20   out/N20
python harness/harness_render.py 1000 out/N1000
pip install --target /tmp/cvtools colorspacious      # scratch, not a package dependency
PYTHONPATH=/tmp/cvtools python harness/determinism_access.py main out/checks
python harness/probe_availability.py out/checks/availability.json
python harness/inventory_api.py out/api_inventory.json
```

The harness asserts that `pprof_py` was imported from the clone, uses fixed seeds and the Agg backend, and never modifies the package. `harness/gen.py` is a throwaway generator (log-normal volumes, planted outliers in both directions, tiny zero-event providers, patient clusters), kept outside the package.
