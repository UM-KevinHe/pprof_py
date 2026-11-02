(validation_overview)=
# Validation overview

How closely does `pprof_py` reproduce the R implementations it replaces? This section answers that question with two
kinds of evidence, kept separate because they answer different questions.

| Evidence | Question it answers | Where |
|---|---|---|
| **Package test suite** (1,268 tests) | Does the code still match the frozen R outputs committed under `tests/data/`? | runs on every push (`.github/workflows/tests.yml`) |
| **Live comparisons** on synthetic data | Do the R packages and `pprof_py`, run side by side on the same machine and the same input, compute the same numbers for every provider? | {doc}`logistic_fixed_effect` (`pprof::logis_fe()`, and the patient-clustered tests against `pprofR`); the survival family in {doc}`../survival/VALIDATION_REPORT` |

The frozen outputs pin the code to one R run (and were themselves regenerated on a second machine, see the {ref}`survival report <survival_validation_refs_repro>`); a live comparison traces every number to a version, a commit, a data hash
and a call, and is repeated at several data sizes, so it also shows where a difference comes from: a stopping rule, a
bound on a degenerate provider or a different definition of a standard error, as the notes below illustrate.

## Precision at a glance

<div class="pprof-interactive">
<iframe src="../_static/validation/ladder.html" title="Precision ladder, interactive" loading="lazy" width="100%" height="720" style="display:block;width:100%;border:1px solid rgba(128,128,128,0.35);border-radius:4px;background:transparent"></iframe>
<p class="pprof-interactive-links"><a href="../_static/validation/ladder.html" target="_blank" rel="noopener">Open in a new tab</a> · hover a dot for its value and source; a table view lists every value.</p>
</div>
<script>
(function () {
  if (window.pprofInteractiveHeights) return;             /* one listener per page */
  window.pprofInteractiveHeights = true;
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (e.origin !== window.location.origin || !d || d.pprof !== 'height') return;   /* same origin only */
    var frames = document.querySelectorAll('.pprof-interactive iframe');
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].contentWindow !== e.source) continue;                             /* only our own frame */
      var px = Math.ceil(Number(d.height));
      if (px >= 200 && px <= 20000) frames[i].style.height = px + 'px';
    }
  });
})();
</script>

<details class="pprof-static-figure">
<summary>Static figure (SVG, as used in print)</summary>

```{figure} ../_static/validation/precision_ladder.svg
:alt: Largest absolute difference between R and Python per comparison and component, on a log axis with precision bands
:width: 100%

Largest absolute difference between the R and Python values, per comparison and component: the live comparison on
`pprof::ExampleDataBinary` and the five synthetic benchmark sizes. Shaded bands: below 1e−10 is the floating-point
limit; 1e−10 to 1e−5 reflects library or operation-order effects; 1e−5 to 1e−3 reflects convergence tolerance or
rounding of the R output; above 1e−2 calls for an explanation. The one value in that band is γ at 7,000 providers under
R's default tolerance `1e-5`; rerun with `tol = 1e-8` (orange), β and γ agree to 4e−16 and 5e−15, which is also why
the floating-point band is otherwise empty: the live comparison keeps R at its default tolerance, so the agreement is
limited by R's stopping rule, not by the arithmetic. The {ref}`report <validation_logistic_fe>` explains the finding.
```

</details>

```{note}
**Matching R's variance definition.** Two robust standard errors are available for the provider tests.
`variance="robust"` (since 0.5.0; the default is `variance="model"`) is the full sandwich of the joint (γ, β) fit at the average case mix;
`variance="robust_fixed_beta"` treats β as known, which is the form the R implementation computes. A comparison with R
output must therefore request `robust_fixed_beta`, together with the null model, transform and critical value the R
run used; with `variance="robust"` the standardized ratios still agree but the z statistics, and so the flags of borderline
providers, do not. The reasoning behind the new default is in {ref}`the changelog <changelog>` (0.7.0).
```

## Coverage

| Family | Live comparison | Package tests against frozen R output |
|---|---|---|
| Logistic fixed effect | yes ({doc}`logistic_fixed_effect`): `pprof::logis_fe()` on the example data and five benchmark sizes; the four patient-clustered tests against `pprofR` on five clustered cohorts up to 5 M rows ({ref}`section <validation_logistic_clustered>`) | SerBIN (`tests/data/serbin_r`), summary (`logis_fe_summary_r`), AOH; the four patient-clustered tests against pprofR 0.2.0 (`r_reference/pprofr_clustered_*`) |
| Logistic random effect / three-stage | planned | `tests/data/three_stage`, `re_profile` |
| Linear fixed / random effect | planned | linear tests (statsmodels, lme4 one-off checks) |
| Penalized and group lasso | planned (λ path against `grplasso`) | `tests/data/cox_group_lasso`, `provider_group_lasso` |
| Cox family | {doc}`../survival/VALIDATION_REPORT` (frozen R runs, re-run 2026-09-20) | `tests/data/cox_smr`, `iur_r` |

The figures on these pages are regenerated from the JSON run records in `validation/data/` by
`validation/make_figures.py`; the scripts that produced the records are listed in each record's provenance block.

(validation_run_log)=
## Run log

The validation runs behind these pages, newest first. Each run has a JSON record in `validation/data/` with the
versions, commits, data hashes, calls and every value; the package test suite runs on every push and is not listed.

| Date | Comparison | `pprof_py` | R side | Data | Result | Record |
|---|---|---|---|---|---|---|
| 2026-10-30 | Patient-clustered tests (`robust_wald`, `stabilized_wald`, `robust_score`, `stabilized_score`) live against `pprofR::test_fe()`, R at `tol` 1e−5, 1e−8 and 1e−12 ({ref}`report <validation_logistic_clustered>`) | 0.7.0, `745ee3d` | pprofR 0.2.0 (`f23c288`), R 4.5.0, macOS arm64 | five synthetic clustered cohorts, 20,053 to 4,996,790 rows, 200 to 7,000 providers | statistics to 9.0e−5 at R's default `tol`, to 1.4e−9 run to convergence; flags identical throughout | `clustered_tests.json` |
| 2026-10-25 | Frozen R references regenerated on a second machine and compared file by file; package tests rerun against the regenerated set ({ref}`report <survival_validation_refs_repro>`) | 0.7.0 | R 4.5.0, survival 3.8.3, glmnet 4.1.10, pprofR 0.2.0, macOS arm64 (committed: R 4.3.3, survival 3.5.8, glmnet 4.1.8, Ubuntu) | the committed inputs under `r_reference/data/`, 117 output files | 45 files identical, 70 within 5.5e−12; the two Cox cross-validation files differ through glmnet 4.1-8 vs 4.1-10 (confirmed with 4.1-8); 142 of 143 tests pass | `frozen_refs_repro.json` |
| 2026-10-13, extended 2026-10-25 | Scaling benchmark of `pprof::logis_fe()` against `LogisticFixedEffectModel`, R at its default `tol` and rerun at 1e−8 ({ref}`report <validation_logistic_fe>`) | 0.7.0, `9320766` | pprof 1.0.3 (CRAN), R 4.5.0, macOS arm64 | five synthetic datasets, 20,000 to 5,000,000 rows, 200 to 7,000 providers | fit 2–2.7× faster in Python; β and γ to 5.3e−15 at `tol` 1e−8; R's default `tol` leaves small-provider γ off by up to 0.11 | `benchmark.json` |
| 2026-10-13 | Live comparison of `pprof::logis_fe()` against `LogisticFixedEffectModel`: β, γ, standardized measures, Wald, score and exact tests ({ref}`report <validation_logistic_fe>`) | 0.7.0, `9320766` | pprof 1.0.3 (CRAN), R 4.5.0, macOS arm64 | `pprof::ExampleDataBinary`, 7,944 rows, 100 providers | β to 1.0e−6, γ to 7.5e−7, flags identical for all 100 providers; 3 zero-event providers recorded | `logistic_fe_report.json` |
| 2026-09-20 | Survival family against frozen R runs (CoxPH, penalized, competing risks, selection; see {doc}`../survival/VALIDATION_REPORT`) | 0.4.0, `4e98f58` | survival 3.5.8, glmnet 4.1.8, R 4.3.3, Ubuntu | the generated datasets under `r_reference/data/` | documented in the survival report, with tolerances per quantity | `r_reference/results/`, tests under `tests/survival/` |
