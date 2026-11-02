(validation_logistic_fe)=
# Validation report: logistic fixed-effect family

This report covers `LogisticFixedEffectModel`, its standardized measures and its provider tests. It complements the
survival family's {doc}`../survival/VALIDATION_REPORT` and follows the same structure: what was compared against what,
with which tolerances, with which result, and what is not covered.

```{note}
Snapshot of the runs listed below (2026-10-13, extended on 2026-10-25 and 2026-10-30; `pprof_py` 0.7.0, commits `9320766`
and `745ee3d`). The first two comparisons use the CRAN release of `pprof` (1.0.3, identical to the main branch of
`UM-KevinHe/pprof`); the patient-clustered tests use `pprofR` 0.2.0 (branch `Yubo` of `UM-KevinHe/pprof`, commit
`f23c288`, the version the package tests reference). Both R packages were compiled from source on macOS (Apple Silicon,
arm64) with R 4.5.0; Python 3.13 / NumPy 2.2.6. All data is synthetic: the example dataset bundled with the R package,
five generated datasets of increasing size and five generated patient-clustered cohorts.
```

## Summary

| Check | Result |
|---|---|
| Live comparison, `logis_fe()` vs `LogisticFixedEffectModel` on `pprof::ExampleDataBinary` (7,944 rows, 100 providers) | **β to 1.0e−6, γ to 7.5e−7, standardized ratios to 3.2e−7; Wald, score and exact flags identical for all 100 providers** |
| Zero-event providers (3 of 100) | γ held at different bounds by the two packages (R −6.2 to −10.7, Python −10.2 to −10.9); recorded, not counted |
| Scaling benchmark, 20,000 to 5,000,000 rows (200 to 7,000 providers) | fit: Python 2–2.7× faster than R's C++ (4.6 s vs 12.5 s at 5 M rows); exact test: Python 4–7× slower up to 1 M rows (14.7 s vs 3.4 s), 1.7× faster at 5 M rows (31 s vs 53 s) |
| Side finding: R's default `tol = 1e-5` | leaves γ of providers with 2–5 events off by up to 0.11; at `tol = 1e-8` all 7,000 γ agree with Python to 5.3e−15 (β to 4.4e−16) at no measurable cost in time |
| Patient-clustered tests, `pprofR::test_fe()` vs `test(test_method=...)` on five clustered cohorts (20,053 to 4,996,790 rows, 200 to 7,000 providers) | **robust and stabilized Wald and score statistics to 9.0e−5 with R at its default tolerance, to 1.4e−9 with R run to convergence; p-values to 2.1e−9; flags identical for every provider, test and cohort** |
| Side finding: string ids | with provider and patient ids as strings, Python's fit of the 5 M-row cohort takes 14.7 s; recoded to integers, 4.9 s. pprofR fits it in 2.0 s |

## Live comparison against `pprof::logis_fe()`

**Protocol.** One csv is written once and read by both sides (SHA-256 recorded). R fits `logis_fe()` with package
defaults (SerBIN, `tol = 1e-5`, `bound = 10`, `cutoff = 10`), computes indirect and direct standardized ratios and rates
with `SM_output()` (reference: the median γ) and runs `test()` with the `wald`, `score` and `exact.poisbinom` methods.
Python fits `LogisticFixedEffectModel().fit()` with package defaults (SerBIN, `tol = 1e-8`, `bound = 10`,
`cutoff = 10`) and calls `calculate_standardized_measures()` and `test()` with matching arguments. Results are aligned by
provider id, never by row position, and compared as max |R − Py| over providers; for the tests, the share of identical
flags.

**Tolerances.** Set from the weaker stopping rule, not from the results: R stops when the change between iterations
falls below 1e−5 and Python below 1e−8, so estimates can only be expected to agree to about 1e−5. Standardized rates are
percentages, so their tolerance is 1e−4; p-values 1e−3; flags must be identical.

| Quantity | max \|R − Py\| | Tolerance | Verdict |
|---|---|---|---|
| β (5 covariates) | 1.03e−6 | 1e−5 | agree |
| γ (97 providers with events) | 7.50e−7 | 1e−5 | agree |
| Indirect standardized ratio / rate | 1.50e−7 / 5.81e−6 | 1e−5 / 1e−4 | agree |
| Direct standardized ratio / rate | 3.22e−7 / 1.25e−5 | 1e−5 / 1e−4 | agree |
| Wald test: flags / p-values | 100 / 100 identical; 5.4e−5 | identical / 1e−3 | agree |
| Score test: flags / p-values | 100 / 100 identical; 9.2e−7 | identical / 1e−3 | agree |
| Exact (Poisson-binomial) test: flags / p-values | 100 / 100 identical; 9.1e−7 | identical / 1e−3 | agree |

<div class="pprof-interactive">
<iframe src="../_static/validation/explorer.html" title="Per-provider explorer, interactive" loading="lazy" width="100%" height="1500" style="display:block;width:100%;border:1px solid rgba(128,128,128,0.35);border-radius:4px;background:transparent"></iframe>
<p class="pprof-interactive-links"><a href="../_static/validation/explorer.html" target="_blank" rel="noopener">Open in a new tab</a> · choose a quantity, hover providers, sort the linked table; Figure 3 is a Bland–Altman view.</p>
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

```{figure} ../_static/validation/logistic_fe_live.svg
:alt: Left, provider effects from R against Python with the identity line; right, per-provider differences on a log scale against the tolerance
:width: 100%

Left: γ̂ from R against γ̂ from Python for all 100 providers; the three zero-event providers (orange) are the only ones
off the identity line. Right: per-provider |R − Py| for six quantities on a log scale, with the tolerance in orange.
```

</details>

**Zero-event providers.** A provider with no events (or only events) has a maximum-likelihood effect of −∞ (or +∞).
Both packages hold such a γ at a bound and document that it is unreliable, but they do not stop at the same place.
Quantities that depend on that γ directly (direct standardization, the Wald test) therefore differ for those providers;
quantities that do not (indirect standardization, the score and exact tests) still agree. These providers are excluded
from the verdict and the difference is reported rather than hidden by a tolerance.

**Why the Wald p-values differ more than the others.** The two Wald tests use different standard errors: `pprof_py` uses
the variance at the average case mix (`variances_["gamma_case_mix"]`), R the diagonal of the inverse information. On
this data the two are close (5.4e−5 in p); with uncentred covariates and small providers the gap can widen. The score
and exact tests do not depend on that standard error and agree to 1e−6.

## Scaling benchmark

Five synthetic datasets (5 covariates, provider effects ~ N(−0.7, 0.6), about 35% events) from 20,000 rows / 200
providers to 5,000,000 rows / 7,000 providers: 7,000 providers is the scale of a national facility-level measure, and
the last two sizes bracket the rows per provider seen in practice (about 140 and about 700). Single machine (Apple
Silicon, 64 GB RAM); R single-threaded, Python with NumPy's default threading.

| Rows / providers | Fit R | Fit Python | Exact test R | Exact test Python | Wald + score |
|---|---|---|---|---|---|
| 20,000 / 200 | 0.04 s | 0.03 s | 0.06 s | 0.43 s | < 0.01 s |
| 100,000 / 1,000 | 0.32 s | 0.13 s | 0.31 s | 2.1 s | ≤ 0.02 s |
| 500,000 / 3,500 | 1.67 s | 0.55 s | 1.69 s | 7.4 s | ≤ 0.06 s |
| 1,000,000 / 7,000 | 2.58 s | 1.11 s | 3.38 s | 14.7 s | ≤ 0.10 s |
| 5,000,000 / 7,000 | 12.5 s | 4.63 s | 53.4 s | 31.1 s | ≤ 0.53 s |

<div class="pprof-interactive">
<iframe src="../_static/validation/benchmark.html" title="Scaling benchmark, interactive" loading="lazy" width="100%" height="1050" style="display:block;width:100%;border:1px solid rgba(128,128,128,0.35);border-radius:4px;background:transparent"></iframe>
<p class="pprof-interactive-links"><a href="../_static/validation/benchmark.html" target="_blank" rel="noopener">Open in a new tab</a> · hover a point or a bar for the exact timing; the table below lists every number.</p>
</div>

<details class="pprof-static-figure">
<summary>Static figure (SVG, as used in print)</summary>

```{figure} ../_static/validation/benchmark.svg
:alt: Runtime of fit and exact test for R and Python against data size, both axes logarithmic
:width: 85%

Runtime against data size. The fit scales linearly on both sides with Python 2–2.7× faster. The exact
Poisson-binomial provider test is slower in Python by a factor of 4–7 up to 1,000,000 rows, but its cost grows with
the number of providers, whereas R's grows with the rows per provider: at 5,000,000 rows (about 700 rows per
provider) R takes 53 s and Python 31 s.
```

</details>

**Side finding: R's default tolerance and small providers.** In the benchmark β agrees to 1e−7 at every size, but the
largest γ difference at R's default `tol = 1e-5` grows with the number of providers with very few events (0.005 at
1,000 providers; 0.114 at 7,000 providers and 1,000,000 rows, entirely in providers with 2 events: 0.114; 4 events:
0.016; 50 or more events: 0). At 5,000,000 rows the same 7,000 providers have about 700 rows each, few have so few
events, and the difference is 9.5e−4. Rerunning R with `tol = 1e-8` on the same generated data removes it at every
size, at a cost of 0.1 to 0.3 s in the R fit:

| Rows / providers | max \|R − Py\| in γ, R at `tol = 1e-5` | R at `tol = 1e-8` | R fit, 1e−5 → 1e−8 |
|---|---|---|---|
| 20,000 / 200 | 7.4e−7 | 7.4e−7 | 0.04 → 0.06 s |
| 100,000 / 1,000 | 4.6e−3 | 9.5e−6 | 0.32 → 0.40 s |
| 500,000 / 3,500 | 5.3e−4 | 1.2e−7 | 1.67 → 1.97 s |
| 1,000,000 / 7,000 | 1.1e−1 | 5.3e−15 | 2.58 → 2.92 s |
| 5,000,000 / 7,000 | 9.5e−4 | 3.9e−7 | 12.5 → 12.8 s |

The cause is the stopping rule, not the algorithm: the Newton iteration converges slowly in directions with large |γ|
and little information, and a tolerance on the change between iterations stops it too early there. When comparing
against an R run, use the tolerance that run used rather than the package default; `pprof_py` stops at 1e−8 by
default.

(validation_logistic_clustered)=
## Live comparison of the patient-clustered tests against `pprofR::test_fe()`

`pprof_py` 0.7.0 added the four patient-clustered provider tests of `pprofR` (`robust_wald`, `stabilized_wald`,
`robust_score`, `stabilized_score`), which treat a patient's rows as correlated. The package tests check them against
R values frozen under `r_reference/` on seven cohorts of at most 5,000 rows. This comparison runs the two packages side
by side on five larger cohorts, 20,053 to 4,996,790 rows, with the same questions as above: does every statistic agree,
by how much, and how do the run times compare.

**Data.** Five synthetic cohorts generated by `gen_data.py` (seed 2026; SHA-256 of each file in the run record). Each
patient has 1 to 8 rows; a patient effect (sd 0.8) and an AR(1) term (correlation 0.5 between consecutive rows) make a
patient's rows correlated; 30% of patients also visit one or two other providers, so a patient can form clusters at
several providers and the rows of patients and providers interleave in time. Provider effects are N(−0.3, 0.5), three
covariates with β = (0.5, −0.4, 0.3), Bernoulli outcome (41% events). No provider has fewer than 11 patients and none
has zero or only events, so there are no degenerate providers to set aside.

**Protocol.** One csv is written once and read by both sides. R: `fe_data_prep()` (`cutoff = 10`) and `logis_fe()`
(SerBIN, `bound = 10`) three times: at its default `tol = 1e-5` with step-halving, at `tol = 1e-8`, and at
`tol = 1e-12` without step-halving (`backtrack = FALSE`), which is how pprofR's own reference values are made; then
`test_fe()` with the four clustered tests and the plain score test, `confint(test = "robust_wald")` for the interval
and `summary_fe_covar()` with the model-based and the robust Wald test for β. Python: `fit(obs_id_var="patient",
order_var="time")` with package defaults (`tol = 1e-8`), `test(test_method=...)` for the same five tests and
`summary(variance_type="robust")` for β. Results are aligned by provider id. Compared per provider: γ, the statistic,
the p-value and the flag of each test, the standard error of the two Wald tests and the robust Wald interval; per
covariate: β, its robust standard error and statistic. Tolerances as above: 1e−5 for estimates and statistics, 1e−3
for p-values, flags identical.

| max \|R − Py\| over providers, R at its default `tol = 1e-5` | 20,053 rows | 100,209 | 499,424 | 999,067 | 4,996,790 |
|---|---|---|---|---|---|
| γ | 4.2e−7 | 4.1e−7 | 7.5e−6 | 3.9e−5 | 5.5e−7 |
| robust Wald z (SE) | 5.4e−8 (1.2e−7) | 7.1e−8 (8.2e−8) | 1.0e−5 (1.9e−6) | 9.0e−5 (7.4e−6) | 5.2e−7 (4.2e−8) |
| stabilized Wald z (SE) | 4.2e−7 (6.3e−8) | 4.7e−7 (5.4e−8) | 7.5e−6 (8.8e−7) | 1.7e−5 (5.6e−6) | 1.9e−6 (2.5e−8) |
| robust score z | 5.6e−10 | 1.1e−10 | 7.3e−10 | 8.9e−10 | 4.1e−11 |
| stabilized score z | 4.5e−10 | 1.1e−10 | 6.8e−10 | 9.5e−10 | 4.2e−11 |
| p-values, largest of the five tests | 7.2e−10 | 1.3e−10 | 1.4e−9 | 2.1e−9 | 5.6e−11 |
| β with robust SE (β, SE) | 4.5e−10, 4.3e−11 | 5.9e−11, 3.2e−12 | 4.5e−10, 1.4e−11 | 5.3e−10, 1.5e−11 | 2.7e−11, 2.0e−13 |
| flags identical, five tests | 200 of 200 | 1,000 of 1,000 | 3,500 of 3,500 | 7,000 of 7,000 | 7,000 of 7,000 |

Every difference is inside its tolerance and every flag is identical, with R at its default tolerance and in both
reruns. The Wald statistics follow γ: their differences are the differences in γ divided by the standard error, and
the γ differences are R's stopping rule, as in the scaling benchmark above. The score statistics do not depend on the
provider effects beyond the median, so they agree to 1e−9 at every size and tolerance; the same holds for β and the
p-values.

**Running R to convergence.** With `tol = 1e-8`, the three cohorts of 499,424 rows and more agree with Python to
5.1e−15 in γ and 1.4e−13 in the statistics: both packages iterate the same Newton scheme (SerBIN) from the same start,
and when they take the same number of steps the iterates agree to rounding. The two smallest cohorts stay at 4.2e−7
and 1.6e−7 in γ, unchanged when Python's own tolerance is tightened to 1e−12: there pprofR's step-halving stops one
step early. Without it (`tol = 1e-12`, `backtrack = FALSE`) those two agree to 6.2e−14, and the largest difference
over all five cohorts and every quantity is 1.4e−9 (robust Wald z at 999,067 rows), the size of the last Newton step.

<div class="pprof-interactive">
<iframe src="../_static/validation/clustered_tests.html" title="Patient-clustered tests: largest difference per cohort and quantity, interactive" loading="lazy" width="100%" height="1500" style="display:block;width:100%;border:1px solid rgba(128,128,128,0.35);border-radius:4px;background:transparent"></iframe>
<p class="pprof-interactive-links"><a href="../_static/validation/clustered_tests.html" target="_blank" rel="noopener">Open in a new tab</a> · hover a dot for its value and R run; the tables hold the flags, the times and every value.</p>
</div>

<details class="pprof-static-figure">
<summary>Static figure (SVG, as used in print)</summary>

```{figure} ../_static/validation/clustered_tests.svg
:alt: One row per cohort, one dot per quantity and R run: the largest absolute difference between pprofR and pprof_py on a log scale with the precision bands
:width: 100%

Largest |R − Py| over providers for each quantity (γ, the five statistics, their p-values, the Wald standard errors
and interval, β with its robust standard error) and each R run. Green: pprofR at its default tolerance; orange: rerun
at `tol = 1e-8`; blue: rerun at `tol = 1e-12` without step-halving. Each run's largest value in a row is labelled.
```

</details>

**Run times** (seconds, single machine; R single-threaded, fit time without its data preparation; Python fit time
including data preparation):

| Cohort | Fit · R | Fit · Py, string ids | Fit · Py, integer ids | Robust Wald · R / Py | Stabilized Wald · R / Py | Score · R / Py |
|---|---|---|---|---|---|---|
| 20,053 rows, 200 providers | 0.01 | 0.05 | 0.03 | 0.004 / 0.003 | 0.004 / 0.006 | 0.003 / 0.001 |
| 100,209 rows, 1,000 providers | 0.04 | 0.26 | 0.14 | 0.016 / 0.009 | 0.017 / 0.028 | 0.019 / 0.004 |
| 499,424 rows, 3,500 providers | 0.21 | 1.31 | 0.58 | 0.088 / 0.061 | 0.088 / 0.18 | 0.059 / 0.014 |
| 999,067 rows, 7,000 providers | 0.38 | 2.73 | 1.17 | 0.17 / 0.18 | 0.19 / 0.55 | 0.13 / 0.027 |
| 4,996,790 rows, 7,000 providers | 2.01 | 14.72 | 4.88 | 1.15 / 1.17 | 1.25 / 3.43 | 0.63 / 0.11 |

Three things stand out. `pprofR` fits the 5 M-row cohort in 2.0 s, where the CRAN `pprof` of the scaling benchmark
takes 12.5 s on a dataset of the same size: it is a different implementation of the same algorithm. Python's fit takes
14.7 s on this cohort against 4.6 s in the benchmark; the difference is the ids. The csv carries provider and patient
ids as strings, and recoding them to integers before `fit()` brings the time to 4.9 s, so about ten seconds go into
handling string ids during data preparation at this size (fitting without `obs_id_var` saves only one second). The
two robust tests cost the same on both sides; the two stabilized tests, which build the pooled correlation matrix
R̂, take 2.7× longer in Python at 5 M rows (3.4 s against 1.25 s); the plain score test is 6× faster in Python.

The run record is `validation/data/clustered_tests.json` (versions, commits, data hashes, the calls, every value); the
scripts are `gen_data.py`, `run.R` and `run.py`.

## What is and isn't covered

- Live comparisons use the CRAN build of `pprof`, so they cover the published algorithm and its single-threaded
  implementation only.
- One example dataset and one set of defaults for the live comparison of `pprof::logis_fe()`; confidence intervals,
  the empirical null and the Firth correction were not compared live. The four patient-clustered tests are compared
  live against `pprofR` above (statistics, p-values, flags, Wald standard errors and the robust Wald interval) on
  Bernoulli cohorts; binomial rows (`n_var`) and the `cutoff` screening are exercised only in the package tests.
- Neither side's `cutoff` screened out a provider on the live data, so the screening logic is not exercised here.
- The provider tests were compared on the example dataset only; on the benchmark datasets the comparison covers β and γ.
- Run records (JSON) are in `validation/data/`; each carries the versions, commits, data hashes and calls of its run.
