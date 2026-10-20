(validation_logistic_fe)=
# Validation report: logistic fixed-effect family

This report covers `LogisticFixedEffectModel`, its standardized measures and its provider tests. It complements the
survival family's {doc}`../survival/VALIDATION_REPORT` and follows the same structure: what was compared against what,
with which tolerances, with which result, and what is not covered.

```{note}
Snapshot of the runs listed below (2026-10-13, `pprof_py` 0.7.0, commit `9320766`). Both comparisons use the CRAN
release of `pprof` (1.0.3, identical to the main branch of `UM-KevinHe/pprof`), compiled from source on macOS (Apple
Silicon, arm64) with R 4.5.0 and Python 3.13 / NumPy 2.2.6. All data is synthetic: the example dataset bundled with
the R package and four generated datasets of increasing size.
```

## Summary

| Check | Result |
|---|---|
| Live comparison, `logis_fe()` vs `LogisticFixedEffectModel` on `pprof::ExampleDataBinary` (7,944 rows, 100 providers) | **β to 1.0e−6, γ to 7.5e−7, standardized ratios to 3.2e−7; Wald, score and exact flags identical for all 100 providers** |
| Zero-event providers (3 of 100) | γ held at different bounds by the two packages (R −6.2 to −10.7, Python −10.2 to −10.9); recorded, not counted |
| Scaling benchmark, 20,000 to 1,000,000 rows (200 to 7,000 providers) | fit: Python ≈ 2× faster than R's C++ (1.1 s vs 2.6 s at 1 M rows); exact test: Python 4–7× slower (14.7 s vs 3.4 s) |
| Side finding: R's default `tol = 1e-5` | leaves γ of providers with 2–5 events off by up to 0.11; at `tol = 1e-8` all 7,000 γ agree with Python to 5.3e−15 (β to 4.4e−16) at no measurable cost in time |

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

Four synthetic datasets (5 covariates, provider effects ~ N(−0.7, 0.6), about 35% events) from 20,000 rows / 200
providers to 1,000,000 rows / 7,000 providers, the scale of a national facility-level measure. Single machine (Apple
Silicon, 64 GB RAM); R single-threaded.

| Rows / providers | Fit R | Fit Python | Exact test R | Exact test Python | Wald + score |
|---|---|---|---|---|---|
| 20,000 / 200 | 0.04 s | 0.03 s | 0.06 s | 0.43 s | < 0.01 s |
| 100,000 / 1,000 | 0.32 s | 0.13 s | 0.31 s | 2.1 s | ≤ 0.02 s |
| 500,000 / 3,500 | 1.67 s | 0.55 s | 1.69 s | 7.4 s | ≤ 0.06 s |
| 1,000,000 / 7,000 | 2.58 s | 1.11 s | 3.38 s | 14.7 s | ≤ 0.10 s |

<div class="pprof-interactive">
<iframe src="../_static/validation/benchmark.html" title="Scaling benchmark, interactive" loading="lazy" width="100%" height="1050" style="display:block;width:100%;border:1px solid rgba(128,128,128,0.35);border-radius:4px;background:transparent"></iframe>
<p class="pprof-interactive-links"><a href="../_static/validation/benchmark.html" target="_blank" rel="noopener">Open in a new tab</a> · hover a point or a bar for the exact timing; the table below lists every number.</p>
</div>

<details class="pprof-static-figure">
<summary>Static figure (SVG, as used in print)</summary>

```{figure} ../_static/validation/benchmark.svg
:alt: Runtime of fit and exact test for R and Python against data size, both axes logarithmic
:width: 85%

Runtime against data size. The fit scales linearly on both sides with Python about twice as fast; the exact
Poisson-binomial provider test is the one step where Python is slower, by a factor of 4–7.
```

</details>

**Side finding: R's default tolerance and small providers.** In the benchmark β agrees to 1e−7 at every size, but the
largest γ difference grows with the number of providers (0.005 at 1,000; 0.114 at 7,000), entirely in providers with
very few events (2 events: 0.114; 4 events: 0.016; 50 or more events: 0). Rerunning R with `tol = 1e-8` instead of the
default 1e−5 on the same generated data, all 7,000 γ agree with Python to 5.3e−15 and β to 4.4e−16, and the R fit takes
2.7 s instead of 2.6 s. The cause is the stopping rule, not the algorithm: the Newton iteration converges slowly in
directions with large |γ| and little information. When comparing against an R run, use
the tolerance that run used rather than the package default; `pprof_py` stops at 1e−8 by default.

## What is and isn't covered

- Live comparisons use the CRAN build of `pprof`, so they cover the published algorithm and its single-threaded
  implementation only.
- One example dataset and one set of defaults for the live comparison; confidence intervals, the empirical null, the
  robust variances and the Firth correction were not compared live.
- Neither side's `cutoff` screened out a provider on the live data, so the screening logic is not exercised here.
- The provider tests were compared on the example dataset only; on the benchmark datasets the comparison covers β and γ.
- Run records (JSON) are in `validation/data/`; each carries the versions, commits, data hash and calls of its run.
