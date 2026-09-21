(diagnostics-guide)=
# Pre-Fit Diagnostics and R Validation

Two tools for checking work before and after fitting: `preflight_report`
inspects your data *before* you call `fit()`, and `validate_against_r`
is the tool behind this package's own
[R compatibility notes](../survival/R_COMPATIBILITY) and
[validation report](../survival/VALIDATION_REPORT) — useful to know
about even if you never run it yourself, since it's what actually
produced the coefficient-level agreement numbers those pages cite.

## `preflight_report`: the same checks `fit()` runs, as a report

```python
from pprof_py.diagnostics.survival import preflight_report

report = preflight_report(
    cohort, covariates=["age", "sex", "diabetes", "comorbidity_count"],
    event="death", duration="time", label="ESRD cohort",
)
print(report.report())
```
```
Pre-flight data diagnostics — ESRD cohort
==========================================
Rows: 4,000

Missing / non-finite values: none found in mapped columns.

Events: 259
Distinct event times: 241 (global; avg tie size 1.1x); largest tied
group in the data overall (no strata given): 2 events at t=0.984

Covariate summary:
                    min      mean   max        std
age                19.8  62.14245  95.0  13.796640
sex                 0.0   0.45575   1.0   0.498100
diabetes            0.0   0.45275   1.0   0.497825
comorbidity_count   0.0   1.32475   7.0   1.128098

No fatal issues found -- fit() should accept this data as-is.
```

`preflight_report` uses the **same column-role vocabulary** as
`fit()` itself — `duration` or `start`/`stop`, `event`, `covariates`,
`strata`, `offset`, `sample_weight` — so the call above mirrors
whatever `CoxPH.fit()` call you're about to make, just with column
*names* (strings) instead of arrays, since it works directly against
the DataFrame before anything is extracted. The returned
`PreflightResult` separates `.fatal` issues (would make `fit()` raise)
from `.warnings` (would fit fine, but worth knowing — a singleton
stratum, a very large tie group, an unusually wide covariate range);
`.report()` renders both into the readable text shown above, and the
structured fields (`n_events`, `max_tie_size`, `n_singleton_strata`,
`covariate_summary`, and more) are available directly on the result
object for programmatic use rather than string-parsing the report.

## `validate_against_r`: what produced the R compatibility numbers

```python
from pprof_py.diagnostics.survival.validate_against_r import run_validation

run_validation(spec_path="validation_spec.yaml")
```

This is the machinery behind
[`R_COMPATIBILITY.md`](../survival/R_COMPATIBILITY) and
[`VALIDATION_REPORT.md`](../survival/VALIDATION_REPORT) — both state
directly that every claim in them "was checked against actual R
output during development," and `run_validation` (plus
`generate_r_script`, which writes the matching `survival::coxph()`
call for a given fitting specification) is that checking process,
not a separate reimplementation of it. It fits the Python side
(`CoxPH`), generates the equivalent R script, and — given either a
live R environment or a `r_results_override` pointing at
already-computed R output — compares coefficients (`rtol=1e-4`
default), log-likelihoods (`rtol=1e-5`), and baseline hazard values
(`rtol=1e-3`), returning a text report of any mismatches. Most
readers of this documentation won't run this directly — its output is
already summarized in the two pages linked above — but if you're
extending this package with a new estimator and want the same
R-comparison rigor the existing survival family has, this is the tool
that provides it, not something to rebuild from scratch.

## What's next

[`deviance_statistics.md`](deviance-statistics-guide) covers the
lower-level numerical building blocks (`saturated_log_likelihood`,
`cox_deviance`, `deviance_ratio`, `bootstrap_cv_se`) that
`deviance_ratio_path_` and every cross-validated Cox-family class's
`cv_mean_deviance_` are actually built from.
