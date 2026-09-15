# This file has been superseded

The top-level [`README.md`](../../README.md) now covers all model
families (logistic, linear, survival), installation, quick-start
examples, validation, performance, and the full package layout.

For survival-specific technical documentation, see:

- [`R_COMPATIBILITY.md`](R_COMPATIBILITY.md) — R convention verification
- [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) — numerical validation results
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — survival-model layering and design rationale

The original content below is retained for reference but is no longer
maintained. All module paths below use the old `coxph` layout; current
paths are documented in the files above.

---

# CoxPH (`pprof_py.models.survival`) — ARCHIVED

> This document was originally the standalone `coxph` package's README.
> That package has been merged into `pprof_py` as its survival-model
> family; the implementation, validation, and behavior described below
> are unchanged -- only the import path (`from pprof_py import CoxPH`)
> and internal module locations (`pprof_py/models/survival/`,
> `pprof_py/algorithms/survival/`, `pprof_py/inference/survival/`) moved.

Native Python/NumPy/pandas Cox Proportional Hazards regression, built to
reproduce [R's `survival::coxph()`](https://cran.r-project.org/package=survival)
as closely as numerically possible — see `docs/VALIDATION_REPORT.md` for
the actual numbers (coefficients, standard errors, log-likelihood,
baseline hazard, and martingale residuals all matching real R output to
1e-8–1e-14 relative error across every supported feature combination) and
`docs/R_COMPATIBILITY.md` for the specific conventions matched, including
a couple of genuinely non-obvious ones (see especially the offset/basehaz
finding).

Built for a specific motivating use case: reproducing a two-stage
CoxPH SMR/SHR (indirect standardization) workflow — a stratified,
offset, weighted, left-truncated Cox model in Stage 1, whose linear
predictor becomes the offset for an offset-only Stage 2 — without
depending on R at runtime.

## Install

```bash
pip install -e .
```

Requires `numpy`, `pandas`, `scipy`, `scikit-learn` (for
`sklearn.base.BaseEstimator`, giving `get_params`/`set_params` for free),
and `numba` (JIT-compiles the hot per-stratum kernels — see Performance
below; a pure-Python fallback covers environments where numba genuinely
can't be installed, at a real cost in speed).

## Quick start

```python
from pprof_py import CoxPH

model = CoxPH(ties="breslow")  # "breslow" (default, matches phregSHR) or "efron" (R's own default)
model.fit(X, duration=time, event=event)

model.coef_                 # coefficients
model.standard_errors_      # model-based (not robust) SE
model.summary()             # coef / exp(coef) / se / z / p / CI, as a DataFrame
model.baseline_hazard_      # DataFrame: stratum, time, hazard, survival
model.martingale_residuals_
```

### Left truncation / counting-process data

```python
model.fit(X, start=start, stop=stop, event=event)
```

### Strata, offset, weights — any combination

```python
model.fit(
    X, start=start, stop=stop, event=event,
    strata=provider,       # own baseline hazard per stratum, shared coefficients
    offset=log_exposure,   # unpenalized, fixed term in the linear predictor
    sample_weight=weight,  # case/frequency weights
)
```

### The two-stage SMR/SHR pattern

```python
stage1 = CoxPH().fit(
    X1, start=start, stop=stop, event=event,
    strata=provider, offset=offset1, sample_weight=weight,
)
xbeta = stage1.predict_linear(X1, offset=offset1)

# Stage 2 can have covariates, or none at all (R's `~ offset(xbeta)`,
# no other terms) -- a genuinely empty (n, 0) design matrix is a
# first-class input here, not a special case:
import pandas as pd
stage2 = CoxPH().fit(
    pd.DataFrame(index=range(len(xbeta))), start=start, stop=stop, event=event,
    offset=xbeta, sample_weight=weight,
)
stage2.log_likelihood_
stage2.baseline_hazard_   # the "expected" side of an SHR/SMR observed-vs-expected comparison
```

### Prediction

```python
model.predict_linear(X_new, offset=offset_new)          # X @ coef_ + offset
model.predict_partial_hazard(X_new, offset=offset_new)   # exp(...)
model.predict_cumulative_hazard(X_new, stratum=...)      # per-subject H(t), DataFrame indexed by time
model.predict_survival_function(X_new, stratum=...)      # exp(-H(t))
```

## What's implemented

| Feature                                                                                           | Status                                                                                                                                      |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Right-censored data                                                                               | ✅ validated against R                                                                                                                      |
| Left truncation / `(start, stop]`                                                                 | ✅ validated against R                                                                                                                      |
| Strata (own baseline hazard, shared coefficients)                                                 | ✅ validated against R                                                                                                                      |
| Offset                                                                                            | ✅ validated against R, including the `basehaz` offset-mean subtlety                                                                        |
| Case weights                                                                                      | ✅ validated against R (model-based SE only, not robust — see `naive.var` note in R-compatibility notes)                                    |
| Breslow ties (default)                                                                            | ✅ validated against R                                                                                                                      |
| Efron ties (R's own default)                                                                      | ✅ validated against R — coefficients, baseline hazard, _and_ martingale residuals each needed their own formula; see R-compatibility notes |
| Coefficients, SE, covariance, Wald z/p, CI, log-likelihood                                        | ✅                                                                                                                                          |
| Baseline cumulative hazard / survival                                                             | ✅ (`basehaz(centered=FALSE)`-equivalent)                                                                                                   |
| Martingale residuals                                                                              | ✅ (left-truncation-correct; Efron-correct)                                                                                                 |
| Two-stage SMR _and_ SHR patterns (Section 34)                                                     | ✅ both validated against R end to end                                                                                                      |
| Prediction (linear, partial hazard, cumulative hazard, survival)                                  | ✅                                                                                                                                          |
| `scikit-learn` conventions (`BaseEstimator`, `get_params`/`set_params`, `coef_`-style attributes) | ✅                                                                                                                                          |
| Exact ties                                                                                        | ❌ not implemented — `NotImplementedError`, see R-compatibility notes                                                                       |
| Robust/sandwich variance, clustering                                                              | ❌ not implemented                                                                                                                          |
| Penalization (L1/L2/elastic net)                                                                  | ❌ not implemented                                                                                                                          |
| Time-dependent covariates, frailty, competing risks                                               | ❌ not implemented                                                                                                                          |
| Formula interface (`Surv(time, event) ~ x1 + x2`)                                                 | ❌ not implemented — pass a numeric design matrix                                                                                           |
| Spark/distributed backend                                                                         | ❌ not implemented — see architecture notes for the seam it would slot into                                                                 |

Everything in the ❌ list is deferred deliberately, not accidentally —
see the "Development Process" staging in the original project brief and
`docs/R_COMPATIBILITY.md`'s closing section for what's known to differ.

## Performance

Correctness came first, but this has now had two real optimization
passes, both driven by profiling against production-realistic data
rather than guessing:

**Pass 1 (pure NumPy):** `cox_partial_likelihood`, `compute_baseline_hazard`,
and `martingale_residuals` originally re-scanned the _entire_ dataset
with a boolean mask for _each_ stratum (`utils/grouping.py` replaced
that with a single sort + contiguous slicing, O(n log n) instead of
O(n \* n_strata)), and the risk-set sweep batched its per-observation
`np.outer` calls into one matrix multiply per event time.

**Pass 2 (numba):** the risk-set sweep, the Breslow and Efron
log-likelihood/score/information accumulation, the baseline-hazard
increments, and the martingale-residual algorithm are now numba-compiled
(`@njit(cache=True)`), used automatically when numba is installed
(it's a default dependency; a pure-Python fallback — the batched version
from Pass 1 — covers environments where it isn't). Every one of these
kernels has a paired test in `tests/test_engine_self_consistency.py`
checking it against its own pure-Python fallback at random (not just
fitted) coefficient values, on top of the R-comparison suite, since a
compiled and an interpreted version of the same formula are exactly the
kind of thing that can quietly diverge.

One thing worth knowing if this runs in an ephemeral environment (a
fresh container or job cluster per run): `cache=True` persists compiled
code to disk, so the ~15-20s one-time JIT compilation cost is paid once
per machine and reused after that — but only if the environment has
persistent disk across runs. In a truly ephemeral environment, that cost
is paid on every invocation.

Indicative wall-clock times on this project's development machine (1
vCPU), fitting a stratified, weighted, offset, left-truncated model (the
full SHR-shaped combination), with a warm numba cache:

| n       | strata | covariates | ties    | fit time |
| ------- | ------ | ---------- | ------- | -------- |
| 2,000   | 10     | 2          | breslow | 0.14s    |
| 10,000  | 50     | 2          | breslow | 0.52s    |
| 50,000  | 250    | 2          | breslow | 2.73s    |
| 200,000 | 3,000  | 6          | breslow | 1.8s     |
| 200,000 | 3,000  | 6          | efron   | 1.9s     |
| 50,000  | 700    | 57         | breslow | 3.8s     |
| 50,000  | 700    | 57         | efron   | 1.8s     |

(The first optimization pass alone got the 200k-row/6-covariate case to
12.0s for Breslow — but left Efron completely unimproved at 17.2s,
because its sweep hadn't been rewired to the compiled kernel; that gap,
and a similar one at higher covariate counts, is what the numba pass
specifically closed. Both the 200k-row and the 57-covariate cases were
checked against real R at that exact scale, not just extrapolated from
the smaller sizes — see `docs/VALIDATION_REPORT.md` and
`coxph/diagnostics/validate_against_r.py` for the tool that ran it;
coefficients, SE, log-likelihood, and baseline hazard all matched to
1e-14–1e-16 relative error at both scales, for both tie methods, so
neither optimization pass traded correctness for speed.)

Fine for exploratory and large production use up to roughly this scale
and well beyond it. The remaining cost is now genuinely the per-stratum
sweep itself, compiled — not Python-level overhead around it, and not
tie-method-specific overhead (Breslow and Efron are now close in cost to
each other, having both been compiled the same way). The natural next
win, if this stops being fast enough, is vectorizing across strata into
a single pass rather than looping over them even in compiled code
(deferred so far because most workloads, including the ones benchmarked
above, are dominated by _many small_ strata rather than a few huge ones,
which limits how much a naive across-strata vectorization would help
without also batching multiple strata's event times together) — or, if
the ~15-20s one-time JIT compilation cost matters more than steady-state
speed in your deployment (see the note on ephemeral environments above),
ahead-of-time compilation to skip paying it at all.

## Validating against your own data

`coxph/diagnostics/` has two tools for exactly the situation of trusting
this package on data it wasn't validated against here:

- **`preflight.py`** inspects a DataFrame for the same things `fit()`
  would reject (missing values, non-numeric columns, start>=stop
  violations, non-binary event codes) plus things worth knowing before
  fitting even though they won't raise (singleton or zero-event strata,
  near-constant covariates, extreme offsets, and — the one that actually
  matters for performance — the largest tied event group _within a
  single stratum_, not the largest in the data overall, which can be a
  very different and much less alarming number).
- **`validate_against_r.py`** fits both this package and real R on the
  same data (right-censored or the two-stage SMR/SHR pattern) and
  reports a structured diff, the same rigor as
  `tests/test_r_comparison.py` but pointed at any dataset via a JSON
  spec rather than the synthetic ones in `r_reference/`. If R isn't
  available in the current environment (e.g. a Databricks cluster), it
  still runs the preflight checks and the Python fit, and writes out a
  ready-to-run R script for a machine that does have R — bring the
  results back and pass `--r-results <dir>` to complete the comparison
  without R ever needing to be installed here. Nothing in either tool
  makes a network call or writes data anywhere outside the output
  directory you specify.

```bash
python -m coxph.diagnostics.validate_against_r --spec my_model.json
```

See `EXAMPLE_SPEC` at the top of `validate_against_r.py` for the JSON
shape (a `stage1`, and an optional `stage2` whose offset is
automatically stage 1's fitted linear predictor).

## Running the tests

```bash
pip install -e ".[dev]"  # or: pip install pytest statsmodels lifelines
python r_reference/generate_data.py         # (re)generate shared synthetic datasets
Rscript r_reference/run_all.R r_reference   # fit the same data in real R -- requires R + the survival package
pytest tests/ -v
```

`tests/test_engine_self_consistency.py` (naive brute-force cross-check)
and `tests/test_validation.py` (input validation) do not require R.
`tests/test_r_comparison.py` reads pre-generated results from
`r_reference/results/` and will report file-not-found errors if the two
commands above haven't been run first.

## Package layout

See `docs/ARCHITECTURE.md` for the full rationale; in short:

```
coxph/
    data/         input validation, the plain-NumPy SurvivalData container
    algorithms/   risk-set sweep, tie-handling (Breslow + Efron implemented; exact stubbed), Newton-Raphson
    statistics/   inference (SE/CI/p), baseline hazard, martingale residuals
    models/       the public CoxPH estimator
    utils/        numerical stability helpers, efficient stratum grouping
    diagnostics/  preflight data checks + the R-comparison harness for your own data
```
