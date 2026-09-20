(survival_validation_report)=
# Validation report

Regenerated on 2026-09-20 from a fresh clone (commit `4e98f58`, version 0.4.0) against real R output: R 4.3.3, `survival` 3.5.8, `glmnet` 4.1.8, on Python 3.12.3 with NumPy 1.26.4, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.17.1 and numba 0.59.1. The reference data and R results were produced by the scripts in `pprof_py/r_reference/`; the exact commands, and the workaround for the tests' split data directories, are in {ref}`survival_validation_tools`.

The numbers below come from that run. Last-digit differences from earlier reports are expected: they depend on the platform's BLAS and on R's build.

## Summary

| Area | Result of the fresh run |
|---|---|
| Phase 1–2 — `CoxPH` vs `coxph()` (`test_r_comparison.py`) | **55 of 55 checks pass** |
| Phase 3 — penalized regression vs `glmnet` (`test_penalized_r_comparison.py`) | all tests pass |
| Phase 3 — variable selection vs `step()` (`test_selector_r_comparison.py`) | **not reproducible** — the datasets it reads have no generator in the repository |
| Phase 4 — cause-specific hazards, robust variance, `tmerge` | pass when both sides use the same tie method |
| Phase 4 — Fine–Gray | right-censored agrees with R; **left-truncated does not** |
| Whole `pprof_py/tests/survival` directory | 227 tests: 200 passed, 26 failed, 1 skipped — categorised in {ref}`survival_validation_tools` |

## Phase 1–2: `CoxPH` against `survival::coxph()` (`test_r_comparison.py`)

Tolerances (from the test file): coefficients, standard errors and z relative 1e-5; log-likelihood relative 1e-6; baseline hazard relative 1e-4; martingale residuals absolute 1e-4 (1e-3 for the `basic` dataset). Every check passed with several orders of magnitude to spare, so no pass depends on a loosened tolerance. Run `python pprof_py/tests/survival/test_r_comparison.py` to print this table.

| Check | Abs diff | Rel diff | Tolerance | Status |
|---|---|---|---|---|
| basic: coef | 3.331e-15 | 9.55e-15 | 1.0e-05 | PASS |
| basic: se | 6.106e-16 | 9.42e-15 | 1.0e-05 | PASS |
| basic: z | 1.350e-13 | 1.46e-14 | 1.0e-05 | PASS |
| basic: loglik(beta) | 1.364e-12 | 9.83e-16 | 1.0e-06 | PASS |
| basic: loglik(null) | 4.547e-13 | 3.12e-16 | 1.0e-06 | PASS |
| basic: baseline hazard | 9.135e-13 | 2.68e-13 | 1.0e-04 | PASS |
| basic: martingale residuals | 2.398e-14 | 4.33e-13 | 1.0e-03 | PASS |
| left_truncation: coef | 1.665e-15 | 4.56e-15 | 1.0e-05 | PASS |
| left_truncation: se | 9.714e-17 | 1.35e-15 | 1.0e-05 | PASS |
| left_truncation: loglik(beta) | 4.547e-13 | 4.58e-16 | 1.0e-06 | PASS |
| left_truncation: baseline hazard | 2.975e-14 | 8.27e-15 | 1.0e-04 | PASS |
| left_truncation: martingale residuals | 1.199e-14 | n/a | 1.0e-04 | PASS |
| strata: coef | 6.106e-16 | 1.25e-15 | 1.0e-05 | PASS |
| strata: se | 5.551e-17 | 8.77e-16 | 1.0e-05 | PASS |
| strata: loglik(beta) | 1.592e-12 | 1.33e-15 | 1.0e-06 | PASS |
| strata: baseline hazard | 2.487e-14 | 8.19e-15 | 1.0e-04 | PASS |
| offset: coef | 1.943e-15 | 4.92e-15 | 1.0e-05 | PASS |
| offset: se | 4.857e-16 | 7.10e-15 | 1.0e-05 | PASS |
| offset: loglik(beta) | 5.002e-12 | 4.85e-15 | 1.0e-06 | PASS |
| offset: baseline hazard | 8.082e-14 | 1.77e-14 | 1.0e-04 | PASS |
| weights: coef | 1.160e-14 | 2.76e-14 | 1.0e-05 | PASS |
| weights: se (model-based, not robust) | 5.551e-17 | 1.06e-15 | 1.0e-05 | PASS |
| weights: loglik(beta) | 1.819e-12 | 1.00e-15 | 1.0e-06 | PASS |
| weights: baseline hazard | 2.824e-13 | 7.08e-14 | 1.0e-04 | PASS |
| combined: coef | 1.190e-09 | 3.03e-09 | 1.0e-05 | PASS |
| combined: se | 6.799e-12 | 1.57e-10 | 1.0e-05 | PASS |
| combined: loglik(beta) | 6.366e-12 | 2.49e-15 | 1.0e-06 | PASS |
| combined: baseline hazard | 5.453e-09 | 1.42e-09 | 1.0e-04 | PASS |
| combined: martingale residuals | 4.969e-09 | n/a | 1.0e-04 | PASS |
| two_stage stage1: coef | 1.190e-09 | 3.03e-09 | 1.0e-05 | PASS |
| two_stage: xbeta (stage1 -> stage2 offset) | 4.285e-09 | 3.78e-07 | 1.0e-05 | PASS |
| two_stage stage2: loglik | 5.514e-08 | 1.47e-11 | 1.0e-06 | PASS |
| two_stage stage2: baseline hazard | 7.244e-10 | 5.96e-10 | 1.0e-04 | PASS |
| smr stage1: coef | 2.776e-16 | 9.38e-16 | 1.0e-05 | PASS |
| smr stage1: se | 4.163e-17 | 7.30e-16 | 1.0e-05 | PASS |
| smr: xbeta (stage1 -> stage2 offset) | 3.553e-15 | 1.24e-13 | 1.0e-05 | PASS |
| smr stage2: coef (covariate alongside the offset) | 3.830e-15 | 9.52e-15 | 1.0e-05 | PASS |
| smr stage2: se | 1.527e-16 | 2.79e-15 | 1.0e-05 | PASS |
| smr stage2: baseline hazard | 1.332e-14 | 4.49e-15 | 1.0e-04 | PASS |
| smr stage2: martingale residuals | 1.621e-14 | n/a | 1.0e-04 | PASS |
| efron/basic: coef | 1.313e-09 | 3.24e-09 | 1.0e-05 | PASS |
| efron/basic: se | 1.528e-11 | 2.40e-10 | 1.0e-05 | PASS |
| efron/basic: loglik(beta) | 1.364e-12 | 1.00e-15 | 1.0e-06 | PASS |
| efron/basic: baseline hazard | 3.908e-09 | 1.24e-09 | 1.0e-04 | PASS |
| efron/basic: martingale residuals | 8.851e-09 | n/a | 1.0e-04 | PASS |
| efron/weights: coef | 4.829e-15 | 1.08e-14 | 1.0e-05 | PASS |
| efron/weights: se (naive/model-based, not vcov's robust) | 1.665e-16 | 2.93e-15 | 1.0e-05 | PASS |
| efron/weights: loglik(beta) | 4.547e-12 | 2.54e-15 | 1.0e-06 | PASS |
| efron/weights: se matches naive.var, NOT var/vcov (robust) | 1.665e-16 | 2.93e-15 | 1.0e-05 | PASS |
| efron/weights: baseline hazard | 1.332e-12 | 3.11e-13 | 1.0e-04 | PASS |
| efron/left_truncation: coef | 1.776e-15 | 4.79e-15 | 1.0e-05 | PASS |
| efron/left_truncation: se | 1.665e-16 | 2.42e-15 | 1.0e-05 | PASS |
| efron/left_truncation: loglik(beta) | 2.160e-12 | 2.21e-15 | 1.0e-06 | PASS |
| efron/left_truncation: baseline hazard | 1.865e-14 | 4.95e-15 | 1.0e-04 | PASS |
| efron/left_truncation: martingale residuals | 1.421e-14 | n/a | 1.0e-04 | PASS |

**55 checks, 0 failing test functions.** Standard errors are compared with R's model-based `se(coef)` / `fit$naive.var`, never `vcov()`, which is the *robust* variance once weights are present.

## Production-scale checks (reported by the authors; not re-run)

The standing suite uses datasets of a few hundred to ~900 rows. The package has separately been checked against real R at production-realistic scale, using `diagnostics/survival/validate_against_r.py`:

1. **200,000 rows, 3,000 strata, 6 covariates**, with offset, weights and left truncation together, under both Breslow and Efron ties: coefficients and SEs matched R to 1e-14–1e-16 relative error, log-likelihood to ~2e-15 (Breslow) / ~1.4e-14 (Efron), and the baseline hazard over all 3,000 strata (21,590 matched time points) to a maximum relative error of ~4.5e-14 (Breslow) / ~2.6e-14 (Efron).
2. **50,000 rows, 700 strata, 57 covariates**, checked the same way, with the same result.

Neither dataset is in the repository. They were not regenerated in the 2026-09-20 re-run; the tool that produced them is documented in {ref}`survival_validation_tools`. Production-scale checking is what surfaced the performance problem that led to the numba kernels — and showed that the first attempt had only accelerated the Breslow path, which correctness tests alone would not have caught.

## Engine self-consistency: compiled vs. pure Python

Every numba-compiled kernel (the risk-set sweep, both tie methods' likelihood/score/information accumulation, both tie methods' baseline-hazard increments, the martingale-residual algorithm, and the robust-variance score kernels) has a paired pure-Python implementation it is checked against directly, at random coefficient values, independent of R (`pprof_py/tests/survival/test_engine_self_consistency.py`). For this kind of numerical code, "still matches R" and "the performance problem is actually fixed" are two separate claims that both need checking.

## Phase 3: penalized regression (`test_penalized_r_comparison.py`)

Validated against R's `glmnet(family="cox")` (glmnet 4.1.8). For every scenario the exact λ sequence glmnet settled on is passed to `PenalizedCoxPH(lambda_path=...)`, so coefficients are compared at identical λ. Every test in the module passes in the fresh run.

Tolerances: coefficient paths absolute 1e-4 (5e-4 for strata/truncation), λ relative 1e-6, CV deviance absolute 0.01.

| Test | Tolerance | Status |
|---|---|---|
| wide lasso (20 covariates, alpha=1): coef path | 1e-4 | PASS |
| wide ridge (alpha=0): coef path (excl. extreme λ[0]) | 1e-4 | PASS |
| wide elastic net (alpha=0.5): coef path | 1e-4 | PASS |
| wide lasso, `standardize=False`: coef path | 1e-4 | PASS |
| wide lasso, first 3 variables unpenalized: coef path | 1e-4 | PASS |
| `lambda_max` vs glmnet's own λ[0] | 1e-6 | PASS |
| auto-generated λ grid (first N points) | 1e-6 | PASS |
| strata: penalized coef path | 5e-4 | PASS |
| offset: penalized coef path | 1e-4 | PASS |
| weights: penalized coef path | 1e-4 | PASS |
| left truncation: penalized coef path | 5e-4 | PASS |
| combined (strata + offset + weights + start/stop) | 5e-4 | PASS |
| basic, heavy ties (27 unique times / 500 rows) | 1e-4 | PASS |
| cross-validation: `cvm` vs `cv.glmnet` (shared fold IDs) | 0.01 | PASS |
| cross-validation: `lambda.min` and `lambda.1se` | 1e-6 | PASS |
| cross-validation: coef at `lambda.min` | 1e-4 | PASS |

`test_penalized_self_consistency.py` additionally verifies that the penalized path at λ → 0 converges to the unpenalized `CoxPH` solution for both Breslow and Efron ties — the only coverage for Efron-tie penalization, since glmnet implements Breslow only.

## Phase 3: variable selection (`test_selector_r_comparison.py`) — not reproducible

`CoxPHSelector` was reported to match R's `extractAIC()`-driven stepwise search step by step (forward, backward, both; AIC and BIC with `k = log(n_events)`; forced variables; strata + offset + weights). Those results cannot be regenerated from the repository: `run_selection.R` and the tests read `selector_test_data.csv` and `selector_strata_data.csv`, and no script in `pprof_py/r_reference/` writes them. On a fresh clone ten of the selector tests fail with `FileNotFoundError`. Until a generator is added, treat the selector's R agreement as reported, not reproduced.

## Phase 4: competing risks, robust variance, time-dependent covariates

Fresh-run status with **the same tie method (Efron) on both sides**. The committed tests fit with this package's default (Breslow) while the R scripts use `coxph()`'s default (Efron); run as committed, the cause-specific and robust checks disagree with R by ~3e-3 for that reason alone.

| Check | Tolerance | Status |
|---|---|---|
| Cause-specific coefficients and SEs, causes 1 and 2 — `cr_simple`, `cr_truncated` | 1e-5 | PASS |
| Fine–Gray transform `fgstart`, `fgstop`, row-for-row — both datasets | 1e-6 | PASS |
| Fine–Gray IPCW weights `fgwt` — `cr_simple` (checked outside the committed tests) | — | PASS (max difference 6.7e-16) |
| Fine–Gray IPCW weights — `cr_truncated` (checked outside the committed tests) | — | **FAIL** — 842 of 3,469 rows differ, by up to 0.445 |
| Fine–Gray fit coefficients — `cr_simple` | 1e-5 | PASS (0.63353294, −0.42879805, identical to R) |
| Fine–Gray fit SE — `cr_simple` | 1e-3 | committed test **fails** because it compares the robust SE with R's model-based `se(coef)`; against R's `robust se` it agrees to 6 digits (0.083041, 0.088158) |
| Fine–Gray fit coefficients — `cr_truncated` | 1e-5 | **FAIL** — 0.37006611, −0.49444791 here vs 0.36671541, −0.49339411 in R |
| Robust + strata + truncation: coefficients, robust SE, naive SE | 1e-5 / 1e-3 / 1e-5 | PASS |
| `tmerge`: `tstart`, `tstop`, death indicator row-for-row; fitted coefficient and SE | 1e-6 / 1e-9 / 1e-5 | PASS |
| `test_timedep.py` (R's own bundled `tmerge` cases, transcribed) | — | PASS |
| `test_finegray_transform.py::test_r_test3_left_truncation` (R's bundled left-truncation case) | 1e-7 | **FAIL** — 2 of 8 weights differ (0.5933 vs 0.5273; 0.7031 vs 0.6250) |

`test_robust_variance.py` and `test_score_residuals.py` cross-check the sandwich variance and score residuals against `statsmodels.PHReg` (installed in the fresh run) and `lifelines` (optional; skipped here).

## What is and isn't covered

Every check above compares this package's output with real R output on shared synthetic datasets built to contain the features that make these models hard to get exactly right: ties from coarse time granularity, staggered entry, unbalanced provider-sized strata, non-trivial weights, correlated predictors, competing events and clustered observations.

It does **not** cover: exact ties (not implemented); real CMS claims data; group lasso, provider-penalized and discrete-time estimators (no R reference — internal tests only); the reported production-scale runs (not re-run); or `FineGrayPH` on left-truncated data (currently disagrees with R).
