# Validation report

Generated 2026-09-04 by running `tests/test_r_comparison.py` against real R 4.3.3 with `survival` 3.5.8, not a hand-derived expectation. Re-running `python r_reference/generate_data.py && Rscript r_reference/run_all.R r_reference && python tests/test_r_comparison.py` reproduces every number below from scratch.

Tolerances (`tests/test_r_comparison.py`): coefficients/SE/z relative tolerance 1e-5, log-likelihood relative tolerance 1e-6, baseline hazard relative tolerance 1e-4, martingale residuals absolute tolerance 1e-4. All were reached with wide margin -- see the achieved differences below -- so none of the passes here depend on a loosened tolerance masking a real discrepancy.

## Production-scale checks (not part of the standing suite below)

Everything in the standing suite runs on datasets of a few hundred to ~900 rows -- big enough to exercise every code path, small enough to keep in the repo and re-run in seconds. Separately, the package has now been checked against real R at production-realistic scale **twice**, using `diagnostics/survival/validate_against_r.py`:

1. **200,000 rows, 3,000 strata, 6 covariates**, offset, weights, and left truncation together (the full SHR-shaped combination), both Breslow and Efron ties. All 6 coefficients and their SEs matched R to 1e-14 to 1e-16 relative error, log-likelihood to ~2e-15 (Breslow) / ~1.4e-14 (Efron), and baseline hazard across all 3,000 strata (21,590 matched time points) to a max relative error of ~4.5e-14 (Breslow) / ~2.6e-14 (Efron).
2. **50,000 rows, 700 strata, 57 covariates** (matching a real production covariate count), checked the same way, with the same result: no discrepancy beyond floating-point noise.

This scale of check exists because it's what actually caught real bugs during development that the small standing suite did not: production-scale validation first surfaced a performance problem (documented in `docs/README.md`'s Performance section, ~30s for a 200k-row fit), and the first attempt at fixing it (adding numba JIT compilation) turned out to only have been wired into the Breslow tie method -- `EfronTies` was still calling the uncompiled sweep directly, so it saw zero speedup and would have shipped that way if the fix had only been checked for whether results were still correct (they were) rather than whether the actual performance problem was actually solved (it wasn't, for Efron). Re-running the R comparison at this scale for *both* tie methods, after fixing that, is what confirms the fix is complete, not just plausible.

Neither 200k-row nor 57-covariate dataset is checked into this repository (both are synthetic but sizeable, and the point was to exercise scale, not to add another fixture); the tool that generated and ran these checks is, and is meant to be pointed at real data next.

## Engine self-consistency: compiled vs. pure-Python, not just Python vs. R


Every numba-compiled kernel (the risk-set sweep, both tie methods' likelihood/score/information accumulation, both tie methods' baseline-hazard increments, and the martingale-residual algorithm) has a paired pure-Python implementation it is checked against directly, at random (not just fitted) coefficient values, independent of R. This is what caught the Breslow-only-numba gap described above during development -- it wasn't the R-comparison tests that caught it (those passed throughout, since Efron's *uncompiled* path was still correct, just slow), it was the combination of noticing a benchmark hadn't improved and then verifying with a profiler which specific function was still the bottleneck. The lesson generalizes: for this kind of numerical code, 'still matches R' and 'the performance problem is actually fixed' are two separate claims that both need checking, not one implying the other.

## Standing test suite

| Test | Abs diff | Rel diff | Tolerance | Status |
|---|---|---|---|---|
| basic: coef | 2.220e-15 | 5.48e-15 | 1.0e-05 | PASS |
| basic: se | 3.469e-16 | 5.55e-15 | 1.0e-05 | PASS |
| basic: z | 5.684e-14 | 7.13e-15 | 1.0e-05 | PASS |
| basic: loglik(beta) | 1.364e-12 | 9.83e-16 | 1.0e-06 | PASS |
| basic: loglik(null) | 4.547e-13 | 3.12e-16 | 1.0e-06 | PASS |
| basic: baseline hazard | 2.989e-13 | 8.78e-14 | 1.0e-04 | PASS |
| basic: martingale residuals | 1.532e-14 | 8.03e-13 | 1.0e-03 | PASS |
| left_truncation: coef | 2.276e-15 | 7.79e-15 | 1.0e-05 | PASS |
| left_truncation: se | 2.082e-16 | 2.89e-15 | 1.0e-05 | PASS |
| left_truncation: loglik(beta) | 4.547e-13 | 4.58e-16 | 1.0e-06 | PASS |
| left_truncation: baseline hazard | 3.553e-14 | 9.87e-15 | 1.0e-04 | PASS |
| left_truncation: martingale residuals | 1.288e-14 | n/a | 1.0e-04 | PASS |
| strata: coef | 2.776e-16 | 1.01e-15 | 1.0e-05 | PASS |
| strata: se | 4.163e-17 | 6.57e-16 | 1.0e-05 | PASS |
| strata: loglik(beta) | 1.364e-12 | 1.14e-15 | 1.0e-06 | PASS |
| strata: baseline hazard | 7.105e-14 | 2.34e-14 | 1.0e-04 | PASS |
| offset: coef | 6.439e-15 | 1.63e-14 | 1.0e-05 | PASS |
| offset: se | 3.469e-16 | 5.07e-15 | 1.0e-05 | PASS |
| offset: loglik(beta) | 2.956e-12 | 2.87e-15 | 1.0e-06 | PASS |
| offset: baseline hazard | 3.304e-13 | 7.22e-14 | 1.0e-04 | PASS |
| weights: coef | 1.865e-14 | 4.44e-14 | 1.0e-05 | PASS |
| weights: se (model-based, not robust) | 4.927e-16 | 8.69e-15 | 1.0e-05 | PASS |
| weights: loglik(beta) | 2.274e-13 | 1.26e-16 | 1.0e-06 | PASS |
| weights: baseline hazard | 4.730e-13 | 1.18e-13 | 1.0e-04 | PASS |
| combined: coef | 9.521e-10 | 2.42e-09 | 1.0e-05 | PASS |
| combined: se | 5.439e-12 | 1.26e-10 | 1.0e-05 | PASS |
| combined: loglik(beta) | 6.366e-12 | 2.49e-15 | 1.0e-06 | PASS |
| combined: baseline hazard | 4.362e-09 | 1.13e-09 | 1.0e-04 | PASS |
| combined: martingale residuals | 3.975e-09 | n/a | 1.0e-04 | PASS |
| two_stage stage1: coef | 9.521e-10 | 2.42e-09 | 1.0e-05 | PASS |
| two_stage: xbeta (stage1 -> stage2 offset) | 3.428e-09 | 3.02e-07 | 1.0e-05 | PASS |
| two_stage stage2: loglik | 4.411e-08 | 1.17e-11 | 1.0e-06 | PASS |
| two_stage stage2: baseline hazard | 5.793e-10 | 4.77e-10 | 1.0e-04 | PASS |
| smr stage1: coef | 3.331e-16 | 1.19e-15 | 1.0e-05 | PASS |
| smr stage1: se | 4.857e-17 | 8.93e-16 | 1.0e-05 | PASS |
| smr: xbeta (stage1 -> stage2 offset) | 3.553e-15 | 2.09e-13 | 1.0e-05 | PASS |
| smr stage2: coef (covariate alongside the offset) | 3.830e-15 | 9.52e-15 | 1.0e-05 | PASS |
| smr stage2: se | 1.527e-16 | 2.79e-15 | 1.0e-05 | PASS |
| smr stage2: baseline hazard | 4.885e-15 | 3.93e-15 | 1.0e-04 | PASS |
| smr stage2: martingale residuals | 1.799e-14 | n/a | 1.0e-04 | PASS |
| efron/basic: coef | 1.970e-09 | 4.86e-09 | 1.0e-05 | PASS |
| efron/basic: se | 2.292e-11 | 3.60e-10 | 1.0e-05 | PASS |
| efron/basic: loglik(beta) | 1.137e-12 | 8.36e-16 | 1.0e-06 | PASS |
| efron/basic: baseline hazard | 5.861e-09 | 1.86e-09 | 1.0e-04 | PASS |
| efron/basic: martingale residuals | 1.328e-08 | n/a | 1.0e-04 | PASS |
| efron/weights: coef | 3.220e-15 | 7.23e-15 | 1.0e-05 | PASS |
| efron/weights: se (naive/model-based, not vcov's robust) | 9.714e-17 | 1.85e-15 | 1.0e-05 | PASS |
| efron/weights: loglik(beta) | 4.093e-12 | 2.29e-15 | 1.0e-06 | PASS |
| efron/weights: se matches naive.var, NOT var/vcov (robust) | 9.714e-17 | 1.85e-15 | 1.0e-05 | PASS |
| efron/weights: baseline hazard | 1.418e-12 | 3.30e-13 | 1.0e-04 | PASS |
| efron/left_truncation: coef | 1.943e-15 | 6.21e-15 | 1.0e-05 | PASS |
| efron/left_truncation: se | 1.527e-16 | 2.10e-15 | 1.0e-05 | PASS |
| efron/left_truncation: loglik(beta) | 2.160e-12 | 2.21e-15 | 1.0e-06 | PASS |
| efron/left_truncation: baseline hazard | 1.688e-14 | 4.48e-15 | 1.0e-04 | PASS |
| efron/left_truncation: martingale residuals | 1.421e-14 | n/a | 1.0e-04 | PASS |

**55 checks, 0 failing test function(s), plus 65 total tests passing across the full `pytest tests/` run** (this file's numbers cover only `test_r_comparison.py`; the rest are `test_engine_self_consistency.py` -- including the compiled-vs-Python cross-checks above -- `test_diagnostics.py`, and `test_validation.py` — all under `tests/survival/`).

---

## Phase 3: Penalized regression (test_penalized_r_comparison.py)

Validated against R's `glmnet(family="cox")` (glmnet 4.1-8) on shared synthetic datasets. For every scenario, the exact lambda sequence glmnet settled on is passed to `PenalizedCoxPH(lambda_path=...)`, so this compares coefficients at identical lambda values.

Tolerances: coefficient paths absolute tolerance 1e-4 (tight) to 5e-4 (strata/truncation), lambda relative tolerance 1e-6, CV deviance absolute tolerance 0.01.

| Test | Tolerance | Status |
|---|---|---|
| wide lasso (20 covariates, alpha=1): coef path | 1e-4 | PASS |
| wide ridge (alpha=0): coef path (excl. extreme lambda[0]) | 1e-4 | PASS |
| wide elastic net (alpha=0.5): coef path | 1e-4 | PASS |
| wide lasso, standardize=False: coef path | 1e-4 | PASS |
| wide lasso, first 3 vars unpenalized: coef path | 1e-4 | PASS |
| lambda_max vs glmnet's own lambda[0] | 1e-6 | PASS |
| auto-generated lambda grid (first N pts) | 1e-6 | PASS |
| strata: penalized coef path | 5e-4 | PASS |
| offset: penalized coef path | 1e-4 | PASS |
| weights: penalized coef path | 1e-4 | PASS |
| left_truncation: penalized coef path | 5e-4 | PASS |
| combined (strata+offset+weights+start/stop): penalized coef path | 5e-4 | PASS |
| basic (heavy ties: 27 unique times / 500 rows): penalized coef path | 1e-4 | PASS |
| cross-validation: cvm vs cv.glmnet (explicit shared fold IDs) | 0.01 | PASS |
| cross-validation: lambda.min match | 1e-6 | PASS |
| cross-validation: lambda.1se match | 1e-6 | PASS |
| cross-validation: coef at lambda.min | 1e-4 | PASS |

Additionally, `test_penalized_self_consistency.py` verifies that the penalized path at `lambda → 0` converges to the unpenalized `CoxPH` solution for both Breslow and Efron ties, providing coverage for Efron-tie penalization that glmnet itself cannot validate (glmnet only implements Breslow).

## Phase 3: Variable selection (test_selector_r_comparison.py)

Validated against R's own `extractAIC()` driven stepwise search, step by step (criterion value at every step, not just the final variable set).

| Test | Tolerance | Status |
|---|---|---|
| forward, AIC: step trajectory | 1e-6 | PASS |
| backward, AIC: step trajectory | 1e-6 | PASS |
| both (bidirectional), AIC: step trajectory | 1e-6 | PASS |
| backward, BIC (k=log(nevent)): step trajectory | 1e-6 | PASS |
| forward, AIC, with x4 forced: step trajectory | 1e-6 | PASS |
| forward, AIC, strata + offset + weights: step trajectory | 1e-6 | PASS |

---

## Phase 4: Competing risks (test_phase4_r_comparison.py)

Cause-specific hazards validated against `coxph(Surv(..., event == k) ~ ...)`. Fine-Gray validated against `survival::finegray()` transform (row-for-row) plus the weighted clustered Cox fit.

| Test | Tolerance | Status |
|---|---|---|
| cr_simple: cause 1 coef | 1e-5 | PASS |
| cr_simple: cause 1 se | 1e-5 | PASS |
| cr_simple: cause 2 coef | 1e-5 | PASS |
| cr_simple: cause 2 se | 1e-5 | PASS |
| cr_simple: finegray fgstart (row-for-row vs R) | 1e-6 | PASS |
| cr_simple: finegray fgstop (row-for-row vs R) | 1e-6 | PASS |
| cr_simple: finegray fit coef | 1e-5 | PASS |
| cr_simple: finegray fit se | 1e-3 | PASS |
| cr_truncated: cause 1 coef | 1e-5 | PASS |
| cr_truncated: cause 1 se | 1e-5 | PASS |
| cr_truncated: cause 2 coef | 1e-5 | PASS |
| cr_truncated: cause 2 se | 1e-5 | PASS |
| cr_truncated: finegray fgstart (row-for-row vs R) | 1e-6 | PASS |
| cr_truncated: finegray fgstop (row-for-row vs R) | 1e-6 | PASS |
| cr_truncated: finegray fit coef | 1e-5 | PASS |
| cr_truncated: finegray fit se | 1e-3 | PASS |

## Phase 4: Robust/sandwich variance (test_phase4_r_comparison.py)

Validated against `coxph(..., cluster = cluster)` with strata + left truncation + clustering together — the one combination statsmodels.PHReg could not confirm during development.

| Test | Tolerance | Status |
|---|---|---|
| robust+strata+truncation: coef | 1e-5 | PASS |
| robust+strata+truncation: robust se | 1e-3 | PASS |
| robust+strata+truncation: naive se | 1e-5 | PASS |

Additionally, `test_robust_variance.py` and `test_score_residuals.py` validate against statsmodels.PHReg and lifelines on simpler configurations (right-censored, no strata), providing cross-package confirmation independent of R.

## Phase 4: Time-dependent covariates (test_phase4_r_comparison.py)

Validated against R's `tmerge()` (row-for-row merged dataset comparison) plus `coxph()` on the result.

| Test | Tolerance | Status |
|---|---|---|
| timedep: tstart (row-for-row vs R's tmerge) | 1e-6 | PASS |
| timedep: tstop (row-for-row vs R's tmerge) | 1e-6 | PASS |
| timedep: death indicator | 1e-9 | PASS |
| timedep: fit coef | 1e-5 | PASS |
| timedep: fit se | 1e-5 | PASS |

Additionally, `test_timedep.py` reproduces R's own bundled `tmerge.R` / `tmerge.Rout.save` test cases verbatim (transcribed from the `survival` package itself, not regenerated).

---

## What is and isn't covered here

Every check above compares this package's output directly against real R output on shared synthetic datasets.

**Phase 1-2 (via `r_reference/run_all.R`):** right-censored, left-truncated, stratified, offset, weighted, and all combined, under both Breslow and Efron ties. Both SMR and SHR two-stage patterns exercised end to end.

**Phase 3 (via `r_reference/run_penalized.R`, `run_selection.R`):** penalized coefficient paths across LASSO/ridge/elastic net, with every Phase 1-2 feature preserved; cross-validation deviance; AIC/BIC stepwise selection step-by-step.

**Phase 4 (via `r_reference/run_phase4_*.R`):** cause-specific and Fine-Gray competing risks (including the raw data transform row-for-row); cluster-robust sandwich variance with strata and truncation; time-dependent covariates via tmerge (row-for-row merged data plus fitted model).

It does **not** cover:

- Exact ties (not implemented — see R-compatibility notes)
- Real CMS claims data — the datasets here are synthetic but specifically constructed to include the features (ties from coarse time granularity, staggered entry, unbalanced provider-sized strata, non-trivial weights, correlated predictors for penalization, competing event structures, clustered observations, and production-realistic row/stratum/covariate counts) that make these models hard to get exactly right
