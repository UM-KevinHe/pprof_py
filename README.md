# pprof_py

**General statistical-computing package for provider profiling and healthcare analytics.**

`pprof_py` provides validated, production-quality implementations of risk-adjusted statistical models — logistic, linear, and Cox proportional hazards — together with standardized measure computation, hypothesis testing, and visualization for evaluating healthcare provider performance.

Implemented in Python with NumPy, validated against R reference implementations, and designed for large-scale provider data.

## Models

| Family                 | Classes                                                                                                                                              | Key features                                                                                                                                                                                                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic**           | `LogisticFixedEffectModel`, `LogisticRandomEffectModel`, `LogisticMixedEffectModel`                                                                  | SerBIN algorithm for large-_m_ fixed effects; PIRLS+Laplace and Newton-Raphson+Gauss-Hermite for random/mixed effects; direct and indirect standardization; provider tests (Wald, score, Poisson-binomial exact `poibin_exact`, bootstrap `bootstrap_exact`, resampling)                                                                      |
| **Penalized logistic** | `PenalizedLogistic`, `PenalizedLogisticCV`, `GroupLassoLogistic`, `GroupLassoLogisticCV`, `ProviderPenalizedLogistic`, `ProviderPenalizedLogisticCV` | Elastic net / ridge / LASSO with coordinate descent; group lasso for structured variable selection; two-stage provider + penalized covariate profiling; built-in cross-validation                                                                                                   |
| **Linear**             | `LinearFixedEffectModel`, `LinearRandomEffectModel`                                                                                                  | Profile-based fixed effects; pure-Python lme4-style REML/ML random intercepts (single and crossed); direct and indirect standardization                                                                                                                                             |
| **Penalized linear**   | `PenalizedLinear`, `PenalizedLinearCV`, `GroupLassoLinear`                                                                                           | Elastic net / ridge / LASSO for linear models; group lasso; cross-validation                                                                                                                                                                                                        |
| **Survival**           | `CoxPH`, `FrailtyCoxPH`, `TimeVaryingCoxPH`, `PenalizedCoxPH`, `PenalizedCoxPHCV`, `GroupLassoCoxPH`, `GroupLassoCoxPHCV`, `ProviderPenalizedCoxPH`  | Breslow and Efron ties; strata, offset, weights, left truncation; robust/sandwich and clustered variance; shared Gamma frailty (EM); time-varying coefficients; validated against R's `survival::coxph()` to 1e⁻⁸–1e⁻¹⁴ relative error; numba-compiled kernels; two-stage SMR/SHR workflow; group lasso and provider-penalized extensions |
| **Discrete survival**  | `DiscreteSurvival`, `DiscreteSurvivalCV`, `ProviderPenalizedDiscreteSurvival`, `ProviderPenalizedDiscreteSurvivalCV`                                 | Discrete-time survival with penalized covariates; three-layer provider + baseline hazard + covariate architecture; elastic net and group lasso penalties                                                                                                                            |
| **Competing risks**    | `CauseSpecificCoxPH`, `FineGrayPH`                                                                                                                   | Cause-specific hazards; Fine-Gray subdistribution hazards                                                                                                                                                                                                                           |
| **Variable selection** | `CoxPHSelector`                                                                                                                                      | Stepwise forward/backward/bidirectional selection with AIC, BIC, or p-value criteria                                                                                                                                                                                                |

## Quick start

```python
from pprof_py import (
    # Logistic
    LogisticFixedEffectModel, LogisticRandomEffectModel,
    LogisticMixedEffectModel,
    PenalizedLogistic, PenalizedLogisticCV,
    GroupLassoLogistic, GroupLassoLogisticCV,
    ProviderPenalizedLogistic, ProviderPenalizedLogisticCV,
    # Linear
    LinearFixedEffectModel, LinearRandomEffectModel,
    PenalizedLinear, PenalizedLinearCV, GroupLassoLinear,
    # Survival
    CoxPH, FrailtyCoxPH, TimeVaryingCoxPH,
    PenalizedCoxPH, PenalizedCoxPHCV,
    GroupLassoCoxPH, GroupLassoCoxPHCV, ProviderPenalizedCoxPH,
    DiscreteSurvival, DiscreteSurvivalCV,
    ProviderPenalizedDiscreteSurvival, ProviderPenalizedDiscreteSurvivalCV,
    CauseSpecificCoxPH, FineGrayPH,
    # Selection
    CoxPHSelector,
)
```

### Cox proportional hazards

```python
model = CoxPH(ties="breslow")
model.fit(X, duration=time, event=event)

model.coef_                 # coefficients
model.standard_errors_      # model-based SE
model.summary()             # coef / exp(coef) / se / z / p / CI
model.baseline_hazard_      # DataFrame: stratum, time, hazard, survival
model.martingale_residuals_
```

Left truncation, strata, offset, weights, and robust/clustered variance — any combination:

```python
model = CoxPH(robust=True)     # sandwich variance (each row is its own cluster)
model.fit(
    X, start=start, stop=stop, event=event,
    strata=provider,
    offset=log_exposure,
    sample_weight=weight,
    cluster=cluster_id,        # clustered sandwich variance (implies robust)
)
```

### Two-stage SMR/SHR (indirect standardization)

```python
import pandas as pd

stage1 = CoxPH().fit(
    X1, start=start, stop=stop, event=event,
    strata=provider, offset=offset1, sample_weight=weight,
)
xbeta = stage1.predict_linear(X1, offset=offset1)

stage2 = CoxPH().fit(
    pd.DataFrame(index=range(len(xbeta))),
    start=start, stop=stop, event=event,
    offset=xbeta, sample_weight=weight,
)
stage2.baseline_hazard_   # the "expected" side of an SMR/SHR comparison
```

### Prediction

```python
model.predict_linear(X_new, offset=offset_new)          # X @ coef_ + offset
model.predict_partial_hazard(X_new, offset=offset_new)   # exp(...)
model.predict_cumulative_hazard(X_new, stratum=...)      # per-subject H(t), at the stratum's event times
model.predict_survival_function(X_new, stratum=...)      # exp(-H(t))
```

### Logistic fixed effect model

```python
model = LogisticFixedEffectModel()
model.fit(df, y_var='event', x_vars=['x1', 'x2'], group_var='provider')   # or the array form: model.fit(X, y, groups)
model.summary()
model.test()
model.calculate_standardized_measures()
```

### Linear random effect model

```python
model = LinearRandomEffectModel(verbose=False)
model.fit(data, y_var='outcome', x_vars=['x1', 'x2'], group_var='provider', reml=True)

model.coefficients_['beta']       # fixed effects
model.coefficients_['alpha']      # BLUPs (random intercepts)
model.random_effect_sd_           # {group_var: sigma_u}
model.sigma_                      # residual SD
model.summary()
model.test(null=0)                # null must be numeric for the linear random-effect model
model.calculate_standardized_measures(stdz='indirect')
model.plot_funnel()
model.plot_provider_effects()
```

### Logistic random effect model

```python
model = LogisticRandomEffectModel(verbose=False)
model.fit(data, y_var='event', x_vars=['x1', 'x2'], group_var='provider')

model.coefficients_['beta']       # fixed effects (log-odds)
model.get_random_effects()        # BLUPs
model.test(null='median', test_method='wald')
model.calculate_standardized_measures(stdz='indirect')
model.plot_funnel()
model.plot_standardized_measures(stdz='indirect', measure='ratio')
```

### Penalized Cox regression

```python
from pprof_py import PenalizedCoxPHCV

model = PenalizedCoxPHCV(alpha=0.5, n_lambda=50)  # elastic net
model.fit(X, duration=time, event=event)
model.coef_           # coefficients at the selected lambda (lambda_min_ by default)
model.lambda_min_     # lambda with the minimum cross-validated deviance
model.lambda_1se_     # largest lambda within one standard error of the minimum
```

## Installation

```bash
git clone https://github.com/UM-KevinHe/pprof_py.git
cd pprof_py
pip install .
```

Requires Python ≥ 3.9.

**Core dependencies:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `numba`, `fast_poibin`.

**Optional:**

```bash
pip install ".[dev]"            # pytest + statsmodels (cross-validation tests)
pip install ".[random-effect]"  # nlopt (optional solver for LogisticRandomEffectModel)
```

## Validation

The CoxPH implementation is validated against real R 4.3.3 / `survival` 3.5.8 output on shared synthetic datasets covering right-censored, left-truncated, stratified, offset, weighted data, and every combination — under both Breslow and Efron ties. Coefficients, standard errors, log-likelihood, baseline hazard, and martingale residuals all match to 1e⁻⁸–1e⁻¹⁴ relative error. Robust/sandwich and clustered variance estimates are validated against R's `survival::coxph(robust=TRUE, cluster=...)` (with the same tie method on both sides — R defaults to Efron, this package to Breslow). Penalized regression (`PenalizedCoxPH` / `PenalizedCoxPHCV`) is validated against R's `glmnet(family="cox")`.

The `LinearRandomEffectModel` is validated against R's `lme4::lmer` (REML and ML, weighted and unweighted) with beta, sigma, RE SD, log-likelihood, and BLUP errors at 10⁻⁷–10⁻⁹ (re-checked on 2026-09-20 against lme4 1.1.35.1: largest difference 2e-7; the comparison script is not in the repository). `LogisticRandomEffectModel` agrees with `glmer(nAGQ=1)` to about 2e-5, and the penalized linear and logistic estimators reproduce glmnet's λ sequence to 5e-14 (details in [`reference/`](pprof_py/docs/source/reference/)).

Production-scale checks (200,000 rows / 3,000 strata / 6 covariates, and 50,000 rows / 57 covariates) confirm agreement to 1e⁻¹⁴–1e⁻¹⁶ relative error — no discrepancy beyond floating-point noise.

A fresh re-run on 2026-09-20 (R 4.3.3, `survival` 3.5.8, `glmnet` 4.1.8) reproduced all 55 Phase 1–2 checks and the penalized-regression checks; Fine–Gray regression on **left-truncated** data does not match R (see *Known limitations*). Group lasso, provider-penalized and discrete-time models have no R reference.

See [`docs/source/survival/`](pprof_py/docs/source/survival/) for the full validation report, R compatibility notes, and architecture documentation. Per-estimator reference pages are in [`docs/source/reference/`](pprof_py/docs/source/reference/).

## Performance

Cox fitting uses numba-compiled kernels (`@njit(cache=True)`) for the risk-set sweep, tie-method accumulation, baseline hazard increments, and martingale residuals. numba is installed with the package; if it cannot be imported, a pure-Python fallback (much slower on large data) is used automatically.

Indicative wall-clock times (1 vCPU, warm numba cache, full SHR-shaped combination):

| n       | strata | covariates | tie method | fit time |
| ------- | ------ | ---------- | ---------- | -------- |
| 2,000   | 10     | 2          | breslow    | 0.14 s   |
| 50,000  | 250    | 2          | breslow    | 2.73 s   |
| 200,000 | 3,000  | 6          | breslow    | 1.8 s    |
| 200,000 | 3,000  | 6          | efron      | 1.9 s    |
| 50,000  | 700    | 57         | efron      | 1.8 s    |

## Feature matrix — Survival models

| Feature                                                              | Status                                            |
| -------------------------------------------------------------------- | ------------------------------------------------- |
| Right-censored data                                                  | ✅ validated against R                            |
| Left truncation / `(start, stop]`                                    | ✅ validated against R                            |
| Strata (own baseline hazard, shared coefficients)                    | ✅ validated against R                            |
| Offset (including the `basehaz` offset-mean subtlety)                | ✅ validated against R                            |
| Case weights (model-based SE)                                        | ✅ validated against R                            |
| Breslow ties (default)                                               | ✅ validated against R                            |
| Efron ties                                                           | ✅ validated against R                            |
| Coefficients, SE, covariance, Wald z/p, CI, log-likelihood           | ✅                                                |
| Baseline cumulative hazard / survival                                | ✅                                                |
| Martingale residuals (left-truncation-correct, Efron-correct)        | ✅                                                |
| Two-stage SMR and SHR patterns                                       | ✅ validated end to end                           |
| Prediction (linear, partial hazard, cumulative hazard, survival)     | ✅                                                |
| Penalized regression (ridge / LASSO / elastic net + CV)              | ✅ validated against `glmnet`                     |
| Cause-specific hazards                                               | ✅ validated against R (same ties on both sides)  |
| Fine-Gray subdistribution hazards                                    | ✅ right-censored · ⚠️ left-truncated: weights differ from R |
| Group lasso / provider-penalized Cox                                 | ✅ implemented; no R reference (internal tests)   |
| Discrete-time survival (penalized, provider)                         | ✅ implemented; no R reference (internal tests)   |
| Shared Gamma frailty (EM algorithm)                                  | ✅ implemented (`FrailtyCoxPH`)                   |
| Time-varying coefficients                                            | ✅ implemented (`TimeVaryingCoxPH`)               |
| Automated variable selection (forward/backward/both)                 | ✅                                                |
| scikit-learn conventions (`BaseEstimator`, `coef_`-style attributes) | ✅                                                |
| Exact ties                                                           | ❌ not implemented                                |
| Robust/sandwich variance, clustering                                 | ✅ validated against R                            |
| Formula interface                                                    | ❌ not implemented — pass a numeric design matrix |

## Known limitations (survival)

- `FineGrayPH` with left-truncated data does not reproduce R's `finegray()` weights (coefficients differ by ~3e-3 on the reference dataset).
- `CoxPH(fit_intercept=True)` fits, but every `predict_*` method then raises `ValueError`.
- `CoxPH` does not warn when `max_iter` is reached; check `converged_`.
- `ties="exact"` is not implemented; `GroupLassoCoxPH(method="MM")` is not implemented.
- The R-comparison tests for variable selection need datasets that no script in the repository generates.

## Known limitations (linear and logistic)

- `LinearRandomEffectModel.test` needs a numeric `null`; `test()` and `calculate_standardized_measures()` support one grouping factor only.
- `LogisticRandomEffectModel` uses `nlopt` (optional) for its default optimizer and silently falls back to another one without it — set `optimizer_stage1="powell"` explicitly if `nlopt` is not installed.
- `LogisticFixedEffectModel.fit(n_var=...)` accepts a trials-count column but the response is still validated as 0/1 — use one row per patient.

## Package layout

```
pprof_py/
├── models/
│   ├── logistic/      fixed_effect, random_effect, mixed_effect, penalized, group_lasso, provider_penalized
│   ├── linear/        fixed_effect, random_effect, penalized, group_lasso
│   └── survival/      coxph, frailty_coxph, time_varying_coxph, penalized_coxph, group_lasso_coxph, provider_coxph, discrete_survival, provider_discrete_survival, competing_risks
├── algorithms/
│   ├── logistic/      fixed_effect, likelihood, provider_effects
│   ├── linear/        fixed_effect, likelihood
│   └── survival/      cox_likelihood, frailty, time_varying, risk_sets, ties, optimization, partial_likelihood, penalty, coordinate_descent, provider_effects, discrete_survival, finegray
├── data/              validation, preparation, survival_data, survival_validation, timedep
├── inference/
│   ├── logistic/      fixed_effect, random_effect, mixed_effect
│   ├── linear/        fixed_effect, random_effect
│   └── survival/      inference, baseline, residuals, robust, empirical_null
├── measures/
│   ├── logistic/      standardized_measures, confidence_intervals, provider_tests
│   ├── linear/        fixed_effect, random_effect
│   └── iur/           bootstrap, split_half, direct
├── statistics/        deviance (saturated log-likelihood, Cox deviance, deviance ratio)
├── selection/         CoxPHSelector (AIC/BIC/p-value criteria), criteria
├── diagnostics/       preflight checks, R-comparison harness
├── plotting/          caterpillar, funnel, coefficient forest, style system
└── utils/             numerical helpers, grouping, misc
```

See [`docs/source/survival/ARCHITECTURE.md`](pprof_py/docs/source/survival/ARCHITECTURE.md) for the full layering rationale.

## Running the tests

```bash
pip install -e ".[dev]"
pytest pprof_py/tests/survival -q        # engine self-consistency + input validation need no R
```

The R-comparison tests read reference data that must be generated first (needs R with `survival` and `glmnet`):

```bash
cd pprof_py/r_reference
python generate_data.py && python generate_penalized_data.py && python generate_phase4_data.py
Rscript run_all.R . && Rscript run_penalized.R . \
  && Rscript run_phase4_competing_risks.R . && Rscript run_phase4_robust.R . && Rscript run_phase4_timedep.R .
cd ../..
ln -s ../r_reference pprof_py/tests/r_reference   # the tests currently look in two places
pytest pprof_py/tests/survival -q
```

A fresh run currently ends with 200 passed, 26 failed, 1 skipped; the failures are explained in
[`R_COMPATIBILITY.md`](pprof_py/docs/source/survival/R_COMPATIBILITY.md).

## Documentation

Full Sphinx documentation lives under `pprof_py/docs/` and can be built with:

```bash
pip install ".[docs]"
cd pprof_py/docs
make html
```

The documentation includes:

- **Getting started** — introduction, model architecture, and changelog
- **Core concepts** — direct vs. indirect standardization, fixed vs. random effects
- **Statistical models** — detailed methodology for linear, logistic, and survival models
- **Survival analysis** — 15 chapters (00–14) from survival-data foundations through group lasso, provider-penalized Cox, and discrete-time survival
- **Tutorials** — hands-on walkthroughs for linear and logistic fixed-effect and random-effect models
- **Reference** — quick-reference cheat sheets for all model families (linear, logistic, penalized, survival), plus guides for empirical null calibration, plotting, data preparation, deviance statistics, diagnostics, and inter-unit reliability
- **Validation** — R compatibility notes, numerical validation report, architecture design
- **API reference** — autodoc-generated from source docstrings

## License

MIT © 2025 Kevin He. See [LICENSE.md](LICENSE.md).

## Contact

If you encounter any problems or bugs, please contact:

- taoxu@umich.edu
- kevinhe@umich.edu

## References

1. Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. _Journal of Statistical Software_, 67(1), 1–48. https://doi.org/10.18637/jss.v067.i01

2. He, K., Kalbfleisch, J. D., Li, Y., & Li, Y. (2013). Evaluating hospital readmission rates in dialysis facilities; adjusting for hospital effects. _Lifetime Data Analysis_, 19, 490–512. https://doi.org/10.1007/s10985-013-9264-6

3. He, K. (2019). Indirect and direct standardization for evaluating transplant centers. _Journal of Hospital Administration_, 8(1), 9–14. https://doi.org/10.5430/jha.v8n1p9

4. Hsiao, C. (2022). _Analysis of Panel Data_ (No. 64). Cambridge University Press.

5. Wu, W., Kuriakose, J. P., Weng, W., Burney, R. E., & He, K. (2023). Test-specific funnel plots for healthcare provider profiling leveraging individual- and summary-level information. _Health Services and Outcomes Research Methodology_, 23(1), 45–58. https://doi.org/10.1007/s10742-022-00287-3

6. Wu, W., Yang, Y., Kang, J., & He, K. (2022). Improving large-scale estimation and inference for profiling health care providers. _Statistics in Medicine_, 41(15), 2840–2853. https://doi.org/10.1002/sim.9387
