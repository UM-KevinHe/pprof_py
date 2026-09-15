# pprof_py

**General statistical-computing package for provider profiling and healthcare analytics.**

`pprof_py` provides validated, production-quality implementations of risk-adjusted statistical models — logistic, linear, and Cox proportional hazards — together with standardized measure computation, hypothesis testing, and visualization for evaluating healthcare provider performance.

Implemented in Python with NumPy, validated against R reference implementations, and designed for large-scale provider data.

## Models

| Family | Classes | Key features |
| --- | --- | --- |
| **Logistic** | `LogisticFixedEffectModel`, `LogisticRandomEffectModel`, `LogisticMixedEffectModel` | SerBIN algorithm for large-*m* fixed effects; PIRLS+Laplace and Newton-Raphson+Gauss-Hermite for random/mixed effects; direct and indirect standardization; Wald, score, exact, and bootstrap hypothesis tests |
| **Linear** | `LinearFixedEffectModel`, `LinearRandomEffectModel` | Profile-based fixed effects; WLS and statsmodels-backed mixed effects; direct and indirect standardization |
| **Survival** | `CoxPH`, `PenalizedCoxPH`, `PenalizedCoxPHCV` | Breslow and Efron ties; strata, offset, weights, left truncation; validated against R's `survival::coxph()` to 1e⁻⁸–1e⁻¹⁴ relative error; numba-compiled kernels; two-stage SMR/SHR workflow |
| **Competing risks** | `CauseSpecificCoxPH`, `FineGrayPH` | Cause-specific hazards; Fine-Gray subdistribution hazards |
| **Variable selection** | `CoxPHSelector` | Stepwise forward/backward/bidirectional selection with AIC, BIC, or p-value criteria |

## Quick start

```python
from pprof_py import CoxPH, LogisticFixedEffectModel, LinearFixedEffectModel
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

Left truncation, strata, offset, and weights — any combination:

```python
model.fit(
    X, start=start, stop=stop, event=event,
    strata=provider,
    offset=log_exposure,
    sample_weight=weight,
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
model.predict_cumulative_hazard(X_new, stratum=...)      # per-subject H(t)
model.predict_survival_function(X_new, stratum=...)      # exp(-H(t))
```

### Logistic fixed effect model

```python
model = LogisticFixedEffectModel()
model.fit(y, X, group)
model.summary()
model.test()
model.calculate_standardized_measures()
```

### Penalized Cox regression

```python
from pprof_py import PenalizedCoxPHCV

model = PenalizedCoxPHCV(alpha=0.5, n_lambda=50)  # elastic net
model.fit(X, duration=time, event=event)
model.coef_           # coefficients at best lambda
model.lambda_best_    # cross-validated lambda
```

## Installation

```bash
git clone https://github.com/UM-KevinHe/pprof_py.git
cd pprof_py
pip install .
```

Requires Python ≥ 3.9.

**Core dependencies:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `numba`, `fast_poibin`.

**Optional:**

```bash
pip install ".[dev]"            # pytest
pip install ".[random-effect]"  # nlopt (optional solver for LogisticRandomEffectModel)
```

## Validation

The CoxPH implementation is validated against real R 4.3.3 / `survival` 3.5.8 output on shared synthetic datasets covering right-censored, left-truncated, stratified, offset, weighted data, and every combination — under both Breslow and Efron ties. Coefficients, standard errors, log-likelihood, baseline hazard, and martingale residuals all match to 1e⁻⁸–1e⁻¹⁴ relative error. Penalized regression (`PenalizedCoxPH` / `PenalizedCoxPHCV`) is validated against R's `glmnet(family="cox")`.

Production-scale checks (200,000 rows / 3,000 strata / 6 covariates, and 50,000 rows / 57 covariates) confirm agreement to 1e⁻¹⁴–1e⁻¹⁶ relative error — no discrepancy beyond floating-point noise.

See [`docs/source/validation/`](pprof_py/docs/source/validation/) for the full validation report, R compatibility notes, and architecture documentation.

## Performance

Cox fitting uses numba-compiled kernels (`@njit(cache=True)`) for the risk-set sweep, tie-method accumulation, baseline hazard increments, and martingale residuals. A pure-Python fallback is available when numba is not installed.

Indicative wall-clock times (1 vCPU, warm numba cache, full SHR-shaped combination):

| n | strata | covariates | tie method | fit time |
| --- | --- | --- | --- | --- |
| 2,000 | 10 | 2 | breslow | 0.14 s |
| 50,000 | 250 | 2 | breslow | 2.73 s |
| 200,000 | 3,000 | 6 | breslow | 1.8 s |
| 200,000 | 3,000 | 6 | efron | 1.9 s |
| 50,000 | 700 | 57 | efron | 1.8 s |

## Feature matrix — Survival models

| Feature | Status |
| --- | --- |
| Right-censored data | ✅ validated against R |
| Left truncation / `(start, stop]` | ✅ validated against R |
| Strata (own baseline hazard, shared coefficients) | ✅ validated against R |
| Offset (including the `basehaz` offset-mean subtlety) | ✅ validated against R |
| Case weights (model-based SE) | ✅ validated against R |
| Breslow ties (default) | ✅ validated against R |
| Efron ties | ✅ validated against R |
| Coefficients, SE, covariance, Wald z/p, CI, log-likelihood | ✅ |
| Baseline cumulative hazard / survival | ✅ |
| Martingale residuals (left-truncation-correct, Efron-correct) | ✅ |
| Two-stage SMR and SHR patterns | ✅ validated end to end |
| Prediction (linear, partial hazard, cumulative hazard, survival) | ✅ |
| Penalized regression (ridge / LASSO / elastic net + CV) | ✅ validated against `glmnet` |
| Competing risks (cause-specific, Fine-Gray) | ✅ |
| Automated variable selection (forward/backward/both) | ✅ |
| scikit-learn conventions (`BaseEstimator`, `coef_`-style attributes) | ✅ |
| Exact ties | ❌ not implemented |
| Robust/sandwich variance, clustering | ❌ not implemented |
| Formula interface | ❌ not implemented — pass a numeric design matrix |

## Package layout

```
pprof_py/
├── models/
│   ├── logistic/      fixed_effect, random_effect, mixed_effect
│   ├── linear/        fixed_effect, random_effect
│   └── survival/      coxph, penalized_coxph, competing_risks
├── algorithms/
│   ├── logistic/      serbin, ban
│   ├── linear/
│   └── survival/      cox_likelihood, risk_sets, ties, optimization, penalty, finegray, coordinate_descent
├── data/              validation, preparation, survival_data, timedep
├── inference/
│   ├── logistic/      fixed_effect, random_effect, mixed_effect
│   ├── linear/        fixed_effect, random_effect
│   └── survival/      inference, baseline, residuals, robust, deviance, empirical_null
├── measures/
│   ├── logistic/      standardized_measures, confidence_intervals, provider_tests
│   ├── linear/        fixed_effect, random_effect
│   └── iur/           bootstrap, split_half, direct
├── selection/         CoxPHSelector (AIC/BIC/p-value criteria)
├── diagnostics/       preflight checks, R-comparison harness
├── plotting/          caterpillar, funnel, coefficient forest, style system
└── utils/             numerical helpers, grouping, misc
```

See [`docs/source/validation/architecture.md`](pprof_py/docs/source/validation/architecture.md) for the full layering rationale.

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The R-comparison tests (`test_r_comparison.py`) require pre-generated R reference data:

```bash
python r_reference/generate_data.py
Rscript r_reference/run_all.R r_reference
pytest tests/ -v
```

Engine self-consistency tests and input validation tests do not require R.

## Documentation

Full Sphinx documentation lives under `pprof_py/docs/` and can be built with:

```bash
cd pprof_py/docs
pip install sphinx sphinx-rtd-theme myst-parser sphinxcontrib-bibtex matplotlib
make html
```

The documentation includes:

- **Getting started** — introduction and base model architecture
- **Core concepts** — direct vs. indirect standardization, fixed vs. random effects
- **Statistical models** — detailed methodology for linear, logistic, and survival models
- **Survival analysis tutorial** — 11 chapters from survival-data foundations through a complete facility-profiling case study
- **Validation** — R compatibility notes, numerical validation report, architecture design
- **API reference** — autodoc-generated from source docstrings

## License

MIT © 2025 Kevin He. See [LICENSE.md](LICENSE.md).

## Contact

If you encounter any problems or bugs, please contact:

- xhliuu@umich.edu
- lfluo@umich.edu
- taoxu@umich.edu
- kevinhe@umich.edu

## References

1. Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software*, 67(1), 1–48. https://doi.org/10.18637/jss.v067.i01

2. He, K., Kalbfleisch, J. D., Li, Y., & Li, Y. (2013). Evaluating hospital readmission rates in dialysis facilities; adjusting for hospital effects. *Lifetime Data Analysis*, 19, 490–512. https://doi.org/10.1007/s10985-013-9264-6

3. He, K. (2019). Indirect and direct standardization for evaluating transplant centers. *Journal of Hospital Administration*, 8(1), 9–14. https://doi.org/10.5430/jha.v8n1p9

4. Hsiao, C. (2022). *Analysis of Panel Data* (No. 64). Cambridge University Press.

5. Wu, W., Kuriakose, J. P., Weng, W., Burney, R. E., & He, K. (2023). Test-specific funnel plots for healthcare provider profiling leveraging individual- and summary-level information. *Health Services and Outcomes Research Methodology*, 23(1), 45–58. https://doi.org/10.1007/s10742-022-00287-3

6. Wu, W., Yang, Y., Kang, J., & He, K. (2022). Improving large-scale estimation and inference for profiling health care providers. *Statistics in Medicine*, 41(15), 2840–2853. https://doi.org/10.1002/sim.9387
