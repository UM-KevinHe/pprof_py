# pprof_py Documentation

**pprof_py** is a Python package for provider profiling and healthcare
analytics. It provides validated, production-quality implementations of
risk-adjusted statistical models — logistic, linear, and survival —
together with standardized measure computation, hypothesis testing, and
visualization for evaluating healthcare provider performance.

Implemented in Python with NumPy and validated against R reference
implementations, `pprof_py` is designed for large-scale provider data
and reproducible statistical analyses.

```{note}
**Version 0.4.0** — All model families (logistic, linear, survival) now
include penalized, group lasso, provider-penalized, and discrete-survival
variants. Import all models from the package root:
`from pprof_py import CoxPH, PenalizedLogistic, DiscreteSurvival, ...`
```

## Installation

```bash
git clone https://github.com/UM-KevinHe/pprof_py.git
cd pprof_py
pip install .
```

Requires `numpy`, `pandas`, `scipy`, `scikit-learn`, and `numba`
(optional, for JIT-compiled survival kernels).

```{toctree}
:maxdepth: 2
:caption: Package Overview

introduction
Model Architecture <model_architecture>
changelog
```

```{toctree}
:maxdepth: 2
:caption: Core Concepts

direct_vs_indirect_standardization
fixed_vs_random_effects
```

```{toctree}
:maxdepth: 2
:caption: Statistical Models — Linear

linear/linear_fixed_effect_model
linear/linear_random_effect_model
linear/penalized_linear
linear/group_lasso_linear
```

```{toctree}
:maxdepth: 2
:caption: Statistical Models — Logistic

logistic/logistic_fixed_effect_model
logistic/logistic_random_effect_model
logistic/penalized_logistic
logistic/group_lasso_logistic
logistic/provider_penalized_logistic
logistic/logistic_mixed_effect_model
```

```{toctree}
:maxdepth: 2
:caption: Survival Analysis — Cox Proportional Hazards

survival/00_start_here
survival/01_survival_data_foundations
survival/02_the_cox_model
survival/03_fitting_your_first_model
survival/04_indirect_standardization_smr_shr
survival/05_robust_and_clustered_variance
survival/06_time_dependent_covariates
survival/07_competing_risks
survival/08_penalized_regression
survival/09_variable_selection
survival/10_complete_case_study
survival/11_group_lasso_cox
survival/12_provider_penalized_cox
survival/13_discrete_survival
survival/14_provider_discrete_survival
```

```{toctree}
:maxdepth: 2
:caption: Validation and R Compatibility

survival/R_COMPATIBILITY
survival/VALIDATION_REPORT
survival/ARCHITECTURE
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/linear_models
reference/logistic_models
reference/penalized_models
reference/coxph
reference/penalized
reference/competing_risks_and_selection
reference/inference_utilities
reference/measures_tests_plots
reference/data_and_iur
reference/empirical_null
reference/plotting_guide
reference/data_preparation
reference/deviance_statistics
reference/diagnostics_guide
reference/inter_unit_reliability
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

linear/linear_fe_tutorial
linear/linear_re_tutorial
logistic/logistic_fe_tutorial
logistic/logistic_re_tutorial
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api
```

## Getting Help

If you encounter any problems or bugs, contact us:

- xhliuu@umich.edu
- lfluo@umich.edu
- taoxu@umich.edu
- kevinhe@umich.edu

## References

```{bibliography} references.bib
:all:
:keyprefix: idx-
```
