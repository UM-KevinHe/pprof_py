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
**Version 0.6.0** — The presentation layer (`pprof_py.presentation`): funnel plots
whose limits come from the same test as the flags, interval plots and ten more
displays, tables in HTML, Markdown, LaTeX, text and Excel, and self-contained
HTML reports, with a gallery and a page on reading each display. Statistical
results are unchanged from 0.5.0. This release requires Python 3.10 or newer and
changes how the plotting calls behave; see the [changelog](changelog.md) and the
[migration guide](presentation/migration.md). Import all models from the package
root: `from pprof_py import CoxPH, LogisticThreeStageModel, PenalizedLogistic, ...`
```

## Installation

```bash
git clone https://github.com/UM-KevinHe/pprof_py.git
cd pprof_py
pip install .
```

Requires `numpy`, `pandas`, `scipy`, and `numba`
(optional, for JIT-compiled survival kernels).

## Where to start

| I want to… | Go to |
|---|---|
| fit a model | [Model Architecture](model_architecture.md) for the three families and the shared `fit()` / `test()` conventions, then the start page of the family you need: [linear](linear/00_start_here.md), [logistic](logistic/00_start_here.md) or [survival](survival/00_start_here.md) |
| test providers against a reference | [Inference for Provider Effects](inference_provider_effects_theory.md) for the tests, the method list in the [measures, tests, intervals and plots reference](reference/measures_tests_plots.md), and [Empirical Null Calibration](empirical_null_theory.md) when many providers are tested at once |
| produce standardized measures | [Direct vs Indirect Standardization](direct_vs_indirect_standardization.md), [Inference for Standardized Measures](inference_standardized_measures_theory.md), and the SMR/SHR chapter of the survival guide ([Chapter 4](survival/04_indirect_standardization_smr_shr.md)) |
| make a funnel plot or a report | [Presentation](presentation/index.md): the gallery of displays, funnel limits, tables and HTML reports; the [plotting guide](reference/plotting_guide.md) for the model plotting methods |
| check how the package was validated | [R compatibility notes](survival/R_COMPATIBILITY.md) and the [validation report](survival/VALIDATION_REPORT.md); each penalized chapter ends with its own comparison against R |
| look up an API | [API Reference](api.md) for signatures and docstrings; the *Reference* section for argument and attribute lists by family |

## How the documentation is organized

- **Package Overview** — what provider profiling is, how the models are organized, and the changelog.
- **Core Concepts** — the theory shared by every family: standardization, fixed versus random effects, empirical null calibration, reliability, and inference for covariates, provider effects and standardized measures.
- **Statistical Models — Linear / Logistic** — one chapter per model class, each with the method, a running example and the R counterpart.
- **Survival Analysis** — a chaptered guide from survival data to penalized and discrete-time Cox models.
- **Presentation** — figures, tables and reports built from fitted results, followed by one page on reading each display.
- **Validation and R Compatibility** — what was checked against R, how, and what is known to differ.
- **Reference** — argument and attribute lists, data preparation, diagnostics and the plotting guide.
- **Tutorials** — end-to-end worked analyses for the linear and logistic profiling models.
- **API Reference** — the docstrings of every public class and function.

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
empirical_null_theory
inter_unit_reliability_theory
inference_covariate_effects_theory
inference_provider_effects_theory
inference_standardized_measures_theory
```

```{toctree}
:maxdepth: 2
:caption: Statistical Models — Linear

linear/00_start_here
linear/linear_fixed_effect_model
linear/linear_random_effect_model
linear/penalized_linear
linear/group_lasso_linear
```

```{toctree}
:maxdepth: 2
:caption: Statistical Models — Logistic

logistic/00_start_here
logistic/logistic_fixed_effect_model
logistic/logistic_random_effect_model
logistic/penalized_logistic
logistic/group_lasso_logistic
logistic/provider_penalized_logistic
logistic/logistic_three_stage_model
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
:caption: Presentation

Overview <presentation/index>
presentation/decision_guide
presentation/gallery
presentation/tables
presentation/theme_export_accessibility
presentation/migration
```

```{toctree}
:maxdepth: 2
:caption: Presentation — the displays

Reading the displays <presentation/displays/index>
presentation/displays/funnel
presentation/displays/caterpillar
presentation/displays/observed_expected
presentation/displays/forest
presentation/displays/data_quality
presentation/displays/null_calibration
presentation/displays/reliability
presentation/displays/provider_variation
presentation/displays/shrinkage
presentation/displays/flag_stability
presentation/displays/several_measures
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
reference/funnel_limits
reference/presentation
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
