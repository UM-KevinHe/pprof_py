(introduction)=
# Introduction

Welcome to the documentation for **pprof_py**, a Python package designed
for robust provider profiling through advanced statistical modeling.

## What is Provider Profiling?

Provider profiling is a critical analytical process in healthcare and
other service industries. It involves the systematic assessment and
comparison of the performance of service providers (e.g., hospitals,
physicians, clinics) based on specific metrics. These metrics often
reflect quality of care, efficiency, patient outcomes, or adherence to
standards.

The primary goal of provider profiling is to identify variations in
performance, highlight best practices, and pinpoint areas requiring
improvement. Effective profiling relies on fair and accurate
comparisons, which necessitates adjusting for differences in the
underlying risk factors of the populations served by different
providers. Without such risk adjustment, comparisons can be misleading,
potentially penalizing providers who care for sicker or more complex
populations.

## Why pprof_py?

The **pprof_py** package offers a comprehensive suite of tools to
conduct sophisticated provider profiling. It aims to provide
accessible, efficient, and statistically sound methods for:

- **Risk Adjustment:** Implementing models that account for
  patient-level or case-mix differences, ensuring fairer comparisons.
- **Performance Measurement:** Calculating standardized measures that
  quantify provider performance relative to an expected baseline.
- **Statistical Inference:** Enabling hypothesis testing to determine
  if observed differences in performance are statistically significant.
- **Large-Scale Data Handling:** Designed with considerations for
  efficiency when working with substantial datasets common in healthcare
  analytics.

## Core Models in pprof_py

To address diverse analytical needs and data characteristics,
**pprof_py** implements a range of statistical models across three
families:

1. **Linear Models:**
   Suitable for continuous outcome variables. Includes fixed effects
   (`LinearFixedEffectModel`), random effects
   (`LinearRandomEffectModel`), penalized (`PenalizedLinear`,
   `PenalizedLinearCV`), and group lasso (`GroupLassoLinear`).

2. **Logistic Models:**
   Designed for binary outcome variables (e.g., mortality,
   readmission). Available as fixed effects
   (`LogisticFixedEffectModel`), random effects
   (`LogisticRandomEffectModel`), mixed effects
   (`LogisticMixedEffectModel`), penalized (`PenalizedLogistic`,
   `PenalizedLogisticCV`), group lasso (`GroupLassoLogistic`,
   `GroupLassoLogisticCV`), and provider-penalized
   (`ProviderPenalizedLogistic`, `ProviderPenalizedLogisticCV`).

3. **Survival / Cox Proportional Hazards Models:**
   For time-to-event data with censoring, the cornerstone of provider
   profiling for mortality and hospitalization outcomes. Includes:

   - `CoxPH` — standard Cox regression with Breslow/Efron ties,
     strata, offsets, weights, and left truncation, validated against
     R's `survival::coxph()` to 1e-8–1e-14 relative error.
   - `PenalizedCoxPH` / `PenalizedCoxPHCV` — Ridge, LASSO, and
     elastic net penalized Cox regression with cross-validation.
   - `GroupLassoCoxPH` / `GroupLassoCoxPHCV` — Group lasso for Cox
     models.
   - `ProviderPenalizedCoxPH` — Provider-penalized Cox regression.
   - `DiscreteSurvival` / `DiscreteSurvivalCV` — Discrete-time
     survival models with cross-validation.
   - `ProviderPenalizedDiscreteSurvival` /
     `ProviderPenalizedDiscreteSurvivalCV` — Provider-penalized
     discrete survival.
   - `CauseSpecificCoxPH` / `FineGrayPH` — Competing risks analysis.
   - `CoxPHSelector` — Automated stepwise variable selection (forward,
     backward, bidirectional) using AIC, BIC, or p-value criteria.

## Getting Started

This documentation covers installation, model fitting, result
interpretation, statistical methodology, and the full API. The Survival
Analysis section provides a comprehensive, chapter-by-chapter tutorial
from survival-data foundations through a complete case study.

All models are importable from the package root:

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
    CoxPH, PenalizedCoxPH, PenalizedCoxPHCV,
    GroupLassoCoxPH, GroupLassoCoxPHCV, ProviderPenalizedCoxPH,
    DiscreteSurvival, DiscreteSurvivalCV,
    ProviderPenalizedDiscreteSurvival, ProviderPenalizedDiscreteSurvivalCV,
    CauseSpecificCoxPH, FineGrayPH,
    # Selection
    CoxPHSelector,
)
```
