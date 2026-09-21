(changelog)=
# Changelog

```{note}
This changelog is reconstructed from git history (commit messages,
dates, and the `v0.1-legacy` tag) and from `pprof_py.__version__`.
Treat specific dates as accurate (taken from commit timestamps) and
feature attributions as a best reconstruction. There is no released
`0.3.0`: `pyproject.toml` goes from `0.2.0` directly to `0.4.0`.
```

## 0.4.1 — current (July 2025)

- **Shared Gamma-frailty Cox model** (`FrailtyCoxPH`) and
  **time-varying-coefficient Cox model** (`TimeVaryingCoxPH`) migrated
  from `coxph_package` into `pprof_py` with full `__init__.py` exports,
  API reference, README, and changelog entries.
- Algorithm modules `algorithms/survival/frailty.py` and
  `algorithms/survival/time_varying.py` added.
- Documentation cleanup: removed all internal review artifacts
  (ISSUE/CODE_ISSUES/K-code references) from docs and Code_review
  files; fixed broken links in README, index, and reference pages.

## 0.4.0 (September 2026)

The largest commit in the repository's history, adding the shared
elastic-net/group-lasso coordinate-descent engine
(`algorithms/coordinate_descent.py`, `algorithms/penalty.py`) and
extending all three model families with penalized, group-lasso,
provider-penalized, and discrete-survival estimators:

- **Penalized regression** for logistic and linear outcomes
  (`PenalizedLogistic`, `PenalizedLogisticCV`, `PenalizedLinear`,
  `PenalizedLinearCV`) — see the
  [penalized logistic chapter](logistic/penalized_logistic).
- **Group lasso** across all three model families (`GroupLassoLogistic`,
  `GroupLassoLogisticCV`, `GroupLassoLinear`, `GroupLassoCoxPH`,
  `GroupLassoCoxPHCV`) — see the
  [group lasso chapter](logistic/group_lasso_logistic).
- **Provider-penalized models** (`ProviderPenalizedLogistic`,
  `ProviderPenalizedLogisticCV`, `ProviderPenalizedCoxPH`) — the
  package's own original methodological contribution; see the
  [provider-penalized chapter](logistic/provider_penalized_logistic).
- **Discrete-time survival models**, plain and provider-penalized
  (`DiscreteSurvival`, `DiscreteSurvivalCV`,
  `ProviderPenalizedDiscreteSurvival`,
  `ProviderPenalizedDiscreteSurvivalCV`) — see
  [Chapter 13](survival/13_discrete_survival).
- **`LogisticMixedEffectModel`** — Stage 3 of the He et al. (2013)
  three-stage SRR approach; see the
  [mixed-effect chapter](logistic/logistic_mixed_effect_model).

## 0.2.0 — the survival/Cox foundation

The "Replace the pprof_oy with refactored code" commit (2026-09-15,
464 files changed) added the entire survival/Cox module: `CoxPH`,
`PenalizedCoxPH`/`PenalizedCoxPHCV`, `CauseSpecificCoxPH`/`FineGrayPH`
(competing risks), `CoxPHSelector`, the indirect-standardization
(SMR/SHR) machinery (see
[Chapter 4](survival/04_indirect_standardization_smr_shr)),
robust/clustered variance, time-dependent-covariate data preparation
(`tmerge`, `build_skeleton`), and the
[diagnostics and R-validation infrastructure](diagnostics-guide).
This is the point where a linear/logistic-only package became a
provider-profiling package with a full survival-analysis arm.

## v0.1-legacy (tagged; last commit 2025-05-17)

The original release: fixed-effect and random-effect models for linear
and logistic outcomes only. Per git history: first commit and README
(2025-03-07), Sphinx scaffolding (2025-04-19),
`LogisticFixedEffectModel` (2025-05-06),
`LinearFixedEffectModel`/`LinearRandomEffectModel` plotting methods
(2025-05-07), `LogisticRandomEffectModel` (2025-05-12), and
documentation fixes through 2025-05-17. No survival/Cox module, no
penalized regression, no `measures.iur`. See the
[logistic FE](logistic/logistic_fixed_effect_model) and
[logistic RE](logistic_random_effect_model_stats) reference pages.

## Undated: `measures.iur`

The [inter-unit reliability](inter-unit-reliability-guide) module
(`BootstrapIUR`, `SplitHalfIUR`, `DirectIUR`, `ratio_measure`) exists
in the `0.4.0` codebase but is not re-exported at the package root
and is not attributable to a specific commit or version from the
available git history.
