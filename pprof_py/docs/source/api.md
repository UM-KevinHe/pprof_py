(api)=
# API Reference

This section provides the API reference for the `pprof_py` package.
All primary model classes are importable directly from `pprof_py`.

```{contents}
:local:
:depth: 2
```

## Survival Models

```{eval-rst}
.. autoclass:: pprof_py.CoxPH
   :members:
   :show-inheritance:
```

### Frailty and Time-Varying Cox Models

```{eval-rst}
.. autoclass:: pprof_py.FrailtyCoxPH
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.TimeVaryingCoxPH
   :members:
   :show-inheritance:
```

### Penalized Cox Models

```{eval-rst}
.. autoclass:: pprof_py.PenalizedCoxPH
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.PenalizedCoxPHCV
   :members:
   :show-inheritance:
```

### Competing Risks

```{eval-rst}
.. autoclass:: pprof_py.CauseSpecificCoxPH
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.FineGrayPH
   :members:
   :show-inheritance:
```

### Group Lasso Cox Models

```{eval-rst}
.. autoclass:: pprof_py.GroupLassoCoxPH
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.GroupLassoCoxPHCV
   :members:
   :show-inheritance:
```

### Provider-Penalized Cox Model

```{eval-rst}
.. autoclass:: pprof_py.ProviderPenalizedCoxPH
   :members:
   :show-inheritance:
```

### Discrete-Time Survival Models

```{eval-rst}
.. autoclass:: pprof_py.DiscreteSurvival
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.DiscreteSurvivalCV
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.ProviderPenalizedDiscreteSurvival
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.ProviderPenalizedDiscreteSurvivalCV
   :members:
   :show-inheritance:
```

### Survival Data Preparation and Diagnostics

```{eval-rst}
.. autoclass:: pprof_py.data.timedep.UpdateStream

.. autofunction:: pprof_py.data.timedep.build_skeleton

.. autofunction:: pprof_py.data.timedep.tmerge

.. autofunction:: pprof_py.data.timedep.survsplit

.. autofunction:: pprof_py.data.survival_validation.validate_fit_inputs

.. autofunction:: pprof_py.diagnostics.survival.preflight_report

.. autoclass:: pprof_py.diagnostics.survival.PreflightResult
   :members: report

.. autofunction:: pprof_py.diagnostics.survival.validate_against_r.run_validation
```

### Survival Inference Utilities

```{eval-rst}
.. autofunction:: pprof_py.inference.survival.compute_baseline_hazard

.. autofunction:: pprof_py.inference.survival.martingale_residuals

.. autofunction:: pprof_py.inference.survival.residuals.score_residuals

.. autofunction:: pprof_py.inference.survival.residuals.dfbeta_residuals

.. autofunction:: pprof_py.inference.survival.robust.cluster_score_residuals

.. autofunction:: pprof_py.inference.survival.robust.robust_covariance

.. autofunction:: pprof_py.inference.survival.poisson_exact_test

.. autofunction:: pprof_py.inference.survival.poisson_midp_zscore

.. autofunction:: pprof_py.inference.survival.log_ratio_zscore

.. autofunction:: pprof_py.inference.survival.fit_robust_location_scale

.. autofunction:: pprof_py.inference.survival.fit_empirical_null

.. autofunction:: pprof_py.inference.survival.fit_grouped_empirical_null

.. autofunction:: pprof_py.inference.survival.assign_quantile_groups

.. autofunction:: pprof_py.inference.survival.adjust_empirical_null

.. autofunction:: pprof_py.inference.survival.poisson_confidence_bounds

.. autofunction:: pprof_py.inference.survival.log_ratio_confidence_intervals

.. autofunction:: pprof_py.statistics.deviance.saturated_log_likelihood

.. autofunction:: pprof_py.statistics.deviance.cox_deviance

.. autofunction:: pprof_py.statistics.deviance.deviance_ratio

.. autofunction:: pprof_py.statistics.deviance.bootstrap_cv_se

.. autofunction:: pprof_py.selection.aic

.. autofunction:: pprof_py.selection.bic

.. autofunction:: pprof_py.algorithms.survival.finegray.finegray_transform

.. autoclass:: pprof_py.algorithms.survival.finegray.FineGrayData
```

## Logistic Models

```{eval-rst}
.. autoclass:: pprof_py.LogisticFixedEffectModel
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.LogisticRandomEffectModel
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.LogisticMixedEffectModel
   :members:
   :show-inheritance:
```

### Penalized Logistic Models

```{eval-rst}
.. autoclass:: pprof_py.PenalizedLogistic
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.PenalizedLogisticCV
   :members:
   :show-inheritance:
```

### Group Lasso Logistic Models

```{eval-rst}
.. autoclass:: pprof_py.GroupLassoLogistic
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.GroupLassoLogisticCV
   :members:
   :show-inheritance:
```

### Provider-Penalized Logistic Models

```{eval-rst}
.. autoclass:: pprof_py.ProviderPenalizedLogistic
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.ProviderPenalizedLogisticCV
   :members:
   :show-inheritance:
```

## Linear Models

```{eval-rst}
.. autoclass:: pprof_py.LinearFixedEffectModel
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.LinearRandomEffectModel
   :members:
   :show-inheritance:
```

### Penalized Linear Models

```{eval-rst}
.. autoclass:: pprof_py.PenalizedLinear
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.PenalizedLinearCV
   :members:
   :show-inheritance:
```

### Group Lasso Linear Model

```{eval-rst}
.. autoclass:: pprof_py.GroupLassoLinear
   :members:
   :show-inheritance:
```

## Variable Selection

```{eval-rst}
.. autoclass:: pprof_py.CoxPHSelector
   :members:
   :show-inheritance:
```

## Inter-Unit Reliability (IUR)

```{eval-rst}
.. autoclass:: pprof_py.measures.iur.bootstrap.BootstrapIUR
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.measures.iur.split_half.SplitHalfIUR
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.measures.iur.direct.DirectIUR
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.measures.iur._core.IURDecomposition
   :members:
```

```{eval-rst}
.. autofunction:: pprof_py.measures.iur.measures.ratio_measure
```

## Inference

```{eval-rst}
.. autofunction:: pprof_py.huber_location_scale
```

```{eval-rst}
.. autofunction:: pprof_py.estimate_empirical_null
```

## Plotting

```{eval-rst}
.. autofunction:: pprof_py.plot_caterpillar
```

## Utilities

```{eval-rst}
.. autofunction:: pprof_py.setup_logger
```

```{eval-rst}
.. autofunction:: pprof_py.proc_freq
```

```{eval-rst}
.. autofunction:: pprof_py.sigmoid
```

## Exceptions

```{eval-rst}
.. autoclass:: pprof_py.NotFittedError
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.data.survival_validation.SurvivalDataError
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.survival.penalized_coxph.DegenerateFeatureWarning
   :show-inheritance:
```
