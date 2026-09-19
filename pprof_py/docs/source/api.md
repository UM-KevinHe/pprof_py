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
.. autoclass:: pprof_py.models.survival.CoxPH
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.survival.PenalizedCoxPH
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.survival.PenalizedCoxPHCV
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.survival.CauseSpecificCoxPH
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.survival.FineGrayPH
   :members:
   :undoc-members:
   :show-inheritance:
```

### Group Lasso Cox Models

```{eval-rst}
.. autoclass:: pprof_py.models.survival.GroupLassoCoxPH
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.survival.GroupLassoCoxPHCV
   :members:
   :undoc-members:
   :show-inheritance:
```

### Provider-Penalized Cox Model

```{eval-rst}
.. autoclass:: pprof_py.models.survival.ProviderPenalizedCoxPH
   :members:
   :undoc-members:
   :show-inheritance:
```

### Discrete-Time Survival Models

```{eval-rst}
.. autoclass:: pprof_py.models.survival.DiscreteSurvival
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.survival.DiscreteSurvivalCV
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.survival.ProviderPenalizedDiscreteSurvival
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.survival.ProviderPenalizedDiscreteSurvivalCV
   :members:
   :undoc-members:
   :show-inheritance:
```

## Logistic Models

```{eval-rst}
.. autoclass:: pprof_py.models.logistic.LogisticFixedEffectModel
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.logistic.LogisticRandomEffectModel
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.logistic.LogisticMixedEffectModel
   :members:
   :undoc-members:
   :show-inheritance:
```

### Penalized Logistic Models

```{eval-rst}
.. autoclass:: pprof_py.models.logistic.PenalizedLogistic
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.logistic.PenalizedLogisticCV
   :members:
   :undoc-members:
   :show-inheritance:
```

### Group Lasso Logistic Models

```{eval-rst}
.. autoclass:: pprof_py.models.logistic.GroupLassoLogistic
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.logistic.GroupLassoLogisticCV
   :members:
   :undoc-members:
   :show-inheritance:
```

### Provider-Penalized Logistic Models

```{eval-rst}
.. autoclass:: pprof_py.models.logistic.ProviderPenalizedLogistic
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.logistic.ProviderPenalizedLogisticCV
   :members:
   :undoc-members:
   :show-inheritance:
```

## Linear Models

```{eval-rst}
.. autoclass:: pprof_py.models.linear.LinearFixedEffectModel
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.linear.LinearRandomEffectModel
   :members:
   :undoc-members:
   :show-inheritance:
```

### Penalized Linear Models

```{eval-rst}
.. autoclass:: pprof_py.models.linear.PenalizedLinear
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pprof_py.models.linear.PenalizedLinearCV
   :members:
   :undoc-members:
   :show-inheritance:
```

### Group Lasso Linear Model

```{eval-rst}
.. autoclass:: pprof_py.models.linear.GroupLassoLinear
   :members:
   :undoc-members:
   :show-inheritance:
```

## Variable Selection

```{eval-rst}
.. autoclass:: pprof_py.selection.CoxPHSelector
   :members:
   :undoc-members:
   :show-inheritance:
```

## Inference

```{eval-rst}
.. autofunction:: pprof_py.inference.huber_location_scale
```

```{eval-rst}
.. autofunction:: pprof_py.inference.estimate_empirical_null
```

## Plotting

```{eval-rst}
.. autofunction:: pprof_py.plotting.plot_caterpillar
```

## Utilities

```{eval-rst}
.. autofunction:: pprof_py.utils.setup_logger
```

```{eval-rst}
.. autofunction:: pprof_py.utils.proc_freq
```

```{eval-rst}
.. autofunction:: pprof_py.utils.sigmoid
```

## Exceptions

```{eval-rst}
.. autoclass:: pprof_py.exceptions.NotFittedError
   :members:
   :show-inheritance:
```
