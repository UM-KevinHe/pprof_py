(model_architecture)=
# Model Architecture

`pprof_py` organizes its models into three families — **logistic**,
**linear**, and **survival** — each with its own estimator classes.
Rather than forcing all models through a single abstract base class, the
package uses a combination of **a small common base class** and
**structural-typing protocols** to provide a consistent API while
respecting each model family's distinct statistical requirements.

```{contents} Table of Contents
:local:
:depth: 2
```

## Design Principles

**No shared abstract base class.** An earlier version of this package
defined an abstract `BaseModel` class. It was removed because the
statistical models in `pprof_py` differ fundamentally in their inputs,
fitting procedures, and outputs — a logistic fixed-effect model, a
linear random-effect model, and a Cox proportional hazards model share
almost no fitting logic. Forcing them into a common inheritance tree
added complexity without adding value.

**One base class.** Every model class inherits `pprof_py.base.ProviderModel`, which fixes these conventions:

- Constructor arguments define model *configuration* (e.g., `ties`,
  `confidence_level`).
- `fit()` performs all data-dependent work.
- Fitted attributes end with a trailing underscore (e.g., `coef_`,
  `standard_errors_`, `log_likelihood_`).
- `get_params()` / `set_params()` return and modify constructor
  parameters, not fitted results, and the `repr` lists the parameters
  that differ from their defaults.

pprof_py does not depend on scikit-learn.

**Structural-typing protocols.** Common *interface contracts* — methods
like `summary()`, `test()`, and plotting methods — are defined as
`typing.Protocol` classes in `pprof_py.models.mixins`. A model
satisfies a protocol simply by implementing the required methods; no
explicit inheritance is needed (though it is allowed).

## Model Families

| Family | Module | Classes |
|--------|--------|---------|
| Logistic | `pprof_py.models.logistic` | `LogisticFixedEffectModel`, `LogisticRandomEffectModel`, `LogisticFERandomClusterModel`, `PenalizedLogistic`, `PenalizedLogisticCV`, `GroupLassoLogistic`, `GroupLassoLogisticCV`, `ProviderPenalizedLogistic`, `ProviderPenalizedLogisticCV` |
| Linear | `pprof_py.models.linear` | `LinearFixedEffectModel`, `LinearRandomEffectModel`, `PenalizedLinear`, `PenalizedLinearCV`, `GroupLassoLinear` |
| Survival | `pprof_py.models.survival` | `CoxPH`, `PenalizedCoxPH`, `PenalizedCoxPHCV`, `GroupLassoCoxPH`, `GroupLassoCoxPHCV`, `ProviderPenalizedCoxPH`, `DiscreteSurvival`, `DiscreteSurvivalCV`, `ProviderPenalizedDiscreteSurvival`, `ProviderPenalizedDiscreteSurvivalCV`, `CauseSpecificCoxPH`, `FineGrayPH` |
| Selection | `pprof_py.selection` | `CoxPHSelector` |

All models are importable from the package root:

```python
from pprof_py import (
    # Logistic
    LogisticFixedEffectModel, LogisticRandomEffectModel,
    LogisticFERandomClusterModel,
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

(protocol_contracts)=
## Protocol Contracts

The following protocols are defined in `pprof_py.models.mixins`. They
are `@runtime_checkable`, so you can use `isinstance()` checks at
runtime if needed.

### SummaryMixin

Models that provide a `summary()` method satisfy this protocol.

```python
class SummaryMixin(Protocol):
    def summary(
        self,
        level: float = 0.95,
        null: float = 0,
        alternative: str = "two_sided",
    ) -> Any: ...
```

The summary typically includes estimated coefficients, standard errors,
confidence intervals, and p-values:

$$
\hat{\beta} \pm z_{\alpha/2} \cdot SE(\hat{\beta})
$$

### TestMixin

Models that provide a `test()` method for hypothesis testing.

```python
class TestMixin(Protocol):
    def test(self, *args, **kwargs) -> Any: ...
```

Common tests include the Wald Z-test:

$$
z = \frac{\hat{\beta} - \beta_0}{SE(\hat{\beta})}
$$

The specific tests available depend on the model family (e.g.,
likelihood-ratio tests, score tests, exact tests for logistic models;
Wald tests for survival models).

### PlotMixin

Models that provide standard diagnostic and summary plots.

```python
class PlotMixin(Protocol):
    def plot_funnel(self, *args, **kwargs) -> None: ...
    def plot_residuals(self, *args, **kwargs) -> None: ...
    def plot_qq(self, *args, **kwargs) -> None: ...
    def plot_provider_effects(self, *args, **kwargs) -> None: ...
    def plot_standardized_measures(self, *args, **kwargs) -> None: ...
    def plot_coefficient_forest(self, *args, **kwargs) -> None: ...
```

All plotting uses Matplotlib as the primary backend. The package-wide
visual style is defined in `pprof_py.plotting.style`.

Each model family exposes its plotting methods through a dedicated
`PlottingMixin` (e.g., `plotting.linear.LinearRandomEffectPlottingMixin`,
`plotting.logistic.LogisticRandomEffectPlottingMixin`,
`plotting.logistic.LogisticFixedEffectPlottingMixin`). Both linear models (fixed and random effect) provide `plot_residuals` and `plot_qq`; on `LogisticFixedEffectModel` they raise `NotImplementedError`, and `LogisticRandomEffectModel` and `LogisticFERandomClusterModel` have neither.

## Common Fitted Attributes

Each model family exposes fitted attributes using a trailing underscore.
The naming conventions differ slightly between families.

**Survival models** (`CoxPH`, `PenalizedCoxPH`, etc.):

| Attribute | Description |
|-----------|-------------|
| `coef_` | Estimated model coefficients (array) |
| `standard_errors_` | Standard errors of the coefficients |
| `covariance_` | Variance-covariance matrix of the coefficients |
| `log_likelihood_` | Maximized log-likelihood |
| `n_obs_` | Number of observations used in fitting |
| `n_events_` | Number of events (uncensored observations) |
| `baseline_hazard_` | DataFrame of baseline cumulative hazard / survival per stratum |
| `martingale_residuals_` | Martingale residuals (array) |
| `convergence_message_` | Diagnostic message from the optimizer (if applicable) |

**Linear and logistic models** (`LinearRandomEffectModel`,
`LogisticRandomEffectModel`, etc.):

| Attribute | Description |
|-----------|-------------|
| `coefficients_` | Dict with `'beta'` (fixed effects, `pd.Series`) and `'alpha'` (random effects / BLUPs, `pd.Series` or dict) |
| `variances_` | Dict with `'beta'` (VCV `pd.DataFrame`) and `'alpha'` (random-effect variance) |
| `sigma_` | Residual standard deviation (linear) or random-effect SD dict (logistic) |
| `random_effect_sd_` | Dict mapping each group variable to its estimated RE standard deviation (linear only) |
| `loglike_` | Log-likelihood at convergence |
| `aic_` / `bic_` | Information criteria |
| `fitted_` | Fitted values (linear scale or probabilities) |
| `residuals_` | Residuals (response scale) |
| `converged_` | Boolean convergence flag |

This table describes the random-effect classes. The fixed-effect classes have `coefficients_['gamma']` (provider effects) instead of `'alpha'`, `variances_['gamma']`, no `loglike_` or `converged_`, and `LogisticFixedEffectModel` adds `auc_`. Exact shapes and the different `summary()` layouts are listed in the reference ({ref}`ll_ref_conventions`).

## Typical Workflow

**Survival model:**

```python
from pprof_py import CoxPH

# 1. Configure
model = CoxPH(ties="breslow", confidence_level=0.95)

# 2. Fit
model.fit(
    X, duration=time, event=event,
    strata=provider, offset=log_exposure,
)

# 3. Inspect
model.coef_
model.standard_errors_
model.summary()

# 4. Predict
model.predict_linear(X_new)
model.predict_survival_function(X_new)
```

**Logistic fixed-effect model:**

```python
from pprof_py import LogisticFixedEffectModel

model = LogisticFixedEffectModel()
model.fit(df, y_var='event', x_vars=['x1', 'x2'], provider_var='provider')
model.summary()
model.test()
model.calculate_standardized_measures()
```

**Linear random-effect model:**

```python
from pprof_py import LinearRandomEffectModel

model = LinearRandomEffectModel(verbose=False)
model.fit(data, y_var='outcome', x_vars=['x1', 'x2'], provider_var='provider')

model.coefficients_['beta']       # fixed effects
model.coefficients_['alpha']      # BLUPs
model.random_effect_sd_           # {provider_var: sigma_u}
model.summary()
model.test()                     # reference 0 (the random-effect mean) by default
model.calculate_standardized_measures(stdz='indirect')
model.plot_funnel()
model.plot_provider_effects()
```

**Logistic random-effect model:**

```python
from pprof_py import LogisticRandomEffectModel

model = LogisticRandomEffectModel(verbose=False)
model.fit(data, y_var='event', x_vars=['x1', 'x2'], provider_var='provider')

model.coefficients_['beta']       # fixed effects (log-odds)
model.get_random_effects()        # BLUPs
model.test(test_method='wald')   # reference 0 by default; reference='median' also works
model.calculate_standardized_measures(stdz='indirect')
model.plot_funnel()
model.plot_provider_effects()
model.plot_standardized_measures(stdz='indirect', measure='ratio')
model.plot_coefficient_forest()
```
