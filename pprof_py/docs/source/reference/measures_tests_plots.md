(ll_ref_measures)=
# Reference: standardized measures, provider tests, confidence intervals and plots

The fixed-effect, random-effect and mixed-effect classes share a provider-profiling workflow: estimate provider effects → standardize them →
test them against a null → build intervals → plot. The theory (direct versus indirect standardization) is in
[Direct vs indirect standardization](direct_vs_indirect_standardization); this page lists the methods and what they return.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
m = 20                                            # providers
provider = np.repeat(np.arange(m), rng.integers(60, 120, m))   # rows sorted by provider
n = len(provider)
x = rng.normal(size=(n, 3))
gamma = rng.normal(0, 0.4, m)
df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "provider": provider,
                   "cluster": rng.integers(0, 8, n)})
df["y"] = 1 + x @ [0.5, -0.3, 0.2] + gamma[provider] + rng.normal(size=n)          # continuous outcome
eta = -1 + x @ [0.5, -0.4, 0.2] + gamma[provider]
df["event"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)                 # binary outcome
X_COLS = ["x1", "x2", "x3"]
```

```python
import matplotlib
matplotlib.use("Agg")
from pprof_py import LinearFixedEffectModel, LogisticFixedEffectModel

lin = LinearFixedEffectModel().fit(df, y_var="y", x_vars=X_COLS, group_var="provider")
log = LogisticFixedEffectModel().fit(df, y_var="event", x_vars=X_COLS, group_var="provider")

lin.calculate_standardized_measures(stdz=["indirect", "direct"])["direct"].head()
log.calculate_confidence_intervals(option="SM", stdz="indirect", measure="ratio", test_method="wald")["indirect_ratio"].head()
lin.test(reference="median").head()
log.test(test_method="wald").head()
log.test_standardized(measure="direct_rate").head()
lin.plot_funnel(); lin.plot_provider_effects(); lin.plot_coefficient_forest()
log.plot_standardized_measures(stdz="indirect", measure="ratio", test_method="wald")
```

## `null`, `reference`, `stdz`, `alternative`

- `null` (standardized measures, confidence intervals and plots): `"median"` (the median provider effect, γ or α), `"mean"` (the
  group-size-weighted mean effect) or a number on the effect scale.
- `reference` (`test` and `test_standardized`): the reference effect γ₀ the tests compare against, in the same three forms. The default is
  `"median"`, except `0` (the random-effect mean, as in R pprof) for the random-effect classes.
- `stdz`: `"indirect"`, `"direct"` or a list of both.
- `alternative`: `"two_sided"` (or `"two-sided"`), `"less"` or `"greater"`. Confidence intervals for the provider effects themselves (`option="gamma"` / `"alpha"`) are two-sided only.

## `calculate_standardized_measures`

| Estimator | Returned dict | Columns |
|---|---|---|
| `LinearFixedEffectModel`, `LinearRandomEffectModel` | `"indirect"` | `group_id`, `indirect_difference`, `observed`, `expected` |
| | `"direct"` | `group_id`, `direct_difference`, `observed`, `expected` |
| `LogisticFixedEffectModel` | `"indirect"` | `group_id`, `indirect_ratio`, `indirect_rate`, `observed`, `expected` |
| | `"direct"` | `group_id`, `direct_ratio`, `direct_rate`, `observed`, `expected`, `n_pop` |
| `LogisticRandomEffectModel`, `LogisticMixedEffectModel` | `"indirect"` | as the fixed-effect logistic indirect table |

Signatures: `providers=None, stdz="indirect", null="median"` for the linear and mixed classes; the logistic fixed-effect version adds
`include_extreme_obs=False, extreme_obs_total_n=None`; the logistic random-effect version adds a leading `group_var=None`.

## `calculate_confidence_intervals`

```text
LogisticFixedEffectModel.calculate_confidence_intervals(
    providers: 'Optional[Union[list, np.ndarray]]' = None,
    level: 'float' = 0.95,
    option: 'str' = 'SM',
    stdz: 'Union[str, list]' = 'indirect',
    null: 'Union[str, float]' = 'median',
    measure: 'Union[str, list]' = ('rate', 'ratio'),
    alternative: 'str' = 'two_sided',
    test_method: 'str' = 'exact',
)
```

```text
LogisticRandomEffectModel.calculate_confidence_intervals(
    group_var: 'Optional[str]' = None,
    providers: 'Optional[Union[List, Array]]' = None,
    level: 'float' = 0.95,
    option: 'str' = 'SM',
    stdz: 'Union[str, List[str]]' = 'indirect',
    null: 'Union[str, float]' = 'median',
    measure: 'Union[str, List[str]]' = ('rate', 'ratio'),
    alternative: 'str' = 'two_sided',
)
```

| Estimator | `option` | Returned dict (keys → columns) |
|---|---|---|
| Linear fixed effect | `"gamma"` | `"gamma_ci"` → `group_id`, `gamma`, `lower`, `upper` |
| | `"SM"` | `"indirect_ci"`, `"direct_ci"` → the standardized-measure columns plus `lower`, `upper` (subject to K2) |
| Linear random effect | `"alpha"` | `"alpha_ci"` → `group_id`, `alpha`, `alpha_lower`, `alpha_upper` |
| | `"SM"` | `"indirect_ci"` (K2), `"direct_ci"` |
| Logistic fixed effect | `"gamma"` | `"gamma_ci"` → `group_id`, `gamma`, `gamma_lower`, `gamma_upper` |
| | `"SM"` | one key per `stdz` × `measure`, e.g. `"indirect_ratio"` → `group_id`, `indirect_ratio`, `indirect_rate`, `observed`, `expected`, `ci_ratio_lower`, `ci_ratio_upper` |

For the logistic fixed-effect model the interval method is `test_method` ∈ {`"wald"`, `"score"`, `"exact"`}; the standardized-measure
intervals are the transformed provider-effect intervals, which must therefore be two-sided.

## `test` and `test_standardized`

Every provider test follows one contract: a statistic per provider (chosen with `test_method` where a family offers several), the
reference effect `reference`, a null model (`null_model`, the theoretical N(0, 1) by default; see {ref}`empirical-null-guide`) and one
result table.

```text
LogisticFixedEffectModel.test(
    providers=None,
    *,
    test_method: 'str' = 'poibin_exact',
    reference='median',
    null_model=None,
    alternative: 'str' = 'two_sided',
    level: 'float' = 0.95,
    critical: 'Optional[float]' = None,
    interval: 'str' = 'inversion',
    n_resample: 'int' = 10000,
    seed=None,
)
```

```text
LogisticFixedEffectModel.test_standardized(
    measure: 'str' = 'direct_rate',
    *,
    providers=None,
    null_value='reference',
    transform='auto',
    null_model=None,
    population=None,
    reference='median',
    variance: 'str' = 'model',
    indirect_variance: 'str' = 'null',
    alternative: 'str' = 'two_sided',
    level: 'float' = 0.95,
    critical: 'Optional[float]' = None,
    interval: 'str' = 'inversion',
    bounds='auto',
)
```

```text
LogisticRandomEffectModel.test(
    providers=None,
    *,
    group_var: 'Optional[str]' = None,
    test_method: 'str' = 'wald',
    reference=0.0,
    null_model=None,
    alternative: 'str' = 'two_sided',
    level: 'float' = 0.95,
    critical: 'Optional[float]' = None,
    interval: 'str' = 'inversion',
    n_resample: 'int' = 10000,
    seed=None,
)
```

```text
LogisticMixedEffectModel.test(
    providers=None,
    *,
    test_method: 'str' = 'resampling',
    reference='median',
    null_model=None,
    alternative: 'str' = 'two_sided',
    level: 'float' = 0.95,
    critical: 'Optional[float]' = None,
    n_resample: 'int' = 10000,
    seed=None,
)
```

```text
LinearFixedEffectModel.test(
    providers=None,
    *,
    reference='median',
    null_model=None,
    alternative: 'str' = 'two_sided',
    level: 'float' = 0.95,
    critical: 'Optional[float]' = None,
    interval: 'str' = 'inversion',
)
```

```text
LinearRandomEffectModel.test(
    providers=None,
    *,
    reference=0.0,
    null_model=None,
    alternative: 'str' = 'two_sided',
    level: 'float' = 0.95,
    critical: 'Optional[float]' = None,
    interval: 'str' = 'inversion',
)
```

| Estimator | `test_method` (default first) | Default `reference` | Intervals |
|---|---|---|---|
| `LogisticFixedEffectModel.test` | `"poibin_exact"`, `"score"`, `"wald"`, `"bootstrap_exact"` | `"median"` | Wald only |
| `LogisticRandomEffectModel.test` | `"wald"`, `"poibin_exact"`, `"resampling"` | `0` | Wald only |
| `LogisticMixedEffectModel.test` | `"exact"`, `"poibin_exact"`, `"resampling"` | `"median"` | inverted test (`"exact"`, `"poibin_exact"`) |
| `LinearFixedEffectModel.test` | Wald with a Student-t reference on n − p − m degrees of freedom | `"median"` | t intervals |
| `LinearRandomEffectModel.test` | Wald (normal reference) | `0` | normal intervals |

- **Exact and Monte Carlo tests** (`"poibin_exact"`, `"bootstrap_exact"`, `"resampling"`) test the provider's event count with its
  effect set to γ₀. Two-sided p-values are mid-p; one-sided p-values are `P(X >= O)` or `P(X <= O)`, as in R pprof. `"resampling"`
  draws the other random effects from their posterior (He et al. 2013). The Monte Carlo tests use `n_resample` draws and `seed`; a
  simulated tail probability of zero is replaced by `0.5 / n_resample`, except in `LogisticMixedEffectModel.test`, which uses the
  exact tails of the same null for those providers. The mixed-effect model's default `"exact"` draws each cluster's effect once for
  all of a provider's patients in that cluster and computes the count's distribution exactly.
- **Binomial outcomes** (a logistic fixed-effect model fitted with `n_var`) are weighted by their trials in the score, exact and bootstrap
  tests; the exact test expands trials up to 20,000 per provider.
- **The logistic Wald test** uses the normal reference, as R pprof does. It is unreliable for providers at the numerical bound of γ
  (see `pprof_py.inference.at_bound`).
- **`providers`** restricts the rows reported; γ₀ and any empirical null always use all providers.
- **`test_standardized`** tests a standardized measure (`measure`: `"direct_rate"`, `"direct_ratio"`, `"indirect_rate"`,
  `"indirect_ratio"` or `"gamma"`). Its null value defaults to the measure at γ₀ (`null_value="reference"`), so it agrees with a test of
  γ = γ₀. Indirect measures use the variance of the observed count under γ₀ (`indirect_variance="null"`, a score-type test) and an
  identity working scale (`transform="auto"`); `indirect_variance="fitted"` with `transform="log"` reproduces earlier versions'
  construction. `variance="robust"` uses sandwich variances, `population=` sets the standard population for direct measures, and
  `bounds="auto"` clips identity-scale intervals to the measure's range. There is no fixed scale factor: earlier versions divided every
  z-statistic by 1.81 by default, which `null_model=FixedNull(sd=1.81)` reproduces.

Every `test()` and `test_standardized()` returns a `DataFrame` indexed by `provider` with the columns
`pprof_py.inference.PROVIDER_TEST_COLUMNS`:

| Column | Meaning |
|---|---|
| `estimate`, `se` | The provider effect (γ̂ or BLUP) or standardized measure, and its standard error (Wald-type statistics only; otherwise `NaN`) |
| `null_value` | The value under the null on the estimate's scale (γ₀ for provider-effect tests) |
| `transformed`, `se_transformed`, `null_transformed` | The same on the test's working scale (identity for provider effects) |
| `z_raw` | The test statistic as a z-value. Exact and Monte Carlo p-values are converted so that the normal tail reproduces them (the sign follows the smaller tail, so a provider essentially at its expected count can have either sign; flags are unaffected), and a t-statistic enters as Φ⁻¹(F_t(t)) |
| `null_mean`, `null_sd`, `null_group` | The null each provider is calibrated against (0, 1 and `NaN` under the theoretical null) |
| `z_adjusted` | `(z_raw - null_mean) / null_sd` |
| `p_value` | From `z_adjusted` and `alternative` |
| `flag` | Nullable integer: `1` above the null value, `-1` below, `0` not significant, `NA` not tested |
| `ci_lower`, `ci_upper` | Interval for `estimate`, inverted from the calibrated test (`NaN` where the method has no interval) |

`result.attrs` records `null_model`, `alternative`, `level`, `critical`, `interval` and, for provider-effect tests, `test_method` and
`reference`.

`"poibin_exact"` needs the `fast_poibin` package (installed with `pprof_py`; the model documents' `pip install poibin` is outdated).

## Plotting methods

All plots use Matplotlib and return `None` (they draw on the current figure). Common arguments: `group_ids` (subset of providers), `level`,
`use_flags` (colour by `flag`), `null`, and `**plot_kwargs` (titles, sizes; see the docstrings). The plots pass their `null` to `test()` as
`reference`; a provider the test could not evaluate (`flag` is `NA`) is drawn as not flagged.

| Estimator | Methods |
|---|---|
| `LinearFixedEffectModel`, `LinearRandomEffectModel` | `plot_funnel(stdz="indirect", null="median", target=0.0, alpha=0.05, …)`, `plot_provider_effects(…, test_method=None)`, `plot_standardized_measures(…, measure="difference", …)`, `plot_coefficient_forest(orientation="vertical", refline_value=0.0, …)`, `plot_residuals(…)`, `plot_qq(…)` |
| `LogisticFixedEffectModel` | `plot_funnel(test_method="score", target=1.0, …)`, `plot_provider_effects(test_method="wald", …)`, `plot_standardized_measures(measure="ratio", test_method="score", …)`, `plot_coefficient_forest(…)`; `plot_residuals` and `plot_qq` raise `NotImplementedError` |
| `LogisticRandomEffectModel` | as above with a leading `group_var=None` and `test_method="wald"` defaults; no residual plots |
| `LogisticMixedEffectModel` | no plotting methods |

`plot_standardized_measures` on the linear models draws the intervals of K2. Two standalone functions are exported: `pprof_py.plot_caterpillar(df, estimate_col="estimate",
ci_lower_col="lower", ci_upper_col="upper", group_col=None, flag_col=None, …)` and `pprof_py.plotting.plot_funnel(df, limits_df, *, estimate_col, precision_col,
flag_col, target, alpha_levels, …)`. Shared styling constants live in `pprof_py.plotting.style`. (The module docstring of `pprof_py.plotting.coefficients`
mentions coefficient paths, but the module only defines `plot_caterpillar`.)
