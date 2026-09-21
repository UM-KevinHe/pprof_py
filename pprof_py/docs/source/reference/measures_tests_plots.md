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
lin.test(null="median").head()
log.test(test_method="wald", null="median").head()
log.test_standardized(measure="direct_rate").head()
lin.plot_funnel(); lin.plot_provider_effects(); lin.plot_coefficient_forest()
log.plot_standardized_measures(stdz="indirect", measure="ratio", test_method="wald")
```

## `null`, `stdz`, `alternative`

- `null`: `"median"` (the median provider effect, γ or α), `"mean"` (the group-size-weighted mean effect) or a number on the effect scale.
  **Exception:** `LinearRandomEffectModel.test` needs a number (K4 in {ref}`ll_ref_conventions`).
- `stdz`: `"indirect"`, `"direct"` or a list of both.
- `alternative`: `"two_sided"`, `"less"` or `"greater"`. Confidence intervals for the provider effects themselves (`option="gamma"` / `"alpha"`) are two-sided only.

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

```text
LogisticFixedEffectModel.test(
    providers: 'Optional[Union[list, np.ndarray]]' = None,
    level: 'float' = 0.95,
    test_method: 'str' = 'poibin_exact',
    score_modified: 'bool' = True,
    null: 'Union[str, float]' = 'median',
    n_bootstrap: 'int' = 10000,
    alternative: 'str' = 'two_sided',
)
```

```text
LogisticFixedEffectModel.test_standardized(
    providers: 'Optional[Union[list, np.ndarray]]' = None,
    measure: 'str' = 'direct_rate',
    null: 'Union[str, float]' = 'mean',
    level: 'float' = 0.95,
    variance_type: 'str' = 'model',
    empirical_null: 'bool' = False,
    groupwise: 'bool' = True,
    n_groups: 'int' = 4,
    remove_outliers: 'bool' = True,
    scale: 'float' = 1.81,
    alternative: 'str' = 'two_sided',
    z_scale: 'str' = 'auto',
    include_extreme_obs: 'bool' = False,
    extreme_obs_total_n: 'Optional[float]' = None,
)
```

```text
LogisticRandomEffectModel.test(
    group_var: 'Optional[str]' = None,
    providers: 'Optional[Union[List, Array]]' = None,
    level: 'float' = 0.95,
    test_method: 'str' = 'wald',
    null: 'Union[str, float]' = 'median',
    alternative: 'str' = 'two_sided',
    n_resample: 'int' = 10000,
    empirical_null: 'bool' = False,
    n_strata: 'int' = 4,
    strata_var: 'Optional[Array]' = None,
    seed: 'int' = 1,
)
```

```text
LinearFixedEffectModel.test(
    providers: 'Optional[Union[list, np.ndarray]]' = None,
    level: 'float' = 0.95,
    null: 'Union[str, float]' = 'median',
    alternative: 'str' = 'two_sided',
)
```

```text
LinearRandomEffectModel.test(
    providers: 'Optional[Union[list, np.ndarray]]' = None,
    level: 'float' = 0.95,
    null: 'float' = 0,
    alternative: 'str' = 'two_sided',
)
```

| Estimator | Result |
|---|---|
| Linear and logistic fixed / random effect (`test`) | `DataFrame` indexed by provider: `flag`, `p_value`, `stat`, `std_error`. `flag` = 1 above the null, −1 below, 0 not flagged. `LinearFixedEffectModel.test` also stores `result.attrs["provider_size"]`. |
| `LogisticRandomEffectModel.test(test_method="resampling")` | `flag`, `p_value`, `p_theo`, `z_score`, `srr` |
| `LogisticMixedEffectModel.test` | `provider_id`, `gamma`, `srr`, `obs`, `exp`, `p_theo`, `z_score`, `p_empi`, `flag` |
| `LogisticFixedEffectModel.test_standardized` | `estimate`, `se`, `transformed`, `se_transformed`, `null_value`, `z_score`, `intercept`, `scale`, `z_calibrated`, `flag`, `p_value`, `ci_lower`, `ci_upper` (`empirical_null=True` calibrates the z-scores with an empirical null fitted within provider-size groups) |

`"poibin_exact"` needs the `fast_poibin` package (installed with `pprof_py`; the model documents' `pip install poibin` is outdated).

## Plotting methods

All plots use Matplotlib and return `None` (they draw on the current figure). Common arguments: `group_ids` (subset of providers), `level`,
`use_flags` (colour by `flag`), `null`, and `**plot_kwargs` (titles, sizes; see the docstrings).

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
