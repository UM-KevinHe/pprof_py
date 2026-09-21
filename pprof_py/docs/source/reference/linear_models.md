(ll_ref_linear)=
# Reference: linear fixed-effect and random-effect models

Both estimators fit `y = Xβ + provider effect + ε`. The fixed-effect model estimates one free intercept γ per provider by least
squares; the random-effect model treats provider intercepts as random draws (lme4-style, REML by default).

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

## `LinearFixedEffectModel`

```python
from pprof_py import LinearFixedEffectModel

fe = LinearFixedEffectModel().fit(df, y_var="y", x_vars=X_COLS, group_var="provider")
fe.summary()                                  # covariate table
fe.coefficients_["gamma"].shape               # (m, 1)
fe.test().head()                              # provider tests vs the median provider effect
```

- **Constructor:** `gamma_var_option="complete"` (default: the variance of γ includes the uncertainty in β) or `"simplified"` (σ²/n_i).
- **`fit(X, y=None, groups=None, x_vars=None, y_var=None, group_var=None)`** — pass either a `DataFrame` plus column names, or arrays
  `fit(X, y, groups)`. No intercept is added.
- **Fitted attributes:** `coefficients_`, `variances_`, `sigma_` (residual SD, divisor n − m − p), `aic_`, `bic_`, `fitted_`, `residuals_`,
  `xbeta_`, `outcome_`, `groups_`, `group_indices_`, `group_sizes_`, `covariate_names_`, `gamma_var_option`.
- **Methods:** `predict(X, groups=None, x_vars=None, group_var=None)`; `score(X, y, groups)` (R²); `get_fitted_params()` (dict with
  `coefficients`, `variances`, `sigma`, `aic`, `bic`); `summary(covariates=None, level=0.95, null=0, alternative="two_sided")`;
  the measure, test and plot methods in {ref}`ll_ref_measures`.

## `LinearRandomEffectModel`

```python
from pprof_py import LinearRandomEffectModel

re = LinearRandomEffectModel(verbose=False).fit(df, y_var="y", x_vars=X_COLS, group_var="provider")
re.random_effect_sd_                          # {"provider": sigma_u}
re.sigma_                                     # residual SD
re.coefficients_["alpha"].head()              # BLUPs (random intercepts), Series
re.fixed_effects_table()                      # Estimate, Std. Error, t value, df, Pr(>|t|)
re.test(null=0).head()                        # null must be numeric (not 'median')
```

| Constructor parameter | Default | Meaning |
|---|---|---|
| `reml` | `True` | REML (lme4's default) or ML; `fit(reml=…)` overrides it. |
| `optimizer` | `"powell"` | `"powell"` or `"nelder-mead"` for the variance components. |
| `max_iter_outer`, `tol_outer` | `200`, `1e-8` | Optimizer limits. |
| `max_iter_inner` | `100` | Kept for API symmetry (the Gaussian solve is direct). |
| `theta_upper` | `inf` | Upper bound on the relative random-effect SDs (the lower bound is 0). |
| `verbose` | `True` | Print progress. |

`fit(X, y_var, x_vars=None, group_vars=None, group_var=None, offset_var=None, weights_var=None, include_intercept=True, reml=None, verbose=None)`:
`X` must be a `DataFrame`; `weights_var` names inverse-residual-variance prior weights (lme4's `weights=`); `offset_var` a known offset;
`group_vars` several random-intercept factors (crossed).

**Fitted attributes:** `coefficients_` (`"beta"` and `"alpha"`, both `Series`), `variances_`, `sigma_`, `random_effect_sd_` (dict by factor),
`loglike_`, `reml_`, `aic_`, `bic_`, `converged_`, `theta_`, `fitted_`, `residuals_`, `xbeta_`, `outcome_`, `groups_`, `group_indices_`,
`group_sizes_`, `covariate_names_`, `nobs_`, `n_fixed_effects_`, `n_random_effects_`, `outer_iterations_`, `objective_`, `pwrss_`,
`ldL2_`, `ussq_`, `residual_variance_`, `optimizer_result_`.

**Methods:** `summary()` (same columns as the fixed-effect model), `fixed_effects_table()`, `conf_int(level=0.95)` (`lower`, `upper`),
`standard_errors()`, `residual_standard_error()`, `fitted_values()`, `get_sigma()`, `get_random_effect_sd(group_var=None)`,
`get_random_effects(group_var=None)`, `predict(X, *, x_vars=None, group_vars=None, group_var=None, offset_var=None, use_re=False)`
(fixed effects only by default; `use_re=True` adds BLUPs for known levels and 0 for unknown ones, lme4's conditional convention), plus
the measure, test and plot methods in {ref}`ll_ref_measures`.

With several grouping factors `fit` works and `random_effect_sd_` has one entry per factor, but the provider-profiling methods (`test()`, `calculate_standardized_measures()`) currently support a single grouping factor only.

### Agreement with lme4

The repository contains no lme4 comparison. A fresh check with R 4.3.3 and `lme4` 1.1.35.1 on one synthetic dataset (1,834 rows, 20
providers, three covariates; outcome `y`) gave these largest absolute differences:

| Setting | β | σ | RE SD | log-likelihood | BLUPs |
|---|---|---|---|---|---|
| REML, unweighted | 1e-10 | 2e-10 | 8e-9 | 3e-12 | 3e-9 |
| REML, weighted | 3e-10 | 8e-10 | 3e-8 | 2e-12 | 1e-8 |
| ML, unweighted | 1e-9 | 3e-9 | 1e-7 | 2e-12 | 5e-8 |
| ML, weighted | 2e-9 | 6e-9 | 2e-7 | 5e-12 | 8e-8 |

This supports the README's "10⁻⁷–10⁻⁹" statement (the random-effect SD and BLUPs are the loosest quantities). To repeat it:

```text
lmer(y ~ x1 + x2 + x3 + (1|provider), data = d, REML = TRUE, weights = w)     # R
LinearRandomEffectModel(reml=True).fit(d, y_var="y", x_vars=[...], group_var="provider", weights_var="w")   # Python
compare fixef(), sigma(), VarCorr()$provider, logLik(), ranef()$provider with beta, sigma_, random_effect_sd_, loglike_, alpha
```
