(ll_ref_logistic)=
# Reference: logistic fixed-effect, random-effect and mixed-effect models

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

## `LogisticFixedEffectModel`

Logit model with one free effect γ per provider, fitted by the SerBIN algorithm (default) or BAN; suitable for many providers.

```python
from pprof_py import LogisticFixedEffectModel

fe = LogisticFixedEffectModel().fit(df, y_var="event", x_vars=X_COLS, provider_var="provider")   # X first; not fit(y, X, group)
fe.summary()                                   # Wald table (test_method="wald")
fe.summary(test_method="score")                # or "lr"
fe.test(test_method="poibin_exact").head()     # exact Poisson-binomial provider test (default)
fe.calculate_standardized_measures()["indirect"].head()
```

| Constructor parameter | Default | Meaning |
|---|---|---|
| `algorithm` | `"Serbin"` | `"Serbin"` or `"Ban"`; on test data the two agree to 4e-9. |
| `use_dataprep` | `True` | Run `DataPrep` first: missing-value and zero-variance checks (errors), correlation and VIF checks (logged warnings), provider screening. |
| `screen_providers` | `True` | Exclude small providers. In a test, a provider with 5 rows was dropped at the default cutoff while a provider with no events was kept. |
| `cutoff` | `10` | Screening keeps providers with more than `cutoff` records. |
| `log_event_providers` | `True` | Log providers with no or all events. |
| `threshold_cor`, `threshold_vif` | `0.9`, `10` | Multicollinearity thresholds. |

`fit(X, y=None, provider_id=None, x_vars=None, y_var=None, provider_var=None, n_var=None, obs_id_var=None, use_dataprep=None, screen_providers=None,
log_event_providers=None, cutoff=None, threshold_cor=None, threshold_vif=None, max_iter=10000, tol=1e-5, bound=10.0, backtrack=True)` —
the `None` overrides replace the constructor values for this call; `bound` clips γ; `obs_id_var` (observation/patient id) enables cluster-robust
variances (`summary(variance_type="robust")`); `n_var` accepts a trials-count column but the response is still validated as 0/1 — use one row per patient.

**Fitted attributes:** `coefficients_` (`"beta"` `(p,)`, `"gamma"` `(m,)`), `variances_`, `robust_variances_`, `fitted_` (probabilities),
`xbeta_`, `aic_`, `bic_`, `auc_`, `provider_ids_`, `provider_indices_`, `provider_sizes_`, `covariate_names_`, `outcome_`, `obs_ids_`, `N_`, `X`, `algorithm`,
`algorithm_type`, `use_dataprep`, `dataprep_options`.

**Methods:** `predict(X, provider_id=None, x_vars=None, provider_var=None)` (probabilities); `score(...)` (accuracy);
`summary(covariates=None, level=0.95, null=0, alternative="two_sided", test_method="wald", variance_type="model")`;
`add_providers(provider_ids, gamma, se_gamma=0.01, group_sizes=None)` appends providers left out of the fit (for example zero-event
providers, with γ = −17, or all-event providers, with γ = 17) so that they appear in standardized measures — this reproduces the R
`PPPW` workflow; `get_fitted_params()`; the measure, test and plot methods in {ref}`ll_ref_measures`.

## `LogisticRandomEffectModel`

Bernoulli-logit GLMM with random intercepts, fitted like `lme4::glmer` (penalized IRLS with a Laplace second stage).

```python
from pprof_py import LogisticRandomEffectModel

re = LogisticRandomEffectModel(verbose=False).fit(df, y_var="event", x_vars=X_COLS, provider_var="provider")
re.coefficients_["beta"]                       # fixed effects (log-odds), Series
re.sigma_                                      # {"provider": random-effect SD}
re.get_random_effects().head()                 # BLUPs
re.predict(df, x_vars=X_COLS, type="link")[:3] # or type="response" (probabilities)
re.test(test_method="wald").head()
```

| Constructor parameter | Default | Meaning |
|---|---|---|
| `max_iter_pirls`, `tol_pirls` | `100`, `1e-8` | Inner penalized-IRLS limits. |
| `max_iter_outer`, `tol_outer` | `200`, `1e-7` | Outer optimizer limits. |
| `sigma_upper` | `inf` | Numerical upper bound for σ. |
| `stage2` | `True` | Run the Laplace (nAGQ = 1) second stage. |
| `optimizer_stage1`, `optimizer_stage2` | `"bobyqa"`, `"nelder-mead"` | `"bobyqa"` requires `nlopt` (`pip install nlopt`); without it the code silently falls back — set `"powell"` explicitly if `nlopt` is not installed. |
| `verbose` | `True` | Print progress. |

`fit(X, y_var, x_vars=None, provider_var=None, cluster_vars=None, offset_var=None, include_intercept=True, verbose=None)`. Several grouping
factors are accepted (`sigma_` then has one entry per factor); the measure methods take `provider_var` to choose the factor.

**Fitted attributes:** `coefficients_`, `variances_`, `sigma_` (dict), `loglike_`, `aic_`, `bic_`, `converged_`, `pirls_converged_`, `pirls_iterations_`,
`fitted_`, `residuals_`, `outcome_`, `groups_`, `group_sizes_`, `covariate_names_`, `pwrss_`, `ldL2_` and optimizer diagnostics.
**Methods:** `summary()`, `get_random_effects(var=None)`, `get_sigma(var=None)`, `predict(X, *, x_vars=None, re_vars=None,
offset_var=None, use_re=False, type="response")`, `pearson_residuals()`, `deviance_residuals()`, and the methods in {ref}`ll_ref_measures`.

**Agreement with lme4.** On the synthetic dataset used for {ref}`ll_ref_linear` (binary outcome), `glmer(nAGQ = 1)` and this model agreed to
about 2e-5 in β, the random-effect SD, the log-likelihood and the BLUPs — looser than the linear model, and not covered by any test in the repository.

## `LogisticMixedEffectModel`

Fixed provider effects γ and a random cluster effect α ~ N(0, σ²): `logit P(Y = 1) = γ_provider + α_cluster + Xβ`. The random effect is
integrated out by Gauss–Hermite quadrature, providers are updated by Newton–Raphson. **Starting values are required.**

```python
import numpy as np
from pprof_py import LogisticMixedEffectModel

mixed = LogisticMixedEffectModel(n_nodes=10, max_iter=200)
mixed.fit(df, y_var="event", x_vars=X_COLS, provider_var="provider", cluster_var="cluster",
          gamma_init=np.zeros(m), beta_init=np.zeros(3), sigma_init=0.5, verbose=False)
# mixed.summary(stage1_model=fe) reports the Stage 1 Wald table for beta
mixed.test(test_method="resampling", n_resample=200).head()
```

- **Constructor:** `n_nodes=20`, `max_iter=10000`, `tol=1e-5`, `bound=10.0`, `bound_mode="relative"` (γ clipped to
  `median ± bound`; `"absolute"` clips to `±bound` as in R), `convergence_criterion="relative"` (or `"max_delta_gamma"`).
- **`fit(data, y_var, x_vars, provider_var, cluster_var, gamma_init, beta_init, sigma_init, obs_var=None, verbose=True, stage1_model=None)`:** `gamma_init`
  has one entry per provider, `beta_init` one per covariate. `y_var` is normally a boundary-adjusted outcome (`Y_adj`) that keeps γ finite;
  `obs_var` names the true 0/1 outcome used for observed counts and resampling p-values (defaults to `y_var`).
- **Attributes:** `gamma_`, `beta_`, `sigma_`, `xbeta_`, `fitted_`, `alpha_mean_`, `alpha_var_`, `alpha_mean_cluster_`, `alpha_var_cluster_`,
  `provider_ids_`, `cluster_ids_`, `n_providers_`, `n_clusters_`, `iterations_`, `convergence_`, `coefficients_`.
- **Methods:** `summary(stage1_model=None, covariates=None, level=0.95, null=0.0, alternative="two_sided")` (the Stage 1 Wald table); `calculate_standardized_measures(providers=None, stdz="indirect", null="median")`;
  `test(providers=None, *, test_method="exact", reference="median", null_model=None, alternative="two_sided", level=0.95,
  critical=None, n_resample=10000, seed=None)`, which returns the shared result table of {ref}`ll_ref_measures`.
