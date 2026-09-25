(survival_ref_penalized)=
# Reference: penalized Cox estimators

Five estimators fit Cox models over a path of penalty strengths λ. All reuse `CoxPH`'s likelihood engine, so strata,
offsets, weights, left truncation and both tie methods carry over unchanged.

| Estimator | Penalty | Extra |
|---|---|---|
| `PenalizedCoxPH` / `PenalizedCoxPHCV` | Ridge / lasso / elastic net | glmnet conventions; K-fold CV |
| `GroupLassoCoxPH` / `GroupLassoCoxPHCV` | Group lasso / sparse group lasso | Groups of features selected together |
| `ProviderPenalizedCoxPH` | Any of the above on β | Unpenalized provider effects γ |

Chapter 8 of the tutorial explains penalization; this page lists what the classes accept and expose.
Common names and defaults are compared across families in {ref}`survival_ref_conventions`.

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
n = 400
X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
provider = rng.integers(0, 12, n)
latent = rng.exponential(1 / np.exp(0.6 * X["x0"] - 0.4 * X["x1"]))
censor = rng.uniform(0.5, 3.0, n)
time = np.round(np.minimum(latent, censor), 3) + 0.001
event = (latent <= censor).astype(int)
```

## `PenalizedCoxPH`

```python
from pprof_py import PenalizedCoxPH

path = PenalizedCoxPH(alpha=0.5, n_lambda=30).fit(X, duration=time, event=event)
path.summary()                                   # one row per lambda
path.coef_at(path.lambda_path_[10])              # interpolated in log(lambda)
path.nonzero_features(path.lambda_path_[10])     # names with nonzero coefficients
```

| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | `1.0` | glmnet mixing: `0` ridge, `1` lasso. |
| `n_lambda` | `100` | Length of the automatic path. |
| `lambda_min_ratio` | `None` | Smallest λ as a fraction of `lambda_max_`; auto = `1e-2` if `n_obs < n_features`, else `1e-4`. |
| `lambda_path` | `None` | `None` = automatic; a float = fit exactly that λ; a sequence = exactly those λ (fit in descending order, warm-started). |
| `penalty_factor` | `None` | Per-feature multiplier; `0` = never penalized. Rescaled internally to sum to `n_features`. |
| `standardize` | `True` | Scale columns by their weighted population SD (no centering) before fitting; coefficients are returned in original units. |
| `ties` | `"breslow"` | `"breslow"` or `"efron"` (glmnet supports Breslow only). |
| `fit_intercept` | `False` | Leave off. |
| `max_outer_iter`, `outer_tol`, `max_inner_iter`, `inner_tol` | `100`, `1e-9`, `1000`, `1e-10` | Proximal-Newton outer loop and coordinate-descent inner loop. |

`fit(X, duration, event, start, stop, strata, offset, sample_weight)` — there is no `cluster`.

**Attributes.** `coef_path_` `(n_lambda_, p)`, `lambda_path_` (descending), `lambda_max_`,
`log_likelihood_path_`, `deviance_ratio_path_` (glmnet's `dev.ratio`), `n_nonzero_path_`, `converged_path_`,
`n_iter_path_`, `log_likelihood_null_`, `column_scale_`, `n_obs_`, `n_events_`, `n_features_in_`,
`feature_names_in_`. `coef_` and `lambda_` exist only for a single-λ fit.

**Methods.** `coef_at(lambda_value)`, `nonzero_features(lambda_value=None)`,
`predict_linear / predict_partial_hazard / predict(X, offset=None, lambda_value=None)`, `summary()` (columns `lambda`,
`n_nonzero`, `deviance_ratio`, `log_likelihood`, `converged`). For a multi-λ path, pass `lambda_value` to the
predict methods.

## `PenalizedCoxPHCV`

```python
from pprof_py import PenalizedCoxPHCV

cv = PenalizedCoxPHCV(alpha=1.0, n_lambda=30, n_folds=5, random_state=0).fit(X, duration=time, event=event)
cv.lambda_min_, cv.lambda_1se_, cv.coef_
cv.summary()          # lambda, n_nonzero, cv_mean_deviance, cv_se_deviance
```

Accepts every `PenalizedCoxPH` parameter plus `n_folds=10`, `fold_id`, `se_method` (`"analytical"` or
`"bootstrap"` — bootstrap supports Breslow ties only), `n_bootstrap=100`, `random_state`, and
`select` (`"lambda_min"` default, or `"lambda_1se"`). It computes the Verweij–Van Houwelingen grouped deviance per
fold, as `cv.glmnet(family="cox")` does. Attributes: `lambda_path_`, `cv_mean_deviance_`, `cv_se_deviance_`,
`lambda_min_`, `lambda_1se_`, `coef_`, `final_estimator_` (a `PenalizedCoxPH` refit on all data at the selected λ),
`full_fit_`, `fold_id_`, `cv_n_folds_`, `n_nonzero_path_`. The `predict_*` methods use `coef_` and take no `lambda_value`.

## `GroupLassoCoxPH` and `GroupLassoCoxPHCV`

Fits the path for the penalty

```
lambda * [ (1 - alpha) * sum_g m_g * ||beta_g||_2  +  alpha * sum_j pf_j * |beta_j| ]
```

where `m_g` are group multipliers (default `sqrt(group size)`) and `pf_j` the per-feature penalty factors.

```python
from pprof_py import GroupLassoCoxPH

groups = np.array([1, 1, 2, 2, 3, 0])      # 0 = unpenalized; one label per column of X
gl = GroupLassoCoxPH(groups=groups, n_lambda=20).fit(X, duration=time, event=event)
gl.active_group_labels(which=-1)           # groups with nonzero norm at the last lambda
gl.group_norms_.shape                      # (n_lambda_, n_groups_)
```

| Parameter | Default | Notes |
|---|---|---|
| `groups` | *required* | Integer label per feature; `0` = unpenalized; labels must be contiguous once sorted. |
| `alpha` | `0.0` | **`0` = pure group lasso**, `0 < alpha < 1` sparse group lasso, `1` = lasso (groups ignored). This is the opposite orientation from `PenalizedCoxPH`'s default. |
| `group_multiplier` | `None` | Per-group `m_g`; default `sqrt(group_size)`. |
| `penalty_factor` | `None` | L1-part factors; only active when `alpha > 0`. |
| `n_lambda`, `lambda_min_ratio`, `lambda_path`, `standardize`, `ties`, `fit_intercept` | as `PenalizedCoxPH` | |
| `method` | `"proximal_newton"` | `"MM"` is accepted by the constructor but `fit` raises `NotImplementedError`. |
| `orthogonalize` | `False` | Accepted; documented as a placeholder for the (unimplemented) MM method. |
| `use_active_set` | `False` | Active-set screening in the inner solver. (The class docstring names this parameter `active_set`; the argument is `use_active_set`.) |
| `max_outer_iter`, `outer_tol`, `max_inner_iter`, `inner_tol` | `100`, `1e-9`, `1000`, `1e-10` | |

Extra attributes beyond the path attributes above: `group_norms_`, `active_groups_` (one boolean mask per λ),
`df_path_`, `groups_` (canonicalised labels), `group_sizes_`, `n_groups_`, `group_weights_`.
`GroupLassoCoxPHCV` adds the CV parameters and attributes of `PenalizedCoxPHCV` (event-stratified folds; λ values where any
fold produced a non-finite deviance are discarded; a warning is raised if a stratum is missing from a training fold).
`se_method="bootstrap"` runs in the current code even though the class docstring still calls it a placeholder.

## `ProviderPenalizedCoxPH`

The linear predictor is `eta = gamma_provider + X @ beta + offset`. The provider effects `gamma` are **not penalized**;
β is penalized. Fitting alternates a Newton update of γ (clamped to `median(γ) ± provider_bound`) with the penalized
proximal-Newton update of β, warm-starting both along the λ path.

```python
from pprof_py import ProviderPenalizedCoxPH

pp = ProviderPenalizedCoxPH(n_lambda=15).fit(X, duration=time, event=event, provider_id=provider)
lam = pp.lambda_path_[5]
pp.predict_provider_effect(lambda_value=lam)                        # gamma for every provider
pp.predict_provider_effect(provider_id=[0, 1], lambda_value=lam)       # selected providers
pp.predict_linear_with_provider(X.iloc[:4], provider[:4], lambda_value=lam)
```

Parameters: `penalty_type` (`"elastic_net"` default, `"group_lasso"`, `"sparse_group_lasso"`), `alpha=1.0`, `groups`,
`group_multiplier`, `penalty_factor`, the λ-path parameters, `standardize`, `ties`, `provider_bound=10.0`,
`provider_backtrack=False` (placeholder), `provider_max_iter=20`, `provider_tol=1e-6`, and solver tolerances
(`outer_tol` defaults to `1e-7` here, looser than the `1e-9` of the non-provider models).

`fit(..., provider_id=...)` **requires** `provider_id` (a `ValueError` is raised otherwise). Extra attributes:
`gamma_path_` `(n_lambda_, n_providers_)`, `provider_labels_`, `n_providers_`, `n_provider_iter_path_`,
`provider_converged_path_`. `summary()` adds `n_provider_iter` and `provider_converged` columns. For a multi-λ fit,
`predict_provider_effect` and `predict_linear_with_provider` **require** `lambda_value`; a single-λ fit
(`lambda_path=0.05`) exposes `coef_`, `gamma_` and `lambda_`.

## Validation status

| Estimator | Reference | Status |
|---|---|---|
| `PenalizedCoxPH` / `CV` | R `glmnet` 4.1.8 (`family="cox"`, `cv.glmnet`) | All tests in `test_penalized_r_comparison.py` pass on a fresh run (coefficient paths to 1e-4–5e-4, λ to 1e-6, CV deviance to 0.01); Efron ties checked by self-consistency against `CoxPH(ties="efron")` as λ → 0 |
| `GroupLassoCoxPH` / `CV` | none | Internal tests only |
| `ProviderPenalizedCoxPH` | none | Internal tests only |

See {ref}`survival_validation_tools` for how to re-run the comparisons.
