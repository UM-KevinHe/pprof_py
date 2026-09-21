(ll_ref_penalized)=
# Reference: penalized, group-lasso and provider-penalized linear and logistic models

Elastic-net (glmnet-style), group-lasso and provider-penalized estimators, fitted along a path of penalty strengths λ by proximal-Newton
with coordinate descent. All follow the scikit-learn interface (`get_params`, `set_params`, `clone`).

| Estimator | Outcome | Penalty | Notes |
|---|---|---|---|
| `PenalizedLinear`, `PenalizedLinearCV` | Gaussian | ridge / lasso / elastic net | CV picks λ by K-fold mean squared error |
| `GroupLassoLinear` | Gaussian | group / sparse group lasso | no cross-validated variant |
| `PenalizedLogistic`, `PenalizedLogisticCV` | binary | ridge / lasso / elastic net | CV picks λ by deviance |
| `GroupLassoLogistic`, `GroupLassoLogisticCV` | binary | group / sparse group lasso | |
| `ProviderPenalizedLogistic`, `ProviderPenalizedLogisticCV` | binary | as above on β; provider effects γ unpenalized | two-layer provider profiling |

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
from pprof_py import PenalizedLinear, PenalizedLinearCV, PenalizedLogistic, GroupLassoLinear, ProviderPenalizedLogistic

Xa, y, yb = df[X_COLS].to_numpy(), df["y"].to_numpy(), df["event"].to_numpy()

path = PenalizedLinear(alpha=0.5, n_lambda=20).fit(Xa, y)      # a path: only *_path_ attributes
path.coef_at(path.lambda_path_[5])                              # coefficients at one lambda (interpolated)
path.summary(which=-1)                                          # feature, coef, nonzero at one path step

cv = PenalizedLinearCV(n_lambda=20, n_folds=5, random_state=0).fit(Xa, y)
cv.lambda_min_, cv.lambda_1se_, cv.coef_                        # coef_ is the 1-SE solution (use_1se=True by default)

groups = np.array([1, 1, 2])                                    # one label per column; 0 = unpenalized
gl = GroupLassoLinear(groups=groups, n_lambda=10).fit(Xa, y)
gl.active_group_labels()                                        # groups with nonzero norm at the last lambda

lg = PenalizedLogistic(alpha=1.0, n_lambda=20).fit(Xa, yb)
lg.predict_proba(Xa[:3], lambda_value=lg.lambda_path_[5])

pp = ProviderPenalizedLogistic(n_lambda=8).fit(Xa, yb, provider_id=provider)
pp.predict_provider_effect().head()                             # provider, gamma at the last lambda
```

## Conventions shared by all of them

- **Path estimators** (`PenalizedLinear`, `GroupLassoLinear`, `PenalizedLogistic`, `GroupLassoLogistic`, `ProviderPenalizedLogistic`) expose
  `coef_path_` (`n_lambda × p`), `intercept_path_`, `lambda_path_` (descending) and further `*_path_` arrays. They have **no `coef_`**;
  use `coef_at(lambda_value)` (interpolated), `intercept_at` (logistic), or index the path.
- **`*CV` estimators** refit on all data and expose `coef_`, `lambda_`, `lambda_min_`, `lambda_1se_`, `cv_mean_*_`, `cv_se_*_`, `cv_std_*_`, and `model_`
  (the refit path estimator). With the default `use_1se=True`, `lambda_` is `lambda_1se_`. The error attributes are named for the loss:
  `cv_*_mse_` (linear) and `cv_*_deviance_` (logistic).
- `lambda_path` may be `None` (automatic path of `n_lambda` values from `lambda_max_`), or explicit values. The automatic path always returns
  `n_lambda` points; glmnet may stop earlier.
- `standardize=True` scales columns by their population standard deviation before penalizing and returns coefficients on the original scale.
  `penalty_factor` rescales the penalty per feature (`0` = unpenalized). `fit_intercept=True` by default (unlike the survival estimators).
- Weights and offsets: every `fit` takes `sample_weight` and `offset`.
- `predict(X, lambda_value=None)` uses the last λ on the path when `lambda_value` is omitted (path estimators) or `coef_` (CV estimators).
- A predictor with zero weighted variance triggers that module's own `DegenerateFeatureWarning`.

## Signatures

```text
PenalizedLinear(
    alpha: 'float' = 1.0,
    n_lambda: 'int' = 100,
    lambda_min_ratio: 'Optional[float]' = None,
    lambda_path: 'Optional[np.ndarray]' = None,
    penalty_factor: 'Optional[np.ndarray]' = None,
    standardize: 'bool' = True,
    fit_intercept: 'bool' = True,
    max_outer_iter: 'int' = 100,
    outer_tol: 'float' = 1e-09,
    max_inner_iter: 'int' = 1000,
    inner_tol: 'float' = 1e-10,
)
```

```text
PenalizedLinearCV(
    alpha: 'float' = 1.0,
    n_lambda: 'int' = 100,
    lambda_min_ratio: 'Optional[float]' = None,
    lambda_path: 'Optional[np.ndarray]' = None,
    penalty_factor: 'Optional[np.ndarray]' = None,
    standardize: 'bool' = True,
    fit_intercept: 'bool' = True,
    n_folds: 'int' = 10,
    fold_id: 'Optional[np.ndarray]' = None,
    use_1se: 'bool' = True,
    random_state: 'Optional[int]' = None,
    max_outer_iter: 'int' = 100,
    outer_tol: 'float' = 1e-09,
    max_inner_iter: 'int' = 1000,
    inner_tol: 'float' = 1e-10,
)
```

```text
GroupLassoLinear(
    groups: 'np.ndarray',
    alpha: 'float' = 0.0,
    n_lambda: 'int' = 100,
    lambda_min_ratio: 'Optional[float]' = None,
    lambda_path: 'Optional[np.ndarray]' = None,
    penalty_factor: 'Optional[np.ndarray]' = None,
    group_multiplier: 'Optional[np.ndarray]' = None,
    standardize: 'bool' = True,
    fit_intercept: 'bool' = True,
    use_active_set: 'bool' = True,
    max_outer_iter: 'int' = 100,
    outer_tol: 'float' = 1e-09,
    max_inner_iter: 'int' = 1000,
    inner_tol: 'float' = 1e-10,
)
```

```text
PenalizedLogistic(
    alpha: 'float' = 1.0,
    n_lambda: 'int' = 100,
    lambda_min_ratio: 'Optional[float]' = None,
    lambda_path: 'Optional[np.ndarray]' = None,
    penalty_factor: 'Optional[np.ndarray]' = None,
    standardize: 'bool' = True,
    fit_intercept: 'bool' = True,
    max_outer_iter: 'int' = 100,
    outer_tol: 'float' = 1e-09,
    max_inner_iter: 'int' = 1000,
    inner_tol: 'float' = 1e-10,
    use_active_set: 'bool' = False,
)
```

```text
GroupLassoLogistic(
    groups: 'np.ndarray',
    alpha: 'float' = 0.0,
    n_lambda: 'int' = 100,
    lambda_min_ratio: 'Optional[float]' = None,
    lambda_path: 'Optional[np.ndarray]' = None,
    penalty_factor: 'Optional[np.ndarray]' = None,
    group_multiplier: 'Optional[np.ndarray]' = None,
    standardize: 'bool' = True,
    fit_intercept: 'bool' = True,
    use_active_set: 'bool' = True,
    max_outer_iter: 'int' = 100,
    outer_tol: 'float' = 1e-09,
    max_inner_iter: 'int' = 1000,
    inner_tol: 'float' = 1e-10,
)
```

```text
ProviderPenalizedLogistic(
    penalty_type: 'str' = 'elastic_net',
    alpha: 'float' = 1.0,
    groups: 'Optional[np.ndarray]' = None,
    group_multiplier: 'Optional[np.ndarray]' = None,
    gamma_bound: 'float' = 10.0,
    n_lambda: 'int' = 100,
    lambda_min_ratio: 'Optional[float]' = None,
    lambda_path: 'Optional[np.ndarray]' = None,
    penalty_factor: 'Optional[np.ndarray]' = None,
    standardize: 'bool' = True,
    fit_intercept: 'bool' = True,
    max_outer_iter: 'int' = 100,
    outer_tol: 'float' = 1e-07,
    max_inner_iter: 'int' = 1000,
    inner_tol: 'float' = 1e-10,
    provider_max_iter: 'int' = 10,
)
```

The remaining CV classes take the same arguments as their path counterparts plus the cross-validation settings:

```text
PenalizedLogisticCV(
    alpha: 'float' = 1.0,
    n_lambda: 'int' = 100,
    lambda_min_ratio: 'Optional[float]' = None,
    lambda_path: 'Optional[np.ndarray]' = None,
    penalty_factor: 'Optional[np.ndarray]' = None,
    standardize: 'bool' = True,
    fit_intercept: 'bool' = True,
    n_folds: 'int' = 10,
    fold_id: 'Optional[np.ndarray]' = None,
    use_1se: 'bool' = True,
    random_state: 'Optional[int]' = None,
    max_outer_iter: 'int' = 100,
    outer_tol: 'float' = 1e-09,
    max_inner_iter: 'int' = 1000,
    inner_tol: 'float' = 1e-10,
)
```

`fit` signatures: `fit(X, y, sample_weight=None, offset=None)`; provider estimators `fit(X, y, provider_id, sample_weight=None, offset=None)`.

## Fitted attributes

| Estimator | Attributes |
|---|---|
| `PenalizedLinear` | `coef_path_`, `intercept_path_`, `lambda_path_`, `lambda_max_`, `lambda_min_ratio_`, `penalty_factor_`, `column_scale_`, `deviance_path_`, `deviance_ratio_path_`, `null_deviance_`, `log_likelihood_path_`, `n_nonzero_path_`, `converged_path_`, `n_obs_`, `n_features_in_`, `feature_names_in_` |
| `PenalizedLinearCV` | `coef_`, `coef_path_`, `lambda_`, `lambda_min_`, `lambda_1se_`, `lambda_path_`, `cv_mean_mse_`, `cv_se_mse_`, `cv_std_mse_`, `model_` |
| `GroupLassoLinear` | `coef_path_`, `intercept_path_`, `lambda_path_`, `lambda_max_`, `column_scale_`, `deviance_path_`, `deviance_ratio_path_`, `null_deviance_`, `active_groups_path_`, `converged_path_`, `n_obs_`, `n_features_in_`, `feature_names_in_` (**no** `n_nonzero_path_`) |
| `PenalizedLogistic` | as `PenalizedLinear` plus `n_iter_path_` |
| `PenalizedLogisticCV` | `coef_`, `intercept_`, `coef_path_`, `intercept_path_`, `lambda_`, `lambda_min_`, `lambda_1se_`, `lambda_min_idx_`, `lambda_1se_idx_`, `lambda_path_`, `cv_mean_deviance_`, `cv_se_deviance_`, `cv_std_deviance_`, `model_` |
| `GroupLassoLogistic` | as `PenalizedLogistic` plus `active_groups_path_`, `group_norms_path_`, `group_sizes_`, `group_weights_`, `groups_`, `n_groups_` |
| `GroupLassoLogisticCV` | `coef_`, `coef_path_`, `lambda_`, `lambda_min_`, `lambda_1se_`, `lambda_path_`, `cv_mean_deviance_`, `cv_se_deviance_`, `cv_std_deviance_`, `model_` |
| `ProviderPenalizedLogistic` | as `PenalizedLogistic` plus `gamma_path_` (`n_lambda × n_providers`), `provider_labels_`, `n_providers_` |
| `ProviderPenalizedLogisticCV` | as `PenalizedLogisticCV` plus `gamma_`, `gamma_path_`, `provider_labels_`, `fold_assignment_` |

## Methods

| Estimator | Methods |
|---|---|
| `PenalizedLinear` | `coef_at(lambda_value)`, `predict(X, lambda_value=None)`, `summary(which=-1)` → `feature`, `coef`, `nonzero` |
| `PenalizedLinearCV` | `predict(X, lambda_value=None)` |
| `GroupLassoLinear` | `coef_at`, `predict`, `active_group_labels(which=-1)` (1-indexed group labels) |
| `PenalizedLogistic` | `coef_at`, `intercept_at`, `predict_proba(X, lambda_value=None)`, `predict(X, lambda_value=None, threshold=0.5)`, `summary(which=-1)` |
| `PenalizedLogisticCV`, `GroupLassoLogisticCV` | `predict_proba`, `predict` |
| `GroupLassoLogistic` | `coef_at`, `active_group_labels`, `predict_proba`, `predict` |
| `ProviderPenalizedLogistic` | `predict_provider_effect(which=-1)` → `provider`, `gamma`; `predict_proba(X, provider_id=None, lambda_value=None, which=-1)`; `predict(..., threshold=0.5)` |
| `ProviderPenalizedLogisticCV` | the same, with `which=None` (the selected λ) |

## Agreement with glmnet

The repository has no R-comparison test for these estimators (the docstring of `PenalizedLinear` states that it matches `glmnet(family="gaussian")`).
A fresh check against R 4.3.3 / glmnet 4.1.8 on a synthetic dataset (1,834 rows, three covariates, `nlambda = 100`):

| Family | α | λ sequence (max relative difference) | Coefficients at glmnet's own λ values (max abs difference) |
|---|---|---|---|
| Gaussian | 1 | 2e-14 | 3e-8 |
| Gaussian | 0.5 | 2e-14 | 9e-3 — glmnet's own solution is off by that much: `PenalizedLinear` matches an exact elastic-net solver (scikit-learn, `tol=1e-14`) to 6e-11 |
| Binomial | 1 | 5e-14 | 4e-5 |
| Binomial | 0.5 | 3e-14 | 2e-5 |

glmnet stopped early (45–61 of 100 λ values); `pprof_py` returned all 100, so the comparison uses the first values that both paths share.
