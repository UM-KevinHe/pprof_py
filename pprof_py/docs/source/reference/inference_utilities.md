(survival_ref_inference)=
# Reference: inference utilities for survival models

Functions used after (or beneath) a fit. They take plain NumPy arrays, so a fitted `CoxPH` need not be involved.
The four inputs shared by most of them come from `validate_fit_inputs`:

```python
import numpy as np, pandas as pd
from pprof_py import CoxPH
from pprof_py.data.survival_validation import validate_fit_inputs

rng = np.random.default_rng(0)
n = 300
X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
latent = rng.exponential(1 / np.exp(0.5 * X["a"]))
censor = rng.uniform(0.5, 3.0, n)
time = np.round(np.minimum(latent, censor), 3) + 0.001
event = (latent <= censor).astype(int)

model = CoxPH().fit(X, duration=time, event=event)
data = validate_fit_inputs(X, duration=time, event=event)   # keys: X, start, stop, event, offset, weight, strata_codes, strata_labels, feature_names
eta = data["X"] @ model.coef_ + data["offset"]
```

## Residuals (`pprof_py.inference.survival.residuals`)

| Function | Returns |
|---|---|
| `martingale_residuals(X, start, stop, event, eta, weight, strata_codes, strata_labels, baseline_hazard=None, ties="breslow")` | `(n,)` martingale residuals — what `CoxPH.martingale_residuals_` stores. `ties` selects the algorithm (Efron has its own). |
| `score_residuals(X, start, stop, event, eta, weight, strata_codes, strata_labels, ties="breslow")` | `(n, p)` score residuals `U` (R's `residuals(type="score")`). |
| `dfbeta_residuals(score_resid, weight, naive_covariance)` | `(n, p)`: `weight_i * U_i @ naive_covariance` (R's `dfbeta`). |

```python
from pprof_py.inference.survival.residuals import score_residuals, dfbeta_residuals

U = score_residuals(data["X"], data["start"], data["stop"], data["event"], eta,
                    data["weight"], data["strata_codes"], data["strata_labels"], ties="breslow")
dfbeta = dfbeta_residuals(U, data["weight"], model.naive_covariance_)
U.shape, dfbeta.shape
```

`martingale_residuals` is also exported from `pprof_py.inference.survival`; `score_residuals` and `dfbeta_residuals` must be
imported from the `residuals` module.

## Robust (sandwich) variance (`pprof_py.inference.survival.robust`)

`cluster_score_residuals(X, start, stop, event, weight, eta, strata, cluster, ties="breslow")` returns `(cluster_scores, cluster_labels)`
— score residuals summed within each cluster (memory `O(n_clusters · p)`). `robust_covariance(naive_covariance, cluster_scores)`
returns the sandwich `V_naive · UᵀU · V_naive`. `CoxPH` calls both when `robust=True` or `cluster=` is given. Numba kernels are used
when available; without numba a much slower pure-Python fallback runs and a warning is issued.

## Baseline hazard (`pprof_py.inference.survival.compute_baseline_hazard`)

`compute_baseline_hazard(X, start, stop, event, eta, weight, strata_codes, strata_labels, ties="breslow")` returns the
`stratum, time, hazard, survival` table of the cumulative baseline hazard at `eta = 0`, using the formula of the requested tie method.
`CoxPH.baseline_hazard_` is this table multiplied by `exp(mean offset)` to match R's `basehaz(centered = FALSE)`.

## Deviance helpers (`pprof_py.statistics.deviance`)

| Function | Purpose |
|---|---|
| `saturated_log_likelihood(stop, event, weight, strata_codes)` | Saturated log-likelihood (glmnet's `coxnet.deviance` convention). |
| `cox_deviance(log_likelihood, lsat)` | `2 * (lsat - log_likelihood)`. |
| `deviance_ratio(log_likelihood, log_likelihood_null, lsat)` | glmnet's `dev.ratio`. |
| `bootstrap_cv_se(eta_matrix, start, stop, event, weight, strata_codes, n_bootstrap=100, random_state=None, ties="breslow")` | Bootstrap SE of the cross-validated deviance. |
| `discrete_survival_loss(y_expanded, p_hat_expanded)` | Binary cross-entropy deviance for the discrete-time models. |

## Provider-profiling calibration: Poisson tests and the empirical null

These functions (in `pprof_py.inference.survival`, all exported there) turn per-provider observed and expected counts — for example the
two sides of an SMR/SHR from the two-stage workflow — into calibrated Z-scores, p-values and intervals.

```python
from pprof_py.inference.survival import (
    poisson_midp_zscore, poisson_exact_test, adjust_empirical_null, fit_empirical_null,
    log_ratio_zscore, log_ratio_confidence_intervals,
)

obs = rng.poisson(20, 60).astype(float)
exp_ = rng.uniform(15, 25, 60)
size = exp_ * 10                                   # any provider-size measure

z = poisson_midp_zscore(obs, exp_)                 # mid-p Poisson Z-scores
null = fit_empirical_null(z)                       # {'intercept', 'scale'}: one robust null for all providers
adj = adjust_empirical_null(z, size=size)          # z_adj, p_value, intercept, scale, group, params
p, lo, hi = poisson_exact_test(obs, exp_)          # classical two-sided test and CI on the ratio scale
```

| Function | Purpose |
|---|---|
| `poisson_midp_zscore(obs, exp)` | Mid-p Z-scores from Poisson counts (`p` floored at `1e-6`). |
| `poisson_exact_test(obs, exp, alpha=0.05, normal_threshold=100.0)` | Returns `(p_value, lower_ratio, upper_ratio)`: two-sided (not mid-p) p-value clipped to `[0, 0.999]`; exact chi-square interval when `exp < normal_threshold`, normal approximation above. |
| `log_ratio_zscore(ratio, stderr, zero_method="score")` | `log(ratio) / stderr`; zero ratios use `-1 / stderr` (`"score"`) or `NaN` (`"exclude"`). `stderr` must be finite and positive. |
| `fit_empirical_null(z, psi="bisquare", method="M", tuning=None, maxiter=1000, tol=1e-8)` | One `{'intercept', 'scale'}` for all providers: `rlm(z ~ 1)` with R's settings (least-squares start); `method="MM"` is available. |
| `fit_grouped_empirical_null(z, size, n_groups=4, ...)` | Group-specific `intercept` and `scale` (groups 1..`n_groups`) and each provider's `group`. Groups are quantiles of `size` (a size equal to a break joins the lower group); missing sizes are left ungrouped (`NaN`), as R's `cut()` does. |
| `adjust_empirical_null(z, size=None, n_groups=4, group_labels=None, common_mean=False, ...)` | Fits the null and returns adjusted Z-scores and p-values; `NaN` inputs stay `NaN`. |
| `poisson_confidence_bounds(obs, exp, p_value, intercept, scale, alpha, upper_cap)` | Calibrated bounds on expected counts by root-finding. |
| `log_ratio_confidence_intervals(ratio, log_ratio_z, stderr, intercept, scale, alpha)` | Dict with `test_stat`, `p_value`, `lower`, `upper` (log-normal intervals). |

These wrappers run on the shared empirical-null layer in `pprof_py.inference` ({ref}`empirical-null-guide`), which the logistic
and linear provider tests use. Their R names are available as aliases in `pprof_py.inference.survival.empirical_null`:
`cal_Z_htaz`, `empirical_null_overall`, `empirical_null_groupwise`, `empirical_null_adjust` and `smr_ci_bounds`.
