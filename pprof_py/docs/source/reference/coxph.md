(survival_ref_coxph)=
# Reference: `CoxPH`

`CoxPH` is the core estimator of the survival module: Cox proportional-hazards
regression fit by Newton–Raphson on the (Breslow or Efron) partial likelihood, designed to
reproduce R's `survival::coxph()`. Every other Cox-based estimator in the package
(penalized, group lasso, competing risks, selection) reuses its likelihood engine.

This page is a *reference*. For a guided introduction see the
[tutorial chapters](survival_start); for how each convention was checked against R see
{ref}`r_compatibility`.

```{note}
Everything on this page describes the code at version 0.4.0 and was checked by running it.
Where the behaviour is surprising, or differs from R, it is called out explicitly.
```

## Minimal example

```python
import numpy as np
import pandas as pd
from pprof_py import CoxPH

rng = np.random.default_rng(0)
n = 400
X = pd.DataFrame({
    "age": rng.normal(size=n),
    "sex": rng.integers(0, 2, n).astype(float),
    "bmi": rng.normal(size=n),
})
provider = rng.integers(0, 12, n)
latent = rng.exponential(1 / np.exp(0.5 * X["age"] - 0.3 * X["sex"]))
censor = rng.uniform(0.5, 3.0, n)
time = np.round(np.minimum(latent, censor), 3) + 0.001
event = (latent <= censor).astype(int)

model = CoxPH(ties="efron").fit(X, duration=time, event=event)
model.summary()          # coef, exp(coef), se(coef), z, p, lower_95%, upper_95%
model.converged_         # always check this -- see "Convergence" below
```

## Constructor parameters

| Parameter | Default | Meaning |
|---|---|---|
| `ties` | `"breslow"` | Tie-handling convention: `"breslow"` or `"efron"`. `"exact"` is recognised but `fit` raises `NotImplementedError`; any other string raises `ValueError`. A `TieMethod` instance is also accepted. **R's own default is Efron**, so pass `ties="efron"` when comparing with a plain `coxph()` call. |
| `fit_intercept` | `False` | Append a constant column named `"intercept"`. A Cox model has no identifiable intercept (it is absorbed by the baseline hazard), so leave this off. See [known limitations](coxph_limitations). |
| `max_iter` | `20` | Maximum Newton–Raphson iterations (R: `iter.max`). |
| `eps` | `1e-9` | Convergence tolerance on the *relative change in log-likelihood* between accepted steps (R: `eps`). |
| `confidence_level` | `0.95` | Level for `confidence_intervals_` and the `summary()` interval columns. |
| `robust` | `False` | Report the cluster-robust (sandwich) covariance instead of the model-based one. Without `cluster=`, every row is its own cluster. With fewer than 30 clusters it warns: the sandwich estimator can then understate standard errors (no small-sample correction is applied, as in R's `survival`). |

`robust` is a *constructor* argument. `fit()` has no `robust` keyword — passing one raises
`TypeError`.

## `fit()`

```text
CoxPH.fit(X, duration=None, event=None, start=None, stop=None,
          strata=None, offset=None, sample_weight=None, cluster=None) -> CoxPH
```

| Argument | Required | Meaning |
|---|---|---|
| `X` | yes | Covariates: `DataFrame`, 2-D array, or 1-D array/`Series` (one column). Must be numeric, finite. A zero-column `DataFrame` is allowed (the second stage of the two-stage SMR/SHR workflow uses one). |
| `duration` | one form | Follow-up time for right-censored data (equivalent to `start=0`). **Mutually exclusive** with `start`/`stop`. |
| `start`, `stop` | other form | Counting-process interval `(start, stop]` per row: left truncation, delayed entry, or time-varying covariates. Both are required together. |
| `event` | yes | Event indicator, coded **0/1 (or boolean)**. R's 1/2 coding is *not* auto-detected and raises `SurvivalDataError`. |
| `strata` | no | Stratum label per row (any sortable dtype). Each stratum gets its own baseline hazard; coefficients are shared. |
| `offset` | no | Known term added to the linear predictor (coefficient fixed at 1). |
| `sample_weight` | no | Non-negative case weights. Zero is allowed (the row stays in the data but has no influence). |
| `cluster` | no | Cluster label per row. **Implies robust variance**, regardless of the constructor's `robust`. |

Rows may be in any order. The full list of input checks, and the exception raised by each, is in
{ref}`survival_ref_data`.

## Fitted attributes

| Attribute | Type / shape | Meaning |
|---|---|---|
| `coef_` | `(p,)` | Estimated log hazard ratios. |
| `standard_errors_` | `(p,)` | Square root of the diagonal of `covariance_` (robust when `robust_` is `True`, otherwise model-based). |
| `covariance_` | `(p, p)` | The covariance that `standard_errors_`, `z_scores_`, `p_values_` and the intervals are built from. |
| `z_scores_`, `p_values_` | `(p,)` | Wald statistics and two-sided normal p-values. |
| `confidence_intervals_` | `(p, 2)` | Wald intervals on the **coefficient** scale (exponentiate for hazard ratios). |
| `naive_covariance_`, `naive_standard_errors_` | `(p, p)`, `(p,)` | Model-based (inverse-information) covariance. Always populated. R: `fit$naive.var`. |
| `robust_covariance_` | `(p, p)` or `None` | Sandwich covariance; `None` unless robust inference was requested. |
| `robust_` | `bool` | Whether `covariance_` is the robust one. |
| `n_clusters_` | `int` | Number of clusters used (`0` when not robust; `n_obs_` when `robust=True` without `cluster=`). |
| `cluster_labels_` | array or `None` | The cluster labels behind `n_clusters_`. |
| `log_likelihood_`, `log_likelihood_null_` | `float` | Partial log-likelihood at `coef_` and at `beta = 0` (offset included). R: `fit$loglik[2]`, `fit$loglik[1]`. |
| `n_iter_`, `converged_`, `convergence_message_` | `int`, `bool`, `str` | Optimiser diagnostics. |
| `n_obs_`, `n_events_` | `int` | Rows, and rows with `event == 1` (a raw count — weights do not change it). |
| `n_features_in_`, `feature_names_in_` | `int`, array | Feature count and names (`"x0", "x1", …` for arrays; `"intercept"` appended if `fit_intercept`). |
| `baseline_hazard_` | `DataFrame` | Baseline **cumulative** hazard, one row per stratum × distinct event time; columns `stratum`, `time`, `hazard`, `survival`. See below. |
| `martingale_residuals_` | `(n_obs_,)` | Martingale residuals, aligned with the input rows (left-truncation- and Efron-correct). |

## Methods

| Method | Returns |
|---|---|
| `predict_linear(X, offset=None)` | `X @ coef_ + offset`, uncentered. Feed it back as `offset=` to build the second stage of a two-stage model. |
| `predict_partial_hazard(X, offset=None)` / `predict(...)` | `exp(predict_linear(...))`. |
| `predict_cumulative_hazard(X, offset=None, stratum=None)` | `DataFrame`, rows = the baseline event times of the chosen stratum (index named `time`), one column per row of `X`. `stratum` is required when the model has more than one stratum. |
| `predict_survival_function(X, offset=None, stratum=None)` | `exp(-cumulative hazard)`, same shape. |
| `score(X, duration=…, event=…, start=…, stop=…, strata=…, offset=…, sample_weight=…)` | The partial log-likelihood of the supplied data at the fitted coefficients (a `float`). |
| `summary()` | `DataFrame` indexed by feature name with columns `coef`, `exp(coef)`, `se(coef)`, `z`, `p`, `lower_<level>%`, `upper_<level>%`. |

`predict_cumulative_hazard` / `predict_survival_function` are evaluated **only at the stratum's
event times** — there is no `times=` argument. To read the step function at arbitrary times, forward-fill:

```python
H = model.predict_cumulative_hazard(X.iloc[:3])
query = np.array([0.0, 0.5, 1.0, 5.0])
H_at_query = H.reindex(H.index.union(query)).ffill().fillna(0.0).loc[query]
```

## Behaviour worth knowing

**Convergence.** Newton–Raphson starts at `beta = 0`; a step that would lower the log-likelihood is
halved (up to 20 times). Iteration stops when the relative log-likelihood change is below `eps`.
If `max_iter` is reached, `fit` **does not raise and does not warn** — it sets
`converged_ = False` and records the reason in `convergence_message_`. Check `converged_` in
production code. If the information matrix is singular the step falls back to a pseudo-inverse and
`convergence_message_` says so (the message is replaced if the iteration limit is also reached).
An unusually large standard error is the usual symptom of collinearity.

**Robust and clustered variance.** `CoxPH(robust=True)` treats each row as a cluster;
`fit(..., cluster=ids)` clusters on `ids` (and implies robust). For counting-process data with several
rows per subject, always pass the subject id as `cluster`. `coef_` never changes; only the covariance does.
With robust inference off, `standard_errors_` equals R's `se(coef)`; with it on, it equals R's `robust se`.
(R reports a robust variance automatically when case weights are non-integer or a cluster term is present;
`CoxPH` does not — it stays model-based unless you ask.)

**`baseline_hazard_`.** It is the cumulative hazard *at `X = 0` and `offset = mean(offset)`* (the weighted
mean when weights are given), exactly matching R's `basehaz(fit, centered = FALSE)`. Without an offset this is simply
the hazard at `X = 0`. `predict_*` methods do **not** use this table; they use the un-shifted hazard at
`X = 0, offset = 0`, so predictions on new data are not biased by the training offsets. It has one row per
distinct *event* time (R's table also repeats values at censoring-only times).

**Ties.** `breslow` (default) and `efron` are implemented, each with a numba-compiled kernel and a
pure-Python fallback used when numba is not importable. Efron uses its own formulas for the baseline hazard
and martingale residuals, matching R's `agsurv5`/`agmart3`.

**Missing data.** Never imputed or dropped: NaN/inf raises `SurvivalDataError`.

## Two-stage SMR / SHR

```python
stage1 = CoxPH().fit(X, start=np.zeros(n), stop=time, event=event, strata=provider)
xbeta = stage1.predict_linear(X)

stage2 = CoxPH().fit(
    pd.DataFrame(index=range(n)),          # no covariates in stage 2
    start=np.zeros(n), stop=time, event=event, offset=xbeta,
)
stage2.baseline_hazard_                    # the "expected" side of the comparison
```

`stage2.coef_` is an empty array and `stage2.summary()` an empty table; the result of interest is the baseline hazard.
The full method is in Chapter 4 of the tutorial.

## R parity cheat-sheet

| R | `pprof_py` |
|---|---|
| `coxph(Surv(t, e) ~ x)` | `CoxPH(ties="efron").fit(X, duration=t, event=e)` (R defaults to Efron) |
| `Surv(start, stop, e)` | `start=`, `stop=`, `event=` |
| `strata(g)`, `offset(o)`, `weights=w` | `strata=g`, `offset=o`, `sample_weight=w` |
| `cluster(id)` | `cluster=id` |
| `summary(fit)$coefficients[, "se(coef)"]` | `naive_standard_errors_` (equals `standard_errors_` when not robust) |
| `robust se` column | `standard_errors_` with `robust=True` / `cluster=` |
| `fit$naive.var` | `naive_covariance_` |
| `fit$loglik` | `[log_likelihood_null_, log_likelihood_]` |
| `basehaz(fit, centered = FALSE)` | `baseline_hazard_` (event times only) |
| `residuals(fit, type = "martingale")` | `martingale_residuals_` |
| `nobs(fit)` | `n_events_` (R's `nobs.coxph` counts events) |
| `extractAIC(fit)` | `pprof_py.selection.aic(model)` |
| `predict(fit, type = "lp")` | `predict_linear` — uncentered. R centres by default; use `reference = "zero"` in R to compare. |

(coxph_limitations)=
## Known limitations

- `fit_intercept=True` fits, and `summary()` shows an `intercept` row, but **every `predict_*` method
  raises `ValueError`** (a matrix-shape mismatch: the intercept column is not appended to new data).
  Leave `fit_intercept=False`.
- `ties="exact"` raises `NotImplementedError`.
- No formula interface and no `na_action`; supply a numeric design matrix.
- No convergence warning (see above).
- Prediction is limited to a stratum's observed event times (see the forward-fill recipe).
