(survival_ref_competing)=
# Reference: competing risks and variable selection

Chapters 7 and 9 of the tutorial explain the ideas. This page lists what the classes accept, expose and guarantee.

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
n = 400
X = pd.DataFrame({"age": rng.normal(size=n), "sex": rng.integers(0, 2, n).astype(float), "bmi": rng.normal(size=n)})
latent = rng.exponential(1 / np.exp(0.5 * X["age"] - 0.3 * X["sex"]))
censor = rng.uniform(0.5, 3.0, n)
time = np.round(np.minimum(latent, censor), 3) + 0.001
event = (latent <= censor).astype(int)
cause = np.where(event == 1, rng.choice([1, 2], n), 0)     # 0 = censored, 1 and 2 = causes
```

## `CauseSpecificCoxPH`

One ordinary `CoxPH` per cause: for cause *k*, events of every other cause are treated as censoring.

```python
from pprof_py import CauseSpecificCoxPH

cs = CauseSpecificCoxPH(ties="efron").fit(X, event=cause, duration=time)
cs.causes_            # the causes fitted (default: every distinct non-zero event code)
cs[1].summary()       # the fitted CoxPH for cause 1  (cs.models_ is the underlying dict)
cs.summary()          # all causes stacked, with leading `cause` and `covariate` columns
```

- Constructor: `ties`, `fit_intercept`, `max_iter`, `eps`, `confidence_level` (as `CoxPH`; there is no `robust`).
- `fit(X, event, duration=None, start=None, stop=None, causes=None, strata=None, offset=None, sample_weight=None)`.
  **`event` precedes `duration`.** `event == 0` means censored; any other value is a cause label.
- The wrapper has no `predict_*` methods; call them on `cs[k]`.
- `1 - cs[k].predict_survival_function(...)` is **not** a cumulative incidence function (Chapter 7 shows the bias).

## `FineGrayPH`

Fine–Gray subdistribution-hazard regression for one cause. `fit` expands the data with `finegray_transform`
(inverse-probability-of-censoring weights) and then fits a weighted, **cluster-robust** `CoxPH`, clustering on each
pseudo-observation's original subject.

```python
from pprof_py import FineGrayPH

fg = FineGrayPH(ties="efron").fit(X, event=cause, failcode=1, duration=time, id=np.arange(n))
fg.summary()
cif = 1 - fg.predict_survival_function(X.iloc[:5])     # cumulative incidence of cause 1
```

- Constructor: as `CauseSpecificCoxPH`. `fit(X, event, failcode, duration=None, start=None, stop=None, id=None, strata=None, sample_weight=None)`
  — there is no `offset`. Pass `id` whenever a subject can occupy several rows.
- Attributes: `coef_`, `standard_errors_` (robust), `covariance_`, `z_scores_`, `p_values_`, `confidence_intervals_`,
  `log_likelihood_`, `log_likelihood_null_`, `n_iter_`, `converged_`, `n_obs_`, `n_events_`, `n_features_in_`,
  `feature_names_in_`, `failcode_`, `model_` (the underlying weighted `CoxPH`), `source_row_`, `source_subject_`
  (original row / subject of every pseudo-observation).
- Methods: `predict_linear`, `predict_partial_hazard`, `predict`, `predict_cumulative_hazard`,
  `predict_survival_function` (the subdistribution *survival*; `1 −` it is the CIF), `summary`.

### Validation status — read before using with delayed entry

Checked against R 4.3.3 / `survival` 3.5.8 with **Efron ties on both sides** (`FineGrayPH`'s default is Breslow;
R's is Efron):

| Data | Transform vs `finegray()` | Coefficients vs R | Robust SE vs R |
|---|---|---|---|
| Right-censored (`competing_risks_simple`) | weights agree to 7e-16 | agree to 8 digits | agree to 6 digits |
| Left-truncated (`competing_risks_truncated`) | `fgstart`/`fgstop` agree; **weights differ in 842 of 3,469 rows, up to 0.445** | differ by 3.4e-3 (0.3670 vs 0.3701) | differ by 0.3–0.5% |

With left truncation the IPCW weights do not reproduce R's. Until that is resolved, treat `FineGrayPH` results on
left-truncated data as unvalidated, and compare `finegray_transform(...).weight` with R's `finegray()` on your data
(see {ref}`survival_validation_tools`). Cause-specific models are unaffected: they are ordinary `CoxPH` fits.

## `CoxPHSelector`

Greedy forward, backward or bidirectional selection built on repeated `CoxPH` fits (it never recomputes a likelihood itself).

```python
from pprof_py import CoxPHSelector

sel = CoxPHSelector(direction="forward", criterion="aic").fit(X, duration=time, event=event)
sel.selected_variables_
sel.selection_history_          # step, action, variable, n_variables, variables, aic
sel.final_model_.summary()      # a fitted CoxPH on the selected variables
```

| Parameter | Default | Meaning |
|---|---|---|
| `direction` | `"forward"` | `"forward"`, `"backward"` or `"both"`. |
| `criterion` | `"aic"` | `"aic"`, `"bic"` or `"pvalue"`. |
| `p_enter`, `p_remove` | `0.05`, `0.10` | Used only with `criterion="pvalue"`: add a candidate if its p-value is below `p_enter`; drop a non-forced variable if its p-value exceeds `p_remove`. |
| `ties`, `fit_intercept`, `max_iter`, `eps`, `confidence_level` | as `CoxPH` | Passed to every fit. |
| `max_steps` | `None` | Safety cap; default `4 * len(candidates) + 10`. |

`fit(X, duration=None, start=None, stop=None, event=None, strata=None, offset=None, sample_weight=None, forced=None, candidates=None)`
— `X` must be a `DataFrame`; `forced` variables are always kept; `candidates` restricts the pool. The history's last column is the
criterion (`aic`, `bic`) or `p_value`. Attributes: `selected_variables_`, `selection_history_`, `final_model_`, `n_steps_`.
`summary()` returns the final model's coefficient table.

Criteria, also importable as `pprof_py.selection.aic(model)` / `bic(model)`: `AIC = -2·loglik + 2·p`;
`BIC = -2·loglik + log(n_events)·p` — the penalty uses the number of **events**, as R's `step()` does through `nobs.coxph`.

**Validation status.** The R comparison tests exist, but the datasets they read (`selector_test_data.csv`,
`selector_strata_data.csv`) have no generator anywhere in the repository, so those tests cannot run on a fresh clone.
