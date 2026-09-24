(ll_ref_data)=
# Reference: input validation, `DataPrep`, inter-unit reliability and utilities

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

## Input validation

`validate_and_convert_inputs` is what the fixed-effect models call to turn a `DataFrame` (plus column names) or arrays into NumPy arrays.

```python
from pprof_py.data import validate_and_convert_inputs, check_missingness, check_variation, run_structural_checks

v = validate_and_convert_inputs(df, x_vars=X_COLS, y_var="event", group_var="provider")
v.X.shape, v.y.shape, v.groups.shape, v.covariate_names
```

```text
validate_and_convert_inputs(X, y=None, groups=None, x_vars=None, y_var=None, group_var=None, *,
                            n_var=None, obs_id_var=None, use_dataprep=False, dataprep_options=None) -> ValidatedInputs
```

The returned `ValidatedInputs` has `X`, `y`, `groups`, `N` (trials per row) and `obs_ids` — both `None` unless `n_var` / `obs_id_var` are given — and `covariate_names`. The structural checks in `pprof_py.data`
take a `DataFrame` and column names:

| Function | Behaviour |
|---|---|
| `check_missingness(data, columns)` | `ValueError("Missing values found in the data.")` if any of `columns` has a missing value |
| `check_variation(data, x_columns)` | `ValueError("Covariates with zero variance found.")` |
| `check_correlation(data, x_columns, threshold=0.9)` | logs a warning for pairs above the threshold |
| `check_vif(data, x_columns, threshold=10)` | logs a warning for covariates above the threshold |
| `run_structural_checks(data, y_col, x_cols, group_col, *, threshold_cor=0.9)` | runs the default pipeline; returns `None` |

The logging-only checks accept a keyword-only `_logger`; `tests/test_infrastructure.py` still passes a logger positionally, which is why several of its tests fail.

## `DataPrep`

```python
from pprof_py.data import DataPrep, DataPrepOptions

clean = DataPrep(df, "event", X_COLS, "provider", options=DataPrepOptions(), check=True).data_prep()   # cleaned DataFrame
```

`DataPrep(data, Y_char, X_char, prov_char, options=None, check=True, logging=logging)` runs `check_missingness`, `check_variation`, `check_correlation`, `check_vif`,
`provider_screening`, `filter_small_providers` and `log_no_all_event_providers` in turn; `data_prep()` returns the cleaned frame. `DataPrepOptions` defaults:
`cutoff=10`, `screen_providers=False`, `log_event_providers=False`, `threshold_cor=0.9`, `threshold_vif=10`, `binary_response=False`. `LogisticFixedEffectModel` builds these options from its own constructor arguments (`cutoff`, `screen_providers`, `log_event_providers`, `threshold_cor`, `threshold_vif`).

## Inter-unit reliability (IUR)

**`BootstrapIUR`** — Bootstrap-based Inter-Unit Reliability estimation.

**`SplitHalfIUR`** — Split-half correlation-based IUR estimation.

**`DirectIUR`** — Direct IUR from pre-computed group-level estimates and SEs.

```python
from pprof_py.measures.iur import BootstrapIUR, SplitHalfIUR, DirectIUR
from pprof_py import LogisticFixedEffectModel

obs = df["event"].to_numpy(float)
exp = 1 / (1 + np.exp(-(-1 + 0.3 * df["x1"].to_numpy())))          # expected probability per patient

boot = BootstrapIUR(n_boot=20, seed=1).fit(obs, exp, provider)      # patient-level observed, expected, provider ids
boot.iur_, boot.s2_between_, boot.s2_within_
boot.stratified_iur()
boot.decile_table()

split = SplitHalfIUR(n_iter=5, seed=1).fit(obs, exp, provider)
split.summary()                                                     # one column per correlation / kappa variant

fe = LogisticFixedEffectModel().fit(df, y_var="event", x_vars=X_COLS, group_var="provider")
direct = DirectIUR().fit(fe.group_sizes_, fe.coefficients_["gamma"], np.sqrt(fe.variances_["gamma"]))
direct.iur_
direct.decile_table()
```

| Estimator | `fit(...)` | Attributes | Methods |
|---|---|---|---|
| `BootstrapIUR(n_boot=100, measure_fn=None, seed=123)` | `fit(obs, exp, groups)` | `iur_`, `iur_groups_`, `s2_between_`, `s2_within_`, `n_prime_`, `n_groups_`, `group_labels_`, `group_sizes_`, `measure_`, `measure_bootstrap_` | `stratified_iur(stratify_var=None, stratify_cut=None)`, `decile_table(stratify_var=None, n_quantiles=10)` |
| `SplitHalfIUR(n_iter=10, category_probs=None, measure_fn=None, seed=None)` | `fit(obs, exp, groups)` | `iur_pearson_`, `iur_spearman_`, `iur_kendall_`, `iur_spearman_cat_`, `iur_kendall_cat_`, `iur_kappa_`, `iur_all_`, `n_groups_` | `summary()` |
| `DirectIUR()` | `fit(sizes, estimates, standard_errors)` | `iur_`, `s2_between_`, `s2_within_`, `n_prime_`, `n_groups_`, `sizes_` | `decile_table(n_quantiles=10)` (columns `min`, `decile 1`…`decile 10`, `max`) |

`ratio_measure(obs, exp, groups)` is the default `measure_fn` (observed-to-expected ratio per group); pass your own function with the same signature to reliability-test another measure.

## Utilities

```python
from pprof_py import proc_freq, setup_logger, sigmoid

proc_freq(df, ["provider"])                        # logs frequency, percent and cumulative percent; returns None
sigmoid(np.array([-2.0, 0.0, 2.0]))
```

```text
setup_logger(name: str, level=20, log_to_file: bool = False, log_dir: str = None, time_zone: str = 'US/Eastern') -> logging.Logger
proc_freq(df: pandas.DataFrame, columns: list)
sigmoid(x: 'np.ndarray') -> 'np.ndarray'
```

The empirical-null tools used by every provider test live in `pprof_py.inference` ({ref}`empirical-null-guide`); the survival module keeps
R-compatible wrappers over them ({ref}`survival_ref_inference`).
