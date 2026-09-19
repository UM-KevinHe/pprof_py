"""Input validation for grouped/provider models.

Parallel to ``survival_validation.py`` (CoxPH), this module provides:

1. **Structural-check free functions** -- generic data-quality checks
   usable by any model family:

   - `check_missingness` -- raises on NaN values
   - `check_variation` -- raises on zero-variance covariates
   - `check_correlation` -- warns on highly correlated covariate pairs
   - `check_vif` -- warns on high variance inflation (opt-in, not in
     the default pipeline)
   - `run_structural_checks` -- orchestrator calling the first three

2. **Input conversion** -- `validate_and_convert_inputs` converts
   DataFrame or array inputs into ``ValidatedInputs`` (a plain-NumPy
   container analogous to ``SurvivalData``), optionally applying
   structural checks and provider screening via ``DataPrep``.

Originally part of ``DataPrep`` (``data/preparation.py``) and
``models/base.py``; extracted here as the canonical location.
``models/__init__.py`` re-exports the symbols for backward
compatibility.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import check_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural checks -- free functions
# ---------------------------------------------------------------------------

def check_missingness(
    data: pd.DataFrame,
    columns: List[str],
    *,
    _logger: logging.Logger = None,
) -> None:
    """Raise ``ValueError`` if any *columns* contain missing values.

    Parameters
    ----------
    data : pd.DataFrame
    columns : list of str
        Column names to check (typically ``x_cols + [y_col, group_col]``).
    _logger : logging.Logger, optional
        Logger instance; defaults to the module-level logger.
    """
    log = _logger or logger
    log.info("Checking missingness of variables ...")
    missing = data[columns].isnull().sum()
    if missing.any():
        bad = ", ".join(missing.index[missing > 0])
        log.error(f"Missing values found in columns: {bad}")
        raise ValueError("Missing values found in the data.")
    log.info("Missing values NOT found. Checking missingness completed!")


def check_variation(
    data: pd.DataFrame,
    x_columns: List[str],
    *,
    _logger: logging.Logger = None,
) -> None:
    """Raise ``ValueError`` if any covariate has zero variance.

    Parameters
    ----------
    data : pd.DataFrame
    x_columns : list of str
        Covariate column names.
    """
    log = _logger or logger
    log.info("Checking variation in covariates ...")
    selector = VarianceThreshold()
    selector.fit(data[x_columns])
    zero_var = [
        col for col, var in zip(x_columns, selector.variances_) if var == 0
    ]
    if zero_var:
        log.error(f"Covariates with zero variance: {', '.join(zero_var)}")
        raise ValueError("Covariates with zero variance found.")
    log.info("Checking variation in covariates completed!")


def check_correlation(
    data: pd.DataFrame,
    x_columns: List[str],
    threshold: float = 0.9,
    *,
    _logger: logging.Logger = None,
) -> None:
    """Log a warning if any covariate pair exceeds *threshold* correlation.

    Parameters
    ----------
    data : pd.DataFrame
    x_columns : list of str
    threshold : float, default 0.9
    """
    log = _logger or logger
    log.info("Checking pairwise correlation among covariates ...")
    corr = data[x_columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high = [
        (c1, c2)
        for c1 in upper.columns
        for c2 in upper.index
        if upper[c1][c2] > threshold
    ]
    if high:
        log.warning(f"Highly correlated covariates found: {high}")
    else:
        log.info("No highly correlated covariates found.")


def _variance_inflation_factors(X: np.ndarray) -> np.ndarray:
    """Compute VIF for each column of *X* without statsmodels.

    Uses the correlation-matrix-inverse method:
    ``VIF_j = diag(inv(corr(X)))_j``.

    Falls back to a column-wise OLS approach when the correlation matrix
    is singular (perfect collinearity).
    """
    corr = np.corrcoef(X, rowvar=False)
    try:
        inv_corr = np.linalg.inv(corr)
        return np.diag(inv_corr)
    except np.linalg.LinAlgError:
        # Singular correlation matrix -- fall back to R^2 per column
        n, p = X.shape
        vifs = np.empty(p)
        for j in range(p):
            mask = np.ones(p, dtype=bool)
            mask[j] = False
            X_other = np.column_stack([np.ones(n), X[:, mask]])
            y_j = X[:, j]
            beta, *_ = np.linalg.lstsq(X_other, y_j, rcond=None)
            ss_res = np.sum((y_j - X_other @ beta) ** 2)
            ss_tot = np.sum((y_j - y_j.mean()) ** 2)
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            vifs[j] = 1.0 / (1.0 - r_sq) if r_sq < 1.0 else np.inf
        return vifs


def check_vif(
    data: pd.DataFrame,
    x_columns: List[str],
    threshold: int = 10,
    *,
    _logger: logging.Logger = None,
) -> None:
    """Log a warning if any covariate's VIF exceeds *threshold*.

    This check is **opt-in** -- :func:`run_structural_checks` does not
    call it by default.  Invoke it explicitly when VIF diagnostics are
    desired.

    Parameters
    ----------
    data : pd.DataFrame
    x_columns : list of str
    threshold : int, default 10
    """
    log = _logger or logger
    log.info("Checking VIF of covariates ...")
    vif_values = _variance_inflation_factors(data[x_columns].values)
    vif_data = pd.DataFrame({
        "variables": x_columns,
        "VIF": vif_values,
    })
    high_vif = vif_data[vif_data["VIF"] > threshold]
    if not high_vif.empty:
        log.warning(
            f"High VIF found in variables: {', '.join(high_vif['variables'])}"
        )
    else:
        log.info("No high VIF found.")


def run_structural_checks(
    data: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    group_col: str,
    *,
    threshold_cor: float = 0.9,
    _logger: logging.Logger = None,
) -> None:
    """Run the default structural-check pipeline.

    Calls :func:`check_missingness`, :func:`check_variation`, and
    :func:`check_correlation`.  VIF is deliberately excluded -- call
    :func:`check_vif` explicitly when needed.

    Parameters
    ----------
    data : pd.DataFrame
    y_col, group_col : str
    x_cols : list of str
    threshold_cor : float, default 0.9
    """
    columns = list(x_cols) + [y_col, group_col]
    check_missingness(data, columns, _logger=_logger)
    check_variation(data, x_cols, _logger=_logger)
    check_correlation(data, x_cols, threshold=threshold_cor, _logger=_logger)


# ---------------------------------------------------------------------------
# Validated-input container
# ---------------------------------------------------------------------------

@dataclass
class ValidatedInputs:
    """Result container from :func:`validate_and_convert_inputs`.

    Callers unpack whichever fields they need and store them on ``self``
    explicitly.  Parallel to
    :class:`~pprof_py.data.survival_data.SurvivalData` for CoxPH.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    covariate_names: List[str]
    obs_ids: Optional[np.ndarray] = None
    N: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Main entry point -- parallel to survival_validation.validate_fit_inputs
# ---------------------------------------------------------------------------

def validate_and_convert_inputs(
    X,
    y=None,
    groups=None,
    x_vars=None,
    y_var=None,
    group_var=None,
    *,
    n_var: Optional[str] = None,
    obs_id_var: Optional[str] = None,
    use_dataprep: bool = False,
    dataprep_options=None,
) -> ValidatedInputs:
    """Validate and convert input data to NumPy arrays for modeling.

    Stateless, free-function entry point for grouped/provider models,
    parallel to :func:`~pprof_py.data.survival_validation.validate_fit_inputs`
    for CoxPH.  Supports both DataFrames and array-like inputs; optionally
    applies structural checks and provider screening via
    :class:`~pprof_py.data.preparation.DataPrep`.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        Design matrix (covariates) or complete dataset.
    y, groups : array-like, optional
        Response and group identifiers (required when *X* is array-like).
    x_vars, y_var, group_var : str or list of str, optional
        Column names (required when *X* is a DataFrame).
    n_var : str, optional
        Column for binomial N (trials per observation).
    obs_id_var : str, optional
        Column for observation-level identifiers.
    use_dataprep : bool, default False
        Apply structural checks and provider screening.
    dataprep_options : DataPrepOptions, optional
        Screening/threshold knobs.  When *use_dataprep* is True and this
        is ``None``, default ``DataPrepOptions()`` is used.  Replaces the
        former flat keyword arguments (``cutoff``, ``screen_providers``,
        ``log_event_providers``, ``threshold_cor``, ``threshold_vif``).

    Returns
    -------
    ValidatedInputs
    """
    obs_ids = None
    N_ = None

    # -- Handle DataFrame inputs -----------------------------------------
    if isinstance(X, pd.DataFrame):
        if x_vars is None or group_var is None:
            raise ValueError(
                "When providing a DataFrame, `x_vars` and `group_var` "
                "must be specified."
            )
        covariate_names = list(x_vars)
        data_for_prep = X.copy()
        X_array = data_for_prep[x_vars].to_numpy()
        y_array = (
            data_for_prep[y_var].to_numpy() if y_var else np.zeros(X.shape[0])
        )
        groups_array = data_for_prep[group_var].to_numpy()
        if obs_id_var is not None and obs_id_var in data_for_prep.columns:
            obs_ids = data_for_prep[obs_id_var].to_numpy()
        if n_var is not None and n_var in data_for_prep.columns:
            N_ = data_for_prep[n_var].to_numpy().astype(float)

    # -- Handle array-like inputs ----------------------------------------
    else:
        X_array = check_array(X, ensure_2d=True, dtype=np.float64)
        groups_array = (
            check_array(groups, ensure_2d=False)
            if groups is not None
            else np.zeros(X_array.shape[0])
        )
        y_array = (
            check_array(y, ensure_2d=False, dtype=np.float64)
            if y is not None
            else np.zeros(X_array.shape[0])
        )
        covariate_names = [f"X{i}" for i in range(X_array.shape[1])]
        data_for_prep = pd.DataFrame(X_array, columns=covariate_names)
        data_for_prep["y"] = y_array
        data_for_prep["groups"] = groups_array

    # -- Dimension checks ------------------------------------------------
    if len(groups_array) != len(X_array):
        raise ValueError("X and groups must have the same number of samples.")
    if len(y_array) != 0 and len(y_array) != len(X_array):
        raise ValueError(
            "y must have the same number of samples as X if provided."
        )

    # -- DataPrep (structural checks + provider screening) ---------------
    if use_dataprep:
        # Lazy import to avoid circular dependency:
        # preparation.py imports check functions from this module;
        # this module imports DataPrep only at call time.
        from .preparation import DataPrep, DataPrepOptions

        if dataprep_options is None:
            dataprep_options = DataPrepOptions()

        if isinstance(X, pd.DataFrame):
            Y_char = y_var if y_var else "y"
            X_char = x_vars
            prov_char = group_var
        else:
            Y_char = "y"
            X_char = covariate_names
            prov_char = "groups"

        dataprep = DataPrep(
            data=data_for_prep,
            Y_char=Y_char,
            X_char=X_char,
            prov_char=prov_char,
            check=True,
            options=dataprep_options,
        )
        prepared_data = dataprep.data_prep()

        X_array = prepared_data[X_char].to_numpy()
        y_array = prepared_data[Y_char].to_numpy()
        groups_array = prepared_data[prov_char].to_numpy()
        if obs_id_var is not None and obs_id_var in prepared_data.columns:
            obs_ids = prepared_data[obs_id_var].to_numpy()

    return ValidatedInputs(
        X=X_array,
        y=y_array,
        groups=groups_array,
        covariate_names=covariate_names,
        obs_ids=obs_ids,
        N=N_,
    )
