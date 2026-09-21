"""Input validation for CoxPH model fitting.

Mirrors the checks R's coxph()/Surv() perform before fitting, translated
into explicit Python exceptions with actionable messages. Nothing here
silently modifies user data beyond converting to a consistent numpy
dtype -- see `validate_fit_inputs` for exactly what is and is not
permitted.
"""
from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


class SurvivalDataError(ValueError):
    """Structurally invalid survival data: bad intervals, mismatched
    lengths, non-finite values, and similar."""


def _as_1d_float(arr, name: str) -> np.ndarray:
    if isinstance(arr, (pd.Series, pd.Index)):
        arr = arr.to_numpy()
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise SurvivalDataError(f"`{name}` must be 1-dimensional, got shape {arr.shape}")
    try:
        arr = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"`{name}` must be numeric, got dtype {arr.dtype}") from exc
    return arr


def validate_X(X) -> Tuple[np.ndarray, List[str]]:
    """Validate and convert a design matrix. Returns (X_array, feature_names)."""
    feature_names = None
    if isinstance(X, pd.DataFrame):
        feature_names = [str(c) for c in X.columns]
        X_arr = X.to_numpy(dtype=np.float64, copy=True)
    elif isinstance(X, pd.Series):
        feature_names = [str(X.name) if X.name is not None else "x0"]
        X_arr = X.to_numpy(dtype=np.float64, copy=True).reshape(-1, 1)
    else:
        X_arr = np.asarray(X, dtype=np.float64)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.ndim != 2:
        raise SurvivalDataError(f"X must be 2-dimensional, got shape {X_arr.shape}")
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
    if not np.all(np.isfinite(X_arr)):
        raise SurvivalDataError(
            "X contains NaN or infinite values; CoxPH requires complete, finite "
            "covariate data (no implicit imputation is performed)."
        )
    return X_arr, feature_names


def validate_X_predict(
    X,
    feature_names_in: List[str],
    n_features_in: int,
) -> np.ndarray:
    """Validate a design matrix for prediction against the fitted schema.

    For DataFrames: checks that column names match ``feature_names_in``
    exactly (no missing, extra, or duplicate columns).  If the columns
    are the same set but in a different order, the DataFrame is silently
    reordered to match the fit-time order -- this prevents the silent
    wrong-answer bug described in COX_USER_REVIEW Finding 1.

    For plain arrays: checks that the feature count matches
    ``n_features_in``.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features_in)
    """
    if isinstance(X, pd.DataFrame):
        cols = [str(c) for c in X.columns]
        if len(cols) != len(set(cols)):
            raise ValueError(
                "X has duplicate column names; prediction requires unique columns."
            )
        fit_set = set(feature_names_in)
        pred_set = set(cols)
        missing = fit_set - pred_set
        extra = pred_set - fit_set
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            if extra:
                parts.append(f"unexpected: {sorted(extra)}")
            raise ValueError(
                f"X columns do not match the fitted feature names. "
                + "; ".join(parts)
                + f".  Expected: {feature_names_in}"
            )
        # Reorder to match fit-time column order.
        X = X[feature_names_in]
        X_arr = X.to_numpy(dtype=np.float64, copy=True)
    elif isinstance(X, pd.Series):
        X_arr = X.to_numpy(dtype=np.float64, copy=True).reshape(-1, 1)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

    if X_arr.ndim != 2:
        raise SurvivalDataError(f"X must be 2-dimensional, got shape {X_arr.shape}")
    if X_arr.shape[1] != n_features_in:
        raise ValueError(
            f"X has {X_arr.shape[1]} feature(s), but the model was fitted "
            f"with {n_features_in}. Provide the same features used at fit time."
        )
    if not np.all(np.isfinite(X_arr)):
        raise SurvivalDataError(
            "X contains NaN or infinite values; predictions require finite covariates."
        )
    return X_arr


def validate_predict_offset(
    offset,
    n_samples: int,
) -> np.ndarray:
    """Validate an offset array for prediction.

    Ensures the offset is 1-dimensional with the correct length and
    contains only finite values.  Prevents the shape-broadcasting bug
    (a (3,1) offset with 3 patients produces a (3,3) linear predictor
    and 9 survival curves for 3 people).
    """
    if offset is None:
        return np.zeros(n_samples, dtype=np.float64)
    offset_arr = np.asarray(offset, dtype=np.float64)
    if offset_arr.ndim == 0:
        # Scalar offset: broadcast to all samples.
        offset_arr = np.full(n_samples, float(offset_arr), dtype=np.float64)
    if offset_arr.ndim != 1:
        raise ValueError(
            f"offset must be 1-dimensional, got shape {offset_arr.shape}. "
            f"If you have a column vector, use .ravel() or .squeeze()."
        )
    if offset_arr.shape[0] != n_samples:
        raise ValueError(
            f"offset has {offset_arr.shape[0]} element(s) but X has "
            f"{n_samples} row(s); they must match."
        )
    if not np.all(np.isfinite(offset_arr)):
        raise ValueError(
            "offset contains NaN or infinite values; predictions require "
            "finite offsets."
        )
    return offset_arr


def validate_fit_inputs(
    X,
    duration=None,
    event=None,
    start=None,
    stop=None,
    strata=None,
    offset=None,
    sample_weight=None,
) -> dict:
    """Validate and normalize all `fit()` inputs into plain numpy arrays.

    Exactly one of `duration` or (`start`, `stop`) must be given, matching
    R's `Surv(time, event)` vs. `Surv(start, stop, event)` forms.
    `duration` is treated as `stop` with `start` implicitly 0 for every
    row -- ordinary right-censored data is simply the start=0 special
    case of the general counting-process form used internally throughout
    this package (see algorithms/risk_sets.py).
    """
    X_arr, feature_names = validate_X(X)
    n = X_arr.shape[0]

    if duration is not None and (start is not None or stop is not None):
        raise ValueError("Pass either `duration`, or (`start`, `stop`) -- not both.")
    if duration is None and stop is None:
        raise ValueError("Must supply either `duration` or (`start`, `stop`).")

    if duration is not None:
        stop_arr = _as_1d_float(duration, "duration")
        start_arr = np.zeros(n, dtype=np.float64)
    else:
        if start is None:
            raise ValueError(
                "`start` is required when `stop` is given (left-truncated / "
                "counting-process form)."
            )
        start_arr = _as_1d_float(start, "start")
        stop_arr = _as_1d_float(stop, "stop")

    if event is None:
        raise ValueError("`event` is required.")
    event_raw = event.to_numpy() if isinstance(event, (pd.Series, pd.Index)) else np.asarray(event)
    event_arr = event_raw.astype(np.float64)
    finite_vals = event_arr[np.isfinite(event_arr)]
    unique_vals = set(np.unique(finite_vals).tolist())
    if not unique_vals <= {0.0, 1.0}:
        raise SurvivalDataError(
            f"`event` must be binary (0/1 or boolean); got values {sorted(unique_vals)}. "
            "(R's Surv() also accepts 1/2 coding for censored/event -- that convention "
            "is not auto-detected here, to avoid silently misinterpreting an ordinary "
            "0/1 status column. Recode explicitly if you're porting a 1/2-coded R model.)"
        )

    for name, arr in [("start", start_arr), ("stop", stop_arr), ("event", event_arr)]:
        if arr.shape[0] != n:
            raise SurvivalDataError(
                f"`{name}` has length {arr.shape[0]}, expected {n} (must match X's row count)."
            )
        if not np.all(np.isfinite(arr)):
            raise SurvivalDataError(f"`{name}` contains NaN or infinite values.")

    if np.any(start_arr >= stop_arr):
        n_bad = int(np.sum(start_arr >= stop_arr))
        raise SurvivalDataError(
            f"{n_bad} observation(s) have start >= stop. Every interval must satisfy "
            "start < stop (R's Surv(start, stop, event) rejects zero-length and "
            "negative-length intervals the same way)."
        )

    if offset is None:
        offset_arr = np.zeros(n, dtype=np.float64)
    else:
        offset_arr = _as_1d_float(offset, "offset")
        if offset_arr.shape[0] != n:
            raise SurvivalDataError(f"`offset` has length {offset_arr.shape[0]}, expected {n}.")
        if not np.all(np.isfinite(offset_arr)):
            raise SurvivalDataError("`offset` contains NaN or infinite values.")

    if sample_weight is None:
        weight_arr = np.ones(n, dtype=np.float64)
    else:
        weight_arr = _as_1d_float(sample_weight, "sample_weight")
        if weight_arr.shape[0] != n:
            raise SurvivalDataError(
                f"`sample_weight` has length {weight_arr.shape[0]}, expected {n}."
            )
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0):
            raise SurvivalDataError(
                "`sample_weight` must be finite and non-negative. (A weight of exactly "
                "0 is allowed -- it drops that observation's influence on the fit while "
                "keeping it in the data; see docs/R_COMPATIBILITY.md, question 15.)"
            )

    if strata is None:
        strata_codes = np.zeros(n, dtype=np.int64)
        strata_labels = np.array([0])
    else:
        strata_raw = strata.to_numpy() if isinstance(strata, (pd.Series, pd.Index)) else np.asarray(strata)
        if strata_raw.shape[0] != n:
            raise SurvivalDataError(f"`strata` has length {strata_raw.shape[0]}, expected {n}.")
        strata_labels, strata_codes = np.unique(strata_raw, return_inverse=True)

    return dict(
        X=X_arr,
        start=start_arr,
        stop=stop_arr,
        event=event_arr,
        offset=offset_arr,
        weight=weight_arr,
        strata_codes=np.asarray(strata_codes, dtype=np.int64),
        strata_labels=strata_labels,
        feature_names=feature_names,
    )
