"""Edge cases for input validation -- these don't need R, just checking
that malformed input is rejected with a clear, specific error rather than
failing deep inside the optimizer or silently producing a wrong fit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pprof_py import CoxPH
from pprof_py.data.survival_validation import validate_fit_inputs, SurvivalDataError


def _toy(n=20, p=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    stop = rng.uniform(0.1, 5, size=n)
    event = rng.integers(0, 2, size=n)
    return X, stop, event


def test_requires_duration_or_start_stop():
    X, stop, event = _toy()
    with pytest.raises(ValueError, match="duration"):
        validate_fit_inputs(X, event=event)


def test_rejects_both_duration_and_start_stop():
    X, stop, event = _toy()
    with pytest.raises(ValueError, match="not both"):
        validate_fit_inputs(X, duration=stop, start=np.zeros(len(stop)), stop=stop, event=event)


def test_requires_event():
    X, stop, event = _toy()
    with pytest.raises(ValueError, match="event"):
        validate_fit_inputs(X, duration=stop)


def test_rejects_non_binary_event():
    X, stop, event = _toy()
    bad_event = event.copy()
    bad_event[0] = 2  # R's 1/2 coding is deliberately not auto-detected
    with pytest.raises(SurvivalDataError, match="binary"):
        validate_fit_inputs(X, duration=stop, event=bad_event)


def test_rejects_start_ge_stop():
    X, stop, event = _toy()
    start = stop.copy()  # zero-length intervals throughout
    with pytest.raises(SurvivalDataError, match="start >= stop"):
        validate_fit_inputs(X, start=start, stop=stop, event=event)


def test_rejects_mismatched_length():
    X, stop, event = _toy(n=20)
    with pytest.raises(SurvivalDataError, match="length"):
        validate_fit_inputs(X, duration=stop[:-1], event=event)


def test_rejects_nan_in_X():
    X, stop, event = _toy()
    X[0, 0] = np.nan
    with pytest.raises(SurvivalDataError, match="NaN or infinite"):
        validate_fit_inputs(X, duration=stop, event=event)


def test_rejects_negative_weight():
    X, stop, event = _toy()
    weight = np.ones(len(stop))
    weight[0] = -1.0
    with pytest.raises(SurvivalDataError, match="non-negative"):
        validate_fit_inputs(X, duration=stop, event=event, sample_weight=weight)


def test_allows_zero_weight():
    """A weight of exactly 0 is a legitimate way to drop an observation's
    influence while keeping it in the data -- must NOT raise."""
    X, stop, event = _toy()
    weight = np.ones(len(stop))
    weight[0] = 0.0
    clean = validate_fit_inputs(X, duration=stop, event=event, sample_weight=weight)
    assert clean["weight"][0] == 0.0


def test_predict_before_fit_raises():
    from pprof_py.models.survival.coxph import NotFittedError

    model = CoxPH()
    X, stop, event = _toy()
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_accepts_pandas_dataframe_and_preserves_feature_names():
    X, stop, event = _toy(p=2)
    df = pd.DataFrame(X, columns=["age", "bmi"])
    model = CoxPH().fit(df, duration=stop, event=event)
    assert list(model.feature_names_in_) == ["age", "bmi"]


def test_boolean_event_accepted():
    X, stop, event = _toy()
    model = CoxPH().fit(X, duration=stop, event=event.astype(bool))
    assert model.converged_


def test_empty_design_matrix_offset_only_model_fits():
    """The real Stage-2 SHR/SMR pattern: coxph(Surv(...) ~ offset(x)),
    no covariates at all."""
    X, stop, event = _toy()
    empty_X = pd.DataFrame(index=range(len(stop)))
    offset = np.random.default_rng(0).normal(size=len(stop))
    model = CoxPH().fit(empty_X, duration=stop, event=event, offset=offset)
    assert model.n_features_in_ == 0
    assert np.isfinite(model.log_likelihood_)
