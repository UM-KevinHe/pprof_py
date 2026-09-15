"""Tests for `models/competing_risks.py` (CauseSpecificCoxPH, FineGrayPH).

`algorithms/competing_risks.py`'s own test file already validates the
Fine-Gray *data transform* against R's exact worked examples; these tests
instead check that the estimator wrappers correctly reuse `CoxPH` --
that `FineGrayPH.fit(...)` gives bit-identical results to manually
calling `finegray_transform` + `CoxPH.fit(...)`, and that cause-specific
fitting is exactly what plain per-cause `CoxPH` calls would give.
"""
import numpy as np
import pandas as pd
import pytest

from coxph.models.coxph import CoxPH
from coxph.models.competing_risks import CauseSpecificCoxPH, FineGrayPH
from coxph.algorithms.competing_risks import finegray_transform


def _synthetic_competing_risks(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    risk1 = np.exp(0.5 * X[:, 0])
    risk2 = np.exp(-0.3 * X[:, 1])
    t1 = rng.exponential(1 / (0.3 * risk1))
    t2 = rng.exponential(1 / (0.2 * risk2))
    censor = rng.exponential(3, n)
    stop = np.minimum(np.minimum(t1, t2), censor)
    event = np.where(stop == t1, 1.0, np.where(stop == t2, 2.0, 0.0))
    return X, stop, event


def test_cause_specific_matches_manual_recoding():
    X, stop, event = _synthetic_competing_risks()
    csc = CauseSpecificCoxPH(ties="efron").fit(X, event=event, duration=stop)

    for cause in (1.0, 2.0):
        manual = CoxPH(ties="efron").fit(X, duration=stop, event=(event == cause).astype(float))
        np.testing.assert_allclose(csc[cause].coef_, manual.coef_)
        np.testing.assert_allclose(csc[cause].standard_errors_, manual.standard_errors_)


def test_cause_specific_auto_detects_causes():
    X, stop, event = _synthetic_competing_risks()
    csc = CauseSpecificCoxPH().fit(X, event=event, duration=stop)
    assert sorted(csc.causes_.tolist()) == [1.0, 2.0]


def test_cause_specific_rejects_unobserved_cause():
    X, stop, event = _synthetic_competing_risks()
    with pytest.raises(ValueError):
        CauseSpecificCoxPH().fit(X, event=event, duration=stop, causes=[1, 99])


def test_cause_specific_summary_has_one_block_per_cause():
    X, stop, event = _synthetic_competing_risks()
    csc = CauseSpecificCoxPH().fit(X, event=event, duration=stop)
    summary = csc.summary()
    assert sorted(summary["cause"].unique().tolist()) == [1.0, 2.0]
    assert len(summary) == 2 * X.shape[1]


def test_finegray_matches_manual_transform_plus_coxph():
    X, stop, event = _synthetic_competing_risks()
    fg = FineGrayPH(ties="efron").fit(X, event=event, failcode=1.0, duration=stop)

    transformed = finegray_transform(np.zeros_like(stop), stop, event, failcode=1.0)
    manual = CoxPH(ties="efron").fit(
        X[transformed.row], start=transformed.start, stop=transformed.stop,
        event=transformed.status, sample_weight=transformed.weight, cluster=transformed.row,
    )
    np.testing.assert_allclose(fg.coef_, manual.coef_)
    np.testing.assert_allclose(fg.standard_errors_, manual.standard_errors_)
    np.testing.assert_array_equal(fg.source_row_, transformed.row)
    # this is the whole point of clustering: robust SE should differ from naive
    assert not np.allclose(fg.standard_errors_, fg.model_.naive_covariance_.diagonal() ** 0.5)


def test_finegray_preserves_dataframe_column_names():
    X, stop, event = _synthetic_competing_risks()
    df = pd.DataFrame(X, columns=["age", "sex"])
    fg = FineGrayPH().fit(df, event=event, failcode=1.0, duration=stop)
    assert list(fg.feature_names_in_) == ["age", "sex"]
    assert list(fg.summary().index) == ["age", "sex"]
    assert list(fg.model_.summary().index) == ["age", "sex"]


def test_finegray_accepts_plain_ndarray():
    X, stop, event = _synthetic_competing_risks()
    fg = FineGrayPH().fit(X, event=event, failcode=1.0, duration=stop)
    assert fg.coef_.shape == (2,)


def test_finegray_n_obs_is_original_subject_count_not_expanded_rows():
    X, stop, event = _synthetic_competing_risks(n=150)
    fg = FineGrayPH().fit(X, event=event, failcode=1.0, duration=stop)
    assert fg.n_obs_ == 150
    assert fg.model_.n_obs_ > 150  # pseudo-observations strictly outnumber subjects whenever anyone has a competing event


def test_finegray_duration_and_start_stop_both_given_raises():
    X, stop, event = _synthetic_competing_risks(n=50)
    start = np.zeros_like(stop)
    with pytest.raises(ValueError):
        FineGrayPH().fit(X, event=event, failcode=1.0, duration=stop, start=start, stop=stop)


def test_finegray_predict_survival_function_is_a_valid_curve():
    X, stop, event = _synthetic_competing_risks(n=150)
    fg = FineGrayPH().fit(X, event=event, failcode=1.0, duration=stop)
    curve = fg.predict_survival_function(X[:5])
    assert curve.shape[1] == 5
    assert np.all(curve.to_numpy() >= -1e-9) and np.all(curve.to_numpy() <= 1.0 + 1e-9)
    # a cumulative-incidence curve (1 - this) must be non-decreasing in time
    cif = 1.0 - curve.to_numpy()
    assert np.all(np.diff(cif, axis=0) >= -1e-9)


def test_finegray_strata_and_sample_weight_passthrough():
    X, stop, event = _synthetic_competing_risks(n=200, seed=2)
    rng = np.random.default_rng(3)
    strata = rng.choice(["a", "b"], size=200)
    sw = rng.uniform(0.5, 1.5, size=200)
    fg = FineGrayPH().fit(X, event=event, failcode=1.0, duration=stop, strata=strata, sample_weight=sw)

    transformed = finegray_transform(np.zeros_like(stop), stop, event, failcode=1.0, strata=strata, sample_weight=sw)
    manual = CoxPH().fit(
        X[transformed.row], start=transformed.start, stop=transformed.stop,
        event=transformed.status, strata=strata[transformed.row], sample_weight=transformed.weight,
        cluster=transformed.row,
    )
    np.testing.assert_allclose(fg.coef_, manual.coef_)
