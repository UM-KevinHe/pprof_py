"""Covariate tests against R pprofR f23c2889.

``summary(variance_type="model")`` and ``summary(variance_type="robust")`` of LogisticFixedEffectModel against
``summary_fe_covar(test="wald")`` and ``summary_fe_covar(test="robust_wald")`` on the reference cohorts of
``test_pprofr_clustered_reference.py``. The robust form is the patient-clustered sandwich; s1_spanning has patients
treated at more than one provider.
"""
import warnings

import pytest

from pprof_py import LogisticFixedEffectModel
from pprof_py.tests.logistic.test_pprofr_clustered_reference import (COHORT_NAMES, COHORTS, COVARIATES, INPUTS, _read,
                                                                     excess)


@pytest.fixture(scope="module")
def fits():
    inputs, meta = _read(INPUTS), _read(COHORTS).set_index("cohort")
    out = {}
    for name in COHORT_NAMES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[name] = LogisticFixedEffectModel().fit(
                inputs[inputs["cohort"] == name], y_var="Y", x_vars=["x1", "x2", "x3"], provider_var="provider",
                n_var="N" if bool(meta.loc[name, "binomial"]) else None, obs_id_var="patient")
    return out


def _r(name, test):
    table = _read(COVARIATES)
    return table[(table["cohort"] == name) & (table["test"] == test)].set_index("covariate")


@pytest.mark.parametrize("variance, test", [("model", "wald"), ("robust", "robust_wald")])
@pytest.mark.parametrize("name", COHORT_NAMES)
def test_covariate_tests_match_pprofr(fits, name, variance, test):
    R = _r(name, test)
    res = fits[name].summary(variance_type=variance).reindex(R.index)
    for mine, theirs in (("estimate", "beta"), ("std_error", "se"), ("stat", "stat"), ("p_value", "p")):
        assert excess(res[mine], R[theirs]) <= 1, mine


def test_the_model_based_errors_miss_the_robust_ones(fits):
    """Negative control: the comparison tells the two variances apart."""
    R = _r("s1_spanning", "robust_wald")
    res = fits["s1_spanning"].summary(variance_type="model").reindex(R.index)
    assert excess(res["std_error"], R["se"]) > 1e3
