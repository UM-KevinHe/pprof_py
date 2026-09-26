"""ProviderPenalizedLogistic.test(): the fixed-effect count test at a point of the path (C1)."""
import warnings

import numpy as np
import pytest
from scipy.special import expit

from pprof_py import ProviderPenalizedLogistic, ProviderPenalizedLogisticCV
from pprof_py.inference.effect_tests import EXACT_P_FLOOR, poibin_tails, z_from_tails


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    n = 600
    X = rng.normal(size=(n, 3))
    prov = rng.integers(0, 12, n)
    y = rng.binomial(1, expit(-1 + X @ [0.6, -0.4, 0.0] + rng.normal(0, 0.6, 12)[prov]))
    return X, y, prov


@pytest.fixture(scope="module")
def model(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ProviderPenalizedLogistic(n_lambda=8).fit(*data)


def test_z_is_the_poisson_binomial_test_at_the_path_point(data, model):
    X, y, prov = data
    which = 4
    res = model.test(which=which)
    g0 = res["null_value"].iloc[0]
    fixed = X @ model.coef_path_[which] + model.intercept_path_[which]
    j = 3
    rows = prov == model.provider_labels_[j]
    t = poibin_tails(y[rows].sum(), expit(g0 + fixed[rows]))
    z = z_from_tails(np.array([t[0]]), np.array([t[1]]), "two_sided", EXACT_P_FLOOR)[0]
    assert res["z_raw"].iloc[j] == z
    assert res["estimate"].iloc[j] == model.gamma_path_[which][j]


def test_limits_are_dual_to_the_flags(model):
    res = model.test()
    g0 = res["null_value"].iloc[0]
    assert ((res["flag"] != 0) == ((res["ci_lower"] > g0) | (res["ci_upper"] < g0))).all()
    assert model.test(lambda_value=model.lambda_path_[2]).equals(model.test(which=2))


def test_cv_tests_at_the_selected_lambda(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = ProviderPenalizedLogisticCV(n_lambda=8, n_folds=3, random_state=0).fit(*data)
    idx = cv.lambda_1se_idx_ if cv.se_rule == "1se" else cv.lambda_min_idx_
    assert cv.test().equals(cv.model_.test(which=idx))


def test_rejects_other_methods_and_weighted_fits(data, model):
    with pytest.raises(ValueError, match="poibin_exact"):
        model.test(test_method="wald")
    X, y, prov = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = ProviderPenalizedLogistic(n_lambda=4).fit(X, y, prov, sample_weight=np.full(len(y), 2.0))
    with pytest.raises(ValueError, match="unweighted"):
        weighted.test()
