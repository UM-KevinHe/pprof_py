"""Every CV class exposes the same attribute surface (C4)."""
import warnings

import numpy as np
import pytest

from pprof_py import (DiscreteSurvivalCV, GroupLassoCoxPHCV, GroupLassoLogisticCV, PenalizedCoxPHCV, PenalizedLinearCV,
                      PenalizedLogisticCV, ProviderPenalizedDiscreteSurvivalCV, ProviderPenalizedLogisticCV)

SURFACE = ("lambda_path_", "cv_mean_deviance_", "cv_se_deviance_", "lambda_min_", "lambda_1se_", "lambda_", "model_", "coef_")


def _data():
    rng = np.random.default_rng(12)
    n, p = 300, 4
    X = rng.normal(size=(n, p))
    eta = X @ np.array([0.8, -0.5, 0.0, 0.3])
    prov = rng.integers(0, 6, n)
    t = rng.exponential(np.exp(-eta))
    c = rng.exponential(2.0, n)
    return dict(X=X, yb=rng.binomial(1, 1 / (1 + np.exp(-eta))), yl=eta + rng.normal(size=n), prov=prov,
                dur=np.minimum(t, c), ev=(t <= c).astype(int), tdisc=np.clip(np.ceil(np.minimum(t, c) * 3), 1, 4).astype(int))


CASES = {
    "PenalizedLogisticCV": lambda d: PenalizedLogisticCV(n_lambda=6, n_folds=3, random_state=0).fit(d["X"], d["yb"]),
    "PenalizedLinearCV": lambda d: PenalizedLinearCV(n_lambda=6, n_folds=3, random_state=0).fit(d["X"], d["yl"]),
    "GroupLassoLogisticCV": lambda d: GroupLassoLogisticCV(groups=[1, 1, 2, 2], n_lambda=6, n_folds=3, random_state=0).fit(d["X"], d["yb"]),
    "ProviderPenalizedLogisticCV": lambda d: ProviderPenalizedLogisticCV(n_lambda=6, n_folds=3, random_state=0).fit(d["X"], d["yb"], d["prov"]),
    "PenalizedCoxPHCV": lambda d: PenalizedCoxPHCV(n_lambda=6, n_folds=3, random_state=0).fit(d["X"], duration=d["dur"], event=d["ev"]),
    "GroupLassoCoxPHCV": lambda d: GroupLassoCoxPHCV(groups=[1, 1, 2, 2], n_lambda=6, n_folds=3, random_state=0).fit(d["X"], duration=d["dur"], event=d["ev"]),
    "DiscreteSurvivalCV": lambda d: DiscreteSurvivalCV(n_lambda=6, n_folds=3, random_state=0).fit(d["X"], d["tdisc"], d["ev"]),
    "ProviderPenalizedDiscreteSurvivalCV": lambda d: ProviderPenalizedDiscreteSurvivalCV(n_lambda=6, n_folds=3, random_state=0).fit(d["X"], d["tdisc"], d["ev"], d["prov"]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_cv_class_has_the_same_attributes(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = CASES[name](_data())
    missing = [a for a in SURFACE if getattr(cv, a, None) is None]
    assert not missing, f"{name} lacks {missing}"
    assert len(cv.cv_mean_deviance_) == len(cv.cv_se_deviance_) == len(cv.lambda_path_)
    assert cv.lambda_ == (cv.lambda_1se_ if cv.se_rule == "1se" else cv.lambda_min_)
    assert cv.lambda_1se_ >= cv.lambda_min_
    assert not any(hasattr(cv, old) for old in ("final_estimator_", "best_model_", "cv_mean_", "cv_se_"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_model_is_the_full_data_path(name):
    """C22: every CV class keeps the full-data path in ``model_``; ``coef_`` is its point at ``lambda_``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = CASES[name](_data())
    path = np.asarray(cv.lambda_path_)
    assert np.array_equal(np.asarray(cv.model_.lambda_path_), path)
    if hasattr(cv.model_, "coef_at"):
        at_selected = cv.model_.coef_at(cv.lambda_)
    else:                                          # the provider path classes index the path instead
        at_selected = np.asarray(cv.model_.coef_path_)[int(np.argmin(np.abs(np.log(path) - np.log(cv.lambda_))))]
    np.testing.assert_allclose(np.asarray(cv.coef_).ravel(), np.asarray(at_selected).ravel(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", ["PenalizedCoxPHCV", "GroupLassoCoxPHCV"])
def test_cox_cv_model_reads_either_selected_lambda(name):
    """C22: under se_rule='1se', model_.coef_at(lambda_min_) is the se_rule='min' selection, and predictions use coef_."""
    d = _data()
    kw = dict(n_lambda=8, n_folds=3, random_state=0)
    cls = PenalizedCoxPHCV if name == "PenalizedCoxPHCV" else GroupLassoCoxPHCV
    extra = {} if name == "PenalizedCoxPHCV" else dict(groups=[1, 1, 2, 2])
    one_se = cls(se_rule="1se", **extra, **kw).fit(d["X"], duration=d["dur"], event=d["ev"])
    at_min = cls(se_rule="min", **extra, **kw).fit(d["X"], duration=d["dur"], event=d["ev"])
    assert np.array_equal(one_se.model_.coef_at(one_se.lambda_min_), at_min.coef_)
    np.testing.assert_allclose(one_se.predict_linear(d["X"][:20]), d["X"][:20] @ one_se.coef_, rtol=0, atol=1e-12)
