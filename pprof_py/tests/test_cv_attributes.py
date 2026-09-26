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
