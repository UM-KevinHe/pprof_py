"""C19 and C20: the columns a group-penalty null point fits, and DiscreteSurvival's lasso-only penalty."""
import numpy as np
import pytest

from pprof_py import (DiscreteSurvival, DiscreteSurvivalCV, GroupLassoCoxPH, GroupLassoLinear, GroupLassoLogistic,
                      ProviderPenalizedCoxPH, ProviderPenalizedLogistic)
from pprof_py.algorithms.coordinate_descent import compute_group_lambda_max
from pprof_py.algorithms.penalty import unpenalized_columns

GROUPS = np.array([1, 1, 1, 2, 2, 0, 3, 3])
PF_ZERO_IN_GROUP = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])     # column 1 sits in penalized group 1


def _data(seed=11, n=500, K=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8)) + 0.5 * rng.normal(size=(n, 1))
    beta = np.array([0.7, -0.5, 0.4, 0.0, 0.0, 0.3, 0.25, -0.2])
    prov = np.sort(rng.integers(0, K, size=n))
    eta = X @ beta + rng.normal(scale=0.4, size=K)[prov]
    y = rng.binomial(1, 1 / (1 + np.exp(-(eta - 0.3)))).astype(float)
    t = rng.exponential(scale=np.exp(-eta))
    c = rng.exponential(scale=np.quantile(t, 0.75), size=n)
    return X, y, X @ beta + rng.normal(size=n), np.minimum(t, c) + 1e-6, (t <= c).astype(float), prov


def test_unpenalized_columns_rule():
    groups = np.array([0, 1, 1, 2, 2])
    pf = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    gw = np.array([np.sqrt(2.0), 0.0])
    # alpha < 1: a zero penalty factor inside a penalized group leaves the group term; group 2 has no group term, so
    # its column with pf = 0 carries no penalty at all.
    assert unpenalized_columns(groups, gw, pf, 0.5).tolist() == [True, False, False, True, False]
    assert unpenalized_columns(groups, gw, pf, 0.0).tolist() == [True, False, False, True, True]
    assert unpenalized_columns(groups, gw, pf, 1.0).tolist() == [True, True, False, True, False]


@pytest.mark.parametrize("name", ["GroupLassoLogistic", "GroupLassoLinear", "ProviderPenalizedLogistic",
                                  "GroupLassoCoxPH", "ProviderPenalizedCoxPH"])
def test_penalty_factor_is_inert_under_the_pure_group_lasso(name):
    """With alpha = 0 the penalty has no L1 term, so penalty factors change nothing, including the null point."""
    X, y, yl, t, e, prov = _data()
    fits = []
    for pf in (np.ones(8), PF_ZERO_IN_GROUP):
        kw = dict(groups=GROUPS, penalty_factor=pf, n_lambda=8, lambda_min_ratio=0.05)
        if name == "GroupLassoLogistic":
            fits.append(GroupLassoLogistic(**kw).fit(X, y))
        elif name == "GroupLassoLinear":
            fits.append(GroupLassoLinear(**kw).fit(X, yl))
        elif name == "ProviderPenalizedLogistic":
            fits.append(ProviderPenalizedLogistic(penalty_type="group_lasso", **kw).fit(X, y, provider_id=prov))
        elif name == "GroupLassoCoxPH":
            fits.append(GroupLassoCoxPH(**kw).fit(X, duration=t, event=e))
        else:
            fits.append(ProviderPenalizedCoxPH(penalty_type="group_lasso", **kw).fit(X, duration=t, event=e,
                                                                                   provider_id=prov))
    a, b = fits
    assert a.lambda_max_ == b.lambda_max_
    assert np.array_equal(a.coef_path_, b.coef_path_)
    assert b.coef_path_[0][1] == 0.0                      # the pf = 0 column is not fitted at the null point


def test_lambda_max_with_a_zero_multiplier_group():
    score = np.array([0.3, -0.2, 0.5, 0.1])
    groups = np.array([1, 1, 2, 2])
    pf = np.ones(4)
    # Group 2 has no group term, so it is zero iff |g_j| <= lam * alpha * pf_j: lambda_max = 0.5 / 0.4.
    assert compute_group_lambda_max(score, 1.0, groups, np.array([10.0, 0.0]), pf, 0.4) == pytest.approx(0.5 / 0.4)
    X, y, _yl, _t, _e, _prov = _data()
    m = GroupLassoLogistic(groups=np.array([1, 1, 1, 2, 2, 3, 3, 3]), alpha=0.4, group_multiplier=np.array([1.0, 0.0, 1.0]),
                           n_lambda=6, lambda_min_ratio=0.1).fit(X, y)
    assert np.all(np.abs(m.coef_path_[0]) < 1e-12)
    assert np.any(np.abs(m.coef_path_[1]) > 0)


@pytest.mark.parametrize("penalty_type", ["group_lasso", "sparse_group_lasso"])
def test_discrete_survival_fits_the_lasso_only(penalty_type):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    time = rng.integers(1, 5, size=200)
    event = rng.binomial(1, 0.7, size=200)
    with pytest.raises(ValueError, match="lasso only"):
        DiscreteSurvival(penalty_type=penalty_type).fit(X, time, event)
    with pytest.raises(ValueError, match="lasso only"):
        DiscreteSurvivalCV(n_folds=3, random_state=0, penalty_type=penalty_type).fit(X, time, event)
    with pytest.raises(TypeError):
        DiscreteSurvival(groups=[1, 1, 2])
    assert DiscreteSurvival(n_lambda=5).fit(X, time, event).coef_path_.shape == (5, 3)


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_group_lasso_cox_null_point_is_exact(alpha):
    """C23: at lambda >= lambda_max GroupLassoCoxPH returns the null point exactly, as the provider classes do."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 5))
    t = rng.exponential(1 / np.exp(X @ [0.5, 0.3, 0.0, 0.0, 0.2]))
    ev = (rng.uniform(size=400) < 0.7).astype(float)
    m = GroupLassoCoxPH(groups=np.array([0, 1, 1, 2, 2]), alpha=alpha, n_lambda=10).fit(X, duration=t, event=ev)
    assert np.all(m.coef_path_[0][1:] == 0.0)
    assert m.coef_path_[0][0] != 0.0              # the unpenalized column is fitted at the null point
    assert m.converged_path_[0] and m.kkt_violation_path_[0] <= 1e-12
