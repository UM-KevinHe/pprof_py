"""Cox group paths (C8b): the standardized group lasso on the shared exact block solver, parity with R's
grplasso::Strat.cox, ProviderPenalizedCoxPH as GroupLassoCoxPH with provider dummies, and the group paths'
convergence flags (C18)."""
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from pprof_py import GroupLassoCoxPH, GroupLassoLinear, GroupLassoLogistic, ProviderPenalizedCoxPH
from pprof_py.algorithms.survival.cox_likelihood import cox_partial_likelihood

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "data" / "cox_group_lasso"
GROUPS = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])


@pytest.fixture(scope="module")
def golden():
    raw = pd.read_csv(GOLDEN / "raw.csv")
    X = raw[[f"z{j}" for j in range(1, 10)]].to_numpy()
    return X, raw["time"].to_numpy(), raw["event"].to_numpy(float), json.loads((GOLDEN / "r_paths.json").read_text())


def _cohort(n=600, K=5, rho=0.5, seed=7):
    rng = np.random.RandomState(seed)
    cols = []
    for _g in range(3):
        base = rng.normal(size=(n, 1))
        cols.append(np.sqrt(rho) * base + np.sqrt(1 - rho) * rng.normal(size=(n, 3)))
    X = np.column_stack(cols)
    prov = np.sort(rng.randint(0, K, size=n))
    eta = X @ np.array([0.6, -0.45, 0.35, 0, 0, 0, 0.25, -0.18, 0.12]) + rng.normal(scale=0.4, size=K)[prov]
    t = rng.exponential(scale=np.exp(-eta))
    c = rng.exponential(scale=np.quantile(t, 0.75), size=n)
    return X, np.minimum(t, c) + 1e-6, (t <= c).astype(float), prov


def _score(Xs, t, d, beta, offset=None):
    n = len(t)
    return cox_partial_likelihood(Xs, np.zeros(n), t, d, beta, offset=np.zeros(n) if offset is None else offset,
                                  weight=np.ones(n), strata=np.zeros(n, dtype=np.intp), ties="breslow")[1] / n


def _kkt(s, b, lam, groups, mult, Q=None):
    worst = 0.0
    for gi, g in enumerate(sorted(set(groups[groups > 0]))):
        ix = groups == g
        bg, sg = (b[ix], s[ix]) if Q is None else (np.linalg.solve(Q[g], b[ix]), Q[g].T @ s[ix])
        nb = np.linalg.norm(bg)
        v = np.linalg.norm(sg) - lam * mult[gi] if nb == 0 else np.linalg.norm(sg - lam * mult[gi] * bg / nb)
        worst = max(worst, v)
    return max(worst, np.max(np.abs(s[groups == 0]), initial=0.0))


def _centered_q(Xs, groups):
    Zc = Xs - Xs.mean(0)
    Q = {}
    for g in sorted(set(groups[groups > 0])):
        _u, d, vt = np.linalg.svd(Zc[:, groups == g] / np.sqrt(len(Xs)), full_matrices=False)
        Q[g] = vt.T / d
    return Q


# --- R parity: Strat.cox without prov.char (one stratum) ----------------------------------------------------

@pytest.mark.parametrize("case", ["strat_cox", "strat_cox_multiplier"])
def test_group_path_matches_r_strat_cox(golden, case):
    X, t, d, r = golden
    R = r[case]
    kw = dict(groups=np.array(r["group"]))
    if case == "strat_cox_multiplier":
        kw["group_multiplier"] = np.array(r["group_multiplier"])
    m = GroupLassoCoxPH(lambda_path=np.array(R["lambda"]), outer_tol=1e-12, **kw).fit(X, duration=t, event=d)
    assert np.max(np.abs(m.coef_path_.T - np.array(R["beta"]))) < 1e-8
    assert np.ptp(m.coef_path_[:, 6]) > 0.1               # the unpenalized column is refitted along the path
    free = GroupLassoCoxPH(n_lambda=30, lambda_min_ratio=0.01, **kw).fit(X, duration=t, event=d)
    assert free.lambda_max_ == pytest.approx(R["lambda"][0] - 1e-5, rel=1e-10)   # R pads its first lambda
    assert free.lambda_path_[0] == pytest.approx(free.lambda_max_, rel=1e-12)
    assert np.all(free.coef_path_[0][np.array(r["group"]) > 0] == 0.0)
    assert free.converged_path_.all() and np.nanmax(free.kkt_violation_path_) < 1e-8


# --- Optimality ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("orthogonalize", [True, False])
def test_group_path_satisfies_kkt(orthogonalize):
    X, t, d, _prov = _cohort()
    m = GroupLassoCoxPH(groups=GROUPS, orthogonalize=orthogonalize, n_lambda=15, lambda_min_ratio=0.02
                        ).fit(X, duration=t, event=d)
    sd = X.std(0)
    Q = _centered_q(X / sd, GROUPS) if orthogonalize else None
    worst = max(_kkt(_score(X / sd, t, d, m.coef_path_[i] * sd), m.coef_path_[i] * sd, lam, GROUPS,
                     np.sqrt([3.0, 3.0, 3.0]), Q) for i, lam in enumerate(m.lambda_path_))
    assert worst < 1e-6


# --- ProviderPenalizedCoxPH: GroupLassoCoxPH with the provider dummies unpenalized -------------------------

@pytest.mark.parametrize("multiplier", [None, np.array([1.0, 1.5, 0.8])])
def test_provider_group_path_is_group_lasso_with_provider_dummies(multiplier):
    X, t, d, prov = _cohort()
    K = prov.max() + 1
    D = (prov[:, None] == np.arange(1, K)[None, :]).astype(float)
    kw = {} if multiplier is None else dict(group_multiplier=multiplier)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = ProviderPenalizedCoxPH(penalty_type="group_lasso", groups=GROUPS, n_lambda=12, lambda_min_ratio=0.02,
                                    provider_tol=1e-10, outer_tol=1e-10, provider_max_iter=500, **kw
                                    ).fit(X, duration=t, event=d, provider_id=prov)
    groups_aug = np.r_[GROUPS, np.zeros(K - 1, dtype=int)]
    gl = GroupLassoCoxPH(groups=groups_aug, lambda_path=pp.lambda_path_, outer_tol=1e-12, **kw
                         ).fit(np.c_[X, D], duration=t, event=d)
    free = GroupLassoCoxPH(groups=groups_aug, n_lambda=12, lambda_min_ratio=0.02, **kw).fit(np.c_[X, D], duration=t,
                                                                                              event=d)
    assert pp.lambda_max_ == pytest.approx(free.lambda_max_, rel=1e-8)
    assert np.max(np.abs(pp.coef_path_ - gl.coef_path_[:, :9])) < 1e-8
    assert np.max(np.abs((pp.gamma_path_[:, 1:] - pp.gamma_path_[:, :1]) - gl.coef_path_[:, 9:])) < 1e-8
    assert np.all(pp.coef_path_[0] == 0.0) and pp.converged_path_.all()


def test_provider_group_path_refits_unpenalized_columns():
    X, t, d, prov = _cohort()
    groups = np.array([1, 1, 1, 2, 2, 2, 0, 3, 3])
    pp = ProviderPenalizedCoxPH(penalty_type="group_lasso", groups=groups, n_lambda=10, lambda_min_ratio=0.05
                                ).fit(X, duration=t, event=d, provider_id=prov)
    assert pp.coef_path_[0][6] != 0.0                     # at its null-point MLE, not zero
    assert np.ptp(pp.coef_path_[:, 6]) > 0.0
    assert np.all(pp.coef_path_[0][groups > 0] == 0.0)


def test_provider_cox_alpha_follows_the_penalty_type():
    X, t, d, prov = _cohort(n=300)
    fit = lambda **kw: ProviderPenalizedCoxPH(n_lambda=3, **kw).fit(X, duration=t, event=d, provider_id=prov)
    assert fit().alpha_ == 1.0
    assert fit(penalty_type="group_lasso", groups=GROUPS).alpha_ == 0.0
    assert fit(penalty_type="sparse_group_lasso", alpha=0.4, groups=GROUPS).alpha_ == 0.4
    with pytest.raises(ValueError, match="pure group lasso"):
        fit(penalty_type="group_lasso", alpha=1.0, groups=GROUPS)
    with pytest.raises(ValueError, match="needs alpha"):
        fit(penalty_type="sparse_group_lasso", groups=GROUPS)


def test_provider_cox_first_point_is_the_null_point():
    X, t, d, prov = _cohort()
    for kw in (dict(penalty_type="group_lasso", groups=GROUPS), dict()):
        pp = ProviderPenalizedCoxPH(n_lambda=5, **kw).fit(X, duration=t, event=d, provider_id=prov)
        assert np.all(pp.coef_path_[0] == 0.0)
        assert pp.n_provider_iter_path_[0] == 0


# --- C18: convergence is certified by the KKT residual -------------------------------------------------------

def test_group_paths_report_convergence_at_their_solutions():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 9)) + rng.normal(size=(600, 1))
    beta = np.array([0.9, -0.6, 0.5, 0, 0, 0, 0.3, -0.2, 0.1])
    y = rng.binomial(1, 1 / (1 + np.exp(-(X @ beta))))
    for m in (GroupLassoLogistic(groups=GROUPS, n_lambda=20, lambda_min_ratio=0.01).fit(X, y),
              GroupLassoLinear(groups=GROUPS, n_lambda=20, lambda_min_ratio=0.01).fit(X, X @ beta + rng.normal(size=600))):
        assert m.converged_path_.all()
        assert np.nanmax(m.kkt_violation_path_) < 1e-8
        assert np.median(m.n_iter_path_) < 50
