"""Provider-penalized group paths (C8): the standardized group lasso, an exact block solver, lambda_max
at the null point with the provider effects fitted, and parity with R grplasso's grp.lasso / pp.lasso."""
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from pprof_py import GroupLassoLinear, GroupLassoLogistic, ProviderPenalizedLogistic
from pprof_py.algorithms.coordinate_descent import (
    _sparse_group_coordinate_descent_python, _group_block_majorizers, solve_sparse_group_penalized_quadratic,
)
from pprof_py.algorithms.penalty import fit_group_multipliers, validate_groups

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "data" / "provider_group_lasso"
TIGHT = dict(outer_tol=1e-10, max_outer_iter=5000, provider_max_iter=50)


@pytest.fixture(scope="module")
def golden():
    raw = pd.read_csv(GOLDEN / "raw.csv")
    X = raw[[f"z{j}" for j in range(1, 10)]].to_numpy()
    return X, raw["y"].to_numpy(float), raw["prov"].to_numpy(), json.loads((GOLDEN / "r_paths.json").read_text())


def _fit(X, y, prov, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ProviderPenalizedLogistic(**kw).fit(X, y, provider_id=prov)


# --- R parity: grp.lasso(prov.char=) and pp.lasso(prov.char=) ------------------------------------------

@pytest.mark.parametrize("case", ["grp_lasso", "grp_lasso_multiplier"])
def test_group_path_matches_r_grp_lasso(golden, case):
    X, y, prov, r = golden
    R = r[case]
    kw = dict(penalty_type="group_lasso", groups=np.array(r["group"]))
    if case == "grp_lasso_multiplier":
        kw["group_multiplier"] = np.array(r["group_multiplier"])
    m = _fit(X, y, prov, lambda_path=np.array(R["lambda"]), **kw, **TIGHT)
    assert list(m.provider_labels_) == R["providers"]
    assert np.max(np.abs(m.coef_path_.T - np.array(R["beta"]))) < 1e-8
    # R stops after one iteration at its first lambda (no coefficient moves), so its gamma there is one
    # Newton step from the pooled logit; compare the provider effects from the second lambda on.
    effects = (m.gamma_path_ + m.intercept_path_[:, None]).T
    assert np.max(np.abs(effects - np.array(R["gamma"]))[:, 1:]) < 1e-8
    assert np.ptp(m.coef_path_[:, 6]) > 0.05          # the unpenalized column is refitted along the path
    free = _fit(X, y, prov, n_lambda=30, lambda_min_ratio=0.01, **kw)
    assert free.lambda_max_ == pytest.approx(R["lambda"][0] - 1e-5, rel=1e-10)   # R pads its first lambda


def test_lasso_path_matches_r_pp_lasso(golden):
    X, y, prov, r = golden
    R = r["pp_lasso"]
    pf = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1.0])
    # Penalty factors are rescaled to sum to p (glmnet); R's pp.lasso keeps them at 1, so lambda = lambda_R * 8/9.
    scale = pf.sum() / len(pf)
    m = _fit(X, y, prov, penalty_type="elastic_net", alpha=1.0, penalty_factor=pf,
             lambda_path=np.array(R["lambda"]) * scale, **TIGHT)
    assert np.max(np.abs(m.coef_path_.T - np.array(R["beta"]))) < 1e-8
    effects = (m.gamma_path_ + m.intercept_path_[:, None]).T
    assert np.max(np.abs(effects - np.array(R["gamma"]))[:, 1:]) < 1e-8
    free = _fit(X, y, prov, penalty_type="elastic_net", penalty_factor=pf, n_lambda=30, lambda_min_ratio=0.01)
    assert free.lambda_max_ == pytest.approx((R["lambda"][0] - 1e-5) * scale, rel=1e-10)


def test_lambda_max_uses_the_fitted_provider_effects(golden):
    X, y, prov, _ = golden
    Xp, groups = X[:, [0, 1, 2, 3, 4, 5, 7, 8]], np.array([1, 1, 1, 2, 2, 2, 3, 3])
    m = _fit(Xp, y, prov, penalty_type="group_lasso", groups=groups, n_lambda=5)
    # Independent null point: every covariate penalized, so the provider effects are the provider logits.
    Z = (Xp - Xp.mean(0)) / Xp.std(0)
    labels, idx = np.unique(prov, return_inverse=True)
    ybar = np.bincount(idx, weights=y) / np.bincount(idx)
    r = y - ybar[idx]
    lam_max = 0.0
    for g in (1, 2, 3):
        Zg = Z[:, groups == g]
        u, d, vt = np.linalg.svd(Zg / np.sqrt(len(y)), full_matrices=False)
        Zt = Zg @ (vt.T / d)
        lam_max = max(lam_max, np.linalg.norm(Zt.T @ r) / len(y) / np.sqrt(Zg.shape[1]))
    assert m.lambda_max_ == pytest.approx(lam_max, rel=1e-9)
    assert np.all(m.coef_path_[0] == 0.0)
    assert np.any(m.coef_path_[1] != 0.0)


# --- One provider: GroupLassoLogistic's path ---------------------------------------------------------------

def test_single_provider_is_group_lasso_logistic(golden):
    X, y, _prov, r = golden
    groups = np.array(r["group"])
    one = np.zeros(len(y))
    gl = GroupLassoLogistic(groups=groups, n_lambda=25, lambda_min_ratio=0.01, outer_tol=1e-12).fit(X, y)
    pp = _fit(X, y, one, penalty_type="group_lasso", groups=groups, n_lambda=25, lambda_min_ratio=0.01, **TIGHT)
    assert pp.lambda_max_ == pytest.approx(gl.lambda_max_, rel=1e-12)
    assert np.allclose(pp.lambda_path_, gl.lambda_path_, rtol=1e-12, atol=0)
    assert np.max(np.abs(pp.coef_path_ - gl.coef_path_)) < 1e-7
    assert np.max(np.abs(pp.intercept_path_ + pp.gamma_path_[:, 0] - gl.intercept_path_)) < 1e-7


def test_group_multiplier_is_honoured():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 5))
    y = rng.binomial(1, 1 / (1 + np.exp(-(X @ np.array([0.8, -0.5, 0.3, 0.0, 0.4])))))
    groups = np.array([1, 1, 2, 2, 3])
    base = GroupLassoLogistic(groups=groups, n_lambda=5).fit(X, y)
    doubled = GroupLassoLogistic(groups=groups, n_lambda=5,
                                 group_multiplier=2 * np.sqrt([2.0, 2.0, 1.0])).fit(X, y)
    assert doubled.lambda_max_ == pytest.approx(base.lambda_max_ / 2, rel=1e-12)
    yl = X @ np.array([0.8, -0.5, 0.3, 0.0, 0.4]) + rng.normal(size=400)
    lin = GroupLassoLinear(groups=groups, n_lambda=5).fit(X, yl)
    lin2 = GroupLassoLinear(groups=groups, n_lambda=5, group_multiplier=2 * np.sqrt([2.0, 2.0, 1.0])).fit(X, yl)
    assert lin2.lambda_max_ == pytest.approx(lin.lambda_max_ / 2, rel=1e-12)
    one = np.zeros(len(y))
    m = np.array([1.0, 0.5, 2.0])
    gl = GroupLassoLogistic(groups=groups, n_lambda=12, group_multiplier=m, outer_tol=1e-12).fit(X, y)
    pp = _fit(X, y, one, penalty_type="group_lasso", groups=groups, n_lambda=12, group_multiplier=m, **TIGHT)
    assert np.max(np.abs(pp.coef_path_ - gl.coef_path_)) < 1e-7


def test_fit_group_multipliers_drops_groups_without_columns():
    groups, _s, _n = validate_groups(np.array([1, 1, 2, 2, 0, 3]), 6)
    keep = np.array([True, True, False, False, True, True])
    assert np.array_equal(fit_group_multipliers(None, groups, keep), np.sqrt([2.0, 1.0]))
    assert np.array_equal(fit_group_multipliers(np.array([1.0, 7.0, 3.0]), groups, keep), [1.0, 3.0])
    with pytest.raises(ValueError):
        fit_group_multipliers(np.array([1.0, 3.0]), groups, keep)


# --- Optimality of the multi-provider path ------------------------------------------------------------------

def test_multi_provider_path_satisfies_kkt(golden):
    X, y, prov, r = golden
    groups = np.array(r["group"])
    m = _fit(X, y, prov, penalty_type="group_lasso", groups=groups, n_lambda=20, lambda_min_ratio=0.01)
    Z = (X - X.mean(0)) / X.std(0)
    Q = {}
    for g in (1, 2, 3):
        ix = groups == g
        _u, d, vt = np.linalg.svd(Z[:, ix] / np.sqrt(len(y)), full_matrices=False)
        Q[g] = vt.T / d
    _labels, idx = np.unique(prov, return_inverse=True)
    worst = 0.0
    for i, lam in enumerate(m.lambda_path_):
        eta = X @ m.coef_path_[i] + m.intercept_path_[i] + m.gamma_path_[i][idx]
        res = y - 1 / (1 + np.exp(-eta))
        assert np.max(np.abs(np.bincount(idx, weights=res))) < 1e-6           # provider scores
        b = m.coef_path_[i] * X.std(0)
        s = Z.T @ res / len(y)
        assert abs(s[6]) < 1e-6                                                # unpenalized column
        for g, Qg in Q.items():
            ix = groups == g
            bt = np.linalg.solve(Qg, b[ix])
            st = Qg.T @ s[ix]
            nb = np.linalg.norm(bt)
            mg = np.sqrt(ix.sum())
            v = np.linalg.norm(st) - lam * mg if nb == 0 else np.linalg.norm(st - lam * mg * bt / nb)
            worst = max(worst, v)
    assert worst < 1e-6


# --- The block solver ---------------------------------------------------------------------------------------

def _fista(A, lt, lam, alpha, pf, gs, ge, gw, n_iter=100000):
    L = np.linalg.eigvalsh(A).max()
    b = np.zeros(len(lt)); z = b.copy(); t = 1.0
    for _ in range(n_iter):
        v = z - (A @ z - lt) / L
        v = np.sign(v) * np.maximum(np.abs(v) - lam * alpha * pf / L, 0)
        bn = v.copy()
        for s, e, w in zip(gs, ge, gw):
            nv = np.linalg.norm(v[s:e]); thr = lam * (1 - alpha) * w / L
            bn[s:e] = 0.0 if nv <= thr else v[s:e] * (1 - thr / nv)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = bn + (t - 1) / tn * (bn - b)
        if np.max(np.abs(bn - b)) < 1e-15:
            return bn
        b, t = bn, tn
    return b


@pytest.mark.parametrize("alpha", [0.0, 0.3])
def test_block_solver_is_exact_for_any_block(alpha):
    rng = np.random.default_rng(11)
    M = rng.normal(size=(7, 7))
    A = M @ M.T / 7 + 0.05 * np.eye(7)
    lt = rng.normal(size=7)
    gs, ge = np.array([0, 3, 5], dtype=np.intp), np.array([3, 5, 7], dtype=np.intp)
    gw, pf = np.sqrt([3.0, 2.0, 2.0]), np.ones(7)
    lam = 0.15
    ref = _fista(A, lt, lam, alpha, pf, gs, ge, gw)
    b, _it, _chg = solve_sparse_group_penalized_quadratic(A, lt, np.zeros(7), lam, alpha, pf, gs, ge, gw,
                                                          tol=1e-14, max_iter=100000)
    assert np.max(np.abs(b - ref)) < 1e-9
    major, lmax = _group_block_majorizers(A, gs, ge, gw, lam, alpha)
    bp, _it, _chg = _sparse_group_coordinate_descent_python(A, lt, np.zeros(7), lam, alpha, pf, gs, ge, gw,
                                                            np.ones(3, dtype=bool), 1e-14, 100000, major, lmax)
    assert np.max(np.abs(bp - ref)) < 1e-9


def test_scalar_blocks_keep_the_one_pass_update():
    A = 0.25 * np.eye(6)
    lt = np.array([0.3, -0.2, 0.1, 0.05, 0.02, -0.01])
    gs, ge, gw = np.array([0, 3], dtype=np.intp), np.array([3, 6], dtype=np.intp), np.sqrt([3.0, 3.0])
    major, _lmax = _group_block_majorizers(A, gs, ge, gw, 0.05, 0.0)
    assert not major.any()
    b, _it, _chg = solve_sparse_group_penalized_quadratic(A, lt, np.zeros(6), 0.05, 0.0, np.ones(6), gs, ge, gw)
    exact = np.zeros(6)
    for s, e, w in zip(gs, ge, gw):
        z = lt[s:e]; nz = np.linalg.norm(z)
        exact[s:e] = 0.0 if nz <= 0.05 * w else z / 0.25 * (1 - 0.05 * w / nz)
    assert np.array_equal(b, exact)


# --- Parameters ---------------------------------------------------------------------------------------------

def test_alpha_follows_the_penalty_type(golden):
    X, y, prov, r = golden
    groups = np.array(r["group"])
    assert _fit(X, y, prov, n_lambda=3).alpha_ == 1.0
    assert _fit(X, y, prov, penalty_type="group_lasso", groups=groups, n_lambda=3).alpha_ == 0.0
    assert _fit(X, y, prov, penalty_type="sparse_group_lasso", alpha=0.4, groups=groups, n_lambda=3).alpha_ == 0.4
    with pytest.raises(ValueError, match="pure group lasso"):
        _fit(X, y, prov, penalty_type="group_lasso", alpha=0.5, groups=groups, n_lambda=3)
    with pytest.raises(ValueError, match="needs alpha"):
        _fit(X, y, prov, penalty_type="sparse_group_lasso", groups=groups, n_lambda=3)
    with pytest.raises(ValueError, match="penalty_type"):
        _fit(X, y, prov, penalty_type="lasso", n_lambda=3)


def test_unorthogonalized_group_path_is_the_plain_group_lasso(golden):
    X, y, prov, r = golden
    groups = np.array(r["group"])
    m = _fit(X, y, prov, penalty_type="group_lasso", groups=groups, orthogonalize=False, n_lambda=12,
             lambda_min_ratio=0.02)
    Z = (X - X.mean(0)) / X.std(0)
    _labels, idx = np.unique(prov, return_inverse=True)
    worst = 0.0
    for i, lam in enumerate(m.lambda_path_):
        eta = X @ m.coef_path_[i] + m.intercept_path_[i] + m.gamma_path_[i][idx]
        s = Z.T @ (y - 1 / (1 + np.exp(-eta))) / len(y)
        b = m.coef_path_[i] * X.std(0)
        for g in (1, 2, 3):
            ix = groups == g
            nb, mg = np.linalg.norm(b[ix]), np.sqrt(ix.sum())
            v = np.linalg.norm(s[ix]) - lam * mg if nb == 0 else np.linalg.norm(s[ix] - lam * mg * b[ix] / nb)
            worst = max(worst, v)
    assert worst < 1e-6
