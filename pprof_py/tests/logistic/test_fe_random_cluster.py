"""LogisticFERandomClusterModel inference: exact tails, inverted-test intervals, the
resampling floor fallback, Stage 1 summary routing, and the Stage 3 fit options."""
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from pprof_py import LogisticRandomEffectModel, LogisticFixedEffectModel, LogisticFERandomClusterModel
from pprof_py.inference import FixedNull
from pprof_py.models.logistic.fe_random_cluster import _marginal_loglik
from pprof_py.inference.effect_tests import (EXACT_P_FLOOR, clustered_poibin_tails, integrated_poibin_tails,
                                             poibin_tails, resample_tails, z_from_tails)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _pmf_enumerate(p):
    """Poisson-binomial pmf by enumerating all outcomes (small n only)."""
    pmf = np.zeros(p.size + 1)
    for ys in itertools.product((0, 1), repeat=p.size):
        ys = np.array(ys)
        pmf[ys.sum()] += np.prod(np.where(ys == 1, p, 1 - p))
    return pmf


def _mid_tails(pmf, o):
    return pmf[o + 1:].sum() + 0.5 * pmf[o], pmf[:o].sum() + 0.5 * pmf[o], pmf[o:].sum(), pmf[:o + 1].sum()


class TestExactTails:
    def test_clustered_tails_match_quadrature(self):
        eta = np.array([-1.0, -0.2, 0.4, -1.5, 0.1, -0.6, 0.8])
        cluster = np.array([0, 0, 0, 1, 1, 1, 1])
        mean_c, var_c = np.array([0.3, -0.4]), np.array([0.09, 0.25])
        cells = []
        for h in (0, 1):
            e = eta[cluster == h]
            k = np.arange(e.size + 1)
            f = lambda a, kk: _pmf_enumerate(expit(e + a))[kk] * norm.pdf(a, mean_c[h], np.sqrt(var_c[h]))
            cells.append(np.array([quad(f, -np.inf, np.inf, args=(kk,), epsabs=1e-14)[0] for kk in k]))
        pmf = np.convolve(*cells)
        for o in range(eta.size + 1):
            np.testing.assert_allclose(clustered_poibin_tails(o, eta, cluster, mean_c, var_c), _mid_tails(pmf, o),
                                       rtol=0, atol=1e-10)

    def test_integrated_tails_match_quadrature(self):
        eta = np.array([-1.2, -0.3, 0.5, -2.0, 0.0])
        var = np.array([0.04, 0.2, 0.0, 0.5, 0.1])
        p = np.array([quad(lambda a: expit(e + a) * norm.pdf(a, 0, np.sqrt(v)), -np.inf, np.inf, epsabs=1e-14)[0]
                      if v > 0 else expit(e) for e, v in zip(eta, var)])
        for o in range(eta.size + 1):
            np.testing.assert_allclose(integrated_poibin_tails(o, eta, var), _mid_tails(_pmf_enumerate(p), o),
                                       rtol=0, atol=1e-10)

    def test_integrated_tails_are_the_resampling_limit(self):
        rng = np.random.default_rng(11)
        eta, mean, var = rng.normal(-1, 0.5, 40), rng.normal(0, 0.3, 40), np.full(40, 0.2)
        mc = resample_tails(14, eta, mean, var, 0.0, 200_000, np.random.default_rng(5))
        ex = integrated_poibin_tails(14, eta + mean, var)
        assert abs(mc[0] - ex[0]) < 5 * np.sqrt(ex[0] * (1 - ex[0]) / 200_000)

    def test_default_nodes_suffice(self):
        # test() uses 32 nodes: tail error below 1e-9 up to a posterior variance of 1
        rng = np.random.default_rng(2)
        eta, cluster = rng.normal(-1, 0.6, 60), rng.integers(0, 3, 60)
        mean_c, var_c = np.array([0.2, -0.3, 0.5]), np.array([0.3, 1.0, 0.1])
        np.testing.assert_allclose(clustered_poibin_tails(20, eta, cluster, mean_c, var_c, n_nodes=32),
                                   clustered_poibin_tails(20, eta, cluster, mean_c, var_c, n_nodes=120),
                                   rtol=0, atol=1e-9)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(20260925)
    n_prov, n_clust = 36, 9
    rows = []
    for j in range(n_prov):
        hs = rng.choice(n_clust, rng.integers(1, 4), replace=False)
        rows += [(j, h) for h in rng.choice(hs, rng.integers(25, 110))]
    df = pd.DataFrame(rows, columns=["provider", "cluster"])
    df["x1"], df["x2"] = rng.normal(size=len(df)), rng.binomial(1, 0.4, len(df))
    g = rng.normal(-1.2, 0.3, n_prov)
    g[[3, 17]] += [1.0, -1.0]
    a = rng.normal(0, 0.35, n_clust)
    df["y"] = rng.binomial(1, expit(g[df.provider] + a[df.cluster] + 0.5 * df.x1 - 0.3 * df.x2)).astype(float)
    df.loc[df.provider == 5, "y"] = 0.0                               # a zero-event provider
    df["combo"] = df.provider * 100 + df.cluster
    fe = LogisticFixedEffectModel(use_dataprep=False, screen_providers=False)
    _quiet(fe.fit, X=df, y_var="y", x_vars=["x1", "x2"], provider_var="combo")
    beta = np.asarray(fe.coefficients_["beta"], dtype=float).ravel()
    me = LogisticFERandomClusterModel()
    _quiet(me.fit, df, y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="cluster",
           gamma_init=np.full(n_prov, -1.2), beta=beta, sigma=0.35, verbose=False)
    return df, fe, me, beta


class TestTest:
    def test_exact_is_default(self, fitted):
        _, _, me, _ = fitted
        a, b = me.test(), me.test(test_method="exact")
        pd.testing.assert_frame_equal(a, b)
        assert a.attrs["test_method"] == "exact" and a.attrs["limits"] == "test inversion"

    def test_poibin_exact_is_the_plug_in_test(self, fitted):
        _, _, me, _ = fitted
        res = me.test(test_method="poibin_exact")
        g0 = np.median(me.gamma_)
        idx = np.asarray(me._provider_idx)
        tails = np.array([poibin_tails(me._obs[idx == j].sum(), expit(g0 + me.alpha_mean_[idx == j] + me.xbeta_[idx == j]))
                          for j in range(me.n_providers_)])
        z = z_from_tails(tails[:, 0], tails[:, 1], "two_sided", EXACT_P_FLOOR)
        np.testing.assert_array_equal(res["z_raw"].to_numpy(), z)

    @pytest.mark.parametrize("null", [None, FixedNull(mean=0.4, sd=1.6)])
    @pytest.mark.parametrize("method", ["exact", "poibin_exact"])
    def test_limits_exclude_reference_iff_flagged(self, fitted, method, null):
        _, _, me, _ = fitted
        res = me.test(test_method=method, null_model=null)
        g0 = res["null_value"].iloc[0]
        excludes = (res["ci_lower"] > g0) | (res["ci_upper"] < g0)
        np.testing.assert_array_equal(excludes.to_numpy(), (res["flag"] != 0).to_numpy())

    def test_limits_reproduce_the_target_z(self, fitted):
        _, _, me, _ = fitted
        res = me.test(test_method="exact", null_model=FixedNull(mean=0.4, sd=1.6), providers=me.provider_ids_[:4])
        me_ref = me.test(test_method="exact", null_model=FixedNull(mean=0.4, sd=1.6), reference=float(res["ci_lower"].iloc[1]))
        assert abs(me_ref["z_raw"].iloc[1] - (0.4 + norm.isf(0.025) * 1.6)) < 1e-6

    def test_one_sided_limits(self, fitted):
        _, _, me, _ = fitted
        up, down = me.test(alternative="greater"), me.test(alternative="less")
        assert np.isinf(up["ci_upper"]).all() and np.isfinite(up["ci_lower"]).any()
        assert np.isinf(down["ci_lower"]).all() and np.isfinite(down["ci_upper"]).any()

    def test_zero_event_provider_has_open_lower_limit(self, fitted):
        _, _, me, _ = fitted
        res = me.test()
        assert res.loc[5, "ci_lower"] == -np.inf and np.isfinite(res.loc[5, "ci_upper"])

    def test_resampling_floor_fallback(self, fitted):
        _, _, me, _ = fitted
        n = 300
        with pytest.warns(UserWarning, match="Monte Carlo floor"):
            res = me.test(test_method="resampling", n_resample=n, seed=4)
        assert res["ci_lower"].isna().all()
        g0 = np.median(me.gamma_)
        idx = np.asarray(me._provider_idx)
        rng = np.random.default_rng(4)
        tails = np.array([resample_tails(me._obs[idx == j].sum(), me.xbeta_[idx == j], me.alpha_mean_[idx == j],
                                         me.alpha_var_[idx == j], g0, n, rng) for j in range(me.n_providers_)])
        floor = 0.5 / n
        at_floor = np.minimum(tails[:, 0], tails[:, 1]) <= floor
        assert at_floor.any() and not at_floor.all()
        z_mc = z_from_tails(tails[:, 0], tails[:, 1], "two_sided", floor)
        np.testing.assert_array_equal(res["z_raw"].to_numpy()[~at_floor], z_mc[~at_floor])
        for j in np.flatnonzero(at_floor):
            t = integrated_poibin_tails(me._obs[idx == j].sum(), g0 + me.alpha_mean_[idx == j] + me.xbeta_[idx == j],
                                        me.alpha_var_[idx == j])
            z_exact = z_from_tails(np.array([t[0]]), np.array([t[1]]), "two_sided", EXACT_P_FLOOR)[0]
            assert abs(res["z_raw"].iloc[j] - z_exact) < 1e-10

    def test_unknown_method(self, fitted):
        with pytest.raises(ValueError, match="not supported"):
            fitted[2].test(test_method="poibin")


class TestConfidenceIntervals:
    def test_gamma_matches_test(self, fitted):
        _, _, me, _ = fitted
        ci = me.calculate_confidence_intervals(option="gamma")["gamma_ci"]
        res = me.test()
        np.testing.assert_array_equal(ci["gamma_lower"].to_numpy(), res["ci_lower"].to_numpy())
        np.testing.assert_array_equal(ci["gamma_upper"].to_numpy(), res["ci_upper"].to_numpy())

    def test_standardized_measures_map_the_gamma_limits(self, fitted):
        _, _, me, _ = fitted
        out = me.calculate_confidence_intervals(stdz=["indirect", "direct"], measure=["ratio", "rate"])
        assert set(out) == {"indirect_ratio", "indirect_rate", "direct_ratio", "direct_rate"}
        lo = me.calculate_confidence_intervals(option="gamma")["gamma_ci"]["gamma_lower"].to_numpy()
        idx = np.asarray(me._provider_idx)
        j = 2
        expected = out["indirect_ratio"]["expected"].iloc[j]
        assert np.isclose(out["indirect_ratio"]["ci_ratio_lower"].iloc[j],
                          expit(lo[j] + me.alpha_mean_[idx == j] + me.xbeta_[idx == j]).sum() / expected)
        ratio = out["indirect_ratio"]
        assert (ratio["ci_ratio_lower"] <= ratio["ci_ratio_upper"]).all()
        assert out["indirect_ratio"].loc[5, "ci_ratio_lower"] == 0.0            # open lower limit maps to 0
        with pytest.raises(ValueError, match="deterministic"):
            me.calculate_confidence_intervals(test_method="resampling")


def _stage2(df, beta, provider="provider"):
    re_ = LogisticRandomEffectModel(verbose=False)
    _quiet(re_.fit, df.assign(off=df[["x1", "x2"]].to_numpy(float) @ beta), y_var="y", x_vars=None,
           provider_var=provider, cluster_vars=["cluster"], offset_var="off", verbose=False)
    return re_


class TestStages:
    def test_stages_give_the_explicit_fit(self, fitted):
        df, fe, _, beta = fitted
        re_ = _stage2(df, beta)
        staged = LogisticFERandomClusterModel()
        _quiet(staged.fit, df, y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="cluster",
               stage1=fe, stage2=re_, verbose=False)
        start = re_.get_random_effects("provider").reindex(staged.provider_ids_).to_numpy() + \
            float(re_.coefficients_["beta"]["(Intercept)"])
        explicit = LogisticFERandomClusterModel()
        _quiet(explicit.fit, df, y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="cluster",
               beta=beta, sigma=re_.sigma_["cluster"], gamma_init=start, verbose=False)
        assert np.array_equal(staged.gamma_, explicit.gamma_)
        assert staged.sigma_ == re_.sigma_["cluster"] and np.array_equal(staged.beta_, beta)

    def test_start_matches_providers_by_id_across_id_types(self, fitted):
        df, fe, _, beta = fitted
        text = df.assign(provider=df["provider"].astype(str))         # Stage 2 sorts '1', '10', '11', ...
        re_ = _stage2(text, beta)
        assert list(re_.get_random_effects("provider").index[:3]) == ["0", "1", "10"]
        staged = LogisticFERandomClusterModel()
        _quiet(staged.fit, df, y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="cluster",
               stage1=fe, stage2=_stage2(text, beta), verbose=False)
        blup = re_.get_random_effects("provider")
        start = np.array([blup[str(p)] for p in staged.provider_ids_]) + float(re_.coefficients_["beta"]["(Intercept)"])
        explicit = LogisticFERandomClusterModel()
        _quiet(explicit.fit, df, y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="cluster",
               beta=beta, sigma=re_.sigma_["cluster"], gamma_init=start, verbose=False)
        assert np.array_equal(staged.gamma_, explicit.gamma_)

    def test_stages_or_values_not_both(self, fitted):
        df, fe, _, beta = fitted
        args = dict(y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="cluster", verbose=False)
        with pytest.raises(ValueError, match="not both"):
            LogisticFERandomClusterModel().fit(df, stage1=fe, stage2=_stage2(df, beta), sigma=0.3, **args)
        with pytest.raises(ValueError, match="both stage1 and stage2"):
            LogisticFERandomClusterModel().fit(df, stage1=fe, **args)
        with pytest.raises(ValueError, match="missing"):
            LogisticFERandomClusterModel().fit(df, beta=beta, sigma=0.3, **args)

    def test_rejects_a_stage2_without_the_cluster_factor(self, fitted):
        df, fe, _, beta = fitted
        single = LogisticRandomEffectModel(verbose=False)
        _quiet(single.fit, df, y_var="y", x_vars=None, provider_var="provider", verbose=False)
        with pytest.raises(ValueError, match="cluster_vars"):
            LogisticFERandomClusterModel().fit(df, "y", ["x1", "x2"], "provider", "cluster", stage1=fe, stage2=single)

    def test_rejects_a_stage1_with_other_covariates(self, fitted):
        df, fe, _, beta = fitted
        with pytest.raises(ValueError, match="not x_vars"):
            LogisticFERandomClusterModel().fit(df, "y", ["x1"], "provider", "cluster", stage1=fe, stage2=_stage2(df, beta))


class TestMarginalEstimator:
    @staticmethod
    def _fit(df, beta, estimator, start):
        model = LogisticFERandomClusterModel(estimator=estimator, tol=1e-9)
        return _quiet(model.fit, df, y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="cluster",
                      beta=beta, sigma=0.35, gamma_init=start, verbose=False)

    def test_is_unique_and_at_least_as_likely_as_he2013(self, fitted):
        df, _, _, beta = fitted
        a, b = self._fit(df, beta, "marginal", np.full(36, -1.2)), self._fit(df, beta, "marginal", np.zeros(36))
        he = self._fit(df, beta, "he2013", np.full(36, -1.2))
        assert a.converged_ and b.converged_
        assert np.max(np.abs(a.gamma_ - b.gamma_)) < 1e-8
        assert a.loglik_ >= he.loglik_ - 1e-9

    def test_the_estimate_is_stationary(self, fitted):
        # central differences of the marginal log-likelihood itself (nodes re-centred at every point)
        df, _, _, beta = fitted
        m = self._fit(df, beta, "marginal", np.full(36, -1.2))
        args = (m.xbeta_, m._y, np.asarray(m._provider_idx), np.asarray(m._cluster_idx), m.sigma_, m.n_nodes,
                m.n_providers_, m.n_clusters_)
        free = [j for j in range(36) if abs(m.gamma_[j] - np.median(m.gamma_)) < m.bound - 1e-6]
        for j in free[:8]:
            step = np.zeros(36)
            step[j] = 1e-4
            slope = (_marginal_loglik(m.gamma_ + step, *args) - _marginal_loglik(m.gamma_ - step, *args)) / 2e-4
            assert abs(slope) < 1e-5

    def test_tests_and_limits_work(self, fitted):
        df, _, _, beta = fitted
        m = self._fit(df, beta, "marginal", np.full(36, -1.2))
        for method in ("exact", "poibin_exact"):
            res = m.test(test_method=method)
            g0 = res["null_value"].iloc[0]
            assert ((res["flag"] != 0) == ((res["ci_lower"] > g0) | (res["ci_upper"] < g0))).all()
        assert np.isfinite(m.test(test_method="resampling", n_resample=500, seed=1)["z_raw"]).all()

    def test_rejects_an_unknown_estimator(self, fitted):
        df, _, _, beta = fitted
        with pytest.raises(ValueError, match="estimator"):
            LogisticFERandomClusterModel(estimator="exact").fit(df, "y", ["x1", "x2"], "provider", "cluster", beta=beta,
                                                                sigma=0.35, gamma_init=np.zeros(36))


class TestSummary:
    def test_routes_to_stage1_wald(self, fitted):
        df, fe, me, beta = fitted
        pd.testing.assert_frame_equal(me.summary(stage1=fe), fe.summary(test_method="wald"))

    def test_stage1_from_fit(self, fitted):
        df, fe, _, beta = fitted
        me = LogisticFERandomClusterModel()
        _quiet(me.fit, df, y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="cluster",
               stage1=fe, stage2=_stage2(df, beta), verbose=False)
        assert me.stage1_ is fe
        pd.testing.assert_frame_equal(me.summary(), fe.summary(test_method="wald"))

    def test_requires_stage1(self, fitted):
        with pytest.raises(ValueError, match="requires the Stage 1 model"):
            fitted[2].summary()

    def test_rejects_a_different_stage1(self, fitted):
        df, _, me, _ = fitted
        other = LogisticFixedEffectModel(use_dataprep=False, screen_providers=False)
        _quiet(other.fit, X=df, y_var="y", x_vars=["x1", "x2"], provider_var="provider")
        with pytest.raises(ValueError, match="differs from the beta"):
            me.summary(stage1=other)


class TestFitOptions:
    def test_update_sigma_removed(self):
        with pytest.raises(TypeError):
            LogisticFERandomClusterModel(estimator="he2013", update_sigma=True)

    def test_converged_attributes(self, fitted):
        me = fitted[2]
        assert me.converged_ is True and 0.0 <= me.convergence_ < me.tol

    def test_bound_modes(self, fitted):
        df, _, me, beta = fitted
        med = np.median(me.gamma_)
        assert np.isclose(me.gamma_[5], med - me.bound)                      # zero-event provider, relative clamp
        ab = LogisticFERandomClusterModel(estimator="he2013", bound_mode="absolute")
        _quiet(ab.fit, df, y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="cluster",
               gamma_init=np.full(36, -1.2), beta=beta, sigma=0.35, verbose=False)
        assert ab.gamma_[5] == -ab.bound
        with pytest.raises(ValueError, match="bound_mode"):
            LogisticFERandomClusterModel(estimator="he2013", bound_mode="median").fit(df, "y", ["x1", "x2"], "provider", "cluster",
                                                             gamma_init=np.zeros(36), beta=beta, sigma=0.35,
                                                             verbose=False)

    def test_max_delta_gamma_from_the_solution(self, fitted):
        df, _, me, beta = fitted
        args = dict(y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="cluster", beta=beta,
                    sigma=0.35, verbose=False)
        first = LogisticFERandomClusterModel(estimator="he2013", convergence_criterion="max_delta_gamma")
        _quiet(first.fit, df, gamma_init=np.full(36, -1.2), **args)
        again = LogisticFERandomClusterModel(estimator="he2013", convergence_criterion="max_delta_gamma")
        _quiet(again.fit, df, gamma_init=first.gamma_, **args)        # a start at the solution stops at once
        assert first.converged_ and again.converged_ and again.iterations_ == 1
        tight = LogisticFERandomClusterModel(estimator="he2013", convergence_criterion="max_delta_gamma", tol=1e-12)
        _quiet(tight.fit, df, gamma_init=np.full(36, -1.2), **args)
        np.testing.assert_allclose(first.gamma_, tight.gamma_, atol=1e-4)

    def test_stationary_start_counts_as_converged(self):
        # every provider starts at the absolute bound with no events: the objective never moves, so the
        # relative criterion is 0/0 (formerly NaN, which ended the loop silently)
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"provider": np.repeat(np.arange(6), 30), "cluster": np.tile(np.arange(3), 60),
                           "x1": rng.normal(size=180), "y": 0.0})
        me = LogisticFERandomClusterModel(estimator="he2013", bound_mode="absolute", convergence_criterion="relative")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            me.fit(df, "y", ["x1"], "provider", "cluster", gamma_init=np.full(6, -10.0), beta=np.array([0.2]), sigma=0.3,
                   verbose=False)
        assert me.converged_ and me.convergence_ == 0.0 and me.iterations_ == 2

    @staticmethod
    def _sparse_provider_data():
        # provider 7's patients each sit alone in a cluster: with a huge sigma their posterior means push
        # p to ~1, so the provider's Newton information falls to ~0
        rng = np.random.default_rng(0)
        big = pd.DataFrame({"provider": rng.integers(0, 7, 280), "cluster": rng.integers(0, 4, 280)})
        lone = pd.DataFrame({"provider": 7, "cluster": 4 + np.arange(3)})
        df = pd.concat([big, lone], ignore_index=True)
        df["x1"] = rng.normal(size=len(df))
        df["y"] = rng.binomial(1, 0.4, len(df)).astype(float)
        df.loc[df.provider == 7, "y"] = 1.0
        return df

    def test_newton_information_floor(self):
        me = LogisticFERandomClusterModel(estimator="he2013", max_iter=300, bound_mode="absolute", convergence_criterion="relative")
        with pytest.warns(RuntimeWarning, match="Newton information"):
            me.fit(self._sparse_provider_data(), "y", ["x1"], "provider", "cluster", gamma_init=np.zeros(8),
                   beta=np.array([0.1]), sigma=30.0, verbose=False)
        assert me.converged_ and np.isfinite(me.gamma_).all() and np.all(np.abs(me.gamma_) <= me.bound)

    def test_breakdown_is_reported(self):
        me = LogisticFERandomClusterModel(estimator="he2013", max_iter=300, bound_mode="relative")
        with pytest.warns(RuntimeWarning, match="Did not converge"):
            _ = me.fit(self._sparse_provider_data(), "y", ["x1"], "provider", "cluster", gamma_init=np.zeros(8),
                       beta=np.array([0.1]), sigma=30.0, verbose=False)
        assert me.converged_ is False and me.convergence_ == np.inf
