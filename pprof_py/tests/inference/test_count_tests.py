"""The shared count-test component and the provider tests built on it."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from pprof_py import LogisticFixedEffectModel, LogisticRandomEffectModel
from pprof_py.inference.count_tests import ClusterMixture, MonteCarlo, PlugIn, count_test, rows_by_provider
from pprof_py.inference.effect_tests import EXACT_P_FLOOR, poibin_tails, z_from_tails


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    m, k = 40, 6
    prov = np.repeat(np.arange(m), rng.integers(30, 90, m))
    clus = rng.integers(0, k, prov.size)
    x = rng.normal(size=(prov.size, 2))
    y = rng.binomial(1, expit(-1.2 + x @ [0.5, -0.3] + rng.normal(0, 0.4, m)[prov] + rng.normal(0, 0.5, k)[clus]))
    y = y.astype(float)
    y[prov == 3] = 0.0                                              # a zero-event provider
    df = pd.DataFrame(x, columns=["x1", "x2"])
    df["y"], df["provider"], df["cluster"] = y, prov, clus
    return df


@pytest.fixture(scope="module")
def re_clustered(data):
    return _quiet(LogisticRandomEffectModel(verbose=False).fit, data, y_var="y", x_vars=["x1", "x2"],
                  provider_var="provider", cluster_vars=["cluster"], verbose=False)


def test_rows_by_provider_keeps_data_order():
    assert [r.tolist() for r in rows_by_provider(np.array([2, 0, 2, 1, 0, 2]), 3)] == [[1, 4], [3], [0, 2, 5]]


def test_plug_in_matches_the_kernel_and_its_limits_invert_the_test():
    rng = np.random.default_rng(3)
    eta = [rng.normal(-1, 0.5, n) for n in (40, 60, 25)]
    obs = np.array([12.0, 9.0, 0.0])
    nulls = [PlugIn(prob=lambda g, e=e: expit(g + e)) for e in eta]
    z, limits = count_test(obs, nulls, 0.0, alternative="two_sided", start=np.zeros(3))
    direct = np.array([poibin_tails(o, expit(e)) for o, e in zip(obs, eta)])
    assert np.array_equal(z, z_from_tails(direct[:, 0], direct[:, 1], "two_sided", EXACT_P_FLOOR))
    c = norm.ppf(0.975)
    lower, upper = limits(np.full(3, c), np.full(3, -c))
    for j in (0, 1):
        for g, target in ((lower[j], c), (upper[j], -c)):
            zg, _ = count_test(obs[j:j + 1], nulls[j:j + 1], g, alternative="two_sided", start=[0.0])
            assert abs(zg[0] - target) < 1e-6
    assert lower[2] == -np.inf and np.isfinite(upper[2])            # no events: the lower limit is open


def test_monte_carlo_floor_falls_back_to_the_exact_null():
    exact = PlugIn(prob=lambda g: np.full(50, 0.02))
    nulls = [MonteCarlo(simulate=lambda o, g: (0.0, 1.0, 0.0, 1.0), n_resample=1000, exact=exact)]
    with pytest.warns(UserWarning, match="1 of 1"):
        z, limits = count_test(np.array([12.0]), nulls, 0.0, alternative="two_sided",
                               floor_message="{n} of {total} at {floor}")
    t = poibin_tails(12.0, np.full(50, 0.02))
    assert limits is None
    assert z[0] == z_from_tails(np.array([t[0]]), np.array([t[1]]), "two_sided", EXACT_P_FLOOR)[0]


def test_nulls_are_all_exact_or_all_monte_carlo():
    plug = PlugIn(prob=lambda g: np.full(5, 0.3))
    simulated = MonteCarlo(simulate=lambda o, g: (0.5, 0.5, 0.5, 0.5), n_resample=100)
    with pytest.raises(ValueError, match="all exact or all Monte Carlo"):
        count_test(np.array([1.0, 1.0]), [plug, simulated], 0.0, alternative="two_sided", start=np.zeros(2))


def test_random_effect_exact_limits_are_dual_to_its_flags(re_clustered):
    res = re_clustered.test(test_method="exact")
    g0 = res["null_value"].iloc[0]
    assert np.isfinite(res["z_raw"]).all()
    assert ((res["flag"] != 0) == ((res["ci_lower"] > g0) | (res["ci_upper"] < g0))).all()
    zero_event = [label for label in res.index if str(label) == "3"][0]
    assert res.loc[zero_event, "ci_lower"] == -np.inf


def test_random_effect_exact_needs_exactly_one_cluster_factor(data):
    single = _quiet(LogisticRandomEffectModel(verbose=False).fit, data, y_var="y", x_vars=["x1", "x2"],
                    provider_var="provider", verbose=False)
    with pytest.raises(ValueError, match="exactly one cluster factor"):
        single.test(test_method="exact")


def test_random_effect_exact_matches_a_shared_cluster_simulation(re_clustered):
    m = re_clustered
    g0 = float(m.test(test_method="exact")["null_value"].iloc[0])
    xb, y = np.asarray(m.xbeta_, float).ravel(), np.asarray(m._y, float).ravel()
    prov, clus = np.asarray(m._group_indices[0]).ravel(), np.asarray(m._group_indices[1]).ravel()
    mean_c = m.get_random_effects("cluster").to_numpy(float)
    var_c = m._get_posterior_se("cluster").to_numpy(float) ** 2
    rng, draws = np.random.default_rng(5), 4000
    for j in range(0, 40, 8):
        rows = np.flatnonzero(prov == j)
        obs, cl = y[rows].sum(), clus[rows]
        levels, local = np.unique(cl, return_inverse=True)
        u = mean_c[levels] + np.sqrt(var_c[levels]) * rng.standard_normal((draws, levels.size))   # one draw per cluster
        counts = (rng.random((draws, rows.size)) < expit(g0 + xb[rows] + u[:, local])).sum(axis=1)
        tails = ClusterMixture(eta=lambda g: g + xb[rows], cluster=cl, mean=mean_c, var=var_c, n_nodes=32).tails(obs, g0)
        for simulated, exact in (((counts >= obs).mean(), tails[2]), ((counts <= obs).mean(), tails[3])):
            assert abs(simulated - exact) <= 4 * np.sqrt(max(exact * (1 - exact), 1e-4) / draws)


def test_random_effect_resampling_floor_uses_exact_tails(re_clustered):
    with pytest.warns(UserWarning, match="Monte Carlo floor"):
        res = re_clustered.test(test_method="resampling", n_resample=50, seed=1)
    assert np.isfinite(res["z_raw"]).all()


def test_fixed_effect_exact_intervals_are_the_inverted_test(data):
    fe = _quiet(LogisticFixedEffectModel().fit, data[["x1", "x2"]].to_numpy(), data["y"].to_numpy(),
                data["provider"].to_numpy())
    res = fe.test(test_method="poibin_exact")
    ci = fe.calculate_confidence_intervals(option="gamma", test_method="exact")
    ci = (ci["gamma_ci"] if isinstance(ci, dict) else ci).set_index("provider_id").reindex(res.index)
    assert np.array_equal(ci["gamma_lower"].to_numpy(), res["ci_lower"].to_numpy(), equal_nan=True)
    assert np.array_equal(ci["gamma_upper"].to_numpy(), res["ci_upper"].to_numpy(), equal_nan=True)
    g0 = res["null_value"].iloc[0]
    assert ((res["flag"] != 0) == ((res["ci_lower"] > g0) | (res["ci_upper"] < g0))).all()
