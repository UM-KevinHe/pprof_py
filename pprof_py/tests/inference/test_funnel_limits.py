"""funnel_limits: control limits that agree with test() flags by construction (S4; ADR-003, D1, D10, D11)."""
import numpy as np
import pandas as pd
import pytest

from pprof_py import (CoxPH, LinearFixedEffectModel, LinearRandomEffectModel, LogisticFixedEffectModel,
                      LogisticRandomEffectModel, LogisticThreeStageModel)
from pprof_py.inference import EmpiricalNull, FunnelLimits, funnel_limits
from pprof_py.inference._recording import record, recording
from pprof_py.inference.effect_tests import (_clustered_pmf, _pmf_tails, clustered_poibin_tails, poibin_tails,
                                             poibin_tails_all)
from pprof_py.inference.funnel import build_funnel_limits


def _logistic(n_providers=60, seed=7):
    rng = np.random.default_rng(seed)
    size = np.maximum(12, rng.lognormal(np.log(50), 0.7, n_providers).astype(int))
    size[:2] = 4                                                    # excluded by data preparation (cutoff 10)
    gamma = rng.normal(-1.2, 0.45, n_providers)
    gamma[2:8] += [1.1, -1.1, 0.9, -0.9, 1.4, -1.4]                 # planted outliers in both directions
    pid = np.repeat(np.arange(n_providers), size)
    x = rng.normal(size=(pid.size, 2))
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(gamma[pid] + x @ [0.5, -0.3]))))
    y[pid == 8] = 0                                                 # a provider with no events
    return pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "provider_id": [f"P{j:03d}" for j in pid]})


def _linear(n_providers=50, seed=8):
    rng = np.random.default_rng(seed)
    size = rng.integers(15, 80, n_providers)
    effect = rng.normal(0.0, 0.5, n_providers)
    effect[:4] += [1.5, -1.5, 1.0, -1.0]
    pid = np.repeat(np.arange(n_providers), size)
    x = rng.normal(size=(pid.size, 2))
    y = effect[pid] + x @ [0.4, -0.2] + rng.normal(0.0, 1.0, pid.size)
    return pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "provider_id": pid})


def _crossed(seed=11, n_fac=30, n_hosp=8):
    rng = np.random.default_rng(seed)
    fq, he = rng.normal(0, 0.3, n_fac), rng.normal(0, 0.4, n_hosp)
    rows = []
    for f in range(n_fac):
        hosp = rng.choice(n_hosp, rng.integers(1, 4), replace=False)
        rows.append(pd.DataFrame({"fac": f + 1, "hosp": rng.choice(hosp, rng.integers(40, 120)) + 1}))
    df = pd.concat(rows, ignore_index=True)
    df["x1"], df["x2"] = rng.normal(size=len(df)), rng.binomial(1, 0.4, len(df))
    lo = -1.2 + 0.4 * df["x1"] + 0.3 * df["x2"] + fq[df["fac"] - 1] + he[df["hosp"] - 1]
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lo)))
    return df


@pytest.fixture(scope="module")
def fe():
    m = LogisticFixedEffectModel()
    m.fit(_logistic(), y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    return m


@pytest.fixture(scope="module")
def re():
    m = LogisticRandomEffectModel(verbose=False)
    m.fit(_logistic(), y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    return m


@pytest.fixture(scope="module")
def three_stage():
    return LogisticThreeStageModel().fit(_crossed(), "y", ["x1", "x2"], "fac", "hosp")


@pytest.fixture(scope="module")
def linear():
    d = _linear()
    lfe, lre = LinearFixedEffectModel(), LinearRandomEffectModel()
    lfe.fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    lre.fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    return lfe, lre


@pytest.fixture(scope="module")
def cox():
    rng = np.random.default_rng(29)
    k = 60
    size = rng.integers(20, 120, k)
    g = rng.normal(0, 0.25, k)
    g[rng.choice(k, 4, replace=False)] += 0.7
    prov = np.repeat(np.arange(k), size)
    X = rng.normal(size=(prov.size, 2))
    t = rng.exponential(1.0 / (0.3 * np.exp(X @ [0.5, -0.3] + g[prov])))
    c = rng.uniform(0.5, 3.0, prov.size)
    stop, event = np.minimum(t, c), (t <= c).astype(float)
    m = CoxPH(ties="breslow").fit(X, duration=stop, event=event, strata=prov)
    return m, X, dict(duration=stop, event=event, provider_id=prov)


def _assert_s4(fl, discrete):
    """Tested providers lie outside their own limits exactly when flagged; discrete limits sit at half-integers."""
    p = fl.providers
    flag = p["flag"].to_numpy(dtype=float, na_value=np.nan)
    tested = ~np.isnan(flag)
    assert tested.any()
    est, lo, up = (p[c].to_numpy(dtype=float) for c in ("estimate", "lower", "upper"))
    outside = (est > up) | (est < lo)
    np.testing.assert_array_equal(outside[tested], flag[tested] != 0)
    if discrete:
        assert not np.any(np.isclose(est, up, rtol=0, atol=1e-12) | np.isclose(est, lo, rtol=0, atol=1e-12))
        e = p["expected"].to_numpy(dtype=float)
        for lim in (up, lo):
            fin = np.isfinite(lim)
            np.testing.assert_allclose((lim[fin] * e[fin]) % 1.0, 0.5, atol=1e-8)


def _group_key(s):
    return s.astype(object).where(s.notna(), "none")


def _ends_match(fl):
    """Exact curves pass through the limits of the providers at both ends of the precision grid."""
    p, c = fl.providers, fl.curves[fl.curves["test_level"]]
    assert c["null_group"].dtype == p["null_group"].dtype
    for key, cg in c.groupby(_group_key(c["null_group"]), sort=False):
        mask = (_group_key(p["null_group"]) == key).to_numpy() & p["flag"].notna().to_numpy()
        prec = p["precision"].to_numpy(dtype=float)[mask]
        for end in (0, -1):
            i = np.flatnonzero(mask)[np.nanargmin(prec) if end == 0 else np.nanargmax(prec)]
            np.testing.assert_allclose(cg[["lower", "upper"]].to_numpy(dtype=float)[end],
                                       p[["lower", "upper"]].to_numpy(dtype=float)[i], rtol=1e-12)


# --------------------------------------------------------------------------------------------- logistic FE
@pytest.mark.parametrize("test_method", ["score", "poibin_exact", "wald"])
@pytest.mark.parametrize("null", ["theoretical", "empirical"])
def test_logistic_fe_limits_agree_with_flags(fe, test_method, null):
    null_model = None if null == "theoretical" else EmpiricalNull.fitter()
    fl = fe.funnel_limits(test_method=test_method, null_model=null_model)
    _assert_s4(fl, discrete=test_method == "poibin_exact")
    assert fl.attrs["test_method"] == test_method
    assert fl.attrs["curve_kind"] == ("poisson_reference" if test_method == "poibin_exact" else "exact")
    if test_method != "poibin_exact":
        _ends_match(fl)


def test_the_funnel_default_is_the_score_test(fe):
    fl = funnel_limits(fe)
    assert isinstance(fl, FunnelLimits)
    assert fl.attrs["test_method"] == "score" and fl.attrs["precision_kind"] == "inverse_null_variance"


def test_the_test_result_is_returned_unchanged(fe):
    fl = fe.funnel_limits(test_method="poibin_exact")
    pd.testing.assert_frame_equal(fl.test, fe.test(test_method="poibin_exact"))
    pd.testing.assert_series_equal(fl.providers["flag"], fl.test["flag"])
    assert list(fl.providers.index) == list(fl.test.index)


@pytest.mark.parametrize("kw", [{"critical": 3.0}, {"alternative": "greater"}, {"alternative": "less"},
                                {"reference": "mean"}, {"level": 0.9}])
@pytest.mark.parametrize("test_method", ["score", "poibin_exact"])
def test_settings_carry_over(fe, kw, test_method):
    fl = fe.funnel_limits(test_method=test_method, **kw)
    _assert_s4(fl, discrete=test_method == "poibin_exact")
    if kw.get("alternative") == "greater":
        assert np.all(fl.providers["lower"] == -np.inf)
    if kw.get("alternative") == "less":
        assert np.all(fl.providers["upper"] == np.inf)
    if "critical" in kw:
        assert set(fl.curves.loc[fl.curves["test_level"], "critical"]) == {3.0}


def test_subset_rows_equal_the_full_rows(fe):
    full = fe.funnel_limits(test_method="poibin_exact")
    ids = list(full.providers.index[[3, 10, 20]])
    sub = funnel_limits(fe, ids, test_method="poibin_exact")
    pd.testing.assert_frame_equal(sub.providers, full.providers.loc[ids])


@pytest.mark.parametrize("test_method", ["score", "poibin_exact"])
def test_extra_levels_are_curves_only(fe, test_method):
    fl = fe.funnel_limits(test_method=test_method, levels=(0.95, 0.998))
    assert fl.attrs["levels"] == (0.95, 0.998)
    c95, c998 = (fl.curves[fl.curves["level"] == v].reset_index(drop=True) for v in (0.95, 0.998))
    assert c95["test_level"].all() and not c998["test_level"].any()
    np.testing.assert_array_equal(c95["precision"], c998["precision"])
    assert np.all(c998["upper"] >= c95["upper"]) and np.all(c998["lower"] <= c95["lower"])
    _assert_s4(fl, discrete=test_method == "poibin_exact")


def test_disagreement_is_detected(fe):
    """Negative control: a flag that no longer matches the null must make the build fail."""
    for test_method in ("poibin_exact", "score"):
        def tampered(test_method=test_method):
            res = fe.test(test_method=test_method)
            i = int(np.flatnonzero(res["flag"].to_numpy(dtype=float, na_value=np.nan) == 0)[0])
            res.iloc[i, res.columns.get_loc("flag")] = 1
            return res
        with pytest.raises(RuntimeError, match="disagree"):
            build_funnel_limits(fe, tampered, order=fe.provider_ids_)


def test_monte_carlo_tests_have_no_limits(fe):
    with pytest.raises(ValueError, match="Monte Carlo"):
        fe.funnel_limits(test_method="bootstrap_exact")


def test_invalid_levels(fe):
    with pytest.raises(ValueError, match="levels"):
        fe.funnel_limits(levels=(0.95, 1.2))


# -------------------------------------------------------------------------------------------- count kernels
def test_all_count_tails_equal_single_count_tails():
    rng = np.random.default_rng(0)
    probs, trials = rng.uniform(0.02, 0.6, 37), rng.integers(1, 4, 37)
    up, lo, ge, le, expected = poibin_tails_all(probs, trials)
    for o in range(up.size):
        assert (up[o], lo[o], ge[o], le[o]) == poibin_tails(o, probs, trials)
    assert expected == pytest.approx(float(np.sum(probs * trials)))


def test_cluster_pmf_tails_equal_the_kernel():
    rng = np.random.default_rng(1)
    eta, cluster = rng.normal(-1, 0.5, 25), rng.integers(0, 3, 25)
    mean, var = rng.normal(0, 0.3, 3), rng.uniform(0.01, 0.2, 3)
    pmf = _clustered_pmf(eta, cluster, mean, var, 20)
    for o in range(pmf.size):
        assert _pmf_tails(pmf, o) == clustered_poibin_tails(o, eta, cluster, mean, var, 20)


def test_recording_is_inert(fe):
    record("count", value=1)                                        # no active recording: nothing happens
    with recording() as outer:
        a = fe.test(test_method="poibin_exact")
        with recording() as inner:
            fe.test(test_method="score")
    pd.testing.assert_frame_equal(a, fe.test(test_method="poibin_exact"))
    assert [r["kind"] for r in outer] == ["count"] and [r["kind"] for r in inner] == ["score"]


# ----------------------------------------------------------------------------------------- other families
def test_random_effect_count_funnel(re):
    fl = re.funnel_limits()
    _assert_s4(fl, discrete=True)
    assert fl.attrs["test_method"] == "poibin_exact" and fl.attrs["reference"] == 0.0


@pytest.mark.parametrize("test_method,match", [("wald", "ADR-004"), ("resampling", "Monte Carlo")])
def test_random_effect_refusals(re, test_method, match):
    with pytest.raises(ValueError, match=match):
        re.funnel_limits(test_method=test_method)


@pytest.mark.parametrize("test_method", ["exact", "poibin_exact"])
def test_three_stage_count_funnel(three_stage, test_method):
    fl = three_stage.funnel_limits(test_method=test_method)
    _assert_s4(fl, discrete=True)
    assert fl.attrs["model"] == "LogisticFERandomClusterModel"


@pytest.mark.parametrize("null", ["theoretical", "empirical"])
def test_linear_fixed_effect_wald_funnel(linear, null):
    lfe, _ = linear
    fl = lfe.funnel_limits(null_model=None if null == "theoretical" else EmpiricalNull.fitter(), levels=(0.95, 0.99))
    _assert_s4(fl, discrete=False)
    _ends_match(fl)
    assert fl.attrs["estimate_kind"] == "effect" and "Student-t" in fl.attrs["limit_rule"]
    assert fl.providers["observed"].isna().all()


def test_linear_random_effect_has_no_funnel(linear):
    _, lre = linear
    with pytest.raises(TypeError, match="ADR-004"):
        funnel_limits(lre)


@pytest.mark.parametrize("kw", [{}, {"test_method": "exact"}, {"null_model": EmpiricalNull.fitter()}])
def test_coxph_funnel(cox, kw):
    m, X, data = cox
    fl = m.funnel_limits(X, **data, **kw)
    _assert_s4(fl, discrete=True)
    _ends_match(fl)
    assert fl.attrs["measure"] == "indirect_ratio" and fl.attrs["reference"] == 1.0
    assert fl.attrs["curve_kind"] == "exact"


def test_unsupported_models():
    with pytest.raises(TypeError, match="no funnel limits"):
        funnel_limits(object())
