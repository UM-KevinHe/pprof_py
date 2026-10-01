"""Coefficient forest and table: values equal summary(), exponentiation as a tested derivation, drawn data (spec §2.4)."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pprof_py import (CoxPH, LinearFixedEffectModel, LinearRandomEffectModel, LogisticFixedEffectModel,
                      LogisticRandomEffectModel)
from pprof_py.presentation import CapabilityError, CoefficientProfile, coefficient_table, forest
from pprof_py.presentation.formatting import fmt_interval


def _binary(seed=4, n=30):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n), rng.integers(30, 90, n))
    x = rng.normal(size=(pid.size, 2))
    y = rng.binomial(1, 1 / (1 + np.exp(-(rng.normal(-1.0, 0.5, n)[pid] + x @ [0.5, -0.3]))))
    return pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "provider_id": pid})


@pytest.fixture(scope="module")
def models():
    d = _binary()
    rng = np.random.default_rng(5)
    dn = d.assign(y=rng.normal(size=len(d)) + 0.4 * d["x1"])
    fe, re = LogisticFixedEffectModel(), LogisticRandomEffectModel(verbose=False)
    lfe, lre = LinearFixedEffectModel(), LinearRandomEffectModel()
    for m, data in ((fe, d), (re, d), (lfe, dn), (lre, dn)):
        m.fit(data, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    prov = np.repeat(np.arange(15), 40)
    X = pd.DataFrame(rng.normal(size=(prov.size, 2)), columns=["a", "b"])
    t = rng.exponential(1 / (0.3 * np.exp(X.to_numpy() @ [0.5, -0.3])))
    c = rng.uniform(0.5, 3, prov.size)
    cox = CoxPH(ties="breslow").fit(X, duration=np.minimum(t, c), event=(t <= c).astype(float), strata=prov)
    return {"fe": fe, "re": re, "lfe": lfe, "lre": lre, "cox": cox}


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


@pytest.mark.parametrize("key,cols", [
    ("fe", ("estimate", "std_error", "ci_lower", "ci_upper", "p_value")),
    ("re", ("Estimate", "Std.Error", "ci_lower", "ci_upper", "Pr(>|z|)")),
    ("lfe", ("estimate", "std_error", "ci_lower", "ci_upper", "p_value")),
    ("lre", ("estimate", "std_error", "ci_lower", "ci_upper", "p_value")),
    ("cox", ("coef", "se(coef)", "lower_95%", "upper_95%", "p")),
])
def test_profile_values_are_the_summary(models, key, cols):
    m = models[key]
    s = m.summary()
    prof = CoefficientProfile.from_model(m)
    f = prof.data
    assert "(Intercept)" not in f.index
    rows = [t for t in s.index if t != "(Intercept)"]
    np.testing.assert_array_equal(f.to_numpy(), s.loc[rows, list(cols)].to_numpy(dtype=float))
    if "(Intercept)" in s.index:
        assert CoefficientProfile.from_model(m, include_intercept=True).data.index[0] == "(Intercept)"


@pytest.mark.parametrize("level", [0.9, 0.95, 0.99])
def test_random_effect_summary_intervals(models, level):
    s = models["re"].summary(level=level)
    half = norm.ppf(1 - (1 - level) / 2) * s["Std.Error"]
    np.testing.assert_allclose(s["ci_lower"], s["Estimate"] - half, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s["ci_upper"], s["Estimate"] + half, rtol=0, atol=1e-12)
    assert (((s["ci_lower"] > 0) | (s["ci_upper"] < 0)) == (s["Pr(>|z|)"] < 1 - level)).all()


def test_exponentiation_is_exact_and_limited(models):
    prof = CoefficientProfile.from_model(models["fe"])
    ratio = prof.exponentiate()
    f, g = prof.data, ratio.data
    for c in ("estimate", "ci_lower", "ci_upper"):
        np.testing.assert_array_equal(g[c].to_numpy(), np.exp(f[c].to_numpy()))
    np.testing.assert_array_equal(g[["se", "p_value"]].to_numpy(), f[["se", "p_value"]].to_numpy())
    assert ratio.provenance["scale"] == "odds_ratio" and ratio.provenance["exponentiated"]
    with pytest.raises(CapabilityError, match="exponentiate"):
        ratio.exponentiate()
    with pytest.raises(CapabilityError, match="difference"):
        CoefficientProfile.from_model(models["lfe"]).exponentiate()
    with pytest.raises(CapabilityError, match="95%"):
        CoefficientProfile.from_model(models["cox"], level=0.9)


@pytest.mark.parametrize("key,exp", [("fe", True), ("cox", True), ("lfe", False), ("re", True)])
def test_forest_draws_the_profile(models, key, exp):
    m = models[key]
    r = forest(m)
    prof = CoefficientProfile.from_model(m)
    f = (prof.exponentiate() if exp else prof).data
    k = len(f)
    segs = np.asarray(_gid(r.figure, "coefficient-intervals")[0].get_segments())
    np.testing.assert_array_equal(segs[:, 0, 0], f["ci_lower"].to_numpy())
    np.testing.assert_array_equal(segs[:, 1, 0], f["ci_upper"].to_numpy())
    np.testing.assert_array_equal(segs[:, 0, 1], np.arange(k - 1, -1, -1))           # first term at the top
    pts = np.asarray(_gid(r.figure, "coefficient-estimates")[0].get_offsets())
    np.testing.assert_array_equal(pts[:, 0], f["estimate"].to_numpy())
    (ref,) = _gid(r.figure, "reference")
    assert np.all(np.asarray(ref.get_xdata()) == (1.0 if exp else 0.0))
    assert r.axes.get_xscale() == ("log" if exp else "linear")
    texts = [t.get_text() for t in _gid(r.figure, "coefficient-text")]
    assert texts == list(fmt_interval(f["estimate"], f["ci_lower"], f["ci_upper"], 2))
    assert [t.get_text() for t in r.axes.get_yticklabels()] == list(f.index)
    assert "Associations are not causal" in r.long_description and r.kind == "forest"


def test_terms_ticks_and_determinism(models):
    r = forest(models["cox"], terms=["b", "a"])
    assert [t.get_text() for t in r.axes.get_yticklabels()] == ["b", "a"]
    with pytest.raises(ValueError, match="unknown terms"):
        forest(models["cox"], terms=["zzz"])
    r.to_bytes("png")
    labels = [t.get_text() for t in r.axes.get_xticklabels() if t.get_text()]
    assert len(labels) >= 3                                       # narrow ratio range: plain ticks
    a, b = forest(models["fe"]), forest(models["fe"])
    for fmt in ("svg", "pdf", "png"):
        assert a.to_bytes(fmt) == b.to_bytes(fmt) == a.to_bytes(fmt)


def test_coefficient_table(models):
    t = coefficient_table(models["fe"])
    prof = CoefficientProfile.from_model(models["fe"]).exponentiate()
    assert [c.header for c in t.spec.columns] == ["Covariate", "Odds ratio (95% CI)", "p-value"]
    assert list(t.spec.cells["estimate_ci"]) == list(fmt_interval(prof.data["estimate"], prof.data["ci_lower"],
                                                                  prof.data["ci_upper"], 2))
    pd.testing.assert_frame_equal(t.to_frame(), prof.data, check_names=False)
    assert "not causal" in dict(t.spec.notes)["a"] and "exponentiated" in dict(t.spec.notes)["a"]
    assert "Odds ratio (95% CI)\u1d43" in t.to_markdown()
    lin = coefficient_table(models["lfe"])
    assert lin.spec.columns[1].header == "Coefficient (outcome units) (95% CI)"
