"""ProviderProfile: presentation data equal to its sources, with statuses, provenance and capabilities (spec §6)."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pprof_py import (CoxPH, LinearFixedEffectModel, LinearRandomEffectModel, LogisticFixedEffectModel,
                      LogisticRandomEffectModel)
from pprof_py.inference import degenerate_providers
from pprof_py.presentation import PROFILE_COLUMNS, STATUSES, CapabilityError, ProviderProfile
from pprof_py.presentation.data._profile import TEST_COLUMNS


def _logistic(n_providers=40, seed=5):
    rng = np.random.default_rng(seed)
    size = rng.integers(25, 90, n_providers)
    size[:2] = [4, 7]                                               # removed by data preparation
    gamma = rng.normal(-1.0, 0.4, n_providers)
    gamma[2:6] += [1.0, -1.0, 0.8, -0.8]
    pid = np.repeat(np.arange(n_providers), size)
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(gamma[pid] + 0.5 * x))))
    y[pid == 6] = 0                                                 # no events
    return pd.DataFrame({"y": y, "x1": x, "provider_id": [f"P{j:02d}" for j in pid]})


def _linear(n_providers=30, seed=6):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_providers), rng.integers(15, 60, n_providers))
    x = rng.normal(size=pid.size)
    y = rng.normal(0, 0.5, n_providers)[pid] + 0.4 * x + rng.normal(size=pid.size)
    return pd.DataFrame({"y": y, "x1": x, "provider_id": pid})


@pytest.fixture(scope="module")
def fe():
    m = LogisticFixedEffectModel()
    m.fit(_logistic(), y_var="y", x_vars=["x1"], provider_var="provider_id")
    return m


@pytest.fixture(scope="module")
def linear():
    d = _linear()
    lfe, lre = LinearFixedEffectModel(), LinearRandomEffectModel()
    lfe.fit(d, y_var="y", x_vars=["x1"], provider_var="provider_id")
    lre.fit(d, y_var="y", x_vars=["x1"], provider_var="provider_id")
    return lfe, lre


@pytest.fixture(scope="module")
def cox():
    rng = np.random.default_rng(3)
    prov = np.repeat(np.arange(15), 40)
    X = rng.normal(size=(prov.size, 2))
    t = rng.exponential(1 / (0.3 * np.exp(X @ [0.5, -0.3])))
    c = rng.uniform(0.5, 3, prov.size)
    data = dict(duration=np.minimum(t, c), event=(t <= c).astype(float), provider_id=prov)
    return CoxPH(ties="breslow").fit(X, duration=data["duration"], event=data["event"], strata=prov), X, data


def _equal(a, b):
    return np.array_equal(pd.Series(a).to_numpy(dtype=float, na_value=np.nan),
                          pd.Series(b).to_numpy(dtype=float, na_value=np.nan), equal_nan=True)


# ------------------------------------------------------------------------------- values equal their sources
def test_test_columns_equal_the_test(fe):
    prof, res = ProviderProfile.from_model(fe), fe.test()
    f = prof.data
    assert list(f.columns) == list(PROFILE_COLUMNS) and f.index.equals(res.index) and f.index.name == "provider_id"
    for c in TEST_COLUMNS:
        if c == "null_group":
            assert f[c].isna().equals(res[c].isna())
        else:
            assert _equal(f[c], res[c]), c
    d = degenerate_providers(fe)
    assert _equal(f["observed"], d["events"]) and _equal(f["denominator"], d["trials"])
    assert f["zero_events"].astype(bool).equals(d["zero_events"]) and f["finite_estimate"].astype(bool).equals(
        d["finite_estimate"])


def test_funnel_columns_equal_the_funnel_limits(fe):
    prof, fl = ProviderProfile.from_model(fe, limits=True, levels=(0.95, 0.998)), fe.funnel_limits(levels=(0.95, 0.998))
    f = prof.data
    for a, b in (("funnel_estimate", "estimate"), ("funnel_precision", "precision"), ("funnel_lower", "lower"),
                 ("funnel_upper", "upper"), ("observed", "observed"), ("expected", "expected"), ("flag", "flag")):
        assert _equal(f[a], fl.providers[b]), a
    pd.testing.assert_frame_equal(prof.funnel_curves, fl.curves)
    assert prof.provenance["test_method"] == "score" and prof.provenance["funnel"]["levels"] == (0.95, 0.998)


def test_linear_denominators_are_the_provider_sizes(linear):
    lfe, lre = linear
    for m in (lfe, lre):
        f = ProviderProfile.from_model(m).data
        sizes = pd.Series(np.asarray(m.provider_sizes_, dtype=float), index=np.asarray(m.provider_ids_))
        assert _equal(f["denominator"], sizes.loc[f.index]) and f["finite_estimate"].all()
        assert f["observed"].isna().all()


def test_coxph_counts_are_the_test_columns(cox):
    m, X, data = cox
    prof, res = ProviderProfile.from_model(m, X, **data), m.test(X, **data)
    f = prof.data
    for c in ("observed", "expected", "person_time"):
        assert _equal(f[c], res[c]), c
    assert _equal(f["denominator"], res["expected"]) and prof.provenance["denominator_kind"] == "expected"
    np.testing.assert_array_equal(f["zero_events"].astype(bool), res["observed"].to_numpy() == 0)
    assert prof.provenance["measure"] == "indirect_ratio" and prof.provenance["reference"] == 1.0


def test_subset_keeps_the_source_order(fe):
    ids = list(fe.provider_ids_[[7, 2, 4]])
    prof = ProviderProfile.from_model(fe, providers=ids)
    assert list(prof.data.index) == [i for i in fe.provider_ids_ if i in ids]
    assert prof.provenance["providers"] == ids


# --------------------------------------------------------------------------------- statuses and provenance
def test_statuses_follow_the_flags(fe):
    res = fe.test().copy()
    res.iloc[0, res.columns.get_loc("flag")] = pd.NA
    res.iloc[0, res.columns.get_loc("ci_lower")] = np.nan
    res.iloc[0, res.columns.get_loc("ci_upper")] = np.nan
    f = ProviderProfile.from_test(res, model=fe).data
    expected = res["flag"].map({1: "above", -1: "below", 0: "not_different"}).fillna("not_tested")
    assert list(f["status"].astype(str)) == list(expected)
    assert list(f["status"].cat.categories) == list(STATUSES)
    assert not f["has_interval"].iloc[0] and f["has_interval"].iloc[1:].all()


def test_status_counts(fe):
    prof = ProviderProfile.from_model(fe)
    counts = prof.status_counts()
    assert sum(counts[s] for s in STATUSES) == len(prof)
    assert counts["zero_events"] == 1 and counts["no_finite_estimate"] == 1 and counts["excluded"] == 2


def test_provenance_essentials(fe, linear):
    p = ProviderProfile.from_model(fe).provenance
    assert p["model"] == "LogisticFixedEffectModel" and p["estimator"] == "fixed effect (unshrunken)"
    assert p["scale"] == "log_odds" and p["test_method"] == "poibin_exact" and p["reference"] == "median"
    assert p["reference_value"] == fe.test().attrs["reference"] and p["level"] == 0.95
    assert p["null_model"]["kind"] == "theoretical" and p["covariates"] == ("x1",) and p["excluded"] == 2
    lre = linear[1]
    q = ProviderProfile.from_model(lre).provenance
    assert q["estimator"] == "random effect (shrunken BLUP)" and q["scale"] == "difference" and q["reference"] == 0.0


def test_random_effect_count_profile(fe):
    re = LogisticRandomEffectModel(verbose=False)
    re.fit(_logistic(), y_var="y", x_vars=["x1"], provider_var="provider_id")
    prof = ProviderProfile.from_model(re, limits=True)
    assert prof.provenance["test_method"] == "poibin_exact" and "funnel_limits" in prof.capabilities
    assert prof.data["finite_estimate"].all() and prof.provenance["excluded"] is None


# --------------------------------------------------------------------------------------------- capabilities
def test_capabilities_and_require(fe):
    exact = ProviderProfile.from_model(fe)
    assert {"flags", "intervals", "denominator", "observed", "degeneracy"} <= exact.capabilities
    assert "funnel_limits" not in exact.capabilities
    with pytest.raises(CapabilityError, match="limits=True"):
        exact.require("funnel", "funnel_limits")
    score = ProviderProfile.from_model(fe, limits=True)
    assert "intervals" not in score.capabilities                    # the score test has no intervals
    with pytest.raises(CapabilityError, match="intervals"):
        score.require("caterpillar", "intervals")
    bare = ProviderProfile.from_test(fe.test())
    with pytest.raises(CapabilityError, match="denominators="):
        bare.require("provider table", "denominator")


def test_funnels_that_cannot_agree_with_their_flags(linear, fe):
    with pytest.raises(CapabilityError, match="ADR-004"):
        ProviderProfile.from_model(linear[1], limits=True)
    re = LogisticRandomEffectModel(verbose=False)
    with pytest.raises(ValueError, match="ADR-004"):
        ProviderProfile.from_model(re, limits=True, test_method="wald")


def test_from_test_with_explicit_denominators(fe):
    res = fe.test()
    n = pd.Series(np.arange(len(res), dtype=float) + 10.0, index=res.index[::-1])   # aligned by provider, not position
    prof = ProviderProfile.from_test(res, denominators=n, denominator_kind="patients")
    assert _equal(prof.data["denominator"], n.loc[res.index]) and prof.provenance["denominator_kind"] == "patients"
    with pytest.raises(ValueError, match="provider-test columns"):
        ProviderProfile.from_test(res.drop(columns="flag"))


# ------------------------------------------------------------------------------------------------ from_frame
def _frame(ids):
    return pd.DataFrame({"id": ids, "est": [1.4, 0.6, 1.0, 1.1], "lo": [1.1, 0.4, 0.8, 0.9], "hi": [1.8, 0.9, 1.25, 1.3],
                         "flag": [1, -1, 0, 0], "null": 1.0, "n": [120, 80, 40, 60]})


@pytest.mark.parametrize("ids", [[3, 1, 2, 0], ["c", "a", "b", "d"], pd.Categorical(["c", "a", "b", "d"])])
def test_from_frame_keeps_ids_and_order(ids):
    roles = {"provider_id": "id", "estimate": "est", "ci_lower": "lo", "ci_upper": "hi", "flag": "flag",
             "null_value": "null", "denominator": "n"}
    prof = ProviderProfile.from_frame(_frame(ids), roles=roles, provenance={"level": 0.95, "test_method": "custom"})
    assert list(prof.data.index) == list(ids)
    assert list(prof.data["status"].astype(str)) == ["above", "below", "not_different", "not_different"]
    assert prof.provenance["reference"] is None and prof.provenance["test_method"] == "custom"
    assert prof.provenance["s3_violations"] == ()


def test_from_frame_validation():
    roles = {"provider_id": "id", "estimate": "est", "flag": "flag"}
    with pytest.raises(ValueError, match="unique"):
        ProviderProfile.from_frame(_frame([1, 1, 2, 3]), roles=roles)
    with pytest.raises(ValueError, match="at least one"):
        ProviderProfile.from_frame(_frame([1, 2, 3, 4]).iloc[:0], roles=roles)
    with pytest.raises(ValueError, match="flag must be"):
        ProviderProfile.from_frame(_frame([1, 2, 3, 4]).assign(flag=[2, 0, 0, 0]), roles=roles)
    with pytest.raises(ValueError, match="numeric"):
        ProviderProfile.from_frame(_frame([1, 2, 3, 4]).assign(est=["a", "b", "c", "d"]), roles=roles)
    with pytest.raises(ValueError, match="unknown roles"):
        ProviderProfile.from_frame(_frame([1, 2, 3, 4]), roles={**roles, "rank": "n"})
    with pytest.raises(ValueError, match="'flag'"):
        ProviderProfile.from_frame(_frame([1, 2, 3, 4]), roles={"estimate": "est"})
    one = ProviderProfile.from_frame(_frame([1, 2, 3, 4]).iloc[:1], roles=roles)
    assert len(one) == 1


def test_from_frame_warns_when_intervals_or_limits_contradict_flags():
    roles = {"provider_id": "id", "estimate": "est", "ci_lower": "lo", "ci_upper": "hi", "flag": "flag",
             "null_value": "null"}
    bad = _frame([1, 2, 3, 4]).assign(flag=[0, -1, 0, 1])             # 1 excludes the null but is not flagged; 4 is
    with pytest.warns(UserWarning, match="S3"):
        prof = ProviderProfile.from_frame(bad, roles=roles)
    assert prof.provenance["s3_violations"] == (1, 4)
    funnel = _frame([1, 2, 3, 4]).assign(fl=0.7, fu=1.05)           # provider 4 (1.1) lies outside but is not flagged
    with pytest.warns(UserWarning, match="S4"):
        prof = ProviderProfile.from_frame(funnel, roles={"provider_id": "id", "estimate": "est", "flag": "flag",
                                                         "funnel_estimate": "est", "funnel_lower": "fl",
                                                         "funnel_upper": "fu"})
    assert prof.provenance["s4_violations"] == (4,) and "funnel_limits" in prof.capabilities


# ------------------------------------------------------------------------------------- immutability, volume
def test_profiles_are_immutable(fe):
    prof = ProviderProfile.from_model(fe)
    with pytest.raises(AttributeError, match="immutable"):
        prof.extra = 1
    copy = prof.data
    copy["estimate"] = 0.0
    assert not (prof.data["estimate"] == 0.0).all()
    with pytest.raises(TypeError):
        prof.provenance["level"] = 0.5


def test_minimum_volume_suppresses_without_changing_flags(fe):
    prof = ProviderProfile.from_model(fe)
    small = prof.with_min_volume(40)
    f, g = prof.data, small.data
    low = g["denominator"] < 40
    assert low.any() and (g.loc[low, "status"] == "suppressed").all()
    assert g.loc[~low, "status"].equals(f.loc[~low, "status"]) and _equal(g["flag"], f["flag"])
    assert small.provenance["min_volume"] == 40.0 and small.provenance["suppressed"] == int(low.sum())
    assert small.status_counts()["suppressed"] == int(low.sum())
    with pytest.raises(CapabilityError, match="denominator"):
        ProviderProfile.from_test(fe.test()).with_min_volume(40)


def test_presentation_imports_no_matplotlib():
    code = "import sys, pprof_py.presentation; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
