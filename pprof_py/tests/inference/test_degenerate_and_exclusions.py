"""Zero-event status, the at_bound guard, excluded-provider records and CoxPH metadata (D4, D13; ADR-005)."""
import numpy as np
import pandas as pd
import pytest

from pprof_py import (CoxPH, LinearFixedEffectModel, LogisticFixedEffectModel, LogisticRandomEffectModel,
                      LogisticThreeStageModel)
from pprof_py.data.glmm_prep import glmm_data_prep
from pprof_py.data.preparation import DataPrep, DataPrepOptions
from pprof_py.inference import at_bound, degenerate_providers


def _logistic(n_providers=40, seed=5):
    rng = np.random.default_rng(seed)
    size = rng.integers(25, 90, n_providers)
    size[:3] = [4, 7, 10]                                           # at most cutoff=10 records: excluded
    pid = np.repeat(np.arange(n_providers), size)
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(rng.normal(-1.0, 0.4, n_providers)[pid] + 0.5 * x))))
    y[pid == 5] = 0                                                 # no events
    y[pid == 6] = 1                                                 # only events
    return pd.DataFrame({"y": y, "x1": x, "provider_id": [f"P{j:02d}" for j in pid]})


def _crossed(seed=11, n_fac=24, n_hosp=6):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fac):
        hosp = rng.choice(n_hosp, rng.integers(1, 3), replace=False)
        rows.append(pd.DataFrame({"fac": f + 1, "hosp": rng.choice(hosp, 6 if f < 2 else rng.integers(40, 100)) + 1}))
    df = pd.concat(rows, ignore_index=True)
    df["x1"] = rng.normal(size=len(df))
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-(-1.0 + 0.4 * df["x1"]))))
    return df


@pytest.fixture(scope="module")
def data():
    return _logistic()


@pytest.fixture(scope="module")
def fe(data):
    m = LogisticFixedEffectModel()
    m.fit(data, y_var="y", x_vars=["x1"], provider_var="provider_id")
    return m


def test_degenerate_providers_fixed_effects(fe, data):
    d = degenerate_providers(fe)
    kept = data.groupby("provider_id")["y"].agg(["sum", "size"]).loc[d.index]
    np.testing.assert_array_equal(d["events"], kept["sum"])
    np.testing.assert_array_equal(d["trials"], kept["size"])
    assert d.loc["P05", "zero_events"] and d.loc["P06", "all_events"]
    np.testing.assert_array_equal(d["finite_estimate"], ~at_bound(fe))
    assert not d.loc["P05", "finite_estimate"] and not d.loc["P06", "finite_estimate"]
    assert list(d.columns) == ["events", "trials", "zero_events", "all_events", "finite_estimate"]


def test_degenerate_providers_random_effects(data):
    m = LogisticRandomEffectModel(verbose=False)
    m.fit(data, y_var="y", x_vars=["x1"], provider_var="provider_id")
    d = degenerate_providers(m)
    assert d["finite_estimate"].all()                               # BLUPs are always finite
    assert bool(d.loc["P05", "zero_events"]) and bool(d.loc["P06", "all_events"])
    assert int(d["trials"].sum()) == len(data)                      # no data preparation in this model


def test_degenerate_providers_three_stage():
    m = LogisticThreeStageModel().fit(_crossed(), "y", ["x1"], "fac", "hosp")
    d = degenerate_providers(m)
    assert len(d) == m.stage3_.n_providers_ and d.index.name == "provider_id"
    np.testing.assert_array_equal(d["finite_estimate"], ~(d["zero_events"] | d["all_events"]))


def test_degenerate_providers_refuses_other_models(data):
    lin = LinearFixedEffectModel()
    lin.fit(data.assign(y=data["y"] + 0.5 * data["x1"]), y_var="y", x_vars=["x1"], provider_var="provider_id")
    with pytest.raises(TypeError, match="binary-outcome"):
        degenerate_providers(lin)
    with pytest.raises(TypeError, match="binary-outcome"):
        degenerate_providers(CoxPH())


def test_at_bound_outside_its_families(data):
    re = LogisticRandomEffectModel(verbose=False)
    re.fit(data, y_var="y", x_vars=["x1"], provider_var="provider_id")
    with pytest.raises(TypeError, match="fixed provider effects"):
        at_bound(re)
    lin = LinearFixedEffectModel()
    lin.fit(data.assign(y=data["y"] + 0.5 * data["x1"]), y_var="y", x_vars=["x1"], provider_var="provider_id")
    with pytest.raises(TypeError, match="continuous outcome"):
        at_bound(lin)


def test_at_bound_unchanged_for_logistic_fixed_effects(fe):
    flagged = set(np.asarray(fe.provider_ids_)[at_bound(fe)])
    assert {"P05", "P06"} <= flagged


def test_dataprep_records_excluded_providers(data):
    prep = DataPrep(data.copy(), Y_char="y", X_char=["x1"], prov_char="provider_id",
                    options=DataPrepOptions(screen_providers=True))
    assert prep.excluded_providers_.empty
    out = prep.data_prep()
    rec = prep.excluded_providers_
    assert list(rec.index) == ["P00", "P01", "P02"] and rec.index.name == "provider_id"
    assert list(rec["n_records"]) == [4, 7, 10] and set(rec["reason"]) == {"at most 10 records"}
    assert not set(rec.index) & set(out["provider_id"])


def test_models_record_excluded_providers(fe, data):
    assert list(fe.excluded_providers_.index) == ["P00", "P01", "P02"]
    off = LogisticFixedEffectModel(use_dataprep=False, screen_providers=False)
    off.fit(data, y_var="y", x_vars=["x1"], provider_var="provider_id")
    assert off.excluded_providers_ is None                         # the model did not prepare the data
    lin = LinearFixedEffectModel()
    lin.fit(data.assign(y=data["y"] + 0.5 * data["x1"]), y_var="y", x_vars=["x1"], provider_var="provider_id")
    assert lin.excluded_providers_ is None


def test_glmm_prep_and_three_stage_record_excluded_providers():
    df = _crossed()
    prep = glmm_data_prep(df, "y", "fac", "hosp", cutoff=10)
    assert list(prep.excluded_providers.index) == [1, 2] and list(prep.excluded_providers["n_records"]) == [6, 6]
    m = LogisticThreeStageModel().fit(df, "y", ["x1"], "fac", "hosp")
    pd.testing.assert_frame_equal(m.excluded_providers_, prep.excluded_providers)


def test_coxph_test_metadata():
    rng = np.random.default_rng(3)
    prov = np.repeat(np.arange(12), 40)
    X = rng.normal(size=(prov.size, 2))
    t = rng.exponential(1.0 / (0.3 * np.exp(X @ [0.5, -0.3])))
    c = rng.uniform(0.5, 3.0, prov.size)
    stop, event = np.minimum(t, c), (t <= c).astype(float)
    m = CoxPH(ties="breslow").fit(X, duration=stop, event=event, strata=prov)
    for method in ("midp", "exact"):
        res = m.test(X, duration=stop, event=event, provider_id=prov, test_method=method)
        assert res.attrs["measure"] == "indirect_ratio" and res.attrs["reference"] == 1.0
