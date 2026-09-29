"""CoxPH.calculate_standardized_measures: the indirect and direct standardized ratios of the SMR tutorial
(Section 3), against R's survival (basehaz of the stratified stage-1 and offset-only stage-2 models), the
definitions integrated directly, and survival Chapter 4's hand-built two-stage pipeline."""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from pprof_py import CoxPH

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "data" / "cox_smr"


@pytest.fixture(scope="module")
def golden():
    d = pd.read_csv(GOLDEN / "raw.csv")
    return d, json.loads((GOLDEN / "r_smr.json").read_text())


def _measures(model, d, **kw):
    X = d[["x1", "x2", "x3"]]
    return model.calculate_standardized_measures(X, start=d["start"], stop=d["stop"], event=d["event"],
                                                 provider_id=d["prov"], **kw)


def test_matches_r_two_stage_and_stratified_baselines(golden):
    d, r = golden
    m = CoxPH(ties="breslow").fit(d[["x1", "x2", "x3"]], start=d["start"], stop=d["stop"], event=d["event"],
                                  strata=d["prov"])
    res = _measures(m, d, stdz=["indirect", "direct"])
    ind, dr = res["indirect"], res["direct"]
    assert ind["provider_id"].tolist() == r["providers"]
    assert np.array_equal(ind["observed"], r["observed"])
    assert np.allclose(ind["expected"], r["indirect_expected"], rtol=1e-10, atol=0)
    assert np.allclose(ind["indirect_ratio"], np.array(r["observed"]) / np.array(r["indirect_expected"]), rtol=1e-10)
    assert np.allclose(dr["expected"], r["direct_expected"], rtol=1e-10, atol=0)
    assert np.allclose(dr["direct_ratio"], np.array(r["direct_expected"]) / r["total_observed"], rtol=1e-10)
    assert ind["expected"].sum() == pytest.approx(r["total_observed"], rel=1e-12)     # sum_j E_j = O


def _data(seed=4, n=400, K=8):
    rng = np.random.default_rng(seed)
    prov = rng.integers(0, K, n)
    X = rng.normal(size=(n, 2))
    entry = np.round(rng.uniform(0, 1, n) * (rng.uniform(size=n) < 0.3), 1)
    t = entry + np.round(rng.exponential(size=n) * np.exp(-(X @ [0.4, -0.3] + rng.normal(0, 0.4, K)[prov])), 1) + 0.1
    c = entry + np.round(rng.exponential(2.0, n), 1) + 0.1
    return X, entry, np.minimum(t, c), (t <= c).astype(float), prov


def test_direct_expected_is_the_integral_of_each_providers_baseline():
    X, entry, stop, event, prov = _data()
    m = CoxPH(ties="breslow").fit(X, start=entry, stop=stop, event=event, strata=prov)
    dr = m.calculate_standardized_measures(X, start=entry, stop=stop, event=event, provider_id=prov, stdz="direct")["direct"]
    risk = np.exp(X @ m.coef_)
    for j in range(prov.max() + 1):
        rows = prov == j
        times = np.unique(stop[rows & (event == 1)])
        inc = [np.sum(event[rows & (stop == u)]) / np.sum(risk[rows & (entry < u) & (stop >= u)]) for u in times]
        cum = np.concatenate([[0.0], np.cumsum(inc)])
        at = lambda t: cum[np.searchsorted(times, t, side="right")]                   # noqa: E731
        assert dr["expected"][j] == pytest.approx(np.sum(risk * (at(stop) - at(entry))), rel=1e-12)


def test_indirect_matches_the_chapter_4_pipeline():
    X, _entry, stop, event, prov = _data()
    stage1 = CoxPH(ties="breslow").fit(X, duration=stop, event=event, strata=prov)
    xbeta = stage1.predict_linear(X)
    stage2 = CoxPH(ties="breslow").fit(pd.DataFrame(index=range(len(stop))), duration=stop, event=event, offset=xbeta)
    bh = stage2.baseline_hazard_
    k = np.searchsorted(bh["time"].to_numpy(), stop, side="right") - 1
    base = np.where(k >= 0, bh["hazard"].to_numpy()[np.clip(k, 0, None)], 0.0) / np.exp(np.mean(xbeta))
    manual = pd.Series(np.exp(xbeta) * base).groupby(prov).sum().to_numpy()
    ind = stage1.calculate_standardized_measures(X, duration=stop, event=event, provider_id=prov)["indirect"]
    assert np.allclose(ind["expected"], manual, rtol=1e-12, atol=0)


def test_single_provider_and_pooled_fits():
    X, entry, stop, event, prov = _data()
    one = np.zeros(len(stop))
    m = CoxPH(ties="breslow").fit(X, start=entry, stop=stop, event=event)
    res = m.calculate_standardized_measures(X, start=entry, stop=stop, event=event, provider_id=one,
                                            stdz=["indirect", "direct"])
    assert res["indirect"]["indirect_ratio"][0] == pytest.approx(1.0, rel=1e-12)
    assert res["direct"]["direct_ratio"][0] == pytest.approx(1.0, rel=1e-12)
    pooled = m.calculate_standardized_measures(X, start=entry, stop=stop, event=event, provider_id=prov)["indirect"]
    assert pooled["expected"].sum() == pytest.approx(event.sum(), rel=1e-12)
    assert np.allclose(pooled["person_time"], pd.Series(stop - entry).groupby(prov).sum())


def test_arguments():
    X, entry, stop, event, prov = _data()
    m = CoxPH().fit(X, start=entry, stop=stop, event=event, strata=prov)
    sub = m.calculate_standardized_measures(X, start=entry, stop=stop, event=event, provider_id=prov,
                                            providers=[2, 5], stdz="direct")["direct"]
    assert sub["provider_id"].tolist() == [2, 5]
    with pytest.raises(ValueError, match="stdz"):
        m.calculate_standardized_measures(X, start=entry, stop=stop, event=event, provider_id=prov, stdz="both")
    with pytest.raises(ValueError, match="one entry per row"):
        m.calculate_standardized_measures(X, start=entry, stop=stop, event=event, provider_id=prov[:-1])
