"""Data preparation: R's glmm.data.prep port and DataPrep's screening and binomial trials."""
import json
import logging
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel
from pprof_py.data import DataPrep, DataPrepOptions, glmm_data_prep

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "data" / "glmm_prep"


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(GOLDEN / "raw.csv")


@pytest.fixture(scope="module")
def prep(raw):
    return glmm_data_prep(raw, "Y", "fac", "hosp", cutoff=10)


def test_matches_r_glmm_data_prep(prep):
    r = pd.read_csv(GOLDEN / "prep_R.csv")
    meta = json.loads((GOLDEN / "prep_R.json").read_text())
    d = prep.data
    assert np.array_equal(d["rid"].to_numpy(), r["rid"].to_numpy())                  # same rows, same order
    assert np.array_equal(d["fac"].astype(int).to_numpy(), r["fac"].to_numpy())
    assert np.array_equal(d["hosp"].astype(int).to_numpy(), r["hosp"].to_numpy())
    assert np.allclose(d["y_adj"].to_numpy(), r["Y.adj"].to_numpy(), rtol=0, atol=1e-12)
    assert np.array_equal(d["provider_size"].to_numpy(), r["fac.size"].to_numpy())
    assert np.array_equal(d["cell_id"].to_numpy(), r["prov_ID"].to_numpy())
    assert np.array_equal(d["included"].to_numpy(), r["included"].to_numpy())
    assert np.array_equal(prep.cell_sizes, np.asarray(meta["n_fac_hosp"]))              # cluster-major, as R
    assert (prep.n_providers, prep.n_clusters) == (meta["fac"][0] if isinstance(meta["fac"], list) else meta["fac"],
                                                   meta["hosp"][0] if isinstance(meta["hosp"], list) else meta["hosp"])


def test_keeps_providers_with_more_than_cutoff_records(raw, prep):
    sizes = raw.groupby("fac").size()
    assert (sizes == 10).any() and (sizes == 11).any()
    assert set(prep.data["fac"].astype(int)) == set(sizes.index[sizes > 10])


def test_adjusts_providers_with_no_or_all_events(prep):
    d = prep.data
    events = d.groupby("fac", observed=True)["Y"].transform("sum")
    none, every = events == 0, events == d["provider_size"]
    assert none.any() and every.any()
    assert np.allclose(d.loc[none, "y_adj"], 0.01 / d.loc[none, "provider_size"])
    assert np.allclose(d.loc[every, "y_adj"], 1 - 0.01 / d.loc[every, "provider_size"])
    assert np.array_equal(d.loc[~none & ~every, "y_adj"], d.loc[~none & ~every, "Y"].astype(float))


def test_rejects_a_non_binary_outcome(raw):
    with pytest.raises(ValueError, match="binary"):
        glmm_data_prep(raw.assign(Y=raw["Y"] * 2), "Y", "fac", "hosp")


def test_dataprep_keeps_providers_with_more_than_cutoff_records():
    df = pd.DataFrame({"prov": np.repeat(["a", "b", "c"], [10, 11, 30]), "x": np.arange(51.0) % 7,
                       "y": np.tile([0, 1], 26)[:51]})
    out = DataPrep(df, "y", ["x"], "prov", options=DataPrepOptions(screen_providers=True, cutoff=10),
                   check=False, logging=logging).data_prep()
    assert sorted(out["prov"].unique()) == ["b", "c"]


def test_binomial_fit_through_dataprep_keeps_trials_aligned():
    rng = np.random.default_rng(4)
    prov = rng.permutation(np.repeat(np.arange(12), 15))                  # rows deliberately not sorted by provider
    n = rng.integers(1, 9, prov.size)
    x = rng.normal(size=prov.size)
    y = rng.binomial(n, 1 / (1 + np.exp(-(-0.5 + 0.7 * x + rng.normal(0, 0.4, 12)[prov]))))
    df = pd.DataFrame({"x": x, "y": y.astype(float), "n": n.astype(float), "prov": prov})
    fits = []
    for use in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticFixedEffectModel(use_dataprep=use, screen_providers=False)
            fits.append(m.fit(df, y_var="y", x_vars=["x"], provider_var="prov", n_var="n"))
    assert np.allclose(np.ravel(fits[0].coefficients_["beta"]), np.ravel(fits[1].coefficients_["beta"]), atol=1e-8)
    assert np.allclose(np.ravel(fits[0].coefficients_["gamma"]), np.ravel(fits[1].coefficients_["gamma"]), atol=1e-8)
    with pytest.raises(ValueError, match="0 <= y <= n"):
        LogisticFixedEffectModel(use_dataprep=True).fit(df.assign(y=df["n"] + 1), y_var="y", x_vars=["x"],
                                                        provider_var="prov", n_var="n")
