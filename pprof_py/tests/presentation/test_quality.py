"""Data-quality panel and table: every provider accounted for; unrecorded counts never shown as zero (S6, S11)."""
import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel
from pprof_py.data.preparation import exclusion_record
from pprof_py.inference import degenerate_providers
from pprof_py.presentation import ProviderProfile, data_quality, data_quality_table
from pprof_py.presentation.data._quality import quality_accounting, quality_groups


def _logistic(n_providers=40, seed=5):
    rng = np.random.default_rng(seed)
    size = rng.integers(15, 90, n_providers)
    size[:2] = [4, 7]                                               # excluded by data preparation
    pid = np.repeat(np.arange(n_providers), size)
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(rng.normal(-1.0, 0.5, n_providers)[pid] + 0.5 * x))))
    y[pid == 6] = 0                                                 # no events
    return pd.DataFrame({"y": y, "x1": x, "provider_id": [f"P{j:02d}" for j in pid]})


@pytest.fixture(scope="module")
def profile():
    fe = LogisticFixedEffectModel()
    fe.fit(_logistic(), y_var="y", x_vars=["x1"], provider_var="provider_id")
    return ProviderProfile.from_model(fe).with_min_volume(25), fe


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def test_accounting_counts_every_provider(profile):
    prof, fe = profile
    rows = {key: count for _, count, key in quality_accounting(prof)}
    c = prof.status_counts()
    assert rows["excluded"] == len(fe.excluded_providers_) == 2 and rows["total"] == len(prof) + 2
    assert rows["above"] + rows["below"] + rows["not_different"] + rows["not_tested"] + rows["suppressed"] == len(prof)
    degenerate = int((~degenerate_providers(fe)["finite_estimate"]).sum())
    assert rows["no_finite_estimate"] == c["no_finite_estimate"] == degenerate == 2
    assert rows["suppressed"] == c["suppressed"] > 0
    groups = quality_groups(prof)
    assert len(groups) == len(prof) and set(groups) <= {"Analysed", "No finite estimate", "Not tested", "Suppressed"}


def test_figure_draws_the_accounting_and_the_volumes(profile):
    prof, fe = profile
    r = data_quality(prof)
    for _, count, key in quality_accounting(prof):
        (bar,) = [p for a in _gid(r.figure, f"accounting-{key}") for p in getattr(a, "patches", [a])]
        assert bar.get_width() == count
    groups, f = quality_groups(prof), prof.data
    for name in set(groups):
        (pts,) = _gid(r.figure, f"volume-{name.lower().replace(' ', '_')}")
        np.testing.assert_array_equal(np.sort(np.asarray(pts.get_offsets())[:, 0]),
                                      np.sort(f.loc[groups == name, "denominator"].to_numpy()))
    status = f["status"].astype(str)
    nofinite = ~f["finite_estimate"].astype(bool) & ~status.isin(["suppressed", "not_tested"])   # independent of the code
    assert nofinite.sum() == 1
    (ne,) = _gid(r.figure, "volume-no_finite_estimate")
    np.testing.assert_array_equal(np.asarray(ne.get_offsets())[:, 0], f.loc[nofinite, "denominator"].to_numpy())
    (ex,) = _gid(r.figure, "volume-excluded")
    np.testing.assert_array_equal(np.sort(np.asarray(ex.get_offsets())[:, 0]),
                                  np.sort(fe.excluded_providers_["n_records"].to_numpy(dtype=float)))
    (line,) = _gid(r.figure, "minimum-volume")
    assert np.all(np.asarray(line.get_xdata()) == 25.0)
    assert r.alt_text.startswith(f"Data-quality summary: {len(prof) + 2} providers in the data, 2 excluded")
    assert r.kind == "data_quality" and r.counts["excluded"] == 2


def test_unrecorded_exclusions_and_missing_denominators(profile):
    prof, fe = profile
    bare = ProviderProfile.from_test(fe.test())                     # no model: no denominators, exclusions unknown
    rows = {key: count for _, count, key in quality_accounting(bare)}
    assert rows["excluded"] is None and rows["total"] is None
    r = data_quality(bare)
    assert [t for t in r.figure.findobj() if getattr(t, "get_text", lambda: "")().strip() == "not recorded"]
    assert "were not recorded" in r.long_description and "volumes are not shown" in r.long_description
    t = data_quality_table(bare)
    assert t.spec.cells.loc["excluded", "count"] == "not recorded" and t.spec.cells.loc["total", "share"] == "\u2014"


def test_excluded_providers_stay_off_an_axis_in_other_units():
    df = pd.DataFrame({"id": ["A", "B", "C"], "est": [1.0, 1.2, 0.8], "lo": [0.8, 1.05, 0.6], "hi": [1.2, 1.4, 1.0],
                       "flag": [0, 1, 0], "nv": 1.0, "e": [10.0, 30.0, 20.0]})
    prof = ProviderProfile.from_frame(df, roles={"provider_id": "id", "estimate": "est", "ci_lower": "lo",
                                                 "ci_upper": "hi", "flag": "flag", "null_value": "nv",
                                                 "denominator": "e"},
                                      provenance={"denominator_kind": "expected"},
                                      excluded=exclusion_record(["Z"], [3], "at most 10 records"))
    r = data_quality(prof)
    assert not _gid(r.figure, "volume-excluded") and "counted but not placed" in r.long_description
    assert r.counts["excluded"] == 1


def test_tables(profile):
    prof, fe = profile
    t = data_quality_table(prof)
    cells = t.spec.cells
    assert cells.loc["total", "count"] == str(len(prof) + 2) and cells.loc["excluded", "share"] == f"{200 / (len(prof) + 2):.1f}%"
    assert list(t.to_frame()["count"]) == [float(c) for _, c, _ in quality_accounting(prof)]
    d = data_quality_table(prof, details=True)
    issues = dict(zip(d.spec.cells["provider"], d.spec.cells["issue"]))
    assert issues["P00"] == "excluded: at most 10 records" and issues["P06"] == "no finite estimate; zero events"
    small = prof.data.index[prof.data["status"] == "suppressed"]
    assert all(issues[p].startswith("suppressed") for p in small)
    assert issues["P02"] == "suppressed; no finite estimate; zero events"   # small and without events


def test_export_is_deterministic(profile):
    prof, _ = profile
    a, b = data_quality(prof), data_quality(prof)
    for fmt in ("svg", "pdf", "png"):
        assert a.to_bytes(fmt) == b.to_bytes(fmt) == a.to_bytes(fmt)
