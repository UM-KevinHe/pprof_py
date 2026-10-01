"""Null calibration: histograms and null densities from test() outputs; flags compared, never re-decided (S9)."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pprof_py import LogisticFixedEffectModel
from pprof_py.inference import EmpiricalNull
from pprof_py.presentation import ProviderProfile, null_calibration, null_calibration_table


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(11)
    n = 240
    size = rng.integers(20, 160, n)
    gamma = rng.normal(-1.3, 0.45, n)                                # overdispersed relative to the model
    pid = np.repeat(np.arange(n), size)
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(gamma[pid] + 0.5 * x))))
    fe = LogisticFixedEffectModel()
    fe.fit(pd.DataFrame({"y": y, "x1": x, "provider_id": pid}), y_var="y", x_vars=["x1"], provider_var="provider_id")
    return fe, EmpiricalNull.fitter(size=fe.provider_sizes_, n_groups=2)


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def test_histograms_and_densities_come_from_the_test(setup):
    fe, emp = setup
    r = null_calibration(fe, test_method="score", null_model=emp)
    res = fe.test(test_method="score", null_model=emp)
    for g in sorted(res["null_group"].dropna().unique()):
        z = res.loc[res["null_group"] == g, "z_raw"].to_numpy()
        (hist,) = _gid(r.figure, f"calibration-histogram-{g}")
        values, edges = hist.get_data()[:2]
        np.testing.assert_array_equal(values, np.histogram(z, bins=edges)[0])
        (curve,) = _gid(r.figure, f"calibration-fitted-{g}")
        xg, yg = curve.get_xdata(), curve.get_ydata()
        mean, sd = res.loc[res["null_group"] == g, ["null_mean", "null_sd"]].iloc[0]
        assert abs(xg[np.argmax(yg)] - mean) <= (xg[1] - xg[0])
        width = edges[1] - edges[0]
        np.testing.assert_allclose(yg, z.size * width * norm.pdf(xg, mean, sd), rtol=1e-12)


def test_flags_are_compared_not_recomputed(setup):
    fe, emp = setup
    r = null_calibration(fe, test_method="score", null_model=emp)
    fitted, theo = fe.test(test_method="score", null_model=emp), fe.test(test_method="score")
    changed = int((fitted["flag"].fillna(0).to_numpy() != theo["flag"].fillna(0).to_numpy()).sum())
    assert r.counts["flagged_fitted"] == int((fitted["flag"] != 0).sum())
    assert r.counts["flagged_theoretical"] == int((theo["flag"] != 0).sum()) and r.counts["changed"] == changed
    assert f"{changed} flags change" in r.alt_text and "not necessarily many outlying providers" in r.long_description
    t = null_calibration_table(fe, test_method="score", null_model=emp)
    v = t.to_frame()
    assert int(v["changed"].sum()) == changed and int(v["providers"].sum()) == len(fitted)
    groups = fitted.groupby("null_group")
    np.testing.assert_array_equal(v["null_sd"].to_numpy(), groups["null_sd"].first().to_numpy())
    html = t.to_html()
    assert 'colspan="2" class="center">Fitted null</th>' in html and 'colspan="3" class="center">Flagged</th>' in html


def test_theoretical_null_and_profile_sources(setup):
    fe, _ = setup
    r = null_calibration(fe, test_method="score")
    assert not _gid(r.figure, "calibration-fitted-all") and "nothing to compare" in r.long_description
    assert "theoretical null N(0, 1)" in r.alt_text
    prof = ProviderProfile.from_model(fe, test_method="score", null_model=EmpiricalNull.fitter())
    rp = null_calibration(prof)
    assert "cannot be tested again" in rp.long_description and "changed" not in rp.counts
    assert "Changed" not in [c.header for c in null_calibration_table(prof).spec.columns]


def test_exact_tests_carry_the_converted_statistic_caveat(setup):
    fe, _ = setup
    r = null_calibration(fe, test_method="poibin_exact", null_model=EmpiricalNull.fitter())
    assert "converted statistic" in r.long_description


def test_export_is_deterministic(setup):
    fe, emp = setup
    a, b = null_calibration(fe, test_method="score", null_model=emp), null_calibration(fe, test_method="score",
                                                                                      null_model=emp)
    for fmt in ("svg", "pdf", "png"):
        assert a.to_bytes(fmt) == b.to_bytes(fmt) == a.to_bytes(fmt)
