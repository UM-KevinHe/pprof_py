"""Shrinkage: paired estimates equal each test's estimate minus its null value; drawn data equal the pairs."""
import numpy as np
import pandas as pd
import pytest

from pprof_py import LinearRandomEffectModel, LogisticFixedEffectModel, LogisticRandomEffectModel
from pprof_py.presentation import CapabilityError, shrinkage, shrinkage_table
from pprof_py.presentation.data._shrinkage import shrinkage_pairs


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(7)
    n = 40
    size = np.r_[rng.integers(12, 40, 15), rng.integers(40, 300, n - 15)]
    pid = np.repeat(np.arange(n), size)
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(rng.normal(-1.2, 0.45, n)[pid] + 0.5 * x))))
    y[pid == 3] = 0                                                  # no events: no finite fixed effect
    d = pd.DataFrame({"y": y, "x1": x, "provider_id": [f"P{j:02d}" for j in pid]})
    fe, re = LogisticFixedEffectModel(), LogisticRandomEffectModel(verbose=False)
    for m in (fe, re):
        m.fit(d, y_var="y", x_vars=["x1"], provider_var="provider_id")
    return fe, re, d


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def test_pairs_are_the_tests(models):
    fe, re, _ = models
    p = shrinkage_pairs(fe, re)
    tf, tr = fe.test(reference="mean"), re.test()
    np.testing.assert_array_equal(p.frame["fixed"].to_numpy(), (tf["estimate"] - tf["null_value"]).to_numpy())
    np.testing.assert_array_equal(p.frame["random"].to_numpy(),
                                  (tr["estimate"] - tr["null_value"]).reindex(p.frame.index).to_numpy())
    assert (tr["null_value"] == 0).all() and p.provenance["fe_reference"] == "mean"
    assert not p.frame.loc["P03", "fixed_finite"] and p.frame["fixed_finite"].sum() == len(p.frame) - 1
    records = models[2].groupby("provider_id").size()                 # independent of the code: rows per provider
    np.testing.assert_array_equal(p.frame["volume"].to_numpy(), records.reindex(p.frame.index).to_numpy(dtype=float))


def test_figure_draws_the_pairs(models):
    fe, re, _ = models
    r = shrinkage(fe, re)
    p = shrinkage_pairs(fe, re).frame
    fin = p["fixed_finite"].to_numpy()
    (pts,) = _gid(r.figure, "shrinkage-points")
    np.testing.assert_array_equal(np.asarray(pts.get_offsets()), p.loc[fin, ["fixed", "random"]].to_numpy())
    (off,) = _gid(r.figure, "shrinkage-offscale-left")
    np.testing.assert_array_equal(np.asarray(off.get_offsets())[:, 1], p.loc[~fin, "random"].to_numpy())
    assert np.asarray(off.get_offsets())[0, 0] == r.axes.get_xlim()[0]
    (diag,) = _gid(r.figure, "no-shrinkage")
    np.testing.assert_array_equal(diag.get_xdata(), diag.get_ydata())
    assert r.axes.get_xlim() == r.axes.get_ylim()
    assert r.counts["no_finite_fixed"] == 1 and "BLUPs are not the true effects" in r.long_description
    assert "1 provider without a finite fixed-effect estimate (no events or only events) is drawn" in r.long_description
    a, b = shrinkage(fe, re), shrinkage(fe, re)
    for fmt in ("svg", "pdf", "png"):
        assert a.to_bytes(fmt) == b.to_bytes(fmt)


def test_table_and_capabilities(models):
    fe, re, d = models
    t = shrinkage_table(fe, re)
    v = t.to_frame()
    fin = shrinkage_pairs(fe, re).frame["fixed_finite"]
    np.testing.assert_array_equal(v.loc[fin, "change"].to_numpy(), (v.loc[fin, "random"] - v.loc[fin, "fixed"]).to_numpy())
    assert t.spec.cells.loc["P03", "fixed"] == "NE" and t.spec.cells.loc["P03", "change"] == "NE"
    assert list(v.index) == list(fe.test().index.intersection(re.test().index, sort=False))   # source order
    with pytest.raises(CapabilityError, match="same family"):
        lre = LinearRandomEffectModel()
        lre.fit(d.assign(y=d["y"].astype(float)), y_var="y", x_vars=["x1"], provider_var="provider_id")
        shrinkage(fe, lre)
    with pytest.raises(CapabilityError, match="same family"):
        shrinkage(re, fe)                                            # arguments swapped
