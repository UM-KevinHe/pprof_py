"""Several measures: collection, small multiples, agreement and table equal the member profiles (S6: none dropped)."""
import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel
from pprof_py.presentation import (CapabilityError, ProfileCollection, ProviderProfile, measure_agreement, multi_measure,
                                   multi_measure_table)


@pytest.fixture(scope="module")
def col():
    rng = np.random.default_rng(12)
    n = 30
    pid = np.repeat(np.arange(n), rng.integers(40, 250, n))
    x = rng.normal(size=pid.size)
    ids = np.array([f"P{j:02d}" for j in pid])
    g1 = rng.normal(-1.2, 0.5, n)
    g2 = -1.0 + 0.6 * (g1 + 1.2) + rng.normal(0, 0.35, n)
    out = {}
    for label, g, keep in (("A", g1, np.ones(pid.size, bool)), ("B", g2, ~np.isin(ids, ["P05", "P17"]))):
        y = rng.binomial(1, 1 / (1 + np.exp(-(g[pid] + 0.4 * x))))
        m = LogisticFixedEffectModel()
        m.fit(pd.DataFrame({"y": y[keep], "x1": x[keep], "provider_id": ids[keep]}), y_var="y", x_vars=["x1"],
              provider_var="provider_id")
        out[label] = ProviderProfile.from_model(m, test_method="wald")
    return ProfileCollection(out)


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def test_collection(col):
    assert col.labels == ["A", "B"] and len(col.providers()) == 30 and len(col.common()) == 28
    assert set(col.providers().difference(col.common())) == {"P05", "P17"}
    with pytest.raises(ValueError):
        ProfileCollection({})
    with pytest.raises(AttributeError):
        col.x = 1
    with pytest.raises(CapabilityError, match="at least 2"):
        multi_measure(ProfileCollection({"A": col["A"]}))


def test_small_multiples_draw_each_measure(col):
    r = multi_measure(col)
    ids = col.providers()
    ids = ids[np.argsort(-col["A"].data["estimate"].reindex(ids).to_numpy(), kind="stable")]
    assert [t.get_text() for t in r.axes.get_yticklabels()] == [str(i) for i in ids]
    y = np.arange(len(ids) - 1, -1, -1, dtype=float)
    for j, label in enumerate(col.labels):
        f = col[label].data.reindex(ids)
        present = f["status"].notna().to_numpy()
        for key in ("above", "below", "not_different"):
            m = present & (f["status"] == key).to_numpy()
            if m.any():
                (pts,) = _gid(r.figure, f"multi-{key}-{j}")
                np.testing.assert_array_equal(np.asarray(pts.get_offsets()), np.c_[f["estimate"].to_numpy()[m], y[m]])
        (segs,) = _gid(r.figure, f"multi-intervals-{j}")
        s = np.asarray(segs.pprof_bounds if hasattr(segs, "pprof_bounds") else segs.get_segments())   # bars: D79
        np.testing.assert_array_equal(s[:, 0, 0], f["ci_lower"].to_numpy()[present])
        np.testing.assert_array_equal(s[:, 1, 0], f["ci_upper"].to_numpy()[present])
        assert len(_gid(r.figure, f"multi-missing-{j}")) == int((~present).sum())
    assert "not a composite" in r.long_description and "2 providers not in this measure (n/a)" in r.long_description


def test_agreement_counts_the_tests_flags(col):
    r = measure_agreement(col)
    ids = col.common()
    a, b = col["A"].data.loc[ids, "flag"].astype(float), col["B"].data.loc[ids, "flag"].astype(float)
    expected = {"same": int(((a != 0) & (a == b)).sum()), "opposite": int(((a != 0) & (a == -b)).sum()),
                "one": int(((a != 0) != (b != 0)).sum()), "neither": int(((a == 0) & (b == 0)).sum())}
    assert {k: r.counts[k] for k in expected} == expected
    pts = np.concatenate([np.asarray(p.get_offsets()) for k in ("same", "opposite", "one", "neither")
                          for p in _gid(r.figure, f"agreement-{k}")])
    want = np.c_[col["A"].data.loc[ids, "estimate"], col["B"].data.loc[ids, "estimate"]]
    np.testing.assert_array_equal(np.sort(pts, axis=0), np.sort(want, axis=0))
    assert "attenuated by estimation noise" in r.long_description


def test_table_and_determinism(col):
    t = multi_measure_table(col)
    assert t.spec.cells.loc["P05", "e1"] == "\u2014" and t.spec.cells.loc["P05", "f1"] == "\u2014"
    assert "| A: Estimate (CI) | A: Flag | B: Estimate (CI) | B: Flag |" in t.to_markdown()
    np.testing.assert_array_equal(t.to_frame()["A: estimate"].to_numpy(),
                                  col["A"].data["estimate"].reindex(col.providers()).to_numpy())
    for make in (lambda: multi_measure(col), lambda: measure_agreement(col)):
        x, y = make(), make()
        for fmt in ("svg", "pdf", "png"):
            assert x.to_bytes(fmt) == y.to_bytes(fmt)


def test_agreement_does_not_place_providers_without_a_finite_estimate():
    from pprof_py.presentation._synthetic import provider_data

    d = provider_data(60)
    zero = d.attrs["planted"]["zero_events"]
    fits = {}
    for label, outcome in (("A", "y"), ("B", "y2")):
        m = LogisticFixedEffectModel()
        m.fit(d, y_var=outcome, x_vars=["x1"], provider_var="provider_id")
        fits[label] = ProviderProfile.from_model(m, test_method="wald")
    col = ProfileCollection(fits)
    r = measure_agreement(col)
    placed = np.concatenate([np.asarray(p.get_offsets()) for k in ("same", "opposite", "one", "neither", "untested")
                             for p in _gid(r.figure, f"agreement-{k}")])
    zero_in = [z for z in zero if z in col.common()]
    assert zero_in and r.counts["not_placed"] >= len(zero_in) and len(placed) == len(col.common()) - r.counts["not_placed"]
    est = col["A"].data.loc[zero_in, "estimate"].to_numpy()
    assert not np.isin(placed[:, 0], est).any()                                 # the bound estimates are not drawn
    lo, hi = r.axes.get_xlim()
    assert hi - lo < 10 and "without a finite estimate" in r.long_description
