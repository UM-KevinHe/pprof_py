"""Flag stability: every flag comes from a test() call per scenario; the display only lines them up (S9)."""
import numpy as np
import pandas as pd
import pytest

from pprof_py import CoxPH, LogisticFixedEffectModel
from pprof_py.inference import EmpiricalNull
from pprof_py.presentation import flag_stability, flag_stability_table
from pprof_py.presentation.data._stability import changing, flag_scenarios


@pytest.fixture(scope="module")
def fe():
    rng = np.random.default_rng(11)
    n = 50
    pid = np.repeat(np.arange(n), rng.integers(30, 150, n))
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(rng.normal(-1.2, 0.5, n)[pid] + 0.5 * x))))
    m = LogisticFixedEffectModel()
    m.fit(pd.DataFrame({"y": y, "x1": x, "provider_id": [f"P{j:02d}" for j in pid]}), y_var="y", x_vars=["x1"],
          provider_var="provider_id")
    return m


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def test_default_scenarios_are_separate_tests(fe):
    sc = flag_scenarios(fe, (), {"test_method": "score"})
    assert list(sc.flags.columns) == ["Base", "Reference: mean", "Empirical null"]
    expected = {"Base": fe.test(test_method="score"), "Reference: mean": fe.test(test_method="score", reference="mean"),
                "Empirical null": fe.test(test_method="score", null_model=EmpiricalNull.fitter())}
    for label, res in expected.items():
        pd.testing.assert_series_equal(sc.flags[label], res["flag"].astype("Int64"), check_names=False)
    assert "median provider effect" in sc.descriptions["Base"]
    assert "size-weighted mean provider effect" in sc.descriptions["Reference: mean"]
    assert list(flag_scenarios(fe, (), {"reference": "mean"}).flags.columns)[1] == "Reference: median"
    custom = flag_scenarios(fe, (), {"test_method": "score"}, {"Wald": {"test_method": "wald"}})
    pd.testing.assert_series_equal(custom.flags["Wald"], fe.test(test_method="wald")["flag"].astype("Int64"),
                                   check_names=False)


def test_coxph_has_no_reference_scenario():
    rng = np.random.default_rng(29)
    prov = np.repeat(np.arange(25), 60)
    X = rng.normal(size=(prov.size, 2))
    t = rng.exponential(1 / (0.3 * np.exp(X @ [0.5, -0.3])))
    c = rng.uniform(0.5, 3, prov.size)
    data = dict(duration=np.minimum(t, c), event=(t <= c).astype(float), provider_id=prov)
    cox = CoxPH(ties="breslow").fit(X, duration=data["duration"], event=data["event"], strata=prov)
    assert list(flag_scenarios(cox, (X,), data).flags.columns) == ["Base", "Empirical null"]


def test_figure_draws_the_matrix(fe):
    r = flag_stability(fe, test_method="score")
    sc = flag_scenarios(fe, (), {"test_method": "score"})
    f = sc.flags
    flagged = ((f == 1) | (f == -1)).fillna(False).to_numpy(dtype=bool).any(axis=1)
    rows = f.index[flagged][np.argsort(-sc.estimate[flagged].to_numpy(), kind="stable")]
    assert [t.get_text() for t in r.axes.get_yticklabels()] == [str(p) for p in rows]
    vals = f.loc[rows].to_numpy(dtype=float, na_value=np.nan)
    y = np.arange(len(rows) - 1, -1, -1, dtype=float)
    for value, key in ((1, "above"), (-1, "below"), (0, "not_different")):
        ii, jj = np.nonzero(vals == value)
        if ii.size:
            (pts,) = _gid(r.figure, f"stability-{key}")
            np.testing.assert_array_equal(np.asarray(pts.get_offsets()), np.c_[jj.astype(float), y[ii]])
    ch = changing(sc)[flagged][np.argsort(-sc.estimate[flagged].to_numpy(), kind="stable")]
    (marks,) = _gid(r.figure, "stability-changes")
    np.testing.assert_array_equal(np.asarray(marks.get_offsets())[:, 1], y[ch])
    assert r.counts["changing"] == int(changing(sc).sum()) and "does not make a flag robust" in r.long_description
    a, b = flag_stability(fe, test_method="score"), flag_stability(fe, test_method="score")
    for fmt in ("svg", "pdf", "png"):
        assert a.to_bytes(fmt) == b.to_bytes(fmt)


def test_tables(fe):
    sc = flag_scenarios(fe, (), {"test_method": "score"})
    v = flag_stability_table(fe, test_method="score").to_frame()
    base = sc.flags["Base"]
    for label in sc.flags.columns:
        assert v.loc[label, "above"] == (sc.flags[label] == 1).sum() and v.loc[label, "below"] == (sc.flags[label] == -1).sum()
        if label != "Base":
            assert v.loc[label, "changed"] == int((sc.flags[label].fillna(9) != base.fillna(9)).sum())
    d = flag_stability_table(fe, test_method="score", details=True)
    assert list(d.to_frame().index) == list(sc.flags.index[changing(sc)])        # source order, no ranking


def test_change_detection_counts_not_tested_as_a_status():
    from pprof_py.presentation.data._stability import FlagScenarios, stability_summary

    flags = pd.DataFrame({"Base": [0, 0, 1, pd.NA, 1], "Alt": [pd.NA, 0, 1, pd.NA, -1]}, index=list("abcde"),
                         dtype="Int64")
    sc = FlagScenarios(flags=flags, estimate=pd.Series(0.0, index=flags.index), descriptions={}, model="m")
    assert list(changing(sc)) == [True, False, False, False, True]           # 0 -> NT and 1 -> -1 change
    assert stability_summary(sc).loc["Alt", "changed"] == 2 and stability_summary(sc).loc["Alt", "not_tested"] == 2
