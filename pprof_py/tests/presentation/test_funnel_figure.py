"""funnel(): drawn artists equal the profile; accessibility, layout, deterministic export and errors (spec §2.1, §7, §9)."""
import hashlib
import subprocess
import sys
import textwrap
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.legend import Legend
from matplotlib.text import Text

from pprof_py import CoxPH, LinearFixedEffectModel, LinearRandomEffectModel, LogisticFixedEffectModel
from pprof_py.presentation import CapabilityError, FigureResult, ProviderProfile, funnel


def _logistic(n_providers=40, seed=5):
    rng = np.random.default_rng(seed)
    size = rng.integers(25, 90, n_providers)
    size[:2] = [4, 7]
    gamma = rng.normal(-1.0, 0.4, n_providers)
    gamma[2:6] += [1.0, -1.0, 0.8, -0.8]
    pid = np.repeat(np.arange(n_providers), size)
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(gamma[pid] + 0.5 * x))))
    y[pid == 6] = 0
    return pd.DataFrame({"y": y, "x1": x, "provider_id": [f"P{j:02d}" for j in pid]})


def _synthetic(n, *, seed=0, scale="ratio", n_zero=0, n_nofinite=0, n_nan=0):
    """A funnel profile from a plain frame (no model): limits 1 +/- 1.96/sqrt(E), flags consistent with them."""
    rng = np.random.default_rng(seed)
    e = rng.lognormal(np.log(20.0), 0.8, n)
    se = 1.0 / np.sqrt(e)
    null = 1.0 if scale == "ratio" else -2.0
    est = null + rng.normal(0.0, 1.2, n) * se
    if scale == "ratio":
        est = np.maximum(est, 0.0)
        est[:n_zero] = 0.0
    finite = np.ones(n, dtype=bool)
    finite[n_zero:n_zero + n_nofinite] = False
    est[n_zero:n_zero + n_nofinite] = null - 50.0                  # a solver clamp, never drawn as an estimate
    lo, hi = null - 1.96 * se, null + 1.96 * se
    flag = np.where(est > hi, 1, np.where(est < lo, -1, 0))
    prec = e.copy()
    prec[n - n_nan:] = np.nan
    df = pd.DataFrame({"id": [f"H{i:05d}" for i in range(n)], "est": est, "prec": prec, "lo": lo, "hi": hi,
                       "flag": flag, "nv": null, "zero": est == 0.0, "fin": finite})
    roles = {"provider_id": "id", "estimate": "est", "flag": "flag", "null_value": "nv", "funnel_estimate": "est",
             "funnel_precision": "prec", "funnel_lower": "lo", "funnel_upper": "hi", "zero_events": "zero",
             "finite_estimate": "fin"}
    prov = {"scale": scale, "level": 0.95, "alternative": "two_sided", "test_method": "custom",
            "estimator": "custom estimates"}
    return ProviderProfile.from_frame(df, roles=roles, provenance=prov)


@pytest.fixture(scope="module")
def fe():
    m = LogisticFixedEffectModel()
    m.fit(_logistic(), y_var="y", x_vars=["x1"], provider_var="provider_id")
    return m


@pytest.fixture(scope="module")
def profiles(fe):
    return {"score": ProviderProfile.from_model(fe, limits=True, levels=(0.95, 0.998)),
            "exact": ProviderProfile.from_model(fe, limits=True, test_method="poibin_exact")}


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def _drawn(result):
    FigureCanvasAgg(result.figure)
    result.figure.canvas.draw()
    return result.figure.canvas.get_renderer()


# --------------------------------------------------------------------------------- drawn data equal the profile
@pytest.mark.parametrize("which", ["score", "exact"])
def test_points_are_the_profile(profiles, which):
    prof = profiles[which]
    r, f = funnel(prof), prof.data
    for key in ("above", "below", "not_different", "not_tested"):
        sel = (f["status"] == key).to_numpy()
        arts = _gid(r.figure, f"status-{key}")
        if not sel.any():
            assert not arts
            continue
        (coll,) = arts
        np.testing.assert_array_equal(np.asarray(coll.get_offsets()),
                                      np.c_[f["funnel_precision"].to_numpy()[sel], f["funnel_estimate"].to_numpy()[sel]])
    zero = f["zero_events"].fillna(False).to_numpy(dtype=bool)
    (coll,) = _gid(r.figure, "zero-events")
    assert len(coll.get_offsets()) == zero.sum() == r.counts["zero_events"]
    assert np.all(np.asarray(coll.get_offsets())[:, 1] == 0.0)
    (ref,) = _gid(r.figure, "reference")
    assert np.all(np.asarray(ref.get_ydata()) == 1.0)


def test_curves_and_marks_are_the_limits(profiles):
    prof = profiles["score"]
    r, c = funnel(prof), prof.funnel_curves
    for lev in c["level"].unique():
        cg = c[c["level"] == lev]
        for side in ("upper", "lower"):
            (line,) = _gid(r.figure, f"limit-curve-{float(lev):g}--{side}")
            np.testing.assert_array_equal(line.get_xdata(), cg["precision"].to_numpy())
            np.testing.assert_array_equal(line.get_ydata(), cg[side].to_numpy())
    assert not _gid(r.figure, "limit-marks-upper")
    prof = profiles["exact"]
    r, f = funnel(prof), prof.data
    for side, col in (("lower", "funnel_lower"), ("upper", "funnel_upper")):
        (coll,) = _gid(r.figure, f"limit-marks-{side}")
        v = f[col].to_numpy()
        m = np.isfinite(v)
        np.testing.assert_array_equal(np.asarray(coll.get_offsets()), np.c_[f["funnel_precision"].to_numpy()[m], v[m]])


def test_legend_reports_counts(profiles):
    prof = profiles["exact"]
    r, c = funnel(prof), prof.status_counts()
    (legend,) = r.figure.findobj(Legend)
    texts = [t.get_text() for t in legend.get_texts()]
    assert f"Above reference ({c['above']})" in texts and f"Below reference ({c['below']})" in texts
    assert f"Not different ({c['not_different']})" in texts
    assert f"Zero events, O/E = 0 ({c['zero_events']})" in texts and "Exact 95% limits, each provider" in texts


# ------------------------------------------------------------------------------- accessibility and layout
def test_text_is_at_least_7pt_and_rows_do_not_overlap(profiles):
    r = funnel(profiles["exact"])
    rend = _drawn(r)
    texts = [t for t in r.figure.findobj(Text) if t.get_visible() and t.get_text().strip()]
    assert min(t.get_fontsize() for t in texts) >= 7.0
    width = r.figure.bbox.width
    note = [t for t in texts if t.get_text().startswith(f"{len(profiles['exact'])} providers")]
    assert len(note) == 1
    nb, ab = note[0].get_window_extent(rend), r.axes.get_tightbbox(rend)
    assert nb.x0 >= 0 and nb.x1 <= width + 0.5 and nb.y1 <= ab.y0 + 0.5
    assert r.axes.get_window_extent(rend).width >= 0.5 * width


def test_alt_text_and_export_metadata(profiles):
    prof = profiles["score"]
    r, c = funnel(prof), prof.status_counts()
    assert r.alt_text.startswith(f"Funnel plot of {len(prof)} providers")
    assert f"{c['above']} above and {c['below']} below the reference at the 95% level" in r.alt_text
    assert r.long_description.startswith(r.alt_text) and "score test" in r.long_description
    svg = r.to_svg()
    assert f'<title id="pprof_py-title">{escape(r.alt_text)}</title>' in svg and '<desc id="pprof_py-desc">' in svg
    assert 'role="img"' in svg and "<dc:date>" not in svg
    assert b"/Title" in r.to_bytes("pdf") and b"Title" in r.to_bytes("png")


def test_export_is_deterministic(profiles):
    a, b = funnel(profiles["exact"]), funnel(profiles["exact"])
    for fmt in ("svg", "pdf", "png"):
        assert a.to_bytes(fmt) == b.to_bytes(fmt) == a.to_bytes(fmt)


def test_export_is_deterministic_across_processes_without_pyplot():
    script = textwrap.dedent("""
        import hashlib, sys, warnings
        warnings.simplefilter("ignore")
        sys.path.insert(0, %r)
        from test_funnel_figure import _synthetic
        from pprof_py.presentation import funnel
        r = funnel(_synthetic(300, n_zero=4))
        print(" ".join(hashlib.sha256(r.to_bytes(f)).hexdigest()[:16] for f in ("svg", "pdf", "png")))
        print("matplotlib.pyplot" in sys.modules)
    """ % str(__import__("pathlib").Path(__file__).parent))
    runs = [subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout
            for _ in range(2)]
    assert runs[0] == runs[1] and runs[0].strip().endswith("False")


# ----------------------------------------------------------------------------------------------- edge cases
def test_dense_mode_rasterizes_the_bulk_and_keeps_flags_on_top():
    r = funnel(_synthetic(2500, seed=3))
    (bulk,) = _gid(r.figure, "status-not_different")
    assert bulk.get_rasterized()
    for key in ("above", "below"):
        (coll,) = _gid(r.figure, f"status-{key}")
        assert not coll.get_rasterized() and coll.get_zorder() > bulk.get_zorder()
    assert all(c.get_rasterized() for c in _gid(r.figure, "limit-marks-upper"))
    assert r.counts["above"] + r.counts["below"] + r.counts["not_different"] == 2500


def test_no_finite_estimate_sits_at_the_axis_edge():
    r = funnel(_synthetic(80, seed=4, scale="log_odds", n_nofinite=3))
    (coll,) = _gid(r.figure, "no-finite-estimate")
    ylo, yhi = r.axes.get_ylim()
    yv = np.asarray(coll.get_offsets())[:, 1]
    assert len(yv) == 3 and np.all(yv < ylo + 0.05 * (yhi - ylo)) and ylo > -40.0
    assert r.axes.get_ylabel() == "Provider effect (log-odds)"


def test_providers_without_precision_are_counted_not_dropped():
    r = funnel(_synthetic(50, seed=5, n_nan=2))
    assert "2 provider(s) without a finite precision or estimate are not drawn." in r.long_description
    drawn = sum(len(c.get_offsets()) for k in ("above", "below", "not_different") for c in _gid(r.figure, f"status-{k}"))
    assert drawn == 48


def test_result_object(profiles, tmp_path):
    r = funnel(profiles["score"], size="double", title="Readmissions")
    assert isinstance(r, FigureResult) and r.kind == "funnel"
    fig, ax = r
    assert fig is r.figure and ax is r.axes and ax.get_title(loc="left") == "Readmissions"
    assert fig.get_figwidth() == pytest.approx(175.0 / 25.4)
    with pytest.raises(AttributeError):
        r.kind = "other"
    for suffix in ("svg", "pdf", "png"):
        assert r.save(tmp_path / f"f.{suffix}").read_bytes() == r.to_bytes(suffix)
    with pytest.raises(ValueError, match="format"):
        r.to_bytes("jpeg")


# --------------------------------------------------------------------------------------- sources and errors
def test_models_as_sources(fe):
    r = funnel(fe)
    assert r.provenance["test_method"] == "score" and r.axes.get_xlabel().startswith("Precision under the null")
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"y": rng.normal(size=600), "x1": rng.normal(size=600), "provider_id": np.repeat(np.arange(20), 30)})
    lin = LinearFixedEffectModel()
    lin.fit(d, y_var="y", x_vars=["x1"], provider_var="provider_id")
    assert funnel(lin).axes.get_ylabel() == "Provider effect (difference)"
    prov = np.repeat(np.arange(15), 40)
    X = rng.normal(size=(prov.size, 2))
    t = rng.exponential(1 / (0.3 * np.exp(X @ [0.5, -0.3])))
    c = rng.uniform(0.5, 3, prov.size)
    data = dict(duration=np.minimum(t, c), event=(t <= c).astype(float), provider_id=prov)
    cox = CoxPH(ties="breslow").fit(X, duration=data["duration"], event=data["event"], strata=prov)
    assert funnel(cox, X, **data).axes.get_xlabel().startswith("Expected events")


def test_capability_errors(fe):
    with pytest.raises(CapabilityError, match="test\\(\\) result"):
        funnel(fe.test())
    with pytest.raises(CapabilityError, match="limits=True"):
        funnel(ProviderProfile.from_model(fe))
    with pytest.raises(TypeError, match="come from the profile"):
        funnel(ProviderProfile.from_model(fe, limits=True), test_method="score")
    with pytest.raises(CapabilityError, match="ADR-004"):
        funnel(LinearRandomEffectModel())
