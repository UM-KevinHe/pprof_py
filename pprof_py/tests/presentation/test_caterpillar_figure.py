"""caterpillar(): drawn intervals, points and volumes equal the profile; ordering, edge cases, errors (spec §2.2)."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text

from pprof_py import CoxPH, LinearFixedEffectModel, LogisticFixedEffectModel
from pprof_py.presentation import CapabilityError, ProviderProfile, caterpillar


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


def _synthetic(n, *, seed=0, ties=0, n_nofinite=0, wald_like=False):
    """An interval profile from a plain frame: intervals est +/- 1.96 se, flags consistent with them, volumes."""
    rng = np.random.default_rng(seed)
    vol = rng.integers(10, 400, n).astype(float)
    se = 1.0 / np.sqrt(vol / 10.0)
    est = rng.normal(0.0, 0.6, n)
    est[1:1 + ties] = est[0]                                      # tied estimates keep their source order
    lo, hi = est - 1.96 * se, est + 1.96 * se
    finite = np.ones(n, dtype=bool)
    k = slice(n - n_nofinite, n)
    finite[k] = False
    est[k] = -30.0                                                # a solver clamp, never drawn
    lo[k], hi[k] = (-120.0, 90.0) if wald_like else (-np.inf, -0.5)
    flag = np.where(lo > 0.0, 1, np.where(hi < 0.0, -1, 0))
    df = pd.DataFrame({"id": [f"H{i:05d}" for i in range(n)], "est": est, "lo": lo, "hi": hi, "flag": flag, "nv": 0.0,
                       "n": vol, "fin": finite})
    roles = {"provider_id": "id", "estimate": "est", "ci_lower": "lo", "ci_upper": "hi", "flag": "flag",
             "null_value": "nv", "denominator": "n", "finite_estimate": "fin"}
    prov = {"scale": "log_odds", "level": 0.95, "alternative": "two_sided", "test_method": "custom",
            "estimator": "custom estimates", "denominator_kind": "records"}
    return ProviderProfile.from_frame(df, roles=roles, provenance=prov)


@pytest.fixture(scope="module")
def fe():
    m = LogisticFixedEffectModel()
    m.fit(_logistic(), y_var="y", x_vars=["x1"], provider_var="provider_id")
    return m


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def _bounds(coll):
    """The interval bounds a collection draws: bars keep them in ``pprof_bounds`` (their round ends are trimmed onto
    them after layout); classic lines draw them as segments."""
    return np.asarray(coll.pprof_bounds if hasattr(coll, "pprof_bounds") else coll.get_segments())


def _rows(prof):
    """Row of each provider: estimate order, ties in source order (what the plot promises)."""
    f = prof.data
    order = np.argsort(f["estimate"].to_numpy(), kind="stable")
    row = np.empty(len(f))
    row[order] = np.arange(len(f))
    return f, row


# ------------------------------------------------------------------------------ drawn data equal the profile
@pytest.mark.parametrize("theme", ["publication", "classic"])
def test_points_and_intervals_are_the_profile_in_estimate_order(fe, theme):
    prof = ProviderProfile.from_model(fe)
    r = caterpillar(prof, theme=theme)
    f, row = _rows(prof)
    status = f["status"].astype(str).to_numpy()
    finite = f["finite_estimate"].astype(bool).to_numpy()
    xlo, xhi = r.axes.get_xlim()
    lo_all, hi_all = f["ci_lower"].to_numpy(), f["ci_upper"].to_numpy()
    inside_lo = np.isfinite(lo_all) & (lo_all > xlo) & (lo_all < xhi)
    inside_hi = np.isfinite(hi_all) & (hi_all > xlo) & (hi_all < xhi)
    drawn_interval = finite | (inside_lo ^ inside_hi)            # ADR-005: from the edge to a finite bound
    assert (~finite & drawn_interval).any()                      # the zero-event provider's exact interval is drawn
    for key in ("above", "below", "not_different"):
        sel = (status == key) & finite
        (pts,) = _gid(r.figure, f"status-{key}")
        np.testing.assert_array_equal(np.asarray(pts.get_offsets()), np.c_[f["estimate"].to_numpy()[sel], row[sel]])
        seg = (status == key) & drawn_interval
        coll = _gid(r.figure, f"interval-{key}")[0]
        # bars keep their bounds in pprof_bounds (the drawn round ends are trimmed onto them: test_caterpillar_identity)
        segs = np.asarray(coll.pprof_bounds if hasattr(coll, "pprof_bounds") else coll.get_segments())
        np.testing.assert_array_equal(segs[:, 0, 0], np.clip(np.nan_to_num(lo_all[seg], neginf=xlo), xlo, xhi))
        np.testing.assert_array_equal(segs[:, 1, 0], np.clip(np.nan_to_num(hi_all[seg], posinf=xhi), xlo, xhi))
        np.testing.assert_array_equal(segs[:, 0, 1], row[seg])
    (ref,) = _gid(r.figure, "reference")
    assert np.all(np.asarray(ref.get_xdata()) == prof.provenance["null_value"])


def test_ties_keep_the_source_order():
    prof = _synthetic(20, seed=1, ties=3)
    r = caterpillar(prof)
    labels = [t.get_text() for t in r.axes.get_yticklabels()]
    tied = [lab for lab in labels if lab in ("H00000", "H00001", "H00002", "H00003")]
    assert tied == ["H00000", "H00001", "H00002", "H00003"]


def test_volume_bars_are_the_denominators(fe):
    prof = ProviderProfile.from_model(fe)
    r = caterpillar(prof)
    f, row = _rows(prof)
    (bars,) = _gid(r.figure, "volume")
    verts = [p.vertices for p in bars.get_paths()]
    widths = np.array([v[:, 0].max() for v in verts])
    centres = np.array([v[:, 1].mean() for v in verts])
    order = np.argsort(centres)
    np.testing.assert_array_equal(widths[order], f["denominator"].to_numpy()[np.argsort(row)])


def test_rows_are_labelled_up_to_60_providers(fe):
    prof = ProviderProfile.from_model(fe)
    r = caterpillar(prof, highlight=["P10"])
    f, row = _rows(prof)
    labels = r.axes.get_yticklabels()
    assert [t.get_text() for t in labels] == [str(i) for i in f.index[np.argsort(row)]]
    assert [t.get_fontweight() for t in labels if t.get_text() == "P10"] == ["semibold"]     # SemiBold file (D80)
    c = caterpillar(prof, highlight=["P10"], theme="classic")
    assert [t.get_fontweight() for t in c.axes.get_yticklabels() if t.get_text() == "P10"] == ["bold"]
    big = caterpillar(_synthetic(150, seed=2), highlight=["H00007"])
    assert not [t for t in big.axes.get_yticklabels() if t.get_text()]
    assert [t for t in big.figure.findobj(Text) if t.get_text() == "H00007"]


@pytest.mark.parametrize("theme", ["publication", "notebook"])
def test_row_labels_do_not_overlap(fe, theme):
    r = caterpillar(fe, theme=theme)
    FigureCanvasAgg(r.figure)
    r.figure.canvas.draw()
    rend = r.figure.canvas.get_renderer()
    boxes = sorted((t.get_window_extent(rend) for t in r.axes.get_yticklabels() if t.get_text()), key=lambda b: b.y0)
    assert len(boxes) == len(ProviderProfile.from_model(fe))
    assert all(lower.y1 <= upper.y0 + 0.5 for lower, upper in zip(boxes, boxes[1:]))


def test_axis_says_order_not_rank(fe):
    r = caterpillar(fe)
    assert r.axes.get_ylabel() == "Providers, ordered by estimate"
    labels = [t.get_text() for t in r.figure.findobj(Text) if t.get_text().strip()]
    labels = [t for t in labels if "the order is not a ranking" not in t]       # the footnote's own disclaimer
    assert labels and not any("rank" in t.lower() for t in labels)
    assert "the order is not a ranking" in r.long_description


# ----------------------------------------------------------------------------------------------- edge cases
def test_no_finite_estimate_is_marked_at_the_edge_with_its_one_sided_interval():
    prof = _synthetic(30, seed=3, n_nofinite=2)
    r = caterpillar(prof)
    xlo, xhi = r.axes.get_xlim()
    assert xlo > -29.0                                           # the clamp never enters the axis range
    (marks,) = _gid(r.figure, "no-finite-estimate-lower")
    assert np.all(np.asarray(marks.get_offsets())[:, 0] == xlo) and len(marks.get_offsets()) == 2
    segs = np.concatenate([_bounds(c) for k in ("above", "below", "not_different", "not_tested")
                           for c in _gid(r.figure, f"interval-{k}")])
    edge = segs[np.isclose(segs[:, 1, 0], -0.5)]
    assert len(edge) == 2 and np.all(edge[:, 0, 0] == xlo)        # edge to the finite bound
    wald = caterpillar(_synthetic(30, seed=3, n_nofinite=2, wald_like=True))
    segs = np.concatenate([_bounds(c) for k in ("above", "below", "not_different")
                           for c in _gid(wald.figure, f"interval-{k}")])
    assert len(segs) == 28                                       # no segment when neither bound is finite on the axis
    assert "without a finite estimate, marked at the axis edge" in wald.long_description


def test_zero_events_on_a_ratio_scale(fe):
    prof = ProviderProfile.from_test(fe.test_standardized(measure="indirect_ratio"), model=fe)
    r = caterpillar(prof)
    (zero,) = _gid(r.figure, "zero-events")
    assert len(zero.get_offsets()) == prof.status_counts()["zero_events"] == 1
    assert np.all(np.asarray(zero.get_offsets())[:, 0] == 0.0)
    assert r.axes.get_xlabel().startswith("Indirectly standardized ratio (O/E)")
    assert "score test" in r.long_description and "where the measure equals 1.00" in r.long_description


def test_dense_mode_rasterizes_the_bulk_and_keeps_flags_on_top():
    r = caterpillar(_synthetic(2500, seed=4))
    for gid in ("status-not_different", "interval-not_different", "volume"):
        (a,) = _gid(r.figure, gid)
        assert a.get_rasterized(), gid
    for key in ("above", "below"):
        (pts,) = _gid(r.figure, f"status-{key}")
        assert not pts.get_rasterized()
        assert pts.get_zorder() > _gid(r.figure, "status-not_different")[0].get_zorder()


# ------------------------------------------------------------------------------------- errors and options
def test_capability_errors_and_volume_option(fe):
    with pytest.raises(CapabilityError, match="intervals"):
        caterpillar(ProviderProfile.from_model(fe, limits=True))           # the score test has no intervals
    with pytest.raises(CapabilityError, match="volume=False"):
        caterpillar(fe.test())                                              # a bare test() result has no denominators
    r = caterpillar(fe.test(), volume=False)
    assert "Denominators are not shown." in r.long_description and not _gid(r.figure, "volume")
    with pytest.raises(TypeError, match="come from the profile"):
        caterpillar(ProviderProfile.from_model(fe), test_method="wald")


def test_export_is_deterministic(fe):
    prof = ProviderProfile.from_model(fe)
    a, b = caterpillar(prof), caterpillar(prof)
    for fmt in ("svg", "pdf", "png"):
        assert a.to_bytes(fmt) == b.to_bytes(fmt) == a.to_bytes(fmt)


def test_text_layout_and_alt_text(fe):
    prof = ProviderProfile.from_model(fe)
    r = caterpillar(prof, size="single")
    FigureCanvasAgg(r.figure)
    r.figure.canvas.draw()
    rend = r.figure.canvas.get_renderer()
    texts = [t for t in r.figure.findobj(Text) if t.get_visible() and t.get_text().strip()]
    assert min(t.get_fontsize() for t in texts) >= 7.0
    (note,) = [t for t in texts if t.get_text().startswith(f"{len(prof)} providers")]
    nb = note.get_window_extent(rend)
    assert nb.x0 >= 0 and nb.x1 <= r.figure.bbox.width + 0.5 and nb.y1 <= r.axes.get_tightbbox(rend).y0 + 0.5
    c = prof.status_counts()
    assert r.alt_text.startswith(f"Interval plot of {len(prof)} providers ordered by estimate")
    assert f"{c['above']} above and {c['below']} below the reference" in r.alt_text and "Volume panel" in r.alt_text


def test_models_as_sources():
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"y": rng.normal(size=600), "x1": rng.normal(size=600), "provider_id": np.repeat(np.arange(20), 30)})
    lin = LinearFixedEffectModel()
    lin.fit(d, y_var="y", x_vars=["x1"], provider_var="provider_id")
    assert caterpillar(lin).axes.get_xlabel().startswith("Provider effect (difference)")
    prov = np.repeat(np.arange(15), 40)
    X = rng.normal(size=(prov.size, 2))
    t = rng.exponential(1 / (0.3 * np.exp(X @ [0.5, -0.3])))
    c = rng.uniform(0.5, 3, prov.size)
    data = dict(duration=np.minimum(t, c), event=(t <= c).astype(float), provider_id=prov)
    cox = CoxPH(ties="breslow").fit(X, duration=data["duration"], event=data["event"], strata=prov)
    r = caterpillar(cox, X, **data)
    assert r.axes.get_xlabel().startswith("Observed / expected (O/E)") and r.kind == "caterpillar"
