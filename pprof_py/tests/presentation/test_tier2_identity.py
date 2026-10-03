"""The ten secondary displays in the package's visual identity (D85, D86): the title row above the key on top, the short
publication footnote (data quality and null calibration keep the full one), halos on filled marks, interval bars
trimmed onto their bounds in the forest and several-measures plots, and highlight rings, emphasis and bands. Each
check has its classic counterpart."""
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.legend import Legend
from matplotlib.text import Text

from pprof_py import presentation as P
from pprof_py.presentation import Theme

DISPLAYS = {
    "observed_expected": lambda s, **kw: P.observed_expected(s["exact"], **kw),
    "forest": lambda s, **kw: P.forest(s["fe"], **kw),
    "reliability": lambda s, **kw: P.reliability(s["iur"], **kw),
    "provider_variation": lambda s, **kw: P.provider_variation(s["re"], **kw),
    "shrinkage": lambda s, **kw: P.shrinkage(s["fe"], s["re"], **kw),
    "flag_stability": lambda s, **kw: P.flag_stability(s["fe"], test_method="score", **kw),
    "null_calibration": lambda s, **kw: P.null_calibration(s["fe"], test_method="score", null_model=s["null"], **kw),
    "data_quality": lambda s, **kw: P.data_quality(s["exact"], **kw),
    "multi_measure": lambda s, **kw: P.multi_measure(s["col"], **kw),
    "measure_agreement": lambda s, **kw: P.measure_agreement(s["col"], **kw),
}
FULL = {"data_quality", "null_calibration"}                  # their footnotes define what they draw


@pytest.fixture(scope="module")
def s():
    from pprof_py import LogisticFixedEffectModel, LogisticRandomEffectModel
    from pprof_py.inference import EmpiricalNull
    from pprof_py.measures.iur import BootstrapIUR
    from pprof_py.presentation._synthetic import provider_data
    d = provider_data(60)
    fe, re, fe2 = LogisticFixedEffectModel(), LogisticRandomEffectModel(verbose=False), LogisticFixedEffectModel()
    fe.fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    re.fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    fe2.fit(d, y_var="y2", x_vars=["x1"], provider_var="provider_id")
    iur = BootstrapIUR(n_boot=20).fit(d["y"].to_numpy(float), np.full(len(d), d["y"].mean()), d["provider_id"].to_numpy())
    col = P.ProfileCollection({"A": P.ProviderProfile.from_model(fe, test_method="wald"),
                               "B": P.ProviderProfile.from_model(fe2, test_method="wald")})
    return {"fe": fe, "re": re, "iur": iur, "col": col, "null": EmpiricalNull.fitter(size=fe.provider_sizes_, n_groups=2),
            "exact": P.ProviderProfile.from_model(fe, test_method="poibin_exact", limits=True)}


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def _row_titles(fig):
    return [sf._suptitle for sf in fig.subfigs if getattr(sf, "_suptitle", None) is not None]


def _footnote(fig):
    return " ".join(t.get_text().replace("\n", " ") for t in fig.subfigs[-1].texts)


@pytest.mark.parametrize("name", list(DISPLAYS))
def test_title_row_above_the_key_and_classic_keeps_its_layout(s, name):
    r = DISPLAYS[name](s, title="Title")
    rd = r.figure.canvas.get_renderer()
    (head,) = _row_titles(r.figure)
    assert head.get_text() == "Title" and head.get_fontproperties().get_file().endswith("IBMPlexSans-SemiBold.ttf")
    data_top = max(a.get_window_extent(rd).y1 for a in r.figure.axes if a.get_visible() and a.has_data())
    for legend in r.figure.findobj(Legend):
        assert legend.get_window_extent(rd).y0 > data_top
    c = DISPLAYS[name](s, title="Title", theme="classic")
    assert not _row_titles(c.figure) and "Title" in [t.get_text() for t in c.figure.findobj(Text)]


@pytest.mark.parametrize("name", list(DISPLAYS))
def test_publication_footnote_is_short_and_the_caption_full(s, name):
    pub, rep = DISPLAYS[name](s), DISPLAYS[name](s, theme="report")
    assert pub.long_description == pub.alt_text + " " + pub.caption
    if name in FULL:
        assert _footnote(pub.figure) == pub.caption
    else:
        assert _footnote(pub.figure) != pub.caption and len(_footnote(pub.figure)) < len(pub.caption)
    assert _footnote(rep.figure) == rep.caption


@pytest.mark.parametrize("name,gid", [("observed_expected", "status-not_different"),
                                      ("flag_stability", "stability-not_different"),
                                      ("multi_measure", "multi-not_different-0")])
def test_filled_marks_have_halos_and_classic_keeps_coloured_edges(s, name, gid):
    th, c = Theme(), Theme.classic()
    (coll,) = _gid(DISPLAYS[name](s).figure, gid)
    assert to_hex(coll.get_edgecolor()[0]) == th.background.lower() and coll.get_linewidths()[0] == pytest.approx(th.halo)
    (coll_c,) = _gid(DISPLAYS[name](s, theme="classic").figure, gid)
    assert to_hex(coll_c.get_edgecolor()[0]) == c.status["not_different"].color.lower()


@pytest.mark.parametrize("name,gid", [("forest", "coefficient-intervals"), ("multi_measure", "multi-intervals-0")])
def test_interval_bars_end_on_their_bounds(s, name, gid):
    r = DISPLAYS[name](s)
    (coll,) = _gid(r.figure, gid)
    ax = coll.axes
    width = coll.get_linewidths()[0]
    one = ax.figure.dpi_scale_trans.transform([(0.0, 0.0), (width / 72.0 / 2.0, 0.0)])
    radius = one[1, 0] - one[0, 0]
    lo_edge, hi_edge = sorted(ax.transData.transform([(v, 0.0) for v in ax.get_xlim()])[:, 0])
    assert coll.get_capstyle() == "round"
    checked = 0
    for d, b in zip(np.asarray(coll.get_segments()), np.asarray(coll.pprof_bounds)):
        (dl, _), (dh, _) = ax.transData.transform(d)
        (bl, _), (bh, _) = ax.transData.transform(b)
        if bh - bl < 2 * radius:
            continue
        if bl > lo_edge + 1e-6:
            assert dl - radius == pytest.approx(bl, abs=1e-6); checked += 1
        if bh < hi_edge - 1e-6:
            assert dh + radius == pytest.approx(bh, abs=1e-6); checked += 1
    assert checked >= 4
    (coll_c,) = _gid(DISPLAYS[name](s, theme="classic").figure, gid)
    assert not hasattr(coll_c, "pprof_bounds")


@pytest.mark.parametrize("name", ["observed_expected", "shrinkage", "measure_agreement", "multi_measure"])
def test_highlight_gets_a_ring_or_band_and_an_emphasized_label(s, name):
    pid = str(s["exact"].data.index[2])
    r = DISPLAYS[name](s, highlight=[pid])
    marks = _gid(r.figure, "highlight-band" if name == "multi_measure" else "highlight-ring")
    assert marks
    (lab,) = [t for t in r.figure.findobj(Text) if t.get_gid() == "emphasis"]
    assert lab.get_text() == pid and lab.get_fontproperties().get_file().endswith("IBMPlexSans-SemiBold.ttf")
    c = DISPLAYS[name](s, highlight=[pid], theme="classic")
    assert not _gid(c.figure, "highlight-ring") and not _gid(c.figure, "highlight-band")
    assert not [t for t in c.figure.findobj(Text) if t.get_gid() == "emphasis"]
