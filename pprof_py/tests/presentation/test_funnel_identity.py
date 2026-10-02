"""The funnel in the package's visual identity (D76 to D78): the shaded corridor only where the curve reproduces the
flags, haloed markers, the key above the data under a title row, a short publication footnote with the full text
as the caption, and a ring and emphasized label for highlighted providers. Each check has its classic counterpart."""
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import SubFigure
from matplotlib.legend import Legend
from matplotlib.text import Text

from pprof_py.presentation import ProviderProfile, Theme, funnel


@pytest.fixture(scope="module")
def fe():
    from pprof_py import LogisticFixedEffectModel
    from pprof_py.presentation._synthetic import provider_data
    m = LogisticFixedEffectModel()
    m.fit(provider_data(60), y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    return m


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def _note(fig, n):
    return [t.get_text().replace("\n", " ") for t in fig.findobj(Text) if t.get_text().startswith(f"{n} providers")]


def test_corridor_is_the_test_level_curve_where_it_reproduces_the_flags(fe):
    r = funnel(fe)
    (poly,) = _gid(r.figure, "corridor")
    c = ProviderProfile.from_model(fe, limits=True, levels=(0.95, 0.998)).funnel_curves
    c = c[np.isclose(c["level"].to_numpy(float), 0.95)]
    verts = np.vstack([p.vertices for p in poly.get_paths()])
    for col in ("lower", "upper"):
        assert np.isin(np.round(c[col].to_numpy(float), 9), np.round(verts[:, 1], 9)).all(), col
    assert np.isclose(verts[:, 0].min(), c["precision"].min()) and np.isclose(verts[:, 0].max(), c["precision"].max())
    labels = [t.get_text() for t in r.figure.findobj(Legend)[0].get_texts()]
    assert "Not flagged at 95%" in labels


def test_no_corridor_for_exact_marks_a_grouped_null_or_classic(fe):
    from pprof_py.inference import EmpiricalNull
    exact = funnel(fe, test_method="poibin_exact")
    grouped = funnel(fe, null_model=EmpiricalNull.fitter(size=fe.provider_sizes_, n_groups=3))
    classic = funnel(fe, theme="classic")
    for r in (exact, grouped, classic):
        assert not _gid(r.figure, "corridor")
        assert "Not flagged at 95%" not in [t.get_text() for t in r.figure.findobj(Legend)[0].get_texts()]


def test_filled_markers_have_a_halo_and_classic_keeps_coloured_edges(fe):
    th, c = Theme(), Theme.classic()
    (nd,) = _gid(funnel(fe).figure, "status-not_different")
    assert to_hex(nd.get_edgecolor()[0]) == th.background.lower() and nd.get_linewidths()[0] == pytest.approx(th.halo)
    (zero,) = _gid(funnel(fe).figure, "zero-events")                  # hollow marks keep their coloured edge
    assert to_hex(zero.get_edgecolor()[0]) == th.status["no_finite_estimate"].color.lower()
    (nd_c,) = _gid(funnel(fe, theme=c).figure, "status-not_different")
    assert to_hex(nd_c.get_edgecolor()[0]) == c.status["not_different"].color.lower()


def test_key_sits_above_the_data_under_the_title_and_classic_keeps_it_below(fe):
    r = funnel(fe, theme="report", title="Readmission ratios")
    renderer = r.figure.canvas.get_renderer()
    (legend,) = r.figure.findobj(Legend)
    assert legend.get_window_extent(renderer).y0 > r.axes.get_window_extent(renderer).y1
    titles = [sf._suptitle for sf in r.figure.findobj(SubFigure) if getattr(sf, "_suptitle", None) is not None]
    assert [t.get_text() for t in titles] == ["Readmission ratios"] and not r.axes.get_title(loc="left")
    assert titles[0].get_window_extent(renderer).y0 >= legend.get_window_extent(renderer).y1 - 1.0
    assert titles[0].get_fontproperties().get_file().endswith("IBMPlexSans-SemiBold.ttf")
    c = funnel(fe, theme="classic", title="Readmission ratios")
    (legend_c,) = c.figure.findobj(Legend)
    rc = c.figure.canvas.get_renderer()
    assert legend_c.get_window_extent(rc).y1 < c.axes.get_window_extent(rc).y0
    assert c.axes.get_title(loc="left") == "Readmission ratios"


def test_publication_footnote_is_short_and_the_caption_is_full(fe):
    n = len(ProviderProfile.from_model(fe, limits=True).data)
    pub = funnel(fe)
    (shown,) = _note(pub.figure, n)
    assert "Limits:" not in shown and "Estimates:" not in shown and "Test: score test" in shown
    assert "Limits:" in pub.caption and pub.long_description == pub.alt_text + " " + pub.caption
    assert "The 99.8% curves are for reference only." in shown
    (full,) = _note(funnel(fe, theme="report").figure, n)
    assert full == funnel(fe, theme="report").caption


def test_highlighted_provider_gets_a_ring_and_an_emphasized_label(fe):
    pid = str(ProviderProfile.from_model(fe, limits=True).data.index[3])
    r = funnel(fe, highlight=[pid], label_flagged=False)
    (ring,) = _gid(r.figure, "highlight-ring")
    assert len(ring.get_offsets()) == 1
    (label,) = [t for t in r.figure.findobj(Text) if t.get_gid() == "emphasis"]
    assert label.get_text() == pid and label.get_fontproperties().get_file().endswith("IBMPlexSans-SemiBold.ttf")
    c = funnel(fe, theme="classic", highlight=[pid], label_flagged=False)
    assert not _gid(c.figure, "highlight-ring") and not [t for t in c.figure.findobj(Text) if t.get_gid() == "emphasis"]
    assert pid in [t.get_text() for t in c.figure.findobj(Text)]
