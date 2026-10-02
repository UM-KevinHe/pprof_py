"""The interval plot in the package's visual identity (D79 to D81): round-ended bars whose drawn ends sit exactly on
the bounds, darker haloed marks on the bars, thin volume bars, the title row and key above, highlight bands and
rings, the short publication footnote, and group_by="status". Each check has its classic counterpart."""
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import SubFigure
from matplotlib.legend import Legend
from matplotlib.text import Text

from pprof_py.presentation import ProviderProfile, Theme, caterpillar


@pytest.fixture(scope="module")
def fe():
    from pprof_py import LogisticFixedEffectModel
    from pprof_py.presentation._synthetic import provider_data
    m = LogisticFixedEffectModel()
    m.fit(provider_data(40), y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    return m


@pytest.fixture(scope="module")
def ratio(fe):
    return ProviderProfile.from_test(fe.test_standardized(measure="indirect_ratio"), model=fe)


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def _bars(r):
    return [c for k in ("above", "below", "not_different", "not_tested") for c in _gid(r.figure, f"interval-{k}")]


@pytest.mark.parametrize("dense", [False, True])
def test_bar_ends_sit_exactly_on_the_bounds(fe, ratio, dense):
    from pprof_py.presentation._synthetic import provider_data
    if dense:
        from pprof_py import LogisticFixedEffectModel
        m = LogisticFixedEffectModel()
        m.fit(provider_data(200), y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
        r = caterpillar(ProviderProfile.from_model(m, test_method="poibin_exact"))
        width = Theme().lines.interval_bar_dense
    else:
        r, width = caterpillar(ratio), Theme().lines.interval_bar
    before = [c.get_segments() for c in _bars(r)]
    r.to_bytes("png")                                    # saving redraws; the trimmed ends must not move
    ax = r.axes
    one = ax.figure.dpi_scale_trans.transform([(0.0, 0.0), (width / 72.0 / 2.0, 0.0)])
    radius = one[1, 0] - one[0, 0]
    edge_lo, edge_hi = sorted(ax.transData.transform([(v, 0.0) for v in ax.get_xlim()])[:, 0])
    checked = 0
    for coll, seg0 in zip(_bars(r), before):
        assert coll.get_capstyle() == "round"
        drawn, bounds = np.asarray(coll.get_segments()), np.asarray(coll.pprof_bounds)
        np.testing.assert_array_equal(drawn, np.asarray(seg0))
        for d, b in zip(drawn, bounds):
            (dl, _), (dh, _) = ax.transData.transform(d)
            (bl, _), (bh, _) = ax.transData.transform(b)
            if bh - bl < 2 * radius:
                continue                                 # shorter than its own width: a dot at the centre
            if bl > edge_lo + 1e-6:
                assert dl - radius == pytest.approx(bl, abs=1e-6); checked += 1
            if bh < edge_hi - 1e-6:
                assert dh + radius == pytest.approx(bh, abs=1e-6); checked += 1
    assert checked > 20
    c = caterpillar(ratio, theme="classic")
    for coll in _bars(c):
        assert not hasattr(coll, "pprof_bounds")


def test_marks_on_bars_are_a_shade_darker_with_halos_and_classic_keeps_status_colours(ratio):
    th = Theme()
    (nd,) = _gid(caterpillar(ratio).figure, "status-not_different")
    assert to_hex(nd.get_facecolor()[0]) == th.bar_marks["not_different"].lower()
    assert to_hex(nd.get_edgecolor()[0]) == th.background.lower()
    (nd_c,) = _gid(caterpillar(ratio, theme="classic").figure, "status-not_different")
    assert to_hex(nd_c.get_facecolor()[0]) == Theme.classic().status["not_different"].color.lower()


def test_volume_bars_are_thin_in_the_identity(ratio):
    for theme, half in (("publication", Theme().volume_half[0]), ("classic", 0.35)):
        (bars,) = _gid(caterpillar(ratio, theme=theme).figure, "volume")
        heights = {round(float(np.ptp(p.vertices[:, 1])), 9) for p in bars.get_paths()}
        assert heights == {round(2 * half, 9)}


def test_group_by_status_orders_sections_with_headers(ratio):
    r = caterpillar(ratio, group_by="status")
    f = ratio.data
    rows = {}
    for key in ("above", "not_different", "below"):
        (pts,) = _gid(r.figure, f"status-{key}")
        off = np.asarray(pts.get_offsets())
        rows[key] = off
        assert np.all(np.diff(off[np.argsort(off[:, 1]), 0]) >= 0)        # by estimate within the group
    assert rows["above"][:, 1].min() > rows["not_different"][:, 1].max() > rows["below"][:, 1].max()
    heads = [t.get_text() for t in r.figure.findobj(Text) if t.get_text().endswith(")") and "reference (" in t.get_text()]
    counts = f["status"].value_counts()
    assert f"Above reference ({counts['above']})" in heads and f"Below reference ({counts['below']})" in heads
    assert r.axes.get_ylabel() == "Providers, grouped by test result"
    assert "grouped by the test's result" in r.caption and "grouped by test result" in r.alt_text
    assert _gid(r.figure, "section-rule")
    assert caterpillar(ratio, group_by="status", theme="classic").axes.get_ylabel() == "Providers, grouped by test result"
    with pytest.raises(ValueError, match="group_by"):
        caterpillar(ratio, group_by="volume")


def test_title_row_and_key_above_and_classic_keeps_them_in_place(ratio):
    r = caterpillar(ratio, theme="report", title="Ratios")
    rd = r.figure.canvas.get_renderer()
    (legend,) = r.figure.findobj(Legend)
    assert legend.get_window_extent(rd).y0 > r.axes.get_window_extent(rd).y1
    titles = [sf._suptitle for sf in r.figure.findobj(SubFigure) if getattr(sf, "_suptitle", None) is not None]
    assert [t.get_text() for t in titles] == ["Ratios"]
    assert titles[0].get_fontproperties().get_file().endswith("IBMPlexSans-SemiBold.ttf")
    c = caterpillar(ratio, theme="classic", title="Ratios")
    rc = c.figure.canvas.get_renderer()
    (legend_c,) = c.figure.findobj(Legend)
    assert legend_c.get_window_extent(rc).y1 < c.axes.get_window_extent(rc).y0 and c.axes.get_title(loc="left") == "Ratios"


def test_highlight_band_ring_and_emphasis(ratio):
    pid = str(ratio.data.index[5])
    r = caterpillar(ratio, highlight=[pid])
    assert len(_gid(r.figure, "highlight-band")) == 2                      # both panels
    (lab,) = [t for t in r.axes.get_yticklabels() if t.get_text() == pid]
    assert lab.get_gid() == "emphasis" and lab.get_fontproperties().get_file().endswith("IBMPlexSans-SemiBold.ttf")
    c = caterpillar(ratio, highlight=[pid], theme="classic")
    assert not _gid(c.figure, "highlight-band") and not _gid(c.figure, "highlight-ring")


def test_publication_footnote_is_short_and_the_caption_full(ratio):
    n = len(ratio.data)
    r = caterpillar(ratio)
    (shown,) = [t.get_text().replace("\n", " ") for t in r.figure.findobj(Text) if t.get_text().startswith(f"{n} providers")]
    assert "Estimates:" not in shown and "not a ranking" in shown and "same test as the flags" in shown
    assert "Estimates:" in r.caption and r.long_description == r.alt_text + " " + r.caption
    (full,) = [t.get_text().replace("\n", " ") for t in caterpillar(ratio, theme="report").figure.findobj(Text)
               if t.get_text().startswith(f"{n} providers")]
    assert full == caterpillar(ratio, theme="report").caption
