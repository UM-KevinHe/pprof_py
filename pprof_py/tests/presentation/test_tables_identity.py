"""Provider tables in the package's visual identity (D82 to D84): HTML styled from the theme's tokens with status glyphs
coloured and named, the inline interval column, and group_by="status" in every format. Each check has its classic
counterpart; the 0.6.0 HTML golden is checked in test_tables.py."""
import re

import numpy as np
import pytest

from pprof_py.presentation import ProviderProfile, Theme, provider_table
from pprof_py.presentation.tables._spec import TABLE_CSS, table_css, table_markup


@pytest.fixture(scope="module")
def ratio():
    from pprof_py import LogisticFixedEffectModel
    from pprof_py.presentation._synthetic import provider_data
    m = LogisticFixedEffectModel()
    m.fit(provider_data(40), y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    return ProviderProfile.from_test(m.test_standardized(measure="indirect_ratio"), model=m)


def test_identity_css_comes_from_the_theme_tokens_and_classic_is_0_6_0(ratio):
    th = Theme()
    css = table_css(th)
    assert css.startswith('.pprof-table{border-collapse:collapse;font-family:"IBM Plex Sans",')
    for token in (th.ink, th.muted, th.grid_color, th.status["above"].color, th.status["below"].color):
        assert token in css
    assert "#123456" in table_css(th.derive(status={"above": {"color": "#123456"}}))
    assert table_css(Theme.classic()) == table_css(None) == TABLE_CSS
    assert provider_table(ratio).theme == Theme() and provider_table(ratio, theme="classic").theme == Theme.classic()


def test_reports_keep_the_0_6_0_markup_until_they_are_restyled(ratio):
    spec = provider_table(ratio).spec
    assert table_markup(spec) == table_markup(spec, theme=Theme.classic())
    assert table_markup(spec, caption_prefix="Table 1. ") != table_markup(spec, caption_prefix="Table 1. ", theme=Theme())


def test_status_cells_are_coloured_glyphs_with_their_word(ratio):
    html = provider_table(ratio).to_html()
    assert '<td class="left"><span class="pp-glyph pp-above" aria-hidden="true">\u25b2</span>Above</td>' in html
    assert '<span class="pp-glyph pp-not_different" aria-hidden="true">\u25cf</span>Not different' in html
    classic = provider_table(ratio, theme="classic").to_html()
    assert "pp-glyph" not in classic and '<td class="center">\u25b2</td>' in classic


def _bars(html):
    return re.findall(r'<line x1="([\d.]+)" y1="7.0" x2="([\d.]+)" y2="7.0" stroke="[^"]+" stroke-width="3.0" '
                      r'stroke-linecap="round"></line>', html)


def test_inline_intervals_sit_exactly_on_their_bounds_on_one_scale(ratio):
    t = provider_table(ratio, intervals=True)
    html = t.to_html()
    f = ratio.data
    lo, hi, nv = (f[c].to_numpy(float) for c in ("ci_lower", "ci_upper", "null_value"))
    bars = _bars(html)
    assert len(bars) == int((np.isfinite(lo) & np.isfinite(hi)).sum()) == len(f)
    vals = np.r_[lo, hi, nv]
    a, b = vals.min(), vals.max()
    a, b = a - 0.04 * (b - a), b + 0.04 * (b - a)
    sx = lambda v: 4.0 + 132.0 * (v - a) / (b - a)        # noqa: E731  (140 px wide, 4 px pads)
    for (x0, x1), l, h in zip(bars, lo, hi):
        if sx(h) - sx(l) >= 3.0:
            assert float(x0) - 1.5 == pytest.approx(sx(l), abs=0.006)
            assert float(x1) + 1.5 == pytest.approx(sx(h), abs=0.006)
    ref = re.findall(r'<line x1="([\d.]+)" y1="0" x2="\1" y2="14" stroke="[^"]+" stroke-width="1"></line>', html)
    assert ref and all(float(x) == pytest.approx(sx(1.0), abs=0.006) for x in ref)
    assert html.count('<th scope="col" class="left pp-interval"><svg') == 1
    plain = provider_table(ratio)
    for fmt in ("to_markdown", "to_latex", "to_text"):
        assert getattr(t, fmt)() == getattr(plain, fmt)()
    assert 'class="pp-interval"' not in plain.to_html() and "<svg" not in plain.to_html()     # no interval cells


def test_group_by_status_in_every_format(ratio):
    t = provider_table(ratio, group_by="status", min_volume=25)
    f = ratio.data
    status = np.where(f["denominator"].to_numpy(float) < 25, "suppressed", f["status"].astype(str).to_numpy())
    order = ["above", "not_different", "below", "not_tested", "suppressed"]
    expect = [i for k in order for i in f.index[status == k]]           # provider order within each group
    assert list(t.spec.cells.index) == expect == list(t.to_frame().index)
    labels = [lab for lab, _ in t.spec.groups]
    assert labels[0] == f"Above reference ({(status == 'above').sum()})" and labels[-1].startswith("Suppressed (")
    html = t.to_html()
    assert html.count("<tbody>") == len(labels) and html.count('<tr class="pp-group">') == len(labels)
    md, tex, txt = t.to_markdown(), t.to_latex(), t.to_text()
    for lab in labels:
        assert f"| **{lab}** |" in md and rf"\textit{{{lab}}}" in tex and f"\n{lab}\n" in txt
    assert tex.count(r"\addlinespace") == len(labels) - 1
    assert "Rows are grouped by the test's result" in md
    classic = provider_table(ratio, group_by="status", theme="classic").to_html()
    assert 'class="left" style="font-weight:600"' in classic
    with pytest.raises(ValueError, match="group_by"):
        provider_table(ratio, group_by="volume")
