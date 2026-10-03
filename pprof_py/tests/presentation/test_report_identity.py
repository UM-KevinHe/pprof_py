"""The report in the package's visual identity (D87): the page and its tables styled from the theme's tokens, tables
keeping their inline intervals, an optional embedded IBM Plex Sans, and the classic report unchanged."""
import base64
import re

import pytest

from pprof_py import presentation as P
from pprof_py.presentation.reports._report import _CSS
from pprof_py.presentation.tables._spec import TABLE_CSS
from pprof_py.presentation.theme import _fonts


@pytest.fixture(scope="module")
def parts():
    from pprof_py import LogisticFixedEffectModel
    from pprof_py.presentation._synthetic import provider_data
    m = LogisticFixedEffectModel()
    m.fit(provider_data(30), y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    ratio = P.ProviderProfile.from_test(m.test_standardized(measure="indirect_ratio"), model=m)
    return {"fig": P.funnel(m, theme="report"), "table": P.provider_table(ratio, intervals=True)}


def _report(parts, **kw):
    return P.Report("Profile", subtitle="synthetic", date="2026-10-02", **kw).section("Results") \
        .figure(parts["fig"]).table(parts["table"])


def _style(html):
    return re.search(r"<style>(.*)</style>", html, re.S).group(1)


def test_identity_page_and_tables_take_the_theme_tokens(parts):
    th = P.Theme.report()
    html = _report(parts).to_html()
    css = _style(html)
    assert f'font-family:"{th.typography.family}",system-ui' in css and th.ink in css and th.muted in css
    assert f".note{{background:{th.corridor}" in css and "border-left" not in css     # a tinted box, not a left rule
    assert 'class="pp-glyph pp-above"' in html and 'class="pp-interval"' in html      # identity markup, intervals kept
    assert "<script" not in html and '="http' not in html and "@font-face" not in html
    assert "figure img{max-height:7.5in;width:auto;max-width:100%}" in css            # print: headings keep their figure
    assert html == _report(parts).to_html()                                            # deterministic


def test_classic_report_keeps_the_0_6_0_style_and_markup(parts):
    html = _report(parts, theme="classic").to_html()
    assert _style(html) == f"{_CSS}\n{TABLE_CSS}"
    assert "pp-glyph" not in html and 'class="pp-interval"' not in html


def test_embedded_fonts_are_the_bundled_files(parts):
    html = _report(parts, embed_fonts=True).to_html()
    faces = re.findall(r'@font-face\{font-family:"IBM Plex Sans";src:url\(data:font/ttf;base64,([A-Za-z0-9+/=]+)\) '
                       r'format\("truetype"\);font-weight:(\d+);font-style:normal\}', html)
    assert [w for _, w in faces] == ["400", "600"]
    for (data, _), name in zip(faces, ("IBMPlexSans-Regular.ttf", "IBMPlexSans-SemiBold.ttf")):
        assert base64.b64decode(data) == (_fonts.FONT_DIR / name).read_bytes()
    assert 'Reserved Font Name "Plex"; SIL Open Font License 1.1' in html


def test_missing_font_files_fall_back_to_named_fonts(parts, tmp_path, monkeypatch):
    monkeypatch.setattr(_fonts, "FONT_DIR", tmp_path)
    _fonts._reset()
    try:
        with pytest.warns(UserWarning, match="missing or unreadable"):
            html = _report(parts, embed_fonts=True).to_html()
        assert "@font-face" not in html and '"IBM Plex Sans",system-ui' in html
    finally:
        _fonts._reset()
