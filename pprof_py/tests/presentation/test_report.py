"""HTML report: self-contained, deterministic, accessible; figures and tables embedded unchanged (spec §9, S13)."""
import base64
import re

import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel
from pprof_py.presentation import ProviderProfile, Report, caterpillar, forest, funnel, provider_table
from pprof_py.presentation.tables._spec import table_markup


@pytest.fixture(scope="module")
def parts():
    rng = np.random.default_rng(4)
    pid = np.repeat(np.arange(25), rng.integers(30, 120, 25))
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(rng.normal(-1.0, 0.5, 25)[pid] + 0.5 * x))))
    fe = LogisticFixedEffectModel()
    fe.fit(pd.DataFrame({"y": y, "x1": x, "provider_id": pid}), y_var="y", x_vars=["x1"], provider_var="provider_id")
    prof = ProviderProfile.from_model(fe, test_method="poibin_exact", limits=True)
    return fe, prof, [funnel(prof), caterpillar(prof), forest(fe)], provider_table(prof)


def _build(parts, with_table=True):
    _, _, figs, table = parts
    r = Report("Profile <A> & B", subtitle="demo").section("Results").text("One <b>&</b>.\n\nTwo.")
    r.figure(figs[0], caption="Funnel").figure(figs[1])
    if with_table:
        r.table(table)
    return r.section("Model").figure(figs[2])


def test_report_embeds_figures_and_tables_unchanged(parts):
    _, prof, figs, table = parts
    h = _build(parts).to_html()
    srcs = re.findall(r'src="data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)"', h)
    assert [base64.b64decode(s) for s in srcs] == [f.to_bytes("svg") for f in figs]
    import html as _html

    for f in figs:
        assert f'alt="{_html.escape(f.alt_text)}"' in h and _html.escape(f.long_description) in h
    # the report embeds the table's markup in the report's theme (identity by default, D87; classic: the 0.6.0 markup)
    from pprof_py.presentation import Theme
    assert table_markup(table.spec, caption_prefix="Table 1. ", theme=Theme.report(),
                        intervals=getattr(table, "_intervals", None)) in h
    assert "<figcaption><span class=\"label\">Figure 1.</span> Funnel</figcaption>" in h
    assert f"<span class=\"label\">Figure 2.</span> {_html.escape(figs[1].alt_text)}</figcaption>" in h


def test_report_is_self_contained_accessible_and_deterministic(parts, tmp_path):
    r = _build(parts)
    h = r.to_html()
    visible = re.sub(r'src="data:image/svg\+xml;base64,[A-Za-z0-9+/=]+"', 'src=""', h)
    assert not [t for t in ("<script", "<link", "http://", "https://", "@import") if t in visible]
    ids = re.findall(r'\bid="([^"]+)"', visible)
    assert len(ids) == len(set(ids))
    assert all(f'id="{t}"' in h for t in re.findall(r'aria-(?:describedby|labelledby)="([^"]+)"', h))
    assert "Profile &lt;A&gt; &amp; B" in h and "One &lt;b&gt;&amp;&lt;/b&gt;." in h and '<html lang="en">' in h
    assert _build(parts).to_html() == h and r.save(tmp_path / "r.html").read_bytes() == h.encode("utf-8")
    assert r.outline() == ["Section: Results", "Text", "Figure 1: funnel (Funnel)", "Figure 2: caterpillar",
                           "Table 1: " + parts[3].spec.caption, "Section: Model", "Figure 3: forest",
                           "Appendix: methods and provenance"]


def test_appendix_and_disclosure(parts):
    h = _build(parts).to_html()
    methods = h[h.index('id="methods"'):]
    rows = re.findall(r'<tr><th scope="row" class="left">([^<]+)</th>(.*?)</tr>', methods)
    assert [r[0] for r in rows] == ["Figure 1", "Figure 2", "Table 1", "Figure 3"]
    assert "exact Poisson-binomial test" in rows[0][1] and "LogisticFixedEffectModel" in rows[2][1]
    assert "provider-level values" in h and "provider-level values" not in _build(parts, with_table=False).to_html()
    with pytest.raises(TypeError):
        Report("x").figure("not a figure")
    with pytest.raises(TypeError):
        Report("x").table(pd.DataFrame())
