"""Tables: golden HTML/Markdown/LaTeX/text, structure, values equal to the profile, Excel cells, errors (spec §3)."""
import io
import os
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pprof_py import CoxPH, LinearFixedEffectModel, LogisticFixedEffectModel
from pprof_py.presentation import CapabilityError, ProviderProfile, provider_table

GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("PPROF_UPDATE_GOLDEN") == "1"          # regenerate after a reviewed, intended change


def _profile():
    """Eight providers covering every rule: flags, rounding collision, zero events, NT, NI, suppression, escaping."""
    df = pd.DataFrame({
        "id": ["H01", "H02", "A&B_3", "C|D", "E_5", "H06", "H07", "H08"],
        "est": [1.25, 0.90, 0.58, 1.0049, 0.0, 1.10, 1.30, 1.90],
        "lo": [1.09, 0.71, 0.26, 1.0001, 0.0, 0.80, np.nan, 0.60],
        "hi": [1.43, 1.11, 1.10, 1.012, 0.62, 1.50, np.nan, 4.40],
        "flag": [1, 0, 0, 1, -1, pd.NA, 0, 0],
        "p": [0.0012, 0.35, 0.12, 0.049, 0.0004, np.nan, 0.2, 0.31],
        "n": [2310, 1245, 388, 5021, 40, 60, 75, 8],
        "obs": [214, 82, 9, 512, 0, 7, 10, 2],
        "exp": [171.0, 91.4, 15.6, 509.5, 6.0, 6.4, 7.7, 1.05],
        "nv": 1.0,
        "zero": [False, False, False, False, True, False, False, False]})
    roles = {"provider_id": "id", "estimate": "est", "ci_lower": "lo", "ci_upper": "hi", "flag": "flag", "p_value": "p",
             "denominator": "n", "observed": "obs", "expected": "exp", "null_value": "nv", "zero_events": "zero"}
    prov = {"model": "LogisticFixedEffectModel", "estimator": "fixed effect (unshrunken)", "measure": "indirect_ratio",
            "scale": "ratio", "test_method": "score", "reference": "median", "reference_value": -1.5,
            "null_value": 1.0, "null_model": {"kind": "theoretical", "null_mean": 0.0, "null_sd": 1.0},
            "alternative": "two_sided", "level": 0.95, "interval": "inversion", "denominator_kind": "records",
            "package_version": "0.5.0"}
    return ProviderProfile.from_frame(df, roles=roles, provenance=prov)


@pytest.fixture(scope="module")
def table():
    return provider_table(_profile(), p_values=True, min_volume=10)


@pytest.mark.parametrize("fmt", ["html", "md", "tex", "txt"])
def test_golden_outputs(table, fmt):
    text = {"html": table.to_html(), "md": table.to_markdown(), "tex": table.to_latex(), "txt": table.to_text()}[fmt]
    path = GOLDEN / f"provider_table.{fmt}"
    if UPDATE:
        path.write_text(text, encoding="utf-8")
    assert text == path.read_text(encoding="utf-8")
    assert not any(line != line.rstrip() for line in text.splitlines())   # no trailing whitespace anywhere


def test_classic_html_is_the_0_6_0_golden():
    # the 0.6.0 HTML, byte for byte (D82); the identity HTML is the provider_table.html golden
    t = provider_table(_profile(), p_values=True, min_volume=10, theme="classic")
    assert t.to_html() == (GOLDEN / "provider_table_classic.html").read_text(encoding="utf-8")


def test_cells_follow_the_rules(table):
    cells = table.spec.cells
    assert cells.loc["C|D", "estimate_ci"] == "1.0049 (1.0001\u20131.0120)"     # extra digits: 1.00 would touch 1.00
    assert cells.loc["E_5", "estimate_ci"] == "0.00 (0.00\u20130.62)"            # zero events: finite O/E = 0
    assert cells.loc["H06", "flag"] == "NT" and cells.loc["H07", "estimate_ci"] == "1.30 (NI)"
    assert (cells.loc["H08", ["estimate_ci", "flag", "p_value"]] == "S").all()
    assert cells.loc["H01", "p_value"] == "0.001" and cells.loc["E_5", "p_value"] == "<0.001"
    texts = [t for _, t in table.spec.notes]
    assert any(t.startswith("S: suppressed, fewer than 10 records") for t in texts)
    assert any("extra decimals" in t for t in texts) and any(t.startswith("NI:") for t in texts)


class _Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.sups, self.text = [], [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        if self.tags and self.tags[-1][0] == "sup":
            self.sups.append(data)
        self.text.append(data)


def test_html_is_semantic_and_self_contained(table):
    html = table.to_html()
    p = _Collect()
    p.feed(html)
    tags = [t for t, _ in p.tags]
    assert "caption" in tags and "tfoot" in tags and "script" not in tags and "link" not in tags
    assert not re.search(r"https?://", html)
    heads = [a for t, a in p.tags if t == "th"]
    assert sum(a.get("scope") == "col" for a in heads) == len(table.spec.columns)
    assert sum(a.get("scope") == "row" for a in heads) == len(table.spec.cells)
    markers = {c.marker for c in table.spec.columns if c.marker}
    assert markers == {"a", "b"} and markers <= set(p.sups[len(markers):])      # each header marker has a note
    assert "A&amp;B_3" in html and table.to_html(standalone=False).startswith("<style>")


def test_markdown_latex_and_text_specifics(table):
    md = table.to_markdown()
    assert "| C\\|D |" in md and "|:--|--:|" in md
    tex = table.to_latex()
    assert r"A\&B\_3" in tex and r"$\blacktriangle$" in tex and r"\begin{tabular}" in tex and "\u2212" not in tex
    big = provider_table(ProviderProfile.from_frame(
        pd.DataFrame({"id": range(45), "est": 0.1, "lo": -0.1, "hi": 0.3, "flag": 0, "nv": 0.0}),
        roles={"provider_id": "id", "estimate": "est", "ci_lower": "lo", "ci_upper": "hi", "flag": "flag",
               "null_value": "nv"}))
    assert r"\begin{longtable}" in big.to_latex()
    lines = table.to_text().splitlines()
    assert len({len(lines[1]), len(lines[3])}) == 1                      # rules span the same width


def test_values_equal_the_profile_and_hide_suppressed(table):
    prof = _profile()
    v, f = table.to_frame(), prof.data
    for col in ("estimate", "ci_lower", "ci_upper", "p_value"):
        np.testing.assert_array_equal(v.loc[f.index != "H08", col].to_numpy(),
                                      f.loc[f.index != "H08", col].to_numpy())
        assert np.isnan(v.loc["H08", col])
    assert v.loc["H08", "flag"] is pd.NA and v.loc["H08", "status"] == "suppressed"
    np.testing.assert_array_equal(v["records"].to_numpy(), f["denominator"].to_numpy())
    assert v.attrs["test_method"] == "score"


def test_excel_cells_are_numeric_and_deterministic(table):
    pytest.importorskip("xlsxwriter")
    data = table.to_excel()
    assert data == table.to_excel()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        sheet = z.read("xl/worksheets/sheet1.xml").decode()
        assert "xl/worksheets/sheet2.xml" in z.namelist()
    assert '<pane ySplit="1"' in sheet                                    # frozen header row
    numbers = re.findall(r'<c r="E(\d+)"(?: s="\d+")?><v>([^<]+)</v></c>', sheet)   # column E: estimate
    assert ("2", "1.25") in numbers and ("9", "1.25") not in numbers     # numeric cells; H08's value is suppressed


def test_capability_errors():
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"y": rng.normal(size=600), "x1": rng.normal(size=600), "provider_id": np.repeat(np.arange(20), 30)})
    lin = LinearFixedEffectModel()
    lin.fit(d, y_var="y", x_vars=["x1"], provider_var="provider_id")
    assert "observed" not in provider_table(lin).spec.cells.columns
    with pytest.raises(CapabilityError, match="observed"):
        provider_table(lin, columns=["provider", "observed"])
    with pytest.raises(ValueError, match="unknown columns"):
        provider_table(lin, columns=["rank"])
    with pytest.raises(CapabilityError, match="denominators="):
        provider_table(lin.test(), columns=["provider", "n"])


def test_models_as_sources():
    rng = np.random.default_rng(5)
    pid = np.repeat(np.arange(25), rng.integers(30, 80, 25))
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1.0 + 0.5 * x))))
    fe = LogisticFixedEffectModel()
    fe.fit(pd.DataFrame({"y": y, "x1": x, "provider_id": pid}), y_var="y", x_vars=["x1"], provider_var="provider_id")
    t = provider_table(fe)
    assert list(t.spec.cells.columns) == ["provider", "n", "observed", "estimate_ci", "flag"]
    assert "exact Poisson-binomial test" in dict(t.spec.notes)["b"]
    prov = np.repeat(np.arange(12), 40)
    X = rng.normal(size=(prov.size, 2))
    tt = rng.exponential(1 / (0.3 * np.exp(X @ [0.5, -0.3])))
    c = rng.uniform(0.5, 3, prov.size)
    data = dict(duration=np.minimum(tt, c), event=(tt <= c).astype(float), provider_id=prov)
    cox = CoxPH(ties="breslow").fit(X, duration=data["duration"], event=data["event"], strata=prov)
    tc = provider_table(cox, X, **data)
    assert tc.spec.columns[1].header == "Expected events" and "expected" not in tc.spec.cells.columns
