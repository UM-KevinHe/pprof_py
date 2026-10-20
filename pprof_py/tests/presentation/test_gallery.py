"""The docs gallery extension builds every figure (the docs build would otherwise fail first)."""
import importlib.util
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parents[2] / "docs" / "source" / "_ext" / "pprof_gallery.py"


@pytest.mark.skipif(not EXT.exists(), reason="docs sources not present (installed package)")
def test_gallery_figures():
    spec = importlib.util.spec_from_file_location("pprof_gallery", EXT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    figs = dict(mod.figures())
    assert list(figs) == list(mod.ITEMS)
    from pprof_py.presentation import FigureResult

    assert all(isinstance(f, FigureResult) and f.alt_text for f in figs.values())
    assert figs["measure_agreement"].counts["not_placed"] > 0          # zero-event providers are not placed
    # The page stretches every image to the column width, so all figures share the interval plot's width and its
    # type sizes: one scale, one type size (heights follow each display's content).
    from matplotlib.text import Text

    def type_sizes(fig):
        return {round(t.get_fontsize(), 2) for t in fig.figure.findobj(Text) if t.get_text().strip() and t.get_visible()}

    ref = figs["caterpillar"]
    for name, fig in figs.items():
        assert fig.figure.get_size_inches()[0] == pytest.approx(ref.figure.get_size_inches()[0], abs=1e-9), name
        assert type_sizes(fig) <= type_sizes(ref), name


SHEET = ("Question", "Quantity and source", "Uncertainty", "Denominator", "Reference", "Misreadings and mitigations",
         "Static or interactive", "From 10 to 50,000 providers")


@pytest.mark.skipif(not EXT.exists(), reason="docs sources not present (installed package)")
def test_every_display_has_its_page_and_spec_sheet():
    """Brief §1: each display has a published spec sheet (§5.2) and explains how to read it and how it misleads."""
    spec = importlib.util.spec_from_file_location("pprof_gallery", EXT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pages = EXT.parents[1] / "presentation" / "displays"
    assert set(mod.PAGES) == set(mod.ITEMS)
    for name, page in mod.PAGES.items():
        text = (pages / f"{page}.md").read_text()
        assert f"../_figures/{name}.md" in text, (page, name)
        for heading in ("## Spec sheet", "## How to read it", "## How it can mislead"):
            assert heading in text, (page, heading)
        rows = [line.split("|")[1].strip() for line in text.splitlines() if line.startswith("| ") and "|" in line[2:]]
        assert all(item in rows for item in SHEET), (page, [i for i in SHEET if i not in rows])


INTERACTIVE = EXT.with_name("pprof_interactive.py")


@pytest.fixture(scope="module")
def interactive():
    """The interactive extension and its records, loaded as Sphinx loads it (it imports pprof_gallery by name)."""
    import sys

    sys.path.insert(0, str(EXT.parent))
    try:
        import pprof_gallery
        import pprof_interactive
    finally:
        sys.path.remove(str(EXT.parent))
    return pprof_gallery, pprof_interactive, pprof_interactive.records()


@pytest.mark.skipif(not INTERACTIVE.exists(), reason="docs sources not present (installed package)")
def test_interactive_displays_keep_the_static_figure(interactive):
    """Interactive-first display pages still carry the gallery's static figure, in the collapsed block that prints."""
    _gallery, mod, recs = interactive
    assert set(recs) == set(mod.DISPLAYS)
    for name in mod.DISPLAYS:
        snippet = mod._snippet(name, recs[name]).decode("utf-8")
        static = mod._static_snippet(name)
        assert f"/_static/gallery/{name}.svg" in static, name
        head, sep, tail = snippet.partition('<details class="pprof-static-figure">')
        assert sep and static in tail and static not in head, name
        assert "documentation only" in head, name


@pytest.mark.skipif(not INTERACTIVE.exists(), reason="docs sources not present (installed package)")
def test_interactive_interval_plot_shows_the_validated_values(interactive):
    """The interactive interval plot carries the profile's flags, estimates, intervals and p-values (to 6 decimals)."""
    gallery, mod, recs = interactive
    f = gallery.fitted()["exact"].data
    ids = {str(i): i for i in f.index}
    shown = recs["caterpillar"]["providers"]
    assert shown and len({p["id"] for p in shown}) == len(shown)
    for p in shown:
        r = f.loc[ids[p["id"]]]
        assert p["status"] == str(r["status"]), p["id"]
        for key, col in (("lower", "ci_lower"), ("upper", "ci_upper"), ("p_value", "p_value"),
                         ("observed", "observed"), ("expected", "expected")):
            assert p[key] == mod._num(r[col]), (p["id"], key)
        if p["estimate"] is not None:
            assert p["estimate"] == mod._num(r["estimate"]), p["id"]
