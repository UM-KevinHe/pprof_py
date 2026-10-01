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
