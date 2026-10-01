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
