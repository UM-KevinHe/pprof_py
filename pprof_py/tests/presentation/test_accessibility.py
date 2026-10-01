"""Accessibility requirements as automated tests (brief §7.4, ADR-008).

Thresholds are CIE76 Delta E in CIELAB under the Machado (2009) simulations; the shipped palette's worst cases are
59.1 (above/below, tritan) and 33.7 (below/not different, tritan), and a red/green pair falls to 7.3 (deutan).
"""
import numpy as np
import pytest

from pprof_py.presentation import Theme
from pprof_py.presentation.theme import _accessibility as acc

PRESETS = [Theme.publication(), Theme.notebook(), Theme.report()]


def test_simulation_matrices_keep_white_white():
    for m in acc._MACHADO.values():
        np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-5)


def test_contrast_formula_reference_values():
    assert acc.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert acc.contrast_ratio("#777777") == pytest.approx(4.48, abs=0.01)


@pytest.mark.parametrize("theme", PRESETS, ids=lambda t: t.name)
def test_contrast(theme):
    r = theme.accessibility_report()
    assert min(r["status_contrast"].values()) >= 3.0
    assert min(r["line_contrast"].values()) >= 3.0
    assert r["text_contrast"] >= 4.5


@pytest.mark.parametrize("theme", PRESETS, ids=lambda t: t.name)
def test_colour_vision_and_grayscale(theme):
    r = theme.accessibility_report()
    assert min(r["delta_e"]["above/below"].values()) >= 40
    assert min(min(r["delta_e"][p].values()) for p in ("above/not_different", "below/not_different")) >= 25
    assert r["lightness_gap_above_below"] >= 15
    assert r["unique_encodings"]


def test_red_green_pair_fails_the_colour_vision_check():
    rg = Theme().derive(status={"above": {"color": "#D62728"}, "below": {"color": "#2CA02C"}})
    assert min(rg.accessibility_report()["delta_e"]["above/below"].values()) < 40
