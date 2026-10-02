"""Accessibility requirements as automated tests (brief §7.4, ADR-008).

Thresholds are CIE76 Delta E in CIELAB under the Machado (2009) simulations. The identity palette's worst cases are
65.2 (above/below) and 26.3 (status/not different), the classic palette's 59.1 and 33.7, and a red/green pair falls
to 7.3 (deutan). Status marks also keep 3:1 against the corridor fill, where most not-different providers sit (D73).
"""
import numpy as np
import pytest

from pprof_py.presentation import Theme
from pprof_py.presentation.theme import _accessibility as acc

PRESETS = [Theme.publication(), Theme.notebook(), Theme.report(), Theme.classic(), Theme.classic("notebook"),
           Theme.classic("report")]


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


@pytest.mark.parametrize("theme", PRESETS, ids=lambda t: t.name)
def test_status_marks_keep_contrast_on_the_corridor(theme):
    r = theme.accessibility_report()
    if theme.corridor is None:
        assert r["corridor_contrast"] == {}
    else:
        assert min(r["corridor_contrast"].values()) >= 3.0


@pytest.mark.parametrize("theme", PRESETS, ids=lambda t: t.name)
def test_marks_on_interval_bars_keep_contrast(theme):
    r = theme.accessibility_report()
    if theme.bar_marks is None:
        assert r["bar_mark_contrast"] == {}
    else:
        assert min(r["bar_mark_contrast"].values()) >= 3.0


def test_the_mockup_grey_fails_on_the_corridor():
    # negative control: the design mockups' not-different grey (#87919D) reaches only 2.78:1 on the corridor
    t = Theme().derive(status={"not_different": {"color": "#87919D"}})
    assert min(t.accessibility_report()["corridor_contrast"].values()) < 3.0


def test_red_green_pair_fails_the_colour_vision_check():
    rg = Theme().derive(status={"above": {"color": "#D62728"}, "below": {"color": "#2CA02C"}})
    assert min(rg.accessibility_report()["delta_e"]["above/below"].values()) < 40
