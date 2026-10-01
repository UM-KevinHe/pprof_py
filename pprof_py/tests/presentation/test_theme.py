"""Design tokens: presets, immutability, derivation, Matplotlib settings and fonts."""
import dataclasses

import pytest

from pprof_py.presentation import STATUS_KEYS, Theme, get_theme


def test_presets_and_resolution():
    assert get_theme() == Theme.publication() == Theme()
    assert get_theme("notebook").name == "notebook" and get_theme("report").name == "report"
    t = Theme.notebook()
    assert get_theme(t) is t
    with pytest.raises(ValueError):
        get_theme("dark")
    with pytest.raises(TypeError):
        get_theme(3)


def test_publication_text_is_at_least_7pt():
    assert Theme.publication().typography.minimum >= 7.0
    assert Theme.notebook().typography.minimum >= 9.0 and Theme.report().typography.minimum >= 9.0


def test_themes_are_immutable_and_hashable():
    t = Theme()
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.dpi = 72
    with pytest.raises(TypeError):
        t.status["above"] = t.status["below"]
    with pytest.raises(TypeError):
        t.widths_mm["single"] = 90.0
    assert hash(t) == hash(Theme()) and {t: 1}[Theme()] == 1


def test_derive_returns_a_new_theme():
    t = Theme()
    d = t.derive(typography={"tick": 7.5}, status={"above": {"color": "#8C510A"}}, widths_mm={"single": 89.0},
                 dpi=600)
    assert (d.typography.tick, d.status["above"].color, d.widths_mm["single"], d.dpi) == (7.5, "#8C510A", 89.0, 600)
    assert (t.typography.tick, t.status["above"].color, t.widths_mm["single"], t.dpi) == (7.0, "#B35806", 85.0, 300)
    assert d.status["below"] == t.status["below"] and d.widths_mm["double"] == 175.0
    with pytest.raises(TypeError):
        t.derive(colour="#000000")
    with pytest.raises(KeyError):
        t.derive(status={"worse": {"color": "#000000"}})
    with pytest.raises(ValueError):
        Theme(status={k: v for k, v in t.status.items() if k != "not_tested"})


def test_status_keys_and_direction_neutral_labels():
    t = Theme()
    assert tuple(t.status) == STATUS_KEYS
    labels = " ".join(s.label.lower() for s in t.status.values())
    assert not any(word in labels for word in ("better", "worse", "good", "bad"))


def test_rc_settings_are_valid_and_apply():
    import matplotlib
    for theme in (Theme.publication(), Theme.notebook(), Theme.report()):
        assert set(theme.rc()) <= set(matplotlib.rcParams)
        with theme.rc_context():
            assert matplotlib.rcParams["svg.hashsalt"] == "pprof_py"
            assert matplotlib.rcParams["axes.spines.top"] is False


def test_figure_sizes():
    w, h = Theme().figsize("single", 70)
    assert (round(w * 25.4, 6), round(h * 25.4, 6)) == (85.0, 70.0)
    assert round(Theme().figsize("double")[0] * 25.4, 6) == 175.0
    with pytest.raises(ValueError):
        Theme().figsize("triple")


def test_default_font_ships_with_matplotlib():
    import matplotlib
    from matplotlib import font_manager
    path = font_manager.findfont(font_manager.FontProperties(family=Theme().typography.family),
                                 fallback_to_default=False)
    assert path.startswith(matplotlib.get_data_path()) and path.endswith("DejaVuSans.ttf")
