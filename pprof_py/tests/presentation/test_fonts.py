"""The bundled typeface (D70, D71): files and licence, per-element loading without global state, the fail-safe
fallback to DejaVu Sans, the classic preset and the exported caption."""
import hashlib
import re

import matplotlib
import pytest

from pprof_py.presentation import Theme, funnel
from pprof_py.presentation.theme import _fonts

STYLES = ("Regular", "Italic", "Medium", "SemiBold")


@pytest.fixture(scope="module")
def model():
    from pprof_py import LogisticFixedEffectModel
    from pprof_py.presentation._synthetic import provider_data
    m = LogisticFixedEffectModel()
    m.fit(provider_data(40), y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    return m


@pytest.fixture
def fresh_fonts():
    _fonts._reset()
    yield
    _fonts._reset()


def _text_files(fig):
    from matplotlib.text import Text
    return [(t.get_text(), t.get_fontproperties().get_file()) for t in fig.findobj(Text) if t.get_text()]


def _is_bundled(path):
    return bool(path) and str(path).startswith(str(_fonts.FONT_DIR)) and "IBMPlexSans-" in str(path)


def test_bundled_files_are_the_unmodified_releases_with_their_licence():
    from matplotlib.ft2font import FT2Font
    readme = (_fonts.FONT_DIR / "README.txt").read_text(encoding="utf-8")
    for style in STYLES:
        path = _fonts.FONT_DIR / f"IBMPlexSans-{style}.ttf"
        assert FT2Font(str(path)).family_name == "IBM Plex Sans"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert re.search(rf"IBMPlexSans-{style}\.ttf\s+\d+ bytes\s+sha256 {digest}", readme), style
    licence = (_fonts.FONT_DIR / "OFL.txt").read_text(encoding="utf-8")
    assert 'Copyright \u00a9 2017 IBM Corp. with Reserved Font Name "Plex"' in licence
    assert "SIL OPEN FONT LICENSE Version 1.1" in licence and "\r" not in licence


def test_weights_and_styles_map_to_the_four_bundled_files(fresh_fonts):
    name = lambda *a: _fonts.font_file(*a).rsplit("/", 1)[-1]      # noqa: E731
    assert name("normal", "normal") == name(300, "normal") == "IBMPlexSans-Regular.ttf"
    assert name("medium", "normal") == name(500, "normal") == "IBMPlexSans-Medium.ttf"
    assert name("semibold", "normal") == name("bold", "normal") == name(700, "normal") == "IBMPlexSans-SemiBold.ttf"
    assert name("semibold", "italic") == name("normal", "oblique") == "IBMPlexSans-Italic.ttf"


def test_figure_text_uses_the_bundled_face_and_changes_no_global_state(model, fresh_fonts, caplog):
    from matplotlib import font_manager
    before_rc = dict(matplotlib.rcParams)
    before_fonts = sorted(f.fname for f in font_manager.fontManager.ttflist)
    r = funnel(model)
    for fmt in ("svg", "pdf", "png"):
        r.to_bytes(fmt)
    assert dict(matplotlib.rcParams) == before_rc
    after_fonts = sorted(f.fname for f in font_manager.fontManager.ttflist)
    assert after_fonts == before_fonts and not any(f.startswith(str(_fonts.FONT_DIR)) for f in after_fonts)
    files = _text_files(r.figure)
    assert files and all(_is_bundled(path) for _, path in files), [t for t, p in files if not _is_bundled(p)]
    assert r.axes.xaxis.label.get_fontproperties().get_file().endswith("IBMPlexSans-Medium.ttf")   # label weight
    assert not [rec for rec in caplog.records if "findfont" in rec.getMessage()]                 # no font noise


def test_a_text_with_a_glyph_the_face_lacks_falls_back_as_a_whole(fresh_fonts):
    covered = _fonts.font_properties(_fonts.BUNDLED_FAMILY, 7.0, text="O/E 1.00 (0.91\u20131.46), E\u00b2/V\u2080")
    uncovered = _fonts.font_properties(_fonts.BUNDLED_FAMILY, 7.0, text="\u25b2 Above reference")
    assert _is_bundled(covered.get_file())
    assert uncovered.get_file() is None and uncovered.get_family() == [_fonts.FALLBACK_FAMILY]


@pytest.mark.parametrize("damage", ["missing", "unreadable"])
def test_missing_or_unreadable_files_fall_back_to_dejavu_sans(model, tmp_path, monkeypatch, fresh_fonts, damage):
    if damage == "unreadable":
        for style in STYLES:
            (tmp_path / f"IBMPlexSans-{style}.ttf").write_bytes(b"not a font")
    monkeypatch.setattr(_fonts, "FONT_DIR", tmp_path)
    _fonts._reset()
    with pytest.warns(UserWarning, match="missing or unreadable"):
        r = funnel(model)
    with _no_warning():
        data = r.to_bytes("png")                       # renders; the warning came once, at the first lookup
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert not any(_is_bundled(path) for _, path in _text_files(r.figure))


class _no_warning:
    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings(record=True)
        self._rec = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        assert not [w for w in self._rec if "missing or unreadable" in str(w.message)]
        return False


def test_classic_preset_keeps_dejavu_sans_and_the_0_6_0_tokens(model, fresh_fonts):
    c = Theme.classic()
    assert (c.typography.family, c.status["above"].color, c.status["below"].color, c.grid, c.spines,
            c.corridor) == ("DejaVu Sans", "#B35806", "#542788", False, ("left", "bottom"), None)
    assert c.rc()["font.sans-serif"] == ["DejaVu Sans", "DejaVu Sans"]
    assert Theme.classic("report").typography.title == 13.0 and Theme.classic("notebook").dpi == 150
    r = funnel(model, theme=c)
    assert not any(_is_bundled(path) for _, path in _text_files(r.figure))
    with pytest.raises(ValueError):
        Theme.classic("poster")


def test_caption_carries_the_footnote(model):
    r = funnel(model)
    assert r.caption and r.long_description == r.alt_text + " " + r.caption
    assert "Score test" in r.caption or "score test" in r.caption
