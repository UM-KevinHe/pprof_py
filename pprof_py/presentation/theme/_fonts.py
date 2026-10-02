"""The bundled typeface: IBM Plex Sans, applied per text element and never registered globally.

The publication, notebook and report presets set text in IBM Plex Sans. Its four styles that the renderers use
(Regular, Italic, Medium for labels, SemiBold for titles) ship unmodified in ``fonts/`` under the SIL Open Font
License 1.1; the copyright notice and licence are in ``fonts/OFL.txt`` and the provenance in ``fonts/README.txt``.

Each text element of a figure gets its face through ``FontProperties(fname=...)`` when the figure's layout is
frozen (:func:`apply`). Matplotlib's font manager and rcParams are never changed, so other figures in the same
process are unaffected, and the themes' rcParams name only DejaVu Sans, which ships with Matplotlib.

Fail-safe in two ways. A missing or unreadable font file makes text fall back to DejaVu Sans, with one warning.
A text containing a character the face lacks (the table symbols ``▲ ▼ ●``, a superscript minus) is set in
DejaVu Sans as a whole, so no glyph ever renders as a box.
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, FrozenSet, Optional

FONT_DIR = Path(__file__).resolve().parent / "fonts"
BUNDLED_FAMILY = "IBM Plex Sans"
FALLBACK_FAMILY = "DejaVu Sans"
_FILES = {("normal", "normal"): "IBMPlexSans-Regular.ttf", ("normal", "italic"): "IBMPlexSans-Italic.ttf",
          ("medium", "normal"): "IBMPlexSans-Medium.ttf", ("semibold", "normal"): "IBMPlexSans-SemiBold.ttf"}
_WEIGHTS = {"ultralight": 200, "light": 300, "normal": 400, "regular": 400, "book": 400, "roman": 400,
            "medium": 500, "semibold": 600, "demibold": 600, "demi": 600, "bold": 700, "heavy": 800,
            "extra bold": 800, "black": 900}

_WARNED: set = set()                               # directories already reported (one warning each)

__all__ = ["BUNDLED_FAMILY", "FALLBACK_FAMILY", "FONT_DIR", "apply", "available", "font_file", "font_properties",
           "is_bundled"]


def is_bundled(family: str) -> bool:
    """Whether ``family`` is the bundled face (applied per element) rather than one Matplotlib resolves itself."""
    return family == BUNDLED_FAMILY


def _style_key(weight: Any, style: Any) -> tuple:
    """The bundled file for a requested weight and style: italic of any weight uses the one italic; 450 and below
    use Regular, up to 550 Medium, heavier SemiBold (the heaviest bundled weight)."""
    if str(style).lower() in ("italic", "oblique"):
        return ("normal", "italic")
    if isinstance(weight, (int, float)):
        w = float(weight)
    else:
        w = float(_WEIGHTS.get(str(weight).lower(), 400))
    return ("normal" if w < 450 else "medium" if w < 550 else "semibold", "normal")


@lru_cache(maxsize=None)
def _readable(path: str) -> bool:
    try:
        from matplotlib.ft2font import FT2Font
        FT2Font(path)
        return True
    except Exception:                              # missing, truncated or not a font: fall back
        return False


@lru_cache(maxsize=None)
def font_file(weight: Any = "normal", style: Any = "normal") -> Optional[str]:
    """Path of the bundled file for ``weight`` and ``style``, or ``None`` (with one warning) when it is unavailable."""
    name = _FILES[_style_key(weight, style)]
    path = str(FONT_DIR / name)
    if _readable(path):
        return path
    _warn_unavailable(str(FONT_DIR), name)
    return None


@lru_cache(maxsize=None)
def _warn_unavailable(directory: str, name: str) -> None:
    """Warn about a missing or unreadable bundled font once per directory (cached on the directory only)."""
    _warned_dirs(directory, name)


@lru_cache(maxsize=None)
def _warned_dirs(directory: str, name: str) -> None:
    del name
    if directory in _WARNED:
        return
    _WARNED.add(directory)
    warnings.warn(f"pprof_py: the bundled IBM Plex Sans files are missing or unreadable in {directory}; figure "
                  f"text falls back to {FALLBACK_FAMILY}", UserWarning, stacklevel=4)


def available() -> bool:
    """Whether the bundled Regular style can be loaded (the condition for applying the face at all)."""
    return font_file("normal", "normal") is not None


@lru_cache(maxsize=None)
def _charmap(path: str) -> FrozenSet[int]:
    from matplotlib.ft2font import FT2Font
    return frozenset(FT2Font(path).get_charmap())


def _covers(path: str, text: str) -> bool:
    cmap = _charmap(path)
    return all(ord(ch) in cmap for ch in text if not ch.isspace())


def font_properties(family: str, size: float, weight: Any = "normal", style: Any = "normal",
                    text: Optional[str] = None) -> Any:
    """``FontProperties`` for one text element: the bundled file when ``family`` is the bundled face, the file is
    available and it has every glyph of ``text``; otherwise Matplotlib's own lookup of ``family`` (DejaVu Sans in
    place of the bundled face)."""
    from matplotlib.font_manager import FontProperties
    if is_bundled(family):
        path = font_file(weight, style)
        if path is not None and (text is None or _covers(path, text)):
            # the file decides the face; weight and style are kept so the element still reports what it asked for
            return FontProperties(fname=path, size=size, weight=weight, style=style)
        family = FALLBACK_FAMILY
    return FontProperties(family=family, size=size, weight=weight, style=style)


def apply(fig: Any, family: str, title_weight: Any = "normal", label_weight: Any = "normal") -> None:
    """Give every text element of ``fig`` the face for ``family``, keeping its size and style.

    Axis labels take ``label_weight``, and titles and text tagged ``gid="emphasis"`` (a highlighted provider's
    label, for example) take ``title_weight``; the themes keep rcParams and renderers keep text at normal weight for
    the bundled face, so Matplotlib never looks for a medium or semibold DejaVu Sans. Other text keeps its weight.
    Called once the figure has been drawn (so tick labels exist) and before the layout is frozen, so the frozen
    layout is measured with the face it shows. Does nothing unless ``family`` is the bundled face and its files load;
    if anything fails, the figure keeps DejaVu Sans throughout (never a mixture).
    """
    if not is_bundled(family) or not available():
        return
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.text import Text
    try:
        role = {}
        for ax in fig.findobj(Axes):
            for lab in (ax.xaxis.label, ax.yaxis.label):
                role[id(lab)] = label_weight
            for ttl in (getattr(ax, "title", None), getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
                if ttl is not None:
                    role[id(ttl)] = title_weight
        for f in fig.findobj(lambda a: isinstance(a, (Figure, SubFigure))):
            if getattr(f, "_suptitle", None) is not None:
                role[id(f._suptitle)] = title_weight
        texts = [t for t in fig.findobj(Text)]
        for t in texts:
            if t.get_gid() == "emphasis":
                role[id(t)] = title_weight
        props = []
        for t in texts:
            fp = t.get_fontproperties()
            props.append(font_properties(family, fp.get_size_in_points(), role.get(id(t), fp.get_weight()),
                                         fp.get_style(), t.get_text()))
    except Exception as exc:                       # pragma: no cover - defensive: keep the rc fonts throughout
        warnings.warn(f"pprof_py: could not apply {family} ({exc.__class__.__name__}: {exc}); figure text uses "
                      f"{FALLBACK_FAMILY}", UserWarning, stacklevel=2)
        return
    for t, p in zip(texts, props):
        t.set_fontproperties(p)


def _reset() -> None:
    """Forget cached lookups (tests that move the font directory use this)."""
    for f in (_readable, font_file, _charmap, _warn_unavailable, _warned_dirs):
        f.cache_clear()
    _WARNED.clear()
