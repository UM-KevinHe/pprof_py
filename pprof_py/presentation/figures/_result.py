"""FigureResult: a rendered figure with deterministic export, alt text and provenance (ADR-006; §7.3, §7.4)."""
from __future__ import annotations

import re

import io
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Optional, Union
from xml.sax.saxutils import escape

__all__ = ["FigureResult"]

_FORMATS = ("svg", "pdf", "png")


class FigureResult:
    """A figure built by a presentation renderer.

    Every export renders inside the figure's theme, without timestamps, so the same input, theme and environment
    give byte-identical files. SVG output carries the alt text as ``<title>`` and the long description as ``<desc>``;
    PDF and PNG output carry them as document metadata.

    Attributes
    ----------
    figure : matplotlib.figure.Figure
        The underlying figure (documented escape hatch; no pyplot state is involved).
    axes : matplotlib.axes.Axes
        The main data axes.
    alt_text, long_description : str
        Short alternative text and a longer description built from the presentation data.
    provenance : mapping
        The provenance of the displayed profile.
    counts : mapping
        Providers per status and attribute, as shown.
    kind : str
        The display, for example ``"funnel"``.
    caption : str
        The figure's footnote text (provenance and how to read it), for outlets that set captions outside the
        figure, such as journals.

    Notes
    -----
    ``fig, ax = result`` unpacks to ``(figure, axes)``, as the earlier funnel functions returned.
    """

    __slots__ = ("_figure", "_axes", "_theme", "_alt", "_long", "_provenance", "_counts", "_kind", "_caption")

    def __init__(self, figure: Any, axes: Any, *, theme: Any, alt_text: str, long_description: str,
                 provenance: Mapping[str, Any], counts: Mapping[str, Any], kind: str,
                 caption: Optional[str] = None) -> None:
        for name, value in (("_figure", figure), ("_axes", axes), ("_theme", theme), ("_alt", alt_text),
                            ("_long", long_description), ("_provenance", MappingProxyType(dict(provenance))),
                            ("_counts", MappingProxyType(dict(counts))), ("_kind", kind),
                            ("_caption", "" if caption is None else str(caption))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FigureResult is immutable")

    figure = property(lambda self: self._figure)
    axes = property(lambda self: self._axes)
    theme = property(lambda self: self._theme)
    alt_text = property(lambda self: self._alt)
    long_description = property(lambda self: self._long)
    provenance = property(lambda self: self._provenance)
    counts = property(lambda self: self._counts)
    kind = property(lambda self: self._kind)
    caption = property(lambda self: self._caption)

    def __iter__(self) -> Iterator[Any]:
        return iter((self._figure, self._axes))

    def __repr__(self) -> str:
        return f"FigureResult({self._kind}: {self._alt})"

    def to_bytes(self, format: str = "svg", *, dpi: Optional[float] = None) -> bytes:
        """The figure as ``"svg"``, ``"pdf"`` or ``"png"`` bytes (deterministic; ``dpi`` defaults to the theme's)."""
        fmt = format.lower()
        if fmt not in _FORMATS:
            raise ValueError(f"format must be one of {_FORMATS}, got {format!r}")
        meta: Dict[str, Any] = {"svg": {"Date": None},
                                "pdf": {"CreationDate": None, "ModDate": None, "Title": self._alt,
                                        "Subject": self._long},
                                "png": {"Title": self._alt, "Description": self._long}}[fmt]
        buf = io.BytesIO()
        with self._theme.rc_context():
            self._figure.savefig(buf, format=fmt, metadata=meta, dpi=self._theme.dpi if dpi is None else dpi)
        data = buf.getvalue()
        if fmt == "svg":
            data = _accessible_svg(_canonical_ids(data.decode("utf-8")), self._alt, self._long).encode("utf-8")
        return data

    def to_svg(self) -> str:
        """The figure as an SVG document with ``<title>`` and ``<desc>``."""
        return self.to_bytes("svg").decode("utf-8")

    def save(self, path: Union[str, Path], *, format: Optional[str] = None, dpi: Optional[float] = None) -> Path:
        """Write the figure to ``path``; the format comes from ``format`` or the file extension."""
        p = Path(path)
        fmt = (format or p.suffix.lstrip(".")).lower()
        p.write_bytes(self.to_bytes(fmt, dpi=dpi))
        return p

    def _repr_svg_(self) -> str:
        return self.to_svg()

    def _repr_png_(self) -> bytes:
        return self.to_bytes("png", dpi=150)


_HASHED_ID = re.compile(r'id="([pm][0-9a-f]{10})"')
_HASHED_TOKEN = re.compile(r"\b[pm][0-9a-f]{10}\b")


def _canonical_ids(svg: str) -> str:
    """Rename Matplotlib's content-hashed SVG ids (clip paths ``p...``, markers ``m...``) in order of definition.

    The hashes take the full-precision ``repr`` of coordinates, so last-bit noise from text measurement (which
    depends on earlier drawing and on string-hash order) changed them although the drawing was identical
    (ADR-006). The ids are opaque; the drawn coordinates are unchanged.
    """
    mapping: Dict[str, str] = {}
    for old in _HASHED_ID.findall(svg):
        if old not in mapping:
            mapping[old] = f"{old[0]}{len(mapping) + 1:05d}"
    if not mapping:
        return svg
    return _HASHED_TOKEN.sub(lambda m: mapping.get(m.group(0), m.group(0)), svg)


def _accessible_svg(svg: str, title: str, desc: str) -> str:
    """Label the root ``<svg>`` element for assistive technology with fixed ids (deterministic)."""
    start = svg.index("<svg")
    end = svg.index(">", start)
    head = svg[:end] + ' role="img" aria-labelledby="pprof_py-title pprof_py-desc"'
    return (head + ">\n <title id=\"pprof_py-title\">" + escape(title) + "</title>\n <desc id=\"pprof_py-desc\">"
            + escape(desc) + "</desc>" + svg[end + 1:])
