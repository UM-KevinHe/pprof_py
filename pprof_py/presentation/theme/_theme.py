"""Immutable design tokens.

A :class:`Theme` holds every visual decision: typography, line widths, status encodings, sizes and export settings.
Renderers read tokens from a theme; users customise by deriving a new theme (:meth:`Theme.derive`), not through
per-call styling arguments. The defaults follow the package's house style: restrained, publication-quality and
accessible, with uncertainty and provider volume always in view.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple, Union

STATUS_KEYS: Tuple[str, ...] = ("above", "below", "not_different", "not_tested", "no_finite_estimate")
"""The provider statuses every display encodes (direction relative to the reference, never good or bad)."""


@dataclass(frozen=True)
class StatusStyle:
    """Encoding of one provider status.

    Attributes
    ----------
    color : str
        Hue as ``#RRGGBB`` (the edge colour when the marker is hollow).
    marker : str
        Matplotlib marker code.
    size : float
        Marker area in points squared.
    filled : bool
        Solid (``True``) or hollow marker.
    label : str
        Direction-neutral legend label.
    symbol : str
        Symbol used in tables and text.
    """

    color: str
    marker: str
    size: float
    filled: bool
    label: str
    symbol: str


@dataclass(frozen=True)
class Typography:
    """Font family and sizes in points, by role."""

    family: str = "DejaVu Sans"
    title: float = 8.5
    subtitle: float = 7.5
    label: float = 7.5
    tick: float = 7.0
    legend: float = 7.0
    annotation: float = 7.0
    footnote: float = 7.0

    @property
    def minimum(self) -> float:
        """Smallest size in use."""
        return min(self.title, self.subtitle, self.label, self.tick, self.legend, self.annotation, self.footnote)


@dataclass(frozen=True)
class Lines:
    """Line widths in points."""

    axis: float = 0.6
    data: float = 0.8
    interval: float = 0.8
    interval_dense: float = 0.35
    reference: float = 0.8
    limit: float = 0.7
    grid: float = 0.4


def _status_styles(scale: float = 1.0) -> Dict[str, StatusStyle]:
    return {
        "above": StatusStyle("#B35806", "^", 18.0 * scale, True, "Above reference", "\u25b2"),
        "below": StatusStyle("#542788", "v", 18.0 * scale, True, "Below reference", "\u25bc"),
        "not_different": StatusStyle("#8C8C8C", "o", 7.0 * scale, True, "Not different", "\u25cf"),
        "not_tested": StatusStyle("#4D4D4D", "o", 10.0 * scale, False, "Not tested", "NT"),
        "no_finite_estimate": StatusStyle("#000000", "s", 22.0 * scale, False, "No finite estimate", "NE"),
    }


def _hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


@dataclass(frozen=True)
class Theme:
    """Design tokens for figures, tables and reports.

    Use a preset (:meth:`publication`, the default, :meth:`notebook`, :meth:`report`) and derive variants with
    :meth:`derive`. Themes are immutable and hashable.

    Attributes
    ----------
    name : str
        Preset or user-chosen name.
    typography, lines : Typography, Lines
        Font sizes and line widths.
    status : mapping
        One :class:`StatusStyle` per key in :data:`STATUS_KEYS`.
    ink, muted, reference, limit, volume, background : str
        Text and axis ink, secondary ink, reference line, control limits, volume bars, background.
    grid : bool
        Draw grid lines (off by default) in ``grid_color``.
    level_dashes : mapping
        Dash pattern for each confidence or control level.
    offscale_markers : tuple of str
        Markers for values beyond the lower and upper axis limits.
    widths_mm : mapping
        Named figure widths in millimetres (``"single"``, ``"double"``).
    dpi : int
        Raster resolution for PNG export.
    svg_hashsalt : str
        Fixed salt so SVG element ids are reproducible.
    svg_text_as_paths : bool
        Draw SVG text as paths (identical rendering everywhere) instead of editable text.
    """

    name: str = "publication"
    typography: Typography = field(default_factory=Typography)
    lines: Lines = field(default_factory=Lines)
    status: Mapping[str, StatusStyle] = field(default_factory=_status_styles)
    ink: str = "#1A1A1A"
    muted: str = "#4D4D4D"
    reference: str = "#000000"
    limit: str = "#4D4D4D"
    volume: str = "#8C8C8C"
    background: str = "#FFFFFF"
    grid: bool = False
    grid_color: str = "#D9D9D9"
    level_dashes: Mapping[float, Any] = field(
        default_factory=lambda: {0.95: (0, (4.0, 2.0)), 0.998: (0, (1.0, 1.5))})
    offscale_markers: Tuple[str, str] = ("<", ">")
    widths_mm: Mapping[str, float] = field(default_factory=lambda: {"single": 85.0, "double": 175.0})
    dpi: int = 300
    svg_hashsalt: str = "pprof_py"
    svg_text_as_paths: bool = True

    def __post_init__(self) -> None:
        for name in ("status", "level_dashes", "widths_mm"):
            value = getattr(self, name)
            if not isinstance(value, MappingProxyType):
                object.__setattr__(self, name, MappingProxyType(dict(value)))
        missing = [k for k in STATUS_KEYS if k not in self.status]
        extra = [k for k in self.status if k not in STATUS_KEYS]
        if missing or extra:
            raise ValueError(f"status must have exactly the keys {STATUS_KEYS}; missing {missing}, unknown {extra}")
        for key, style in self.status.items():
            if not isinstance(style, StatusStyle):
                raise TypeError(f"status[{key!r}] must be a StatusStyle, got {type(style).__name__}")
        if self.typography.minimum <= 0:
            raise ValueError("font sizes must be positive")

    def __hash__(self) -> int:
        return hash(tuple((f.name, _hashable(getattr(self, f.name))) for f in dataclasses.fields(self)))

    # presets -----------------------------------------------------------------------------------------------
    @classmethod
    def publication(cls) -> "Theme":
        """Print-ready defaults: 85/175 mm widths, text at least 7 pt at final size, 300 dpi."""
        return cls()

    @classmethod
    def notebook(cls) -> "Theme":
        """Larger text and marks for on-screen work in notebooks."""
        return cls(name="notebook",
                   typography=Typography(title=12.0, subtitle=11.0, label=11.0, tick=10.0, legend=10.0,
                                         annotation=10.0, footnote=9.5),
                   lines=Lines(axis=0.8, data=1.0, interval=1.0, interval_dense=0.45, reference=1.0, limit=0.9,
                               grid=0.5),
                   status=_status_styles(2.0), widths_mm={"single": 120.0, "double": 200.0}, dpi=150)

    @classmethod
    def report(cls) -> "Theme":
        """Defaults for HTML reports and slides."""
        return cls(name="report",
                   typography=Typography(title=13.0, subtitle=11.0, label=11.0, tick=10.0, legend=10.0,
                                         annotation=10.0, footnote=10.0),
                   lines=Lines(axis=0.8, data=1.0, interval=1.0, interval_dense=0.45, reference=1.0, limit=0.9,
                               grid=0.5),
                   status=_status_styles(2.0), widths_mm={"single": 120.0, "double": 180.0}, dpi=200)

    # derivation and use ------------------------------------------------------------------------------------
    def derive(self, **changes: Any) -> "Theme":
        """Return a new theme with some tokens changed.

        Groups take a mapping of the fields to change, for example
        ``theme.derive(typography={"tick": 7.5}, status={"above": {"color": "#8C510A"}})``.
        """
        names = {f.name for f in dataclasses.fields(self)}
        updates: Dict[str, Any] = {}
        for key, value in changes.items():
            if key not in names:
                raise TypeError(f"Theme has no token {key!r}")
            current = getattr(self, key)
            if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
                updates[key] = dataclasses.replace(current, **value)
            elif key == "status" and isinstance(value, Mapping):
                merged = dict(current)
                for status, style in value.items():
                    if status not in merged:
                        raise KeyError(f"unknown status {status!r}; expected one of {STATUS_KEYS}")
                    merged[status] = dataclasses.replace(merged[status], **style) if isinstance(style, Mapping) else style
                updates[key] = merged
            elif key in ("level_dashes", "widths_mm") and isinstance(value, Mapping):
                updates[key] = {**current, **value}
            else:
                updates[key] = value
        return dataclasses.replace(self, **updates)

    def rc(self) -> Dict[str, Any]:
        """Matplotlib rcParams for this theme (renderers apply them while building and while saving)."""
        t, ln = self.typography, self.lines
        return {
            "font.family": "sans-serif", "font.sans-serif": [t.family, "DejaVu Sans"], "font.size": t.tick,
            "axes.titlesize": t.title, "axes.labelsize": t.label, "figure.titlesize": t.title,
            "xtick.labelsize": t.tick, "ytick.labelsize": t.tick, "legend.fontsize": t.legend,
            "axes.linewidth": ln.axis, "xtick.major.width": ln.axis, "ytick.major.width": ln.axis,
            "xtick.minor.width": 0.75 * ln.axis, "ytick.minor.width": 0.75 * ln.axis,
            "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.edgecolor": self.ink, "axes.labelcolor": self.ink, "text.color": self.ink,
            "xtick.color": self.ink, "ytick.color": self.ink,
            "axes.facecolor": self.background, "figure.facecolor": self.background,
            "savefig.facecolor": self.background, "axes.grid": self.grid, "grid.color": self.grid_color,
            "grid.linewidth": ln.grid, "legend.frameon": False, "axes.unicode_minus": True,
            "svg.hashsalt": self.svg_hashsalt, "svg.fonttype": "path" if self.svg_text_as_paths else "none",
            "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": self.dpi, "path.simplify": True,
        }

    def rc_context(self) -> Any:
        """A ``matplotlib.rc_context`` applying :meth:`rc` (Matplotlib is imported only here)."""
        import matplotlib
        return matplotlib.rc_context(self.rc())

    def figsize(self, size: Union[str, float] = "single", height_mm: Optional[float] = None, *,
                aspect: float = 0.8) -> Tuple[float, float]:
        """Figure size in inches for a named width (``"single"``, ``"double"``) or a width in millimetres."""
        if isinstance(size, str):
            if size not in self.widths_mm:
                raise ValueError(f"unknown size {size!r}; choose one of {sorted(self.widths_mm)} or a width in mm")
            width = float(self.widths_mm[size])
        else:
            width = float(size)
        height = width * aspect if height_mm is None else float(height_mm)
        return width / 25.4, height / 25.4

    def accessibility_report(self) -> Dict[str, Any]:
        """Contrast, colour-vision and encoding metrics for this theme (see ``theme._accessibility``)."""
        from ._accessibility import accessibility_report
        return accessibility_report(self)


_PRESETS = {"publication": Theme.publication, "notebook": Theme.notebook, "report": Theme.report}


def get_theme(theme: Union[str, Theme, None] = None) -> Theme:
    """Resolve ``None`` (publication), a preset name or a :class:`Theme`."""
    if theme is None:
        return Theme.publication()
    if isinstance(theme, Theme):
        return theme
    if isinstance(theme, str):
        if theme not in _PRESETS:
            raise ValueError(f"unknown theme {theme!r}; choose one of {sorted(_PRESETS)} or pass a Theme")
        return _PRESETS[theme]()
    raise TypeError(f"theme must be None, a preset name or a Theme, got {type(theme).__name__}")
