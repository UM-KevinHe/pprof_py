"""Immutable design tokens.

A :class:`Theme` holds every visual decision: typography, line widths, status encodings, sizes and export settings.
Renderers read tokens from a theme; users customise by deriving a new theme (:meth:`Theme.derive`), not through
per-call styling arguments. The presets carry the package's visual identity: IBM Plex Sans (bundled, applied per
text element), a direction-neutral copper and petrol status pair, a light grid instead of a frame, and a shaded
corridor for the test's acceptance region, with uncertainty and provider volume always in view. :meth:`Theme.classic`
keeps the 0.6.0 look.
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
    """Font family, sizes in points by role, and the weights of titles and axis labels.

    The family ``"IBM Plex Sans"`` is the bundled face: it is applied to each text element from the package's own
    font files and falls back to DejaVu Sans without them (``theme._fonts``). Any other family is looked up by
    Matplotlib as usual.
    """

    family: str = "IBM Plex Sans"
    title: float = 8.5
    subtitle: float = 7.5
    label: float = 7.5
    tick: float = 7.0
    legend: float = 7.0
    annotation: float = 7.0
    footnote: float = 7.0
    title_weight: str = "semibold"
    label_weight: str = "medium"

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
    reference: float = 0.9
    limit: float = 0.8
    grid: float = 0.5


# Status hues: "identity" is the package's look (copper / petrol, a cool grey that keeps 3:1 on the corridor);
# "classic" is the 0.6.0 dark PuOr pair. Both pass the accessibility tests (ADR-008, D73).
_PALETTES = {
    "identity": {"above": "#C25E1F", "below": "#0F4C63", "not_different": "#7D8793", "not_tested": "#556372",
                 "no_finite_estimate": "#1E2B38"},
    "classic": {"above": "#B35806", "below": "#542788", "not_different": "#8C8C8C", "not_tested": "#4D4D4D",
                "no_finite_estimate": "#000000"},
}


def _status_styles(scale: float = 1.0, palette: str = "identity") -> Dict[str, StatusStyle]:
    c = _PALETTES[palette]
    return {
        "above": StatusStyle(c["above"], "^", 18.0 * scale, True, "Above reference", "\u25b2"),
        "below": StatusStyle(c["below"], "v", 18.0 * scale, True, "Below reference", "\u25bc"),
        "not_different": StatusStyle(c["not_different"], "o", 7.0 * scale, True, "Not different", "\u25cf"),
        "not_tested": StatusStyle(c["not_tested"], "o", 10.0 * scale, False, "Not tested", "NT"),
        "no_finite_estimate": StatusStyle(c["no_finite_estimate"], "s", 22.0 * scale, False, "No finite estimate",
                                          "NE"),
    }


def _hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


@dataclass(frozen=True)
class Theme:
    """Design tokens for figures, tables and reports.

    Use a preset (:meth:`publication`, the default, :meth:`notebook`, :meth:`report`, or :meth:`classic` for the
    0.6.0 look) and derive variants with :meth:`derive`. Themes are immutable and hashable.

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
    corridor : str or None
        Fill for the region between the test's own limits (where a provider is not flagged); ``None`` draws none.
    halo : float
        Width in points of the background-coloured edge that separates markers from lines beneath them.
    spines : tuple of str
        Axis lines drawn (``"left"``, ``"bottom"``); empty draws none and lets the grid carry the scale.
    tick_length : float
        Major tick length in points (minor ticks are 0.6 of it).
    tick_label_color : str or None
        Tick label colour; ``None`` uses ``ink``.
    """

    name: str = "publication"
    typography: Typography = field(default_factory=Typography)
    lines: Lines = field(default_factory=Lines)
    status: Mapping[str, StatusStyle] = field(default_factory=_status_styles)
    ink: str = "#1E2B38"
    muted: str = "#556372"
    reference: str = "#1E2B38"
    limit: str = "#6B7684"
    volume: str = "#87919D"
    background: str = "#FFFFFF"
    grid: bool = True
    grid_color: str = "#E2E7ED"
    level_dashes: Mapping[float, Any] = field(
        default_factory=lambda: {0.95: "solid", 0.998: (0, (0.8, 1.6))})
    offscale_markers: Tuple[str, str] = ("<", ">")
    widths_mm: Mapping[str, float] = field(default_factory=lambda: {"single": 85.0, "double": 175.0})
    dpi: int = 300
    svg_hashsalt: str = "pprof_py"
    svg_text_as_paths: bool = True
    corridor: Optional[str] = "#EAF0F5"
    halo: float = 0.6
    spines: Tuple[str, ...] = ()
    tick_length: float = 0.0
    tick_label_color: Optional[str] = "#556372"

    def __post_init__(self) -> None:
        object.__setattr__(self, "spines", tuple(self.spines))
        if not set(self.spines) <= {"left", "bottom"}:
            raise ValueError(f"spines must name only 'left' and 'bottom', got {self.spines}")
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
        """Print-ready defaults: IBM Plex Sans, 85/175 mm widths, text at least 7 pt at final size, 300 dpi."""
        return cls()

    @classmethod
    def notebook(cls) -> "Theme":
        """Larger text and marks for on-screen work in notebooks."""
        return cls(name="notebook",
                   typography=Typography(title=12.0, subtitle=11.0, label=11.0, tick=10.0, legend=10.0,
                                         annotation=10.0, footnote=9.5),
                   lines=Lines(axis=0.8, data=1.0, interval=1.0, interval_dense=0.45, reference=1.1, limit=1.0,
                               grid=0.6),
                   status=_status_styles(2.0), widths_mm={"single": 120.0, "double": 200.0}, dpi=150, halo=0.8)

    @classmethod
    def report(cls) -> "Theme":
        """Defaults for HTML reports and slides."""
        return cls(name="report",
                   typography=Typography(title=13.0, subtitle=11.0, label=11.0, tick=10.0, legend=10.0,
                                         annotation=10.0, footnote=10.0),
                   lines=Lines(axis=0.8, data=1.0, interval=1.0, interval_dense=0.45, reference=1.1, limit=1.0,
                               grid=0.6),
                   status=_status_styles(2.0), widths_mm={"single": 120.0, "double": 180.0}, dpi=200, halo=0.8)

    @classmethod
    def classic(cls, variant: str = "publication") -> "Theme":
        """The 0.6.0 presets, unchanged: DejaVu Sans, the dark PuOr status pair, an open frame and no grid.

        ``variant`` is ``"publication"`` (the default), ``"notebook"`` or ``"report"``. Within one environment the
        figures are byte-identical to 0.6.0 output with that preset.
        """
        if variant not in ("publication", "notebook", "report"):
            raise ValueError(f"variant must be 'publication', 'notebook' or 'report', got {variant!r}")
        common: Dict[str, Any] = dict(
            ink="#1A1A1A", muted="#4D4D4D", reference="#000000", limit="#4D4D4D", volume="#8C8C8C",
            background="#FFFFFF", grid=False, grid_color="#D9D9D9",
            level_dashes={0.95: (0, (4.0, 2.0)), 0.998: (0, (1.0, 1.5))}, corridor=None, halo=0.0,
            spines=("left", "bottom"), tick_length=3.0, tick_label_color=None)
        face = dict(family="DejaVu Sans", title_weight="normal", label_weight="normal")
        if variant == "publication":
            return cls(name="classic", typography=Typography(**face),
                       lines=Lines(axis=0.6, data=0.8, interval=0.8, interval_dense=0.35, reference=0.8, limit=0.7,
                                   grid=0.4),
                       status=_status_styles(1.0, "classic"), **common)
        wide = Lines(axis=0.8, data=1.0, interval=1.0, interval_dense=0.45, reference=1.0, limit=0.9, grid=0.5)
        if variant == "notebook":
            return cls(name="classic-notebook",
                       typography=Typography(title=12.0, subtitle=11.0, label=11.0, tick=10.0, legend=10.0,
                                             annotation=10.0, footnote=9.5, **face),
                       lines=wide, status=_status_styles(2.0, "classic"),
                       widths_mm={"single": 120.0, "double": 200.0}, dpi=150, **common)
        return cls(name="classic-report",
                   typography=Typography(title=13.0, subtitle=11.0, label=11.0, tick=10.0, legend=10.0,
                                         annotation=10.0, footnote=10.0, **face),
                   lines=wide, status=_status_styles(2.0, "classic"), widths_mm={"single": 120.0, "double": 180.0},
                   dpi=200, **common)

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
        from ._fonts import FALLBACK_FAMILY, is_bundled

        t, ln = self.typography, self.lines
        bundled = is_bundled(t.family)
        faces = [FALLBACK_FAMILY] if bundled else [t.family, FALLBACK_FAMILY]
        # the bundled face takes its title and label weights per element (_fonts.apply); rcParams stay at normal so
        # Matplotlib never searches DejaVu Sans for a weight it does not have
        title_w, label_w = ("normal", "normal") if bundled else (t.title_weight, t.label_weight)
        minor = 0.6 * self.tick_length
        return {
            "font.family": "sans-serif", "font.sans-serif": faces, "font.size": t.tick,
            "axes.titlesize": t.title, "axes.labelsize": t.label, "figure.titlesize": t.title,
            "xtick.labelsize": t.tick, "ytick.labelsize": t.tick, "legend.fontsize": t.legend,
            "axes.linewidth": ln.axis, "xtick.major.width": ln.axis, "ytick.major.width": ln.axis,
            "xtick.minor.width": 0.75 * ln.axis, "ytick.minor.width": 0.75 * ln.axis,
            "xtick.major.size": self.tick_length, "ytick.major.size": self.tick_length, "xtick.minor.size": minor,
            "ytick.minor.size": minor,
            "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": "left" in self.spines,
            "axes.spines.bottom": "bottom" in self.spines, "axes.axisbelow": True if self.grid else "line",
            "xtick.labelcolor": self.tick_label_color or self.ink, "ytick.labelcolor": self.tick_label_color or self.ink,
            "axes.titleweight": title_w, "figure.titleweight": title_w, "axes.labelweight": label_w,
            "axes.edgecolor": self.ink, "axes.labelcolor": self.ink, "text.color": self.ink,
            "xtick.color": self.ink, "ytick.color": self.ink,
            "axes.facecolor": self.background, "figure.facecolor": self.background,
            "savefig.facecolor": self.background, "axes.grid": self.grid, "grid.color": self.grid_color,
            "grid.linewidth": ln.grid, "legend.frameon": False, "axes.unicode_minus": True,
            "svg.hashsalt": self.svg_hashsalt, "svg.fonttype": "path" if self.svg_text_as_paths else "none",
            "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": self.dpi, "path.simplify": True,
            "text.hinting": "no_hinting",   # text keeps its outline width at every raster resolution
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


_PRESETS = {"publication": Theme.publication, "notebook": Theme.notebook, "report": Theme.report,
            "classic": Theme.classic}


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
