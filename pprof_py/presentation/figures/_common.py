"""Shared pieces of the figure renderers: sources, provenance text, layout rows, ticks and labels."""
from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data import CapabilityError, ProviderProfile

PT_MM = 25.4 / 72.0


def resolve_profile(source: Any, args: tuple, kwargs: dict, *, display: str, limits: bool,
                    levels: Optional[Sequence[float]] = None) -> ProviderProfile:
    """A profile from a fitted model (tested once here), a ``test()`` result, or a profile as given."""
    if isinstance(source, ProviderProfile):
        if args or kwargs:
            raise TypeError(f"{display}(): the test settings come from the profile; build a new profile to change "
                            f"them (got {sorted(kwargs) or 'positional arguments'})")
        return source
    if isinstance(source, pd.DataFrame):
        if limits:
            raise CapabilityError(f"{display}() needs control limits from the same test as the flags, and a test() "
                                  "result has none; pass the fitted model, or ProviderProfile.from_model(model, "
                                  "limits=True).")
        return ProviderProfile.from_test(source)
    return ProviderProfile.from_model(source, *args, limits=limits, levels=levels if limits else None, **kwargs)


def text_width_mm(text: str, font_pt: float, family: str) -> float:
    """Rendered width of one line of text, measured with the font's own metrics."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    width, _, _ = TextToPath().get_text_width_height_descent(text, FontProperties(family=family, size=font_pt),
                                                              ismath=False)
    return float(width) * PT_MM


def wrap_measured(text: str, width_mm: float, font_pt: float, family: str) -> str:
    """Greedy line breaking at spaces, by measured width (a word longer than the line keeps its own line)."""
    lines: List[str] = []
    current = ""
    for word in text.split(" "):
        trial = word if not current else current + " " + word
        if current and text_width_mm(trial, font_pt, family) > width_mm:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return "\n".join(lines)


def legend_rows(labels: Sequence[str], width_mm: float, font_pt: float, family: str) -> Tuple[int, float]:
    """Columns and height (mm) of a legend row that fits ``width_mm`` (handle 1.0 em, pad 0.4 em, spacing 1.2 em)."""
    if not labels:
        return 1, 0.0
    em = font_pt * PT_MM
    entry = max(text_width_mm(s, font_pt, family) for s in labels) + 1.4 * em
    ncol = max(1, min(len(labels), int((width_mm + 1.2 * em) // (entry + 1.2 * em))))
    return ncol, math.ceil(len(labels) / ncol) * em * 1.7 + 0.8


def note_rows(text: str, width_mm: float, font_pt: float, family: str) -> Tuple[str, float]:
    """Wrapped footnote and its height (mm)."""
    wrapped = wrap_measured(text, width_mm, font_pt, family)
    return wrapped, (wrapped.count("\n") + 1) * font_pt * PT_MM * 1.3 + 0.8


def scaffold(theme: Any, size: Any, main_mm: float, labels: Sequence[str], note: str, *,
             width_ratios: Optional[Sequence[float]] = None):
    """Figure with three stacked sub-figures: data, legend and footnote, each sized from its content.

    With ``width_ratios`` the data row holds that many axes side by side, sharing the y axis; otherwise one axes.

    Sub-figures keep the legend and footnote at the full figure width; in one grid, constrained layout would align
    them with the data axes and shrink those axes to make room.
    """
    from matplotlib.figure import Figure

    width_in, _ = theme.figsize(size)
    width_mm = width_in * 25.4
    family = theme.typography.family
    usable = (width_mm - 3.0) * 0.98              # rendered text boxes are about 1.5 % wider than the outlines
    ncol, key_mm = legend_rows(labels, usable, theme.typography.legend, family)
    wrapped, note_mm = note_rows(note, usable, theme.typography.footnote, family)
    fig = Figure(figsize=(width_in, (main_mm + key_mm + note_mm) / 25.4), layout="constrained")
    top, key, foot = fig.subfigures(3, 1, height_ratios=[main_mm, max(key_mm, 0.01), note_mm])
    if width_ratios is None:
        ax = top.add_subplot()
    else:
        ax = tuple(top.subplots(1, len(width_ratios), sharey=True,
                                gridspec_kw={"width_ratios": list(width_ratios), "wspace": 0.02}))
    inset = 1.5 / width_mm
    foot.text(inset, 1.0, wrapped, ha="left", va="top", fontsize=theme.typography.footnote, color=theme.muted,
              linespacing=1.25)
    return fig, ax, key, ncol, inset


def freeze_layout(fig: Any) -> None:
    """Run constrained layout once, then switch it off, so every later save draws the same positions.

    Constrained layout restarts from the current positions on each draw and is not idempotent: without this, two
    saves of one figure differ in the last decimals of their coordinates.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig)
    fig.canvas.draw()
    if hasattr(fig, "set_layout_engine"):           # Matplotlib >= 3.6
        fig.set_layout_engine("none")
    else:                                           # Matplotlib 3.5
        fig.set_constrained_layout(False)


def log_ticks(axis: Any) -> None:
    """Plain-number 1-2-5 ticks on a log axis."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.10g}"))
    axis.set_minor_formatter(NullFormatter())


def spread(values: Sequence[float], gap: float, lower: float, upper: float) -> List[float]:
    """Positions, in the same order, moved apart by at least ``gap`` and kept within ``[lower, upper]``."""
    order = np.argsort(values, kind="stable")
    pos = np.asarray(values, dtype=np.float64)[order].copy()
    for i in range(1, pos.size):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    overflow = pos[-1] - upper if pos.size else 0.0
    if overflow > 0:
        pos -= overflow
    for i in range(pos.size - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - gap)
    pos = np.maximum(pos, lower)
    out = np.empty_like(pos)
    out[order] = pos
    return out.tolist()
