"""Shared pieces of the figure renderers: sources, provenance text, layout rows, ticks and labels."""
from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data import CapabilityError, ProviderProfile
from ..formatting import fmt_count, fmt_number

PT_MM = 25.4 / 72.0
_METHODS = {"poibin_exact": "exact Poisson-binomial", "exact": "exact", "score": "score", "wald": "Wald",
            "midp": "mid-p Poisson", "bootstrap_exact": "bootstrap", "resampling": "resampling"}
_ALTERNATIVES = {"two_sided": "two-sided", "greater": "one-sided (above)", "less": "one-sided (below)"}


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


def pct(level: float) -> str:
    return f"{float(level) * 100:g}%"


def null_text(nm: Any) -> str:
    if not isinstance(nm, Mapping):
        return "null model not stated"
    kind = nm.get("kind")
    if kind == "theoretical":
        return "theoretical null N(0, 1)"
    if kind == "fixed":
        return f"fixed null, mean {fmt_number(nm.get('null_mean'), 2)} and SD {fmt_number(nm.get('null_sd'), 2)}"
    if kind == "empirical":
        groups = list(nm.get("groups") or [])
        if len(groups) == 1:
            g = groups[0]
            return f"empirical null, mean {fmt_number(g['null_mean'], 2)} and SD {fmt_number(g['null_sd'], 2)}"
        if groups:
            m = [g["null_mean"] for g in groups]
            s = [g["null_sd"] for g in groups]
            return (f"empirical null in {len(groups)} groups, means {fmt_number(min(m), 2)} to {fmt_number(max(m), 2)} "
                    f"and SDs {fmt_number(min(s), 2)} to {fmt_number(max(s), 2)}")
        return "empirical null"
    return f"{kind} null"


def test_text(p: Mapping[str, Any]) -> str:
    parts = [f"{_METHODS.get(p.get('test_method'), p.get('test_method') or 'unstated')} test"]
    if p.get("alternative"):
        parts.append(_ALTERNATIVES.get(p["alternative"], p["alternative"]))
    if p.get("critical") is not None:
        parts.append(f"critical value {fmt_number(p['critical'], 2)}")
    elif p.get("level") is not None:
        parts.append(f"{pct(p['level'])} level per provider")
    return ", ".join(parts)


def reference_text(p: Mapping[str, Any], ratio: bool) -> str:
    spec, value = p.get("reference"), p.get("reference_value")
    if p.get("measure") == "gamma":
        what = {"median": "median provider effect", "mean": "size-weighted mean provider effect"}.get(spec, "effect")
        text = what if value is None else f"{what} {fmt_number(value, 2)}"
        return text + (", where O/E = 1" if ratio else "")
    if value is not None:
        return f"{'O/E' if p.get('scale') == 'ratio' else 'value'} {fmt_number(value, 2)}"
    return "not stated"


def counts_text(profile: ProviderProfile) -> str:
    c, p = profile.status_counts(), profile.provenance
    bits = [f"{fmt_count(len(profile))} providers"]
    if c["excluded"]:
        bits.append(f"{fmt_count(c['excluded'])} excluded by data preparation")
    if c["not_tested"]:
        bits.append(f"{fmt_count(c['not_tested'])} not tested")
    if c["suppressed"]:
        bits.append(f"{fmt_count(c['suppressed'])} suppressed (fewer than {fmt_number(p.get('min_volume'), 0)} "
                    f"{p.get('min_volume_kind') or 'units'}; not drawn)")
    return "; ".join(bits)


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


def scaffold(theme: Any, size: Any, main_mm: float, labels: Sequence[str], note: str):
    """Figure with three stacked sub-figures: data, legend and footnote, each sized from its content.

    Sub-figures keep the legend and footnote at the full figure width; in one grid, constrained layout would align
    them with the data axes and shrink those axes to make room.
    """
    from matplotlib.figure import Figure

    width_in, _ = theme.figsize(size)
    width_mm = width_in * 25.4
    family = theme.typography.family
    ncol, key_mm = legend_rows(labels, width_mm - 3.0, theme.typography.legend, family)
    wrapped, note_mm = note_rows(note, width_mm - 3.0, theme.typography.footnote, family)
    fig = Figure(figsize=(width_in, (main_mm + key_mm + note_mm) / 25.4), layout="constrained")
    top, key, foot = fig.subfigures(3, 1, height_ratios=[main_mm, max(key_mm, 0.01), note_mm])
    ax = top.add_subplot()
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
