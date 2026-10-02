"""Coefficient forest: covariate effects with their intervals, from ``summary()`` (spec §2.4)."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Union

import numpy as np

from .._provenance import pct
from ..data import CapabilityError
from ..data._coefficients import COEFFICIENT_LABELS as _LABELS
from ..data._coefficients import COEFFICIENT_SHORT as _SHORT
from ..data._coefficients import coefficient_profile
from ..formatting import fmt_count, fmt_interval, fmt_number
from ..theme import Theme, get_theme
from ._common import freeze_layout, log_ticks, scaffold
from ._result import FigureResult

__all__ = ["forest"]

def forest(source: Any, *, exponentiate: Union[str, bool] = "auto", level: float = 0.95,
           include_intercept: bool = False, terms: Optional[Iterable[str]] = None,
           theme: Union[str, Theme, None] = "publication", size: Union[str, float] = "single",
           title: Optional[str] = None) -> FigureResult:
    """Forest plot of covariate effects with their intervals.

    Parameters
    ----------
    source
        A fitted model (its ``summary()``) or a :class:`~pprof_py.presentation.CoefficientProfile`.
    exponentiate : {"auto", True, False}
        Show odds or hazard ratios on a log axis (``"auto"``: for logistic and Cox models).
    level : float, default 0.95
        Interval level (CoxPH reports 95% only).
    include_intercept : bool, default False
        Show the intercept row of random-effect models.
    terms : sequence of str, optional
        Terms to show, in this order (default: all, in model order).
    theme, size, title
        As for the other figures.

    Returns
    -------
    FigureResult

    Notes
    -----
    Each estimate is adjusted for the other covariates and the provider effects. The footnote warns against reading
    the effects causally and against comparing covariates measured in different units.
    """
    th = get_theme(theme)
    prof = coefficient_profile(source, exponentiate=exponentiate, level=level, include_intercept=include_intercept,
                               terms=terms)
    f, prov = prof.data, prof.provenance
    scale = prov.get("scale")
    exp_axis = bool(prov.get("exponentiated"))
    null = float(prov.get("null_value", 1.0 if exp_axis else 0.0))
    lev = prov.get("level") or level
    k = len(f)
    est, lo, hi = (f[c].to_numpy() for c in ("estimate", "ci_lower", "ci_upper"))
    if exp_axis and (np.any(est <= 0) or np.any(lo[np.isfinite(lo)] <= 0)):
        raise CapabilityError("ratios must be positive for a log axis")
    rows = np.arange(k - 1, -1, -1, dtype=float)          # first term at the top
    excl = int(((lo > null) | (hi < null)).sum())
    label = _LABELS.get(scale, "Estimate")
    texts = list(np.asarray(fmt_interval(est, lo, hi, 2), dtype=object))
    note = (f"{fmt_count(k)} term{'s' if k != 1 else ''} from {prov.get('model') or 'the data'}"
            f"{'.summary()' if prov.get('model') else ''}; {pct(lev)} intervals. Each estimate is adjusted for the other "
            "covariates and the provider effects. Associations are not causal, and covariates have their own units, so "
            "their sizes are not directly comparable.")
    if exp_axis:
        note += f" {label}s are the exponentiated coefficients and bounds, computed for display."
    row_mm = max(3.6, th.typography.tick * 25.4 / 72.0 * 1.6)
    from matplotlib.lines import Line2D

    entries = [(Line2D([], [], color=th.ink, lw=th.lines.interval, marker="o", markersize=3.5), f"{pct(lev)} interval")]
    with th.rc_context():
        fig, (ax, tx), key, ncol, inset = scaffold(th, size, k * row_mm + 18.0, [lab for _, lab in entries], note,
                                                   width_ratios=(3.0, 1.45))
        from matplotlib.collections import LineCollection

        segs = np.stack([np.c_[lo, rows], np.c_[hi, rows]], axis=1)
        ax.add_collection(LineCollection(segs, colors=th.ink, linewidths=th.lines.interval, zorder=2.0,
                                         gid="coefficient-intervals"))
        ax.scatter(est, rows, marker="o", s=14.0, color=th.ink, zorder=3.0, gid="coefficient-estimates")
        ax.axvline(null, color=th.reference, lw=th.lines.reference, zorder=1.0, gid="reference")
        vals = np.r_[lo[np.isfinite(lo)], hi[np.isfinite(hi)], est, null]
        a, b = float(vals.min()), float(vals.max())
        if exp_axis:
            ax.set_xscale("log")
            span = np.log10(b / a) if b > a else 1.0
            ax.set_xlim(a / 10 ** (0.06 * span), b * 10 ** (0.06 * span))
            if b / a < 10.0:                              # narrow ratio range: plain ticks a decade locator lacks
                from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter, NullLocator

                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.10g}"))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                log_ticks(ax.xaxis)
        else:
            pad = 0.06 * ((b - a) or 1.0)
            ax.set_xlim(a - pad, b + pad)
        ax.set_ylim(-0.7, k - 0.3)
        ax.set_yticks(rows)
        ax.set_yticklabels(list(f.index), fontsize=th.typography.tick)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(label + (" (log scale)" if exp_axis else "") + f"; no association at {fmt_number(null, 0)}")
        tx.axis("off")
        tx.set_xlim(0.0, 1.0)
        for y, text in zip(rows, texts):
            tx.text(1.0, y, text, ha="right", va="center", fontsize=th.typography.tick, color=th.ink,
                    gid="coefficient-text")
        tx.text(1.0, k - 0.3, f"{_SHORT.get(scale, 'Estimate')} ({pct(lev)} CI)", ha="right", va="bottom",
                fontsize=th.typography.tick, color=th.muted)
        if title:
            ax.set_title(title, loc="left", fontsize=th.typography.title)
        key.legend([h for h, _ in entries], [lab for _, lab in entries], loc="upper left", bbox_to_anchor=(inset, 1.0),
                   ncol=ncol, frameon=False, borderaxespad=0.0, borderpad=0.0, handletextpad=0.4, handlelength=1.4,
                   fontsize=th.typography.legend)
        freeze_layout(fig, th)
    alt = (f"Forest plot of {fmt_count(k)} terms ({label.lower()}s) with {pct(lev)} intervals: "
           f"{fmt_count(excl)} of them exclude {fmt_number(null, 0)}.")
    return FigureResult(fig, ax, theme=th, alt_text=alt, long_description=alt + " " + note, caption=note, provenance=prov,
                        counts={"terms": k, "excluding_null": excl}, kind="forest")
