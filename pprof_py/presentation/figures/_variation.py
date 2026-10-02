"""Between-provider variation: how much real variation exists across providers (spec §5.3 Tier 2)."""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from scipy.stats import norm

from .._provenance import pct
from ..data._variation import VariationSummary, variation_summary
from ..formatting import fmt_count, fmt_number
from ..theme import Theme, get_theme
from ._common import freeze_layout, scaffold
from ._result import FigureResult

__all__ = ["provider_variation"]

_AXIS = {"log_odds": "Provider effect (log-odds; 0 = average)",
         "difference": "Provider effect (outcome units; 0 = average)"}


def _range_text(v: VariationSummary) -> str:
    text = f"{fmt_number(v.range_lower, 2)} to {fmt_number(v.range_upper, 2)}"
    if v.ratio_lower is not None:
        text += f" (odds ratios {fmt_number(v.ratio_lower, 2)} to {fmt_number(v.ratio_upper, 2)})"
    return text


def _sigma_text(v: VariationSummary) -> str:
    if v.lower is None:
        return f"\u03c3 = {fmt_number(v.sigma, 2)} (no interval available)"
    return (f"\u03c3 = {fmt_number(v.sigma, 2)} ({pct(v.level)} {v.interval_method} interval "
            f"{fmt_number(v.lower, 2)}\u2013{fmt_number(v.upper, 2)})")


def provider_variation(model: Any, *, level: float = 0.95, theme: Union[str, Theme, None] = "publication",
                       size: Union[str, float] = "single", title: Optional[str] = None) -> FigureResult:
    """Shrunken provider effects against the fitted between-provider distribution of a random-effect model.

    Parameters
    ----------
    model
        A fitted ``LogisticRandomEffectModel`` (random-effect SD with a profile-likelihood interval) or
        ``LinearRandomEffectModel`` (random-effect SD without an interval).
    level : float, default 0.95
        Level of the SD's interval and of the range of true effects.
    theme, size, title
        As for the other figures.

    Returns
    -------
    FigureResult

    Notes
    -----
    The BLUPs spread less than the true effects, because each is shrunk toward the average by an amount that
    depends on its precision; unshrunken estimates spread more, because they include sampling noise. The fitted
    distribution N(0, sigma^2) is the model's estimate of the true variation; the range -/+ z * sigma assumes it is
    normal.
    """
    th = get_theme(theme)
    v = variation_summary(model, level=level)
    b = v.blups.to_numpy(dtype=float)
    n = b.size
    wide = max(v.sigma, v.upper or v.sigma)
    half = max(float(np.nanmax(np.abs(b))), 3.0 * wide) * 1.05
    edges = np.linspace(-half, half, 25)
    width = float(edges[1] - edges[0])
    counts, _ = np.histogram(b, bins=edges)
    grid = np.linspace(-half, half, 400)
    scale = n * width
    degenerate = not np.isfinite(v.sigma) or v.sigma < 1e-6     # sigma estimated at 0: no density to draw
    drawn = [s_ for s_ in ((v.sigma,) + ((v.lower, v.upper) if v.lower is not None else ())) if s_ and s_ >= 1e-6]
    peak = max([float(counts.max())] + [scale * norm.pdf(0.0, 0.0, s_) for s_ in drawn])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    entries = [(Patch(facecolor=th.volume, edgecolor="none"), "Shrunken provider effects (BLUPs)"),
               (Line2D([], [], color=th.limit, lw=th.lines.data), "Fitted between-provider distribution")]
    if v.lower is not None:
        entries.append((Line2D([], [], color=th.limit, lw=th.lines.grid, linestyle=(0, (3.0, 2.0))),
                        f"At the SD's {pct(level)} interval bounds"))
    note = (f"{fmt_count(n)} providers, {v.model}. Bars: the BLUPs, shrunken estimates that spread less than the true "
            "effects (each is pulled toward the average by an amount that depends on its precision); unshrunken "
            "estimates spread more, because they include sampling noise. Curve: the fitted between-provider "
            f"distribution N(0, \u03c3\u00b2), scaled to the counts; {_sigma_text(v)}. Bracket: the range in which "
            f"{pct(level)} of true provider effects would lie if that distribution is normal, \u00b1"
            f"{fmt_number(v.z, 2)}\u03c3 = {_range_text(v)}, computed for display from \u03c3.")
    if v.lower is None:
        note += " The model reports no interval for \u03c3, so its uncertainty is not shown."
    if not np.isfinite(v.sigma) or v.sigma < 1e-6:                 # degenerate: nothing of the above is drawn
        note = (f"{fmt_count(n)} providers, {v.model}. Bars: the BLUPs. \u03c3 is estimated at 0, so the model detects "
                "no between-provider variation and no fitted distribution or range is drawn"
                + (f"; {_sigma_text(v)}" if v.lower is not None else "") + ".")
    with th.rc_context():
        fig, ax, key, ncol, inset = scaffold(th, size, 60.0, [lab for _, lab in entries], note)
        ax.stairs(counts, edges, fill=True, color=th.volume, linewidth=0, zorder=1.0, gid="variation-histogram")
        if not degenerate:
            ax.plot(grid, scale * norm.pdf(grid, 0.0, v.sigma), color=th.limit, lw=th.lines.data, zorder=3.0,
                    gid="variation-fitted")
        if v.lower is not None:
            for name, s in (("lower", v.lower), ("upper", v.upper)):
                if s >= 1e-6:
                    ax.plot(grid, scale * norm.pdf(grid, 0.0, s), color=th.limit, lw=th.lines.grid,
                            linestyle=(0, (3.0, 2.0)), zorder=2.5, gid=f"variation-bound-{name}")
        yb = 1.18 * peak
        if degenerate:
            ax.text(0.5, 0.62, "\u03c3 is estimated at 0: no between-provider\nvariation is detected", transform=ax.transAxes,
                    ha="center", va="center", fontsize=th.typography.annotation, color=th.ink, gid="variation-degenerate")
        ax.plot([0.0, 0.0], [0.0, 1.06 * peak], color=th.reference, lw=th.lines.reference, zorder=1.5,
                gid="reference")                                # stops below the bracket and its label
        if not degenerate:
            ax.plot([v.range_lower, v.range_upper], [yb, yb], color=th.ink, lw=th.lines.interval, zorder=4.0,
                    gid="variation-range")
            for xv in (v.range_lower, v.range_upper):
                ax.plot([xv, xv], [yb - 0.03 * peak, yb + 0.03 * peak], color=th.ink, lw=th.lines.interval, zorder=4.0)
            ax.text(0.0, yb + 0.05 * peak, f"{pct(level)} of true effects if normal:\n{_range_text(v)}", ha="center",
                    va="bottom", fontsize=th.typography.annotation, color=th.ink, zorder=5.0, linespacing=1.25)
        ax.text(0.02, 0.97, _sigma_text(v), transform=ax.transAxes, ha="left", va="top",
                fontsize=th.typography.annotation, color=th.ink, gid="variation-sigma")
        ax.set_xlim(-half, half)
        ax.set_ylim(0.0, 1.85 * peak)
        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))     # provider counts
        ax.set_xlabel(_AXIS[v.scale])
        ax.set_ylabel("Providers")
        if title:
            ax.set_title(title, loc="left", fontsize=th.typography.title)
        key.legend([h for h, _ in entries], [lab for _, lab in entries], loc="upper left", bbox_to_anchor=(inset, 1.0),
                   ncol=ncol, frameon=False, borderaxespad=0.0, borderpad=0.0, handletextpad=0.4, handlelength=1.6,
                   fontsize=th.typography.legend)
        freeze_layout(fig, th)
    alt = (f"Between-provider variation in {fmt_count(n)} providers: random-effect SD {_sigma_text(v)[4:]}; "
           f"{pct(level)} of true provider effects would lie within {_range_text(v)} under a normal random-effect "
           f"distribution; the BLUPs range from {fmt_number(b.min(), 2)} to {fmt_number(b.max(), 2)}.")
    prov = {"model": v.model, "provider_var": v.provider_var, "scale": v.scale, "sigma": v.sigma, "lower": v.lower,
            "upper": v.upper, "level": v.level, "interval_method": v.interval_method}
    return FigureResult(fig, ax, theme=th, alt_text=alt, long_description=alt + " " + note, caption=note, provenance=prov,
                        counts={"providers": n}, kind="provider_variation")
