"""Shrinkage: how far pooling moves each provider, from its fixed-effect to its random-effect estimate."""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Union

import numpy as np

from ..data._shrinkage import reference_text, shrinkage_pairs
from ..formatting import fmt_count, fmt_number
from ..theme import Theme, get_theme
from ._common import freeze_layout, scaffold
from ._result import FigureResult

__all__ = ["shrinkage"]

_SCALE = {"logistic": "log-odds", "linear": "outcome units"}
_UNITS = {"records": "records", "trials": "trials", "patients": "patients"}


def _sizes(volume: np.ndarray, top: float) -> np.ndarray:
    out = np.full(volume.size, 8.0)
    ok = np.isfinite(volume) & (volume > 0)
    out[ok] = 4.0 + 40.0 * volume[ok] / top
    return out


def shrinkage(fixed: Any, random: Any, *, fe_reference: Any = "mean", highlight: Optional[Iterable[Any]] = None,
              theme: Union[str, Theme, None] = "publication", size: Union[str, float] = "single",
              title: Optional[str] = None) -> FigureResult:
    """Each provider's fixed-effect estimate against its random-effect BLUP, sized by volume.

    Parameters
    ----------
    fixed, random
        A fixed-effect and a random-effect fit of the same family (models, tested here, or profiles).
    fe_reference : {"mean", "median"} or float, default "mean"
        Reference of the fixed-effect test, so that both estimates are deviations from an average provider.
    highlight : iterable, optional
        Providers to label.
    theme, size, title
        As for the other figures.

    Returns
    -------
    FigureResult

    Notes
    -----
    On the diagonal an estimate is not shrunk; on the horizontal line it is shrunk completely to the average.
    Shrinkage depends on the assumed normal random-effect distribution; the BLUPs are not the true effects.
    """
    th = get_theme(theme)
    pairs = shrinkage_pairs(fixed, random, fe_reference=fe_reference)
    f, prov = pairs.frame, pairs.provenance
    x, y = f["fixed"].to_numpy(dtype=float), f["random"].to_numpy(dtype=float)
    vol = f["volume"].to_numpy(dtype=float)
    fin = f["fixed_finite"].to_numpy(dtype=bool) & np.isfinite(x)
    n, k_off = len(f), int((~fin).sum())
    vals = np.r_[x[fin], y[np.isfinite(y)], 0.0]
    a, b = float(vals.min()), float(vals.max())
    pad = 0.06 * ((b - a) or 1.0)
    lo, hi = a - pad, b + pad
    top = float(np.nanmax(vol)) if np.isfinite(vol).any() else 1.0
    sizes = _sizes(vol, top)
    scale = _SCALE.get(prov.get("family"), "effect")
    unit = _UNITS.get(prov.get("denominator_kind"), "volume")
    from matplotlib.lines import Line2D

    keys = []
    if np.isfinite(vol).any():
        for q in (0.1, 0.5, 0.9):
            v = float(np.nanquantile(vol, q))
            v = float(f"{v:.2g}")
            keys.append((Line2D([], [], linestyle="none", marker="o", markersize=math.sqrt(float(_sizes(np.array([v]), top)[0])),
                                markerfacecolor="none", markeredgecolor=th.ink, markeredgewidth=0.5),
                         f"{fmt_count(v)} {unit}"))
    if k_off:
        keys.append((Line2D([], [], linestyle="none", marker="<", markersize=4.5, color=th.muted),
                     "No finite fixed-effect estimate"))
    sigma = prov.get("sigma")
    note = (f"{fmt_count(n)} providers in both fits" + (f" ({fmt_count(pairs.only_fixed)} only in the fixed-effect fit, "
            f"{fmt_count(pairs.only_random)} only in the random-effect fit)" if pairs.only_fixed or pairs.only_random
            else "") + f". Each point is a provider: its fixed-effect estimate ({prov.get('fixed_model')}, unshrunken, "
            f"relative to the {reference_text(pairs)}) against its BLUP ({prov.get('random_model')}, shrunken, "
            f"relative to the model's intercept), on the {scale} scale. On the diagonal an estimate is not shrunk; on "
            f"the horizontal line it is shrunk completely to the average; point area is proportional to {unit}. "
            "Shrinkage depends on the assumed normal random-effect distribution"
            + (f" (\u03c3 = {fmt_number(sigma, 2)})" if sigma is not None else "") + "; BLUPs are not the true effects.")
    if k_off:
        note += (f" {fmt_count(k_off)} provider{'s' if k_off != 1 else ''} without a finite fixed-effect estimate (no "
                 f"events or only events) {'are' if k_off != 1 else 'is'} drawn at the edge.")
    with th.rc_context():
        fig, ax, key, ncol, inset = scaffold(th, size, 70.0, [lab for _, lab in keys], note)
        ax.plot([lo, hi], [lo, hi], color=th.reference, lw=th.lines.reference, zorder=1.0, gid="no-shrinkage")
        ax.axhline(0.0, color=th.volume, lw=th.lines.grid, linestyle=(0, (4.0, 2.0)), zorder=0.9, gid="complete-pooling")
        ax.scatter(x[fin], y[fin], s=sizes[fin], facecolors="none", edgecolors=th.ink, linewidths=0.5, zorder=3.0,
                   rasterized=n > 2000, gid="shrinkage-points")
        if k_off:
            edge = np.where(np.nan_to_num(x[~fin], nan=-1.0) < 0, lo, hi)
            marker_left = edge == lo
            for side, m in (("<", marker_left), (">", ~marker_left)):
                if m.any():
                    ax.scatter(edge[m], y[~fin][m], marker=side, s=22.0, color=th.muted, linewidths=0, zorder=3.5,
                               clip_on=False, gid=f"shrinkage-offscale-{'left' if side == '<' else 'right'}")
        # shrinkage leaves two regions empty: above the diagonal on the right, above the zero line on the left
        span = hi - lo
        ax.text(lo + 0.62 * span, lo + 0.70 * span, "no shrinkage", ha="right", va="bottom",
                fontsize=th.typography.annotation, color=th.reference)
        ax.text(lo + 0.02 * span, 0.0 + 0.015 * span, "complete pooling", ha="left", va="bottom",
                fontsize=th.typography.annotation, color=th.muted)
        if highlight is not None:
            ids = f.index
            for i in np.flatnonzero(np.asarray(ids.isin(list(highlight))) & fin):
                ax.annotate(str(ids[i]), (x[i], y[i]), xytext=(3, 3), textcoords="offset points",
                            fontsize=th.typography.annotation, color=th.ink)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Fixed effect, unshrunken ({scale})")
        ax.set_ylabel(f"Random effect, shrunken ({scale})")
        if title:
            ax.set_title(title, loc="left", fontsize=th.typography.title)
        if keys:
            key.legend([h for h, _ in keys], [lab for _, lab in keys], loc="upper left", bbox_to_anchor=(inset, 1.0),
                       ncol=ncol, frameon=False, borderaxespad=0.0, borderpad=0.0, handletextpad=0.4, handlelength=1.0,
                       fontsize=th.typography.legend)
        freeze_layout(fig)
    alt = (f"Shrinkage of {fmt_count(n)} providers: fixed-effect estimates from {fmt_number(x[fin].min(), 2)} to "
           f"{fmt_number(x[fin].max(), 2)}, random-effect estimates from {fmt_number(np.nanmin(y), 2)} to "
           f"{fmt_number(np.nanmax(y), 2)} ({scale})" + (f"; {fmt_count(k_off)} without a finite fixed-effect estimate"
                                                       if k_off else "") + ".")
    return FigureResult(fig, ax, theme=th, alt_text=alt, long_description=alt + " " + note, provenance=dict(prov),
                        counts={"providers": n, "no_finite_fixed": k_off, "only_fixed": pairs.only_fixed,
                                "only_random": pairs.only_random}, kind="shrinkage")
