"""Several measures of the same providers: small multiples and pairwise agreement (spec §5.3 Tier 2)."""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd

from .._provenance import null_text, test_text
from ..data._collection import ProfileCollection
from ..formatting import fmt_count
from ..theme import Theme, get_theme
from ._common import freeze_layout, halo_edges, interval_bars, ring, scaffold
from ._result import FigureResult

__all__ = ["measure_agreement", "multi_measure"]

_KEYS = ("above", "below", "not_different", "not_tested")
_JOINT = (("same", "Flagged on both, same direction", "D", True),
          ("opposite", "Flagged on both, opposite directions", "s", True),
          ("one", "Flagged on one measure only", "^", False),
          ("neither", "Flagged on neither", "o", True),
          ("untested", "Not tested on at least one", "o", False))


def _collection(source: Any) -> ProfileCollection:
    return source if isinstance(source, ProfileCollection) else ProfileCollection(source)


def _limits(lo: np.ndarray, hi: np.ndarray, est: np.ndarray, null: float) -> tuple:
    vals = np.r_[lo[np.isfinite(lo)], hi[np.isfinite(hi)], est[np.isfinite(est)], null]
    a, b = float(vals.min()), float(vals.max())
    pad = 0.06 * ((b - a) or 1.0)
    return a - pad, b + pad


def multi_measure(source: Any, *, order: str = "first", highlight: Optional[Iterable[Any]] = None,
                  theme: Union[str, Theme, None] = "publication", size: Union[str, float] = "double",
                  title: Optional[str] = None) -> FigureResult:
    """One interval panel per measure, providers in a common row order.

    Parameters
    ----------
    source
        A :class:`~pprof_py.presentation.ProfileCollection`, or a mapping of measure labels to profiles, ``test()``
        results or fitted models.
    order : {"first", "id"}, default "first"
        Rows ordered by the first measure's estimate (for legibility, not a ranking) or by provider id.
    highlight, theme, size, title
        As for the other figures.

    Notes
    -----
    Each panel is on its own measure's scale with its own reference; panels are not to be combined into a composite.
    A provider missing from a measure is marked "n/a" in that panel.
    """
    th = get_theme(theme)
    col = _collection(source)
    col.require("multi_measure")
    labels = col.labels
    k = len(labels)
    ids = col.providers()
    first = col.column("estimate")[labels[0]]
    if order == "first":
        ids = ids[np.argsort(-first.reindex(ids).to_numpy(dtype=float), kind="stable")]
    elif order != "id":
        raise ValueError("order must be 'first' or 'id'")
    n = len(ids)
    dense = n > 60
    row_mm = max(3.6, th.typography.tick * 25.4 / 72.0 * 1.6) if not dense else min(1.2, 110.0 / n)
    y = np.arange(n - 1, -1, -1, dtype=float)
    from matplotlib.lines import Line2D

    entries = []
    for key in _KEYS:
        st = th.status[key]
        entries.append((Line2D([], [], linestyle="none", marker=st.marker, markersize=math.sqrt(st.size),
                               markerfacecolor=st.color if st.filled else "none", markeredgecolor=st.color,
                               markeredgewidth=0.6), st.label))
    parts, alt_parts = [], []
    for label in labels:
        p = col[label]
        c = p.status_counts()
        missing = int(n - len(p))
        parts.append(f"{label}: {test_text(p.provenance)}; {null_text(p.provenance.get('null_model'))}"
                     + (f"; {fmt_count(missing)} providers not in this measure (n/a)" if missing else ""))
        alt_parts.append(f"{label}: {fmt_count(c['above'])} above, {fmt_count(c['below'])} below")
    note = (f"{fmt_count(n)} providers, rows "
            + ("ordered by the first measure's estimate for legibility, not a ranking" if order == "first"
               else "ordered by provider id") + ". " + "; ".join(parts) + ". Each panel has its own scale and "
            "reference; panels are not a composite, and non-overlapping intervals are not a test of a difference "
            "between providers.")
    shown_note = (f"{fmt_count(n)} providers, rows "
                  + ("ordered by the first measure's estimate, not a ranking" if order == "first"
                     else "ordered by provider id") + ". Each panel has its own scale and reference; panels are not a "
                  "composite, and non-overlapping intervals are not a test of a difference between providers."
                  if th.footnote == "short" else note)
    key_top = th.key_position == "top"
    with th.rc_context():
        fig, axes, key, ncol, inset = scaffold(th, size, n * row_mm + 12.0, [lab for _, lab in entries], shown_note,
                                               width_ratios=[1.0] * k, key_top=key_top,
                                               title=title if key_top else None)
        for j, (ax, label) in enumerate(zip(axes, labels)):
            f = col[label].data.reindex(ids)
            est, lo, hi = (f[c].to_numpy(dtype=float) for c in ("estimate", "ci_lower", "ci_upper"))
            null = float(np.nanmedian(f["null_value"].to_numpy(dtype=float)))
            present = f["status"].notna().to_numpy()
            finite = present & np.isfinite(est) & f["finite_estimate"].fillna(True).to_numpy(dtype=bool)
            a, b = _limits(lo[finite], hi[finite], est[finite], null)
            seg = finite & (np.isfinite(lo) | np.isfinite(hi))
            from matplotlib.collections import LineCollection

            segs = np.stack([np.c_[np.clip(np.nan_to_num(lo[seg], nan=a, neginf=a), a, b), y[seg]],
                             np.c_[np.clip(np.nan_to_num(hi[seg], nan=b, posinf=b), a, b), y[seg]]], axis=1)
            if th.interval_bars:                         # status-coloured bars trimmed onto the bounds (D79)
                seg_status = f["status"].astype(object).to_numpy()[seg]
                colors = [th.status[s].color if s in ("above", "below") else th.volume for s in seg_status]
                interval_bars(fig, ax, th, segs, colors, th.lines.interval_bar_dense if dense else th.lines.interval_bar,
                              zorder=2.0, gid=f"multi-intervals-{j}", rasterized=dense)
            else:
                ax.add_collection(LineCollection(segs, colors=th.limit,
                                                 linewidths=th.lines.interval * (0.5 if dense else 1.0),
                                                 zorder=2.0, rasterized=dense, gid=f"multi-intervals-{j}"))
            status = f["status"].astype(object).to_numpy()
            for key_ in _KEYS:
                m = finite & (status == key_)
                if m.any():
                    st = th.status[key_]
                    mark = (th.bar_marks or {}).get(key_, st.color) if th.interval_bars else st.color
                    ec, ew = halo_edges(th, mark, st.filled, 0.6)
                    ax.scatter(est[m], y[m], marker=st.marker, s=st.size * (0.5 if dense else 1.0),
                               facecolors=mark if st.filled else "none", edgecolors=ec, linewidths=ew,
                               zorder=3.0, rasterized=dense, gid=f"multi-{key_}-{j}")
            ax.axvline(null, color=th.reference, lw=th.lines.reference, zorder=1.5, gid=f"multi-reference-{j}")
            if not dense:
                for yi in y[~present]:                      # "n/a", not a dash that reads as a short interval
                    ax.text(null, yi, "n/a", ha="center", va="center", fontsize=th.typography.annotation,
                            color=th.muted, style="italic", zorder=4.0, gid=f"multi-missing-{j}")
                for yi in y[present & ~finite]:
                    ax.text(null, yi, "NE", ha="center", va="center", fontsize=th.typography.annotation,
                            color=th.muted, zorder=4.0)
            ax.set_xlim(a, b)
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_title(label, loc="left", fontsize=th.typography.label)
            ax.set_xlabel("Estimate")
            if j == 0:
                if dense:
                    ax.set_yticks([])
                else:
                    ax.set_yticks(y)
                    ax.set_yticklabels([str(i) for i in ids], fontsize=th.typography.tick)
                ax.tick_params(axis="y", length=0)
        if highlight is not None and not dense:
            want = set(highlight)
            for t in axes[0].get_yticklabels():
                if t.get_text() in {str(w) for w in want}:
                    if th.highlight_ring:
                        t.set_gid("emphasis")              # the title weight with the bundled face (D80)
                    else:
                        t.set_fontweight("bold")
            if th.highlight_wash:
                chosen = [yy for yy, i in zip(y, ids) if str(i) in {str(w) for w in want}]
                for pnl in axes:
                    for r in chosen:
                        pnl.axhspan(r - 0.5, r + 0.5, facecolor=th.highlight_wash, edgecolor="none", zorder=0.3,
                                    gid="highlight-band")
        if title and not key_top:
            fig.suptitle(title, x=inset, ha="left", fontsize=th.typography.title)
        key.legend([h for h, _ in entries], [lab for _, lab in entries], loc="upper left", bbox_to_anchor=(inset, 1.0),
                   ncol=ncol, frameon=False, borderaxespad=0.0, borderpad=0.0, handletextpad=0.4, handlelength=1.0,
                   fontsize=th.typography.legend)
        freeze_layout(fig, th)
    alt = f"Small multiples of {k} measures for {fmt_count(n)} providers: " + "; ".join(alt_parts) + "."
    return FigureResult(fig, axes[0], theme=th, alt_text=alt, long_description=alt + " " + note, caption=note,
                        provenance={"measures": labels}, counts={"providers": n, "measures": k},
                        kind="multi_measure")


def joint_status(fx: pd.Series, fy: pd.Series) -> pd.Series:
    """Joint flag status of two measures (from their tests' flags): same, opposite, one, neither, untested."""
    a, b = fx.astype("float"), fy.astype("float")
    out = pd.Series("neither", index=fx.index, dtype=object)
    out[(a != 0) ^ (b != 0)] = "one"
    out[(a != 0) & (b != 0) & (a == b)] = "same"
    out[(a != 0) & (b != 0) & (a == -b)] = "opposite"
    out[a.isna() | b.isna()] = "untested"
    return out


def measure_agreement(source: Any, x: Optional[str] = None, y: Optional[str] = None, *,
                      highlight: Optional[Iterable[Any]] = None, theme: Union[str, Theme, None] = "publication",
                      size: Union[str, float] = "single", title: Optional[str] = None) -> FigureResult:
    """Two measures' estimates for the providers in both, with interval crosses and joint flag status.

    Notes
    -----
    Agreement between estimates is attenuated by estimation noise, so the cloud understates how closely the true
    provider effects agree; a flag on one measure says nothing by itself about the other.
    """
    th = get_theme(theme)
    col = _collection(source)
    col.require("measure_agreement")
    x = x or col.labels[0]
    y = y or col.labels[1]
    ids = col.common()
    fx, fy = col[x].data.reindex(ids), col[y].data.reindex(ids)
    ex, ey = fx["estimate"].to_numpy(dtype=float), fy["estimate"].to_numpy(dtype=float)
    finite = (fx["finite_estimate"].fillna(True).to_numpy(dtype=bool)
              & fy["finite_estimate"].fillna(True).to_numpy(dtype=bool))
    ok = np.isfinite(ex) & np.isfinite(ey) & finite            # solver-bound estimates would set the axes (M1)
    joint = joint_status(fx["flag"], fy["flag"])
    counts = {key: int(((joint == key).to_numpy() & ok).sum()) for key, *_ in _JOINT}
    not_placed = int((~ok).sum())
    nx, ny = float(np.nanmedian(fx["null_value"])), float(np.nanmedian(fy["null_value"]))
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    entries = [(Line2D([], [], linestyle="none", marker=mk, markersize=4.0 if key != "neither" else 3.0,
                       markerfacecolor=(th.ink if key != "neither" else th.muted) if filled else "none",
                       markeredgecolor=th.ink if key != "neither" else th.muted, markeredgewidth=0.6),
                f"{text} ({fmt_count(counts[key])})") for key, text, mk, filled in _JOINT if counts[key]]
    only_x, only_y = len(col[x]) - len(ids), len(col[y]) - len(ids)
    note = (f"{fmt_count(len(ids))} providers in both measures" + (f" ({fmt_count(only_x)} only in {x}, "
            f"{fmt_count(only_y)} only in {y})" if only_x or only_y else "") + f". Crosses: each measure's "
            f"interval; lines: each measure's reference. {x}: {test_text(col[x].provenance)}; {y}: "
            f"{test_text(col[y].provenance)}. Joint status counts the two tests' flags"
            + (f"; {fmt_count(not_placed)} provider{'s' if not_placed != 1 else ''} without a finite estimate in at "
               "least one measure are not placed" if not_placed else "") + ". Agreement between estimates "
            "is attenuated by estimation noise, so the cloud understates how closely true provider effects agree; a "
            "flag on one measure says nothing by itself about the other.")
    shown_note = (f"{fmt_count(len(ids))} providers in both measures. Crosses: each measure's interval; lines: each "
                  "measure's reference. A flag on one measure says nothing by itself about the other."
                  if th.footnote == "short" else note)
    key_top = th.key_position == "top"
    with th.rc_context():
        fig, ax, key, ncol, inset = scaffold(th, size, 70.0, [lab for _, lab in entries], shown_note,
                                             key_top=key_top, title=title if key_top else None)
        lox, hix = fx["ci_lower"].to_numpy(dtype=float), fx["ci_upper"].to_numpy(dtype=float)
        loy, hiy = fy["ci_lower"].to_numpy(dtype=float), fy["ci_upper"].to_numpy(dtype=float)
        ax_x = _limits(lox[ok], hix[ok], ex[ok], nx)
        ax_y = _limits(loy[ok], hiy[ok], ey[ok], ny)
        hx = ok & np.isfinite(lox) & np.isfinite(hix)
        vy = ok & np.isfinite(loy) & np.isfinite(hiy)
        crosses = [((a, c), (b, c)) for a, b, c in zip(lox[hx], hix[hx], ey[hx])] + \
                  [((c, a), (c, b)) for a, b, c in zip(loy[vy], hiy[vy], ex[vy])]
        ax.add_collection(LineCollection(crosses, colors=th.limit, linewidths=th.lines.grid * 1.2, zorder=1.5,
                                         rasterized=len(ids) > 2000, gid="agreement-crosses"))
        ax.axvline(nx, color=th.reference, lw=th.lines.reference, zorder=1.0, gid="agreement-reference-x")
        ax.axhline(ny, color=th.reference, lw=th.lines.reference, zorder=1.0, gid="agreement-reference-y")
        for key_, _, mk, filled in _JOINT:
            m = ok & (joint == key_).to_numpy()
            if m.any():
                color = th.ink if key_ != "neither" else th.muted
                ec, ew = halo_edges(th, color, filled, 0.6)
                ax.scatter(ex[m], ey[m], marker=mk, s=14.0 if key_ != "neither" else 7.0,
                           facecolors=color if filled else "none", edgecolors=ec, linewidths=ew, zorder=3.0,
                           rasterized=len(ids) > 2000, gid=f"agreement-{key_}")
        if highlight is not None:
            chosen = np.flatnonzero(np.asarray(ids.isin(list(highlight))) & ok)
            ring(ax, th, ex[chosen], ey[chosen])
            for i in chosen:
                ax.annotate(str(ids[i]), (ex[i], ey[i]), xytext=(3, 3), textcoords="offset points",
                            fontsize=th.typography.annotation, color=th.ink,
                            gid="emphasis" if th.highlight_ring else None)
        ax.set_xlim(*ax_x)
        ax.set_ylim(*ax_y)
        ax.set_xlabel(f"{x} (estimate)")
        ax.set_ylabel(f"{y} (estimate)")
        if title and not key_top:
            ax.set_title(title, loc="left", fontsize=th.typography.title)
        key.legend([h for h, _ in entries], [lab for _, lab in entries], loc="upper left", bbox_to_anchor=(inset, 1.0),
                   ncol=ncol, frameon=False, borderaxespad=0.0, borderpad=0.0, handletextpad=0.4, handlelength=1.0,
                   fontsize=th.typography.legend)
        freeze_layout(fig, th)
    alt = (f"Agreement of {x} and {y} for {fmt_count(int(ok.sum()))} providers: {fmt_count(counts['same'])} flagged on both "
           f"in the same direction, {fmt_count(counts['opposite'])} in opposite directions, {fmt_count(counts['one'])} "
           f"on one measure only, {fmt_count(counts['neither'])} on neither.")
    return FigureResult(fig, ax, theme=th, alt_text=alt, long_description=alt + " " + note, caption=note,
                        provenance={"x": x, "y": y}, counts={**counts, "not_placed": not_placed},
                        kind="measure_agreement")
