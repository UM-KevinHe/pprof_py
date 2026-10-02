"""Interval plot ("caterpillar"): provider estimates with their test's intervals and a volume panel (spec §2.2)."""
from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np

from ..formatting import fmt_count, fmt_number
from ..theme import Theme, get_theme
from .._provenance import counts_text, interval_method, null_text, pct, reference_text, test_text
from ..data._resolve import resolve_profile
from ._common import freeze_layout, scaffold, spread
from ._result import FigureResult

__all__ = ["caterpillar"]

_LABEL_MAX = 60          # provider labels need about 3 mm per row at 7 pt (D34)
_RASTER = 2000
_ROW_MM = 3.0
_DENSE_MM = {"single": 85.0, "double": 110.0}
_MEASURES = {"indirect_ratio": "Indirectly standardized ratio (O/E)", "indirect_rate": "Indirectly standardized rate",
             "direct_ratio": "Directly standardized ratio", "direct_rate": "Directly standardized rate"}
_EFFECTS = {"log_odds": "Provider effect (log-odds)", "difference": "Provider effect (difference)"}
_DENOMINATORS = {"records": "Records\n(N)", "trials": "Trials\n(N)", "expected": "Expected\nevents (E)",
                 "patients": "Patients\n(N)", "person_time": "Person-\ntime"}     # two lines: the panel is narrow
_PHRASES = {"records": "records per provider", "trials": "trials per provider", "expected": "expected events per provider",
            "patients": "patients per provider", "person_time": "person-time per provider"}
_ORDER = ("above", "below", "not_different", "not_tested")
_ZORDER = {"not_different": 2.0, "not_tested": 3.0, "below": 4.0, "above": 4.0}


def caterpillar(source: Any, *args: Any, volume: Union[str, bool] = "auto", highlight: Optional[Iterable[Any]] = None,
                theme: Union[str, Theme, None] = "publication", size: Union[str, float] = "double",
                title: Optional[str] = None, **test_kwargs: Any) -> FigureResult:
    """Interval plot of provider estimates with the intervals of their test, and a volume panel.

    Parameters
    ----------
    source
        A fitted model (tested once here with ``test()``'s defaults), a ``test()`` result, or a
        :class:`~pprof_py.presentation.ProviderProfile`. For a standardized-measure scale, pass
        ``ProviderProfile.from_test(model.test_standardized(...), model=model)``.
    *args, **test_kwargs
        Passed to the model's ``test()`` (``test_method``, ``reference``, ``null_model``, ``level``, ``providers``,
        a CoxPH model's data, ...). Not allowed with a profile or a ``test()`` result.
    volume : {"auto", False}
        Draw the volume panel (the default; the profile must have denominators), or omit it with ``False``, which
        the footnote states.
    highlight : iterable, optional
        Providers to emphasise (bold labels, or annotations when the plot has too many rows for labels).
    theme : str or Theme
        A preset name or a :class:`~pprof_py.presentation.Theme`.
    size : {"single", "double"} or float
        Named width, or a width in millimetres.
    title : str, optional
        Title above the plot.

    Returns
    -------
    FigureResult

    Raises
    ------
    CapabilityError
        When the test has no intervals (for example the score test), or when denominators are missing and
        ``volume`` is not ``False``.

    Notes
    -----
    Providers are ordered by estimate for legibility only; the axis says so and is never labelled as a rank.
    Each interval is drawn from its lower to its upper bound, so intervals shifted by an empirical null render as
    they are. A provider without a finite estimate is marked at the axis edge and its exact one-sided interval is
    drawn from that edge (ADR-005); the solver's clamp is never drawn. Up to 60 providers are labelled; above 2,000
    the not-different providers are rasterized and flagged providers are drawn on top.
    """
    th = get_theme(theme)
    prof = resolve_profile(source, args, test_kwargs, display="caterpillar", limits=False)
    prof.require("caterpillar", "intervals")
    if volume is not False:
        prof.require("caterpillar (volume panel; pass volume=False to omit it)", "denominator")
    f, prov = prof.data, prof.provenance
    ids = f.index
    est = f["estimate"].to_numpy(dtype=float)
    lo, hi = f["ci_lower"].to_numpy(dtype=float), f["ci_upper"].to_numpy(dtype=float)
    status = f["status"].astype(str).to_numpy()
    ratio = prov.get("scale") == "ratio"
    nofinite = ~f["finite_estimate"].fillna(True).to_numpy(dtype=bool) & (not ratio)
    zero = f["zero_events"].fillna(False).to_numpy(dtype=bool) & ratio
    has_interval = f["has_interval"].to_numpy(dtype=bool)
    drawn = (status != "suppressed") & (np.isfinite(est) | nofinite)
    order = np.argsort(np.where(np.isfinite(est), est, np.inf), kind="stable")
    order = order[drawn[order]]
    if order.size == 0:
        raise ValueError("no provider has an estimate to draw")
    n = order.size
    row = np.full(len(f), np.nan)
    row[order] = np.arange(n, dtype=float)
    labelled, raster = n <= _LABEL_MAX, n > _RASTER
    ref = _null_value(prov, f)
    level = float(prov.get("level") or 0.95)

    core = drawn & ~nofinite
    xlo, xhi = _x_range(est[core], lo[core], hi[core], ref, ratio)
    seg_lo = np.clip(np.where(np.isfinite(lo), lo, -np.inf), xlo, xhi)
    seg_hi = np.clip(np.where(np.isfinite(hi), hi, np.inf), xlo, xhi)
    clipped = drawn & has_interval & ((~np.isfinite(lo)) | (~np.isfinite(hi)) | (lo < xlo) | (hi > xhi)) & ~nofinite
    inside_lo = np.isfinite(lo) & (lo > xlo) & (lo < xhi)
    inside_hi = np.isfinite(hi) & (hi > xlo) & (hi < xhi)
    segment = drawn & has_interval & (~nofinite | (inside_lo ^ inside_hi))   # ADR-005: edge to the finite bound only

    kind = prov.get("denominator_kind")
    denom = f["denominator"].to_numpy(dtype=float)
    entries = _legend_entries(th, status, drawn, nofinite, zero, level)
    note = _footnote(prof, prov, ratio, int((drawn & nofinite).sum()), int((drawn & ~has_interval).sum()),
                     int(clipped.sum()), volume, kind, level)
    row_mm = max(_ROW_MM, th.typography.tick * 25.4 / 72.0 * 1.25)     # labels never overlap (any preset)
    if isinstance(size, str):
        main_mm = n * row_mm + 16.0 if labelled else _DENSE_MM.get(size, 85.0)
    else:
        main_mm = n * row_mm + 16.0 if labelled else 0.6 * float(size)
    with th.rc_context():
        fig, axes, key, ncol, inset = scaffold(th, size, main_mm, [lab for _, lab in entries], note,
                                               width_ratios=None if volume is False else (4.0, 1.0))
        ax = axes if volume is False else axes[0]
        _draw_intervals(ax, th, row, seg_lo, seg_hi, status, segment, raster, labelled)
        _draw_points(ax, th, row, est, status, drawn, nofinite, zero, raster, labelled, ref, xlo, xhi)
        if ref is not None and np.isfinite(ref):
            ax.axvline(ref, color=th.reference, lw=th.lines.reference, zorder=1.2, gid="reference")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_ylabel("Providers, ordered by estimate")
        ax.set_xlabel(_x_label(prov, ratio) + ("" if ref is None or not np.isfinite(ref)
                                                else f"; reference {fmt_number(ref, 2)}"))
        _label_rows(ax, th, ids, order, row, est, seg_hi, xlo, xhi, labelled, highlight, main_mm, n)
        if volume is not False:
            _draw_volume(axes[1], th, row, denom, drawn, kind, raster, labelled)
        if title:
            ax.set_title(title, loc="left", fontsize=th.typography.title)
        if entries:
            key.legend([h for h, _ in entries], [lab for _, lab in entries], loc="upper left",
                       bbox_to_anchor=(inset, 1.0), ncol=ncol, frameon=False, borderaxespad=0.0, borderpad=0.0,
                       handletextpad=0.4, columnspacing=1.2, handlelength=1.4, fontsize=th.typography.legend)
        freeze_layout(fig, th)
    counts = prof.status_counts()
    alt = _alt_text(n, counts, ratio, level, denom[drawn], kind, volume)
    return FigureResult(fig, ax, theme=th, alt_text=alt, long_description=alt + " " + note, caption=note, provenance=prov,
                        counts=counts, kind="caterpillar")


def _null_value(prov: Any, f: Any) -> Optional[float]:
    if prov.get("null_value") is not None:
        return float(prov["null_value"])
    nv = f["null_value"].dropna().unique()
    return float(nv[0]) if len(nv) == 1 else None


def _x_range(est: np.ndarray, lo: np.ndarray, hi: np.ndarray, ref: Optional[float], ratio: bool) -> Tuple[float, float]:
    vals = np.r_[est, lo, hi]
    vals = vals[np.isfinite(vals)]
    if ref is not None and np.isfinite(ref):
        vals = np.r_[vals, ref]
    a, b = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
    pad = 0.04 * ((b - a) or 1.0)
    return (min(a, 0.0) - pad if ratio and a <= pad else a - pad), b + pad


def _x_label(prov: Any, ratio: bool) -> str:
    measure = prov.get("measure")
    if measure in _MEASURES:
        return "Observed / expected (O/E)" if prov.get("estimator") == "observed/expected ratio" else _MEASURES[measure]
    return _EFFECTS.get(prov.get("scale"), "Estimate")


def _marker(th: Theme, key: str, marker: Optional[str] = None) -> Any:
    from matplotlib.lines import Line2D

    st = th.status[key]
    return Line2D([], [], linestyle="none", marker=marker or st.marker, markersize=math.sqrt(st.size),
                  markerfacecolor=st.color if st.filled else "none", markeredgecolor=st.color, markeredgewidth=0.6)


def _legend_entries(th: Theme, status: np.ndarray, drawn: np.ndarray, nofinite: np.ndarray, zero: np.ndarray,
                    level: float) -> List[Tuple[Any, str]]:
    from matplotlib.lines import Line2D

    entries = [(Line2D([], [], color=th.muted, lw=th.lines.interval), f"{pct(level)} interval")]
    for key in _ORDER:
        k = int(((status == key) & drawn).sum())
        if k:
            entries.append((_marker(th, key), f"{th.status[key].label} ({fmt_count(k)})"))
    if (nofinite & drawn).any():
        entries.append((_marker(th, "no_finite_estimate", th.offscale_markers[0]),
                        f"{th.status['no_finite_estimate'].label}, at the axis edge "
                        f"({fmt_count(int((nofinite & drawn).sum()))})"))
    if (zero & drawn).any():
        entries.append((_marker(th, "no_finite_estimate"), f"Zero events, O/E = 0 ({fmt_count(int((zero & drawn).sum()))})"))
    return entries


def _footnote(prof: Any, prov: Any, ratio: bool, n_nofinite: int, n_nointerval: int, n_clipped: int, volume: Any,
              kind: Optional[str], level: float) -> str:
    bits = [counts_text(prof)]
    if n_nofinite:
        bits.append(f"{fmt_count(n_nofinite)} without a finite estimate, marked at the axis edge (a one-sided "
                    "interval with a finite bound is drawn from the edge)")
    if n_nointerval:
        bits.append(f"{fmt_count(n_nointerval)} without an interval (point only)")
    if n_clipped:
        bits.append(f"{fmt_count(n_clipped)} intervals extend beyond the axis and are drawn to its edge")
    text = "; ".join(bits) + "."
    text += f" Intervals: {pct(level)} {interval_method(prov)}intervals from the same test as the flags"
    if prov.get("alternative") == "two_sided" and not prov.get("s3_violations"):
        text += "; an interval excludes the reference exactly when the provider is flagged"
    text += (f". Test: {test_text(prov)}; {null_text(prov.get('null_model'))}; reference: "
             f"{reference_text(prov, False)}. Estimates: {prov.get('estimator') or 'estimator not stated'}. Providers "
             "are ordered by estimate for legibility; the order is not a ranking.")
    if volume is False:
        text += " Denominators are not shown."
    else:
        text += f" Bars: {_PHRASES.get(kind) or 'denominator per provider'}."
    return text


def _draw_intervals(ax: Any, th: Theme, row: np.ndarray, lo: np.ndarray, hi: np.ndarray, status: np.ndarray,
                    mask: np.ndarray, raster: bool, labelled: bool) -> None:
    from matplotlib.collections import LineCollection

    width = th.lines.interval if labelled else th.lines.interval_dense
    for key in ("not_different", "not_tested", "below", "above"):
        m = mask & (status == key)
        if not m.any():
            continue
        segs = np.stack([np.c_[lo[m], row[m]], np.c_[hi[m], row[m]]], axis=1)
        color = th.status[key].color if key != "not_different" else th.volume
        ax.add_collection(LineCollection(segs, colors=color, linewidths=width, zorder=_ZORDER[key] - 0.5,
                                         rasterized=raster and key == "not_different", gid=f"interval-{key}"))


def _draw_points(ax: Any, th: Theme, row: np.ndarray, est: np.ndarray, status: np.ndarray, drawn: np.ndarray,
                 nofinite: np.ndarray, zero: np.ndarray, raster: bool, labelled: bool, ref: Optional[float],
                 xlo: float, xhi: float) -> None:
    dense = not labelled
    for key in ("not_different", "not_tested", "below", "above"):
        m = drawn & (status == key) & ~nofinite
        if not m.any():
            continue
        st = th.status[key]
        ax.scatter(est[m], row[m], marker=st.marker, s=st.size * (0.5 if dense and key == "not_different" else 1.0),
                   facecolors=st.color if st.filled else "none", edgecolors=st.color, linewidths=0.6,
                   zorder=_ZORDER[key], rasterized=raster and key == "not_different", gid=f"status-{key}")
    ne = th.status["no_finite_estimate"]
    m = drawn & nofinite
    if m.any():                                          # never at the solver's clamp: at the axis edge (ADR-005)
        below = np.nan_to_num(est[m], nan=0.0) < (ref if ref is not None else 0.0)
        for side, sel, marker in (("lower", below, th.offscale_markers[0]), ("upper", ~below, th.offscale_markers[1])):
            if sel.any():
                ax.scatter(np.full(int(sel.sum()), xlo if side == "lower" else xhi), row[m][sel], marker=marker,
                           s=ne.size * (0.5 if dense else 1.0), facecolors=ne.color, edgecolors=ne.color, linewidths=0.6, zorder=5.0,
                           clip_on=False, gid=f"no-finite-estimate-{side}")
    m = drawn & zero
    if m.any():
        ax.scatter(est[m], row[m], marker=ne.marker, s=ne.size * (0.5 if dense else 1.0), facecolors="none",
                   edgecolors=ne.color, linewidths=0.6, zorder=5.0, gid="zero-events")


def _label_rows(ax: Any, th: Theme, ids: Any, order: np.ndarray, row: np.ndarray, est: np.ndarray, seg_hi: np.ndarray,
                xlo: float, xhi: float, labelled: bool, highlight: Optional[Iterable[Any]], main_mm: float,
                n: int) -> None:
    wanted = set() if highlight is None else set(highlight)
    if labelled:
        ax.set_yticks(np.arange(n))
        labels = ax.set_yticklabels([str(ids[i]) for i in order], fontsize=th.typography.tick)
        for text, i in zip(labels, order):
            if ids[i] in wanted:
                text.set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        return
    ax.set_yticks([])
    idx = [i for i in order if ids[i] in wanted]
    if not idx:
        return
    gap = th.typography.annotation * 25.4 / 72.0 * 1.3 / (0.8 * main_mm) * n
    pos = spread([row[i] for i in idx], gap, 0.0, n - 1.0)
    span = xhi - xlo
    for i, yv in zip(idx, pos):
        x_text = min(seg_hi[i] + 0.02 * span, xhi - 0.02 * span)
        ax.annotate(str(ids[i]), (seg_hi[i], row[i]), xytext=(x_text, yv), textcoords="data", ha="left", va="center",
                    fontsize=th.typography.annotation, color=th.ink, zorder=6.0,
                    arrowprops=None if abs(yv - row[i]) < 1e-9 else dict(arrowstyle="-", color=th.muted, lw=0.4))


def _draw_volume(ax: Any, th: Theme, row: np.ndarray, denom: np.ndarray, drawn: np.ndarray, kind: Optional[str],
                 raster: bool, labelled: bool) -> None:
    from matplotlib.collections import PolyCollection
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    m = drawn & np.isfinite(denom)
    half = 0.35 if labelled else 0.5
    y, d = row[m], denom[m]
    verts = np.stack([np.c_[np.zeros_like(d), y - half], np.c_[d, y - half], np.c_[d, y + half],
                      np.c_[np.zeros_like(d), y + half]], axis=1)
    ax.add_collection(PolyCollection(verts, facecolors=th.volume, edgecolors="none", linewidths=0.0, zorder=2.0,
                                     rasterized=raster, gid="volume"))
    ax.set_xlim(0.0, float(d.max()) * 1.05 if d.size and d.max() > 0 else 1.0)
    ax.set_xlabel(_DENOMINATORS.get(kind) or "Denominator")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=kind in ("records", "trials", "patients")))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_count(v) if float(v).is_integer() else fmt_number(v, 1)))
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)


def _alt_text(n: int, c: Any, ratio: bool, level: float, denom: np.ndarray, kind: Optional[str], volume: Any) -> str:
    text = (f"Interval plot of {fmt_count(n)} providers ordered by estimate, with {pct(level)} intervals: "
            f"{fmt_count(c['above'])} above and {fmt_count(c['below'])} below the reference, "
            f"{fmt_count(c['not_different'])} not different")
    if c["not_tested"]:
        text += f", {fmt_count(c['not_tested'])} not tested"
    if c["suppressed"]:
        text += f", {fmt_count(c['suppressed'])} suppressed"
    if c["no_finite_estimate"] and not ratio:
        text += f"; {fmt_count(c['no_finite_estimate'])} without a finite estimate"
    if c["zero_events"] and ratio:
        text += f"; {fmt_count(c['zero_events'])} with zero events"
    d = denom[np.isfinite(denom)]
    if volume is not False and d.size:
        label = _PHRASES.get(kind) or "denominator per provider"
        text += (f". Volume panel: {label} from {fmt_number(d.min(), 0)} to {fmt_number(d.max(), 0)} "
                 f"(median {fmt_number(float(np.median(d)), 0)})")
    return text + "."
