"""Funnel plot: provider estimates against precision, with control limits from the same test (S4; ADR-003)."""
from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..formatting import fmt_count, fmt_number
from ..theme import Theme, get_theme
from .._provenance import counts_text, null_text, pct, reference_text, test_text
from ..data._resolve import resolve_profile
from ._common import PT_MM, freeze_layout, log_ticks, scaffold, spread, text_width_mm
from ._result import FigureResult

__all__ = ["funnel"]

_DENSE = 2000
_LABEL_FLAGGED_MAX = 10
_MAIN_MM = {"single": 62.0, "double": 95.0}
_X_LABELS = {"expected": "Expected events, E", "inverse_null_variance": "Precision under the null, E\u00b2/V\u2080",
             "inverse_variance": "Precision, 1/SE\u00b2"}
_EFFECT_LABELS = {"log_odds": "Provider effect (log-odds)", "difference": "Provider effect (difference)"}
_FALLBACK_DASH = (0, (6.0, 1.5, 1.0, 1.5))
_ORDER = ("above", "below", "not_different", "not_tested")
_ZORDER = {"not_different": 2.0, "not_tested": 3.0, "below": 4.0, "above": 4.0}


def funnel(source: Any, *args: Any, levels: Iterable[float] = (0.95, 0.998), highlight: Optional[Iterable[Any]] = None,
           label_flagged: Union[str, bool] = "auto", theme: Union[str, Theme, None] = "publication",
           size: Union[str, float] = "single", title: Optional[str] = None, **test_kwargs: Any) -> FigureResult:
    """Funnel plot whose control limits come from the same test as the flags.

    Parameters
    ----------
    source
        A fitted model (tested once here, with the funnel's defaults, for example the score test for logistic
        fixed effects), or a :class:`~pprof_py.presentation.ProviderProfile` with funnel limits.
    *args, **test_kwargs
        Passed to the model's ``funnel_limits()``: ``test_method``, ``reference``, ``null_model``, ``level``,
        ``providers``, a CoxPH model's data, ... Not allowed with a profile.
    levels : sequence of float, default (0.95, 0.998)
        Levels of the limit curves. Flags exist only at the test's level; the others are reference curves.
    highlight : iterable, optional
        Providers to label.
    label_flagged : {"auto", True, False}
        Label flagged providers: ``"auto"`` labels them when there are at most 10 and the plot is not dense.
    theme : str or Theme
        A preset name (``"publication"``, ``"notebook"``, ``"report"``) or a :class:`~pprof_py.presentation.Theme`.
    size : {"single", "double"} or float
        Named width, or a width in millimetres.
    title : str, optional
        Title above the plot (publication figures usually carry it in the caption instead).

    Returns
    -------
    FigureResult

    Raises
    ------
    CapabilityError
        When the source has no funnel limits, for example a linear random-effect model (ADR-004).

    Notes
    -----
    A tested provider lies outside its own limits exactly when it is flagged (S4). Exact count tests draw each
    provider's own limits as marks, because their limits depend on more than the expected count; their curves are a
    Poisson reference. Zero-event providers sit at O/E = 0 with an outline marker; above 2,000 providers the
    not-different points are rasterized and flagged providers are drawn on top.
    """
    th = get_theme(theme)
    prof = resolve_profile(source, args, test_kwargs, display="funnel", limits=True, levels=tuple(levels))
    prof.require("funnel", "funnel_limits")
    f, prov = prof.data, prof.provenance
    fa = dict(prov.get("funnel") or {})
    ratio = (fa.get("estimate_kind") or ("ratio" if prov.get("scale") == "ratio" else "effect")) == "ratio"
    curves = prof.funnel_curves
    curve_kind = fa.get("curve_kind")
    exact_curves = curve_kind == "exact" and curves is not None and len(curves) > 0
    test_level = float(prov.get("level") or 0.95)
    shown_levels = tuple(fa.get("levels") or (test_level,))
    x, y = f["funnel_precision"].to_numpy(dtype=float), f["funnel_estimate"].to_numpy(dtype=float)
    lo, hi = f["funnel_lower"].to_numpy(dtype=float), f["funnel_upper"].to_numpy(dtype=float)
    status = f["status"].astype(str).to_numpy()
    zero = f["zero_events"].fillna(False).to_numpy(dtype=bool) & ratio
    nofinite = ~f["finite_estimate"].fillna(True).to_numpy(dtype=bool) & (not ratio)
    drawn = status != "suppressed"
    shown = drawn & np.isfinite(x) & (x > 0) & (np.isfinite(y) | nofinite)
    if not shown.any():
        raise ValueError("no provider has a finite precision and estimate to place on the funnel")
    n_unplaced = int((drawn & ~shown).sum())
    dense = len(f) > _DENSE
    ref = 1.0 if ratio else _null_value(prov, f)
    counts = prof.status_counts()

    exact_marks = "count boundaries" in (fa.get("limit_rule") or "")
    corridor = _corridor_curve(th, curves, exact_curves, test_level)
    entries = _legend_entries(th, status, shown, zero, nofinite, exact_curves, exact_marks, test_level)
    if corridor is not None:
        from matplotlib.patches import Patch
        entries.append((Patch(facecolor=th.corridor, edgecolor=th.limit, linewidth=0.5),
                        f"Not flagged at {pct(test_level)}"))
    note = _footnote(prof, prov, ratio, fa, exact_curves, exact_marks, curve_kind, shown_levels, test_level, n_unplaced)
    shown_note = (_short_footnote(prof, prov, ratio, exact_curves, exact_marks, shown_levels, test_level, n_unplaced)
                  if th.footnote == "short" else note)
    key_top = th.key_position == "top"
    if isinstance(size, str):
        main_mm = _MAIN_MM.get(size, 62.0)              # unknown names are rejected by theme.figsize
    else:
        main_mm = 0.7 * float(size)
    with th.rc_context():
        fig, ax, key, ncol, inset = scaffold(th, size, main_mm, [label for _, label in entries], shown_note,
                                             key_top=key_top, title=title if key_top else None)
        ax.set_xscale("log")
        xs = x[shown]
        x0, x1 = float(xs.min()), float(xs.max())
        span = math.log10(x1 / x0) if x1 > x0 else 1.0
        left = x0 / 10 ** (0.05 * span)
        body = shown & np.isfinite(lo) & np.isfinite(hi)
        if body.any():                                  # keep the acceptance region in view (not only the points)
            body &= x >= np.median(x[body])
        ylo, yhi = _y_range(np.r_[y[shown & ~nofinite & np.isfinite(y)], lo[body], hi[body]], ref, ratio)
        ax.set_ylim(ylo, yhi)
        discrete = "count boundaries" in (fa.get("limit_rule") or "")
        labels = _draw_limits(ax, th, curves, exact_curves, discrete, x, lo, hi, shown, dense, test_level)
        if corridor is not None:                        # where a provider of that precision is not flagged
            ax.fill_between(corridor[0], corridor[1], corridor[2], facecolor=th.corridor, edgecolor="none",
                            linewidth=0.0, zorder=0.6, gid="corridor")
        if ref is not None and np.isfinite(ref):            # ends with the data: the label strip stays clear
            ax.plot([left, x1 * 10 ** (0.015 * span)], [ref, ref], color=th.reference, lw=th.lines.reference,
                    zorder=1.2, solid_capstyle="butt", gid="reference")
            labels.append((float(ref), fmt_number(ref, 2), th.reference))
        strip = max([text_width_mm(t, th.typography.annotation, th.typography.family) for _, t, _ in labels] or [0.0])
        frac = min(0.35, (strip + 2.5) / max(theme_axes_mm(th, size), 1.0))
        right_span = (math.log10(x1 * 10 ** (0.03 * span)) - math.log10(left)) * frac / (1.0 - frac)
        ax.set_xlim(left, x1 * 10 ** (0.03 * span) * 10 ** right_span)
        _draw_points(ax, th, x, y, status, shown, zero, nofinite, dense, ref, ylo, yhi)
        _direct_labels(ax, th, labels, x1 * 10 ** (0.03 * span), ylo, yhi, main_mm)
        _provider_labels(ax, th, f.index, x, y, status, shown & ~nofinite, highlight, label_flagged, dense, ylo, yhi,
                         main_mm)
        log_ticks(ax.xaxis)
        ax.set_xlabel(_X_LABELS.get(fa.get("precision_kind"), "Precision") + " (log scale)")
        ax.set_ylabel("Observed / expected (O/E)" if ratio else _EFFECT_LABELS.get(prov.get("scale"), "Estimate"))
        if title and not key_top:
            ax.set_title(title, loc="left", fontsize=th.typography.title)
        if entries:
            key.legend([h for h, _ in entries], [lab for _, lab in entries], loc="upper left",
                       bbox_to_anchor=(inset, 1.0), ncol=ncol, frameon=False, borderaxespad=0.0, borderpad=0.0,
                       handletextpad=0.4, columnspacing=1.2, handlelength=1.0, fontsize=th.typography.legend)
        freeze_layout(fig, th)
    alt = _alt_text(len(f), counts, ratio, test_level, fa)
    return FigureResult(fig, ax, theme=th, alt_text=alt, long_description=alt + " " + note, caption=note, provenance=prov,
                        counts=counts, kind="funnel")


def theme_axes_mm(th: Theme, size: Union[str, float]) -> float:
    """Approximate width of the data axes: the figure width less the y-axis label and tick margin."""
    return th.figsize(size)[0] * 25.4 - 15.0


def _null_value(prov: Any, f: pd.DataFrame) -> Optional[float]:
    if prov.get("null_value") is not None:
        return float(prov["null_value"])
    nv = f["null_value"].dropna().unique()
    return float(nv[0]) if len(nv) == 1 else None


def _marker(th: Theme, key: str, size: Optional[float] = None) -> Any:
    from matplotlib.lines import Line2D

    st = th.status[key]
    halo = st.filled and th.halo > 0
    return Line2D([], [], linestyle="none", marker=st.marker, markersize=math.sqrt(size or st.size),
                  markerfacecolor=st.color if st.filled else "none",
                  markeredgecolor=th.background if halo else st.color, markeredgewidth=th.halo if halo else 0.6)


def _legend_entries(th: Theme, status: np.ndarray, shown: np.ndarray, zero: np.ndarray, nofinite: np.ndarray,
                    exact_curves: bool, exact_marks: bool, test_level: float) -> List[Tuple[Any, str]]:
    from matplotlib.lines import Line2D

    entries = []
    for key in _ORDER:
        k = int(((status == key) & shown).sum())
        if k:
            entries.append((_marker(th, key), f"{th.status[key].label} ({fmt_count(k)})"))
    if (zero & shown).any():
        entries.append((_marker(th, "no_finite_estimate"), f"Zero events, O/E = 0 ({fmt_count(int((zero & shown).sum()))})"))
    if (nofinite & shown).any():
        entries.append((_marker(th, "no_finite_estimate"),
                        f"{th.status['no_finite_estimate'].label}, at the axis edge ({fmt_count(int((nofinite & shown).sum()))})"))
    if not exact_curves:
        entries.append((Line2D([], [], linestyle="none", marker="_", markersize=6.0, markeredgecolor=th.limit,
                               markeredgewidth=0.8),
                        f"{'Exact ' if exact_marks else ''}{pct(test_level)} limits, each provider"))
    return entries


def _footnote(prof: Any, prov: Any, ratio: bool, fa: dict, exact_curves: bool, exact_marks: bool,
              curve_kind: Optional[str], levels: Tuple[float, ...], test_level: float, n_unplaced: int) -> str:
    rule = fa.get("limit_rule") or "limits supplied with the data"
    others = [pct(v) for v in levels if v != test_level]
    if exact_curves:
        limits = f"the {pct(test_level)} curves reproduce the flags ({rule})"
        if others:
            limits += f"; the {', '.join(others)} curves are for reference only"
    else:
        limits = (f"each provider's {'exact ' if exact_marks else ''}{pct(test_level)} limits (marks) "
                  f"{'reproduce the flags' if exact_marks else 'as supplied'} ({rule})")
        if curve_kind == "poisson_reference":
            limits += f"; the curves are Poisson references at {', '.join(pct(v) for v in levels)}"
    text = (f"{counts_text(prof)}. Limits: {limits}. Test: {test_text(prov)}; {null_text(prov.get('null_model'))}; "
            f"reference: {reference_text(prov, ratio)}. Estimates: {prov.get('estimator') or 'estimator not stated'}.")
    if n_unplaced:
        text += f" {fmt_count(n_unplaced)} provider(s) without a finite precision or estimate are not drawn."
    return text


def _corridor_curve(th: Theme, curves: Optional[pd.DataFrame], exact_curves: bool,
                    test_level: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Precision, lower and upper limit of the test-level curve, when a corridor may be shaded: the theme has one,
    the curves reproduce the flags (``exact_curves``) and there is a single null group. Per-provider limit marks
    (exact count tests) and per-group curves (a grouped empirical null) are never shaded, because one region would
    misplace providers."""
    if not th.corridor or not exact_curves or curves is None or not len(curves):
        return None
    c = curves[np.isclose(curves["level"].to_numpy(dtype=float), test_level)]
    if not len(c) or c["null_group"].nunique(dropna=False) > 1:
        return None
    c = c.sort_values("precision", kind="mergesort")
    return (c["precision"].to_numpy(dtype=float), c["lower"].to_numpy(dtype=float),
            c["upper"].to_numpy(dtype=float))


def _short_footnote(prof: Any, prov: Any, ratio: bool, exact_curves: bool, exact_marks: bool,
                    levels: Tuple[float, ...], test_level: float, n_unplaced: int) -> str:
    """One provenance line for publication figures; the full text is the figure's caption."""
    others = [pct(v) for v in levels if v != test_level]
    text = f"{counts_text(prof)}. Test: {test_text(prov)}; reference: {reference_text(prov, ratio)}."
    nm = prov.get("null_model")
    if isinstance(nm, dict) and nm.get("kind") not in (None, "theoretical"):
        text = text[:-1] + f"; {null_text(nm)}."
    if not exact_curves:
        text += f" Marks: each provider's {'exact ' if exact_marks else ''}{pct(test_level)} limits."
    elif others:
        text += f" The {', '.join(others)} curves are for reference only."
    if n_unplaced:
        text += f" {fmt_count(n_unplaced)} provider(s) without a finite precision or estimate are not drawn."
    return text


def _y_range(values: np.ndarray, ref: Optional[float], ratio: bool) -> Tuple[float, float]:
    vals = values[np.isfinite(values)]
    if ref is not None and np.isfinite(ref):
        vals = np.r_[vals, ref]
    lo_v, hi_v = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
    pad = 0.07 * ((hi_v - lo_v) or 1.0)
    return (min(0.0, lo_v) - pad if ratio else lo_v - pad), hi_v + pad


def _dash(th: Theme, level: float) -> Any:
    return th.level_dashes.get(float(level), _FALLBACK_DASH)


def _draw_limits(ax: Any, th: Theme, curves: Optional[pd.DataFrame], exact_curves: bool, discrete: bool,
                 x: np.ndarray, lo: np.ndarray, hi: np.ndarray, shown: np.ndarray, dense: bool,
                 test_level: float) -> List[Tuple[float, str, str]]:
    """Limit curves (dashed by level; solid and thin for discrete counts, where dashes are illegible) and marks."""
    labels: List[Tuple[float, str, str]] = []
    if curves is not None and len(curves):
        group = curves["null_group"].astype(object).where(curves["null_group"].notna(), "")
        for lev in sorted(curves["level"].unique()):
            at_level = curves["level"] == lev
            own = exact_curves and float(lev) == test_level
            if discrete or not exact_curves:
                color, width, style = (th.limit if own else th.volume), 0.55, "-"
            else:
                color, width, style = th.limit, th.lines.limit, _dash(th, lev)
            last = None
            for g in pd.unique(group[at_level]):
                cg = curves[at_level & (group == g)]
                for col in ("upper", "lower"):
                    ax.plot(cg["precision"].to_numpy(), cg[col].to_numpy(), color=color, lw=width, linestyle=style,
                            zorder=1.0, gid=f"limit-curve-{float(lev):g}-{g}-{col}")
                if last is None or cg["precision"].iloc[-1] > last["precision"].iloc[-1]:
                    last = cg
            if last is not None and np.isfinite(last["upper"].iloc[-1]):
                labels.append((float(last["upper"].iloc[-1]), pct(lev), th.limit if own or not discrete else th.muted))
    if not exact_curves:
        for side, lim in (("lower", lo), ("upper", hi)):
            m = shown & np.isfinite(lim)
            ax.scatter(x[m], lim[m], marker="_", s=16.0, color=th.limit, linewidths=0.8, zorder=1.5, rasterized=dense,
                       gid=f"limit-marks-{side}")
    return labels


def _draw_points(ax: Any, th: Theme, x: np.ndarray, y: np.ndarray, status: np.ndarray, shown: np.ndarray,
                 zero: np.ndarray, nofinite: np.ndarray, dense: bool, ref: Optional[float], ylo: float,
                 yhi: float) -> None:
    for key in ("not_different", "not_tested", "below", "above"):
        m = shown & (status == key) & ~nofinite
        if not m.any():
            continue
        st = th.status[key]
        halo = st.filled and th.halo > 0                 # a background-coloured edge separates marks from lines
        ax.scatter(x[m], y[m], marker=st.marker, s=st.size * (0.6 if dense and key == "not_different" else 1.0),
                   facecolors=st.color if st.filled else "none", edgecolors=th.background if halo else st.color,
                   linewidths=th.halo if halo else 0.6,
                   zorder=_ZORDER[key], rasterized=dense and key == "not_different", gid=f"status-{key}")
    ne = th.status["no_finite_estimate"]
    if (zero & shown).any():
        ax.scatter(x[zero & shown], y[zero & shown], marker=ne.marker, s=ne.size * (0.45 if dense else 1.0),
                   facecolors="none", edgecolors=ne.color, linewidths=0.45 if dense else 0.6, zorder=5.0,
                   gid="zero-events")
    m = nofinite & shown
    if m.any():                                          # never at the solver's clamp: at the axis edge (ADR-005)
        inset = 0.02 * (yhi - ylo)
        side = np.where(np.nan_to_num(y[m], nan=0.0) < (ref if ref is not None else 0.0), ylo + inset, yhi - inset)
        ax.scatter(x[m], side, marker=ne.marker, s=ne.size, facecolors="none", edgecolors=ne.color, linewidths=0.6,
                   zorder=5.0, clip_on=False, gid="no-finite-estimate")


def _gap(th: Theme, ylo: float, yhi: float, main_mm: float) -> float:
    return th.typography.annotation * PT_MM * 1.3 / (0.72 * main_mm) * (yhi - ylo)


def _direct_labels(ax: Any, th: Theme, labels: List[Tuple[float, str, str]], x_text: float, ylo: float, yhi: float,
                   main_mm: float) -> None:
    if not labels:
        return
    pos = spread([v for v, _, _ in labels], _gap(th, ylo, yhi, main_mm), ylo, yhi)
    for (_, text, color), yv in zip(labels, pos):
        ax.text(x_text, yv, text, ha="left", va="center", fontsize=th.typography.annotation, color=color, zorder=6.0)


def _provider_labels(ax: Any, th: Theme, ids: pd.Index, x: np.ndarray, y: np.ndarray, status: np.ndarray,
                     shown: np.ndarray, highlight: Optional[Iterable[Any]], label_flagged: Union[str, bool],
                     dense: bool, ylo: float, yhi: float, main_mm: float) -> None:
    flagged = shown & np.isin(status, ("above", "below"))
    want = np.zeros(len(ids), dtype=bool)
    if highlight is not None:
        want |= np.asarray(ids.isin(list(highlight)), dtype=bool) & shown
    if label_flagged is True or (label_flagged == "auto" and not dense and flagged.sum() <= _LABEL_FLAGGED_MAX):
        want |= flagged
    idx = np.flatnonzero(want)
    if idx.size == 0:
        return
    chosen = (np.asarray(ids.isin(list(highlight)), dtype=bool) & shown) if highlight is not None else np.zeros(len(ids), bool)
    ring = chosen & bool(th.highlight_ring)
    if ring.any():                                       # the selected providers: a ring and an emphasized label
        size = max(s.size for s in th.status.values())
        ax.scatter(x[ring], y[ring], marker="o", s=size * 3.2, facecolors="none", edgecolors=th.highlight_ring,
                   linewidths=0.8, zorder=5.5, gid="highlight-ring")
    pos = spread(y[idx].tolist(), _gap(th, ylo, yhi, main_mm), ylo, yhi)
    for i, yv in zip(idx, pos):
        ax.annotate(str(ids[i]), (x[i], y[i]), xytext=(x[i] * 1.12, yv), textcoords="data", ha="left", va="center",
                    fontsize=th.typography.annotation, color=th.ink, zorder=6.0,
                    gid="emphasis" if ring[i] else None,
                    arrowprops=None if abs(yv - y[i]) < 1e-12 else dict(arrowstyle="-", color=th.muted, lw=0.4))


def _alt_text(n: int, c: Any, ratio: bool, level: float, fa: dict) -> str:
    what = "observed over expected events" if ratio else "provider effects"
    text = (f"Funnel plot of {fmt_count(n)} providers, {what} against precision: {fmt_count(c['above'])} above and "
            f"{fmt_count(c['below'])} below the reference at the {pct(level)} level, "
            f"{fmt_count(c['not_different'])} not different")
    if c["not_tested"]:
        text += f", {fmt_count(c['not_tested'])} not tested"
    if c["suppressed"]:
        text += f", {fmt_count(c['suppressed'])} suppressed"
    if c["zero_events"] and ratio:
        text += f"; {fmt_count(c['zero_events'])} with zero events"
    if c["no_finite_estimate"] and not ratio:
        text += f"; {fmt_count(c['no_finite_estimate'])} without a finite estimate"
    return text + "."
