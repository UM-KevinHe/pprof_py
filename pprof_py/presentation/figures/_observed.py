"""Observed versus expected events: where counts depart from expectation, and at what volume (spec §5.3 Tier 2)."""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd

from .._provenance import counts_text, null_text, pct, test_text
from ..data._resolve import resolve_profile
from ..formatting import fmt_count, fmt_number
from ..theme import Theme, get_theme
from ._common import freeze_layout, scaffold, spread
from ._result import FigureResult

__all__ = ["observed_expected"]

_ORDER = ("above", "below", "not_different", "not_tested")
_ZORDER = {"not_different": 2.0, "not_tested": 3.0, "below": 4.0, "above": 4.0}
_DENSE = 2000


def _sqrt_axis(axis_setter: Any) -> None:
    axis_setter("function", functions=(lambda v: np.sqrt(np.maximum(v, 0.0)), np.square))


def _nice(v: float) -> float:
    e = 10.0 ** np.floor(np.log10(v))
    m = v / e
    return float((1.0 if m < 1.5 else 2.0 if m < 3.0 else 5.0 if m < 7.0 else 10.0) * e)


def sqrt_ticks(top: float) -> list:
    """Nice tick values evenly spaced on a square-root axis from 0 to ``top`` (linear spacing crowds at the top)."""
    raw = np.linspace(0.0, np.sqrt(top), 6)[1:] ** 2
    ticks = sorted({_nice(v) for v in raw if v > 0})
    return [0.0] + [v for v in ticks if v <= top and (v >= 1.0 or top < 3.0)]


def _set_sqrt_ticks(axis: Any, top: float) -> None:
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    axis.set_major_locator(FixedLocator(sqrt_ticks(top)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.10g}"))
    axis.set_minor_locator(NullLocator())


def observed_expected(source: Any, *args: Any, highlight: Optional[Iterable[Any]] = None,
                      theme: Union[str, Theme, None] = "publication", size: Union[str, float] = "single",
                      title: Optional[str] = None, **test_kwargs: Any) -> FigureResult:
    """Observed against expected events on square-root axes, with the test's limits converted to counts.

    Parameters
    ----------
    source
        A fitted model (tested once here with its funnel limits, as :func:`funnel` would), or a
        :class:`~pprof_py.presentation.ProviderProfile` with observed and expected counts (for example from a
        CoxPH test, or built with ``limits=True``).
    *args, **test_kwargs
        Passed to the model's ``funnel_limits()``; not allowed with a profile.
    highlight : iterable, optional
        Providers to label.
    theme, size, title
        As for the other figures.

    Returns
    -------
    FigureResult

    Raises
    ------
    CapabilityError
        When the source has no observed or expected counts.

    Notes
    -----
    On square-root axes Poisson noise has roughly constant spread, so distances from the line O = E are comparable
    across volumes: the same ratio means more excess events at a larger volume. Limits are the funnel limits of the
    same test multiplied by the provider's expected count (count tests: half-integer counts, so no provider lies on
    a limit).
    """
    th = get_theme(theme)
    prof = resolve_profile(source, args, test_kwargs, display="observed_expected", limits=True)
    prof.require("observed_expected", "observed", "expected")
    f, prov = prof.data, prof.provenance
    fa = dict(prov.get("funnel") or {})
    obs, exp_ = f["observed"].to_numpy(dtype=float), f["expected"].to_numpy(dtype=float)
    status = f["status"].astype(str).to_numpy()
    shown = (status != "suppressed") & np.isfinite(obs) & np.isfinite(exp_) & (exp_ > 0)
    lo = f["funnel_lower"].to_numpy(dtype=float) * exp_
    hi = f["funnel_upper"].to_numpy(dtype=float) * exp_
    has_limits = "funnel_limits" in prof.capabilities and fa.get("estimate_kind", "ratio") == "ratio"
    # only the test's own limits as curves (CoxPH); Poisson references of exact count tests are not this test
    curves = (prof.funnel_curves if has_limits and fa.get("precision_kind") == "expected"
              and fa.get("curve_kind") == "exact" else None)
    dense = len(f) > _DENSE
    level = float(prov.get("level") or 0.95)
    top = float(max(np.nanmax(obs[shown]), np.nanmax(exp_[shown]))) * 1.06
    from matplotlib.lines import Line2D

    entries = []
    for key in _ORDER:
        k = int(((status == key) & shown).sum())
        if k:
            st = th.status[key]
            entries.append((Line2D([], [], linestyle="none", marker=st.marker, markersize=math.sqrt(st.size),
                                   markerfacecolor=st.color if st.filled else "none", markeredgecolor=st.color,
                                   markeredgewidth=0.6), f"{st.label} ({fmt_count(k)})"))
    if has_limits:
        entries.append((Line2D([], [], linestyle="none", marker="_", markersize=6.0, markeredgecolor=th.limit,
                               markeredgewidth=0.8), f"{pct(level)} limits of the test, in events"))
    note = (f"{counts_text(prof)}. Observed and expected events per provider on square-root axes, where Poisson noise "
            "has roughly constant spread, so distances from the line O = E are comparable across volumes; the same "
            f"ratio means more excess events at a larger volume. Test: {test_text(prov)}; "
            f"{null_text(prov.get('null_model'))}.")
    if has_limits:
        note += (" Limits: the test's funnel limits multiplied by each provider's expected count"
                 + (" (half-integer counts for count tests)." if "count boundaries" in (fa.get("limit_rule") or "")
                    else "."))
    else:
        note += " The source has no funnel limits, so none are drawn."
    with th.rc_context():
        fig, ax, key, ncol, inset = scaffold(th, size, 68.0, [lab for _, lab in entries], note)
        _sqrt_axis(ax.set_xscale)
        _sqrt_axis(ax.set_yscale)
        ax.set_xlim(0.0, top)
        ax.set_ylim(0.0, top)
        _set_sqrt_ticks(ax.xaxis, top)
        _set_sqrt_ticks(ax.yaxis, top)
        line = np.linspace(0.0, top, 200)
        ax.plot(line, line, color=th.reference, lw=th.lines.reference, zorder=1.2, gid="identity")
        for ratio in (0.5, 2.0):
            ax.plot(line, ratio * line, color=th.volume, lw=th.lines.grid, zorder=0.9, gid=f"ratio-{ratio:g}")
        labels = [(top * 0.97, "O = E", th.reference), (top * 0.97 * 0.5, "O/E = 0.5", th.muted)]
        ax.text(top * 0.48, top * 0.97, "O/E = 2", ha="right", va="bottom", fontsize=th.typography.annotation,
                color=th.muted, zorder=6.0)
        if has_limits:
            if curves is not None and len(curves):
                test_curves = curves[curves["test_level"]]
                for g, cg in test_curves.groupby(test_curves["null_group"].astype(object).where(
                        test_curves["null_group"].notna(), ""), sort=False):
                    for col in ("upper", "lower"):
                        ax.plot(cg["precision"], cg[col] * cg["precision"], color=th.limit, lw=0.55, zorder=1.0,
                                gid=f"count-limit-{col}-{g}")
            for side, lim in (("lower", lo), ("upper", hi)):
                m = shown & np.isfinite(lim) & (lim >= 0)
                ax.scatter(exp_[m], lim[m], marker="_", s=16.0, color=th.limit, linewidths=0.8, zorder=1.5,
                           rasterized=dense, gid=f"count-marks-{side}")
        for key_ in ("not_different", "not_tested", "below", "above"):
            m = shown & (status == key_)
            if not m.any():
                continue
            st = th.status[key_]
            ax.scatter(exp_[m], obs[m], marker=st.marker, s=st.size * (0.6 if dense and key_ == "not_different" else 1.0),
                       facecolors=st.color if st.filled else "none", edgecolors=st.color, linewidths=0.6,
                       zorder=_ZORDER[key_], rasterized=dense and key_ == "not_different", gid=f"status-{key_}")
        for yv, text, color in labels:
            ax.text(top * 0.985, yv, text, ha="right", va="bottom",
                    fontsize=th.typography.annotation, color=color, zorder=6.0)
        _label(ax, th, f.index, exp_, obs, status, shown, highlight, dense, top)
        ax.set_xlabel("Expected events, E (square-root scale)")
        ax.set_ylabel("Observed events, O (square-root scale)")
        if title:
            ax.set_title(title, loc="left", fontsize=th.typography.title)
        key.legend([h for h, _ in entries], [lab for _, lab in entries], loc="upper left", bbox_to_anchor=(inset, 1.0),
                   ncol=ncol, frameon=False, borderaxespad=0.0, borderpad=0.0, handletextpad=0.4, handlelength=1.0,
                   fontsize=th.typography.legend)
        freeze_layout(fig, th)
    c = prof.status_counts()
    excess = obs[shown] - exp_[shown]
    alt = (f"Observed against expected events for {fmt_count(int(shown.sum()))} providers: {fmt_count(c['above'])} above "
           f"and {fmt_count(c['below'])} below the reference at the {pct(level)} level; observed minus expected ranges "
           f"from {fmt_number(excess.min(), 1)} to {fmt_number(excess.max(), 1)} events.")
    return FigureResult(fig, ax, theme=th, alt_text=alt, long_description=alt + " " + note, caption=note, provenance=prov, counts=c,
                        kind="observed_expected")


def _label(ax: Any, th: Theme, ids: pd.Index, x: np.ndarray, y: np.ndarray, status: np.ndarray, shown: np.ndarray,
           highlight: Optional[Iterable[Any]], dense: bool, top: float) -> None:
    flagged = shown & np.isin(status, ("above", "below"))
    want = np.zeros(len(ids), dtype=bool)
    if highlight is not None:
        want |= np.asarray(ids.isin(list(highlight)), dtype=bool) & shown
    if not dense and flagged.sum() <= 10:
        want |= flagged
    idx = np.flatnonzero(want)
    if idx.size == 0:
        return
    roots = np.sqrt(np.maximum(y[idx], 0.0))
    pos = spread(roots.tolist(), 0.035 * math.sqrt(top), 0.0, math.sqrt(top))
    for i, r in zip(idx, pos):
        ax.annotate(str(ids[i]), (x[i], y[i]), xytext=(x[i] * 1.04 + 0.01 * top, r ** 2), textcoords="data",
                    ha="left", va="center", fontsize=th.typography.annotation, color=th.ink, zorder=6.0)
