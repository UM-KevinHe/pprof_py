"""Flag stability: how fragile the flags are across test settings (spec §5.3 Tier 2)."""
from __future__ import annotations

import math
import textwrap
from typing import Any, Mapping, Optional, Union

import numpy as np

from ..data._stability import changing, flag_scenarios, stability_summary
from ..formatting import FLAG_SYMBOLS, fmt_count
from ..theme import Theme, get_theme
from ._common import freeze_layout, scaffold
from ._result import FigureResult

__all__ = ["flag_stability"]

_STATUS = ((1, "above"), (-1, "below"), (0, "not_different"))


def flag_stability(model: Any, *args: Any, scenarios: Optional[Mapping[str, Mapping[str, Any]]] = None,
                   theme: Union[str, Theme, None] = "publication", size: Union[str, float, None] = None,
                   title: Optional[str] = None, **test_kwargs: Any) -> FigureResult:
    """The flags of each provider flagged in at least one scenario, scenario by scenario.

    Parameters
    ----------
    model
        A fitted model; it is tested once per scenario.
    *args, **test_kwargs
        The base scenario: passed to every ``test()`` call (CoxPH needs its data here).
    scenarios : mapping of label to test settings, optional
        Settings that replace the base ones, one test each. Default: an alternative reference (where the test takes
        one), the other null (theoretical or empirical), and for three-stage models the flags at the bounds of
        sigma's interval (``sigma_sensitivity()``).
    theme, size, title
        As for the other figures (``size`` defaults to single width up to four scenarios).

    Returns
    -------
    FigureResult

    Notes
    -----
    Rows are ordered by the base estimate for legibility, not as a ranking. Agreement across a few scenarios does not
    make a flag robust: risk adjustment, data preparation and the model itself are not varied.
    """
    th = get_theme(theme)
    sc = flag_scenarios(model, args, test_kwargs, scenarios)
    f = sc.flags
    labels = list(f.columns)
    k, n = len(labels), len(f)
    flagged = ((f == 1) | (f == -1)).fillna(False).to_numpy(dtype=bool).any(axis=1)
    rows = f.index[flagged]
    order = np.argsort(-sc.estimate.reindex(rows).to_numpy(), kind="stable")
    rows = rows[order]
    m = len(rows)
    change_all = changing(sc)
    change = change_all[flagged][order]
    dense = m > 60
    size = size or ("single" if k <= 4 else "double")
    row_mm = max(3.6, th.typography.tick * 25.4 / 72.0 * 1.6) if not dense else min(1.2, 110.0 / max(m, 1))
    summary = stability_summary(sc)
    up, down = FLAG_SYMBOLS[1], FLAG_SYMBOLS[-1]
    heads = ["\n".join(textwrap.wrap(lab, 13)) + f"\n{up} {summary.loc[lab, 'above']}  {down} {summary.loc[lab, 'below']}"
             for lab in labels]
    from matplotlib.lines import Line2D

    entries = []
    for key in ("above", "below", "not_different", "not_tested"):
        st = th.status[key]
        entries.append((Line2D([], [], linestyle="none", marker=st.marker, markersize=math.sqrt(st.size),
                               markerfacecolor=st.color if st.filled else "none", markeredgecolor=st.color,
                               markeredgewidth=0.6), st.label))
    entries.append((Line2D([], [], linestyle="none", marker="D", markersize=3.0, color=th.ink),
                    "Status differs between scenarios"))
    descs = "; ".join(f"{lab}: {sc.descriptions[lab]}" for lab in labels)
    c_all = int(change_all.sum())
    note = (f"{fmt_count(m)} of {fmt_count(n)} providers flagged in at least one scenario (rows, ordered by the base "
            f"estimate for legibility, not a ranking); {fmt_count(n - m)} never flagged; {fmt_count(c_all)} change "
            f"status between scenarios. Scenarios, each a separate test of {sc.model}: {descs}. Agreement across these "
            f"{k} scenarios does not make a flag robust: risk adjustment, data preparation and the model itself are "
            "not varied." + "".join(" " + n for n in sc.notes))
    with th.rc_context():
        fig, ax, key, ncol, inset = scaffold(th, size, max(m, 1) * row_mm + 16.0, [lab for _, lab in entries], note)
        y = np.arange(m - 1, -1, -1, dtype=float)
        vals = f.loc[rows].to_numpy(dtype=float, na_value=np.nan)
        for value, key_ in _STATUS:
            st = th.status[key_]
            ii, jj = np.nonzero(vals == value)
            if ii.size:
                ax.scatter(jj.astype(float), y[ii], marker=st.marker, s=st.size * (0.5 if dense else 1.0),
                           facecolors=st.color if st.filled else "none", edgecolors=st.color, linewidths=0.6,
                           zorder=3.0, rasterized=dense, gid=f"stability-{key_}")
        ii, jj = np.nonzero(np.isnan(vals))
        if ii.size:
            st = th.status["not_tested"]
            ax.scatter(jj.astype(float), y[ii], marker=st.marker, s=st.size, facecolors="none", edgecolors=st.color,
                       linewidths=0.6, zorder=3.0, rasterized=dense, gid="stability-not_tested")
        if change.any():
            ax.scatter(np.full(int(change.sum()), k - 0.25), y[change], marker="D", s=6.0 if dense else 9.0,
                       color=th.ink, linewidths=0, zorder=3.0, rasterized=dense, gid="stability-changes")
        ax.set_xlim(-0.6, k + 0.1)
        ax.set_ylim(-0.6, max(m, 1) - 0.4)
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(k))
        ax.set_xticklabels(heads, fontsize=th.typography.tick, linespacing=1.15)
        ax.tick_params(axis="x", length=0)
        if dense:
            ax.set_yticks([])
            ax.set_ylabel(f"{fmt_count(m)} providers flagged in a scenario, ordered by the base estimate")
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([str(r) for r in rows], fontsize=th.typography.tick)
            ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        if title:
            ax.set_title(title, loc="left", fontsize=th.typography.title, pad=24.0)
        key.legend([h for h, _ in entries], [lab for _, lab in entries], loc="upper left", bbox_to_anchor=(inset, 1.0),
                   ncol=ncol, frameon=False, borderaxespad=0.0, borderpad=0.0, handletextpad=0.4, handlelength=1.0,
                   fontsize=th.typography.legend)
        freeze_layout(fig)
    stable_flagged = int(m - change.sum())
    alt = (f"Flag stability across {k} scenarios for {fmt_count(n)} providers: {fmt_count(m)} flagged in at least one "
           f"scenario, {fmt_count(stable_flagged)} of them identically in every scenario; {fmt_count(c_all)} providers "
           "change status.")
    return FigureResult(fig, ax, theme=th, alt_text=alt, long_description=alt + " " + note,
                        provenance={"model": sc.model, "scenarios": dict(sc.descriptions)},
                        counts={"providers": n, "flagged_any": m, "changing": c_all, "scenarios": k},
                        kind="flag_stability")
