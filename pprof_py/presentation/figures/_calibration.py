"""Null-calibration diagnostics: raw z-statistics against the theoretical and the fitted null, by null group."""
from __future__ import annotations

import math
from typing import Any, Optional, Union

import numpy as np
from scipy.stats import norm

from .._provenance import null_text, test_text
from ..data._calibration import calibration_profiles, calibration_summary
from ..formatting import fmt_count, fmt_number
from ..theme import Theme, get_theme
from ._common import freeze_layout, scaffold
from ._result import FigureResult

__all__ = ["null_calibration"]

_CONVERTED = ("poibin_exact", "exact", "midp", "bootstrap_exact", "resampling")


def null_calibration(source: Any, *args: Any, theme: Union[str, Theme, None] = "publication",
                     size: Union[str, float] = "double", title: Optional[str] = None, **test_kwargs: Any) -> FigureResult:
    """Raw z-statistics against the theoretical and the fitted null, one panel per null group.

    Parameters
    ----------
    source
        A fitted model, tested here twice: with ``test_kwargs`` (for example ``null_model=EmpiricalNull.fitter()``)
        and with the theoretical null, so the flags under both can be compared. A profile or a ``test()`` result is
        drawn without that comparison.
    *args, **test_kwargs
        Passed to the model's ``test()``.
    theme, size, title
        As for the other figures.

    Returns
    -------
    FigureResult

    Notes
    -----
    Flags are those of the test layer; this display only counts them. The densities are normal curves with the
    test's ``null_mean`` and ``null_sd``, scaled to the counts. A null SD above 1 indicates overdispersion or
    unmodelled variation, not necessarily many outlying providers.
    """
    th = get_theme(theme)
    calibrated, theoretical = calibration_profiles(source, args, test_kwargs, display="null_calibration")
    summary = calibration_summary(calibrated, theoretical)
    prov = calibrated.provenance
    f = calibrated.data
    tested = f["flag"].notna().to_numpy()
    z = f["z_raw"].to_numpy(dtype=float)
    keys = f["null_group"].astype(object).where(f["null_group"].notna(), "all").to_numpy()
    groups = list(summary.index)
    g = len(groups)
    ncol = min(g, 3)
    nrow = math.ceil(g / ncol)
    finite = z[tested & np.isfinite(z)]
    med = float(np.median(finite))                            # a few extreme statistics must not flatten the bulk:
    bound = max(6.0, abs(med) + 6.0 * 1.4826 * float(np.median(np.abs(finite - med))))   # median + 6 robust SDs
    lo = max(float(np.floor(finite.min() * 2) / 2), -np.ceil(bound))
    hi = min(float(np.ceil(finite.max() * 2) / 2), np.ceil(bound))
    beyond = int(((finite < lo) | (finite > hi)).sum())
    edges = np.arange(lo, hi + 0.5, 0.5) if (hi - lo) / 0.5 <= 60 else np.linspace(lo, hi, 61)
    width = float(edges[1] - edges[0])
    grid = np.linspace(lo, hi, 400)
    fitted_differs = bool(((summary["null_mean"] != 0.0) | (summary["null_sd"] != 1.0)).any())
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    entries = [(Patch(facecolor=th.volume, edgecolor="none"), "Raw z-statistics"),
               (Line2D([], [], color=th.reference, lw=th.lines.reference, linestyle=(0, (4.0, 2.0))),
                "Theoretical null N(0, 1)")]
    if fitted_differs:
        entries.append((Line2D([], [], color=th.limit, lw=th.lines.reference), "Fitted null"))
    note = _footnote(prov, summary, theoretical is not None and fitted_differs, len(f), fitted_differs)
    if beyond:
        note += (f" {fmt_count(beyond)} provider{'s' if beyond != 1 else ''} with a raw z beyond the axis "
                 f"(\u00b1{fmt_number(hi, 0)}) {'are' if beyond != 1 else 'is'} counted at its edge.")
    with th.rc_context():
        fig, axes, key, ncol_legend, inset = scaffold(th, size, nrow * 42.0 + 10.0, [lab for _, lab in entries], note,
                                                      grid=(nrow, ncol))
        for i, ax in enumerate(axes):
            if i >= g:
                ax.axis("off")
                continue
            key_ = groups[i]
            row = summary.loc[key_]
            zg = z[tested & (keys == key_) & np.isfinite(z)]
            counts, _ = np.histogram(zg, bins=edges)
            ax.stairs(counts, edges, fill=True, color=th.volume, linewidth=0, zorder=1.0,
                      gid=f"calibration-histogram-{key_}")              # one artist, one SVG id
            scale = zg.size * width
            ax.plot(grid, scale * norm.pdf(grid), color=th.reference, lw=th.lines.reference,
                    linestyle=(0, (4.0, 2.0)), zorder=2.0, gid=f"calibration-theoretical-{key_}")
            if fitted_differs and np.isfinite(row["null_mean"]) and np.isfinite(row["null_sd"]):
                ax.plot(grid, scale * norm.pdf(grid, row["null_mean"], row["null_sd"]), color=th.limit,
                        lw=th.lines.reference, zorder=3.0, gid=f"calibration-fitted-{key_}")
            label = "All providers" if key_ == "all" else f"Group {key_}"
            ax.set_title(f"{label} ({fmt_count(row['providers'])})", loc="left", fontsize=th.typography.label)
            for side, n_out, xpos, ha in (("left", int((zg < lo).sum()), 0.0, "left"),
                                          ("right", int((zg > hi).sum()), 1.0, "right")):
                if n_out:
                    arrow = "\u2190" if side == "left" else "\u2192"
                    text_ = f"{arrow} {fmt_count(n_out)}" if side == "left" else f"{fmt_count(n_out)} {arrow}"
                    ax.text(xpos, 0.03, text_, transform=ax.transAxes, ha=ha, va="bottom",
                            fontsize=th.typography.annotation, color=th.muted, gid=f"calibration-beyond-{side}-{key_}")
            fitted = int(row["above_fitted"] + row["below_fitted"])
            text = (f"mean {fmt_number(row['null_mean'], 2)}, SD {fmt_number(row['null_sd'], 2)}" if fitted_differs
                    else "theoretical null")
            if theoretical is not None and fitted_differs:
                theo = int(row["above_theoretical"] + row["below_theoretical"])
                text += f"\nflagged {fmt_count(theo)} \u2192 {fmt_count(fitted)}"
            else:
                text += f"\nflagged {fmt_count(fitted)}"
            ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=th.typography.annotation,
                    color=th.ink, linespacing=1.3, gid=f"calibration-text-{key_}")
            peak = max(float(counts.max()) if counts.size else 0.0, scale * norm.pdf(0.0),
                       scale * norm.pdf(0.0, 0.0, row["null_sd"]) if np.isfinite(row["null_sd"]) else 0.0)
            ax.set_ylim(0.0, 1.32 * peak if peak > 0 else 1.0)    # headroom for the text above the curves
            ax.set_xlim(lo, hi)
            if i // ncol == nrow - 1 or i + ncol >= g:
                ax.set_xlabel("Raw z-statistic")
            if i % ncol == 0:
                ax.set_ylabel("Providers")
        if title:
            fig.suptitle(title, x=inset, ha="left", fontsize=th.typography.title)
        key.legend([h for h, _ in entries], [lab for _, lab in entries], loc="upper left", bbox_to_anchor=(inset, 1.0),
                   ncol=ncol_legend, frameon=False, borderaxespad=0.0, borderpad=0.0, handletextpad=0.4,
                   handlelength=1.6, fontsize=th.typography.legend)
        freeze_layout(fig, th)
    alt = _alt_text(summary, theoretical is not None and fitted_differs, len(f), fitted_differs)
    counts_out = {"groups": g, "flagged_fitted": int(summary["above_fitted"].sum() + summary["below_fitted"].sum())}
    if theoretical is not None:
        counts_out.update({"flagged_theoretical": int(summary["above_theoretical"].sum()
                                                      + summary["below_theoretical"].sum()),
                           "changed": int(summary["changed"].sum())})
    return FigureResult(fig, axes[0], theme=th, alt_text=alt, long_description=alt + " " + note, caption=note, provenance=prov,
                        counts=counts_out, kind="null_calibration")


def _footnote(prov: Any, summary: Any, compared: bool, n: int, fitted: bool) -> str:
    g = len(summary)
    text = (f"{fmt_count(n)} providers in {fmt_count(g)} null group{'s' if g != 1 else ''}. Bars: raw z-statistics of "
            f"the {test_text(prov)}; dashed: the theoretical N(0, 1) density"
            + (f"; solid: the normal density with the fitted null's mean and SD ({null_text(prov.get('null_model'))})"
               if fitted else "") + ", scaled to the counts.")
    if not fitted:
        text += (" The test used the theoretical null, so there is nothing to compare; pass null_model= (for example "
                 "EmpiricalNull.fitter()) to see how calibration changes the flags.")
    elif compared:
        theo = int(summary["above_theoretical"].sum() + summary["below_theoretical"].sum())
        fitted = int(summary["above_fitted"].sum() + summary["below_fitted"].sum())
        text += (f" Flagged under the theoretical null: {fmt_count(theo)}; under the fitted null: {fmt_count(fitted)}; "
                 f"{fmt_count(int(summary['changed'].sum()))} flags change.")
    else:
        text += " Flags under the theoretical null are not shown: a profile cannot be tested again (pass the model)."
    text += " A null SD above 1 indicates overdispersion or unmodelled variation, not necessarily many outlying providers."
    if prov.get("test_method") in _CONVERTED:
        text += (" For exact and Monte Carlo tests the raw z is a converted statistic whose sign is arbitrary near a "
                 "provider's expected count.")
    return text


def _alt_text(summary: Any, compared: bool, n: int, fitted_null: bool) -> str:
    g = len(summary)
    means, sds = summary["null_mean"].dropna(), summary["null_sd"].dropna()
    text = (f"Null-calibration diagnostic for {fmt_count(n)} providers in {fmt_count(g)} group{'s' if g != 1 else ''}"
            + (f"; fitted null means {fmt_number(means.min(), 2)} to {fmt_number(means.max(), 2)} and SDs "
               f"{fmt_number(sds.min(), 2)} to {fmt_number(sds.max(), 2)}" if len(means) and fitted_null
               else "; theoretical null N(0, 1)"))
    fitted = int(summary["above_fitted"].sum() + summary["below_fitted"].sum())
    if compared:
        theo = int(summary["above_theoretical"].sum() + summary["below_theoretical"].sum())
        text += (f"; flagged under the theoretical null: {fmt_count(theo)}, under the fitted null: {fmt_count(fitted)}; "
                 f"{fmt_count(int(summary['changed'].sum()))} flags change")
    else:
        text += f"; {fmt_count(fitted)} flagged"
    return text + "."
