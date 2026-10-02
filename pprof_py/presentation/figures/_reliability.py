"""Reliability of a measure by provider size: how well it separates providers (spec §5.3 Tier 2)."""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from ..data import CapabilityError
from ..formatting import fmt_count, fmt_number
from ..theme import Theme, get_theme
from ._common import freeze_layout, log_ticks, scaffold
from ._result import FigureResult

__all__ = ["reliability"]


def reliability(iur: Any, *, theme: Union[str, Theme, None] = "publication", size: Union[str, float] = "single",
                title: Optional[str] = None) -> FigureResult:
    """Reliability at each provider's size, the curve it traces, and the overall inter-unit reliability.

    Parameters
    ----------
    iur
        A fitted :class:`~pprof_py.measures.iur.BootstrapIUR` (it stores each provider's reliability,
        ``iur_groups_``, at its size, ``group_sizes_``).
    theme, size, title
        As for the other figures.

    Returns
    -------
    FigureResult

    Raises
    ------
    CapabilityError
        For reliability objects that do not store per-provider reliability (``DirectIUR``, ``SplitHalfIUR``); use
        :func:`~pprof_py.presentation.reliability_table` for those.

    Notes
    -----
    Reliability is a property of the measure at a given volume: the share of the between-provider spread that is
    signal for providers of that size. It is not a score for any provider.
    """
    for attr in ("iur_groups_", "group_sizes_", "iur_", "n_prime_"):
        if getattr(iur, attr, None) is None:
            raise CapabilityError(f"reliability() needs {attr}, which {type(iur).__name__} does not provide; per-provider "
                                  "reliability is stored by a fitted BootstrapIUR. Use reliability_table() for the "
                                  "overall value and the decile table.")
    th = get_theme(theme)
    sizes = np.asarray(iur.group_sizes_, dtype=float)
    rel = np.asarray(iur.iur_groups_, dtype=float)
    order = np.argsort(sizes, kind="stable")
    n = sizes.size
    overall, n_prime = float(iur.iur_), float(iur.n_prime_)
    boot = getattr(iur, "n_boot", None)
    from matplotlib.lines import Line2D

    entries = [(Line2D([], [], linestyle="none", marker="o", markersize=3.0, markerfacecolor=th.ink,
                       markeredgecolor=th.ink), "Provider"),
               (Line2D([], [], color=th.limit, lw=th.lines.data), "Reliability by size"),
               (Line2D([], [], color=th.reference, lw=th.lines.reference, linestyle=(0, (4.0, 2.0))), "Overall IUR")]
    note = (f"{fmt_count(n)} providers. Reliability at each provider's size n, s\u00b2 between / (s\u00b2 between + "
            f"s\u00b2 within / n), from {type(iur).__name__}" + (f" with {fmt_count(boot)} bootstrap resamples" if boot else "")
            + f"; the line joins the providers in size order. The overall IUR, {fmt_number(overall, 2)}, is the "
            f"reliability at the effective size n\u2032 = {fmt_number(n_prime, 0)}. Reliability is a property of the "
            "measure at a given volume (the share of the spread between providers of that size that is signal), not a "
            "score for any provider.")
    with th.rc_context():
        fig, ax, key, ncol, inset = scaffold(th, size, 62.0, [lab for _, lab in entries], note)
        ax.plot(sizes[order], rel[order], color=th.limit, lw=th.lines.data, zorder=2.0, gid="reliability-curve")
        ax.scatter(sizes, rel, s=9.0, color=th.ink, linewidths=0, zorder=3.0, rasterized=n > 2000,
                   gid="reliability-providers")
        ax.axhline(overall, color=th.reference, lw=th.lines.reference, linestyle=(0, (4.0, 2.0)), zorder=1.5,
                   gid="overall-iur")
        ax.axvline(n_prime, color=th.volume, lw=th.lines.grid, linestyle=(0, (1.0, 1.5)), zorder=1.0, gid="n-prime")
        ax.set_xscale("log")
        span = np.log10(sizes.max() / sizes.min()) if sizes.max() > sizes.min() else 1.0
        ax.set_xlim(sizes.min() / 10 ** (0.05 * span), sizes.max() * 10 ** (0.05 * span))
        log_ticks(ax.xaxis)
        ax.set_ylim(0.0, 1.0)
        ax.text(ax.get_xlim()[0] * 10 ** (0.02 * span), overall + 0.02, f"overall IUR {fmt_number(overall, 2)}",
                ha="left", va="bottom", fontsize=th.typography.annotation, color=th.reference)
        ax.text(n_prime * 10 ** (0.01 * span), 0.03, f"n\u2032 = {fmt_number(n_prime, 0)}", ha="left", va="bottom",
                fontsize=th.typography.annotation, color=th.muted)
        ax.set_xlabel("Provider size (log scale)")
        ax.set_ylabel("Reliability")
        if title:
            ax.set_title(title, loc="left", fontsize=th.typography.title)
        key.legend([h for h, _ in entries], [lab for _, lab in entries], loc="upper left", bbox_to_anchor=(inset, 1.0),
                   ncol=ncol, frameon=False, borderaxespad=0.0, borderpad=0.0, handletextpad=0.4, handlelength=1.6,
                   fontsize=th.typography.legend)
        freeze_layout(fig, th)
    alt = (f"Reliability by provider size for {fmt_count(n)} providers: from {fmt_number(rel.min(), 2)} at size "
           f"{fmt_number(sizes[order][0], 0)} to {fmt_number(rel.max(), 2)} at size {fmt_number(sizes[order][-1], 0)}; "
           f"overall IUR {fmt_number(overall, 2)} at the effective size {fmt_number(n_prime, 0)}.")
    prov = {"source": type(iur).__name__, "n_boot": boot, "iur": overall, "n_prime": n_prime}
    return FigureResult(fig, ax, theme=th, alt_text=alt, long_description=alt + " " + note, caption=note, provenance=prov,
                        counts={"providers": n}, kind="reliability")
