"""Data-quality panel: who is missing or unreliable, and why (spec §5.3 Tier 2; S6, S11)."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..data import ProviderProfile
from ..data._quality import quality_accounting, quality_groups
from ..data._resolve import resolve_profile
from ..formatting import fmt_count, fmt_number
from ..theme import Theme, get_theme
from ._common import freeze_layout, log_ticks, scaffold
from ._result import FigureResult

__all__ = ["data_quality"]

_UNITS = {"records": "records", "trials": "trials", "expected": "expected events", "patients": "patients",
          "person_time": "person-time"}


def data_quality(source: Any, *args: Any, theme: Union[str, Theme, None] = "publication",
                 size: Union[str, float] = "double", title: Optional[str] = None, **test_kwargs: Any) -> FigureResult:
    """Data-quality panel: provider accounting and provider volume by group.

    Parameters
    ----------
    source
        A fitted model (tested once here), a ``test()`` result, or a :class:`~pprof_py.presentation.ProviderProfile`.
    *args, **test_kwargs
        Passed to the model's ``test()``; not allowed with a profile or a ``test()`` result.
    theme, size, title
        As for the other figures.

    Returns
    -------
    FigureResult

    Notes
    -----
    Every provider is accounted for: providers excluded by data preparation (when the model recorded them), untested
    and suppressed providers, and providers without a finite estimate. A count that the source does not record is
    shown as "not recorded", never as zero. The volume panel needs denominators; without them only the accounting is
    drawn and the footnote says so.
    """
    th = get_theme(theme)
    prof = resolve_profile(source, args, test_kwargs, display="data_quality", limits=False)
    prov, f = prof.provenance, prof.data
    rows = quality_accounting(prof)
    kind = prov.get("denominator_kind")
    has_volume = "denominator" in prof.capabilities
    groups = quality_groups(prof)
    excluded = prof.excluded
    plot_excluded = excluded is not None and len(excluded) > 0 and kind == "records"
    note = _footnote(prof, rows, kind, has_volume, excluded is not None and len(excluded) > 0 and not plot_excluded)
    main_mm = max(len(rows) * 4.4, 5 * 7.0) + 16.0
    with th.rc_context():
        fig, axes, key, ncol, inset = scaffold(th, size, main_mm, [], note, sharey=False,
                                               width_ratios=None if not has_volume else (1.0, 1.15))
        acc = axes if not has_volume else axes[0]
        _draw_accounting(acc, th, rows)
        if has_volume:
            _draw_volume(axes[1], th, f["denominator"].to_numpy(dtype=float), groups, excluded if plot_excluded else None,
                         kind, prov.get("min_volume"))
        if title:
            acc.set_title(title, loc="left", fontsize=th.typography.title)
        freeze_layout(fig)
    counts = {key: count for _, count, key in rows}
    alt = _alt_text(prof, rows, f["denominator"].to_numpy(dtype=float) if has_volume else None, kind)
    return FigureResult(fig, acc, theme=th, alt_text=alt, long_description=alt + " " + note, provenance=prov,
                        counts=counts, kind="data_quality")


def _color(th: Theme, key: str) -> str:
    if key in ("above", "below", "not_different", "not_tested"):
        return th.status[key].color
    return {"total": th.ink, "excluded": th.muted, "analysed": th.ink}.get(key, th.volume)


def _draw_accounting(ax: Any, th: Theme, rows: List[Tuple[str, Optional[int], str]]) -> None:
    n = len(rows)
    y = np.arange(n - 1, -1, -1, dtype=float)
    top = max([c for _, c, _ in rows if c is not None] or [1])
    for (label, count, key), yi in zip(rows, y):
        if count is None:
            ax.text(0.0, yi, " not recorded", ha="left", va="center", fontsize=th.typography.annotation,
                    color=th.muted, gid=f"accounting-{key}")
            continue
        ax.barh([yi], [count], height=0.62, color=_color(th, key), linewidth=0, zorder=2.0, gid=f"accounting-{key}")
        ax.text(count + 0.012 * top, yi, fmt_count(count), ha="left", va="center", fontsize=th.typography.annotation,
                color=th.ink)
    first_attribute = [i for i, (_, _, key) in enumerate(rows) if key == "no_finite_estimate"][0]
    ax.axhline(y[first_attribute] + 0.5, color=th.volume, lw=th.lines.grid, zorder=1.0, gid="accounting-separator")
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in rows], fontsize=th.typography.tick)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.0, top * 1.22)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Providers")


def _draw_volume(ax: Any, th: Theme, volume: np.ndarray, groups: pd.Series, excluded: Optional[pd.DataFrame],
                 kind: Optional[str], min_volume: Optional[float]) -> None:
    order = ["Analysed", "No finite estimate", "Not tested", "Suppressed"]
    names = [g for g in order if (groups == g).any()]
    data = {g: volume[(groups == g).to_numpy()] for g in names}
    if excluded is not None:
        names.append("Excluded")
        data["Excluded"] = excluded["n_records"].to_numpy(dtype=float)
    rng = np.random.default_rng(20261001)                      # deterministic jitter
    dense = sum(len(v) for v in data.values()) > 2000
    for i, g in enumerate(names):
        v = data[g]
        v = v[np.isfinite(v) & (v > 0)]
        jitter = rng.uniform(-0.28, 0.28, v.size)
        ax.scatter(v, np.full(v.size, float(i)) + jitter, s=5.0 if dense else 9.0, color=th.ink if g == "Analysed" else th.muted,
                   linewidths=0, zorder=2.0, rasterized=dense, gid=f"volume-{g.lower().replace(' ', '_')}")
    vals = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in data.values()]) if data else np.array([1.0])
    a, b = float(vals.min()), float(vals.max())
    span = np.log10(b / a) if b > a else 1.0
    ax.set_xscale("log")
    ax.set_xlim(a / 10 ** (0.08 * span), b * 10 ** (0.08 * span))
    log_ticks(ax.xaxis)
    if min_volume is not None:
        ax.axvline(float(min_volume), color=th.reference, lw=th.lines.reference, linestyle=(0, (4.0, 2.0)),
                   zorder=1.0, gid="minimum-volume")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=th.typography.tick)
    ax.tick_params(axis="y", length=0, labelleft=True)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel(f"{_UNITS.get(kind, 'denominator').capitalize()} per provider (log scale)")


def _footnote(prof: ProviderProfile, rows: Any, kind: Optional[str], has_volume: bool, unit_mismatch: bool) -> str:
    prov = prof.provenance
    ex = prof.excluded
    parts = []
    if ex is None:
        parts.append("Exclusions by data preparation were not recorded for this source")
    elif len(ex):
        reasons = "; ".join(f"{fmt_count(int(n))} with {r}" for r, n in ex.groupby("reason").size().items())
        parts.append(f"Excluded by data preparation: {reasons}")
    else:
        parts.append("Data preparation excluded no provider")
    parts.append("not tested: the test returned no result; no finite estimate: no events or only events (the test "
                 "and its flag still apply); zero events and no interval are attributes of analysed providers")
    if prov.get("min_volume") is not None:
        parts.append(f"suppressed: fewer than {fmt_number(prov['min_volume'], 0)} {_UNITS.get(kind, 'units')} "
                     "(a display rule; flags are unchanged)")
    if not has_volume:
        parts.append("denominators are not available, so provider volumes are not shown")
    elif unit_mismatch:
        parts.append("excluded providers are counted but not placed on the volume axis, whose unit is not records")
    return "; ".join(parts) + "."


def _alt_text(prof: ProviderProfile, rows: Any, volume: Optional[np.ndarray], kind: Optional[str]) -> str:
    c = {key: count for _, count, key in rows}
    head = (f"Data-quality summary: {fmt_count(c['total'])} providers in the data, "
            f"{fmt_count(c['excluded'])} excluded by data preparation, " if c["total"] is not None
            else "Data-quality summary: exclusions not recorded, ")
    text = (head + f"{fmt_count(c['analysed'])} analysed ({fmt_count(c['above'])} above, {fmt_count(c['below'])} below, "
            f"{fmt_count(c['not_different'])} not different, {fmt_count(c['not_tested'])} not tested"
            + (f", {fmt_count(c['suppressed'])} suppressed" if "suppressed" in c else "") + "); "
            f"{fmt_count(c['no_finite_estimate'])} without a finite estimate")
    if volume is not None:
        v = volume[np.isfinite(volume)]
        if v.size:
            text += (f"; {_UNITS.get(kind, 'denominator')} per provider from {fmt_number(v.min(), 0)} to "
                     f"{fmt_number(v.max(), 0)} (median {fmt_number(float(np.median(v)), 0)})")
    return text + "."
