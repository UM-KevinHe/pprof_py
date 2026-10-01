"""Data-quality and exclusions table (table taxonomy item 7; S6, S11)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .._provenance import test_text
from ..data._quality import quality_accounting
from ..data._resolve import resolve_profile
from ..formatting import fmt_count, fmt_number, fmt_percent
from ._provider import TableResult
from ._spec import Column, TableSpec

__all__ = ["data_quality_table"]

_DEFINITIONS = {
    "total": "Providers in the input data",
    "excluded": "Removed by data preparation before the test",
    "analysed": "Providers in the test",
    "above": "Flagged above the reference",
    "not_different": "Not flagged",
    "below": "Flagged below the reference",
    "not_tested": "The test returned no result",
    "suppressed": "Below the minimum volume; values and flag not shown",
    "no_finite_estimate": "No events or only events; the test and its flag still apply",
    "zero_events": "No events",
    "no_interval": "The test reports no interval",
}
_SUB = ("above", "not_different", "below", "not_tested", "suppressed")


def data_quality_table(source: Any, *args: Any, details: bool = False, caption: Optional[str] = None,
                       **test_kwargs: Any) -> TableResult:
    """Provider accounting, or with ``details=True`` one row per provider with a data-quality issue.

    Parameters
    ----------
    source
        A fitted model (tested once here), a ``test()`` result, or a :class:`~pprof_py.presentation.ProviderProfile`.
    details : bool, default False
        List the affected providers (excluded, suppressed, not tested, no finite estimate, zero events, no interval)
        with their volume and issues, instead of the counts.
    caption : str, optional

    Returns
    -------
    TableResult
    """
    prof = resolve_profile(source, args, test_kwargs, display="data_quality_table", limits=False)
    prov = prof.provenance
    source_note = (f"Source: {prov.get('model') or 'user data'}; {test_text(prov)}; pprof_py "
                   f"{prov.get('package_version') or ''}".rstrip() + ".")
    if details:
        return _details(prof, caption, source_note)
    rows = quality_accounting(prof)
    counts = {key: count for _, count, key in rows}
    total, analysed = counts["total"], counts["analysed"]
    labels, count_text, share_text, share, defs, raw = [], [], [], [], [], []
    for label, count, key in rows:
        base = total if key in ("total", "excluded", "analysed") else analysed
        labels.append(("\u2003" if key in _SUB else "") + label)
        count_text.append("not recorded" if count is None else fmt_count(count))
        s = np.nan if count is None or not base else count / base
        share.append(s)
        share_text.append("\u2014" if np.isnan(s) else fmt_percent(s, 1))
        defs.append(_DEFINITIONS[key])
        raw.append(np.nan if count is None else float(count))
    cells = pd.DataFrame({"category": labels, "count": count_text, "share": share_text, "definition": defs},
                         index=[key for _, _, key in rows])
    cols = (Column("category", "Providers", "id"), Column("count", "Count", "count"),
            Column("share", "Share", "percent", marker="a"), Column("definition", "Definition", "text"))
    values = pd.DataFrame({"category": [label for label, _, _ in rows], "count": raw, "share": share},
                          index=pd.Index([key for _, _, key in rows], name="key"))
    note = ("Shares of the providers in the data for the first three rows and of the analysed providers below; "
            "zero events and no interval are attributes of analysed providers and overlap the flags.")
    if total is None:
        note = "Exclusions were not recorded for this source, so shares are of the analysed providers. " + note
    spec = TableSpec(columns=cols, cells=cells, values=values,
                     caption=caption or f"Data quality: {fmt_count(analysed)} analysed providers",
                     notes=(("a", note),), source_note=source_note,
                     number_formats={"count": "#,##0", "share": "0.0%"}, provenance=dict(prov))
    return TableResult(spec)


def _details(prof: Any, caption: Optional[str], source_note: str) -> TableResult:
    f = prof.data
    kind = prof.provenance.get("denominator_kind")
    status = f["status"].astype(str).to_numpy()
    masks = (("suppressed", status == "suppressed"), ("not tested", status == "not_tested"),
             ("no finite estimate", ~f["finite_estimate"].fillna(True).to_numpy(dtype=bool)),
             ("zero events", f["zero_events"].fillna(False).to_numpy(dtype=bool)),
             ("no interval", ~f["has_interval"].to_numpy(dtype=bool)))
    issues = ["; ".join(name for name, m in masks if m[i]) for i in range(len(f))]
    analysed = pd.DataFrame({"volume": f["denominator"].to_numpy(dtype=float), "issue": issues}, index=f.index)
    analysed = analysed[analysed["issue"] != ""]
    ex = prof.excluded
    frames = []
    if ex is not None and len(ex):
        frames.append(pd.DataFrame({"volume": ex["n_records"].to_numpy(dtype=float),
                                    "issue": [f"excluded: {r}" for r in ex["reason"]]}, index=ex.index))
    frames.append(analysed)
    rows = pd.concat(frames)
    rows.index.name = "provider_id"
    unit = {"records": "Records", "trials": "Trials", "expected": "Expected events"}.get(kind, "Volume")
    cells = pd.DataFrame({"provider": [str(i) for i in rows.index],
                          "volume": [fmt_number(v, 1) if kind == "expected" and not str(i).startswith("excluded") else
                                     fmt_count(v) for i, v in zip(rows["issue"], rows["volume"])],
                          "issue": rows["issue"].to_numpy()}, index=rows.index)
    cols = (Column("provider", "Provider", "id"), Column("volume", unit, "count", marker="a"),
            Column("issue", "Issue", "text"))
    note = (f"{unit} of analysed providers; excluded providers show their records, the unit data preparation "
            "screens on." if kind != "records" else "Records per provider.")
    spec = TableSpec(columns=cols, cells=cells, values=rows.copy(),
                     caption=caption or f"Data quality: {fmt_count(len(rows))} providers with an issue",
                     notes=(("a", note),), source_note=source_note, number_formats={"volume": "#,##0"},
                     provenance=dict(prof.provenance))
    return TableResult(spec)
