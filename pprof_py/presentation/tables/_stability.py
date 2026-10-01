"""Flag-sensitivity table (taxonomy item 8): flags per scenario, and the providers whose flag changes."""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ..data._stability import changing, flag_scenarios, stability_summary
from ..formatting import FLAG_SYMBOLS, MISSING, fmt_count
from ._provider import TableResult
from ._spec import Column, TableSpec

__all__ = ["flag_stability_table"]


def flag_stability_table(model: Any, *args: Any, scenarios: Optional[Mapping[str, Mapping[str, Any]]] = None,
                         details: bool = False, caption: Optional[str] = None, **test_kwargs: Any) -> TableResult:
    """Counts per scenario (above, below, not different, not tested, changed from the base), or with
    ``details=True`` the flag of each provider whose status changes, in source order.

    Takes the same arguments as :func:`~pprof_py.presentation.flag_stability`.
    """
    sc = flag_scenarios(model, args, test_kwargs, scenarios)
    labels = list(sc.flags.columns)
    caution = ("Each scenario is a separate test; agreement across them does not make a flag robust (risk adjustment, "
               "data preparation and the model are not varied).")
    caution += "".join(" " + n for n in sc.notes)
    if details:
        sel = changing(sc)
        f = sc.flags.loc[sel]
        sym = {1: FLAG_SYMBOLS[1], -1: FLAG_SYMBOLS[-1], 0: FLAG_SYMBOLS[0]}
        cells = pd.DataFrame({"provider": [str(i) for i in f.index]}, index=f.index)
        for j, lab in enumerate(labels):
            cells[f"s{j}"] = [MISSING["not_tested"] if v is pd.NA else sym[int(v)] for v in f[lab]]
        cols = (Column("provider", "Provider", "id"),) + tuple(
            Column(f"s{j}", lab, "flag", spanner="Flag by scenario") for j, lab in enumerate(labels))
        values = f.astype("float")
        caption = caption or f"Flag sensitivity: {fmt_count(len(f))} providers whose flag changes"
        notes = (("", f"{FLAG_SYMBOLS[1]} above / {FLAG_SYMBOLS[-1]} below the reference, {FLAG_SYMBOLS[0]} not "
                      f"different, {MISSING['not_tested']} not tested. " + caution),)
    else:
        s = stability_summary(sc)
        cells = pd.DataFrame({"scenario": labels, "above": [fmt_count(v) for v in s["above"]],
                              "below": [fmt_count(v) for v in s["below"]],
                              "not_different": [fmt_count(v) for v in s["not_different"]],
                              "not_tested": [fmt_count(v) for v in s["not_tested"]],
                              "changed": [MISSING["not_applicable"] if lab == labels[0] else fmt_count(v)
                                          for lab, v in zip(labels, s["changed"])],
                              "settings": [sc.descriptions[lab] for lab in labels]}, index=labels)
        cols = (Column("scenario", "Scenario", "id"), Column("above", "Above", "count", spanner="Providers"),
                Column("below", "Below", "count", spanner="Providers"),
                Column("not_different", "Not different", "count", spanner="Providers"),
                Column("not_tested", "Not tested", "count", spanner="Providers"),
                Column("changed", "Changed from base", "count", marker="a"), Column("settings", "Test", "text"))
        values = s.astype(float)
        values.loc[labels[0], "changed"] = np.nan
        caption = caption or f"Flag sensitivity: {fmt_count(len(sc.flags))} providers, {len(labels)} scenarios"
        notes = (("a", "Providers whose flag (or tested status) differs from the base scenario. " + caution),)
    spec = TableSpec(columns=cols, cells=cells, values=values, caption=caption, notes=notes,
                     source_note=f"Source: {sc.model}, one test() per scenario.", number_formats={},
                     provenance={"model": sc.model, "scenarios": dict(sc.descriptions)})
    return TableResult(spec)
