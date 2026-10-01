"""Null-calibration table: per null group, the fitted null and the flags under the theoretical and fitted nulls."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from .._provenance import null_text, test_text
from ..data._calibration import calibration_profiles, calibration_summary
from ..formatting import FLAG_SYMBOLS, fmt_count, fmt_number
from ._provider import TableResult
from ._spec import Column, TableSpec

__all__ = ["null_calibration_table"]


def null_calibration_table(source: Any, *args: Any, caption: Optional[str] = None, **test_kwargs: Any) -> TableResult:
    """Per null group: providers, fitted null mean and SD, and flags under the theoretical and the fitted null.

    Takes the same arguments as :func:`~pprof_py.presentation.null_calibration`; a model is tested twice.
    """
    calibrated, theoretical = calibration_profiles(source, args, test_kwargs, display="null_calibration_table")
    s = calibration_summary(calibrated, theoretical)
    prov = calibrated.provenance
    up, down = FLAG_SYMBOLS[1], FLAG_SYMBOLS[-1]

    def flags(a: Any, b: Any) -> list:
        return [f"{up} {fmt_count(int(x))} / {down} {fmt_count(int(y))}" for x, y in zip(a, b)]
    cells = {"group": ["All providers" if k == "all" else str(k) for k in s.index],
             "providers": [fmt_count(int(v)) for v in s["providers"]],
             "mean": [fmt_number(v, 2) for v in s["null_mean"]], "sd": [fmt_number(v, 2) for v in s["null_sd"]],
             "fitted": flags(s["above_fitted"], s["below_fitted"])}
    cols = [Column("group", "Null group", "id"), Column("providers", "Providers", "count"),
            Column("mean", "Null mean", "estimate", spanner="Fitted null"),
            Column("sd", "Null SD", "estimate", spanner="Fitted null")]
    if theoretical is not None:
        cells["theoretical"] = flags(s["above_theoretical"], s["below_theoretical"])
        cells["changed"] = [fmt_count(int(v)) for v in s["changed"]]
        cols += [Column("theoretical", "Theoretical null", "text", spanner="Flagged"),
                 Column("fitted", "Fitted null", "text", spanner="Flagged"),
                 Column("changed", "Changed", "count", marker="a", spanner="Flagged")]
    else:
        cols.append(Column("fitted", "Flagged", "text"))
    note = (f"{up} above / {down} below the reference. Test: {test_text(prov)}; {null_text(prov.get('null_model'))}."
            + (" Changed: providers whose flag differs between the two nulls." if theoretical is not None else
               " Flags under the theoretical null need the model."))
    spec = TableSpec(columns=tuple(cols), cells=pd.DataFrame(cells, index=s.index.astype(str)), values=s.copy(),
                     caption=caption or f"Null calibration: {fmt_count(int(s['providers'].sum()))} providers",
                     notes=((("a" if theoretical is not None else ""), note),),
                     source_note=f"Source: {prov.get('model') or 'user data'}; pprof_py {prov.get('package_version') or ''}".rstrip() + ".",
                     number_formats={"null_mean": "0.00", "null_sd": "0.00"}, provenance=dict(prov))
    return TableResult(spec)
