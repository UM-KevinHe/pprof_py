"""Multi-measure provider table (taxonomy item 3): each measure's estimate and flag under a grouped header."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .._provenance import null_text, test_text
from ..data._collection import ProfileCollection
from ..formatting import MISSING, fmt_count, fmt_flag, fmt_interval, resolve_digits
from ._provider import TableResult
from ._spec import Column, TableSpec

__all__ = ["multi_measure_table"]


def _strings(values: Any) -> np.ndarray:
    return np.asarray([str(v) for v in np.asarray(values, dtype=object)], dtype=object)


def multi_measure_table(source: Any, *, digits: int = 2, caption: Optional[str] = None) -> TableResult:
    """Providers (all measures' union, first-seen order) by measure: estimate with interval, and flag.

    ``\u2014`` marks a provider a measure does not include; ``NE``, ``NT`` and ``S`` follow the provider table.
    Display rounding follows the rounding-collision rule (S12) within each measure.
    """
    col = source if isinstance(source, ProfileCollection) else ProfileCollection(source)
    ids = col.providers()
    cells = {"provider": [str(i) for i in ids]}
    cols = [Column("provider", "Provider", "id")]
    values = {}
    notes = []
    any_unresolved = False
    for j, label in enumerate(col.labels):
        p = col[label]
        f = p.data.reindex(ids)
        present = f["status"].notna().to_numpy()
        est, lo, hi, nv = (f[c].to_numpy(dtype=float) for c in ("estimate", "ci_lower", "ci_upper", "null_value"))
        flag = f["flag"]
        flag_f = flag.to_numpy(dtype=float, na_value=np.nan)
        no_bounds = np.isnan(lo) | np.isnan(hi)
        row_digits, unresolved = resolve_digits(lo, hi, nv, list(np.where(no_bounds | ~present, np.nan, flag_f)),
                                                digits)
        any_unresolved |= bool(np.any(unresolved & present))
        ci = _strings(fmt_interval(est, lo, hi, row_digits))
        nofinite = ~f["finite_estimate"].fillna(True).to_numpy(dtype=bool)
        suppressed = f["status"].astype(object).to_numpy() == "suppressed"
        ci = np.where(nofinite, MISSING["no_finite_estimate"], ci)
        flags = _strings(fmt_flag(flag))
        cells[f"e{j}"] = np.where(~present, MISSING["not_applicable"], np.where(suppressed, MISSING["suppressed"], ci))
        cells[f"f{j}"] = np.where(~present, MISSING["not_applicable"], np.where(suppressed, MISSING["suppressed"], flags))
        cols += [Column(f"e{j}", "Estimate (CI)", "interval", spanner=label),
                 Column(f"f{j}", "Flag", "flag", spanner=label)]
        for c in ("estimate", "ci_lower", "ci_upper", "flag"):
            values[f"{label}: {c}"] = f[c].to_numpy(dtype=float, na_value=np.nan)
        notes.append(f"{label}: {test_text(p.provenance)}; {null_text(p.provenance.get('null_model'))}")
    text = ("; ".join(notes) + f". {MISSING['not_applicable']}: the measure does not include the provider; "
            f"{MISSING['no_finite_estimate']}: no finite estimate; {MISSING['not_tested']}: not tested; "
            f"{MISSING['suppressed']}: suppressed. Each measure has its own scale and reference; the columns are not a "
            "composite.")
    if any_unresolved:
        text += " Some flagged intervals touch the reference even at the maximum precision shown."
    spec = TableSpec(columns=tuple(cols), cells=pd.DataFrame(cells, index=ids),
                     values=pd.DataFrame(values, index=ids),
                     caption=caption or f"{len(col)} measures, {fmt_count(len(ids))} providers",
                     notes=(("", text),), source_note="Source: " + "; ".join(
                         f"{label}: {col[label].provenance.get('model') or 'user data'}" for label in col.labels) + ".",
                     number_formats={}, provenance={"measures": col.labels})
    return TableResult(spec)
