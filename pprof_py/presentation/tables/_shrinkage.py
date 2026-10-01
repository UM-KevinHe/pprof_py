"""Shrinkage table: each provider's fixed-effect and random-effect estimate and the change pooling makes."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..data._shrinkage import reference_text, shrinkage_pairs
from ..formatting import MISSING, fmt_count, fmt_number
from ._provider import TableResult
from ._spec import Column, TableSpec

__all__ = ["shrinkage_table"]


def shrinkage_table(fixed: Any, random: Any, *, fe_reference: Any = "mean", digits: int = 2,
                    caption: Optional[str] = None) -> TableResult:
    """Per provider (in the fixed-effect fit's order): volume, both estimates and the change (random minus fixed).

    Takes the same sources as :func:`~pprof_py.presentation.shrinkage`; providers without a finite fixed-effect
    estimate show ``NE`` for it and for the change.
    """
    pairs = shrinkage_pairs(fixed, random, fe_reference=fe_reference)
    f, prov = pairs.frame.copy(), pairs.provenance
    fin = f["fixed_finite"].to_numpy(dtype=bool) & np.isfinite(f["fixed"].to_numpy(dtype=float))
    f["change"] = np.where(fin, f["random"] - f["fixed"], np.nan)
    ne = MISSING["no_finite_estimate"]
    cells = pd.DataFrame({
        "provider": [str(i) for i in f.index],
        "volume": [fmt_count(v) if np.isfinite(v) else MISSING["not_applicable"] for v in f["volume"]],
        "fixed": [fmt_number(v, digits) if ok else ne for v, ok in zip(f["fixed"], fin)],
        "random": [fmt_number(v, digits) for v in f["random"]],
        "change": [fmt_number(v, digits) if ok else ne for v, ok in zip(f["change"], fin)],
    }, index=f.index)
    unit = {"records": "Records", "trials": "Trials", "patients": "Patients"}.get(prov.get("denominator_kind"), "Volume")
    cols = (Column("provider", "Provider", "id"), Column("volume", unit, "count"),
            Column("fixed", "Fixed effect", "estimate", spanner="Estimate", marker="a"),
            Column("random", "Random effect", "estimate", spanner="Estimate", marker="b"),
            Column("change", "Change", "estimate", marker="c"))
    notes = (("a", f"Unshrunken, {prov.get('fixed_model')}, relative to the {reference_text(pairs)}."),
             ("b", f"Shrunken BLUP, {prov.get('random_model')}, relative to the model's intercept; it depends on the "
                   "assumed normal random-effect distribution and is not the true effect."),
             ("c", f"Random minus fixed, computed for display; {ne}: no finite fixed-effect estimate."))
    fmt = "0." + "0" * digits
    spec = TableSpec(columns=cols, cells=cells, values=f.drop(columns=["fixed_finite"]),
                     caption=caption or f"Shrinkage: {fmt_count(len(f))} providers",
                     notes=notes, source_note=f"Source: {prov.get('fixed_model')} and {prov.get('random_model')}.",
                     number_formats={"fixed": fmt, "random": fmt, "change": fmt, "volume": "#,##0"},
                     provenance=dict(prov))
    return TableResult(spec)
