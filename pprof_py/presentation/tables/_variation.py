"""Between-provider variation table: random-effect SD, its interval and the implied range of true effects."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._provenance import pct
from ..data._variation import variation_summary
from ..formatting import fmt_count, fmt_number
from ._provider import TableResult
from ._spec import Column, TableSpec

__all__ = ["provider_variation_table"]


def provider_variation_table(model: Any, *, level: float = 0.95, caption: Optional[str] = None) -> TableResult:
    """Random-effect SD with its interval, the range of true effects it implies, and the spread of the BLUPs.

    Takes the same arguments as :func:`~pprof_py.presentation.provider_variation`.
    """
    v = variation_summary(model, level=level)
    b = v.blups.to_numpy(dtype=float)
    p = pct(level)
    rows: List[Tuple[str, float, str]] = [
        ("Random-effect SD (\u03c3)", v.sigma, "2"),
        (f"\u03c3, {p} interval: lower", np.nan if v.lower is None else v.lower, "2"),
        (f"\u03c3, {p} interval: upper", np.nan if v.upper is None else v.upper, "2"),
        (f"Range of {p} of true effects: lower", v.range_lower, "2"),
        (f"Range of {p} of true effects: upper", v.range_upper, "2"),
    ]
    if v.ratio_lower is not None:
        rows += [("As odds ratios: lower", v.ratio_lower, "2"), ("As odds ratios: upper", v.ratio_upper, "2")]
    rows += [("Providers", float(b.size), "count"), ("SD of the BLUPs (descriptive)", float(np.std(b, ddof=1)), "2")]
    text = ["not available" if not np.isfinite(val) else (fmt_count(val) if kind == "count" else fmt_number(val, 2))
            for _, val, kind in rows]
    labels = [r[0] for r in rows]
    note = (f"Effects on the {'log-odds' if v.scale == 'log_odds' else 'outcome'} scale, relative to the average "
            f"provider. The range is \u00b1{fmt_number(v.z, 2)}\u03c3 and assumes normal random effects; the BLUPs "
            "spread less than \u03c3 because they are shrunk toward the average."
            + ("" if v.lower is not None else " The model reports no interval for \u03c3."))
    spec = TableSpec(columns=(Column("quantity", "Quantity", "id"), Column("value", "Value", "estimate", marker="a")),
                     cells=pd.DataFrame({"quantity": labels, "value": text}, index=labels),
                     values=pd.DataFrame({"value": [r[1] for r in rows]}, index=pd.Index(labels, name="quantity")),
                     caption=caption or f"Between-provider variation ({v.model})", notes=(("a", note),),
                     source_note=f"Source: {v.model}; {v.interval_method or 'no'} interval for \u03c3.",
                     number_formats={"value": "0.00"}, provenance={"model": v.model, "level": v.level})
    return TableResult(spec)
