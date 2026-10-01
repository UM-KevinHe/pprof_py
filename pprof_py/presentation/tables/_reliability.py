"""Reliability table: overall IUR, its decomposition and reliability by size decile (table taxonomy item 6)."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pandas as pd

from ..data import CapabilityError
from ..formatting import fmt_count, fmt_number
from ._provider import TableResult
from ._spec import Column, TableSpec

__all__ = ["reliability_table"]

_SPLIT = {"iur_kappa": "Kappa (split-half agreement of flags)", "iur_kendall_cat": "Kendall tau (categories)",
          "iur_spearman_cat": "Spearman rho (categories)", "iur_pearson": "Pearson r", "iur_kendall": "Kendall tau",
          "iur_spearman": "Spearman rho"}


def reliability_table(iur: Any, *, n_quantiles: int = 10, caption: Optional[str] = None) -> TableResult:
    """Overall reliability, its variance decomposition, and reliability at representative sizes.

    Parameters
    ----------
    iur
        A fitted ``BootstrapIUR`` or ``DirectIUR`` (overall IUR, effective size, variance components and
        ``decile_table()``), or a fitted ``SplitHalfIUR`` (its ``summary()``).
    n_quantiles : int, default 10
        Size groups of the decile table.
    """
    name = type(iur).__name__
    rows: List[Tuple[str, float, str]] = []
    if hasattr(iur, "decile_table") and getattr(iur, "iur_", None) is not None:
        rows += [("Overall IUR", float(iur.iur_), "ratio"), ("Effective size n\u2032", float(iur.n_prime_), "size"),
                 ("Between-provider variance", float(iur.s2_between_), "variance"),
                 ("Within-provider variance", float(iur.s2_within_), "variance"),
                 ("Providers", float(iur.n_groups_), "count")]
        table = iur.decile_table(n_quantiles=n_quantiles)
        for col in table.columns:
            label = {"min": "Smallest provider", "max": "Largest provider"}.get(col, col.capitalize())
            rows.append((f"Reliability: {label}", float(table[col].iloc[0]), "ratio"))
        note = ("Reliability at a size n is s\u00b2 between / (s\u00b2 between + s\u00b2 within / n); deciles are groups of "
                "providers by size, from the smallest to the largest. Reliability is a property of the measure at a "
                "given volume, not a score for any provider.")
    elif hasattr(iur, "summary") and getattr(iur, "iur_kappa_", None) is not None:
        s = iur.summary()
        for col, label in _SPLIT.items():
            rows.append((label, float(s[col].iloc[0]), "ratio"))
        rows.append(("Providers split", float(s["n_groups"].iloc[0]), "count"))
        note = ("Split-half reliability: agreement between the measure computed on two random halves of each "
                "provider's records, averaged over repeated splits.")
    else:
        raise CapabilityError(f"reliability_table() needs a fitted BootstrapIUR, DirectIUR or SplitHalfIUR, not {name}")
    text = [fmt_count(v) if kind == "count" else fmt_number(v, 0 if kind == "size" else (4 if kind == "variance" else 2))
            for _, v, kind in rows]
    cells = pd.DataFrame({"quantity": [r[0] for r in rows], "value": text}, index=[r[0] for r in rows])
    values = pd.DataFrame({"value": [r[1] for r in rows]}, index=pd.Index([r[0] for r in rows], name="quantity"))
    spec = TableSpec(columns=(Column("quantity", "Quantity", "id"), Column("value", "Value", "estimate", marker="a")),
                     cells=cells, values=values, caption=caption or f"Reliability of the measure ({name})",
                     notes=(("a", note),), source_note=f"Source: {name}.",
                     number_formats={"value": "0.0000"}, provenance={"source": name})
    return TableResult(spec)
