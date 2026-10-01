"""Covariate-effects table: estimates with intervals and p-values from ``summary()`` (table taxonomy item 2)."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd

from .._provenance import pct
from ..data._coefficients import COEFFICIENT_LABELS, coefficient_profile
from ..formatting import fmt_count, fmt_interval, fmt_p
from ._provider import TableResult
from ._spec import Column, TableSpec

__all__ = ["coefficient_table"]


def coefficient_table(source: Any, *, exponentiate: Union[str, bool] = "auto", level: float = 0.95,
                      include_intercept: bool = False, terms: Optional[Iterable[str]] = None, digits: int = 2,
                      caption: Optional[str] = None) -> TableResult:
    """Covariate effects with intervals and p-values, from a model's ``summary()`` or a ``CoefficientProfile``.

    Parameters are as for :func:`~pprof_py.presentation.forest`; ``digits`` sets the decimals of estimates and
    intervals. The footnote states the source, the interval level and that the associations are not causal.
    """
    prof = coefficient_profile(source, exponentiate=exponentiate, level=level, include_intercept=include_intercept,
                               terms=terms)
    f, prov = prof.data, prof.provenance
    label = COEFFICIENT_LABELS.get(prov.get("scale"), "Estimate")
    lev = prov.get("level") or level
    cells = pd.DataFrame({"term": [str(t) for t in f.index],
                          "estimate_ci": np.asarray(fmt_interval(f["estimate"], f["ci_lower"], f["ci_upper"], digits),
                                                    dtype=object),
                          "p_value": np.asarray(fmt_p(f["p_value"].to_numpy()), dtype=object)}, index=f.index)
    cols = (Column("term", "Covariate", "id"), Column("estimate_ci", f"{label} ({pct(lev)} CI)", "interval", marker="a"),
            Column("p_value", "p-value", "p_value"))
    note = (f"{pct(lev)} intervals from {prov.get('model') or 'the data'}{'.summary()' if prov.get('model') else ''}. "
            "Each estimate is adjusted for the other covariates and the provider effects; associations are not causal, "
            "and covariates have their own units.")
    if prov.get("exponentiated"):
        note += f" {label}s are the exponentiated coefficients and bounds; p-values are those of the coefficients."
    fmt = "0." + "0" * digits
    spec = TableSpec(columns=cols, cells=cells, values=f.copy(),
                     caption=caption or f"Covariate effects: {label.lower()}s, {fmt_count(len(f))} terms",
                     notes=(("a", note),),
                     source_note=f"Source: {prov.get('model') or 'user data'}; pprof_py {prov.get('package_version') or ''}".rstrip() + ".",
                     number_formats={"estimate": fmt, "se": "0.000", "ci_lower": fmt, "ci_upper": fmt, "p_value": "0.000"},
                     provenance=dict(prov))
    return TableResult(spec)
