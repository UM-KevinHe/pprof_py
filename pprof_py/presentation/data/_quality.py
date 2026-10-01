"""Provider accounting for data-quality displays: every provider counted once, unrecorded counts kept as None."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ._profile import ProviderProfile

__all__ = ["quality_accounting", "quality_groups"]


def quality_accounting(profile: ProviderProfile) -> List[Tuple[str, Optional[int], str]]:
    """Rows ``(label, count, key)`` of the provider accounting; ``count`` is ``None`` when not recorded."""
    c = profile.status_counts()
    analysed = len(profile)
    excluded = c["excluded"]
    rows: List[Tuple[str, Optional[int], str]] = [
        ("In the data", None if excluded is None else analysed + excluded, "total"),
        ("Excluded by data preparation", excluded, "excluded"),
        ("Analysed", analysed, "analysed"),
        ("Above reference", c["above"], "above"),
        ("Not different", c["not_different"], "not_different"),
        ("Below reference", c["below"], "below"),
        ("Not tested", c["not_tested"], "not_tested"),
    ]
    if profile.provenance.get("min_volume") is not None:
        rows.append(("Suppressed", c["suppressed"], "suppressed"))
    rows += [("No finite estimate", c["no_finite_estimate"], "no_finite_estimate"),
             ("Zero events", c["zero_events"], "zero_events"), ("No interval", c["no_interval"], "no_interval")]
    return rows


def quality_groups(profile: ProviderProfile) -> pd.Series:
    """One mutually exclusive group per analysed provider: suppressed, not tested, no finite estimate, or analysed."""
    f = profile.data
    status = f["status"].astype(str)
    nofinite = ~f["finite_estimate"].fillna(True).astype(bool)
    group = np.where(status == "suppressed", "Suppressed",
                     np.where(status == "not_tested", "Not tested",
                              np.where(nofinite, "No finite estimate", "Analysed")))
    return pd.Series(group, index=f.index, name="group")
