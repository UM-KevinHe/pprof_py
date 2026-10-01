"""Null calibration: per-group null parameters and flags under the theoretical and the fitted null (S9: the test
layer decides every flag; this module only counts them)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ._profile import ProviderProfile
from ._resolve import resolve_profile

__all__ = ["calibration_profiles", "calibration_summary"]


def calibration_profiles(source: Any, args: tuple, kwargs: dict, *, display: str):
    """``(calibrated, theoretical)`` profiles: the requested test, and the same test under the theoretical null.

    ``theoretical`` is ``None`` when the source is not a model (a profile cannot be tested again).
    """
    calibrated = resolve_profile(source, args, kwargs, display=display, limits=False)
    if isinstance(source, (ProviderProfile, pd.DataFrame)):
        return calibrated, None
    theoretical_kwargs = {k: v for k, v in kwargs.items() if k != "null_model"}
    theoretical = ProviderProfile.from_model(source, *args, **theoretical_kwargs)
    return calibrated, theoretical


def calibration_summary(calibrated: ProviderProfile, theoretical: Optional[ProviderProfile]) -> pd.DataFrame:
    """One row per null group: providers, fitted null mean and SD, flags above and below under each null, changes.

    Groups come from the calibrated test's ``null_group`` (one group, labelled ``"all"``, when it has none).
    """
    f = calibrated.data
    tested = f["flag"].notna().to_numpy()
    grp = f["null_group"]
    keys = grp.astype(object).where(grp.notna(), "all").to_numpy()
    flag_c = f["flag"].to_numpy(dtype=float, na_value=np.nan)
    flag_t = (theoretical.data["flag"].reindex(f.index).to_numpy(dtype=float, na_value=np.nan)
              if theoretical is not None else np.full(len(f), np.nan))
    rows = []
    for key in pd.unique(keys[tested]):
        m = tested & (keys == key)
        mean = f.loc[m, "null_mean"].to_numpy(dtype=float)
        sd = f.loc[m, "null_sd"].to_numpy(dtype=float)
        row = {"null_group": key, "providers": int(m.sum()),
               "null_mean": float(mean[0]) if np.all(mean == mean[0]) else np.nan,
               "null_sd": float(sd[0]) if np.all(sd == sd[0]) else np.nan,
               "above_fitted": int((flag_c[m] == 1).sum()), "below_fitted": int((flag_c[m] == -1).sum())}
        if theoretical is not None:
            row.update({"above_theoretical": int((flag_t[m] == 1).sum()),
                        "below_theoretical": int((flag_t[m] == -1).sum()),
                        "changed": int((flag_t[m] != flag_c[m]).sum())})
        rows.append(row)
    out = pd.DataFrame(rows).set_index("null_group")
    try:
        out = out.sort_index()
    except TypeError:
        pass
    return out
