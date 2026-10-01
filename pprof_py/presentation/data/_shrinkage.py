"""Shrinkage: each provider's fixed-effect (unshrunken) and random-effect (shrunken) estimate, aligned.

Fixed-effect estimates are absolute provider effects; random-effect BLUPs are deviations from the model's intercept.
Each is shown relative to its own test's reference, ``estimate - null_value`` (a named derivation, S1): the
random-effect test compares with 0, the fixed-effect test with its ``reference`` (default ``"mean"``, the
size-weighted mean of the provider effects).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ._profile import CapabilityError, ProviderProfile

__all__ = ["ShrinkagePairs", "reference_text", "shrinkage_pairs"]


@dataclass(frozen=True)
class ShrinkagePairs:
    """Paired estimates: ``frame`` (indexed by ``provider_id``: ``fixed``, ``random``, ``volume``,
    ``fixed_finite``), the providers found in only one fit, and provenance."""

    frame: pd.DataFrame
    only_fixed: int
    only_random: int
    provenance: Dict[str, Any]


def _family(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return "logistic" if "Logistic" in name else ("linear" if name.startswith("Linear") else None)


def shrinkage_pairs(fixed: Any, random: Any, *, fe_reference: Any = "mean") -> ShrinkagePairs:
    """Pair a fixed-effect fit with a random-effect fit of the same family.

    Parameters
    ----------
    fixed, random
        Fitted models (tested here: the fixed-effect model with ``reference=fe_reference``, the random-effect model
        with its defaults) or :class:`ProviderProfile` objects.
    fe_reference : {"mean", "median"} or float, default "mean"

    Raises
    ------
    CapabilityError
        When the two sources are not a fixed-effect and a random-effect fit of the same family.
    """
    pf = fixed if isinstance(fixed, ProviderProfile) else ProviderProfile.from_model(fixed, reference=fe_reference)
    pr = random if isinstance(random, ProviderProfile) else ProviderProfile.from_model(random)
    mf, mr = pf.provenance.get("model"), pr.provenance.get("model")
    ff, fr = _family(mf), _family(mr)
    if ff is None or ff != fr or "RandomEffect" in (mf or "") or "RandomEffect" not in (mr or ""):
        raise CapabilityError(f"shrinkage needs a fixed-effect and a random-effect fit of the same family; got {mf} "
                              f"and {mr}")
    f, r = pf.data, pr.data
    common = f.index.intersection(r.index, sort=False)
    vol_f = f["denominator"] if "denominator" in pf.capabilities else pd.Series(np.nan, index=f.index)
    vol_r = r["denominator"] if "denominator" in pr.capabilities else pd.Series(np.nan, index=r.index)
    volume = vol_f.reindex(common).fillna(vol_r.reindex(common))
    frame = pd.DataFrame({
        "fixed": (f.loc[common, "estimate"] - f.loc[common, "null_value"]).to_numpy(dtype=float),
        "random": (r.loc[common, "estimate"] - r.loc[common, "null_value"]).to_numpy(dtype=float),
        "volume": volume.to_numpy(dtype=float),
        "fixed_finite": f.loc[common, "finite_estimate"].fillna(True).to_numpy(dtype=bool),
    }, index=common)
    prov = {"fixed_model": mf, "random_model": mr, "family": ff, "fe_reference": pf.provenance.get("reference"),
            "fe_reference_value": float(f["null_value"].iloc[0]) if len(f) else np.nan,
            "random_reference_value": float(r["null_value"].iloc[0]) if len(r) else np.nan,
            "denominator_kind": pf.provenance.get("denominator_kind") or pr.provenance.get("denominator_kind"),
            "sigma": None}
    if not isinstance(random, ProviderProfile):
        from ._variation import variation_summary

        prov["sigma"] = variation_summary(random).sigma
    return ShrinkagePairs(frame=frame, only_fixed=int(len(f.index.difference(r.index))),
                          only_random=int(len(r.index.difference(f.index))), provenance=prov)


def reference_text(pairs: ShrinkagePairs) -> str:
    ref = pairs.provenance.get("fe_reference")
    if ref == "mean":
        return "size-weighted mean of the fixed effects"
    if ref == "median":
        return "median of the fixed effects"
    return f"reference {_fmt2(pairs.provenance.get('fe_reference_value'), 2)}"


def _fmt2(v: Any, digits: int) -> str:
    from ..formatting import fmt_number

    return fmt_number(v, digits)
