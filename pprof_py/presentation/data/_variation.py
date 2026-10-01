"""Between-provider variation of random-effect models: the random-effect SD, its interval, the BLUPs and the range
of true provider effects implied by a normal random-effect distribution (a named derivation, S1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._profile import CapabilityError

__all__ = ["VariationSummary", "variation_summary"]


@dataclass(frozen=True)
class VariationSummary:
    """Random-effect SD ``sigma`` (with ``lower``/``upper`` when the model provides an interval), the BLUPs, and the
    derived range ``-/+ z * sigma`` of true effects (``ratio_lower``/``ratio_upper``: its exponential, logistic only)."""

    model: str
    provider_var: str
    scale: str                       # "log_odds" or "difference"
    sigma: float
    lower: Optional[float]
    upper: Optional[float]
    level: float
    interval_method: Optional[str]
    blups: pd.Series
    z: float
    range_lower: float
    range_upper: float
    ratio_lower: Optional[float]
    ratio_upper: Optional[float]


def variation_summary(model: Any, *, level: float = 0.95) -> VariationSummary:
    """Collect the between-provider variation of a fitted random-effect model.

    Logistic models: ``sigma_`` (the random-effect SD) and ``profile_sigma(level)``. Linear models:
    ``random_effect_sd_`` (``sigma_`` is their residual SD) and no interval. BLUPs from ``get_random_effects()``.

    Raises
    ------
    CapabilityError
        For models without a random-effect SD (fixed-effect models: use ``reliability()``).
    """
    name = type(model).__name__
    if name == "LogisticRandomEffectModel":
        sds, scale = model.sigma_, "log_odds"
    elif name == "LinearRandomEffectModel":
        sds, scale = model.random_effect_sd_, "difference"
    else:
        raise CapabilityError(f"between-provider variation needs a random-effect model with a random-effect SD; {name} "
                              "has none. For fixed-effect fits, reliability() shows how well the measure separates "
                              "providers.")
    if not isinstance(sds, dict) or not sds:
        raise CapabilityError(f"{name} has no fitted random-effect SD")
    var = next(iter(sds))
    sigma = float(sds[var])
    lower = upper = method = None
    if hasattr(model, "profile_sigma"):
        ps = model.profile_sigma(var=var, level=level)
        lower, upper, method = float(ps.loc[var, "lower"]), float(ps.loc[var, "upper"]), "profile likelihood"
    blups = model.get_random_effects(var=var)
    z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
    lo, hi = -z * sigma, z * sigma
    rl, ru = (float(np.exp(lo)), float(np.exp(hi))) if scale == "log_odds" else (None, None)
    return VariationSummary(model=name, provider_var=str(var), scale=scale, sigma=sigma, lower=lower, upper=upper,
                            level=float(level), interval_method=method, blups=blups, z=z, range_lower=lo,
                            range_upper=hi, ratio_lower=rl, ratio_upper=ru)
