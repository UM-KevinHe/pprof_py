"""Decisions from provider z-statistics: p-values, flags, and intervals.

Everything here works on a :class:`ZFrame` (or, for p-values and flags, plain
z-values) and a null model. Conventions shared by every function:

* ``z_adjusted = (z_raw - null_mean) / null_sd`` using the null model's parameters;
* ``flag`` is +1 when a provider is significantly above its null value, -1
  when significantly below, 0 when not significant, and missing (``<NA>``)
  when the provider could not be tested. Whether "above" is better or worse
  depends on the measure and is left to the caller;
* intervals are reported on the measure's own scale.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from .empirical_null.models import NullModel, TheoreticalNull, _as_z
from .zstat import ZFrame, z_to_t

__all__ = ["calibrate", "p_values", "flags", "intervals", "provider_test", "resolve_null_model",
           "PROVIDER_TEST_COLUMNS"]

PROVIDER_TEST_COLUMNS = (
    "estimate", "se", "null_value", "transformed", "se_transformed", "null_transformed", "z_raw",
    "null_mean", "null_sd", "null_group", "z_adjusted", "p_value", "flag",
    "ci_lower", "ci_upper",
)

_ALTERNATIVES = {"two_sided": "two_sided", "two-sided": "two_sided", "greater": "greater", "less": "less"}


def _alternative(alternative: str) -> str:
    if alternative not in _ALTERNATIVES:
        raise ValueError("alternative must be 'two_sided', 'greater', or 'less'.")
    return _ALTERNATIVES[alternative]


def _check_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1.")
    return 1.0 - level


def resolve_null_model(null_model, z) -> NullModel:
    """``None`` -> :class:`TheoreticalNull`; a NullModel as is; a callable is fitted on ``z``."""
    if null_model is None:
        return TheoreticalNull()
    if isinstance(null_model, NullModel):
        return null_model
    if callable(null_model):
        return null_model(z)
    raise TypeError("null_model must be a NullModel or a callable returning one.")


def calibrate(z, null: NullModel = TheoreticalNull()) -> np.ndarray:
    """Return ``(z - null_mean) / null_sd`` under the null model."""
    zz, _ = _as_z(z)
    loc, scl, _ = null.parameters(z)
    return (zz - loc) / scl


def p_values(z, null: NullModel = TheoreticalNull(), alternative: str = "two_sided") -> np.ndarray:
    """P-values of the calibrated z-statistics (NaN where z is missing)."""
    alt = _alternative(alternative)
    zc = calibrate(z, null)
    if alt == "two_sided":
        return 2.0 * norm.sf(np.abs(zc))
    return norm.sf(zc) if alt == "greater" else norm.cdf(zc)


def flags(z, null: NullModel = TheoreticalNull(), *, alternative: str = "two_sided",
          level: float = 0.95, critical: Optional[float] = None) -> pd.arrays.IntegerArray:
    """Flag providers as significantly above (+1) or below (-1) their null value.

    Parameters
    ----------
    critical : float, optional
        When given, flag where the calibrated z exceeds this value (for example
        1.96, or 3 for three-sigma limits) instead of comparing the p-value
        with ``1 - level``.
    """
    alt = _alternative(alternative)
    zc = calibrate(z, null)
    missing = np.isnan(zc)
    if critical is None:
        alpha = _check_level(level)
        sig = np.where(missing, False, p_values(z, null, alt) < alpha)
        up, down = sig & (zc > 0), sig & (zc < 0)
    else:
        c = float(critical)
        up, down = zc > c, zc < -c
    if alt == "greater":
        down = np.zeros_like(down)
    elif alt == "less":
        up = np.zeros_like(up)
    values = np.where(up, 1, np.where(down, -1, 0)).astype(np.int8)
    return pd.arrays.IntegerArray(values, missing)


def intervals(z: ZFrame, null: NullModel = TheoreticalNull(), *, alternative: str = "two_sided",
              level: float = 0.95, critical: Optional[float] = None, form: str = "inversion",
              bounds: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Confidence limits on the measure's own scale.

    Parameters
    ----------
    form : {"inversion", "scale_only"}
        ``"inversion"`` inverts the calibrated test: on the working scale the
        limits are ``T - (null_mean +/- c * null_sd) * se`` (EmpiNull's Wald
        intervals), so a limit excludes the null value exactly when the
        provider is flagged. ``"scale_only"`` widens by the null SD but ignores
        its mean, ``T -/+ c * null_sd * se``; the two coincide when the mean is 0.
    critical : float, optional
        Critical value ``c``; defaults to the normal quantile for ``level``.
    bounds : (float, float), optional
        Clip both limits to this range (for example ``(0, 1)`` for a rate on
        the identity scale).
    """
    if not isinstance(z, ZFrame) or not z.has_intervals:
        raise ValueError("Intervals need a ZFrame built by z_statistic (with estimates and standard errors).")
    if form not in ("inversion", "scale_only"):
        raise ValueError("form must be 'inversion' or 'scale_only'.")
    alt = _alternative(alternative)
    alpha = _check_level(level)
    c = float(critical) if critical is not None else float(norm.ppf(1.0 - (alpha / 2.0 if alt == "two_sided" else alpha)))
    loc, scl, _ = null.parameters(z)
    t, s = z.transformed, z.se_transformed
    df = z.df
    with np.errstate(invalid="ignore", over="ignore"):   # untestable providers (inf or NaN) are set to NaN below
        if form == "inversion":
            upper_z, lower_z = loc + c * scl, loc - c * scl
        else:
            upper_z, lower_z = c * scl, -c * scl
        if df is not None:                              # Student-t reference: invert through t, not by scaling
            upper_z, lower_z = z_to_t(upper_z, df), z_to_t(lower_z, df)
        lo_t, hi_t = t - upper_z * s, t - lower_z * s
    if alt == "greater":
        hi_t = np.full_like(hi_t, np.inf)
    elif alt == "less":
        lo_t = np.full_like(lo_t, -np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        lo = np.asarray(z.transform.inverse(lo_t), dtype=np.float64)
        hi = np.asarray(z.transform.inverse(hi_t), dtype=np.float64)
    untested = np.isnan(z.z)
    lo, hi = np.where(untested, np.nan, lo), np.where(untested, np.nan, hi)
    if bounds is not None:
        lo, hi = np.clip(lo, bounds[0], bounds[1]), np.clip(hi, bounds[0], bounds[1])
    return lo, hi


def provider_test(z, null: NullModel = TheoreticalNull(), *, alternative: str = "two_sided",
                  level: float = 0.95, critical: Optional[float] = None, interval: str = "inversion",
                  bounds: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    """Calibrate, test, flag, and (when possible) bound every provider.

    Parameters are those of :func:`flags` and :func:`intervals`; ``critical``
    applies to both. Call those functions directly when a method needs
    different settings for flags and intervals.

    Returns
    -------
    pandas.DataFrame
        Indexed by provider, with columns :data:`PROVIDER_TEST_COLUMNS`. The
        estimate, null value, and limits are on the measure's own scale;
        ``transformed``, ``se_transformed``, and ``z_raw`` are on the working
        scale. ``attrs`` records the settings and the null model.
    """
    zf = z if isinstance(z, ZFrame) else ZFrame.from_arrays(*((z.to_numpy(), z.index) if isinstance(z, pd.Series) else (z,)))
    alt = _alternative(alternative)
    loc, scl, grp = null.parameters(zf)
    out = zf.to_frame()
    out["null_mean"] = loc
    out["null_sd"] = scl
    if grp is None:
        out["null_group"] = pd.array([pd.NA] * len(zf), dtype="Int64")
    else:
        grp_s = pd.Series(grp, index=out.index)
        out["null_group"] = grp_s.astype("Int64") if pd.api.types.is_integer_dtype(grp_s) else grp_s
    out["z_adjusted"] = (zf.z - loc) / scl
    out["p_value"] = p_values(zf, null, alt)
    out["flag"] = flags(zf, null, alternative=alt, level=level, critical=critical)
    if zf.has_intervals:
        lo, hi = intervals(zf, null, alternative=alt, level=level, critical=critical, form=interval, bounds=bounds)
    else:
        lo = hi = np.full(len(zf), np.nan)
    out["ci_lower"], out["ci_upper"] = lo, hi
    out = out[list(PROVIDER_TEST_COLUMNS)]
    out.attrs.update({"measure": zf.measure, "transform": zf.transform.name, "alternative": alt,
                      "level": level, "critical": critical, "interval": interval, "bounds": bounds,
                      "null_model": null.describe()})
    return out
