"""The standalone plotting functions as delegates to the presentation layer (single rendering path; D45, D60, D91)."""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd

def _finish(result: Any, save_path: Optional[str], dpi: int) -> Any:
    if save_path:
        result.save(save_path, dpi=dpi)
    return result


def funnel_from_frame(df: pd.DataFrame, limits_df: pd.DataFrame, *, estimate_col: str, precision_col: str,
                      flag_col: Optional[str], target: float, alpha_levels: Optional[List[float]], plot_title: str,
                      save_path: Optional[str], dpi: int) -> Any:
    from ..presentation import ProviderProfile, funnel

    alphas = sorted(set(alpha_levels) if alpha_levels is not None else set(limits_df["alpha"].unique()))
    flag_alpha = max(alphas)                                   # the narrowest limits: the flags' level
    lim = limits_df[limits_df["alpha"].isin(alphas)]
    curves = pd.DataFrame({"level": 1.0 - lim["alpha"].to_numpy(dtype=float),
                           "test_level": (lim["alpha"] == flag_alpha).to_numpy(),
                           "precision": lim["precision"].to_numpy(dtype=float),
                           "lower": lim["control_lower"].to_numpy(dtype=float),
                           "upper": lim["control_upper"].to_numpy(dtype=float)})
    at = curves[curves["test_level"]].sort_values("precision", kind="stable")
    precision = df[precision_col].to_numpy(dtype=float)
    lower = np.interp(precision, at["precision"], at["lower"], left=np.nan, right=np.nan)
    upper = np.interp(precision, at["precision"], at["upper"], left=np.nan, right=np.nan)
    flags = df[flag_col].astype("Int64") if flag_col else pd.Series(pd.NA, index=df.index, dtype="Int64")
    frame = pd.DataFrame({"estimate": df[estimate_col].to_numpy(dtype=float),
                          "funnel_estimate": df[estimate_col].to_numpy(dtype=float), "funnel_precision": precision,
                          "funnel_lower": lower, "funnel_upper": upper, "null_value": float(target),
                          "flag": flags.to_numpy()}, index=df.index)
    frame["flag"] = frame["flag"].astype("Int64")
    levels = tuple(sorted({1.0 - a for a in alphas}))
    prof = ProviderProfile.from_frame(frame, roles={c: c for c in frame.columns},
                                      provenance={"level": 1.0 - flag_alpha,
                                                  "funnel": {"levels": levels, "curve_kind": "supplied",
                                                             "limit_rule": "limits supplied with the data",
                                                             "estimate_kind": "ratio" if target == 1.0 else "effect"}},
                                      curves=curves)
    return _finish(funnel(prof, title=plot_title), save_path, dpi)


def caterpillar_from_frame(df: pd.DataFrame, estimate_col: str, ci_lower_col: Optional[str],
                           ci_upper_col: Optional[str], group_col: Optional[str], flag_col: Optional[str], *,
                           refline_value: Optional[float], plot_title: str, save_path: Optional[str], dpi: int) -> Any:
    lo_col, hi_col = ci_lower_col, ci_upper_col
    if lo_col is None or hi_col is None or lo_col not in df.columns or hi_col not in df.columns:
        hint = (" This frame has 'lower' and 'upper': pass ci_lower_col='lower', ci_upper_col='upper'."
                if {"lower", "upper"} <= set(df.columns) else "")
        raise ValueError(f"plot_caterpillar() needs the interval columns {lo_col!r} and {hi_col!r}; drawing without "
                         f"intervals was removed in 0.7.0.{hint}")
    if refline_value is None:
        raise ValueError("plot_caterpillar() needs refline_value, the reference line; drawing without one was removed "
                         "in 0.7.0.")
    from ..presentation import ProviderProfile, caterpillar

    index = pd.Index(df[group_col]) if group_col else df.index
    flags = df[flag_col].to_numpy() if flag_col else np.full(len(df), pd.NA)
    frame = pd.DataFrame({"estimate": df[estimate_col].to_numpy(dtype=float),
                          "ci_lower": df[lo_col].to_numpy(dtype=float), "ci_upper": df[hi_col].to_numpy(dtype=float),
                          "null_value": float(refline_value), "flag": pd.array(flags, dtype="Int64")}, index=index)
    prof = ProviderProfile.from_frame(frame, roles={c: c for c in frame.columns}, provenance={})
    return _finish(caterpillar(prof, volume=False, title=plot_title), save_path, dpi)   # no volumes in a frame
