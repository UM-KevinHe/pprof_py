"""The standalone plotting functions as delegates to the presentation layer (single rendering path; D45, D60)."""
from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

_REMOVAL = "will be removed in 0.7.0"


def _changed_style(legacy: Callable, name: str, given: Dict[str, Any]) -> List[str]:
    """Styling keywords given with a value other than the earlier default; unknown keywords raise as before."""
    params = inspect.signature(legacy).parameters
    unknown = [k for k in given if k not in params]
    if unknown:
        raise TypeError(f"{name}() got unexpected keyword arguments {unknown}")
    out = []
    for key, value in given.items():
        try:
            same = bool(value == params[key].default)
        except (TypeError, ValueError):
            same = False
        if not same:
            out.append(key)
    return out


def _warn_style(name: str, changed: List[str]) -> None:
    if changed:
        warnings.warn(f"{name}(): the styling keywords {changed} are deprecated and ignored; the presentation theme "
                      f"sets the style (derive a Theme instead). They {_REMOVAL}.", DeprecationWarning, stacklevel=4)


def _finish(result: Any, save_path: Optional[str], dpi: int) -> Any:
    if save_path:
        result.save(save_path, dpi=dpi)
    return result


def funnel_from_frame(df: pd.DataFrame, limits_df: pd.DataFrame, *, estimate_col: str, precision_col: str,
                      flag_col: Optional[str], target: float, alpha_levels: Optional[List[float]], plot_title: str,
                      save_path: Optional[str], dpi: int, ax: Any, style: Dict[str, Any], legacy: Callable) -> Any:
    if ax is not None:
        warnings.warn("plot_funnel(ax=...) keeps the earlier drawing, which is deprecated and " + _REMOVAL + "; use "
                      "pprof_py.presentation.funnel() and its FigureResult.figure to compose figures.",
                      DeprecationWarning, stacklevel=3)
        return legacy(df, limits_df, estimate_col=estimate_col, precision_col=precision_col, flag_col=flag_col,
                      target=target, alpha_levels=alpha_levels, plot_title=plot_title, save_path=save_path, dpi=dpi,
                      ax=ax, **style)
    _warn_style("plot_funnel", _changed_style(legacy, "plot_funnel", style))
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
                           refline_value: Optional[float], sort_by_estimate: bool, orientation: str, plot_title: str,
                           save_path: Optional[str], dpi: int, style: Dict[str, Any], legacy: Callable) -> Any:
    lo_col, hi_col = ci_lower_col, ci_upper_col
    if (lo_col, hi_col) == ("ci_lower", "ci_upper") and not {"ci_lower", "ci_upper"} <= set(df.columns) \
            and {"lower", "upper"} <= set(df.columns):
        warnings.warn("plot_caterpillar() now reads intervals from 'ci_lower' and 'ci_upper' by default (the columns of "
                      "test()); this frame has 'lower' and 'upper', which are used for now. Pass ci_lower_col='lower', "
                      "ci_upper_col='upper' explicitly; this fallback will be removed in 0.7.0.", DeprecationWarning,
                      stacklevel=3)                              # R8's wording, unchanged
        lo_col, hi_col = "lower", "upper"
    reason = None
    if lo_col is None or hi_col is None or lo_col not in df.columns or hi_col not in df.columns:
        reason = "without interval columns"
    elif refline_value is None:
        reason = "without a reference line"
    elif not sort_by_estimate:
        reason = "with sort_by_estimate=False"
    elif orientation != "vertical":
        reason = f"with orientation={orientation!r}"
    if reason:
        warnings.warn(f"plot_caterpillar() {reason} keeps the earlier drawing, which is deprecated and {_REMOVAL}; "
                      "use pprof_py.presentation.caterpillar().", DeprecationWarning, stacklevel=3)
        return legacy(df, estimate_col, lo_col, hi_col, group_col, flag_col, refline_value=refline_value,
                      sort_by_estimate=sort_by_estimate, orientation=orientation, plot_title=plot_title,
                      save_path=save_path, dpi=dpi, **style)
    _warn_style("plot_caterpillar", _changed_style(legacy, "plot_caterpillar", style))
    from ..presentation import ProviderProfile, caterpillar

    index = pd.Index(df[group_col]) if group_col else df.index
    flags = df[flag_col].to_numpy() if flag_col else np.full(len(df), pd.NA)
    frame = pd.DataFrame({"estimate": df[estimate_col].to_numpy(dtype=float),
                          "ci_lower": df[lo_col].to_numpy(dtype=float), "ci_upper": df[hi_col].to_numpy(dtype=float),
                          "null_value": float(refline_value), "flag": pd.array(flags, dtype="Int64")}, index=index)
    prof = ProviderProfile.from_frame(frame, roles={c: c for c in frame.columns}, provenance={})
    return _finish(caterpillar(prof, volume=False, title=plot_title), save_path, dpi)   # no volumes in a frame
