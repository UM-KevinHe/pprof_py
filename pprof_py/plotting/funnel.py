"""Shared funnel-plot renderer.

A pure rendering function that accepts precomputed data and produces a
publication-ready funnel plot.  Both ``LogisticFixedEffectModel`` (O/E
ratio vs. precision) and ``LinearFixedEffectModel`` /
``LinearRandomEffectModel`` (standardized difference vs. group size) use
the *same* visual layout -- scatter points color-coded by flag,
confidence-band fill, control-limit lines, and a target reference line --
the only difference is how the data is prepared before this function is
called.

Follows the same pure-function-over-a-DataFrame-styled-from-style.py
pattern as :func:`~pprof_py.plotting.coefficients.plot_caterpillar`.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
from . import style as _style


def plot_funnel(df: pd.DataFrame, limits_df: pd.DataFrame, *, estimate_col: str = "estimate",
                precision_col: str = "precision", flag_col: Optional[str] = "flag", target: float = 1.0,
                alpha_levels: Optional[List[float]] = None, plot_title: str = "Funnel Plot",
                save_path: Optional[str] = None, dpi: int = _style.SAVE_DPI):
    """Funnel plot from precomputed data and hand-built limits (the escape hatch; D45).

    Draws through :func:`pprof_py.presentation.funnel`: the curves of ``limits_df`` (columns ``precision``,
    ``control_lower``, ``control_upper``, ``alpha``) are drawn as supplied, and each provider's limits are read off
    the curve with the largest ``alpha`` (the flags' level) at its precision, so providers whose flags contradict
    the supplied limits are reported (S4). Returns a :class:`~pprof_py.presentation.FigureResult`, which still
    unpacks as ``fig, ax``. The styling keywords and ``ax=`` were removed in 0.7.0; compose figures from
    ``FigureResult.figure``.
    """
    from ._standalone import funnel_from_frame

    return funnel_from_frame(df, limits_df, estimate_col=estimate_col, precision_col=precision_col, flag_col=flag_col,
                             target=target, alpha_levels=alpha_levels, plot_title=plot_title, save_path=save_path,
                             dpi=dpi)
