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

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import style as _style


def plot_funnel(
    df: pd.DataFrame,
    limits_df: pd.DataFrame,
    *,
    estimate_col: str = "estimate",
    precision_col: str = "precision",
    flag_col: Optional[str] = "flag",
    target: float = 1.0,
    alpha_levels: Optional[List[float]] = None,
    # -- Flag / point styling -------------------------------------------
    labels: List[str] = _style.FLAG_LABELS,
    flag_colors: Dict[int, str] = _style.FLAG_COLORS,
    point_shapes: List[str] = ("v", "o", "^"),
    point_size: float = _style.POINT_SIZE,
    point_alpha: float = _style.POINT_ALPHA,
    edge_color: Optional[str] = None,
    edge_linewidth: float = 0,
    # -- Control-limit styling ------------------------------------------
    cl_line_colors: Optional[Union[str, List[str]]] = _style.COLOR_NEUTRAL_GREY,
    cl_line_styles: Optional[Union[str, List[str]]] = None,
    fill_color: str = _style.CI_FILL_COLOR,
    fill_alpha: float = _style.CI_FILL_ALPHA,
    # -- Reference / target line ----------------------------------------
    target_linestyle: str = _style.REFLINE_LINESTYLE,
    line_size: float = _style.LINE_WIDTH,
    # -- Typography / layout --------------------------------------------
    font_size: float = _style.FONT_SIZE,
    tick_label_size: float = _style.TICK_LABEL_SIZE,
    add_grid: bool = True,
    grid_style: str = _style.GRID_STYLE,
    grid_alpha: float = _style.GRID_ALPHA,
    remove_top_right_spines: bool = True,
    figure_size: Tuple[float, float] = _style.FIGURE_SIZE,
    plot_title: str = "Funnel Plot",
    xlab: str = "Precision",
    ylab: str = "Estimate",
    legend_location: str = "best",
    # -- Output ---------------------------------------------------------
    save_path: Optional[str] = None,
    dpi: int = _style.SAVE_DPI,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Render a funnel plot from precomputed data.

    Parameters
    ----------
    df : pd.DataFrame
        One row per provider/group.  Must contain ``estimate_col``,
        ``precision_col``, and optionally ``flag_col`` (-1/0/1; NaN for a provider without a
        test result, drawn hollow as "Not tested").
    limits_df : pd.DataFrame
        Control-limit curves.  Must contain ``precision``,
        ``control_lower``, ``control_upper``, and ``alpha`` columns.
        Multiple ``alpha`` levels are supported (one set of curves per
        level).
    estimate_col, precision_col, flag_col : str
        Column names in *df*.
    target : float
        Reference-line value on the estimate axis (horizontal).
    alpha_levels : list of float, optional
        Significance levels present in *limits_df*.  If ``None``,
        inferred from ``limits_df["alpha"].unique()``.
    labels, flag_colors, point_shapes
        Cosmetic options for flag-coded scatter points (see style.py).
    save_path : str or None
        If given, the figure is saved here instead of shown.
    ax : matplotlib Axes, optional
        If provided, draw onto this axes instead of creating a new figure.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # ---- figure / axes ------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
    else:
        fig = ax.get_figure()

    # ---- resolve alpha levels ----------------------------------------
    if alpha_levels is None:
        alpha_levels = sorted(limits_df["alpha"].unique())
    else:
        alpha_levels = sorted(alpha_levels)
    outer_alpha = min(alpha_levels)

    # ---- control-limit line style/color maps -------------------------
    if cl_line_styles is None:
        _defaults = ["-", "--", ":", "-."]
        cl_line_styles = [_defaults[i % len(_defaults)] for i in range(len(alpha_levels))]
    elif isinstance(cl_line_styles, str):
        cl_line_styles = [cl_line_styles] * len(alpha_levels)

    if cl_line_colors is None or isinstance(cl_line_colors, str):
        cl_line_colors = [cl_line_colors or _style.COLOR_NEUTRAL_GREY] * len(alpha_levels)

    style_map = dict(zip(alpha_levels, cl_line_styles))
    color_map = dict(zip(alpha_levels, cl_line_colors))

    # ---- sort limits for smooth curves --------------------------------
    limits_sorted = limits_df.sort_values("precision")

    # ---- outer CI fill ------------------------------------------------
    legend_handles, legend_labels = [], []
    outer = limits_sorted[limits_sorted["alpha"] == outer_alpha]
    ci_label = f"{int((1 - outer_alpha) * 100)}% CI"
    fill_h = ax.fill_between(
        outer["precision"],
        outer["control_lower"],
        outer["control_upper"],
        color=fill_color,
        alpha=fill_alpha,
        label=ci_label,
    )
    legend_handles.append(fill_h)
    legend_labels.append(ci_label)

    # ---- control-limit lines -----------------------------------------
    for a in alpha_levels:
        subset = limits_sorted[limits_sorted["alpha"] == a]
        line_label = f"{int((1 - a) * 100)}% CI" if a != outer_alpha else None
        (line_lower,) = ax.plot(
            subset["precision"],
            subset["control_lower"],
            linestyle=style_map[a],
            color=color_map[a],
            linewidth=line_size,
            label=line_label,
        )
        ax.plot(
            subset["precision"],
            subset["control_upper"],
            linestyle=style_map[a],
            color=color_map[a],
            linewidth=line_size,
        )
        if line_label:
            legend_handles.append(line_lower)
            legend_labels.append(line_label)

    # ---- target reference line ----------------------------------------
    ax.axhline(
        y=target,
        color=_style.COLOR_REFERENCE,
        linestyle=target_linestyle,
        linewidth=line_size,
    )

    # ---- scatter points -----------------------------------------------
    flag_map = {-1: 0, 0: 1, 1: 2}
    if flag_col and flag_col in df.columns:
        present_flags = sorted(int(v) for v in pd.unique(df[flag_col].dropna()))
        untested = df[df[flag_col].isna()]            # no test result: drawn hollow, listed last
    else:
        untested = df.iloc[0:0]
        present_flags = [0]
        df = df.copy()
        df["_flag_tmp"] = 0
        flag_col = "_flag_tmp"

    for flag_val in present_flags:
        subset = df[df[flag_col] == flag_val]
        idx = flag_map.get(flag_val, 1)
        count = len(subset)
        lbl = f"{labels[idx]} ({count})"
        h = ax.scatter(
            subset[precision_col],
            subset[estimate_col],
            marker=point_shapes[idx % len(point_shapes)],
            color=flag_colors.get(flag_val, _style.COLOR_PRIMARY),
            s=point_size * 30,
            alpha=point_alpha,
            edgecolor=edge_color,
            linewidth=edge_linewidth if edge_color else 0,
            label=lbl,
        )
        legend_handles.append(h)
        legend_labels.append(lbl)
    if len(untested):
        lbl = f"{_style.UNTESTED_LABEL} ({len(untested)})"
        h = ax.scatter(
            untested[precision_col],
            untested[estimate_col],
            marker=point_shapes[1 % len(point_shapes)],
            facecolors="none",
            edgecolors=_style.COLOR_UNTESTED,
            s=point_size * 30,
            alpha=point_alpha,
            linewidth=0.8,
            label=lbl,
        )
        legend_handles.append(h)
        legend_labels.append(lbl)

    # ---- axes dressing ------------------------------------------------
    ax.set_xlabel(xlab, fontsize=font_size)
    ax.set_ylabel(ylab, fontsize=font_size)
    ax.set_title(plot_title, fontsize=font_size + 2, pad=_style.TITLE_PAD)
    ax.tick_params(axis="both", labelsize=tick_label_size)

    # axis limits
    max_x = df[precision_col].max(skipna=True)
    ax.set_xlim(left=0, right=(max_x * 1.05 if pd.notna(max_x) and max_x > 0 else 1))

    all_y = pd.concat(
        [
            df[estimate_col],
            limits_df["control_lower"],
            limits_df["control_upper"],
        ]
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if not all_y.empty:
        min_y = min(all_y.min(), target)
        max_y = max(all_y.max(), target)
        pad = (max_y - min_y) * 0.1 if (max_y - min_y) > 1e-6 else 0.1
        ax.set_ylim(min_y - pad, max_y + pad)

    if add_grid:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha, axis="both", color=_style.GRID_COLOR)

    if remove_top_right_spines:
        _style.remove_top_right_spines(ax)
        ax.spines["left"].set_linewidth(_style.SPINE_WIDTH)
        ax.spines["bottom"].set_linewidth(_style.SPINE_WIDTH)

    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        fontsize=font_size - 2,
        loc=legend_location,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    elif ax is None:
        plt.show()

    return fig, ax
