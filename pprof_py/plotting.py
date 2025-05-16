import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional, List, Dict, Tuple, Literal


def plot_caterpillar(
    df: pd.DataFrame,
    estimate_col: str = 'estimate',
    ci_lower_col: Optional[str] = 'lower', # Changed default
    ci_upper_col: Optional[str] = 'upper', # Changed default
    group_col: Optional[str] = None, # If None, uses DataFrame index
    flag_col: Optional[str] = None,
    labels: List[str] = ["Lower", "Expected", "Higher"], # For flags -1, 0, 1
    sort_by_estimate: bool = True,
    point_size: float = 2.0,
    # Use flag_colors dict for coloring if flag_col is provided
    point_color_default: str = "#56B4E9", # Default if no flags or flag not in dict
    flag_colors: Dict[int, str] = {-1: "#E69F00", 0: "#56B4E9", 1: "#009E73"},
    point_alpha: float = 0.8,
    line_size: float = 0.8, # For refline
    refline_value: Optional[float] = 0.0, # Default refline at 0
    refline_linestyle: str = '--', # Changed name for clarity
    refline_color: str = "black", # Changed default
    font_size: float = 12,
    tick_label_size: float = 10,
    errorbar_size: float = 0.5,
    errorbar_alpha: float = 0.5, # Matched funnel fill alpha
    # Error bar color will match point color if flags are used
    errorbar_color_default: str = "grey", # Default if no flags or flag not in dict
    figure_size: Tuple[float, float] = (8, 6),
    plot_title: str = "Caterpillar Plot",
    xlab: str = "Estimate", # Swapped axes
    ylab: str = "Group",    # Swapped axes
    legend_location: str = 'best',
    orientation: Literal["vertical", "horizontal"] = "vertical",
    add_grid: bool = True,
    grid_style: str = ':',
    grid_alpha: float = 0.6,
    remove_top_right_spines: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> None:
    """Create a highly customizable caterpillar plot from a DataFrame.

    Plots estimates on the x-axis against sorted groups on the y-axis,
    with optional confidence intervals and color-coding based on flags.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot. Index should represent groups if group_col is None.
    estimate_col : str, default='estimate'
        Column name for point estimates (plotted on x-axis).
    ci_lower_col : str or None, default='lower'
        Column name for lower confidence interval bounds. If None, no error bars.
    ci_upper_col : str or None, default='upper'
        Column name for upper confidence interval bounds. If None, no error bars.
    group_col : str or None, default=None
        Column name for group identifiers. If None, uses the DataFrame index.
    flag_col : str or None, default=None
        Column name for flags (e.g., -1, 0, 1). If provided, points and error bars
        are colored based on flag values using `flag_colors`.
    labels : List[str], default=["Lower", "Expected", "Higher"]
        Labels for flag categories (-1, 0, 1) used in the legend.
    sort_by_estimate : bool, default=True
        Whether to sort the groups on the y-axis by estimate values.
    point_size : float, default=2.0
        Scaling factor for marker size (base size is 30).
    point_color_default : str, default="#56B4E9"
        Default color for points if `flag_col` is None or a flag value is missing
        from `flag_colors`.
    flag_colors : Dict[int, str], default={-1: "#E69F00", 0: "#56B4E9", 1: "#009E73"}
        Dictionary mapping flag values to colors.
    point_alpha : float, default=0.8
        Transparency of points.
    line_size : float, default=0.8
        Thickness for the reference line.
    refline_value : float or None, default=0.0
        Value for the vertical reference line (on the estimate axis). If None, no line.
    refline_linestyle : str, default='--'
        Line style for the reference line.
    refline_color : str, default="black"
        Color for the reference line.
    font_size : float, default=12
        Base font size for labels and title.
    tick_label_size : float, default=10
        Font size for axis tick labels.
    errorbar_size : float, default=0.5
        Thickness (linewidth) of the error bars.
    errorbar_alpha : float, default=0.5
        Transparency of the error bars.
    errorbar_color_default : str, default="grey"
        Default color for error bars if `flag_col` is None or a flag value is missing
        from `flag_colors`. If flags are used, error bars match point colors.
    figure_size : Tuple[float, float], default=(8, 6)
        Figure size for the plot.
    plot_title : str, default="Caterpillar Plot"
        Plot title.
    xlab : str, default="Estimate"
        X-axis label.
    ylab : str, default="Group"
        Y-axis label.
    legend_location : str, default='best'
        Location of the legend if `flag_col` is used.
    orientation : "vertical" (groups on y-axis, estimates on x-axis)
        or "horizontal" (groups on x-axis, estimates on y-axis).
    add_grid : bool, default=True
        Whether to display a grid.
    grid_style : str, default=':'
        Style of the grid lines.
    grid_alpha : float, default=0.6
        Transparency of the grid lines.
    remove_top_right_spines : bool, default=True
        Whether to remove the top and right spines.
    save_path : str or None, default=None
        Path to save the plot. If None, plot is displayed.
    dpi : int, default=300
        Resolution for saving the plot.
    """
    # --- Input Validation ---
    required_cols = [estimate_col]
    plot_ci = False
    if ci_lower_col and ci_upper_col:
        if ci_lower_col in df.columns and ci_upper_col in df.columns:
            required_cols += [ci_lower_col, ci_upper_col]
            plot_ci = True
        else:
            warnings.warn(f"CI columns ('{ci_lower_col}', '{ci_upper_col}') not found; skipping error bars.")
            ci_lower_col, ci_upper_col = None, None

    plot_flags = False
    if flag_col:
        if flag_col in df.columns:
            required_cols.append(flag_col)
            plot_flags = True
        else:
            warnings.warn(f"Flag column '{flag_col}' not found; skipping flag coloring.")
            flag_col = None

    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in DataFrame.")
        required_cols.append(group_col)

    missing = [c for c in required_cols if c not in df.columns and c != df.index.name]
    if missing:
        raise ValueError(f"Missing columns/index: {missing}")

    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    # --- Data Preparation ---
    plot_df = df.copy()
    if group_col:
        plot_df.set_index(group_col, inplace=True)
    if sort_by_estimate:
        plot_df.sort_values(estimate_col, inplace=True)

    n = len(plot_df)
    positions = np.arange(n)
    vals = plot_df[estimate_col].values
    if plot_ci:
        lows  = vals - plot_df[ci_lower_col].values
        highs = plot_df[ci_upper_col].values - vals
        errs  = (lows, highs)
    else:
        errs = None

    # Fixed helper to draw errorbars + points according to orientation
    def _draw(ax, x_arr, y_arr, err_arr, **kwargs):
        # extract once
        alpha      = kwargs.pop("alpha", 1.0)
        color      = kwargs.pop("color", "blue")
        s          = kwargs.pop("s", 50)
        edgecolor  = kwargs.pop("edgecolor", "grey")
        linewidth  = kwargs.pop("linewidth", 0.5)
        ecolor     = kwargs.pop("ecolor", "grey")
        elinewidth = kwargs.pop("elinewidth", 0.8)

        if err_arr is not None:
            eb_kw = {"xerr": err_arr} if orientation == "vertical" else {"yerr": err_arr}
            ax.errorbar(
                x_arr, y_arr, fmt='none',
                **eb_kw,
                ecolor=ecolor,
                elinewidth=elinewidth,
                alpha=alpha
            )

        ax.scatter(
            x_arr, y_arr,
            color=color,
            s=s,
            alpha=alpha,
            marker='o',
            edgecolor=edgecolor,
            linewidth=linewidth
        )

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figure_size)
    legend_handles = {}

    if plot_flags:
        plot_df[flag_col] = plot_df[flag_col].fillna(0).astype(int)
        for flag, subset in plot_df.groupby(flag_col):
            mask = plot_df.index.isin(subset.index)
            pos  = positions[mask]
            if orientation == "vertical":
                x_arr, y_arr = subset[estimate_col].values, pos
                err_arr = ([errs[0][mask], errs[1][mask]] if errs is not None else None)
            else:
                x_arr, y_arr = pos, subset[estimate_col].values
                err_arr = ([errs[0][mask], errs[1][mask]] if errs is not None else None)

            color_point = flag_colors.get(flag, point_color_default)
            color_err   = flag_colors.get(flag, errorbar_color_default)
            _draw(
                ax, x_arr, y_arr, err_arr,
                ecolor=color_err,
                elinewidth=errorbar_size,
                alpha=errorbar_alpha,
                color=color_point,
                s=point_size * 30,
                edgecolor='grey',
                linewidth=0.5
            )
            # build legend handle
            label_idx = {-1:0, 0:1, 1:2}.get(flag, 1)
            handle = ax.scatter([], [], color=color_point, s=point_size*30,
                                alpha=point_alpha, edgecolor='grey', linewidth=0.5)
            legend_handles[flag] = (handle, f"{labels[label_idx]} ({len(subset)})")
    else:
        if orientation == "vertical":
            x_arr, y_arr, err_arr = vals, positions, errs
        else:
            x_arr, y_arr, err_arr = positions, vals, errs

        _draw(
            ax, x_arr, y_arr, err_arr,
            ecolor=errorbar_color_default,
            elinewidth=errorbar_size,
            alpha=errorbar_alpha,
            color=point_color_default,
            s=point_size * 30,
            edgecolor='grey',
            linewidth=0.5
        )

    # Reference line
    if refline_value is not None:
        if orientation == "vertical":
            ax.axvline(refline_value, color=refline_color,
                       linestyle=refline_linestyle, linewidth=line_size)
        else:
            ax.axhline(refline_value, color=refline_color,
                       linestyle=refline_linestyle, linewidth=line_size)

    # Labels & ticks
    if orientation == "vertical":
        ax.set_xlabel(xlab, fontsize=font_size)
        ax.set_ylabel(ylab, fontsize=font_size)
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.set_yticks([])
        if add_grid:
            ax.grid(True, axis='x', linestyle=grid_style, alpha=grid_alpha, color='lightgrey')
    else:
        ax.set_xlabel(xlab, fontsize=font_size)
        ax.set_ylabel(ylab, fontsize=font_size)
        ax.tick_params(axis='y', labelsize=tick_label_size)
        ax.set_xticks([])
        if add_grid:
            ax.grid(True, axis='y', linestyle=grid_style, alpha=grid_alpha, color='lightgrey')

    # Spines
    if remove_top_right_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # thicken remaining
        ax.spines['left' if orientation=="vertical" else 'bottom'].set_linewidth(0.8)
        ax.spines['bottom' if orientation=="vertical" else 'left'].set_linewidth(0.8)

    # Legend
    if plot_flags:
        handles = [legend_handles[f][0] for f in sorted(legend_handles)]
        labels_ = [legend_handles[f][1] for f in sorted(legend_handles)]
        ax.legend(handles=handles, labels=labels_, loc=legend_location,
                  fontsize=font_size-2, title="Flag")

    # Title, layout, save/show
    ax.set_title(plot_title, fontsize=font_size+2, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

