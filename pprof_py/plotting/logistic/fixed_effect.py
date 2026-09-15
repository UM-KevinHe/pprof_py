"""Plotting for `LogisticFixedEffectModel`: funnel plots, provider-effect
and standardized-measure caterpillar plots, and a coefficient forest
plot. Mixed into the model class so that `models/logistic/fixed_effect.py`
can stay focused on configuration, fitting, and prediction (see
AGENTS.md Sections 9, 17-20).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from fast_poibin import PoiBin

from ...plotting import plot_caterpillar
from ...plotting.funnel import plot_funnel as _render_funnel
from ...plotting import style as _style


class _LogisticFEPlottingHost(Protocol):
    """Attribute contract that `FixedEffectPlottingMixin` expects from its
    host class (`LogisticFixedEffectModel`).
    """
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    groups_: Optional[np.ndarray]
    group_indices_: Optional[np.ndarray]
    group_sizes_: Optional[np.ndarray]
    outcome_: Optional[np.ndarray]
    xbeta_: Optional[np.ndarray]
    covariate_names_: list

    def _check_is_fitted(self) -> None: ...
    def calculate_confidence_intervals(self, **kwargs: Any) -> Any: ...
    def calculate_standardized_measures(self, **kwargs: Any) -> dict: ...
    def test(self, **kwargs: Any) -> Any: ...


class FixedEffectPlottingMixin:
    """Funnel, provider-effect, standardized-measure, and coefficient-forest
    plots for `LogisticFixedEffectModel`."""

    def plot_funnel(
        self,
        test_method: str = "score", # "score" or "poibin_exact"
        null: Union[str, float] = "median",
        target: float = 1.0,
        alpha: Union[float, List[float]] = 0.05,
        labels: List[str] = _style.FLAG_LABELS,
        point_colors: List[str] = [_style.COLOR_WARNING, _style.COLOR_PRIMARY, _style.COLOR_SECONDARY],
        point_shapes: List[str] = ['v', 'o', '^'],
        point_size: float = _style.POINT_SIZE,
        point_alpha: float = _style.POINT_ALPHA,
        line_size: float = _style.LINE_WIDTH,
        target_linestyle: str = _style.REFLINE_LINESTYLE,
        font_size: float = _style.FONT_SIZE,
        tick_label_size: float = _style.TICK_LABEL_SIZE,
        cl_line_colors: Optional[Union[str, List[str]]] = _style.COLOR_NEUTRAL_GREY,
        cl_line_styles: Optional[Union[str, List[str]]] = None,
        fill_color: str = _style.CI_FILL_COLOR,
        fill_alpha: float = _style.CI_FILL_ALPHA,
        edge_color: Optional[str] = None,
        edge_linewidth: float = 0,
        add_grid: bool = True,
        grid_style: str = _style.GRID_STYLE,
        grid_alpha: float = _style.GRID_ALPHA,
        remove_top_right_spines: bool = True,
        figure_size: Tuple[float, float] = _style.FIGURE_SIZE,
        plot_title: str = "Funnel Plot (Indirect Standardization)",
        xlab: str = "Precision",
        ylab: str = "Indirectly Standardized Ratio (O/E)",
        legend_location: str = 'best' # Location for legend
     ) -> None:
        """Create a funnel plot comparing provider performance (Indirect Ratio O/E).

        Control limits and point flagging are based on the specified test method
        ('score' or 'poibin_exact'). Precision is calculated as Expected^2 / Variance.

        Parameters:
        -----------
        test_method : str, default="score"
            Method for flagging and control limits: "score" or "poibin_exact".
            "poibin_exact" requires the 'poibin' library.
        null : str or float, default="median"
            Baseline for provider effects (gamma) used in null hypothesis calculations.
            Can be "median" or a specific float value.
        target : float, default=1.0
            Reference performance value (target line on the plot).
        alpha : float or List[float], default=0.05
            Significance level(s) for control limits (e.g., 0.05 for 95% limits).
            If a list is provided, multiple limit boundaries will be drawn.
        labels : List[str], default=["Lower", "Expected", "Higher"]
            Labels for provider performance categories based on flags (-1, 0, 1).
        point_colors : List[str]
            Colors for provider points based on performance flag. Cycles through list.
        point_shapes : List[str]
            Marker shapes for provider points based on performance flag. Cycles through list.
        point_size : float, default=2.0
            Scaling factor for marker size (base size is typically ~20-50).
        point_alpha : float, default=0.8
            Marker transparency (0 to 1).
        line_size : float, default=0.8
            Thickness for target and control limit lines.
        target_linestyle : str, default='--'
            Line style for the target reference line (matplotlib style).
        font_size : float, default=12
            Base font size for labels and title.
        tick_label_size : float, default=10
            Font size for axis tick labels.
        cl_line_colors : str or List[str], optional
            Color(s) for the control limit lines. If a list, cycles through alphas. Defaults to "grey".
        cl_line_styles : str or List[str], optional
            Line style(s) for control limits. If None, defaults based on number of alphas.
            E.g., ['-', '--', ':'] for 3 alpha levels.
        fill_color : str, default="#A6CEE3"
            Fill color for the area between the outermost control limits (smallest alpha).
        fill_alpha : float, default=0.25
            Transparency of the control limit fill area (0 to 1).
        edge_color : str or None, default=None
            Edge color for scatter points. Set to None for no edges.
        edge_linewidth : float, default=0.5
             Line width for scatter point edges.
        add_grid : bool, default=True
             Whether to add a background grid to the plot.
        grid_style : str, default=':'
             Line style for the grid.
        grid_alpha : float, default=0.6
             Transparency for the grid lines.
        remove_top_right_spines : bool, default=True
             Whether to remove the top and right axis lines (spines) for a cleaner look.
        figure_size : Tuple[float, float], default=(8, 6)
            Figure size in inches (width, height).
        plot_title : str, default="Funnel Plot (Indirect Standardization)"
            Title for the plot.
        xlab : str, default="Precision (Expected^2 / Variance)"
            Label for the x-axis.
        ylab : str, default="Indirectly Standardized Ratio (O/E)"
            Label for the y-axis.
                 legend_location : str, default='best'
             Location string for the legend (e.g., 'best', 'upper right', 'lower left',
             'center left', 'upper center', 'center', or tuple (x,y)).

        Raises:
        -------
        ValueError
            If model is not fitted, arguments are invalid, or required attributes missing.

        """

        # --- Input Validation ---
        if self.coefficients_ is None or self.groups_ is None or self.group_sizes_ is None:
            raise ValueError("Model must be fitted with coefficients, groups, and group sizes.")
        if self.outcome_ is None or self.xbeta_ is None or self.group_indices_ is None:
             raise ValueError("Model requires 'outcome_', 'xbeta_', and 'group_indices_' attributes for funnel plot calculations.")
        allowed_tests = ["score", "poibin_exact"]
        if test_method not in allowed_tests:
            raise ValueError(f"Argument 'test_method' must be one of {allowed_tests}.")
        if not isinstance(labels, list) or len(labels) != 3:
            raise ValueError("'labels' must be a list of three strings.")
        if not isinstance(point_colors, list) or len(point_colors) < 1:
            raise ValueError("'point_colors' must be a non-empty list.")
        if not isinstance(point_shapes, list) or len(point_shapes) < 1:
            raise ValueError("'point_shapes' must be a non-empty list.")

        a_list = sorted([alpha] if isinstance(alpha, (float, int)) else alpha)
        if not all(0 < a < 1 for a in a_list):
            raise ValueError("'alpha' must be between 0 and 1.")
        alpha_test = min(a_list) # Use the smallest alpha for flagging

        # --- Data Preparation ---
        # 1. Get Standardized Measures & Set Index
        try:
            sm_results = self.calculate_standardized_measures(stdz="indirect", null=null)
            df = sm_results["indirect"].copy()
            if len(df) == len(self.groups_):
                 df.index = self.groups_
            else: 
                raise ValueError(f"Length mismatch: std measures ({len(df)}) vs groups ({len(self.groups_)}).")
            required_cols = ["indirect_ratio", "observed", "expected"]
            if not all(col in df.columns for col in required_cols): 
                raise ValueError(f"Missing required columns: {required_cols}")
        except (AttributeError, KeyError, ValueError) as e: 
            raise ValueError(f"Failed data prep (std measures). Error: {e}")

        # 2. Calculate Probabilities under Null
        gamma_vals = self.coefficients_["gamma"].flatten()
        if null == "median": 
            gamma_null = np.median(gamma_vals)
        elif isinstance(null, (int, float)): 
            gamma_null = float(null)
        else: 
            raise ValueError("Argument 'null' must be 'median' or a numeric value.")

        pvec_null_all = 1.0 / (1.0 + np.exp(-(gamma_null + self.xbeta_)))
        pvec_null_all = np.clip(pvec_null_all, 1e-10, 1 - 1e-10)
        probs_by_group_idx = [pvec_null_all[self.group_indices_ == i] for i in range(len(self.groups_))]

        # 3. Calculate Variance and Precision
        group_vars_null = np.array([np.sum(p * (1 - p)) for p in probs_by_group_idx])
        group_vars_null_clipped = np.maximum(group_vars_null, 1e-12)
        expected_counts_null = np.array([np.sum(p) for p in probs_by_group_idx])
        df['expected_null'] = expected_counts_null; df['variance_null'] = group_vars_null
        df["precision"] = np.where(df['variance_null'] > 1e-14, np.square(df['expected_null']) / group_vars_null_clipped, np.inf)
        max_finite_precision = df.loc[np.isfinite(df['precision']), 'precision'].max(skipna=True)
        if pd.isna(max_finite_precision): 
            max_finite_precision = 1
        df["precision"].replace(np.inf, max_finite_precision * 1.1, inplace=True)

        # 4. Get Flags
        test_df = self.test(null=null, level=1.0 - alpha_test, test_method=test_method)
        df["flag"] = test_df.loc[df.index, "flag"].astype(int)

        # --- Calculate Control Limits ---
        limits_list = []
        for a in a_list:
            if test_method == "score":
                z_val = norm.ppf(1 - a / 2); se_ratio = np.sqrt(1.0 / df["precision"])
                se_ratio[~np.isfinite(se_ratio)] = 0.0
                control_lower = target - z_val * se_ratio; control_upper = target + z_val * se_ratio
                limits_df = pd.DataFrame(
                    {"precision": df["precision"], "control_lower": np.maximum(0, control_lower), "control_upper": control_upper, "alpha": a}, 
                    index=df.index)
                limits_list.append(limits_df)
            elif test_method == "poibin_exact":
                cl_lower_ratio, cl_upper_ratio, group_precision_list = [], [], []
                for i, g_id in enumerate(df.index):
                     probs_g = probs_by_group_idx[i]
                     expected_g = df.loc[g_id, 'expected_null']
                     precision_g = df.loc[g_id, 'precision']

                     if len(probs_g) == 0 or expected_g < 1e-10:
                         cl_lower_ratio.append(0);
                         cl_upper_ratio.append(np.inf if expected_g < 1e-10 else 0)
                         group_precision_list.append(precision_g)
                         continue
                     pb = PoiBin(probs_g)
                     o_lower = pb.quantile(a / 2)
                     o_upper = pb.quantile(1 - a / 2)
                     limit_lower = o_lower / expected_g
                     limit_upper = o_upper / expected_g
                     cl_lower_ratio.append(limit_lower) 
                     cl_upper_ratio.append(limit_upper)
                     group_precision_list.append(precision_g)
                limits_df = pd.DataFrame(
                    {"precision": group_precision_list, "control_lower": np.maximum(0, cl_lower_ratio), "control_upper": cl_upper_ratio, "alpha": a}, 
                    index=df.index)
                limits_list.append(limits_df)
        limits_all_alphas = pd.concat(limits_list)

        # --- Render via shared funnel primitive --------------------------
        return _render_funnel(
            df=df,
            limits_df=limits_all_alphas,
            estimate_col="indirect_ratio",
            precision_col="precision",
            flag_col="flag",
            target=target,
            alpha_levels=a_list,
            labels=labels,
            flag_colors=dict(zip([-1, 0, 1], point_colors[:3])),
            point_shapes=point_shapes,
            point_size=point_size,
            point_alpha=point_alpha,
            edge_color=edge_color,
            edge_linewidth=edge_linewidth,
            cl_line_colors=cl_line_colors,
            cl_line_styles=cl_line_styles,
            fill_color=fill_color,
            fill_alpha=fill_alpha,
            target_linestyle=target_linestyle,
            line_size=line_size,
            font_size=font_size,
            tick_label_size=tick_label_size,
            add_grid=add_grid,
            grid_style=grid_style,
            grid_alpha=grid_alpha,
            remove_top_right_spines=remove_top_right_spines,
            figure_size=figure_size,
            plot_title=plot_title,
            xlab=xlab,
            ylab=ylab,
            legend_location=legend_location,
        )

    def plot_provider_effects(
        self,
        group_ids=None,
        level: float = 0.95,
        test_method: str = 'wald',
        use_flags: bool = True,
        null: Union[str, float] = 'median',
        **plot_kwargs
    ) -> None:
        """Plot provider-specific effects (gamma) with confidence intervals in a caterpillar plot.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot. If None, all providers are included.
        level : float, default=0.95
            Confidence level for intervals.
        test_method : str, default='wald'
            Method for computing intervals: 'wald', 'score', or 'exact'.
        use_flags : bool, default=True
            Whether to color-code providers based on flags from the test method.
        null : str or float, default='median'
            Null hypothesis for gamma in the test method.
        **plot_kwargs
            Additional arguments passed to plot_caterpillar (e.g., title, point_color).
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted first.")

        # 1) Compute gamma‐CIs
        ci_results = self.calculate_confidence_intervals(
            providers=group_ids,
            level=level,
            option='gamma',
            test_method=test_method
        )
        df = ci_results['gamma_ci']

        # 2) Merge in flags based on the same `null`
        if use_flags:
            test_df = self.test(
                providers=group_ids,
                level=level,
                test_method=test_method,
                null=null
            )
            df = df.merge(
                test_df[['flag']],
                left_on='group_id',
                right_index=True,
                how='left'
            )

        # 3) Figure out what “null” really is
        gamma_vals = self.coefficients_["gamma"].flatten()
        if   null == "median":
            gamma_null = np.median(gamma_vals)
        elif null == "mean":
            gamma_null = np.mean(gamma_vals)
        elif isinstance(null, (int, float)):
            gamma_null = float(null)
        else:
            raise ValueError("`null` must be 'median', 'mean', or a numeric value")

        # 4) Orientation & axis‐labels
        orientation = plot_kwargs.pop('orientation', 'vertical')
        if orientation not in ('vertical','horizontal'):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        est_label = "Gamma Estimate"
        grp_label = "Provider"

        if orientation == 'vertical':
            xlab = plot_kwargs.pop('xlabel', est_label)
            ylab = plot_kwargs.pop('ylabel', grp_label)
        else:
            xlab = plot_kwargs.pop('xlabel', grp_label)
            ylab = plot_kwargs.pop('ylabel', est_label)

        # 5) Use gamma_null as the default ref‑line
        refline_value = plot_kwargs.pop('refline_value', gamma_null)

        # 6) Fire off the caterpillar
        plot_caterpillar(
            df=df,
            estimate_col='gamma',
            ci_lower_col='gamma_lower',
            ci_upper_col='gamma_upper',
            group_col='group_id',
            flag_col='flag' if use_flags else None,
            orientation=orientation,
            refline_value=refline_value,
            xlab=xlab,
            ylab=ylab,
            plot_title=plot_kwargs.pop('title', 'Provider Effects (Gamma)'),
            **plot_kwargs
        )
        
    def plot_standardized_measures(
        self,
        group_ids=None,
        level: float = 0.95,
        stdz: str = 'indirect',
        measure: str = 'ratio',
        test_method: str = 'score',
        use_flags: bool = True,
        null: Union[str, float] = 'median',
        **plot_kwargs
    ) -> None:
        """Plot standardized measures (e.g., indirect ratio or rate) with confidence intervals in a caterpillar plot.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot. If None, all providers are included.
        level : float, default=0.95
            Confidence level for intervals.
        stdz : str, default='indirect'
            Standardization method: 'indirect' or 'direct'.
        measure : str, default='ratio'
            Measure to plot: 'rate' or 'ratio'.
        test_method : str, default='score'
            Method for computing intervals: 'wald', 'score', or 'exact'.
        use_flags : bool, default=True
            Whether to color-code providers based on flags from the test method.
        null : str or float, default='median'
            Null hypothesis for gamma in the test method.
        **plot_kwargs
            Additional arguments passed to plot_caterpillar (e.g., title, point_color).
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted first.")

        # Compute SM‐CIs
        ci_results = self.calculate_confidence_intervals(
            providers=group_ids,
            level=level,
            option='SM',
            stdz=stdz,
            measure=measure,
            test_method=test_method,
            null=null
        )
        key = f"{stdz}_{measure}"
        if key not in ci_results:
            raise ValueError(f"Invalid combination: stdz='{stdz}', measure='{measure}'")
        df = ci_results[key]
        
        # Determine default refline:
        #  - ratios → 1
        #  - rates  → population_rate (stored in df.attrs by _compute_sm_intervals)
        refline_value = plot_kwargs.pop('refline_value', None)
        if refline_value is None:
            if measure == 'ratio':
                refline_value = 1.0
            elif measure == 'rate':
                refline_value = df.attrs.get('population_rate', None)

        # Merge in flags if desired
        if use_flags:
            test_df = self.test(
                providers=group_ids,
                level=level,
                test_method=test_method,
                null=null
            )
            df = df.merge(test_df[['flag']],
                        left_on='group_id',
                        right_index=True,
                        how='left')

        # Orientation
        orientation = plot_kwargs.pop('orientation', 'vertical')
        if orientation not in ('vertical','horizontal'):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")


        # Axis‐labels
        est_label = f"{stdz.capitalize()} {measure.capitalize()}"
        grp_label = "Provider"

        if orientation == 'vertical':
            xlabel = plot_kwargs.pop('xlabel', est_label)
            ylabel = plot_kwargs.pop('ylabel', grp_label)
        else:
            xlabel = plot_kwargs.pop('xlabel', grp_label)
            ylabel = plot_kwargs.pop('ylabel', est_label)

        # Plot
        plot_caterpillar(
            df=df,
            estimate_col=f"{stdz}_{measure}",
            ci_lower_col=f"ci_{measure}_lower",
            ci_upper_col=f"ci_{measure}_upper",
            group_col='group_id',
            flag_col='flag' if use_flags else None,
            orientation=orientation,
            refline_value=refline_value,
            xlab=xlabel,
            ylab=ylabel,
            plot_title=plot_kwargs.pop(
                'title',
                f"{stdz.capitalize()} Standardized {measure.capitalize()}"
            ),
            **plot_kwargs
        )

    def plot_coefficient_forest(
        self,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        refline_value: Optional[float] = 0.0,
        point_color: str = _style.COLOR_ESTIMATE,
        point_alpha: float = _style.POINT_ALPHA,
        edge_color: Optional[str] = None,
        edge_linewidth: float = 0,
        point_size: float = 0.05,
        error_color: str = _style.COLOR_NEUTRAL_GREY,
        capsize: float = _style.CAPSIZE,
        errorbar_size: float = _style.ERRORBAR_SIZE,
        errorbar_alpha: float = _style.ERRORBAR_ALPHA,
        line_color: str = _style.COLOR_REFERENCE,
        line_style: str = _style.REFLINE_LINESTYLE,
        line_size: float = _style.LINE_WIDTH,
        font_size: float = _style.FONT_SIZE,
        tick_label_size: float = _style.TICK_LABEL_SIZE,
        add_grid: bool = True,
        grid_style: str = _style.GRID_STYLE,
        grid_alpha: float = _style.GRID_ALPHA,
        remove_top_right_spines: bool = True,
        figure_size: Tuple[float, float] = _style.FIGURE_SIZE,
        plot_title: str = "Forest Plot of Covariate Coefficients",
        xlab: str = "Coefficient Estimate",
        ylab: str = "Covariate",
        save_path: Optional[str] = None,
        dpi: int = _style.SAVE_DPI
    ) -> None:
        """Create a forest plot of covariate coefficients with 95% confidence intervals.

        Plots each covariate's coefficient estimate and its confidence interval
        in a vertical or horizontal layout, with a reference line at a specified value.

        Parameters
        ----------
        orientation : {'vertical','horizontal'}, default 'vertical'
            'vertical': covariate names on the y-axis, estimates on the x-axis.
            'horizontal': covariate names on the x-axis, estimates on the y-axis.
        refline_value : float or None, default 0.0
            Draws a reference line at this value (vertical or horizontal). None disables it.
        point_color : str, default "#34495E"
            Color of the coefficient marker.
        point_alpha : float, default 0.8
            Opacity of the coefficient marker.
        edge_color : str or None, default None
            Edge color of the marker. None for no edge.
        edge_linewidth : float, default 0
            Width of the marker edge.
        point_size : float, default 0.5
            Scale factor for marker size.
        error_color : str, default "#95A5A6"
            Color of the error bars.
        capsize : float, default 5
            Cap size for error bars.
        errorbar_size : float, default 0.5
            Thickness of the error bar lines.
        errorbar_alpha : float, default 0.5
            Opacity of the error bars.
        line_color : str, default "red"
            Color of the reference line.
        line_style : str, default "--"
            Line style of the reference line.
        line_size : float, default 0.8
            Thickness of the reference line.
        font_size : float, default 12
            Font size for labels and title.
        tick_label_size : float, default 10
            Font size for tick labels.
        add_grid : bool, default True
            Whether to draw a light grid.
        grid_style : str, default ":"
            Line style for the grid.
        grid_alpha : float, default 0.6
            Opacity of the grid.
        remove_top_right_spines : bool, default True
            Hide the top and right spines.
        figure_size : tuple, default (10, 6)
            Size of the figure in inches.
        plot_title : str, default "Forest Plot of Covariate Coefficients"
            Title of the plot.
        xlab : str, default "Coefficient Estimate"
            Label for the x-axis (or y-axis if horizontal).
        ylab : str, default "Covariate"
            Label for the y-axis (or x-axis if horizontal).
        save_path : str or None, default None
            File path to save the figure. If None, the plot is shown.
        dpi : int, default 300
            Resolution (dots per inch) for saving.

        Raises
        ------
        ValueError
            If the model is not fitted or if an invalid orientation is provided.
        """
        # Preconditions
        if self.coefficients_ is None or self.variances_ is None or self.covariate_names_ is None:
            raise ValueError("Model must be fitted before plotting coefficients.")
        if orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        # Compute estimates and 95% CIs
        beta = self.coefficients_["beta"].flatten()
        se_beta = np.sqrt(np.diag(self.variances_["beta"]))
        df_denom = self.fitted_.size - len(beta) - len(self.coefficients_["gamma"])
        crit = t.ppf(1 - 0.05 / 2, df_denom)
        lower = beta - crit * se_beta
        upper = beta + crit * se_beta

        coef_df = (
            pd.DataFrame({
                "covariate": self.covariate_names_,
                "estimate": beta,
                "ci_lower": lower,
                "ci_upper": upper
            })
            .sort_values("estimate")
            .reset_index(drop=True)
        )

        # Positions
        n = len(coef_df)
        positions = np.arange(n)

        # Prepare coordinates and errors
        if orientation == "vertical":
            x_vals, y_vals = coef_df["estimate"], positions
            xerr = [
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"]
            ]
        else:
            x_vals, y_vals = positions, coef_df["estimate"]
            yerr = [
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"]
            ]

        # Plot setup
        fig, ax = plt.subplots(figsize=figure_size)

        # Draw errorbars and points
        if orientation == "vertical":
            ax.errorbar(
                x_vals, y_vals,
                xerr=xerr,
                fmt="o",
                color=point_color,
                ecolor=error_color,
                capsize=capsize,
                markersize=point_size * 30,
                alpha=point_alpha,
                linewidth=edge_linewidth if edge_color else 0,
                markeredgecolor=edge_color
            )
        else:
            ax.errorbar(
                x_vals, y_vals,
                yerr=yerr,
                fmt="o",
                color=point_color,
                ecolor=error_color,
                capsize=capsize,
                markersize=point_size * 30,
                alpha=point_alpha,
                linewidth=edge_linewidth if edge_color else 0,
                markeredgecolor=edge_color
            )

        # Reference line
        if refline_value is not None:
            if orientation == "vertical":
                ax.axvline(refline_value, color=line_color, linestyle=line_style, linewidth=line_size)
            else:
                ax.axhline(refline_value, color=line_color, linestyle=line_style, linewidth=line_size)

        # Labels, ticks, grid
        if orientation == "vertical":
            ax.set_xlabel(xlab, fontsize=font_size)
            ax.set_ylabel(ylab, fontsize=font_size)
            ax.set_yticks(positions)
            ax.set_yticklabels(coef_df["covariate"], fontsize=tick_label_size)
            ax.tick_params(axis="x", labelsize=tick_label_size)
            if add_grid:
                ax.grid(True, axis="x", linestyle=grid_style, alpha=grid_alpha, color=_style.GRID_COLOR)
        else:
            ax.set_xlabel(ylab, fontsize=font_size)
            ax.set_ylabel(xlab, fontsize=font_size)
            ax.set_xticks(positions)
            ax.set_xticklabels(coef_df["covariate"], rotation=45, ha="right", fontsize=tick_label_size)
            ax.tick_params(axis="y", labelsize=tick_label_size)
            if add_grid:
                ax.grid(True, axis="y", linestyle=grid_style, alpha=grid_alpha, color=_style.GRID_COLOR)

        # Spines
        if remove_top_right_spines:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        if orientation == "vertical":
            ax.spines["left"].set_linewidth(_style.SPINE_WIDTH)
            ax.spines["bottom"].set_linewidth(_style.SPINE_WIDTH)
        else:
            ax.spines["bottom"].set_linewidth(_style.SPINE_WIDTH)
            ax.spines["left"].set_linewidth(_style.SPINE_WIDTH)

        # Title & layout
        ax.set_title(plot_title, fontsize=font_size + 2, pad=_style.TITLE_PAD)
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

       
    def plot_residuals(self, *args, **kwargs) -> None:
        """Plot residuals versus fitted probabilities for logistic regression.

        Raises:
        -------
        NotImplementedError
            This method is not implemented as residuals are not computed by default.
        """
        raise NotImplementedError(
            "plot_residuals is not implemented for LogisticFixedEffectModel. "
            "Residual diagnostics are less standard in logistic regression and "
            "residuals are not computed by default in this model."
        )

    def plot_qq(self, *args, **kwargs) -> None:
        """Create a Q-Q plot of the deviance residuals for logistic regression.

        Raises:
        -------
        NotImplementedError
            This method is not implemented as residuals are not computed by default.
        """
        raise NotImplementedError(
            "plot_qq is not implemented for LogisticFixedEffectModel. "
            "Q-Q plots are less meaningful in logistic regression and "
            "residuals are not computed by default in this model."
        )
