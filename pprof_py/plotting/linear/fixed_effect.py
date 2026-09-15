"""Plotting for `LinearFixedEffectModel`: funnel plots, provider-effect
and standardized-measure caterpillar plots, coefficient forest plot, and
residual/Q-Q diagnostic plots. Mixed into the model class so that
`models/linear/fixed_effect.py` can stay focused on configuration,
fitting, and prediction (see AGENTS.md Sections 9, 17-20).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, probplot

from ...plotting import plot_caterpillar
from ...plotting.funnel import plot_funnel as _render_funnel
from ...plotting import style as _style


class _LinearFEPlottingHost(Protocol):
    """Attribute contract that `FixedEffectPlottingMixin` expects from its
    host class (`LinearFixedEffectModel`).
    """
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    residuals_: Optional[np.ndarray]
    sigma_: Optional[float]
    groups_: Optional[np.ndarray]
    group_sizes_: Optional[np.ndarray]
    covariate_names_: list

    def _check_is_fitted(self) -> None: ...
    def calculate_confidence_intervals(self, **kwargs: Any) -> Any: ...
    def calculate_standardized_measures(self, **kwargs: Any) -> dict: ...
    def test(self, **kwargs: Any) -> Any: ...


class FixedEffectPlottingMixin:
    """Funnel, provider-effect, standardized-measure, coefficient-forest,
    and residual/Q-Q plots for `LinearFixedEffectModel`."""

    def plot_funnel(
        self,
        stdz: str = "indirect",
        null: Union[str, float] = "median",
        target: float = 0.0,
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
        edge_color: Optional[str] = _style.COLOR_NEUTRAL_GREY,
        edge_linewidth: float = _style.ERRORBAR_SIZE,
        add_grid: bool = True,
        grid_style: str = _style.GRID_STYLE,
        grid_alpha: float = _style.GRID_ALPHA,
        remove_top_right_spines: bool = True,
        figure_size: Tuple[float, float] = _style.FIGURE_SIZE,
        plot_title: str = "Funnel Plot of Standardized Differences",
        xlab: str = "Precision (Group Size)",
        ylab: str = "Standardized Difference",
        legend_location: str = 'best'
    ) -> None:
        """Create a funnel plot for standardized differences.

        For LinearFixedEffectModel, this plots the indirect standardized difference
        (gamma_i - gamma_null) against group size (as a proxy for precision).
        Control limits are based on the overall model's residual standard deviation (sigma).

        Parameters:
        -----------
        stdz : str, default="indirect"
            Standardization method. Currently, only "indirect" is meaningfully
            supported as both indirect and direct differences simplify to gamma_i - gamma_null.
        null : str or float, default="median"
            Baseline for provider effects (gamma) used in calculating the difference
            and for flagging. Can be "median", "mean", or a specific float value.
        target : float, default=0.0
            Reference value for the difference (target line on the plot).
        alpha : float or List[float], default=0.05
            Significance level(s) for control limits.
        labels : List[str], default=["Lower", "Expected", "Higher"]
            Labels for provider performance categories based on flags (-1, 0, 1).
        point_colors : List[str]
            Colors for provider points based on performance flag.
        point_shapes : List[str]
            Marker shapes for provider points based on performance flag.
        point_size : float, default=2.0
            Scaling factor for marker size.
        point_alpha : float, default=0.8
            Marker transparency.
        line_size : float, default=0.8
            Thickness for target and control limit lines.
        target_linestyle : str, default='--'
            Line style for the target reference line.
        font_size : float, default=12
            Base font size for labels and title.
        tick_label_size : float, default=10
            Font size for axis tick labels.
        cl_line_colors : str or List[str], optional
            Color(s) for the control limit lines. Defaults to "grey".
        cl_line_styles : str or List[str], optional
            Line style(s) for control limits. Defaults based on number of alphas.
        fill_color : str, default="#A6CEE3"
            Fill color for the area between the outermost control limits.
        fill_alpha : float, default=0.25
            Transparency of the control limit fill area.
        edge_color : str or None, default="grey"
            Edge color for scatter points.
        edge_linewidth : float, default=0.5
            Line width for scatter point edges.
        add_grid : bool, default=True
            Whether to add a background grid.
        grid_style : str, default=':'
            Line style for the grid.
        grid_alpha : float, default=0.6
            Transparency for the grid lines.
        remove_top_right_spines : bool, default=True
            Whether to remove the top and right axis lines.
        figure_size : Tuple[float, float], default=(8, 6)
            Figure size in inches.
        plot_title : str, default="Funnel Plot of Standardized Differences"
            Title for the plot.
        xlab : str, default="Precision (Group Size)"
            Label for the x-axis.
        ylab : str, default="Standardized Difference"
            Label for the y-axis.
        legend_location : str, default='best'
            Location string for the legend.
        """
        if self.coefficients_ is None or self.sigma_ is None or self.groups_ is None or self.group_sizes_ is None:
            raise ValueError("Model must be fitted and sigma estimated before plotting funnel plot.")
        if stdz != "indirect":
            warnings.warn("Funnel plot for LinearFixedEffectModel is primarily designed for 'indirect' standardized differences.")

        a_list = sorted([alpha] if isinstance(alpha, (float, int)) else alpha)
        alpha_test = min(a_list)

        sm_info = self.calculate_standardized_measures(stdz=stdz, null=null)
        if stdz not in sm_info or sm_info[stdz].empty:
            warnings.warn(f"No standardized measure data found for '{stdz}'. Cannot plot.")
            return
        df = sm_info[stdz].copy()
        if 'group_id' in df.columns: df.set_index('group_id', inplace=True)
        
        precision_map = pd.Series(self.group_sizes_, index=self.groups_)
        df["precision"] = df.index.map(precision_map)
        df.dropna(subset=['precision'], inplace=True)

        test_df = self.test(null=null, level=1.0 - alpha_test, alternative="two_sided")
        df = df.merge(test_df[['flag']], left_index=True, right_index=True, how='left')
        df["flag"] = df["flag"].fillna(0).astype(int)

        limits_list = []
        for a_val in a_list:
            z_val = norm.ppf(1 - a_val / 2)
            se_for_limits = self.sigma_ / np.sqrt(df["precision"])
            se_for_limits.replace([np.inf, -np.inf], np.nan, inplace=True)
            se_for_limits.fillna(0, inplace=True)
            control_lower = target - z_val * se_for_limits
            control_upper = target + z_val * se_for_limits
            limits_df_a = pd.DataFrame({
                "precision": df["precision"], 
                "control_lower": control_lower,
                "control_upper": control_upper, 
                "alpha": a_val
            }, index=df.index)
            limits_list.append(limits_df_a)
        limits_all_alphas = pd.concat(limits_list)

        # --- Render via shared funnel primitive --------------------------
        return _render_funnel(
            df=df,
            limits_df=limits_all_alphas,
            estimate_col=f"{stdz}_difference",
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
        use_flags: bool = True, 
        null: Union[str, float] = 'median',
        test_method: Optional[str] = None, # Added for consistency with LogisticFE
        **plot_kwargs
    ) -> None:
        """Plots provider fixed effects (gamma) using the plot_caterpillar helper function.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot. If None, all providers are included.
        level : float, default=0.95
            Confidence level for intervals.
        use_flags : bool, default=True
            Whether to color-code providers based on flags from the test method.
        null : str or float, default='median'
            Null hypothesis for gamma used for flagging. Can be 'median', 'mean', or a float.
        test_method : str, optional
             Test method used specifically for generating flags ('wald' is the only one for LinearFE's .test()).
             If None, defaults to 'wald'.
        **plot_kwargs
            Additional arguments passed to plot_caterpillar (e.g., plot_title, orientation).
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("Model must be fitted first.")

        # Get gamma CIs (always two-sided for plotting)
        # For LinearFixedEffectModel, test_method in calculate_CIs is implicitly Wald-like (t-dist)
        ci_results = self.calculate_confidence_intervals(
            providers=group_ids,
            level=level,
            option='gamma',
            alternative='two_sided'
        )
        if 'gamma_ci' not in ci_results or ci_results['gamma_ci'].empty:
            warnings.warn("No gamma CI data. Cannot plot.")
            return
        
        df_plot = ci_results['gamma_ci'] # Has 'group_id', 'gamma', 'lower', 'upper'

        flag_col_name = None
        if use_flags:
            flag_col_name = 'flag'
            # LinearFEModel's test method is t-test based (Wald-like)
            current_test_method = test_method if test_method else 'wald' # Default for LinearFE
            if current_test_method != 'wald':
                warnings.warn(f"LinearFixedEffectModel.test uses a t-test (Wald-like). test_method '{current_test_method}' for flagging will use this underlying test.")

            try:
                test_df = self.test(
                    providers=df_plot['group_id'].unique().tolist(),
                    level=level, 
                    null=null, 
                    alternative='two_sided'
                )
                # Merge flags using left_on='group_id' and right_index=True since test_df is indexed by provider IDs
                df_plot = df_plot.merge(test_df[['flag']], left_on='group_id', right_index=True, how='left')
                df_plot[flag_col_name] = df_plot[flag_col_name].fillna(0).astype(int)
            except Exception as e:
                warnings.warn(f"Could not generate flags. Plotting without flags. Error: {e}")
                flag_col_name = None
        
        gamma_vals = self.coefficients_["gamma"].flatten()
        if null == "median": gamma_null_val = np.median(gamma_vals)
        elif null == "mean": gamma_null_val = np.average(gamma_vals, weights=self.group_sizes_ if self.group_sizes_ is not None else None)
        else: gamma_null_val = float(null)

        # Default orientation is vertical (groups on Y, estimates on X)
        orientation = plot_kwargs.pop('orientation', 'vertical')
        if orientation == 'vertical':
            plot_kwargs.setdefault('xlab', 'Gamma Estimate (Fixed Effect)')
            plot_kwargs.setdefault('ylab', 'Provider')
        else: # horizontal
            plot_kwargs.setdefault('xlab', 'Provider')
            plot_kwargs.setdefault('ylab', 'Gamma Estimate (Fixed Effect)')

        # Use 'plot_title' instead of 'title' to match plot_caterpillar's parameter
        plot_kwargs.setdefault('plot_title', 'Provider Effects (Gamma)')
        plot_kwargs.setdefault('refline_value', gamma_null_val)
        plot_kwargs.setdefault('orientation', orientation)

        plot_caterpillar(
            df=df_plot, 
            estimate_col='gamma', 
            ci_lower_col='lower', 
            ci_upper_col='upper',
            group_col='group_id', 
            flag_col=flag_col_name, 
            **plot_kwargs
        )

    def plot_standardized_measures(
        self,
        group_ids=None, 
        level: float = 0.95, 
        stdz: str = 'indirect',
        measure: str = 'difference',
        use_flags: bool = True, null: Union[str, float] = 'median',
        test_method: Optional[str] = None,
        **plot_kwargs
    ) -> None:
        """Plots standardized differences using plot_caterpillar.
        For LinearFixedEffectModel, standardized measures are differences (gamma_i - gamma_null).

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot.
        level : float, default=0.95
            Confidence level for intervals.
        stdz : str, default='indirect'
            Standardization method ('indirect' or 'direct'). Both result in gamma_i - gamma_null.
        measure : str, default='difference'
            The measure to plot. For linear models, this is always 'difference'.
        use_flags : bool, default=True
            Whether to color-code providers based on flags from the gamma test method.
        null : str or float, default='median'
            Null hypothesis for gamma used for flagging and calculating the difference.
        test_method : str, optional
             Test method used specifically for generating flags. Defaults to 'wald' (t-test).
        **plot_kwargs
            Additional arguments passed to plot_caterpillar.
        """
        if self.coefficients_ is None: raise ValueError("Model must be fitted.")
        if measure != 'difference':
            warnings.warn("For LinearFixedEffectModel, standardized 'measure' is 'difference'.")
        
        # Get SM CIs (which are for the difference: gamma_i - gamma_null)
        ci_results = self.calculate_confidence_intervals(
            providers=group_ids,
            level=level,
            option='SM',
            stdz=stdz,
            null=null,
            alternative='two_sided'
        )
        ci_key = f"{stdz}_ci"
        if ci_key not in ci_results or ci_results[ci_key].empty:
            warnings.warn(f"No SM CI data for '{ci_key}'. Cannot plot.")
            return
        
        df_plot = ci_results[ci_key] # This df should have 'group_id', '{stdz}_difference', 'lower', 'upper'
        estimate_col_name = f"{stdz}_difference"
                
        if estimate_col_name not in df_plot.columns or 'lower' not in df_plot.columns or 'upper' not in df_plot.columns:
             raise ValueError(f"Required columns ('{estimate_col_name}', 'lower', 'upper') not found in SM CI results. Available: {df_plot.columns}")

        flag_col_name = None
        if use_flags:
            flag_col_name = 'flag'
            current_test_method = test_method if test_method else 'wald'
            if current_test_method != 'wald':
                 warnings.warn(f"LinearFixedEffectModel.test uses a t-test (Wald-like). test_method '{current_test_method}' for flagging will use this.")
            try:
                test_df = self.test(providers=df_plot['group_id'].unique().tolist(), 
                                    level=level, 
                                    null=null, 
                                    alternative='two_sided')
                # Merge using left_on='group_id' and right_index=True
                df_plot = df_plot.merge(test_df[['flag']], left_on='group_id', right_index=True, how='left')
                df_plot[flag_col_name] = df_plot[flag_col_name].fillna(0).astype(int)
            except Exception as e:
                warnings.warn(f"Could not generate flags. Plotting without flags. Error: {e}")
                flag_col_name = None

        orientation = plot_kwargs.pop('orientation', 'vertical')
        default_title = f"{stdz.capitalize()} Standardized Difference"
        if orientation == 'vertical':
            default_xlab = f"{stdz.capitalize()} Difference Estimate"
            default_ylab = "Provider"
        else:
            default_xlab = "Provider"
            default_ylab = f"{stdz.capitalize()} Difference Estimate"
        default_refline = 0.0

        plot_kwargs.setdefault('plot_title', default_title)
        plot_kwargs.setdefault('xlab', default_xlab)
        plot_kwargs.setdefault('ylab', default_ylab)
        plot_kwargs.setdefault('refline_value', default_refline)
        plot_kwargs.setdefault('orientation', orientation)

        plot_caterpillar(
            df=df_plot, 
            estimate_col=estimate_col_name,
            ci_lower_col='lower', 
            ci_upper_col='upper',
            group_col='group_id', 
            flag_col=flag_col_name, 
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

    def plot_residuals(
        self,
        figsize: tuple = _style.FIGURE_SIZE,
        point_color: str = _style.COLOR_PRIMARY,
        point_alpha: float = 0.6,
        edge_color: Optional[str] = _style.COLOR_NEUTRAL_GREY,
        edge_linewidth: float = _style.ERRORBAR_SIZE,
        point_size: float = 30,
        line_color: str = _style.COLOR_REFERENCE,
        line_style: str = _style.REFLINE_LINESTYLE,
        line_width: float = 1.5,
        xlabel: str = "Fitted Values",
        ylabel: str = "Residuals",
        title: str = "Residuals vs. Fitted Values",
        font_size: float = 12,
        tick_label_size: float = 10,
        add_grid: bool = True,
        grid_style: str = ':',
        grid_alpha: float = 0.6,
        remove_top_right_spines: bool = True
    ) -> None:
        """Plots residuals versus fitted values.

        This diagnostic plot helps assess model assumptions such as linearity
        and homoscedasticity (constant variance of errors). Ideally, the points
        should show no discernible pattern and be randomly scattered around the
        horizontal line at zero.

        Parameters
        ----------
        figsize : tuple, default=(8, 5)
            Size of the figure to create.
        point_color : str, default="#1F78B4"
            Color for the residual points.
        point_alpha : float, default=0.6
            Transparency level for the points.
        edge_color : str or None, default="grey"
            Edge color for the points. None means no edge.
        edge_linewidth : float, default=0.5
            Width of the point edges if `edge_color` is specified.
        point_size : float, default=30
            Size of the scatter plot markers.
        line_color : str, default="red"
            Color of the horizontal reference line at zero.
        line_style : str, default="--"
            Line style for the reference line.
        line_width : float, default=1.5
            Width of the reference line.
        xlabel : str, default="Fitted Values"
            Label for the x-axis.
        ylabel : str, default="Residuals"
            Label for the y-axis.
        title : str, default="Residuals vs. Fitted Values"
            Title of the plot.
        font_size : float, default=12
            Base font size for labels and title.
        tick_label_size : float, default=10
            Font size for axis tick labels.
        add_grid : bool, default=True
            Whether to add a background grid.
        grid_style : str, default=':'
            Line style for the grid.
        grid_alpha : float, default=0.6
            Transparency for the grid lines.
        remove_top_right_spines : bool, default=True
            Whether to remove the top and right plot spines.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.fitted_ is None or self.residuals_ is None:
            raise ValueError("Model must be fitted before plotting residuals.")

        # Create a new figure and axes for the plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            self.fitted_.flatten(), self.residuals_.flatten(),
            color=point_color, alpha=point_alpha,
            edgecolor=edge_color, linewidth=edge_linewidth if edge_color else 0,
            s=point_size
        )
        ax.axhline(0, color=line_color, linestyle=line_style, linewidth=line_width)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.set_title(title, fontsize=font_size + 2, pad=_style.TITLE_PAD)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if add_grid: ax.grid(True, linestyle=grid_style, alpha=grid_alpha, color='lightgrey')
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(_style.SPINE_WIDTH); ax.spines['bottom'].set_linewidth(_style.SPINE_WIDTH)

        plt.tight_layout()
        plt.show()
        # No return value

    def plot_qq(
        self,
        figsize: tuple = _style.FIGURE_SIZE,
        title: str = "Normal Q-Q Plot of Residuals",
        xlabel: str = "Theoretical Normal Quantiles",
        ylabel: str = "Ordered Residuals",
        font_size: float = 12,
        tick_label_size: float = 10,
        point_color: str = _style.COLOR_PRIMARY,
        line_color: str = _style.COLOR_REFERENCE,
        add_grid: bool = False,
        remove_top_right_spines: bool = True
    ) -> None:
        """Creates a Q-Q plot of residuals against a Normal distribution.

        This plot helps assess the assumption of normally distributed errors,
        which is important for the validity of t-tests and confidence intervals
        in linear models. Points falling approximately along the diagonal line
        suggest normality.

        Parameters
        ----------
        figsize : tuple, default=(7, 6)
            Size of the figure to create.
        title : str, default="Normal Q-Q Plot of Residuals"
            Title of the plot.
        xlabel : str, default="Theoretical Normal Quantiles"
            Label for the x-axis.
        ylabel : str, default="Ordered Residuals"
            Label for the y-axis.
        font_size : float, default=12
            Base font size for labels and title.
        tick_label_size : float, default=10
            Font size for axis tick labels.
        point_color : str, default="#1F78B4"
            Color for the points representing residuals.
        line_color : str, default="red"
            Color for the diagonal reference line.
        add_grid : bool, default=False
            Whether to add a background grid.
        remove_top_right_spines : bool, default=True
            Whether to remove the top and right plot spines.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.residuals_ is None:
            raise ValueError("Model must be fitted before plotting QQ plot.")

        # Create a new figure and axes for the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Ensure residuals are 1D array
        residuals_flat = self.residuals_.flatten()

        # Create the Q-Q plot using scipy.stats.probplot
        try:
            (osm, osr), (slope, intercept, r_sq) = probplot(residuals_flat, dist="norm", fit=True, plot=None)
            # Plot the ordered residuals against theoretical quantiles
            ax.plot(osm, osr, 'o', color=point_color, markersize=5, alpha=0.7)
            # Plot the fitted line
            ax.plot(osm, slope * osm + intercept, color=line_color, linestyle='-', linewidth=1.5)
        except Exception as e:
            warnings.warn(f"Could not generate Q-Q plot data, possibly due to issues with residuals: {e}")
            # Plot empty axes as placeholder if data generation fails
            pass

        ax.set_title(title, fontsize=font_size + 2, pad=_style.TITLE_PAD)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if add_grid: ax.grid(True, linestyle=':', alpha=0.6, color='lightgrey')
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(_style.SPINE_WIDTH); ax.spines['bottom'].set_linewidth(_style.SPINE_WIDTH)

        plt.tight_layout()
        plt.show()
