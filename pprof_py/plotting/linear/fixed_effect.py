"""Plotting for `LinearFixedEffectModel`: funnel plots, provider-effect
and standardized-measure caterpillar plots, coefficient forest plot, and
residual/Q-Q diagnostic plots. Mixed into the model class so that
`models/linear/fixed_effect.py` can stay focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union


import numpy as np
import pandas as pd
from scipy.stats import t, probplot

from ...plotting import style as _style


class _LinearFEPlottingHost(Protocol):
    """Attribute contract that `LinearFixedEffectPlottingMixin` expects from its
    host class (`LinearFixedEffectModel`).
    """
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    residuals_: Optional[np.ndarray]
    sigma_: Optional[float]
    provider_ids_: Optional[np.ndarray]
    provider_sizes_: Optional[np.ndarray]
    covariate_names_: list

    def _check_is_fitted(self) -> None: ...
    def calculate_confidence_intervals(self, **kwargs: Any) -> Any: ...
    def calculate_standardized_measures(self, **kwargs: Any) -> dict: ...
    def test(self, **kwargs: Any) -> Any: ...


class LinearFixedEffectPlottingMixin:
    """Funnel, provider-effect, standardized-measure, coefficient-forest,
    and residual/Q-Q plots for `LinearFixedEffectModel`."""

    def plot_funnel(self, reference: Union[str, float] = "median", alpha: Union[float, List[float]] = 0.05,
                    **kwargs: Any):
        """Funnel plot whose control limits come from the same test as the flags.

        Delegates to :func:`pprof_py.presentation.funnel` and returns its
        :class:`~pprof_py.presentation.FigureResult` (``fig, ax = model.plot_funnel()`` still works). ``alpha`` sets
        the levels of the limit curves; ``save_path``, ``theme``, ``size``, ``title`` and ``highlight`` are passed on.
        Styling keywords and ``target=`` were removed in 0.7.0 and raise a ``TypeError``.

        The funnel shows the provider effects with the limits of their Student-t Wald test; ``stdz=`` was removed
        in 0.7.0.
        """
        from .._delegates import funnel_delegate
        return funnel_delegate(self, "plot_funnel", test_kwargs={"reference": reference}, alpha=alpha, kwargs=kwargs)

    def plot_provider_effects(self, group_ids=None, level: float = 0.95, use_flags: bool = True,
                              reference: Union[str, float] = "median", test_method: Optional[str] = None,
                              **plot_kwargs: Any):
        """Interval plot of the provider effects with the intervals of their own test, and a volume panel.

        Delegates to :func:`pprof_py.presentation.caterpillar` and returns its
        :class:`~pprof_py.presentation.FigureResult`. ``group_ids`` selects providers; ``save_path``, ``theme``,
        ``size``, ``title`` and ``highlight`` are passed on. Styling keywords and ``use_flags=False`` no longer have
        an effect (``DeprecationWarning``; ``TypeError`` from 0.7.0).
        """
        from .._delegates import caterpillar_delegate
        return caterpillar_delegate(self, "plot_provider_effects", use_flags=use_flags, kwargs=plot_kwargs,
                                    ignored=[] if test_method is None else ["test_method= (the test is the Wald test)"],
                                    test_kwargs={"providers": group_ids, "level": level, "reference": reference})

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
        import matplotlib.pyplot as plt
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
            xerr = np.vstack([
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"]
            ])
        else:
            x_vals, y_vals = positions, coef_df["estimate"]
            yerr = np.vstack([
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"]
            ])

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
        import matplotlib.pyplot as plt
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
        import matplotlib.pyplot as plt
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
