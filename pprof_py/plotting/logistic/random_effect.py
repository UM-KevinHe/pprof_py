"""Plotting for ``LogisticRandomEffectModel``: funnel plots,
provider-effect and standardized-measure caterpillar plots, and a
coefficient forest plot.  Mixed into the model class so that
``models/logistic/random_effect.py`` can stay focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import norm

from ...plotting import plot_caterpillar
from ...plotting.funnel import plot_funnel as _render_funnel
from ...plotting import style as _style


# ---------------------------------------------------------------------------
# Protocol: attributes the mixin expects from the host model
# ---------------------------------------------------------------------------

class _LogisticREPlottingHost(Protocol):
    """Attribute contract that ``RandomEffectPlottingMixin`` expects from
    ``LogisticRandomEffectModel``."""
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    residuals_: Optional[np.ndarray]
    groups_: Optional[Dict[str, np.ndarray]]
    group_sizes_: Optional[Dict[str, np.ndarray]]
    xbeta_: Optional[np.ndarray]
    covariate_names_: Optional[list]
    outcome_: Optional[np.ndarray]
    _group_vars: Optional[list]
    _provider_var: Optional[str]
    _group_indices: Optional[list]
    _group_labels: Optional[list]
    _n_groups: Optional[list]
    _y: Optional[np.ndarray]

    def _check_is_fitted(self) -> None: ...
    def get_random_effects(self, var: Optional[str] = None) -> pd.Series: ...
    def _get_posterior_se(self, var: Optional[str] = None) -> pd.Series: ...
    def calculate_confidence_intervals(self, **kwargs: Any) -> Any: ...
    def calculate_standardized_measures(self, **kwargs: Any) -> dict: ...
    def test(self, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class RandomEffectPlottingMixin:
    """Funnel, provider-effect, standardized-measure, and coefficient-forest
    plots for ``LogisticRandomEffectModel``."""

    # ---- private helpers ------------------------------------------------

    def _resolve_group_var(self, group_var: Optional[str] = None) -> str:
        """Resolve and validate the grouping variable."""
        if group_var is None:
            if len(self._group_vars) == 1:
                return self._group_vars[0]
            raise ValueError(
                f"Specify group_var; available: {self._group_vars}"
            )
        if group_var not in self._group_vars:
            raise ValueError(
                f"Unknown group_var '{group_var}'; "
                f"available: {self._group_vars}"
            )
        return group_var

    def _resolve_reference(self, reference, group_var: str) -> float:
        """Convert 'median'/'mean'/numeric *reference* to a float BLUP value."""
        blups = self.get_random_effects(group_var)
        if reference == "median":
            return float(np.median(blups.values))
        if reference == "mean":
            return float(np.mean(blups.values))
        if isinstance(reference, (int, float)):
            return float(reference)
        raise ValueError(
            "null must be 'median', 'mean', or a numeric value."
        )

    # ==================================================================
    # 1. Funnel plot  (indirect standardized ratio O/E)
    # ==================================================================

    def plot_funnel(
        self,
        test_method: str = "wald",
        reference: Union[str, float] = "median",
        target: float = 1.0,
        alpha: Union[float, List[float]] = 0.05,
        labels: List[str] = _style.FLAG_LABELS,
        point_colors: List[str] = [
            _style.COLOR_WARNING,
            _style.COLOR_PRIMARY,
            _style.COLOR_SECONDARY,
        ],
        point_shapes: List[str] = ["v", "o", "^"],
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
        plot_title: str = "Funnel Plot (Indirect Standardized Ratio)",
        xlab: str = "Expected Count (Precision)",
        ylab: str = "Indirectly Standardized Ratio (O/E)",
        legend_location: str = "best",
    ) -> None:
        """Funnel plot of the indirect standardized ratio (O/E) vs expected
        count.  Control limits use the Poisson approximation
        ``target +/- z / sqrt(E)``.

        Parameters
        ----------
        test_method : str, default 'wald'
            Test method passed to ``self.test()`` for flagging.
        reference : str or float, default 'median'
            Baseline BLUP value for expected counts and flagging.
        target : float, default 1.0
            Reference value for the ratio (centre of the funnel).
        alpha : float or list of float, default 0.05
            Significance level(s) for control-limit bands.
        """
        group_var = self._provider_var
        self._check_is_fitted()
        group_var = self._resolve_group_var(group_var)
        gamma_null = self._resolve_reference(reference, group_var)

        a_list = sorted(
            [alpha] if isinstance(alpha, (float, int)) else alpha
        )
        alpha_test = min(a_list)

        # Standardized measures
        sm = self.calculate_standardized_measures(
            stdz="indirect", reference=gamma_null,
        )
        if "indirect" not in sm or sm["indirect"].empty:
            warnings.warn("No indirect SM data. Cannot plot.")
            return
        df = sm["indirect"].copy()
        if "provider_id" in df.columns:
            df.set_index("provider_id", inplace=True)

        # Precision = expected count
        df["precision"] = df["expected"]
        df.dropna(subset=["precision"], inplace=True)

        # Flags from test()
        test_df = self.test(
            test_method=test_method,
            reference=gamma_null,
            level=1.0 - alpha_test,
            alternative="two_sided",
        )
        df = df.merge(
            test_df[["flag"]], left_index=True, right_index=True, how="left",
        )
        df["flag"] = df["flag"].fillna(0).astype(int)

        # Poisson-approximation control limits: target +/- z / sqrt(E)
        limits_list = []
        for a_val in a_list:
            z_val = norm.ppf(1.0 - a_val / 2.0)
            se_curve = np.where(
                df["precision"] > 0,
                1.0 / np.sqrt(df["precision"]),
                0.0,
            )
            control_lower = target - z_val * se_curve
            control_upper = target + z_val * se_curve
            limits_df_a = pd.DataFrame(
                {
                    "precision": df["precision"],
                    "control_lower": control_lower,
                    "control_upper": control_upper,
                    "alpha": a_val,
                },
                index=df.index,
            )
            limits_list.append(limits_df_a)
        limits_all = pd.concat(limits_list)

        return _render_funnel(
            df=df,
            limits_df=limits_all,
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

    # ==================================================================
    # 2. Provider effects caterpillar  (BLUPs on log-odds scale)
    # ==================================================================

    def plot_provider_effects(
        self,
        group_ids=None,
        level: float = 0.95,
        use_flags: bool = True,
        reference: Union[str, float] = 0,
        test_method: str = "wald",
        **plot_kwargs,
    ) -> None:
        """Caterpillar plot of provider random effects (BLUPs) with CIs.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs to plot.
        level : float, default 0.95
            Confidence level for intervals.
        use_flags : bool, default True
            Colour-code providers using ``self.test()`` flags.
        reference : str or float, default 0
            Null hypothesis for flagging (log-odds scale).
        test_method : str, default 'wald'
            Test method for flagging.
        **plot_kwargs
            Forwarded to ``plot_caterpillar``.
        """
        group_var = self._provider_var
        self._check_is_fitted()
        group_var = self._resolve_group_var(group_var)
        null_val = self._resolve_reference(reference, group_var) if isinstance(reference, str) else float(reference)

        ci = self.calculate_confidence_intervals(
            providers=group_ids,
            level=level,
            option="alpha",
            alternative="two_sided",
        )
        if "alpha_ci" not in ci or ci["alpha_ci"].empty:
            warnings.warn("No alpha CI data. Cannot plot.")
            return
        df_plot = ci["alpha_ci"].copy()

        # Flags
        flag_col_name = None
        if use_flags:
            flag_col_name = "flag"
            try:
                test_df = self.test(
                    providers=(
                        df_plot["provider_id"].unique().tolist()
                        if "provider_id" in df_plot.columns
                        else None
                    ),
                    level=level,
                    test_method=test_method,
                    reference=null_val,
                    alternative="two_sided",
                )
                df_plot = df_plot.merge(
                    test_df[["flag"]],
                    left_on="provider_id",
                    right_index=True,
                    how="left",
                )
                df_plot[flag_col_name] = (
                    df_plot[flag_col_name].fillna(0).astype(int)
                )
            except Exception as exc:
                warnings.warn(
                    f"Could not generate flags: {exc}. "
                    "Plotting without flags."
                )
                flag_col_name = None

        orientation = plot_kwargs.pop("orientation", "vertical")
        if orientation == "vertical":
            plot_kwargs.setdefault("xlab", "BLUP (log-odds scale)")
            plot_kwargs.setdefault("ylab", "Provider")
        else:
            plot_kwargs.setdefault("xlab", "Provider")
            plot_kwargs.setdefault("ylab", "BLUP (log-odds scale)")

        plot_kwargs.setdefault("plot_title", "Provider Effects (BLUPs)")
        plot_kwargs.setdefault("refline_value", null_val)
        plot_kwargs.setdefault("orientation", orientation)

        plot_caterpillar(
            df=df_plot,
            estimate_col="alpha",
            ci_lower_col="alpha_lower",
            ci_upper_col="alpha_upper",
            group_col="provider_id",
            flag_col=flag_col_name,
            **plot_kwargs,
        )

    # ==================================================================
    # 3. Standardized-measure caterpillar  (ratio or rate with CIs)
    # ==================================================================

    def plot_standardized_measures(
        self,
        group_ids=None,
        level: float = 0.95,
        stdz: str = "indirect",
        measure: str = "ratio",
        use_flags: bool = True,
        reference: Union[str, float] = "median",
        test_method: str = "wald",
        **plot_kwargs,
    ) -> None:
        """Caterpillar plot of standardized measures (ratio or rate) with CIs.

        Parameters
        ----------
        group_ids : list or np.ndarray, optional
            Subset of provider IDs.
        level : float, default 0.95
            Confidence level.
        stdz : {'indirect', 'direct'}, default 'indirect'
            Standardization method.
        measure : {'ratio', 'rate'}, default 'ratio'
            Which measure to plot.
        use_flags : bool, default True
            Colour-code using flags.
        reference : str or float, default 'median'
            Null BLUP for flagging.
        test_method : str, default 'wald'
            Test method for flagging.
        **plot_kwargs
            Forwarded to ``plot_caterpillar``.
        """
        group_var = self._provider_var
        self._check_is_fitted()
        group_var = self._resolve_group_var(group_var)
        null_val = self._resolve_reference(reference, group_var)

        ci_key = f"{stdz}_{measure}"
        ci = self.calculate_confidence_intervals(
            providers=group_ids,
            level=level,
            option="SM",
            stdz=stdz,
            reference=null_val,
            measure=measure,
            alternative="two_sided",
        )
        if ci_key not in ci or ci[ci_key].empty:
            warnings.warn(
                f"No SM CI data for '{ci_key}'. Cannot plot."
            )
            return
        df_plot = ci[ci_key].copy()
        estimate_col = ci_key  # e.g. 'indirect_ratio'

        if (
            estimate_col not in df_plot.columns
            or "lower" not in df_plot.columns
            or "upper" not in df_plot.columns
        ):
            raise ValueError(
                f"Required columns ('{estimate_col}', 'lower', 'upper') "
                f"not in CI results. Got: {list(df_plot.columns)}"
            )

        # Flags
        flag_col_name = None
        if use_flags:
            flag_col_name = "flag"
            try:
                test_df = self.test(
                    providers=(
                        df_plot["provider_id"].unique().tolist()
                        if "provider_id" in df_plot.columns
                        else None
                    ),
                    level=level,
                    test_method=test_method,
                    reference=null_val,
                    alternative="two_sided",
                )
                df_plot = df_plot.merge(
                    test_df[["flag"]],
                    left_on="provider_id",
                    right_index=True,
                    how="left",
                )
                df_plot[flag_col_name] = (
                    df_plot[flag_col_name].fillna(0).astype(int)
                )
            except Exception as exc:
                warnings.warn(
                    f"Could not generate flags: {exc}. "
                    "Plotting without flags."
                )
                flag_col_name = None

        # Defaults
        refline = 1.0 if measure == "ratio" else None
        orientation = plot_kwargs.pop("orientation", "vertical")
        nice = f"{stdz.capitalize()} {measure.capitalize()}"
        if orientation == "vertical":
            plot_kwargs.setdefault("xlab", nice)
            plot_kwargs.setdefault("ylab", "Provider")
        else:
            plot_kwargs.setdefault("xlab", "Provider")
            plot_kwargs.setdefault("ylab", nice)

        plot_kwargs.setdefault("plot_title", nice)
        plot_kwargs.setdefault("refline_value", refline)
        plot_kwargs.setdefault("orientation", orientation)

        plot_caterpillar(
            df=df_plot,
            estimate_col=estimate_col,
            ci_lower_col="lower",
            ci_upper_col="upper",
            group_col="provider_id",
            flag_col=flag_col_name,
            **plot_kwargs,
        )

    # ==================================================================
    # 4. Coefficient forest plot  (fixed effects, z-based CIs)
    # ==================================================================

    def plot_coefficient_forest(
        self,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        level: float = 0.95,
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
        xlab: str = "Coefficient Estimate (log-odds)",
        ylab: str = "Covariate",
        save_path: Optional[str] = None,
        dpi: int = _style.SAVE_DPI,
    ) -> None:
        """Forest plot of fixed-effect coefficients with Wald CIs.

        Uses the normal distribution (z-based) for intervals, consistent
        with GLMM inference.

        Parameters
        ----------
        orientation : {'vertical', 'horizontal'}, default 'vertical'
            Layout orientation.
        level : float, default 0.95
            Confidence level.
        refline_value : float or None, default 0.0
            Reference line value.  ``None`` disables.
        save_path : str or None
            If given, save figure to this path instead of showing.
        dpi : int, default 300
            Resolution for saving.
        """
        self._check_is_fitted()
        if orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        beta = self.coefficients_["beta"]
        vcov = self.variances_["beta"]
        se = np.sqrt(np.maximum(np.diag(vcov.to_numpy()), 0.0))

        z_crit = norm.ppf(1.0 - (1.0 - level) / 2.0)
        lower = beta.values - z_crit * se
        upper = beta.values + z_crit * se

        coef_df = (
            pd.DataFrame(
                {
                    "covariate": beta.index,
                    "estimate": beta.values,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
            .sort_values("estimate")
            .reset_index(drop=True)
        )

        n = len(coef_df)
        positions = np.arange(n)

        if orientation == "vertical":
            x_vals, y_vals = coef_df["estimate"], positions
            xerr = [
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"],
            ]
        else:
            x_vals, y_vals = positions, coef_df["estimate"]
            yerr = [
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"],
            ]

        fig, ax = plt.subplots(figsize=figure_size)

        err_kw = dict(
            fmt="o",
            color=point_color,
            ecolor=error_color,
            capsize=capsize,
            markersize=point_size * 30,
            alpha=point_alpha,
            linewidth=edge_linewidth if edge_color else 0,
            markeredgecolor=edge_color,
        )
        if orientation == "vertical":
            ax.errorbar(x_vals, y_vals, xerr=xerr, **err_kw)
        else:
            ax.errorbar(x_vals, y_vals, yerr=yerr, **err_kw)

        if refline_value is not None:
            if orientation == "vertical":
                ax.axvline(
                    refline_value, color=line_color,
                    linestyle=line_style, linewidth=line_size,
                )
            else:
                ax.axhline(
                    refline_value, color=line_color,
                    linestyle=line_style, linewidth=line_size,
                )

        if orientation == "vertical":
            ax.set_xlabel(xlab, fontsize=font_size)
            ax.set_ylabel(ylab, fontsize=font_size)
            ax.set_yticks(positions)
            ax.set_yticklabels(coef_df["covariate"], fontsize=tick_label_size)
            ax.tick_params(axis="x", labelsize=tick_label_size)
            if add_grid:
                ax.grid(
                    True, axis="x", linestyle=grid_style,
                    alpha=grid_alpha, color=_style.GRID_COLOR,
                )
        else:
            ax.set_xlabel(ylab, fontsize=font_size)
            ax.set_ylabel(xlab, fontsize=font_size)
            ax.set_xticks(positions)
            ax.set_xticklabels(
                coef_df["covariate"], rotation=45, ha="right",
                fontsize=tick_label_size,
            )
            ax.tick_params(axis="y", labelsize=tick_label_size)
            if add_grid:
                ax.grid(
                    True, axis="y", linestyle=grid_style,
                    alpha=grid_alpha, color=_style.GRID_COLOR,
                )

        if remove_top_right_spines:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(_style.SPINE_WIDTH)
        ax.spines["bottom"].set_linewidth(_style.SPINE_WIDTH)

        ax.set_title(
            plot_title, fontsize=font_size + 2, pad=_style.TITLE_PAD,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
