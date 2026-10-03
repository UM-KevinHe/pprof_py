"""Plotting for ``LogisticRandomEffectModel``: funnel plots,
provider-effect and standardized-measure caterpillar plots, and a
coefficient forest plot.  Mixed into the model class so that
``models/logistic/random_effect.py`` can stay focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union


import numpy as np
import pandas as pd
from scipy.stats import norm

from ...plotting import style as _style


# ---------------------------------------------------------------------------
# Protocol: attributes the mixin expects from the host model
# ---------------------------------------------------------------------------

class _LogisticREPlottingHost(Protocol):
    """Attribute contract that ``LogisticRandomEffectPlottingMixin`` expects from
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

class LogisticRandomEffectPlottingMixin:
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

    def plot_funnel(self, test_method: str = "poibin_exact", reference: Union[str, float] = "median",
                    alpha: Union[float, List[float]] = 0.05, **kwargs: Any):
        """Funnel plot whose control limits come from the same test as the flags.

        Delegates to :func:`pprof_py.presentation.funnel` and returns its
        :class:`~pprof_py.presentation.FigureResult` (``fig, ax = model.plot_funnel()`` still works). ``alpha`` sets
        the levels of the limit curves; ``save_path``, ``theme``, ``size``, ``title`` and ``highlight`` are passed on.
        Styling keywords and ``target=`` were removed in 0.7.0 and raise a ``TypeError``.

        Random-effect funnels show count tests only (ADR-004): the default is the exact count test,
        ``"poibin_exact"``; ``"wald"``, whose shrunken estimates have no funnel that agrees with its flags, raises a
        ``ValueError`` (removed in 0.7.0).
        """
        from .._delegates import funnel_delegate
        if test_method == "wald":
            raise ValueError("plot_funnel(): a Wald test of shrunken estimates has no funnel that agrees with its flags "
                             "(ADR-004); use the default test_method='poibin_exact' or 'exact'. test_method='wald' was "
                             "removed in 0.7.0.")
        return funnel_delegate(self, "plot_funnel", test_kwargs={"test_method": test_method, "reference": reference},
                               alpha=alpha, kwargs=kwargs)

    def plot_provider_effects(self, group_ids=None, level: float = 0.95, use_flags: bool = True,
                              reference: Union[str, float] = 0, test_method: str = "wald", **plot_kwargs: Any):
        """Interval plot of the provider effects with the intervals of their own test, and a volume panel.

        Delegates to :func:`pprof_py.presentation.caterpillar` and returns its
        :class:`~pprof_py.presentation.FigureResult`. ``group_ids`` selects providers; ``save_path``, ``theme``,
        ``size``, ``title`` and ``highlight`` are passed on. Styling keywords and ``use_flags=False`` were removed in
        0.7.0 and raise a ``TypeError``.
        """
        from .._delegates import caterpillar_delegate
        return caterpillar_delegate(self, "plot_provider_effects", use_flags=use_flags, kwargs=plot_kwargs,
                                    test_kwargs={"providers": group_ids, "level": level, "test_method": test_method,
                                                 "reference": reference})

    # ==================================================================
    # 2. Provider effects caterpillar  (BLUPs on log-odds scale)
    # ==================================================================

    # ==================================================================
    # 3. Standardized-measure caterpillar  (ratio or rate with CIs)
    # ==================================================================

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
        import matplotlib.pyplot as plt
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
            xerr = np.vstack([
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"],
            ])
        else:
            x_vals, y_vals = positions, coef_df["estimate"]
            yerr = np.vstack([
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"],
            ])

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
