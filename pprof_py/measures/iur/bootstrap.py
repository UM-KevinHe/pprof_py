"""Bootstrap-based Inter-Unit Reliability (IUR) estimation."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from ...base import ProviderModel

from ._core import (
    ArrayLike,
    _compute_decile_table,
    _iur_decomposition,
)
from ._sampling import _stratified_bootstrap
from .measures import ratio_measure


class BootstrapIUR(ProviderModel):
    """Bootstrap-based Inter-Unit Reliability estimation.

    Estimates the fraction of observed variation in a group-level
    measure that reflects true differences between groups (signal)
    versus within-group sampling noise, using stratified bootstrap
    resampling.

    Parameters
    ----------
    n_boot : int, default=100
        Number of bootstrap iterations.
    measure_fn : callable or None, default=None
        Function with signature ``(obs, exp, groups) -> measures``
        that computes a group-level measure from patient-level data.
        If *None*, uses :func:`~coxph.iur.measures.ratio_measure`
        (``sum(obs) / sum(exp)`` per group).
    seed : int, default=123
        Random seed for reproducibility.

    Attributes
    ----------
    iur_ : float
        Overall (national) IUR.
    n_groups_ : int
        Number of groups.
    iur_groups_ : ndarray of shape (n_groups_,)
        Per-group IUR.
    s2_between_ : float
        Between-group variance component (signal).
    s2_within_ : float
        Within-group variance component (noise), scaled by
        ``n_prime_``.
    n_prime_ : float
        Effective sample size per group.
    measure_ : ndarray of shape (n_groups_,)
        Original (non-bootstrapped) measure per group.
    group_labels_ : ndarray of shape (n_groups_,)
        Sorted unique group identifiers.
    group_sizes_ : ndarray of shape (n_groups_,)
        Number of observations per group.
    measure_bootstrap_ : ndarray of shape (n_groups_, n_boot)
        Bootstrap measure matrix (groups × iterations).

    Examples
    --------
    >>> from pprof_py.measures.iur import BootstrapIUR
    >>> model = BootstrapIUR(n_boot=100, seed=42)
    >>> model.fit(obs, exp, groups)
    >>> print(f"IUR = {model.iur_:.4f}")
    >>> model.decile_table()
    """

    def __init__(
        self,
        n_boot: int = 100,
        measure_fn: Optional[Callable] = None,
        seed: int = 123,
    ):
        self.n_boot = n_boot
        self.measure_fn = measure_fn
        self.seed = seed

    def fit(
        self,
        obs: ArrayLike,
        exp: ArrayLike,
        groups: ArrayLike,
    ) -> "BootstrapIUR":
        """Compute IUR via stratified bootstrap resampling.

        Data is sorted internally by *groups*; the caller need not
        pre-sort.

        Parameters
        ----------
        obs : array-like of shape (n_samples,)
            Observed outcomes (numerator contributions).
        exp : array-like of shape (n_samples,)
            Expected outcomes (denominator contributions).
        groups : array-like of shape (n_samples,)
            Group (facility) identifiers.

        Returns
        -------
        self
            Fitted estimator.
        """
        obs = np.asarray(obs, dtype=np.float64)
        exp = np.asarray(exp, dtype=np.float64)
        groups = np.asarray(groups)

        if not (len(obs) == len(exp) == len(groups)):
            raise ValueError(
                "obs, exp, and groups must have the same length."
            )

        # Sort by group (stable sort preserves within-group order)
        sort_idx = np.argsort(groups, kind="stable")
        obs = obs[sort_idx]
        exp = exp[sort_idx]
        groups = groups[sort_idx]

        # Group metadata
        unique_groups, group_counts = np.unique(
            groups, return_counts=True
        )
        self.group_labels_ = unique_groups
        self.group_sizes_ = group_counts
        self.n_groups_ = len(unique_groups)

        # Resolve measure function
        fn = self.measure_fn if self.measure_fn is not None else ratio_measure

        # Original measure
        self.measure_ = fn(obs, exp, groups)

        # ── Bootstrap ─────────────────────────────────────────────
        rng = np.random.default_rng(self.seed)
        measure_boot = np.empty((self.n_groups_, self.n_boot))

        for b in range(self.n_boot):
            idx = _stratified_bootstrap(group_counts, rng)
            measure_boot[:, b] = fn(obs[idx], exp[idx], groups[idx])

        self.measure_bootstrap_ = measure_boot

        # ── Variance decomposition ────────────────────────────────
        within_vars = measure_boot.var(axis=1, ddof=1)
        decomp = _iur_decomposition(
            group_counts, within_vars, self.measure_
        )

        self.iur_ = decomp.iur
        self.iur_groups_ = decomp.iur_groups
        self.s2_between_ = decomp.s2_between
        self.s2_within_ = decomp.s2_within
        self.n_prime_ = decomp.n_prime

        return self

    # ── Post-fit methods ──────────────────────────────────────────

    def stratified_iur(
        self,
        stratify_var: Optional[ArrayLike] = None,
        stratify_cut: Optional[ArrayLike] = None,
    ) -> pd.DataFrame:
        """Compute IUR for subgroups of groups.

        Parameters
        ----------
        stratify_var : array-like of shape (n_groups_,) or None
            Variable to stratify groups by (already aggregated to
            group level, e.g., summed patient-years per facility).
            If *None*, group sizes are used.
        stratify_cut : array-like or None
            Cut-points defining subgroup boundaries (right-closed).
            If *None*, decile boundaries of *stratify_var* are used.

        Returns
        -------
        result : DataFrame
            Rows: Total, then one per subgroup.  Columns: ``Group``,
            ``IUR``, ``Group size``.
        """
        self._require_fitted("iur_")

        if stratify_var is None:
            stratify_var = self.group_sizes_.astype(np.float64)
        else:
            stratify_var = np.asarray(stratify_var, dtype=np.float64)

        if stratify_cut is None:
            stratify_cut = np.quantile(
                stratify_var, np.arange(0.1, 1.0, 0.1)
            )
        stratify_cut = np.asarray(stratify_cut, dtype=np.float64)

        # Build boolean masks for each subgroup
        masks = []
        for i in range(len(stratify_cut)):
            if i == 0:
                masks.append(stratify_var <= stratify_cut[i])
            else:
                masks.append(
                    (stratify_var > stratify_cut[i - 1])
                    & (stratify_var <= stratify_cut[i])
                )
        masks.append(stratify_var > stratify_cut[-1])

        # Total row
        rows = [
            {
                "Group": "Total",
                "IUR": self.iur_,
                "Group size": self.n_groups_,
            }
        ]

        for i, mask in enumerate(masks):
            n_in = int(mask.sum())
            if n_in == 0:
                continue

            decomp = _iur_decomposition(
                self.group_sizes_[mask],
                self.measure_bootstrap_[mask, :].var(axis=1, ddof=1),
                self.measure_[mask],
            )

            # Label
            lo = stratify_var[mask].min()
            hi = stratify_var[mask].max()
            if i == 0:
                label = f"{lo:.0f}<=size<={stratify_cut[0]:.0f}"
            elif i == len(masks) - 1:
                label = f"{stratify_cut[-1]:.0f}<size<={hi:.0f}"
            else:
                label = (
                    f"{stratify_cut[i - 1]:.0f}<size<="
                    f"{stratify_cut[i]:.0f}"
                )
            rows.append(
                {
                    "Group": label,
                    "IUR": decomp.iur,
                    "Group size": decomp.n_groups,
                }
            )

        return pd.DataFrame(rows)

    def decile_table(
        self,
        stratify_var: Optional[ArrayLike] = None,
        n_quantiles: int = 10,
    ) -> pd.DataFrame:
        """Compute IUR at representative group sizes.

        Uses the fitted variance components to compute the expected
        reliability for groups of different sizes (min, decile
        averages, max).

        Parameters
        ----------
        stratify_var : array-like of shape (n_groups_,) or None
            Variable to determine size grouping.  If *None*, group
            sizes are used.
        n_quantiles : int, default=10
            Number of quantile groups.

        Returns
        -------
        table : DataFrame
            Single-row frame with columns ``min``, ``decile 1`` …
            ``decile K``, ``max``.
        """
        self._require_fitted("iur_")

        sizes = (
            np.asarray(stratify_var, dtype=np.float64)
            if stratify_var is not None
            else self.group_sizes_.astype(np.float64)
        )
        return _compute_decile_table(
            sizes,
            self.s2_between_,
            self.s2_within_,
            n_quantiles=n_quantiles,
        )
