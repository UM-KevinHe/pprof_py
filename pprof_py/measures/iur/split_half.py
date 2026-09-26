"""Split-half correlation-based Inter-Unit Reliability estimation."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from ...base import ProviderModel
from ...utils.metrics import cohen_kappa_score

from ._core import ArrayLike
from ._sampling import _split_half_sample
from .measures import ratio_measure


def _correlation_to_iur(corr: float) -> float:
    """Convert a split-half correlation to IUR (Spearman-Brown formula)."""
    denom = 1.0 + corr
    if denom == 0:
        return 0.0
    return 2.0 * corr / denom


def _categorise(values: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Assign ordinal category labels based on quantile breaks."""
    breaks = np.quantile(values, probs)
    full_breaks = np.concatenate(
        [[values.min() - 1], breaks, [values.max() + 1]]
    )
    return np.digitize(values, full_breaks[1:], right=True) + 1


class SplitHalfIUR(ProviderModel):
    """Split-half correlation-based IUR estimation.

    Estimates reliability by repeatedly splitting each group's
    observations into two random halves, computing the measure on
    each half independently, and measuring the correlation between
    halves across groups.  Six agreement metrics are computed:

    * **Categorised** (ordinal bins): weighted Cohen's kappa,
      Kendall tau, and Spearman rho.
    * **Continuous**: Pearson, Kendall tau, and Spearman rho.

    Each correlation is converted to IUR via the Spearman-Brown
    formula: ``IUR = 2r / (1 + r)``.

    Parameters
    ----------
    n_iter : int, default=10
        Number of split-half iterations.
    category_probs : array-like or None, default=None
        Quantile probabilities used to create ordinal bins.
        If *None*, uses ``[0.1, 0.3, 0.7, 0.9]``.
    measure_fn : callable or None, default=None
        See :class:`BootstrapIUR`.
    seed : int or None, default=None
        Random seed.

    Attributes
    ----------
    n_groups_ : int
        Number of groups.
    iur_kappa_ : float
        Mean IUR from weighted kappa (categorised).
    iur_kendall_cat_ : float
        Mean IUR from Kendall tau (categorised).
    iur_spearman_cat_ : float
        Mean IUR from Spearman rho (categorised).
    iur_pearson_ : float
        Mean IUR from Pearson r (continuous).
    iur_kendall_ : float
        Mean IUR from Kendall tau (continuous).
    iur_spearman_ : float
        Mean IUR from Spearman rho (continuous).
    iur_all_ : ndarray of shape (n_iter, 6)
        Per-iteration IUR values for all six metrics.

    Examples
    --------
    >>> from pprof_py.measures.iur import SplitHalfIUR
    >>> model = SplitHalfIUR(n_iter=10, seed=42)
    >>> model.fit(obs, exp, groups)
    >>> print(f"IUR (Pearson) = {model.iur_pearson_:.4f}")
    """

    def __init__(
        self,
        n_iter: int = 10,
        category_probs: Optional[Sequence[float]] = None,
        measure_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.n_iter = n_iter
        self.category_probs = category_probs
        self.measure_fn = measure_fn
        self.seed = seed

    def fit(
        self,
        obs: ArrayLike,
        exp: ArrayLike,
        groups: ArrayLike,
    ) -> "SplitHalfIUR":
        """Compute split-half IUR.

        Parameters
        ----------
        obs : array-like of shape (n_samples,)
            Observed outcomes.
        exp : array-like of shape (n_samples,)
            Expected outcomes.
        groups : array-like of shape (n_samples,)
            Group identifiers.

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

        probs = (
            np.asarray(self.category_probs, dtype=np.float64)
            if self.category_probs is not None
            else np.array([0.1, 0.3, 0.7, 0.9])
        )

        fn = (
            self.measure_fn
            if self.measure_fn is not None
            else ratio_measure
        )

        # Sort by group
        sort_idx = np.argsort(groups, kind="stable")
        obs = obs[sort_idx]
        exp = exp[sort_idx]
        groups = groups[sort_idx]

        unique_groups, group_counts = np.unique(
            groups, return_counts=True
        )
        self.n_groups_ = len(unique_groups)
        all_indices = np.arange(len(obs))

        # Storage for per-iteration correlations
        kappa_v = np.empty(self.n_iter)
        kendall_cat_v = np.empty(self.n_iter)
        spearman_cat_v = np.empty(self.n_iter)
        pearson_v = np.empty(self.n_iter)
        kendall_v = np.empty(self.n_iter)
        spearman_v = np.empty(self.n_iter)

        for loop in range(self.n_iter):
            rng = np.random.default_rng(
                self.seed + loop if self.seed is not None else loop
            )

            half1_idx = _split_half_sample(group_counts, rng)
            half1_set = set(half1_idx.tolist())
            half2_idx = np.array(
                [j for j in all_indices if j not in half1_set],
                dtype=np.intp,
            )

            # Compute measure on each half
            m1 = fn(obs[half1_idx], exp[half1_idx], groups[half1_idx])
            m2 = fn(obs[half2_idx], exp[half2_idx], groups[half2_idx])

            # ── Categorised metrics ──
            cat1 = _categorise(m1, probs)
            cat2 = _categorise(m2, probs)

            if np.array_equal(cat1, cat2):
                kappa_v[loop] = 1.0
            else:
                kappa_v[loop] = cohen_kappa_score(
                    cat1, cat2, weights="linear"
                )

            tau_cat, _ = kendalltau(cat1, cat2)
            kendall_cat_v[loop] = tau_cat

            rho_cat, _ = spearmanr(cat1, cat2)
            spearman_cat_v[loop] = rho_cat

            # ── Continuous metrics ──
            if m1.std() == 0 or m2.std() == 0:
                pearson_v[loop] = 0.0
            else:
                pearson_v[loop] = np.corrcoef(m1, m2)[0, 1]

            tau, _ = kendalltau(m1, m2)
            kendall_v[loop] = tau

            rho, _ = spearmanr(m1, m2)
            spearman_v[loop] = rho

        # ── Convert correlations to IUR ──
        v_c2i = np.vectorize(_correlation_to_iur)

        iur_kappa = v_c2i(kappa_v)
        iur_kendall_cat = v_c2i(kendall_cat_v)
        iur_spearman_cat = v_c2i(spearman_cat_v)
        iur_pearson = v_c2i(pearson_v)
        iur_kendall = v_c2i(kendall_v)
        iur_spearman = v_c2i(spearman_v)

        self.iur_all_ = np.column_stack(
            [
                iur_kappa,
                iur_kendall_cat,
                iur_spearman_cat,
                iur_pearson,
                iur_kendall,
                iur_spearman,
            ]
        )

        self.iur_kappa_ = float(iur_kappa.mean())
        self.iur_kendall_cat_ = float(iur_kendall_cat.mean())
        self.iur_spearman_cat_ = float(iur_spearman_cat.mean())
        self.iur_pearson_ = float(iur_pearson.mean())
        self.iur_kendall_ = float(iur_kendall.mean())
        self.iur_spearman_ = float(iur_spearman.mean())

        return self

    def summary(self) -> pd.DataFrame:
        """Summary of all IUR estimates.

        Returns
        -------
        df : DataFrame
            Mean IUR for each metric, plus number of groups.
        """
        self._require_fitted("iur_kappa_")
        return pd.DataFrame(
            {
                "iur_kappa": [self.iur_kappa_],
                "iur_kendall_cat": [self.iur_kendall_cat_],
                "iur_spearman_cat": [self.iur_spearman_cat_],
                "iur_pearson": [self.iur_pearson_],
                "iur_kendall": [self.iur_kendall_],
                "iur_spearman": [self.iur_spearman_],
                "n_groups": [self.n_groups_],
            }
        )
