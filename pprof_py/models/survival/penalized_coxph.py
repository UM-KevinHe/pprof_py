"""User-facing penalized Cox estimators: `PenalizedCoxPH` (a
regularization path, analogous to R's `glmnet(family="cox")`) and
`PenalizedCoxPHCV` (k-fold cross-validated lambda selection, analogous
to `cv.glmnet(family="cox")`).

Both are built the same way ``CoxPHSelector`` is built on top of
``CoxPH``: they reuse `data/validation.py` and `data/survival_data.py`
untouched, and reuse `algorithms/survival/cox_likelihood.py` (via a closure
identical in shape to `CoxPH.fit`'s `objective(beta)`) for every
likelihood/score/information evaluation. The new numerical work
(standardization, the penalty, coordinate descent, the lambda path,
cross-validation deviance) lives in `algorithms/penalty.py`,
`algorithms/coordinate_descent.py`, and `statistics/deviance.py`; this
module's job is orchestration, validation, and presenting results in
the same estimator style as ``CoxPH`` -- see ``docs/ARCHITECTURE.md``
and ``docs/R_COMPATIBILITY.md`` for the conventions matched against
glmnet 4.1-8.

Naming note: `alpha` here is glmnet's elastic-net mixing parameter
(0 = ridge, 1 = lasso), NOT scikit-learn's `ElasticNet.alpha` (overall
strength) -- this package validates against R's `survival`/`glmnet`,
so its penalized-regression vocabulary follows glmnet, not sklearn's
`linear_model` module. `lambda` is a Python keyword, so the
regularization-strength argument is `lambda_path` (a full sequence, a
single scalar, or `None` for glmnet's auto-generated grid); the fitted
grid actually used is `lambda_path_`.
"""
from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import pandas as pd
from ...base import ProviderModel

from ...data.survival_validation import validate_fit_inputs, validate_X
from ...data.survival_data import SurvivalData
from ...algorithms.survival.cox_likelihood import cox_partial_likelihood, precompute_stratum_indices
from ...algorithms.survival.penalty import weighted_column_scale, rescale_penalty_factors
from ...algorithms.survival.coordinate_descent import (
    fit_regularization_path,
    compute_lambda_max,
    build_lambda_sequence,
)
from ...algorithms.survival.ties import TieMethod
from ...utils.deviance import saturated_log_likelihood, cox_deviance, deviance_ratio, bootstrap_cv_se
from ...utils.numerical import safe_exp
from .coxph import CoxPH, NotFittedError
from ...exceptions import DegenerateFeatureWarning

logger = logging.getLogger(__name__)


def _validate_common_parameters(
    alpha, n_lambda, lambda_min_ratio, lambda_path, standardize,
    max_outer_iter, outer_tol, max_inner_iter, inner_tol, fit_intercept,
):
    if not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if isinstance(n_lambda, (bool, np.bool_)) or int(n_lambda) != n_lambda or int(n_lambda) < 1:
        raise ValueError(f"n_lambda must be a positive integer, got {n_lambda!r}")
    if lambda_min_ratio is not None:
        ratio = float(lambda_min_ratio)
        if not np.isfinite(ratio) or not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"lambda_min_ratio must be finite and in (0, 1], got {lambda_min_ratio!r}"
            )
    if lambda_path is not None:
        values = np.atleast_1d(np.asarray(lambda_path, dtype=np.float64))
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("lambda_path must contain finite, strictly positive values")
        if np.unique(values).size != values.size:
            raise ValueError("lambda_path must not contain duplicate values")
    if not isinstance(standardize, (bool, np.bool_)):
        raise ValueError("standardize must be a boolean")
    if isinstance(max_outer_iter, (bool, np.bool_)) or int(max_outer_iter) != max_outer_iter or int(max_outer_iter) < 1:
        raise ValueError(f"max_outer_iter must be a positive integer, got {max_outer_iter!r}")
    if not np.isfinite(outer_tol) or float(outer_tol) <= 0:
        raise ValueError(f"outer_tol must be finite and > 0, got {outer_tol!r}")
    if isinstance(max_inner_iter, (bool, np.bool_)) or int(max_inner_iter) != max_inner_iter or int(max_inner_iter) < 1:
        raise ValueError(f"max_inner_iter must be a positive integer, got {max_inner_iter!r}")
    if not np.isfinite(inner_tol) or float(inner_tol) <= 0:
        raise ValueError(f"inner_tol must be finite and > 0, got {inner_tol!r}")
    if fit_intercept:
        raise ValueError("Cox proportional hazards regression does not support an intercept")


def _resolve_lambda_path(
    lambda_path,
    lambda_max,
    lambda_min_ratio,
    n_lambda,
    p_fit,
    n_obs,
    lambda_pad: float = 0.0,
):
    if lambda_path is None:
        if lambda_min_ratio is not None:
            ratio = lambda_min_ratio
        else:
            ratio = 1e-2 if n_obs < p_fit else 1e-4
        if lambda_max < 0 or not np.isfinite(lambda_max):
            raise ValueError(f"lambda_max must be finite and non-negative, got {lambda_max!r}")
        if lambda_max == 0.0:
            # The null score is exactly zero, so beta=0 is already the optimum
            # for every positive lambda. Keep a zero path as a representational
            # edge case rather than constructing log(0). No positive lambda can
            # be inferred from lambda_max in this situation.
            return np.zeros(int(n_lambda), dtype=np.float64), float(ratio)
        return build_lambda_sequence(
            lambda_max, ratio, n_lambda, lambda_pad=lambda_pad,
        ), float(ratio)

    if np.isscalar(lambda_path):
        return np.array([float(lambda_path)], dtype=np.float64), (
            float(lambda_min_ratio) if lambda_min_ratio is not None else (1e-2 if n_obs < p_fit else 1e-4)
        )
    values = np.asarray(lambda_path, dtype=np.float64)
    return np.sort(values)[::-1], (
        float(lambda_min_ratio) if lambda_min_ratio is not None else (1e-2 if n_obs < p_fit else 1e-4)
    )


# ----------------------------------------------------------------------
# Event-stratified CV fold assignment (grplasso §4.3.1)
# ----------------------------------------------------------------------
def stratified_fold_assignment(
    event: np.ndarray,
    n_folds: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Assign CV folds ensuring proportional event/censoring balance.

    All ``grplasso`` CV routines stratify folds by event indicator so
    that each fold has roughly the same event rate.  This matters most
    when events are rare: a purely random split can leave a fold with
    very few or zero events, causing the fold-specific partial
    likelihood to degenerate.

    Parameters
    ----------
    event : ndarray of {0, 1}, shape (n,)
        Event indicator.
    n_folds : int
        Number of CV folds (>= 3).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    fold_id : ndarray of int, shape (n,)
        Fold labels in 0 .. n_folds - 1.
    """
    rng = np.random.RandomState(random_state)
    event = np.asarray(event)
    n = len(event)
    fold = np.empty(n, dtype=np.intp)
    for stratum_mask in [event == 1, event == 0]:
        idx = np.where(stratum_mask)[0]
        if len(idx) == 0:
            continue
        assignments = np.tile(
            np.arange(n_folds), len(idx) // n_folds + 1
        )[:len(idx)]
        fold[idx] = rng.permutation(assignments)
    return fold


# ======================================================================
# Shared base mixin for penalized Cox PH path estimators
# ======================================================================

class _PenalizedCoxPHBase:
    """Mixin providing shared infrastructure for penalized Cox PH
    path estimators (``PenalizedCoxPH``, ``GroupLassoCoxPH``, and
    future ``ProviderPenalizedCoxPH``).

    Subclasses implement ``fit()`` using the helper methods
    ``_prepare_fit_data()``, ``_compute_null_point()``, and
    ``_store_path_results()`` for common boilerplate, then insert
    penalty-specific lambda-max computation and path fitting in
    between.

    Class hierarchy::

        _PenalizedCoxPHBase (mixin)
            ├── PenalizedCoxPH           (elastic net)
            ├── GroupLassoCoxPH          (group / sparse group lasso)
            └── ProviderPenalizedCoxPH   (two-layer provider + covariates)
    """

    # ------------------------------------------------------------------
    # Fit helpers
    # ------------------------------------------------------------------
    def _prepare_fit_data(
        self, X, duration, event, start, stop, strata, offset, sample_weight,
    ) -> SimpleNamespace:
        """Validate inputs, standardize columns, build the objective
        closure.  Shared across all penalized Cox estimators.

        Returns a ``SimpleNamespace`` with:

        * ``data`` -- ``SurvivalData`` instance.
        * ``X_fit`` -- design matrix after degenerate-column removal and
          standardization, shape ``(n, p_fit)``.
        * ``pf_fit`` -- rescaled penalty factors for the fit columns,
          shape ``(p_fit,)``.
        * ``p_fit``, ``p_full`` -- column counts.
        * ``xs_full`` -- per-column scale divisors, shape ``(p_full,)``.
        * ``fit_cols`` -- boolean mask of included columns.
        * ``degenerate`` -- boolean mask of excluded columns.
        * ``feature_names`` -- original feature names.
        * ``penalty_factor_full`` -- user-supplied penalty factors,
          shape ``(p_full,)``.
        * ``objective_fn`` -- ``objective_fn(beta) -> (loglik, score, info)``.
        * ``c`` -- ``1 / sum(weight)``.
        * ``stratum_idx`` -- precomputed stratum index dict.
        """
        clean = validate_fit_inputs(
            X, duration=duration, event=event, start=start, stop=stop,
            strata=strata, offset=offset, sample_weight=sample_weight,
        )
        data = SurvivalData(**clean)
        feature_names = list(data.feature_names)
        p_full = data.n_features

        pf_input = getattr(self, 'penalty_factor', None)
        penalty_factor_full = (
            np.ones(p_full, dtype=np.float64)
            if pf_input is None
            else np.asarray(pf_input, dtype=np.float64)
        )
        if penalty_factor_full.shape != (p_full,):
            raise ValueError(
                f"penalty_factor must have shape ({p_full},), "
                f"got {penalty_factor_full.shape}"
            )
        if not np.all(np.isfinite(penalty_factor_full)):
            raise ValueError("penalty_factor must contain only finite values")
        if np.any(penalty_factor_full < 0):
            raise ValueError("penalty_factor must be non-negative")

        standardize = getattr(self, 'standardize', True)
        xs_full, degenerate = weighted_column_scale(
            data.X, data.weight, standardize=standardize,
        )
        if np.any(degenerate):
            warnings.warn(
                f"{int(np.sum(degenerate))} feature(s) have ~zero weighted "
                "variance and contribute no information to the partial "
                "likelihood regardless of their coefficient; excluding them "
                "from the penalized fit (coefficient fixed at 0): "
                f"{list(np.array(feature_names)[degenerate])}",
                category=DegenerateFeatureWarning, stacklevel=3,
            )
        fit_cols = ~degenerate
        if not np.any(fit_cols):
            raise ValueError(
                "All predictors have zero weighted variance; "
                "no usable predictors remain"
            )
        X_fit = data.X[:, fit_cols] / xs_full[fit_cols]
        pf_fit = rescale_penalty_factors(
            penalty_factor_full[fit_cols], int(fit_cols.sum()),
        )
        p_fit = X_fit.shape[1]

        stratum_idx = precompute_stratum_indices(data.strata_codes)
        ties = getattr(self, 'ties', 'breslow')

        def objective_fn(beta_fit):  # noqa: D401
            """Partial log-likelihood, score, and information at *beta_fit*."""
            return cox_partial_likelihood(
                X_fit, data.start, data.stop, data.event, beta_fit,
                offset=data.offset, weight=data.weight,
                strata=data.strata_codes, ties=ties,
                stratum_indices=stratum_idx,
            )

        c = 1.0 / float(np.sum(data.weight))

        return SimpleNamespace(
            data=data, X_fit=X_fit, pf_fit=pf_fit, p_fit=p_fit,
            p_full=p_full, xs_full=xs_full, fit_cols=fit_cols,
            degenerate=degenerate, feature_names=feature_names,
            penalty_factor_full=penalty_factor_full,
            objective_fn=objective_fn, c=c, stratum_idx=stratum_idx,
        )

    def _compute_null_point(self, prep: SimpleNamespace):
        """Compute the null point for lambda_max calculation.

        Returns ``(beta_null, score_null)`` where ``beta_null`` is zero
        for all penalized features and the unpenalized MLE for any
        always-unpenalized features (matching glmnet's
        ``get_cox_lambda_max``, which pre-fits unpenalized variables
        via ``survival::coxph``).
        """
        beta_null = np.zeros(prep.p_fit)
        always_unpenalized = prep.pf_fit == 0.0
        if np.any(always_unpenalized) and not np.all(always_unpenalized):
            restricted = CoxPH(
                ties=getattr(self, 'ties', 'breslow'),
                max_iter=getattr(self, 'max_outer_iter', 100),
                eps=getattr(self, 'outer_tol', 1e-9),
            )
            restricted.fit(
                prep.X_fit[:, always_unpenalized],
                duration=None, event=prep.data.event,
                start=prep.data.start, stop=prep.data.stop,
                strata=prep.data.strata_codes,
                offset=prep.data.offset,
                sample_weight=prep.data.weight,
            )
            beta_null[always_unpenalized] = restricted.coef_

        _, score_null, _ = prep.objective_fn(beta_null)
        return beta_null, score_null

    def _store_path_results(
        self,
        results,
        prep: SimpleNamespace,
        lambda_sequence: np.ndarray,
        lambda_max: float,
        lambda_min_ratio: float,
    ) -> None:
        """Store common fitted attributes from path results.

        Sets ``coef_path_``, ``lambda_path_``, ``lambda_max_``,
        ``lambda_min_ratio_``, ``log_likelihood_path_``,
        ``converged_path_``, ``n_iter_path_``, ``n_nonzero_path_``,
        ``column_scale_``, ``penalty_factor_``, ``_excluded_features_``,
        ``log_likelihood_null_``, ``_lsat_``, ``deviance_ratio_path_``,
        ``n_obs_``, ``n_events_``, ``n_features_in_``,
        ``feature_names_in_``, and (when a single lambda is fit)
        ``coef_`` and ``lambda_``.
        """
        n_lam = len(results)
        coef_path_fit = np.array(
            [r.beta / prep.xs_full[prep.fit_cols] for r in results],
        )
        coef_path = np.zeros((n_lam, prep.p_full))
        coef_path[:, prep.fit_cols] = coef_path_fit

        self.coef_path_ = coef_path
        self.lambda_path_ = np.asarray(lambda_sequence, dtype=np.float64)
        self.lambda_max_ = float(lambda_max)
        self.lambda_min_ratio_ = float(lambda_min_ratio)
        self.log_likelihood_path_ = np.array(
            [r.log_likelihood for r in results],
        )
        self.converged_path_ = np.array([r.converged for r in results])
        self.n_iter_path_ = np.array([r.n_outer_iter for r in results])
        self.n_nonzero_path_ = np.array(
            [int(np.sum(row != 0.0)) for row in coef_path],
        )
        self.column_scale_ = prep.xs_full
        self.penalty_factor_ = prep.penalty_factor_full
        self._excluded_features_ = prep.degenerate

        # Null log-likelihood and deviance ratio.
        null_loglik, _, _ = cox_partial_likelihood(
            prep.data.X, prep.data.start, prep.data.stop,
            prep.data.event, np.zeros(prep.p_full),
            offset=prep.data.offset, weight=prep.data.weight,
            strata=prep.data.strata_codes,
            ties=getattr(self, 'ties', 'breslow'),
            stratum_indices=prep.stratum_idx,
        )
        self.log_likelihood_null_ = null_loglik
        lsat = saturated_log_likelihood(
            prep.data.stop, prep.data.event,
            prep.data.weight, prep.data.strata_codes,
        )
        self._lsat_ = lsat
        self.deviance_ratio_path_ = np.array(
            [deviance_ratio(ll, null_loglik, lsat)
             for ll in self.log_likelihood_path_],
        )

        self.n_obs_ = prep.data.n_obs
        self.n_events_ = int(np.sum(prep.data.event))
        self.n_features_in_ = prep.p_full
        self.feature_names_in_ = np.array(prep.feature_names)

        if n_lam == 1:
            self.coef_ = coef_path[0]
            self.lambda_ = float(self.lambda_path_[0])

    # ------------------------------------------------------------------
    # Fitted-model queries
    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        cls_name = type(self).__name__
        if not hasattr(self, "coef_path_"):
            raise NotFittedError(
                f"This {cls_name} instance is not fitted yet. "
                "Call `fit` first."
            )

    def coef_at(self, lambda_value: float) -> np.ndarray:
        """Coefficients at an arbitrary lambda, linearly interpolated
        in log(lambda) between the two bracketing fitted grid points
        (glmnet's ``lambda.interp`` convention); clamped to the nearest
        endpoint if ``lambda_value`` is outside the fitted range.
        """
        self._check_is_fitted()
        grid = self.lambda_path_
        lam = float(lambda_value)
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError(
                f"lambda_value must be finite and strictly positive, "
                f"got {lambda_value!r}"
            )
        if np.any(grid <= 0):
            raise ValueError(
                "coef_at() is undefined for a path containing "
                "non-positive lambda values"
            )
        if len(grid) == 1:
            return self.coef_path_[0]
        log_grid = np.log(grid)
        if lam >= grid[0]:
            return self.coef_path_[0]
        if lam <= grid[-1]:
            return self.coef_path_[-1]
        log_lam = np.log(lam)
        upper_idx = np.searchsorted(-log_grid, -log_lam) - 1
        upper_idx = int(np.clip(upper_idx, 0, len(grid) - 2))
        lower_idx = upper_idx + 1
        l_up, l_lo = log_grid[upper_idx], log_grid[lower_idx]
        frac = 0.0 if l_up == l_lo else (log_lam - l_lo) / (l_up - l_lo)
        return (
            frac * self.coef_path_[upper_idx]
            + (1 - frac) * self.coef_path_[lower_idx]
        )

    def nonzero_features(
        self, lambda_value: Optional[float] = None,
    ) -> np.ndarray:
        """Feature names with a nonzero coefficient at ``lambda_value``
        (defaults to ``self.lambda_`` when the fit resolved to a single
        lambda)."""
        self._check_is_fitted()
        if lambda_value is None:
            if not hasattr(self, "coef_"):
                raise ValueError(
                    "lambda_value must be specified when multiple "
                    "lambda values were fitted"
                )
            coef = self.coef_
        else:
            coef = self.coef_at(lambda_value)
        return self.feature_names_in_[coef != 0.0]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _resolve_coef(
        self, lambda_value: Optional[float],
    ) -> np.ndarray:
        self._check_is_fitted()
        if lambda_value is not None:
            return self.coef_at(lambda_value)
        if hasattr(self, "coef_"):
            return self.coef_
        raise ValueError(
            "This fit produced a path with more than one lambda; "
            "pass `lambda_value=` to select which point on the path "
            "to use for prediction."
        )

    def predict_linear(
        self, X, offset=None, lambda_value: Optional[float] = None,
    ) -> np.ndarray:
        """Linear predictor ``X @ coef + offset`` at a given lambda.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        offset : ndarray or None
        lambda_value : float or None
            Query lambda.  Uses ``coef_`` if the path has length 1.

        Returns
        -------
        ndarray, shape (n_new,)
        """
        coef = self._resolve_coef(lambda_value)
        X_arr, _ = validate_X(X)
        offset_arr = (
            np.zeros(X_arr.shape[0])
            if offset is None
            else np.asarray(offset, dtype=np.float64)
        )
        return X_arr @ coef + offset_arr

    def predict_partial_hazard(
        self, X, offset=None, lambda_value: Optional[float] = None,
    ) -> np.ndarray:
        """Partial hazard ``exp(X @ coef + offset)`` at a given lambda.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        offset : ndarray or None
        lambda_value : float or None

        Returns
        -------
        ndarray, shape (n_new,)
        """
        return safe_exp(
            self.predict_linear(X, offset=offset, lambda_value=lambda_value),
        )

    def predict(
        self, X, offset=None, lambda_value: Optional[float] = None,
    ) -> np.ndarray:
        """Alias for ``predict_partial_hazard``.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        offset : ndarray or None
        lambda_value : float or None

        Returns
        -------
        ndarray, shape (n_new,)
        """
        return self.predict_partial_hazard(
            X, offset=offset, lambda_value=lambda_value,
        )

    def summary(self) -> pd.DataFrame:
        """Per-lambda path summary analogous to glmnet's printed
        ``glmnet`` object table (lambda, df, %dev)."""
        self._check_is_fitted()
        return pd.DataFrame(
            {
                "lambda": self.lambda_path_,
                "n_nonzero": self.n_nonzero_path_,
                "deviance_ratio": self.deviance_ratio_path_,
                "log_likelihood": self.log_likelihood_path_,
                "converged": self.converged_path_,
            }
        )


class PenalizedCoxPH(_PenalizedCoxPHBase, ProviderModel):
    """Elastic-net-penalized Cox Proportional Hazards regression, fit
    by proximal Newton + coordinate descent over a lambda path.

    Reuses the exact same (Breslow/Efron) partial-likelihood engine as
    `CoxPH`, so every non-penalization capability -- strata, offset,
    sample weights, start/stop (left-truncated) data -- carries over
    unchanged.  See ``docs/R_COMPATIBILITY.md`` for the numerical
comparison against the package's pinned glmnet reference version. Current glmnet releases support additional Cox options;
    compatibility claims here refer to the explicitly pinned reference
    used by the regression suite. Efron-tie penalized fits use the same
    engine and are validated by self-consistency against this
    package's own (R-validated) unpenalized `CoxPH(ties="efron")` in
    the lambda -> 0 limit, since no independent Efron+penalty
    reference implementation is available.

    Parameters
    ----------
    alpha : float, default 1.0
        Elastic-net mixing parameter in [0, 1], glmnet's convention:
        0 = ridge, 1 = lasso (the default), values in between = elastic
        net. NOT scikit-learn's `ElasticNet.alpha` -- see module
        docstring.
    n_lambda : int, default 100
        Number of lambda values in the auto-generated path (glmnet's
        `nlambda`). Ignored if `lambda_path` is given.
    lambda_min_ratio : float, optional
        Smallest lambda in the auto-generated path, as a fraction of
        `lambda_max_`. Defaults to glmnet's own rule: 1e-2 if
        n_obs < n_features, else 1e-4. Ignored if `lambda_path` is given.
    lambda_path : float, sequence of float, or None, default None
        None auto-generates the path (see `n_lambda`,
        `lambda_min_ratio`). A single float fits exactly that one
        lambda. A sequence fits exactly those lambda values (sorted
        descending internally for warm-starting; `lambda_path_`
        reflects the order actually fit).
    penalty_factor : array-like of shape (n_features,), optional
        Per-feature multiplier on the penalty (glmnet convention): 0
        marks an always-unpenalized feature (always in the model,
        never shrunk); values are rescaled internally to sum to
        n_features, matching glmnet. Defaults to all-ones (every
        feature penalized equally).
    standardize : bool, default True
        Divide each column of X by its weighted population standard
        deviation before fitting so the penalty is comparable across
        features measured in different units, then divide the fitted
        coefficients back by the same scale (glmnet's default
        behavior; does not affect an unpenalized fit but changes which
        coefficients an L1/L2 penalty shrinks first).
    ties, fit_intercept, max_outer_iter, outer_tol, max_inner_iter, inner_tol :
        See `CoxPH` for `ties`/`fit_intercept`; the remaining four
        control the proximal-Newton outer loop and coordinate-descent
        inner loop (`algorithms/coordinate_descent.py`).

    Attributes (set by `fit`)
    -------------------------
    coef_path_ : ndarray, shape (n_lambda_, n_features)
        Fitted coefficients (original units) at every lambda in
        `lambda_path_`, in the same order.
    lambda_path_ : ndarray, shape (n_lambda_,)
        The lambda values actually fit, descending.
    lambda_max_ : float
        Smallest lambda at which every penalized coefficient is 0
        (see `algorithms/coordinate_descent.py::compute_lambda_max`).
        0 if every feature is unpenalized.
    log_likelihood_path_, deviance_ratio_path_, n_nonzero_path_ : ndarray, shape (n_lambda_,)
        Per-lambda partial log-likelihood, glmnet-style deviance ratio
        (`statistics/deviance.py`), and count of exactly-nonzero
        coefficients (glmnet's `df`).
    log_likelihood_null_ : float
        Partial log-likelihood at beta=0 (all features), the
        deviance-ratio denominator.
    column_scale_ : ndarray, shape (n_features,)
        The `xs` divisor applied to each column before fitting
        (all 1s if `standardize=False`).
    coef_, lambda_ : ndarray / float
        Only set when the resolved `lambda_path_` has exactly one
        value (a single explicit `lambda_path` scalar, or a
        user-supplied length-1 sequence) -- the natural case of "just
        fit one penalized model". Use `coef_at()` or index
        `coef_path_` directly otherwise.
    n_obs_, n_events_, n_features_in_, feature_names_in_, converged_path_, n_iter_path_ :
        See `CoxPH` for the analogous non-path attributes.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[Union[float, "np.ndarray"]] = None,
        penalty_factor: Optional[np.ndarray] = None,
        standardize: bool = True,
        ties: Union[str, TieMethod] = "breslow",
        fit_intercept: bool = False,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
    ):
        """Penalized Cox regularization path (R: ``glmnet(family='cox')``)."""
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.penalty_factor = penalty_factor
        self.standardize = standardize
        self.ties = ties
        self.fit_intercept = fit_intercept
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        duration=None,
        event=None,
        start=None,
        stop=None,
        strata=None,
        offset=None,
        sample_weight=None,
    ) -> "PenalizedCoxPH":
        """Fit the elastic-net penalized Cox regularization path.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_samples, n_features)
        duration, event, start, stop, strata, offset, sample_weight
            Same as ``CoxPH.fit``.

        Returns
        -------
        self
        """
        _validate_common_parameters(
            self.alpha, self.n_lambda, self.lambda_min_ratio,
            self.lambda_path, self.standardize, self.max_outer_iter,
            self.outer_tol, self.max_inner_iter, self.inner_tol,
            self.fit_intercept,
        )

        # Common data preparation (shared with GroupLassoCoxPH).
        prep = self._prepare_fit_data(
            X, duration, event, start, stop, strata, offset,
            sample_weight,
        )
        beta_null, score_null = self._compute_null_point(prep)

        # Elastic-net-specific: element-wise lambda_max.
        always_unpenalized = prep.pf_fit == 0.0
        lambda_max = (
            0.0 if np.all(always_unpenalized)
            else compute_lambda_max(
                score_null, prep.c, prep.pf_fit, self.alpha,
            )
        )

        lambda_sequence, lambda_min_ratio = _resolve_lambda_path(
            self.lambda_path, lambda_max, self.lambda_min_ratio,
            self.n_lambda, prep.p_fit, prep.data.n_obs,
        )

        results = fit_regularization_path(
            prep.objective_fn, prep.p_fit, prep.c, self.alpha,
            lambda_sequence, prep.pf_fit,
            beta_warm_start=beta_null,
            outer_max_iter=self.max_outer_iter,
            outer_tol=self.outer_tol,
            inner_max_iter=self.max_inner_iter,
            inner_tol=self.inner_tol,
        )

        # Common post-fit attribute storage.
        self._store_path_results(
            results, prep, lambda_sequence, lambda_max,
            lambda_min_ratio,
        )
        return self

    # coef_at, nonzero_features, predict_*, summary are inherited from
    # _PenalizedCoxPHBase.


# ======================================================================
# Shared base mixin for penalized Cox PH cross-validated estimators
# ======================================================================

class _PenalizedCoxPHCVBase:
    """Mixin providing shared CV infrastructure for penalized Cox PH
    cross-validated estimators.

    Class hierarchy::

        _PenalizedCoxPHCVBase (mixin)
            ├── PenalizedCoxPHCV         (elastic net CV)
            └── GroupLassoCoxPHCV        (group lasso CV)
    """

    def _assign_folds(
        self,
        event: np.ndarray,
        n_obs: int,
        n_folds: int,
        fold_id,
        random_state,
    ):
        """Assign CV folds, event-stratified by default.

        When ``fold_id`` is supplied by the user, validates and
        normalizes the user-supplied assignment.  Otherwise, uses
        event-stratified random assignment (from ``grplasso``
        §4.3.1) to ensure each fold has roughly the same event
        rate.

        Returns ``(fold_id, fold_labels)``.
        """
        if fold_id is not None:
            fold_id_raw = np.asarray(fold_id)
            if fold_id_raw.ndim != 1 or fold_id_raw.shape[0] != n_obs:
                raise ValueError(
                    f"fold_id must be a one-dimensional array of "
                    f"length {n_obs}"
                )
            if not np.all(
                np.isfinite(fold_id_raw.astype(np.float64, copy=False))
            ):
                raise ValueError("fold_id must contain finite values")
            if not np.all(
                np.equal(fold_id_raw, np.floor(fold_id_raw))
            ):
                raise ValueError("fold_id values must be integers")
            fold_id_out = fold_id_raw.astype(np.int64)
            fold_labels = np.unique(fold_id_out)
            k = fold_labels.size
            if not (
                np.array_equal(fold_labels, np.arange(k))
                or np.array_equal(fold_labels, np.arange(1, k + 1))
            ):
                raise ValueError(
                    "fold_id must use contiguous labels 0..K-1 or 1..K"
                )
        else:
            # Event-stratified fold assignment (grplasso §4.3.1).
            fold_id_out = stratified_fold_assignment(
                event, n_folds, random_state,
            )
            fold_labels = np.arange(n_folds)
        return fold_id_out, fold_labels

    def _compute_cv_statistics(
        self,
        cvraw: np.ndarray,
        fold_event_weight: np.ndarray,
    ):
        """Compute CV mean and SE with saturated-lambda elimination.

        Saturated-lambda elimination (§4.3.3): lambda values where
        **any** fold produced a non-finite deviance are dropped,
        since they indicate numerical instability (typically from
        extremely large or small lambda values that cause the
        fold-specific partial likelihood to degenerate).

        Returns ``(cvm, cvsd, n_folds_used)``.
        """
        n_lam = cvraw.shape[1]

        # Saturated-lambda elimination.
        all_finite = np.all(np.isfinite(cvraw), axis=0)
        n_dropped = int(np.sum(~all_finite))
        if n_dropped > 0:
            logger.info(
                "Saturated-lambda elimination: dropped %d/%d lambda "
                "values where at least one fold produced non-finite "
                "deviance.",
                n_dropped, n_lam,
            )

        valid = np.isfinite(cvraw)
        n_folds_used = np.sum(valid, axis=0)
        cvm = np.full(n_lam, np.nan, dtype=np.float64)
        cvsd = np.full(n_lam, np.nan, dtype=np.float64)
        for j in range(n_lam):
            if not all_finite[j]:
                continue  # saturated-lambda elimination
            ok = valid[:, j]
            if np.sum(ok) == 0:
                continue
            wj = fold_event_weight[ok]
            yj = cvraw[ok, j]
            wsum = np.sum(wj)
            cvm[j] = np.sum(wj * yj) / wsum
            n_ok = int(np.sum(ok))
            if n_ok > 1:
                # Match glmnet::cvstats: weighted population variance
                # divided by (N - 1).
                cv_var = np.sum(wj * (yj - cvm[j]) ** 2) / wsum
                cvsd[j] = np.sqrt(cv_var / (n_ok - 1))
            else:
                cvsd[j] = np.nan

        return cvm, cvsd, n_folds_used

    def _select_lambda(
        self,
        lambda_grid: np.ndarray,
        cvm: np.ndarray,
        cvsd: np.ndarray,
    ):
        """Select ``lambda_min`` and ``lambda_1se`` from CV statistics.

        Returns ``(lambda_min, lambda_1se)``.
        """
        valid_lambda = np.isfinite(cvm) & np.isfinite(cvsd)
        if not np.any(valid_lambda):
            raise RuntimeError(
                "Cross-validation produced no valid lambda values"
            )
        min_idx = int(
            np.nanargmin(np.where(valid_lambda, cvm, np.nan))
        )
        lambda_min = float(lambda_grid[min_idx])
        within_1se = valid_lambda & (
            cvm <= cvm[min_idx] + cvsd[min_idx]
        )
        # lambda_grid is descending, so the *first* True is the
        # largest lambda within one SE of the minimum.
        lambda_1se = float(
            lambda_grid[np.flatnonzero(within_1se)[0]]
        )
        return lambda_min, lambda_1se

    def _check_is_fitted(self) -> None:
        cls_name = type(self).__name__
        if not hasattr(self, "final_estimator_"):
            raise NotFittedError(
                f"This {cls_name} instance is not fitted yet. "
                "Call `fit` first."
            )

    def predict_linear(self, X, offset=None) -> np.ndarray:
        """Linear predictor at the selected lambda.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        offset : ndarray or None

        Returns
        -------
        ndarray, shape (n_new,)
        """
        self._check_is_fitted()
        return self.final_estimator_.predict_linear(X, offset=offset)

    def predict_partial_hazard(self, X, offset=None) -> np.ndarray:
        """Partial hazard at the selected lambda.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        offset : ndarray or None

        Returns
        -------
        ndarray, shape (n_new,)
        """
        self._check_is_fitted()
        return self.final_estimator_.predict_partial_hazard(
            X, offset=offset,
        )

    def predict(self, X, offset=None) -> np.ndarray:
        """Alias for ``predict_partial_hazard``.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        offset : ndarray or None

        Returns
        -------
        ndarray, shape (n_new,)
        """
        return self.predict_partial_hazard(X, offset=offset)

    def summary(self) -> pd.DataFrame:
        """Per-lambda cross-validation summary analogous to R's
        ``print(cv.glmnet_fit)`` / plotting data."""
        self._check_is_fitted()
        return pd.DataFrame(
            {
                "lambda": self.lambda_path_,
                "n_nonzero": self.n_nonzero_path_,
                "cv_mean_deviance": self.cv_mean_deviance_,
                "cv_se_deviance": self.cv_se_deviance_,
            }
        )


class PenalizedCoxPHCV(_PenalizedCoxPHCVBase, ProviderModel):
    """K-fold cross-validated lambda selection for `PenalizedCoxPH`,
    analogous to R's `cv.glmnet(family="cox")`.

    Fits the full-data path once to establish the lambda grid
    (`lambda_path_`), then, for each fold, fits `PenalizedCoxPH` on the
    training-minus-fold data *at exactly those same lambda values*
    (rather than letting each fold auto-generate and interpolate onto
    the master grid, as glmnet does internally) -- a simplification
    enabled by `PenalizedCoxPH.fit(lambda_path=...)` accepting an
    explicit sequence, and at least as accurate as interpolation since
    each fold is fit exactly at the lambda of interest. The
    per-fold-per-lambda deviance uses the Verweij & Van Houwelingen
    (1993) grouped method, matching glmnet's default (`grouped=TRUE`)
    -- see `statistics/deviance.py` and docs/R_COMPATIBILITY.md.

    Parameters
    ----------
    n_folds : int, default 10
        Number of cross-validation folds (glmnet's `nfolds`).
    fold_id : array-like of shape (n_obs,), optional
        Explicit fold assignment (1..n_folds or 0..n_folds-1), matching
        glmnet's `foldid` -- primarily useful for reproducing an exact
        R comparison with the same fold assignment.
    se_method : {"analytical", "bootstrap"}, default "analytical"
        How to estimate the standard error of the CV deviance:

        * ``'analytical'`` (default) -- glmnet-style weighted variance
          across folds.
        * ``'bootstrap'`` -- bootstrap resampling of observations,
          preserving within-stratum time ordering (``grplasso``
          §4.3.2). Only Breslow ties are supported.
    n_bootstrap : int, default 100
        Number of bootstrap replicates when ``se_method='bootstrap'``.
    random_state : int, optional
        Used to randomly assign folds when `fold_id` is not given,
        and as the seed for bootstrap resampling.
    se_rule : {"min", "1se"}, default "min"
        Which cross-validated lambda `final_estimator_`/`coef_` uses: the
        minimum CV error (``"min"``) or the largest lambda within one SE of it
        (``"1se"``).
    Remaining parameters are passed through to the underlying
    `PenalizedCoxPH` fits -- see that class.

    Attributes (set by `fit`)
    -------------------------
    lambda_path_, cv_mean_deviance_, cv_se_deviance_ : ndarray, shape (n_lambda_,)
        The fitted lambda grid and, per lambda, the cross-validated
        mean (and standard error of the mean, across folds) deviance
        per event -- glmnet's `cvm`/`cvsd`.
    lambda_min_, lambda_1se_ : float
        The lambda with minimum cross-validated deviance, and the
        largest lambda within one standard error of that minimum
        (glmnet's 1-SE rule).
    final_estimator_ : PenalizedCoxPH
        Fit on the *full* data at `lambda_min_` or `lambda_1se_` (per
        `select`).
    coef_ : ndarray
        `final_estimator_.coef_`.
    fold_id_ : ndarray, shape (n_obs,)
        The fold assignment actually used.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[Union[float, "np.ndarray"]] = None,
        penalty_factor: Optional[np.ndarray] = None,
        standardize: bool = True,
        ties: Union[str, TieMethod] = "breslow",
        fit_intercept: bool = False,
        n_folds: int = 10,
        fold_id: Optional[np.ndarray] = None,
        se_method: str = "analytical",
        n_bootstrap: int = 100,
        random_state: Optional[int] = None,
        se_rule: str = "min",
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
    ):
        """Cross-validated penalized Cox (R: ``cv.glmnet(family='cox')``)."""
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.penalty_factor = penalty_factor
        self.standardize = standardize
        self.ties = ties
        self.fit_intercept = fit_intercept
        self.n_folds = n_folds
        self.fold_id = fold_id
        self.se_method = se_method
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.se_rule = se_rule
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol

    def _base_kwargs(self) -> dict:
        return dict(
            alpha=self.alpha, penalty_factor=self.penalty_factor, standardize=self.standardize,
            ties=self.ties, fit_intercept=self.fit_intercept,
            max_outer_iter=self.max_outer_iter, outer_tol=self.outer_tol,
            max_inner_iter=self.max_inner_iter, inner_tol=self.inner_tol,
        )

    def fit(
        self,
        X,
        duration=None,
        event=None,
        start=None,
        stop=None,
        strata=None,
        offset=None,
        sample_weight=None,
    ) -> "PenalizedCoxPHCV":
        """Fit the cross-validated penalized Cox model.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_samples, n_features)
        duration, event, start, stop, strata, offset, sample_weight
            Same as ``CoxPH.fit``.

        Returns
        -------
        self
        """
        _validate_common_parameters(
            self.alpha, self.n_lambda, self.lambda_min_ratio,
            self.lambda_path, self.standardize,
            self.max_outer_iter, self.outer_tol,
            self.max_inner_iter, self.inner_tol, self.fit_intercept,
        )
        if self.se_rule not in ("min", "1se"):
            raise ValueError(
                f"se_rule must be 'min' or '1se', "
                f"got {self.se_rule!r}"
            )
        if self.se_method not in ("analytical", "bootstrap"):
            raise ValueError(
                f"se_method must be 'analytical' or 'bootstrap', "
                f"got {self.se_method!r}"
            )
        if self.se_method == "bootstrap":
            ties_name = (
                self.ties
                if isinstance(self.ties, str)
                else str(self.ties).lower()
            )
            if ties_name != "breslow":
                raise NotImplementedError(
                    "se_method='bootstrap' only supports "
                    "ties='breslow'"
                )
        if self.n_lambda < 2 and self.lambda_path is None:
            raise ValueError(
                "n_lambda must be at least 2 for cross-validation"
            )
        if (
            self.lambda_path is not None
            and np.atleast_1d(self.lambda_path).size < 2
        ):
            raise ValueError(
                "lambda_path must contain at least 2 values for "
                "cross-validation"
            )
        if (
            isinstance(self.n_folds, (bool, np.bool_))
            or int(self.n_folds) != self.n_folds
            or int(self.n_folds) < 3
        ):
            raise ValueError("n_folds must be an integer >= 3")

        clean = validate_fit_inputs(
            X, duration=duration, event=event, start=start,
            stop=stop, strata=strata, offset=offset,
            sample_weight=sample_weight,
        )
        data = SurvivalData(**clean)
        n = data.n_obs

        # Full-data path fit to establish the lambda grid.
        full_fit = PenalizedCoxPH(
            n_lambda=self.n_lambda,
            lambda_min_ratio=self.lambda_min_ratio,
            lambda_path=self.lambda_path,
            **self._base_kwargs(),
        )
        full_fit.fit(
            data.X, event=data.event, start=data.start,
            stop=data.stop, strata=data.strata_codes,
            offset=data.offset, sample_weight=data.weight,
        )
        lambda_grid = full_fit.lambda_path_
        n_lam = len(lambda_grid)

        # Event-stratified fold assignment (§4.3.1) or user-supplied.
        fold_id, fold_labels = self._assign_folds(
            data.event, n, self.n_folds, self.fold_id,
            self.random_state,
        )
        self.fold_id_ = fold_id

        cvraw = np.full(
            (len(fold_labels), n_lam), np.nan, dtype=np.float64,
        )
        fold_weight = np.zeros(len(fold_labels), dtype=np.float64)
        fold_event_weight = np.zeros(
            len(fold_labels), dtype=np.float64,
        )
        # Out-of-fold linear predictors for bootstrap SE (§4.3.2).
        use_bootstrap = self.se_method == "bootstrap"
        eta_hat = (
            np.full((n, n_lam), np.nan, dtype=np.float64)
            if use_bootstrap
            else None
        )

        # Full-data quantities are independent of the held-out fold.
        lsat_full = saturated_log_likelihood(
            data.stop, data.event, data.weight, data.strata_codes,
        )
        _stratum_idx_full = precompute_stratum_indices(
            data.strata_codes,
        )

        for i, fold in enumerate(fold_labels):
            held_out = fold_id == fold
            train = ~held_out
            fold_weight[i] = float(
                np.sum(data.weight[held_out]),
            )
            fold_event_weight[i] = float(
                np.sum(
                    data.weight[held_out] * data.event[held_out]
                ),
            )
            if fold_event_weight[i] <= 0.0:
                raise ValueError(
                    f"CV fold {fold!r} contains no positive "
                    "event weight"
                )

            fold_fit = PenalizedCoxPH(
                lambda_path=lambda_grid, **self._base_kwargs(),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=DegenerateFeatureWarning,
                )
                fold_fit.fit(
                    data.X[train],
                    event=data.event[train],
                    start=data.start[train],
                    stop=data.stop[train],
                    strata=data.strata_codes[train],
                    offset=data.offset[train],
                    sample_weight=data.weight[train],
                )

            if not np.all(fold_fit.converged_path_):
                bad = np.flatnonzero(~fold_fit.converged_path_)
                warnings.warn(
                    f"Penalized Cox fold {fold!r} did not converge "
                    f"at lambda indices {bad.tolist()}; using the "
                    "last iterate (matches glmnet behavior)",
                    stacklevel=2,
                )

            lsat_train = saturated_log_likelihood(
                data.stop[train], data.event[train],
                data.weight[train], data.strata_codes[train],
            )
            _stratum_idx_train = precompute_stratum_indices(
                data.strata_codes[train],
            )
            for j in range(n_lam):
                beta_j = fold_fit.coef_path_[j]
                loglik_full, _, _ = cox_partial_likelihood(
                    data.X, data.start, data.stop, data.event,
                    beta_j, offset=data.offset,
                    weight=data.weight,
                    strata=data.strata_codes,
                    ties=self.ties,
                    stratum_indices=_stratum_idx_full,
                )
                loglik_train, _, _ = cox_partial_likelihood(
                    data.X[train], data.start[train],
                    data.stop[train], data.event[train], beta_j,
                    offset=data.offset[train],
                    weight=data.weight[train],
                    strata=data.strata_codes[train],
                    ties=self.ties,
                    stratum_indices=_stratum_idx_train,
                )
                dev_full = cox_deviance(loglik_full, lsat_full)
                dev_train = cox_deviance(loglik_train, lsat_train)
                cvraw[i, j] = dev_full - dev_train
            # Normalize by held-out event weight to match glmnet
            # 4.1-8 Cox grouped deviance scale.
            cvraw[i, :] /= fold_event_weight[i]

            # Collect out-of-fold linear predictors for bootstrap SE.
            if use_bootstrap:
                for j in range(n_lam):
                    beta_j = fold_fit.coef_path_[j]
                    eta_hat[held_out, j] = (
                        data.X[held_out] @ beta_j
                        + data.offset[held_out]
                    )

        # CV statistics with saturated-lambda elimination (§4.3.3).
        cvm, cvsd, n_folds_used = self._compute_cv_statistics(
            cvraw, fold_event_weight,
        )

        # Bootstrap SE override (§4.3.2).
        if use_bootstrap:
            cvsd_boot = bootstrap_cv_se(
                eta_hat, data.start, data.stop, data.event,
                data.weight, data.strata_codes,
                n_bootstrap=self.n_bootstrap,
                random_state=self.random_state,
                ties=self.ties,
            )
            # Replace analytical SE with bootstrap SE; keep cvm and
            # the saturated-lambda NaN mask from the analytical pass.
            valid = np.isfinite(cvm)
            cvsd[valid] = cvsd_boot[valid]

        self.lambda_path_ = lambda_grid
        self.cv_mean_deviance_ = cvm
        self.cv_se_deviance_ = cvsd
        self.cv_n_folds_ = n_folds_used

        # Lambda selection: min and 1-SE rule.
        self.lambda_min_, self.lambda_1se_ = self._select_lambda(
            lambda_grid, cvm, cvsd,
        )

        # Final estimator at the selected lambda.
        chosen_lambda = (
            self.lambda_min_
            if self.se_rule == "min"
            else self.lambda_1se_
        )
        self.final_estimator_ = PenalizedCoxPH(
            lambda_path=chosen_lambda, **self._base_kwargs(),
        )
        self.final_estimator_.fit(
            data.X, event=data.event, start=data.start,
            stop=data.stop, strata=data.strata_codes,
            offset=data.offset, sample_weight=data.weight,
        )
        self.coef_ = self.final_estimator_.coef_
        self.n_obs_ = data.n_obs
        self.n_events_ = int(np.sum(data.event))
        self.feature_names_in_ = full_fit.feature_names_in_
        self.n_nonzero_path_ = full_fit.n_nonzero_path_
        self.full_fit_ = full_fit

        return self

    # _check_is_fitted, predict_*, summary are inherited from
    # _PenalizedCoxPHCVBase.
