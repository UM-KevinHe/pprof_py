"""User-facing CoxPH estimator, following scikit-learn conventions.

This module is the only place pandas, sklearn's BaseEstimator, and the
statistical engine (algorithms/, statistics/) meet -- everything below it
is plain NumPy so that a future distributed backend only has to replace
what happens inside `fit`, not the class's public surface.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ...data.survival_validation import validate_fit_inputs, validate_X
from ...data.survival_data import SurvivalData
from ...algorithms.survival.cox_likelihood import cox_partial_likelihood, precompute_stratum_indices
from ...algorithms.survival.optimization import newton_raphson
from ...algorithms.survival.ties import TieMethod
from ...inference.survival.inference import (
    covariance_from_information,
    standard_errors,
    wald_statistics,
    confidence_intervals,
)
from ...inference.survival.robust import cluster_score_residuals, robust_covariance
from ...inference.survival.baseline import compute_baseline_hazard
from ...inference.survival.residuals import martingale_residuals
from ...utils.numerical import col_means, safe_exp


class NotFittedError(RuntimeError):
    """Raised when a fitted-only attribute or method is accessed before `fit()`."""


class CoxPH(BaseEstimator):
    """Cox Proportional Hazards regression, fit by maximizing the
    (Breslow, by default) partial likelihood via Newton-Raphson.

    Designed to reproduce R's `survival::coxph()` as closely as
    numerically possible for the supported feature set: right-censored
    and left-truncated data, strata, offsets, and observation weights.
    See docs/R_COMPATIBILITY.md for the conventions matched and the
    known differences/limitations.

    Parameters
    ----------
    ties : {"breslow", "efron"} or TieMethod, default "breslow"
        Tie-breaking convention for the partial likelihood. Both
        "breslow" (this package's default, matching the existing
        `phregSHR` R workflow) and "efron" (R's own default) are
        implemented, each with a numba-compiled kernel used automatically
        when numba is installed. "exact" is recognized as forward-looking
        API surface (see algorithms/ties.py) but raises
        NotImplementedError if selected.
    fit_intercept : bool, default False
        A Cox model is typically fit *without* an intercept: a constant
        shift to eta is absorbed entirely into the (otherwise
        unidentified) baseline hazard, so an intercept term is not
        separately estimable from the partial likelihood alone. Exposed
        as an explicit, defaulted-off parameter for scikit-learn-style
        symmetry rather than because `True` is usually appropriate.
    max_iter, eps : Newton-Raphson controls -- see algorithms/optimization.py.
        Defaults (20, 1e-9) match `survival::coxph.control()`'s defaults.
    confidence_level : float, default 0.95
        Confidence level used for `confidence_intervals_` and `summary()`.
    robust : bool, default False
        If True, use a cluster-robust sandwich covariance instead of the
        model-based covariance. If `cluster` is not supplied, each input
        row is treated as its own cluster. For counting-process `(start, stop]`
        data with multiple rows per subject, supply the subject identifier
        as `cluster` so those rows are treated as one independent unit.

    Attributes (set by `fit`)
    -------------------------
    coef_, standard_errors_, covariance_, z_scores_, p_values_,
    confidence_intervals_, log_likelihood_, log_likelihood_null_,
    n_iter_, converged_, n_obs_, n_events_, n_features_in_,
    feature_names_in_, baseline_hazard_, martingale_residuals_,
    naive_covariance_, naive_standard_errors_, robust_covariance_,
    robust_, n_clusters_, cluster_labels_
    """

    def __init__(
        self,
        ties: Union[str, TieMethod] = "breslow",
        fit_intercept: bool = False,
        max_iter: int = 20,
        eps: float = 1e-9,
        confidence_level: float = 0.95,
        robust: bool = False,
    ):
        self.ties = ties
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.eps = eps
        self.confidence_level = confidence_level
        self.robust = robust

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
        cluster=None,
    ) -> "CoxPH":
        clean = validate_fit_inputs(
            X,
            duration=duration,
            event=event,
            start=start,
            stop=stop,
            strata=strata,
            offset=offset,
            sample_weight=sample_weight,
        )
        data = SurvivalData(**clean)

        if not isinstance(self.robust, (bool, np.bool_)):
            raise ValueError("robust must be a boolean")

        ties_name = self.ties if isinstance(self.ties, str) else getattr(self.ties, "name", None)
        # Supplying cluster explicitly requests cluster-robust inference.
        # This keeps the public API unambiguous and is the behavior needed by
        # Fine-Gray, where multiple pseudo-rows share the same original subject.
        robust_requested = bool(self.robust or cluster is not None)
        if robust_requested and str(ties_name).lower() == "exact":
            raise NotImplementedError(
                "Robust Cox variance is not implemented for ties=\"exact\"."
            )

        if cluster is not None:
            cluster_arr = np.asarray(cluster)
            if cluster_arr.ndim != 1 or cluster_arr.shape[0] != data.n_obs:
                raise ValueError(
                    f"cluster must be one-dimensional with length {data.n_obs}"
                )
            if cluster_arr.dtype.kind == "f" and not np.all(np.isfinite(cluster_arr)):
                raise ValueError("cluster must not contain NaN or infinite values")
        elif robust_requested:
            # No cluster supplied: treat each input row as its own cluster.
            # This is consistent across right-censored and counting-process
            # data. Users with multiple rows per subject should supply the
            # subject identifier explicitly so all rows are clustered together.
            cluster_arr = np.arange(data.n_obs, dtype=np.int64)
        else:
            cluster_arr = None

        if self.fit_intercept:
            data.X = np.column_stack([data.X, np.ones(data.n_obs)])
            data.feature_names = list(data.feature_names) + ["intercept"]

        p = data.n_features
        x_mean = col_means(data.X)
        Xc = data.X - x_mean  # numerical conditioning only -- see utils/numerical.py

        # Pre-compute per-stratum index arrays ONCE.  The NR loop calls
        # the objective ~4-20 times; without this, each call would
        # re-create 7000+ boolean masks on the full 4.8M-row array.
        _stratum_idx = precompute_stratum_indices(data.strata_codes)

        def objective(beta):
            return cox_partial_likelihood(
                Xc,
                data.start,
                data.stop,
                data.event,
                beta,
                offset=data.offset,
                weight=data.weight,
                strata=data.strata_codes,
                ties=self.ties,
                stratum_indices=_stratum_idx,
            )

        result = newton_raphson(objective, np.zeros(p), max_iter=self.max_iter, eps=self.eps)

        # Centering a constant shift into X changes nothing about beta_hat
        # itself (see utils/numerical.py) -- no back-transform of `beta` is
        # needed. What DOES need care is that every quantity computed below
        # that depends on the *absolute* value of eta (baseline hazard,
        # residuals, predictions) uses the ORIGINAL uncentered data.X, never
        # Xc, so that "baseline" means X=0 in the user's units, not X=mean(X).
        self.coef_ = result.beta
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        self.convergence_message_ = result.message
        self.log_likelihood_ = result.log_likelihood

        null_loglik, _, _ = cox_partial_likelihood(
            Xc,
            data.start,
            data.stop,
            data.event,
            np.zeros(p),
            offset=data.offset,
            weight=data.weight,
            strata=data.strata_codes,
            ties=self.ties,
            stratum_indices=_stratum_idx,
        )
        self.log_likelihood_null_ = null_loglik

        self.naive_covariance_ = covariance_from_information(result.information)
        self.naive_standard_errors_ = standard_errors(self.naive_covariance_)

        if robust_requested:
            eta_fit = data.X @ self.coef_ + data.offset
            cluster_scores, cluster_labels = cluster_score_residuals(
                data.X, data.start, data.stop, data.event, data.weight,
                eta_fit, data.strata_codes, cluster_arr, ties=self.ties,
            )
            self.covariance_ = robust_covariance(self.naive_covariance_, cluster_scores)
            self.robust_covariance_ = self.covariance_.copy()
            self.n_clusters_ = cluster_scores.shape[0]
            self.cluster_labels_ = cluster_labels
        else:
            self.covariance_ = self.naive_covariance_
            self.robust_covariance_ = None
            self.n_clusters_ = 0
            self.cluster_labels_ = None

        self.robust_ = robust_requested
        self.standard_errors_ = standard_errors(self.covariance_)
        self.z_scores_, self.p_values_ = wald_statistics(self.coef_, self.standard_errors_)
        ci_lower, ci_upper = confidence_intervals(self.coef_, self.standard_errors_, self.confidence_level)
        self.confidence_intervals_ = np.column_stack([ci_lower, ci_upper])

        self.n_obs_ = data.n_obs
        self.n_events_ = int(np.sum(data.event))
        self.n_features_in_ = p
        self.feature_names_in_ = np.array(data.feature_names)
        self._strata_labels = data.strata_labels

        eta_fit = data.X @ self.coef_ + data.offset
        # baseline_raw: hazard at (X=0, offset=0) exactly, using whichever
        # tie method fit the model -- this is the reference point
        # predict_cumulative_hazard's exp(eta) multiplier is designed to
        # pair with (see that method's docstring for why it must NOT use
        # the public, R-matching baseline_hazard_ computed further down).
        baseline_raw = compute_baseline_hazard(
            data.X,
            data.start,
            data.stop,
            data.event,
            eta_fit,
            data.weight,
            data.strata_codes,
            data.strata_labels,
            ties=self.ties,
        )
        # Martingale residuals use their own dedicated algorithm, not
        # baseline_raw -- Efron's per-tied-death correction has no
        # counterpart in the baseline hazard table itself; see
        # statistics/residuals.py.
        self.martingale_residuals_ = martingale_residuals(
            data.X,
            data.start,
            data.stop,
            data.event,
            eta_fit,
            data.weight,
            data.strata_codes,
            data.strata_labels,
            ties=self.ties,
        )

        # R's basehaz(fit, centered=FALSE) does NOT report the hazard at
        # (X=0, offset=0): survfit.coxph's reference curve is built from
        # risk scores centered on BOTH mean(X)@beta and mean(offset)
        # (weighted mean if weights were supplied), and basehaz only
        # divides back out the mean(X)@beta part -- the mean(offset) part
        # is silently left in. Net effect, after the mean(X)@beta terms
        # cancel: basehaz(centered=FALSE) == (hazard at X=0, offset=0) *
        # exp(mean(offset)), not the "purer" hazard at offset=0 a naive
        # reading of `centered=FALSE` would suggest. Matched here exactly
        # (verified against R's actual survfit.coxph/basehaz source, not
        # just its documentation) so this attribute is numerically
        # equivalent to R's; see docs/R_COMPATIBILITY.md, question 6.
        offset_mean = float(np.average(data.offset, weights=data.weight))
        self.baseline_hazard_ = baseline_raw.copy()
        self.baseline_hazard_["hazard"] = baseline_raw["hazard"] * np.exp(offset_mean)
        self.baseline_hazard_["survival"] = np.exp(-self.baseline_hazard_["hazard"])

        self._baseline_hazard_raw = baseline_raw

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise NotFittedError("This CoxPH instance is not fitted yet. Call `fit` first.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_linear(self, X, offset=None) -> np.ndarray:
        """X @ coef_ + offset -- usable directly as the offset for a
        second-stage CoxPH model (the two-stage SMR/SHR workflow)."""
        self._check_is_fitted()
        X_arr, _ = validate_X(X)
        if offset is None:
            offset_arr = np.zeros(X_arr.shape[0])
        else:
            offset_arr = np.asarray(offset, dtype=np.float64)
        return X_arr @ self.coef_ + offset_arr

    def predict_partial_hazard(self, X, offset=None) -> np.ndarray:
        """exp(X @ coef_ + offset)."""
        return safe_exp(self.predict_linear(X, offset=offset))

    def predict(self, X, offset=None) -> np.ndarray:
        """Alias for `predict_partial_hazard`, for scikit-learn-style symmetry."""
        return self.predict_partial_hazard(X, offset=offset)

    def predict_cumulative_hazard(self, X, offset=None, stratum=None) -> pd.DataFrame:
        """Per-subject cumulative hazard H_i(t) = H0(t) * exp(eta_i),
        returned as a DataFrame indexed by the baseline event times of
        the requested stratum, one column per row of `X`.

        `stratum` is required if the fitted model has more than one
        stratum (there is no single baseline hazard to apply otherwise);
        it is optional and defaults to the only stratum when the model
        is unstratified.

        Uses the raw baseline hazard at (X=0, offset=0) internally, NOT
        the `baseline_hazard_` public attribute -- the latter matches R's
        `basehaz(centered=FALSE)` convention of reporting the hazard at
        offset=mean(offset) (see `fit`), which would silently bias every
        prediction by exp(mean(offset)) if used here instead.
        """
        self._check_is_fitted()
        partial_hazard = self.predict_partial_hazard(X, offset=offset)
        baseline = self._baseline_hazard_raw
        if stratum is None:
            strata_present = baseline["stratum"].unique()
            if len(strata_present) != 1:
                raise ValueError(
                    "Model has multiple strata; pass `stratum=` to select which "
                    "stratum's baseline hazard to apply for prediction."
                )
            stratum = strata_present[0]
        strat_baseline = baseline[baseline["stratum"] == stratum].sort_values("time")
        if strat_baseline.empty:
            raise ValueError(f"No baseline hazard found for stratum={stratum!r}.")
        times = strat_baseline["time"].to_numpy()
        H0 = strat_baseline["hazard"].to_numpy()
        H = np.outer(H0, partial_hazard)
        return pd.DataFrame(H, index=pd.Index(times, name="time"))

    def predict_survival_function(self, X, offset=None, stratum=None) -> pd.DataFrame:
        """Per-subject baseline-relative survival S_i(t) = exp(-H_i(t))."""
        return np.exp(-self.predict_cumulative_hazard(X, offset=offset, stratum=stratum))

    # ------------------------------------------------------------------
    # Inference / summary
    # ------------------------------------------------------------------
    def score(self, X, duration=None, event=None, start=None, stop=None, strata=None, offset=None, sample_weight=None) -> float:
        """Partial log-likelihood of the given data evaluated at the
        already-fitted coefficients -- the natural analogue of
        scikit-learn's `score()` for a model fit by (partial) maximum
        likelihood. (A concordance-index-based score is a natural future
        addition; not implemented in this version.)
        """
        self._check_is_fitted()
        clean = validate_fit_inputs(
            X, duration=duration, event=event, start=start, stop=stop,
            strata=strata, offset=offset, sample_weight=sample_weight,
        )
        if self.fit_intercept:
            clean["X"] = np.column_stack([clean["X"], np.ones(clean["X"].shape[0])])
        loglik, _, _ = cox_partial_likelihood(
            clean["X"], clean["start"], clean["stop"], clean["event"], self.coef_,
            offset=clean["offset"], weight=clean["weight"], strata=clean["strata_codes"],
            ties=self.ties,
        )
        return loglik

    def summary(self) -> pd.DataFrame:
        """Coefficient table analogous to R's `summary(coxph_fit)$coefficients`
        plus confidence intervals. Numerical values are what's validated
        against R; this does not attempt to reproduce R's console formatting."""
        self._check_is_fitted()
        level_pct = f"{self.confidence_level:.0%}"
        return pd.DataFrame(
            {
                "coef": self.coef_,
                "exp(coef)": np.exp(self.coef_),
                "se(coef)": self.standard_errors_,
                "z": self.z_scores_,
                "p": self.p_values_,
                f"lower_{level_pct}": self.confidence_intervals_[:, 0],
                f"upper_{level_pct}": self.confidence_intervals_[:, 1],
            },
            index=self.feature_names_in_,
        )
