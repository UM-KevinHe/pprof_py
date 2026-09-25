"""Shared-frailty Cox proportional hazards regression.

``FrailtyCoxPH`` is the user-facing estimator for a shared Gamma frailty
model.  The model uses a multiplicative cluster effect

    h_ig(t) = z_g h_0(t) exp(X_i beta + offset_i),

where ``z_g`` has mean 1 and variance ``theta``.  Estimation is performed by
an EM algorithm: each M-step is an ordinary weighted ``CoxPH`` fit, so the
existing Cox likelihood remains the single implementation of risk sets,
strata, weights, offsets, and Breslow/Efron ties.

This first implementation intentionally supports a single shared Gamma
frailty term.  Gaussian/log-normal frailty and nested or multiple frailty
terms are future extensions rather than hidden special cases.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from ...base import ProviderModel
from ...exceptions import ConvergenceWarning

from ...data.survival_validation import validate_fit_inputs
from ...data.survival_data import SurvivalData
from ...inference.survival.baseline import compute_baseline_hazard
from ...algorithms.survival.ties import TieMethod
from ...algorithms.survival.frailty import (
    gamma_posterior_moments,
    group_exposure,
    update_gamma_theta,
)
from ...exceptions import NotFittedError
from .coxph import CoxPH


class FrailtyCoxPH(ProviderModel):
    """Shared Gamma-frailty Cox Proportional Hazards regression.

    The frailty is modeled as a positive, mean-one Gamma random effect shared
    by all rows belonging to the same ``frailty`` group.  The model is fit by
    EM, with an ordinary weighted ``CoxPH`` fit in each M-step.

    Parameters
    ----------
    ties : {"breslow", "efron"} or TieMethod, default "breslow"
        Tie-breaking convention used by the underlying Cox model.
    theta : float or None, default None
        Initial (and, when ``theta_fixed=True``, fixed) frailty variance.
        If None, starts at 0.1.
    theta_fixed : bool, default False
        If True, do not update the frailty variance during EM.
    max_iter : int, default 50
        Maximum number of outer EM iterations. EM's convergence rate for
        the frailty variance is slower when theta is small -- low
        cluster-level heterogeneity carries little information about
        theta, so the E/M alternation contracts slowly even though each
        individual iteration is cheap. A ``ConvergenceWarning`` is raised
        if ``max_iter`` is reached without meeting ``tol``; in that case
        ``theta_`` is typically biased away from zero relative to where a
        longer run would settle, so prefer inspecting ``theta_history_``
        and refitting with a larger ``max_iter`` over trusting the point
        estimate as-is.
    tol : float, default 1e-6
        Relative convergence tolerance for beta and theta.
    cox_max_iter : int, default 20
        Newton-Raphson iterations used by each conditional Cox fit.
    cox_eps : float, default 1e-9
        Convergence tolerance passed to each conditional Cox fit.
    confidence_level : float, default 0.95
        Confidence level for the conditional fixed-effect intervals.
    """

    def __init__(
        self,
        ties: Union[str, TieMethod] = "breslow",
        theta: Optional[float] = None,
        theta_fixed: bool = False,
        max_iter: int = 50,
        tol: float = 1e-6,
        cox_max_iter: int = 20,
        cox_eps: float = 1e-9,
        confidence_level: float = 0.95,
    ):
        self.ties = ties
        self.theta = theta
        self.theta_fixed = theta_fixed
        self.max_iter = max_iter
        self.tol = tol
        self.cox_max_iter = cox_max_iter
        self.cox_eps = cox_eps
        self.confidence_level = confidence_level

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
        frailty=None,
    ) -> "FrailtyCoxPH":
        """Fit the shared Gamma-frailty Cox model.

        ``frailty`` is a one-dimensional group identifier, one value per
        input row.  Repeated identifiers define the shared random effect.
        For counting-process data, pass the subject/facility identifier that
        identifies the independent frailty unit.

        The current implementation does not accept a cluster-robust
        covariance request in addition to frailty.  Frailty and cluster-
        robust GEE inference represent different correlation models.
        """
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

        if frailty is None:
            raise ValueError("`frailty` must be supplied as one group identifier per row")
        groups = np.asarray(frailty)
        if groups.ndim != 1 or groups.shape[0] != data.n_obs:
            raise ValueError(
                f"`frailty` must be one-dimensional with length {data.n_obs}"
            )
        if pd.isna(groups).any():
            raise ValueError("`frailty` must not contain missing values")
        if groups.dtype.kind == "f" and not np.all(np.isfinite(groups)):
            raise ValueError("`frailty` must not contain NaN or infinite values")

        labels, codes = np.unique(groups, return_inverse=True)
        n_groups = labels.shape[0]
        if n_groups < 2:
            raise ValueError("At least two frailty groups are required")

        if not isinstance(self.theta_fixed, (bool, np.bool_)):
            raise ValueError("theta_fixed must be a boolean")
        if self.max_iter < 1 or self.cox_max_iter < 1:
            raise ValueError("max_iter and cox_max_iter must be positive")
        if self.tol <= 0 or self.cox_eps <= 0:
            raise ValueError("tol and cox_eps must be strictly positive")

        theta = 0.1 if self.theta is None else float(self.theta)
        if theta <= 0 or not np.isfinite(theta):
            raise ValueError("theta must be finite and strictly positive")

        base_weight = np.asarray(data.weight, dtype=np.float64)
        posterior_mean = np.ones(n_groups, dtype=np.float64)
        beta_old = np.zeros(data.n_features, dtype=np.float64)
        theta_history = [theta]
        beta_history = []
        converged = False
        conditional_model = None

        for iteration in range(1, self.max_iter + 1):
            frailty_offset = data.offset + np.log(
                np.maximum(posterior_mean[codes], np.finfo(float).tiny)
            )
            conditional_model = CoxPH(
                ties=self.ties,
                fit_intercept=False,
                max_iter=self.cox_max_iter,
                eps=self.cox_eps,
                confidence_level=self.confidence_level,
                robust=False,
            )
            conditional_model.fit(
                data.X,
                start=data.start,
                stop=data.stop,
                event=data.event,
                strata=data.strata_codes,
                offset=frailty_offset,
                sample_weight=base_weight,
            )

            beta = conditional_model.coef_.copy()
            eta = data.X @ beta + data.offset
            exposure = group_exposure(
                eta,
                conditional_model._baseline_hazard_raw,
                codes,
                n_groups,
                data.start,
                data.stop,
                base_weight,
                data.strata_codes,
                data.strata_labels,
            )
            event_count = np.bincount(
                codes,
                weights=base_weight * data.event,
                minlength=n_groups,
            ).astype(np.float64)

            new_mean, new_log_mean = gamma_posterior_moments(
                theta, event_count, exposure
            )
            new_theta = theta if self.theta_fixed else update_gamma_theta(
                theta, new_mean, new_log_mean
            )

            beta_change = np.max(np.abs(beta - beta_old)) if beta.size else 0.0
            beta_scale = max(1.0, np.max(np.abs(beta_old))) if beta.size else 1.0
            theta_change = abs(new_theta - theta) / max(1.0, abs(theta))

            beta_history.append(beta.copy())
            theta_history.append(new_theta)
            beta_old = beta
            posterior_mean = new_mean
            theta = new_theta

            if beta_change / beta_scale < self.tol and theta_change < self.tol:
                converged = True
                break

        if conditional_model is None:
            raise RuntimeError("Frailty fitting did not perform a Cox M-step")

        # Polish the fixed effects once more at the final posterior frailties.
        final_frailty_offset = data.offset + np.log(
            np.maximum(posterior_mean[codes], np.finfo(float).tiny)
        )
        conditional_model = CoxPH(
            ties=self.ties,
            fit_intercept=False,
            max_iter=self.cox_max_iter,
            eps=self.cox_eps,
            confidence_level=self.confidence_level,
            robust=False,
        )
        conditional_model.fit(
            data.X,
            start=data.start,
            stop=data.stop,
            event=data.event,
            strata=data.strata_codes,
            offset=final_frailty_offset,
            sample_weight=base_weight,
        )

        # Re-evaluate the posterior moments at the polished beta/baseline.
        eta_final = data.X @ conditional_model.coef_ + data.offset
        exposure_final = group_exposure(
            eta_final,
            conditional_model._baseline_hazard_raw,
            codes,
            n_groups,
            data.start,
            data.stop,
            base_weight,
            data.strata_codes,
            data.strata_labels,
        )
        event_count_final = np.bincount(
            codes,
            weights=base_weight * data.event,
            minlength=n_groups,
        ).astype(np.float64)
        posterior_mean, posterior_log_mean = gamma_posterior_moments(
            theta, event_count_final, exposure_final
        )
        if not self.theta_fixed:
            theta = update_gamma_theta(theta, posterior_mean, posterior_log_mean)

        self.coef_ = conditional_model.coef_.copy()
        self.standard_errors_ = conditional_model.standard_errors_.copy()
        self.covariance_ = conditional_model.covariance_.copy()
        self.z_scores_ = conditional_model.z_scores_.copy()
        self.p_values_ = conditional_model.p_values_.copy()
        self.confidence_intervals_ = conditional_model.confidence_intervals_.copy()
        self.conditional_standard_errors_ = self.standard_errors_.copy()
        self.conditional_covariance_ = self.covariance_.copy()

        self.theta_ = float(theta)
        self.frailty_variance_ = self.theta_
        self.frailty_ = posterior_mean.copy()
        self.frailty_log_ = np.log(np.maximum(self.frailty_, np.finfo(float).tiny))
        self.frailty_expected_log_ = posterior_log_mean.copy()
        self.frailty_labels_ = labels
        self.frailty_codes_ = codes
        self.n_groups_ = n_groups
        self.n_frailty_groups_ = n_groups

        self.log_likelihood_ = conditional_model.log_likelihood_
        self.log_likelihood_null_ = conditional_model.log_likelihood_null_
        self.n_iter_ = iteration
        self.converged_ = converged
        self.convergence_message_ = (
            "converged" if converged else f"reached max_iter={self.max_iter}"
        )
        if not converged:
            warnings.warn(
                f"FrailtyCoxPH EM did not converge within max_iter={self.max_iter} "
                "outer iterations (see theta_history_/beta_history_ for the "
                "trajectory). This is most often seen when the fitted frailty "
                "variance is small: low cluster-level heterogeneity gives EM "
                "little information about theta, which slows its linear "
                "convergence rate and can require several hundred iterations "
                "even though each individual iteration is cheap. theta_ from a "
                "non-converged fit is typically biased away from zero (away "
                "from the no-frailty case) rather than arbitrary -- refit with "
                "a larger max_iter, or with theta_fixed=True at a value from a "
                "longer run, if this matters for your use case.",
                ConvergenceWarning,
                stacklevel=2,
            )
        self.n_obs_ = data.n_obs
        self.n_events_ = int(np.sum(data.event))
        self.n_features_in_ = data.n_features
        self.feature_names_in_ = np.asarray(data.feature_names)
        self._baseline_hazard_raw = conditional_model._baseline_hazard_raw.copy()
        self.baseline_hazard_ = self._baseline_hazard_raw.copy()
        self.baseline_hazard_["survival"] = np.exp(-self.baseline_hazard_["hazard"])
        self.martingale_residuals_ = conditional_model.martingale_residuals_.copy()
        self.theta_history_ = np.asarray(theta_history, dtype=np.float64)
        self.beta_history_ = np.asarray(beta_history, dtype=np.float64)
        self.n_cox_iterations_ = conditional_model.n_iter_
        self.cox_converged_ = conditional_model.converged_
        self.inference_approximate_ = True
        self.inference_note_ = (
            "standard_errors_ are conditional Cox-model standard errors at the "
            "final posterior frailty weights; a full observed-information/Louis "
            "variance calculation is not yet implemented."
        )
        self._strata_labels = data.strata_labels
        self._conditional_model = conditional_model

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise NotFittedError("This FrailtyCoxPH instance is not fitted yet. Call `fit` first.")

    def predict_linear(self, X, offset=None) -> np.ndarray:
        """Return the fixed-effects linear predictor ``X @ beta + offset``."""
        self._check_is_fitted()
        return self._conditional_model.predict_linear(X, offset=offset)

    def predict_partial_hazard(self, X, offset=None, frailty=None) -> np.ndarray:
        """Return ``exp(X @ beta + offset)`` with optional fitted frailty."""
        self._check_is_fitted()
        baseline_risk = self._conditional_model.predict_partial_hazard(X, offset=offset)
        if frailty is None:
            return baseline_risk
        groups = np.asarray(frailty)
        labels = self.frailty_labels_
        index = {label: i for i, label in enumerate(labels)}
        try:
            z = np.asarray([self.frailty_[index[value]] for value in groups], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"Unknown frailty group {exc.args[0]!r}") from exc
        return baseline_risk * z

    def predict(self, X, offset=None, frailty=None) -> np.ndarray:
        """Alias for ``predict_partial_hazard``."""
        return self.predict_partial_hazard(X, offset=offset, frailty=frailty)

    def predict_cumulative_hazard(self, X, offset=None, stratum=None, frailty=None) -> pd.DataFrame:
        """Return subject-specific cumulative hazard from the fitted frailty model."""
        self._check_is_fitted()
        partial_hazard = self.predict_partial_hazard(X, offset=offset, frailty=frailty)
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
        return pd.DataFrame(
            np.outer(H0, partial_hazard),
            index=pd.Index(times, name="time"),
        )

    def predict_survival_function(self, X, offset=None, stratum=None, frailty=None) -> pd.DataFrame:
        """Return subject-specific survival from the fitted frailty model."""
        return np.exp(
            -self.predict_cumulative_hazard(
                X, offset=offset, stratum=stratum, frailty=frailty
            )
        )

    def summary(self) -> pd.DataFrame:
        """Return a fixed-effect coefficient table in the CoxPH style."""
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
