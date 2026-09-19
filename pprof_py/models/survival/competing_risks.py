"""User-facing competing-risks estimators.

Both classes here are thin wrappers around `CoxPH` -- see
`algorithms/finegray.py`'s module docstring for why competing-risks
regression reduces to an ordinary Cox fit either way:

* `CauseSpecificCoxPH` fits one ordinary `CoxPH` per cause, recoding
  every OTHER cause's events as censoring. No new statistics at all.
* `FineGrayPH` runs `finegray_transform` (the Fine-Gray subdistribution-
  hazard data expansion) and fits one ordinary, weighted, CLUSTERED
  `CoxPH` on the result -- clustered because the pseudo-observations
  `finegray_transform` produces are NOT independent across rows: several
  can share one original subject, so plain model-based standard errors
  would understate the true uncertainty. `cluster=<original subject>` is
  passed to `CoxPH.fit` for exactly this reason (see
  `statistics/robust.py` for the cluster-robust sandwich variance
    this relies on). `coef_` does not
  depend on which variance formula is used.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from ...algorithms.survival.finegray import finegray_transform
from .coxph import CoxPH, NotFittedError
from ...algorithms.survival.ties import TieMethod


class CauseSpecificCoxPH(BaseEstimator):
    """Cause-specific-hazards competing-risks regression.

    Fits one `CoxPH` per requested cause: modeling cause `k` treats every
    event of any OTHER cause as an ordinary censoring at the time it
    occurs (subjects who fail from cause 2 simply leave cause 1's risk
    set at that time, exactly like any other withdrawal) -- there is no
    new algorithm here, only a recoding of `event` repeated once per
    cause, so this class does not add anything `CoxPH` itself does not
    already validate; it exists for convenience (fit every cause in one
    call, index into the result by cause) and to make the "many separate
    CoxPH fits" pattern discoverable and hard to get wrong (e.g.
    forgetting to re-derive the per-cause event indicator).

    Parameters mirror `CoxPH` and are passed through unchanged to every
    per-cause fit.
    """

    def __init__(
        self,
        ties: Union[str, TieMethod] = "breslow",
        fit_intercept: bool = False,
        max_iter: int = 20,
        eps: float = 1e-9,
        confidence_level: float = 0.95,
    ):
        """Cause-specific Cox model for competing risks."""
        self.ties = ties
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.eps = eps
        self.confidence_level = confidence_level

    def _new_coxph(self) -> CoxPH:
        return CoxPH(
            ties=self.ties,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            eps=self.eps,
            confidence_level=self.confidence_level,
        )

    def fit(
        self,
        X,
        event,
        duration=None,
        start=None,
        stop=None,
        causes: Optional[Sequence] = None,
        strata=None,
        offset=None,
        sample_weight=None,
    ) -> "CauseSpecificCoxPH":
        """Fit one CoxPH per cause, recoding other causes as censored.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_samples, n_features)
        event : Series or ndarray
            0 = censored, any other value = cause label.
        duration, start, stop, strata, offset, sample_weight
            Same as ``CoxPH.fit``.
        causes : sequence or None
            Cause codes to fit; defaults to all distinct nonzero values
            in ``event``.

        Returns
        -------
        self
        """
        event_arr = event.to_numpy() if isinstance(event, (pd.Series, pd.Index)) else np.asarray(event)
        event_arr = event_arr.astype(np.float64)

        if causes is None:
            causes_resolved = sorted(np.unique(event_arr[event_arr != 0.0]).tolist())
        else:
            causes_resolved = list(causes)
            missing = [c for c in causes_resolved if not np.any(event_arr == float(c))]
            if missing:
                raise ValueError(f"`causes` includes value(s) not present in `event`: {missing}")
        if not causes_resolved:
            raise ValueError("No nonzero event codes found in `event` -- nothing to fit.")

        self.models_ = {}
        for cause in causes_resolved:
            event_k = (event_arr == float(cause)).astype(np.float64)
            model = self._new_coxph().fit(
                X,
                duration=duration,
                event=event_k,
                start=start,
                stop=stop,
                strata=strata,
                offset=offset,
                sample_weight=sample_weight,
            )
            self.models_[cause] = model

        self.causes_ = np.array(causes_resolved)
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "models_"):
            raise NotFittedError("This CauseSpecificCoxPH instance is not fitted yet. Call 'fit' first.")

    def __getitem__(self, cause) -> CoxPH:
        """`model[cause]` returns the fitted `CoxPH` for that cause --
        every `CoxPH` attribute and `predict_*`/`summary` method is
        available on it directly.
        """
        self._check_is_fitted()
        if cause not in self.models_:
            raise KeyError(f"No model was fit for cause {cause!r}; fitted causes are {self.causes_.tolist()}.")
        return self.models_[cause]

    def summary(self) -> pd.DataFrame:
        """Stacked summary of all cause-specific fits.

        Returns
        -------
        DataFrame
            Every cause's ``CoxPH.summary()`` concatenated with a
            leading ``cause`` column.
        """
        self._check_is_fitted()
        frames = []
        for cause in self.causes_.tolist():
            frame = self.models_[cause].summary().reset_index().rename(columns={"index": "covariate"})
            frame.insert(0, "cause", cause)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


class FineGrayPH(BaseEstimator):
    """Fine-Gray subdistribution-hazard regression for one cause of
    interest, via `finegray_transform` + a weighted, cluster-robust
    `CoxPH` fit (clustered on each pseudo-observation's original
    subject -- see the module docstring for why that matters here).

    `predict_survival_function` on a fitted instance returns the model's
    estimated subdistribution *survival* function; `1 -
    predict_survival_function(...)` is the estimated cumulative
    incidence function for `failcode` -- the quantity Fine-Gray
    regression exists to model directly (unlike cause-specific hazards,
    whose cumulative hazard does not transform into a CIF on its own in
    the presence of competing risks).
    """

    def __init__(
        self,
        ties: Union[str, TieMethod] = "breslow",
        fit_intercept: bool = False,
        max_iter: int = 20,
        eps: float = 1e-9,
        confidence_level: float = 0.95,
    ):
        """Fine-Gray subdistribution hazard model for competing risks."""
        self.ties = ties
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.eps = eps
        self.confidence_level = confidence_level

    def fit(
        self,
        X,
        event,
        failcode,
        duration=None,
        start=None,
        stop=None,
        id=None,
        strata=None,
        sample_weight=None,
    ) -> "FineGrayPH":
        """Fit Fine-Gray subdistribution hazard regression.

        Internally runs ``finegray_transform`` then fits a weighted,
        cluster-robust ``CoxPH`` on the resulting pseudo-observations.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_samples, n_features)
        event : Series or ndarray
            0 = censored, positive = cause label.
        failcode : scalar
            The cause of interest (other causes become censored).
        duration, start, stop : Series or ndarray or None
            Follow-up time specification.
        id : Series or ndarray or None
            Subject identifiers (needed for correct clustering).
        strata, sample_weight : Series or ndarray or None

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            feature_names = [str(c) for c in X.columns]
            X_arr = X.to_numpy(dtype=np.float64, copy=True)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        n = X_arr.shape[0]

        if duration is not None:
            if start is not None or stop is not None:
                raise ValueError("Pass either `duration`, or (`start`, `stop`) -- not both.")
            stop_arr = np.asarray(duration, dtype=np.float64)
            start_arr = np.zeros(n, dtype=np.float64)
        else:
            if stop is None:
                raise ValueError("Must supply either `duration` or (`start`, `stop`).")
            stop_arr = np.asarray(stop, dtype=np.float64)
            start_arr = np.zeros(n, dtype=np.float64) if start is None else np.asarray(start, dtype=np.float64)

        event_arr = event.to_numpy() if isinstance(event, (pd.Series, pd.Index)) else np.asarray(event)
        event_arr = event_arr.astype(np.float64)

        strata_arr = None
        if strata is not None:
            strata_arr = strata.to_numpy() if isinstance(strata, (pd.Series, pd.Index)) else np.asarray(strata)

        sw_arr = None
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=np.float64)

        fg = finegray_transform(
            start_arr, stop_arr, event_arr, failcode=failcode,
            id=id, strata=strata_arr, sample_weight=sw_arr,
        )
        if fg.n_obs == 0:
            raise ValueError(f"failcode={failcode!r} produced no pseudo-observations -- check `event`.")

        self.source_row_ = fg.row
        self.source_subject_ = fg.subject
        self.failcode_ = float(failcode)

        # Re-wrap as a DataFrame with the ORIGINAL column names before
        # handing it back to CoxPH.fit -- otherwise CoxPH sees a bare
        # ndarray and falls back to generic x0/x1/... names, which would
        # then surface (wrongly) in every attribute and method this
        # class delegates to `self.model_` for, e.g. `summary()`.
        X_expanded = pd.DataFrame(X_arr[fg.row], columns=feature_names)
        strata_expanded = strata_arr[fg.row] if strata_arr is not None else None

        self.model_ = CoxPH(
            ties=self.ties,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            eps=self.eps,
            confidence_level=self.confidence_level,
        ).fit(
            X_expanded,
            start=fg.start,
            stop=fg.stop,
            event=fg.status,
            strata=strata_expanded,
            sample_weight=fg.weight,
            cluster=fg.subject,
        )

        # Convenience top-level access to the fitted quantities most
        # people want without reaching into `.model_` -- everything else
        # (predict_*, summary, martingale_residuals_, baseline_hazard_,
        # ...) is available there directly.
        for attr in (
            "coef_", "standard_errors_", "covariance_", "z_scores_", "p_values_",
            "confidence_intervals_", "log_likelihood_", "log_likelihood_null_",
            "n_iter_", "converged_", "n_events_",
        ):
            setattr(self, attr, getattr(self.model_, attr))
        self.n_obs_ = n  # the ORIGINAL subject count, not the expanded pseudo-observation count
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = np.array(feature_names)
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise NotFittedError("This FineGrayPH instance is not fitted yet. Call 'fit' first.")

    def predict_linear(self, X, offset=None) -> np.ndarray:
        """Linear predictor from the underlying CoxPH fit.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        offset : ndarray or None

        Returns
        -------
        ndarray, shape (n_new,)
        """
        self._check_is_fitted()
        return self.model_.predict_linear(X, offset=offset)

    def predict_partial_hazard(self, X, offset=None) -> np.ndarray:
        """Partial hazard ``exp(linear_predictor)``.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        offset : ndarray or None

        Returns
        -------
        ndarray, shape (n_new,)
        """
        self._check_is_fitted()
        return self.model_.predict_partial_hazard(X, offset=offset)

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
        self._check_is_fitted()
        return self.model_.predict(X, offset=offset)

    def predict_cumulative_hazard(self, X, offset=None, stratum=None) -> pd.DataFrame:
        """Estimated *subdistribution* cumulative hazard for `failcode`."""
        self._check_is_fitted()
        return self.model_.predict_cumulative_hazard(X, offset=offset, stratum=stratum)

    def predict_survival_function(self, X, offset=None, stratum=None) -> pd.DataFrame:
        """`1 - ` this is the estimated cumulative incidence function
        for `failcode` -- see the class docstring.
        """
        self._check_is_fitted()
        return self.model_.predict_survival_function(X, offset=offset, stratum=stratum)

    def summary(self) -> pd.DataFrame:
        """Coefficient summary from the underlying Fine-Gray CoxPH fit.

        Returns
        -------
        DataFrame
        """
        self._check_is_fitted()
        return self.model_.summary()