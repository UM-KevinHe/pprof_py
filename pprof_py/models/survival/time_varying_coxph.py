"""Time-varying-coefficient Cox regression.

``TimeVaryingCoxPH`` extends the ordinary Cox model by allowing selected
coefficients to multiply a time-dependent transform of the corresponding
covariate, e.g. ``x * log(t + 20)``.  The transformed term is evaluated at the
actual event time inside the partial-likelihood calculation; it is not
constructed from each subject's final follow-up time.
"""
from __future__ import annotations

from typing import Mapping, Union

import numpy as np
import pandas as pd
from ...base import ProviderModel

from ...data.survival_validation import validate_fit_inputs
from ...data.survival_data import SurvivalData
from ...algorithms.survival.optimization import newton_raphson
from ...algorithms.survival.cox_likelihood import precompute_stratum_indices
from ...algorithms.survival.time_varying import time_varying_cox_partial_likelihood, _resolve_transform_specs
from ...algorithms.survival.ties import TieMethod
from ...inference.survival.inference import covariance_from_information, standard_errors, wald_statistics, confidence_intervals
from ...exceptions import NotFittedError


class TimeVaryingCoxPH(ProviderModel):
    """Cox regression with selected time-varying coefficients.

    Parameters
    ----------
    time_transforms : mapping
        Maps an original feature name or integer column index to a callable
        ``f(x, t)``. The fitted model adds the term ``x * f(x, t)`` with its
        own coefficient. For example::

            {"karno": lambda x, t: np.log(t + 20.0)}

        gives ``beta1 * karno + beta2 * karno * log(t + 20)``.
    ties : {"breslow", "efron"}, default "breslow"
        Tie convention for the partial likelihood.

    Notes
    -----
    This estimator is intentionally separate from :class:`CoxPH` because a
    time-varying coefficient changes the likelihood's design vector at each
    event time. It still uses the same validation, Newton optimizer,
    `(start, stop]` risk-set semantics, strata, weights, and offsets.
    """

    def __init__(
        self,
        time_transforms: Mapping,
        ties: Union[str, TieMethod] = "breslow",
        fit_intercept: bool = False,
        max_iter: int = 20,
        eps: float = 1e-9,
        confidence_level: float = 0.95,
    ):
        self.time_transforms = time_transforms
        self.ties = ties
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.eps = eps
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
    ) -> "TimeVaryingCoxPH":
        clean = validate_fit_inputs(
            X, duration=duration, event=event, start=start, stop=stop,
            strata=strata, offset=offset, sample_weight=sample_weight,
        )
        data = SurvivalData(**clean)
        if self.fit_intercept:
            raise ValueError("fit_intercept=True is not supported for TimeVaryingCoxPH")
        ties_name = self.ties if isinstance(self.ties, str) else getattr(self.ties, "name", None)
        ties_name = str(ties_name).lower()
        if ties_name not in {"breslow", "efron"}:
            raise NotImplementedError("TimeVaryingCoxPH supports ties='breslow' or 'efron'.")

        specs = _resolve_transform_specs(data.X, self.time_transforms, data.feature_names)
        p = data.n_features + len(specs)
        # NOTE (perf, not correctness): this validates strata the same way
        # CoxPH does, but its result is discarded -- time_varying_cox_
        # partial_likelihood re-derives stratum membership via
        # iter_stratum_indices (boolean masking) on every Newton iteration
        # instead of reusing these precomputed indices. The base CoxPH fit
        # specifically moved away from repeated per-iteration boolean
        # masking in favor of indices computed once per fit; this estimator
        # doesn't yet get that optimization, which is likely to matter at
        # the same production scale (millions of rows, thousands of strata)
        # where it mattered for CoxPH. Wiring precompute_stratum_indices's
        # actual output through to time_varying_cox_partial_likelihood
        # (rather than only calling it for validation) would close this gap,
        # but touches the hot loop in a file outside this review's scope.
        _ = precompute_stratum_indices(data.strata_codes)  # validates/access pattern consistently with CoxPH

        def objective(beta):
            return time_varying_cox_partial_likelihood(
                data.X,
                data.start,
                data.stop,
                data.event,
                beta,
                data.offset,
                data.weight,
                data.strata_codes,
                self.time_transforms,
                data.feature_names,
                ties=ties_name,
            )

        result = newton_raphson(objective, np.zeros(p), max_iter=self.max_iter, eps=self.eps)
        self.coef_ = result.beta
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        self.convergence_message_ = result.message
        self.log_likelihood_ = result.log_likelihood
        self.covariance_ = covariance_from_information(result.information)
        self.standard_errors_ = standard_errors(self.covariance_)
        self.z_scores_, self.p_values_ = wald_statistics(self.coef_, self.standard_errors_)
        lo, hi = confidence_intervals(self.coef_, self.standard_errors_, self.confidence_level)
        self.confidence_intervals_ = np.column_stack([lo, hi])

        self.feature_names_in_ = np.asarray(
            list(data.feature_names) + [f"tt({name})" for _, name, _ in specs]
        )
        self.n_features_in_ = p
        self.n_obs_ = data.n_obs
        self.n_events_ = int(np.sum(data.event))
        self._strata_labels = data.strata_labels
        self.time_transform_features_ = [name for _, name, _ in specs]
        self.time_transform_functions_ = [func for _, _, func in specs]

        # A generic one-dimensional baseline-hazard table and ordinary
        # martingale residuals are not directly interpretable for a beta(t)
        # model because the design vector changes with event time. Keep these
        # out of the public fitted attributes rather than reporting quantities
        # that would silently ignore the time-varying terms.
        self._data_ = data
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "coef_"):
            raise NotFittedError("TimeVaryingCoxPH is not fitted yet. Call `fit` first.")

    def coefficient_at(self, t) -> pd.Series:
        """Return the effective coefficient vector beta_j(t).

        This assumes each registered time transform is a function of ``t``
        alone (the documented, common case, e.g. ``lambda x, t: log(t+20)``)
        and evaluates it at a neutral ``x=1`` accordingly. If a transform
        genuinely depends on ``x`` (the partial-likelihood fit itself
        supports this; this convenience method does not), there is no
        single well-defined "coefficient at time t" independent of which
        subject's x you mean, and the value returned here -- the transform
        evaluated at x=1 -- may not correspond to any actual subject.
        """
        self._check_is_fitted()
        p0 = self._data_.n_features
        beta = self.coef_[:p0].copy()
        feature_names = list(self._data_.feature_names)
        for k, func in enumerate(self.time_transform_functions_):
            # Effective coefficient is beta_static + beta_tt * g(t).
            # For the API, g(t) is evaluated with a neutral x=1.
            g = float(np.asarray(func(np.array([1.0]), float(t))).reshape(-1)[0])
            j = feature_names.index(self.time_transform_features_[k])
            beta[j] += self.coef_[p0 + k] * g
        return pd.Series(beta, index=feature_names, name=f"beta({t})")

    def summary(self) -> pd.DataFrame:
        """Return the coefficient table including static and time terms."""
        self._check_is_fitted()
        level = f"{self.confidence_level:.0%}"
        return pd.DataFrame(
            {
                "coef": self.coef_,
                "exp(coef)": np.exp(self.coef_),
                "se(coef)": self.standard_errors_,
                "z": self.z_scores_,
                "p": self.p_values_,
                f"lower_{level}": self.confidence_intervals_[:, 0],
                f"upper_{level}": self.confidence_intervals_[:, 1],
            },
            index=self.feature_names_in_,
        )
