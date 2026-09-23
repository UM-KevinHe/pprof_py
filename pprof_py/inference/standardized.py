"""Model-backed standardized measures for fixed-effect logistic provider models.

:func:`standardized_measure` turns a fitted model into a :class:`MeasureFrame`
(one estimate and standard error per provider) for a direct or indirect
standardized rate or ratio, or for the provider effect itself. It also records
the measure's reference value (its value when the provider effect equals the
reference effect gamma_0), so ``z_statistic(null_value="reference")`` tests
the same hypothesis as a test of the provider effect.

Definitions (``p(g, i) = sigmoid(g + xbeta_i)``, weights ``w_i`` = binomial
trials, sums over the standard population unless stated):

* direct rate: ``sum_i w_i p(gamma_j, i) / sum_i w_i``
* direct ratio: ``sum_i w_i p(gamma_j, i) / E_pop`` with ``E_pop`` the population's total events
* indirect ratio: ``O_j / E_j``, sums over provider j's own observations, ``E_j`` at gamma_0
* indirect rate: indirect ratio times the population's crude rate
* gamma: the fitted provider effect

Standard errors: direct measures and gamma use the delta method on
``SE(gamma_j)`` (model-based or cluster-robust). Indirect measures use the
Poisson-binomial variance of ``O_j``, evaluated at gamma_0 (the default,
score-type) or at the provider's fitted gamma_j (Wald-type). With the variance
at gamma_0 and the identity scale (the default for indirect measures), the
z-statistic is the score statistic ``(O_j - E_j) / sqrt(Var_0(O_j))``. In
simulations with unequal provider sizes this was the only indirect variant
that held the nominal error rate; the fitted variance and the log scale were
anti-conservative for small providers, and the log scale cannot test providers
with no events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..utils.numerical import sigmoid
from .zstat import MeasureFrame

__all__ = ["StandardPopulation", "standardized_measure", "at_bound", "MEASURES"]

MEASURES = ("direct_rate", "direct_ratio", "indirect_rate", "indirect_ratio", "gamma")


@dataclass(frozen=True)
class StandardPopulation:
    """The population over which direct standardization averages.

    Attributes
    ----------
    xbeta : numpy.ndarray
        Covariate linear predictor of each population row.
    weight : numpy.ndarray
        Binomial trials of each row (1 for Bernoulli data).
    events : float or None
        Total events in the population, used for the population's crude rate
        (direct ratio, indirect rate). ``None`` when unknown.
    """

    xbeta: np.ndarray
    weight: np.ndarray
    events: Optional[float]

    @classmethod
    def from_model(cls, model) -> "StandardPopulation":
        """The model's own observations."""
        xb = np.asarray(model.xbeta_, dtype=np.float64).ravel()
        n_obs = getattr(model, "N_", None)
        w = np.ones(xb.size) if n_obs is None else np.asarray(n_obs, dtype=np.float64).ravel()
        return cls(xbeta=xb, weight=w, events=float(np.sum(model.outcome_)))

    def extend(self, *, xbeta, weight, events: Optional[float] = None) -> "StandardPopulation":
        """Add rows (scalars or arrays) to the population.

        For example, observations excluded from the model fit can still count
        in the population: ``extend(xbeta=0.0, weight=n_excluded, events=y_excluded)``.
        Leaving ``events`` as ``None`` makes the population's total events
        unknown, which only matters for measures that need it.
        """
        xb = np.atleast_1d(np.asarray(xbeta, dtype=np.float64)).ravel()
        w = np.atleast_1d(np.asarray(weight, dtype=np.float64)).ravel()
        if xb.size == 1 and w.size > 1:
            xb = np.full(w.size, xb[0])
        if w.size == 1 and xb.size > 1:
            w = np.full(xb.size, w[0])
        if xb.size != w.size:
            raise ValueError("xbeta and weight must have matching lengths.")
        total = None if (self.events is None or events is None) else self.events + float(events)
        return StandardPopulation(np.r_[self.xbeta, xb], np.r_[self.weight, w], total)

    @property
    def total_weight(self) -> float:
        return float(np.sum(self.weight))


def _reference_gamma(gamma: np.ndarray, sizes: np.ndarray, reference) -> float:
    if isinstance(reference, str):
        if reference == "median":
            return float(np.median(gamma))
        if reference == "mean":
            return float(np.average(gamma, weights=sizes))
        raise ValueError("reference must be 'median', 'mean', or a number.")
    return float(reference)


def _gamma_se(model, variance: str) -> np.ndarray:
    if variance == "model":
        return np.sqrt(np.asarray(model.variances_["gamma"], dtype=np.float64).ravel())
    if variance == "robust":
        if getattr(model, "robust_variances_", None) is None:
            raise ValueError("Robust variances are not available; fit the model with obs_id_var.")
        return np.sqrt(np.asarray(model.robust_variances_["gamma"], dtype=np.float64).ravel())
    raise ValueError("variance must be 'model' or 'robust'.")


def standardized_measure(
    model,
    measure: str = "direct_rate",
    *,
    population: Optional[StandardPopulation] = None,
    reference: Union[str, float] = "median",
    variance: str = "model",
    indirect_variance: str = "null",
) -> MeasureFrame:
    """Per-provider standardized measure and standard error from a fitted model.

    Parameters
    ----------
    model : fitted LogisticFixedEffectModel
    measure : {"direct_rate", "direct_ratio", "indirect_rate", "indirect_ratio", "gamma"}
    population : StandardPopulation, optional
        Standard population for direct standardization and for the population
        rate; defaults to the model's own observations.
    reference : "median", "mean", or float
        The reference provider effect gamma_0: the median of the fitted
        effects, their size-weighted mean, or a value on the effect scale.
        It sets the expected counts of indirect measures and every measure's
        reference value.
    variance : {"model", "robust"}
        Standard error of the fitted effects, for direct measures and gamma.
        ``"robust"`` needs a model fitted with ``obs_id_var``.
    indirect_variance : {"null", "fitted"}
        Where the variance of ``O_j`` is evaluated for indirect measures: at
        gamma_0 (default; on the identity scale the test is then the score
        test of gamma_j = gamma_0) or at the provider's fitted gamma_j.

    Returns
    -------
    MeasureFrame
        Indexed by provider, with ``reference_value`` set.
    """
    if getattr(model, "coefficients_", None) is None or getattr(model, "variances_", None) is None:
        raise ValueError("The model must be fitted first.")
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}.")
    if indirect_variance not in ("null", "fitted"):
        raise ValueError("indirect_variance must be 'null' or 'fitted'.")
    if variance == "robust" and measure in ("indirect_rate", "indirect_ratio"):
        raise ValueError("Indirect measures use the Poisson-binomial variance of observed counts "
                         "(see indirect_variance); variance='robust' applies to direct measures and gamma.")

    gamma = np.asarray(model.coefficients_["gamma"], dtype=np.float64).ravel()
    sizes = np.asarray(model.group_sizes_, dtype=np.float64).ravel()
    g0 = _reference_gamma(gamma, sizes, reference)
    pop = StandardPopulation.from_model(model) if population is None else population

    def _population_events() -> float:
        if pop.events is None:
            raise ValueError(f"measure={measure!r} needs the population's total events; pass events= to "
                             "StandardPopulation.extend.")
        return pop.events

    if measure == "gamma":
        return MeasureFrame.from_arrays(gamma, _gamma_se(model, variance), model.groups_,
                                        measure=measure, reference_value=g0)

    if measure in ("direct_rate", "direct_ratio"):
        se_gamma = _gamma_se(model, variance)
        denominator = pop.total_weight if measure == "direct_rate" else _population_events()
        est = np.empty(gamma.size)
        se = np.empty(gamma.size)
        for j, g in enumerate(gamma):
            p = sigmoid(g + pop.xbeta)
            est[j] = np.sum(pop.weight * p) / denominator
            se[j] = np.sum(pop.weight * p * (1.0 - p)) / denominator * se_gamma[j]
        ref = np.sum(pop.weight * sigmoid(g0 + pop.xbeta)) / denominator
        return MeasureFrame.from_arrays(est, se, model.groups_, measure=measure, reference_value=float(ref))

    # indirect measures: provider j's own observations
    idx = np.asarray(model.group_indices_).ravel()
    xb = np.asarray(model.xbeta_, dtype=np.float64).ravel()
    n_obs = getattr(model, "N_", None)
    w = np.ones(xb.size) if n_obs is None else np.asarray(n_obs, dtype=np.float64).ravel()
    y = np.asarray(model.outcome_, dtype=np.float64).ravel()
    p0 = sigmoid(g0 + xb)
    expected = np.bincount(idx, weights=w * p0, minlength=gamma.size)
    observed = np.bincount(idx, weights=y, minlength=gamma.size)
    p_var = p0 if indirect_variance == "null" else sigmoid(gamma[idx] + xb)
    var_o = np.bincount(idx, weights=w * p_var * (1.0 - p_var), minlength=gamma.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(expected > 1e-10, observed / expected, np.nan)
        se = np.where(expected > 1e-10, np.sqrt(var_o) / expected, np.nan)
    if measure == "indirect_ratio":
        return MeasureFrame.from_arrays(ratio, se, model.groups_, measure=measure, reference_value=1.0)
    rate = _population_events() / pop.total_weight
    return MeasureFrame.from_arrays(ratio * rate, se * rate, model.groups_, measure=measure,
                                    reference_value=float(rate))


def at_bound(model, tol: float = 0.1) -> np.ndarray:
    """Providers whose fitted effect sits at the solver's bound (for example all-0 or all-1 outcomes).

    Useful as ``fit_mask=~at_bound(model)`` when fitting an empirical null.
    """
    gamma = np.asarray(model.coefficients_["gamma"], dtype=np.float64).ravel()
    algorithm = getattr(model, "algorithm", None)
    bound = getattr(algorithm, "bound", None) if algorithm is not None else None
    bound = 10.0 if bound is None else float(bound)
    return np.abs(gamma) >= bound - tol
