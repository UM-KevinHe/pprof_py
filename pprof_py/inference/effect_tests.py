"""Provider-effect tests: shared statistics and the test() contract.

Every model family's ``test()`` builds a z-type statistic for each provider's
effect and hands it to :func:`effect_test`, which applies the null model and
returns the fixed schema of :func:`~pprof_py.inference.provider_test`. The
statistics shared by several families live here too:

* :func:`poibin_tails`: exact Poisson-binomial tail probabilities of an observed
  count given per-trial null probabilities (mid-p for two-sided tests,
  ``P(X >= obs)`` / ``P(X <= obs)`` for one-sided tests, as in R pprof);
* :func:`resample_tails`: the same tails by simulation, drawing a random effect
  for each resample from its posterior (He et al. 2013);
* :func:`z_from_tails`: the z-statistic whose normal tail reproduces those
  p-values, computed in the smaller tail so extreme providers keep finite,
  accurate statistics.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from fast_poibin import PoiBin
from scipy.special import expit
from scipy.stats import norm

from .decision import _alternative as normalize_alternative
from .decision import provider_test, resolve_null_model
from .zstat import IDENTITY, ZFrame

__all__ = ["effect_test", "poibin_tails", "resample_tails", "bootstrap_tails", "z_from_tails", "reference_effect",
           "normalize_alternative", "EXACT_P_FLOOR"]

EXACT_P_FLOOR = 1e-300
_MAX_EXACT_TRIALS = 20000


def reference_effect(effects: np.ndarray, sizes: Optional[np.ndarray], reference) -> float:
    """The reference effect gamma_0: ``"median"``, ``"mean"`` (size-weighted), or a number."""
    effects = np.asarray(effects, dtype=np.float64)
    if isinstance(reference, str):
        if reference == "median":
            return float(np.median(effects))
        if reference == "mean":
            return float(np.average(effects, weights=None if sizes is None else np.asarray(sizes, dtype=float)))
        raise ValueError("reference must be 'median', 'mean', or a number.")
    return float(reference)


def poibin_tails(obs: float, probs: np.ndarray, trials: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
    """Exact tails of ``X = sum Bernoulli(p_i)`` at ``obs``: ``(upper_mid, lower_mid, p_ge, p_le)``.

    ``trials`` expands each probability into that many Bernoulli trials (binomial data).
    """
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-10, 1 - 1e-10)
    if trials is not None and np.any(trials != 1):
        n = np.asarray(trials)
        if np.any(n != np.round(n)) or np.any(n < 0):
            raise ValueError("The exact test needs integer binomial trials.")
        if n.sum() > _MAX_EXACT_TRIALS:
            raise ValueError(f"The exact test would need {int(n.sum())} trials for one provider; "
                             "use test_method='score' or 'bootstrap_exact' for large binomial counts.")
        probs = np.repeat(probs, n.astype(int))
    o = int(round(float(obs)))
    pb = PoiBin(probs)
    cdf, pmf = pb.cdf[o], pb.pmf[o]
    cdf_below = pb.cdf[o - 1] if o > 0 else 0.0
    return 1.0 - cdf + 0.5 * pmf, cdf - 0.5 * pmf, (1.0 - cdf_below) if o > 0 else 1.0, cdf


def resample_tails(obs: float, eta_fixed: np.ndarray, re_mean: np.ndarray, re_var: np.ndarray,
                   null_effect: float, n_resample: int, rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """Simulated tails ``(upper_mid, lower_mid, p_ge, p_le)`` of a provider's event count under the null.

    Each resample draws the other random effects from their posterior
    ``N(re_mean, re_var)``, then outcomes ``Bernoulli(expit(null_effect + re + eta_fixed))``.
    """
    eta_fixed = np.asarray(eta_fixed, dtype=np.float64)
    re = rng.normal(np.tile(re_mean, (n_resample, 1)), np.sqrt(np.maximum(np.tile(re_var, (n_resample, 1)), 0.0)))
    sums = rng.binomial(1, expit(null_effect + re + eta_fixed)).sum(axis=1)
    ge, gt = np.mean(sums >= obs), np.mean(sums > obs)
    le, lt = np.mean(sums <= obs), np.mean(sums < obs)
    return (ge + gt) / 2.0, (le + lt) / 2.0, ge, le


def bootstrap_tails(obs: float, probs: np.ndarray, trials: Optional[np.ndarray], n_resample: int,
                    rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """Monte Carlo version of :func:`poibin_tails` (``n_resample`` draws of the provider's count)."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-10, 1 - 1e-10)
    if trials is None or np.all(trials == 1):
        draws = (rng.random((n_resample, probs.size)) < probs).sum(axis=1)
    else:
        draws = rng.binomial(np.asarray(trials).astype(int), probs, size=(n_resample, probs.size)).sum(axis=1)
    ge, gt = np.mean(draws >= obs), np.mean(draws > obs)
    le, lt = np.mean(draws <= obs), np.mean(draws < obs)
    return (ge + gt) / 2.0, (le + lt) / 2.0, ge, le


def z_from_tails(upper: np.ndarray, lower: np.ndarray, alternative: str, floor: float) -> np.ndarray:
    """z-statistics whose normal tails reproduce tail p-values.

    Two-sided: ``upper`` and ``lower`` are mid-p tails summing to 1 and
    ``2 * sf(|z|) = 2 * min(upper, lower)``. Greater: ``sf(z) = upper``.
    Less: ``cdf(z) = lower``. Tails below ``floor`` are raised to it, and
    ``|z|`` is capped at ``Phi^-1(1 - floor)`` so every provider keeps a
    finite statistic.
    """
    upper = np.clip(np.asarray(upper, dtype=np.float64), floor, 1.0)
    lower = np.clip(np.asarray(lower, dtype=np.float64), floor, 1.0)
    if alternative == "two_sided":
        z = np.where(upper <= lower, norm.isf(upper), -norm.isf(lower))
    elif alternative == "greater":
        z = norm.isf(upper)
    else:
        z = -norm.isf(lower)
    z_max = norm.isf(floor)                       # a one-sided p of exactly 1 stays finite (p = 1 to machine precision)
    return np.clip(z, -z_max, z_max)


def effect_test(provider, estimate, z, reference: float, *, se=None, df=None, null_model=None,
                alternative: str = "two_sided", level: float = 0.95, critical: Optional[float] = None,
                interval: str = "inversion", providers=None, test_method: Optional[str] = None) -> pd.DataFrame:
    """Assemble a provider-effect test in the shared schema.

    ``se`` is given only for Wald-type statistics, which then also get
    intervals; ``df`` gives them a Student-t reference.
    """
    est = np.asarray(estimate, dtype=np.float64).ravel()
    zz = np.asarray(z, dtype=np.float64).ravel().copy()
    zz[~np.isfinite(zz)] = np.nan
    se_arr = None if se is None else np.asarray(se, dtype=np.float64).ravel()
    zf = ZFrame(z=zz, index=pd.Index(np.asarray(provider).ravel(), name="provider"), estimate=est,
                se=np.full(est.size, np.nan) if se_arr is None else se_arr,
                transformed=None if se_arr is None else est, se_transformed=se_arr,
                null_value=float(reference), null_transformed=float(reference), transform=IDENTITY,
                measure="gamma", df=df)
    null = resolve_null_model(null_model, zf)
    res = provider_test(zf, null, alternative=alternative, level=level, critical=critical, interval=interval)
    res.attrs.update({"test_method": test_method, "reference": float(reference)})
    if providers is not None:
        res = res.loc[res.index.isin(np.atleast_1d(providers))]
    return res
