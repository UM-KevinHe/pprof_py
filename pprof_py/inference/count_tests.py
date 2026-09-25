"""Provider count tests: one driver for every exact or simulated count null.

A provider's observed event count is compared with its distribution under the
reference effect ``g0``. Each null specification returns that count's tails at
any provider effect ``g``, so one object gives the test (at ``g0``) and exact
confidence limits (by inverting the test in ``g``):

* :class:`PlugIn`: independent Bernoulli (or binomial) outcomes with row
  probabilities ``prob(g)``; exact Poisson-binomial tails.
* :class:`RowMixture`: each row draws its own random effect ``N(0, var_i)`` on
  top of ``eta(g)``; exact tails by Gauss-Hermite integration.
* :class:`ClusterMixture`: each cluster draws one random effect, shared by the
  provider's rows in that cluster; a Gauss-Hermite mixture per cluster,
  convolved across clusters.
* :class:`MonteCarlo`: simulated tails. ``exact`` names the exact null whose
  tails replace simulated tails at the Monte Carlo floor (``None``: no
  replacement).

The models build one null per provider from their own row quantities, with the
arithmetic of the fitted model, and :func:`count_test` does the rest. The
numerical kernels are in :mod:`pprof_py.inference.effect_tests`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .effect_tests import (EXACT_P_FLOOR, clustered_poibin_tails, integrated_poibin_tails, invert_decreasing,
                           normalize_alternative, poibin_tails, z_from_tails)

Tails = Tuple[float, float, float, float]


@dataclass(frozen=True)
class PlugIn:
    """Independent outcomes with probabilities ``prob(g)`` (and binomial ``trials``)."""

    prob: Callable[[float], np.ndarray]
    trials: Optional[np.ndarray] = None

    def tails(self, obs: float, g: float) -> Tails:
        return poibin_tails(obs, self.prob(g), self.trials)


@dataclass(frozen=True)
class RowMixture:
    """Each row draws its own random effect ``N(0, var_i)`` on top of ``eta(g)``."""

    eta: Callable[[float], np.ndarray]
    var: np.ndarray
    n_nodes: int

    def tails(self, obs: float, g: float) -> Tails:
        return integrated_poibin_tails(obs, self.eta(g), self.var, self.n_nodes)


@dataclass(frozen=True)
class ClusterMixture:
    """Each cluster draws one effect ``N(mean_c, var_c)``, shared by its rows; ``eta(g)`` excludes it."""

    eta: Callable[[float], np.ndarray]
    cluster: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    n_nodes: int

    def tails(self, obs: float, g: float) -> Tails:
        return clustered_poibin_tails(obs, self.eta(g), self.cluster, self.mean, self.var, self.n_nodes)


@dataclass(frozen=True)
class MonteCarlo:
    """Simulated tails ``simulate(obs, g)`` from ``n_resample`` draws; ``exact`` is used at the floor."""

    simulate: Callable[[float, float], Tails]
    n_resample: int
    exact: Optional[Union[PlugIn, RowMixture, ClusterMixture]] = None

    def tails(self, obs: float, g: float) -> Tails:
        return self.simulate(obs, g)


CountNull = Union[PlugIn, RowMixture, ClusterMixture, MonteCarlo]


def rows_by_provider(idx: np.ndarray, n_providers: int) -> List[np.ndarray]:
    """Row indices of each provider, in data order within the provider."""
    idx = np.asarray(idx).ravel()
    order = np.argsort(idx, kind="stable")
    edges = np.r_[0, np.cumsum(np.bincount(idx, minlength=n_providers))]
    return [order[edges[j]:edges[j + 1]] for j in range(n_providers)]


def count_test(obs: np.ndarray, nulls: Sequence[CountNull], g0: float, *, alternative: str,
               start: Optional[np.ndarray] = None, wanted: Optional[np.ndarray] = None,
               floor_message: Optional[str] = None
               ) -> Tuple[np.ndarray, Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]]:
    """z-statistics of each provider's count under its null, and exact limits when available.

    Parameters
    ----------
    obs : array
        Observed count of each provider.
    nulls : sequence
        One null per provider: all exact (:class:`PlugIn`, :class:`RowMixture`,
        :class:`ClusterMixture`) or all :class:`MonteCarlo`.
    g0 : float
        Reference effect at which the test is computed.
    alternative : {"two_sided", "greater", "less"}
        Two-sided tests use mid-p tails; one-sided tests use the full tail.
    start : array, optional
        Estimated provider effects, where the inversion starts. Required for
        exact nulls.
    wanted : bool array, optional
        Providers whose limits are computed (default: all).
    floor_message : str, optional
        Warning issued when simulated tails reach the Monte Carlo floor and are
        replaced by exact tails. ``{n}``, ``{total}`` and ``{floor}`` are filled in.

    Returns
    -------
    z : array
        z-statistics whose normal tails reproduce the tail p-values.
    limits : callable or None
        For exact nulls, ``limits(z_at_lower, z_at_upper)`` solving
        ``z(g) = target`` for each wanted provider (the ``limits`` argument of
        :func:`~pprof_py.inference.effect_test`); ``None`` for Monte Carlo nulls.
    """
    alt = normalize_alternative(alternative)
    two = alt == "two_sided"
    n = len(nulls)

    def z_of(tails, floor):
        t = np.atleast_2d(tails)
        return z_from_tails(t[:, 0] if two else t[:, 2], t[:, 1] if two else t[:, 3], alt, floor)

    simulated = [isinstance(null, MonteCarlo) for null in nulls]
    if n and all(simulated):
        sizes = {null.n_resample for null in nulls}
        if len(sizes) != 1:
            raise ValueError("All Monte Carlo nulls must use the same n_resample.")
        floor = 0.5 / sizes.pop()
        tails = np.empty((n, 4))
        for j, null in enumerate(nulls):
            tails[j] = null.tails(obs[j], g0)
        z = z_of(tails, floor)
        if any(null.exact is not None for null in nulls):
            up, lo = (tails[:, 0], tails[:, 1]) if two else (tails[:, 2], tails[:, 3])
            at_floor = (np.minimum(up, lo) if two else (up if alt == "greater" else lo)) <= floor
            at_floor &= np.array([null.exact is not None for null in nulls])
            for j in np.flatnonzero(at_floor):
                z[j] = z_of(nulls[j].exact.tails(obs[j], g0), EXACT_P_FLOOR)[0]
            if at_floor.any() and floor_message is not None:
                warnings.warn(floor_message.format(n=int(at_floor.sum()), total=n, floor=floor), stacklevel=3)
        return z, None
    if any(simulated):
        raise ValueError("Nulls must be all exact or all Monte Carlo.")
    if start is None:
        raise ValueError("Exact nulls need the estimated effects (start) for the limits.")
    z = z_of(np.array([null.tails(obs[j], g0) for j, null in enumerate(nulls)]), EXACT_P_FLOOR)
    start = np.asarray(start, dtype=np.float64).ravel()
    wanted = np.ones(n, dtype=bool) if wanted is None else np.asarray(wanted, dtype=bool)

    def limits(z_lower, z_upper):
        lower = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        for j in np.flatnonzero(wanted):
            def zfun(g, j=j):
                return float(z_of(nulls[j].tails(obs[j], g), EXACT_P_FLOOR)[0])
            lower[j] = invert_decreasing(zfun, start[j], z_lower[j])
            upper[j] = invert_decreasing(zfun, start[j], z_upper[j])
        return lower, upper

    return z, limits
