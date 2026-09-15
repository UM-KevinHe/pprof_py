"""Risk-set construction: tracking who is "at risk" as time moves forward.

The risk set at time t is R(t) = { i : start_i < t <= stop_i } -- a
right-continuous, left-open convention. Concretely this means:
  * an observation censored or having an event exactly AT time t is
    still counted at risk at t (ties on `stop` are inclusive), and
  * an observation entering (via left truncation / staggered entry)
    exactly AT time t is NOT yet at risk at t (ties on `start` are
    exclusive).
This matches R's Surv(start, stop, event) convention (see
docs/R_COMPATIBILITY.md, question 1).

Two implementations of the same sweep exist here, and it matters that
they stay behaviorally identical:

  - `_compute_risk_set_summaries_numba` / `compute_risk_set_summaries`:
    the numba-compiled path (used whenever numba is importable), written
    as explicit per-observation loops because that is what numba
    compiles best -- there is no Python-level per-call overhead to
    amortize away by batching, so there is nothing to gain from it here.
  - `sweep_risk_sets`: the pure-Python fallback (used when numba isn't
    installed, and directly by any tie method that hasn't been ported to
    a numba kernel yet -- see algorithms/ties.py). Here, per-call NumPy
    overhead (one `np.outer` per observation) previously dominated
    runtime at production scale (profiling a 200k-row, 3000-stratum fit
    found 2.6M+ individual `np.outer` calls), so entries and exits
    between consecutive event times are batched into one `searchsorted`
    + one vectorized reduction each, rather than processed one
    observation at a time.

Both compute the identical S0/S1/S2 definitions:

    S0 = sum_{i in R(t)} weight_i * r_i                   (scalar)
    S1 = sum_{i in R(t)} weight_i * r_i * X_i              (p,)
    S2 = sum_{i in R(t)} weight_i * r_i * outer(X_i, X_i)  (p, p)
"""
from __future__ import annotations

from typing import Callable

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


@njit(cache=True)
def _compute_risk_set_summaries_numba(
    start: np.ndarray,
    stop: np.ndarray,
    weight: np.ndarray,
    r: np.ndarray,
    X: np.ndarray,
    event_times: np.ndarray,
):
    n, p = X.shape
    wr = weight * r
    order_start = np.argsort(start)
    order_stop = np.argsort(stop)

    running_S0 = 0.0
    running_S1 = np.zeros(p)
    running_S2 = np.zeros((p, p))

    K = len(event_times)
    S0 = np.empty(K)
    S1 = np.empty((K, p))
    S2 = np.empty((K, p, p))

    ptr_add = 0
    ptr_remove = 0

    for j in range(K):
        t = event_times[j]

        while ptr_add < n and start[order_start[ptr_add]] < t:
            i = order_start[ptr_add]
            wri = wr[i]
            running_S0 += wri
            for a in range(p):
                wa = wri * X[i, a]
                running_S1[a] += wa
                for b in range(p):
                    running_S2[a, b] += wa * X[i, b]
            ptr_add += 1

        while ptr_remove < n and stop[order_stop[ptr_remove]] < t:
            i = order_stop[ptr_remove]
            wri = wr[i]
            running_S0 -= wri
            for a in range(p):
                wa = wri * X[i, a]
                running_S1[a] -= wa
                for b in range(p):
                    running_S2[a, b] -= wa * X[i, b]
            ptr_remove += 1

        S0[j] = running_S0
        for a in range(p):
            S1[j, a] = running_S1[a]
            for b in range(p):
                S2[j, a, b] = running_S2[a, b]

    return S0, S1, S2


def compute_risk_set_summaries(
    start: np.ndarray,
    stop: np.ndarray,
    weight: np.ndarray,
    r: np.ndarray,
    X: np.ndarray,
    event_times: np.ndarray,
):
    """Return arrays S0, S1, S2 for all event times in one stratum.

    When numba is available, the running-risk-set sweep is compiled. A
    pure-Python fallback (the batched `sweep_risk_sets` below) preserves
    identical behavior for environments where numba is not installed.
    """
    if _HAS_NUMBA:
        return _compute_risk_set_summaries_numba(start, stop, weight, r, X, event_times)

    n, p = X.shape
    K = len(event_times)
    S0 = np.empty(K)
    S1 = np.empty((K, p))
    S2 = np.empty((K, p, p))

    def on_step(j, t, s0, s1, s2):
        S0[j] = s0
        S1[j, :] = s1
        S2[j, :, :] = s2

    sweep_risk_sets(start, stop, weight, r, X, event_times, on_step)
    return S0, S1, S2


@njit(cache=True)
def _compute_S0_only_numba(
    start: np.ndarray,
    stop: np.ndarray,
    weight: np.ndarray,
    r: np.ndarray,
    event_times: np.ndarray,
):
    n = len(start)
    wr = weight * r
    order_start = np.argsort(start)
    order_stop = np.argsort(stop)

    running_S0 = 0.0
    K = len(event_times)
    S0 = np.empty(K)

    ptr_add = 0
    ptr_remove = 0

    for j in range(K):
        t = event_times[j]

        while ptr_add < n and start[order_start[ptr_add]] < t:
            running_S0 += wr[order_start[ptr_add]]
            ptr_add += 1

        while ptr_remove < n and stop[order_stop[ptr_remove]] < t:
            running_S0 -= wr[order_stop[ptr_remove]]
            ptr_remove += 1

        S0[j] = running_S0

    return S0


def compute_S0_only(
    start: np.ndarray,
    stop: np.ndarray,
    weight: np.ndarray,
    r: np.ndarray,
    event_times: np.ndarray,
) -> np.ndarray:
    """Like `compute_risk_set_summaries`, but for callers (baseline
    hazard, for either tie method) that only need the scalar S0 -- not
    S1 or S2, which cost O(p) and O(p^2) per observation respectively to
    accumulate and would otherwise be computed and immediately discarded.
    At p=57 that's not a rounding error: it's most of the work.
    """
    if _HAS_NUMBA:
        return _compute_S0_only_numba(start, stop, weight, r, event_times)

    K = len(event_times)
    S0 = np.empty(K)

    def on_step(j, t, s0, s1, s2):
        S0[j] = s0

    dummy_X = np.zeros((len(start), 1))
    sweep_risk_sets(start, stop, weight, r, dummy_X, event_times, on_step)
    return S0


def sweep_risk_sets(
    start: np.ndarray,
    stop: np.ndarray,
    weight: np.ndarray,
    r: np.ndarray,
    X: np.ndarray,
    event_times: np.ndarray,
    callback: Callable[[int, float, float, np.ndarray, np.ndarray], None],
) -> None:
    """Sweep `event_times` (must be sorted ascending, one stratum's worth
    of data) and invoke ``callback(j, t, S0, S1, S2)`` for each.

    `S2` is passed as a *live reference* to the sweep's internal running
    matrix, not a copy -- callbacks must use it immediately (e.g. to
    accumulate into their own array) rather than store the reference,
    since it is mutated in place on the next iteration.

    Entries and exits between two consecutive event times are applied as
    a single batched update (one `searchsorted` to find the batch
    boundary, one vectorized reduction over it) rather than one
    observation at a time -- see the module docstring for why that
    matters at scale. The risk-set MATH is unchanged (a sum of individual
    outer products over a batch equals that batch's matrix product).
    """
    n, p = X.shape
    wr = weight * r

    order_start = np.argsort(start, kind="mergesort")
    order_stop = np.argsort(stop, kind="mergesort")
    start_sorted = start[order_start]
    stop_sorted = stop[order_stop]

    running_S0 = 0.0
    running_S1 = np.zeros(p)
    running_S2 = np.zeros((p, p))

    ptr_add = 0
    ptr_remove = 0

    for j in range(len(event_times)):
        t = event_times[j]

        # Entries: as t passes start_i (t now > start_i), i becomes at risk.
        end_add = np.searchsorted(start_sorted, t, side="left")
        if end_add > ptr_add:
            batch = order_start[ptr_add:end_add]
            Xb = X[batch]
            wrb = wr[batch]
            running_S0 += wrb.sum()
            running_S1 += wrb @ Xb
            running_S2 += Xb.T @ (wrb[:, None] * Xb)
            ptr_add = end_add

        # Exits: once t passes stop_i (t now > stop_i), i leaves the risk set.
        # Note the strict "<", not "<=" -- an exit exactly at t is still
        # included at t (see module docstring).
        end_remove = np.searchsorted(stop_sorted, t, side="left")
        if end_remove > ptr_remove:
            batch = order_stop[ptr_remove:end_remove]
            Xb = X[batch]
            wrb = wr[batch]
            running_S0 -= wrb.sum()
            running_S1 -= wrb @ Xb
            running_S2 -= Xb.T @ (wrb[:, None] * Xb)
            ptr_remove = end_remove

        callback(j, t, running_S0, running_S1, running_S2)
