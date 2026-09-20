"""Tie-handling strategies for the Cox partial likelihood.

R's coxph supports three conventions for scoring simultaneous ("tied")
event times: Breslow (the default in this package), Efron (R's own
default, generally preferred when ties are common), and exact
(a discrete/conditional-logistic likelihood, exact but combinatorially
expensive for large tie sets).

This module isolates *only* the tie-breaking formula. Everything about
*who is in the risk set at time t* -- left truncation, strata, the
half-open (start, stop] convention -- lives in risk_sets.py and is
identical across tie methods; only how the denominator at a tied event
time is built from that risk set differs. Because of that split, adding
Efron or exact tie support later is a matter of adding a class here,
not touching cox_likelihood.py, optimization.py, or CoxPH itself.

Breslow is this package's default (matching the existing `phregSHR` R
workflow this package targets) and Efron is implemented as of this
version too, matched directly against R's C source rather than a
textbook formula (see EfronTies' docstring). Exact is stubbed with
enough of the interface and docstring context that implementing it is
additive rather than a redesign.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .risk_sets import sweep_risk_sets, compute_risk_set_summaries, compute_S0_only, njit, _HAS_NUMBA
from ...utils.numerical import safe_exp


@dataclass
class StratumContribution:
    """One stratum's additive contribution to the total log-likelihood,
    score, and information matrix. Coefficients are shared across strata,
    so the caller (cox_likelihood.cox_partial_likelihood) simply sums
    these across strata."""

    log_likelihood: float
    score: np.ndarray
    information: np.ndarray


class TieMethod:
    """Abstract base for a tie-breaking convention."""

    name: str = "abstract"

    def stratum_contribution(
        self,
        X: np.ndarray,
        start: np.ndarray,
        stop: np.ndarray,
        event: np.ndarray,
        weight: np.ndarray,
        eta: np.ndarray,
    ) -> StratumContribution:
        raise NotImplementedError

    def baseline_hazard_increments(
        self,
        X: np.ndarray,
        start: np.ndarray,
        stop: np.ndarray,
        event: np.ndarray,
        weight: np.ndarray,
        eta: np.ndarray,
    ):
        """Return (event_times, dH0) for one stratum: the distinct event
        times and the Breslow-estimator cumulative-hazard JUMP at each.

        This is a second, separate place a tie method has to define a
        formula (`stratum_contribution` governs the likelihood; this
        governs the hazard estimate) -- the two are related but not
        interchangeable. Getting this wrong silently produces a
        plausible-looking but numerically incorrect `baseline_hazard_`
        even when `coef_`/`standard_errors_`/`log_likelihood_` are all
        exactly right, since those don't depend on this method at all.
        See docs/R_COMPATIBILITY.md, question 2b, for exactly this trap
        (encountered, and fixed, during this package's own development).
        """
        raise NotImplementedError


@njit(cache=True)
def _breslow_accumulate_numba(start, stop, weight, r, X, event_times, d_star):
    """Fused risk-set sweep + Breslow log-likelihood/score/information
    accumulation for one stratum -- mirrors `_efron_accumulate_numba`'s
    structure so the two tie methods get the same treatment: the whole
    per-event-time Hessian assembly happens inside compiled code, not in
    a Python loop over K after the sweep returns arrays. (An earlier
    version of this file's numba work compiled only the sweep itself and
    left that K-loop in Python -- harmless for correctness, since it's
    still exercised by every test here, but it meant Breslow was
    doing less of its work in compiled code than Efron ended up doing,
    for no reason other than which one got fused first.)

    error_code: 0 ok, 1 empty risk set at an event time with a death --
    see `_efron_accumulate_numba`'s docstring for why this is a code
    rather than a raised exception.
    """
    n, p = X.shape
    wr = weight * r
    order_start = np.argsort(start)
    order_stop = np.argsort(stop)

    running_S0 = 0.0
    running_S1 = np.zeros(p)
    running_S2 = np.zeros((p, p))

    K = len(event_times)
    loglik_term = 0.0
    score_term = np.zeros(p)
    hessian = np.zeros((p, p))
    error_code = 0

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

        d_star_j = d_star[j]
        if d_star_j == 0.0:
            continue
        if running_S0 <= 0:
            error_code = 1
            return loglik_term, score_term, hessian, error_code

        xbar = np.empty(p)
        for a in range(p):
            xbar[a] = running_S1[a] / running_S0
        loglik_term -= d_star_j * np.log(running_S0)
        for a in range(p):
            score_term[a] -= d_star_j * xbar[a]
            for b in range(p):
                hessian[a, b] += d_star_j * (running_S2[a, b] / running_S0 - xbar[a] * xbar[b])

    return loglik_term, score_term, hessian, error_code


class BreslowTies(TieMethod):
    """Breslow's approximation: all d_j tied events at time t_j share the
    same risk-set denominator S0(t_j), as if they were d_j independent
    draws (with replacement) from the full risk set.

    Weighted log partial likelihood (see docs/R_COMPATIBILITY.md,
    question 5, for the weight semantics this matches):

        LL = sum_{i: event} w_i * eta_i  -  sum_j d*_j * log( S0(t_j) )

    where d*_j = sum of weights of the observations with an event at
    t_j, and S0(t_j) = sum_{i in R(t_j)} w_i * exp(eta_i).

    Score and (observed) information follow by differentiating this
    expression w.r.t. beta once and twice respectively; see
    cox_likelihood.py's module docstring for the closed forms in
    terms of S0, S1 = sum w*r*X, and S2 = sum w*r*outer(X,X).
    """

    name = "breslow"

    def stratum_contribution(self, X, start, stop, event, weight, eta) -> StratumContribution:
        """Dispatches to the numba-compiled kernel when available,
        falling back to `_stratum_contribution_python` otherwise -- see
        EfronTies.stratum_contribution's docstring for the same pattern
        and why both paths are checked against each other, not just
        assumed consistent (test_engine_self_consistency.py does this
        for both tie methods).
        """
        if not _HAS_NUMBA:
            return self._stratum_contribution_python(X, start, stop, event, weight, eta)

        n, p = X.shape
        event = event.astype(bool)
        r = safe_exp(eta)

        event_stop = stop[event]
        if event_stop.size == 0:
            return StratumContribution(0.0, np.zeros(p), np.zeros((p, p)))

        event_times, inverse = np.unique(event_stop, return_inverse=True)
        K = event_times.shape[0]
        event_weight = weight[event]
        d_star = np.bincount(inverse, weights=event_weight, minlength=K)

        total_event_eta = float(np.sum(event_weight * eta[event]))
        total_event_X = np.sum(event_weight[:, None] * X[event], axis=0)

        loglik_term, score_term, hessian, error_code = _breslow_accumulate_numba(
            start, stop, weight, r, X, event_times, d_star
        )
        if error_code == 1:
            raise FloatingPointError(
                "Empty risk set at an event time with positive event weight. "
                "This means an observation's own (start, stop] interval does "
                "not contain its recorded event time -- a data problem, not a "
                "numerical one; re-check the event/stop time construction."
            )

        log_likelihood = total_event_eta + loglik_term
        score = total_event_X + score_term
        return StratumContribution(log_likelihood, score, hessian)

    def _stratum_contribution_python(self, X, start, stop, event, weight, eta) -> StratumContribution:
        n, p = X.shape
        event = event.astype(bool)
        r = safe_exp(eta)

        event_stop = stop[event]
        if event_stop.size == 0:
            return StratumContribution(0.0, np.zeros(p), np.zeros((p, p)))

        event_times, inverse = np.unique(event_stop, return_inverse=True)
        K = event_times.shape[0]
        event_weight = weight[event]
        d_star = np.bincount(inverse, weights=event_weight, minlength=K)

        total_event_eta = float(np.sum(event_weight * eta[event]))
        total_event_X = np.sum(event_weight[:, None] * X[event], axis=0)

        S0, S1, S2 = compute_risk_set_summaries(start, stop, weight, r, X, event_times)
        hessian = np.zeros((p, p))

        positive = S0 > 0
        bad = (~positive) & (d_star > 0)
        if np.any(bad):
            raise FloatingPointError(
                "Empty risk set at an event time with positive event weight. "
                "This means an observation's own (start, stop] interval does "
                "not contain its recorded event time -- a data problem, not a "
                "numerical one; re-check the event/stop time construction."
            )

        for j in range(K):
            s0 = S0[j]
            if s0 > 0:
                xbar = S1[j] / s0
                hessian[:, :] += d_star[j] * (S2[j] / s0 - np.outer(xbar, xbar))

        log_likelihood = total_event_eta - float(np.sum(d_star[positive] * np.log(S0[positive])))
        xbar_all = np.divide(S1, S0[:, None], out=np.zeros_like(S1), where=S0[:, None] > 0)
        score = total_event_X - np.sum(d_star[:, None] * xbar_all, axis=0)

        return StratumContribution(log_likelihood, score, hessian)

    def baseline_hazard_increments(self, X, start, stop, event, weight, eta):
        n, p = X.shape
        event = event.astype(bool)
        r = safe_exp(eta)
        event_stop = stop[event]
        if event_stop.size == 0:
            return np.array([]), np.array([])

        event_times, inverse = np.unique(event_stop, return_inverse=True)
        K = event_times.shape[0]
        d_star = np.bincount(inverse, weights=weight[event], minlength=K)
        S0 = compute_S0_only(start, stop, weight, r, event_times)

        bad = (S0 <= 0) & (d_star > 0)
        if np.any(bad):
            raise FloatingPointError(
                "Empty risk set at an event time with positive event weight "
                "while computing the baseline hazard."
            )

        dH0 = np.divide(d_star, S0, out=np.zeros_like(d_star), where=S0 > 0)
        return event_times, dH0


@njit(cache=True)
def _compute_death_sums_numba(X, weight, r, event_idx, death_time_idx, K):
    """S0_D/S1_D/S2_D: the same S0/S1/S2 definitions as
    `compute_risk_set_summaries`, but summed over only the tied-death
    rows at each event time rather than the full risk set -- needed for
    Efron's correction. `event_idx` is the row indices (within this
    stratum) that have an event; `death_time_idx[m]` is which of the K
    event times `event_idx[m]`'s own death belongs to. A plain loop over
    event rows only (not the full risk set), since D_j is always small
    even when the risk set is large.
    """
    p = X.shape[1]
    n_events = len(event_idx)
    S0_D = np.zeros(K)
    S1_D = np.zeros((K, p))
    S2_D = np.zeros((K, p, p))
    for m in range(n_events):
        i = event_idx[m]
        j = death_time_idx[m]
        wri = weight[i] * r[i]
        S0_D[j] += wri
        for a in range(p):
            wa = wri * X[i, a]
            S1_D[j, a] += wa
            for b in range(p):
                S2_D[j, a, b] += wa * X[i, b]
    return S0_D, S1_D, S2_D


@njit(cache=True)
def _efron_accumulate_numba(start, stop, weight, r, X, event_times, d_star, d_raw, S0_D, S1_D, S2_D):
    """Fused risk-set sweep + Efron log-likelihood/score/information
    accumulation for one stratum -- the compiled equivalent of
    `EfronTies.stratum_contribution`'s callback-based Python version
    (kept below as `_efron_stratum_contribution_python`, used when numba
    isn't available and as the reference this was checked against).

    Returns (loglik_correction, score_correction, hessian, error_code):
    error_code is 0 for success, 1 for an empty risk set at an event
    time with positive event weight, 2 for a non-positive interpolated
    Efron denominator -- numba's nopython mode does not support raising
    an exception with a dynamically-built message, so the Python wrapper
    (`EfronTies.stratum_contribution`) checks this code and raises the
    same FloatingPointError the pure-Python path would, with the same
    message.
    """
    n, p = X.shape
    wr = weight * r
    order_start = np.argsort(start)
    order_stop = np.argsort(stop)

    running_S0 = 0.0
    running_S1 = np.zeros(p)
    running_S2 = np.zeros((p, p))

    K = len(event_times)
    loglik_correction = 0.0
    score_correction = np.zeros(p)
    hessian = np.zeros((p, p))
    error_code = 0

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

        dj = d_raw[j]
        if dj == 0:
            continue
        d_star_j = d_star[j]

        if dj == 1:
            if running_S0 <= 0:
                error_code = 1
                return loglik_correction, score_correction, hessian, error_code
            xbar = np.empty(p)
            for a in range(p):
                xbar[a] = running_S1[a] / running_S0
            loglik_correction -= d_star_j * np.log(running_S0)
            for a in range(p):
                score_correction[a] -= d_star_j * xbar[a]
                for b in range(p):
                    hessian[a, b] += d_star_j * (running_S2[a, b] / running_S0 - xbar[a] * xbar[b])
            continue

        meanwt_j = d_star_j / dj
        for k in range(1, dj + 1):
            frac = k / dj
            s0_k = (running_S0 - S0_D[j]) + frac * S0_D[j]
            if s0_k <= 0:
                error_code = 2
                return loglik_correction, score_correction, hessian, error_code
            mean_k = np.empty(p)
            for a in range(p):
                s1_k_a = (running_S1[a] - S1_D[j, a]) + frac * S1_D[j, a]
                mean_k[a] = s1_k_a / s0_k
            loglik_correction -= meanwt_j * np.log(s0_k)
            for a in range(p):
                score_correction[a] -= meanwt_j * mean_k[a]
                for b in range(p):
                    s2_k_ab = (running_S2[a, b] - S2_D[j, a, b]) + frac * S2_D[j, a, b]
                    hessian[a, b] += meanwt_j * (s2_k_ab / s0_k - mean_k[a] * mean_k[b])

    return loglik_correction, score_correction, hessian, error_code


@njit(cache=True)
def _efron_baseline_numba(start, stop, weight, r, event_times, d_star, d_raw, S0_D):
    """Fused sweep + Efron baseline-hazard-increment accumulation, S0
    only (no S1/S2 -- baseline hazard doesn't need them, see
    risk_sets.py::compute_S0_only). error_code: 0 ok, 1 empty risk set,
    2 non-positive interpolated denominator -- see
    `_efron_accumulate_numba` for why this is a code, not an exception.
    """
    n = len(start)
    wr = weight * r
    order_start = np.argsort(start)
    order_stop = np.argsort(stop)

    running_S0 = 0.0
    K = len(event_times)
    dH0 = np.zeros(K)
    error_code = 0

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

        dj = d_raw[j]
        if dj == 0:
            dH0[j] = 0.0
            continue
        if running_S0 <= 0:
            error_code = 1
            return dH0, error_code
        if dj == 1:
            dH0[j] = d_star[j] / running_S0
            continue

        meanwt_j = d_star[j] / dj
        total = 0.0
        for k in range(dj):
            denom = running_S0 - (k / dj) * S0_D[j]
            if denom <= 0:
                error_code = 2
                return dH0, error_code
            total += 1.0 / denom
        dH0[j] = meanwt_j * total

    return dH0, error_code


class EfronTies(TieMethod):
    """Efron's approximation: unlike Breslow, which treats all `d_j` tied
    events at time `t_j` as sharing the identical risk-set denominator
    `S0(t_j)`, Efron's method progressively "removes" the tied deaths from
    the denominator one at a time, averaging over the `d_j` possible
    removal positions. This is what `coxph(..., ties="efron")` computes
    in R -- and is R's own default, unlike this package's (Breslow-first,
    matching the existing `phregSHR` R workflow).

    Formula matched exactly to R's actual C implementation for the
    standard (non-penalized) fit path, `src/agfit4.c` in survival 3.5-8
    (read directly via `apt-get source r-cran-survival`, not
    reconstructed from the textbook formula alone -- see
    docs/R_COMPATIBILITY.md, question 2, for why that mattered:
    survival's *other* Efron implementation, in the penalized/frailty
    path `agfit5.c`, uses a visibly different -- and, for an ordinary
    weighted fit, not applicable -- weighting construction, so reading
    the wrong C file would have reproduced the wrong formula convincingly).

    For a tied event time with `d_j` deaths (D_j, a RAW COUNT -- not
    weighted) and weighted death mass `d*_j = sum_{i in D_j} w_i`:

        meanwt_j = d*_j / d_j
        S0_{R\\D} = S0(risk set) - S0(D_j)   (and likewise S1, S2)

        LL_j = sum_{i in D_j} w_i*eta_i
               - meanwt_j * sum_{k=1}^{d_j} log(S0_{R\\D} + (k/d_j)*S0(D_j))

    with the score and information the first and second derivatives of
    that expression, in the same S0/S1/S2 terms as BreslowTies. When
    `d_j == 1` (no actual tie), `meanwt_j == d*_j` and the single k=1 term
    reduces exactly to BreslowTies' formula -- R's own C code takes
    exactly this shortcut (`if (method==0 || deaths==1)`), and this
    implementation was checked to reproduce it: EfronTies and
    BreslowTies agree to machine precision on data with no ties at all.
    """

    name = "efron"

    def stratum_contribution(self, X, start, stop, event, weight, eta) -> StratumContribution:
        """Dispatches to the numba-compiled kernel when available,
        falling back to `_stratum_contribution_python` (the original,
        callback-based implementation this was validated against)
        otherwise. Both must produce identical results by construction --
        `tests/test_engine_self_consistency.py` checks the numba path
        against the brute-force reference the same way the Python path
        already was, and `tests/test_r_comparison.py`'s efron tests run
        with numba installed, so a divergence between the two paths
        would show up as a normal test failure, not silently.
        """
        if not _HAS_NUMBA:
            return self._stratum_contribution_python(X, start, stop, event, weight, eta)

        n, p = X.shape
        event = event.astype(bool)
        r = safe_exp(eta)

        event_stop = stop[event]
        if event_stop.size == 0:
            return StratumContribution(0.0, np.zeros(p), np.zeros((p, p)))

        event_times, inverse = np.unique(event_stop, return_inverse=True)
        K = event_times.shape[0]
        event_weight = weight[event]
        d_star = np.bincount(inverse, weights=event_weight, minlength=K)
        d_raw = np.bincount(inverse, minlength=K).astype(np.int64)

        total_event_eta = float(np.sum(event_weight * eta[event]))
        total_event_X = np.sum(event_weight[:, None] * X[event], axis=0)

        event_idx = np.where(event)[0].astype(np.int64)
        S0_D, S1_D, S2_D = _compute_death_sums_numba(X, weight, r, event_idx, inverse.astype(np.int64), K)

        loglik_correction, score_correction, hessian, error_code = _efron_accumulate_numba(
            start, stop, weight, r, X, event_times, d_star, d_raw, S0_D, S1_D, S2_D
        )
        if error_code == 1:
            raise FloatingPointError(
                "Empty risk set at an event time with positive event "
                "weight; indicates invalid (start, stop] construction."
            )
        if error_code == 2:
            raise FloatingPointError(
                "Non-positive interpolated Efron denominator at an event "
                "time -- indicates invalid (start, stop] construction "
                "rather than a numerical issue."
            )

        log_likelihood = total_event_eta + loglik_correction
        score = total_event_X + score_correction
        return StratumContribution(log_likelihood, score, hessian)

    def _stratum_contribution_python(self, X, start, stop, event, weight, eta) -> StratumContribution:
        n, p = X.shape
        event = event.astype(bool)
        r = safe_exp(eta)

        event_stop = stop[event]
        if event_stop.size == 0:
            return StratumContribution(0.0, np.zeros(p), np.zeros((p, p)))

        event_times, inverse = np.unique(event_stop, return_inverse=True)
        K = event_times.shape[0]
        event_weight = weight[event]
        d_star = np.bincount(inverse, weights=event_weight, minlength=K)  # weighted death mass per time
        d_raw = np.bincount(inverse, minlength=K)                          # RAW count of tied deaths per time

        total_event_eta = float(np.sum(event_weight * eta[event]))
        total_event_X = np.sum(event_weight[:, None] * X[event], axis=0)

        # D_j-restricted sums at each event time: cheap direct reductions,
        # computed once per event time rather than needing the risk-set
        # sweep (D_j is small -- typically far smaller than the full risk
        # set). S2_D is kept as a per-time list of (p, p) matrices rather
        # than a (K, p, p) tensor, since K (distinct event times) can be
        # large even though each individual D_j is small.
        event_idx = np.where(event)[0]
        wr_event = event_weight * r[event_idx]
        S0_D = np.zeros(K)
        S1_D = np.zeros((K, p))
        S2_D = [np.zeros((p, p)) for _ in range(K)]
        np.add.at(S0_D, inverse, wr_event)
        np.add.at(S1_D, inverse, wr_event[:, None] * X[event_idx])
        for local_i, j in enumerate(inverse):
            xi = X[event_idx[local_i]]
            S2_D[j] += wr_event[local_i] * np.outer(xi, xi)

        hessian = np.zeros((p, p))
        loglik_correction = 0.0  # accumulates -meanwt_j * sum_k log(...)
        score_correction = np.zeros(p)

        def on_step(j, t, S0_R, S1_R, S2_R):
            nonlocal loglik_correction, score_correction, hessian
            dj = d_raw[j]
            if dj == 0:
                return
            d_star_j = d_star[j]
            S0_D_j = S0_D[j]
            S1_D_j = S1_D[j]
            S2_D_j = S2_D[j]

            if dj == 1:
                # Exact Breslow shortcut -- also avoids any 0/dj division.
                if S0_R <= 0:
                    raise FloatingPointError(
                        "Empty risk set at an event time with positive event "
                        "weight; indicates invalid (start, stop] construction."
                    )
                xbar = S1_R / S0_R
                loglik_correction -= d_star_j * np.log(S0_R)
                score_correction -= d_star_j * xbar
                hessian[:, :] += d_star_j * (S2_R / S0_R - np.outer(xbar, xbar))
                return

            meanwt_j = d_star_j / dj
            S0_base = S0_R - S0_D_j
            S1_base = S1_R - S1_D_j
            S2_base = S2_R - S2_D_j
            for k in range(1, dj + 1):
                frac = k / dj
                s0_k = S0_base + frac * S0_D_j
                if s0_k <= 0:
                    raise FloatingPointError(
                        "Non-positive interpolated Efron denominator at an event "
                        "time -- indicates invalid (start, stop] construction "
                        "rather than a numerical issue."
                    )
                s1_k = S1_base + frac * S1_D_j
                s2_k = S2_base + frac * S2_D_j
                mean_k = s1_k / s0_k
                loglik_correction -= meanwt_j * np.log(s0_k)
                score_correction -= meanwt_j * mean_k
                hessian[:, :] += meanwt_j * (s2_k / s0_k - np.outer(mean_k, mean_k))

        sweep_risk_sets(start, stop, weight, r, X, event_times, on_step)

        log_likelihood = total_event_eta + loglik_correction
        score = total_event_X + score_correction
        return StratumContribution(log_likelihood, score, hessian)

    def baseline_hazard_increments(self, X, start, stop, event, weight, eta):
        """Dispatches to the numba-compiled kernel when available,
        falling back to `_baseline_hazard_increments_python` otherwise --
        see `stratum_contribution`'s docstring for the same pattern and
        why both paths are checked, not just assumed consistent.
        """
        if not _HAS_NUMBA:
            return self._baseline_hazard_increments_python(X, start, stop, event, weight, eta)

        event = event.astype(bool)
        r = safe_exp(eta)
        event_stop = stop[event]
        if event_stop.size == 0:
            return np.array([]), np.array([])

        event_times, inverse = np.unique(event_stop, return_inverse=True)
        K = event_times.shape[0]
        event_weight = weight[event]
        d_star = np.bincount(inverse, weights=event_weight, minlength=K)
        d_raw = np.bincount(inverse, minlength=K).astype(np.int64)

        event_idx = np.where(event)[0].astype(np.int64)
        S0_D, _, _ = _compute_death_sums_numba(X, weight, r, event_idx, inverse.astype(np.int64), K)

        dH0, error_code = _efron_baseline_numba(start, stop, weight, r, event_times, d_star, d_raw, S0_D)
        if error_code == 1:
            raise FloatingPointError(
                "Empty risk set at an event time with positive event weight "
                "while computing the baseline hazard."
            )
        if error_code == 2:
            raise FloatingPointError(
                "Non-positive interpolated Efron denominator while "
                "computing the baseline hazard -- indicates invalid "
                "(start, stop] construction rather than a numerical issue."
            )
        return event_times, dH0

    def _baseline_hazard_increments_python(self, X, start, stop, event, weight, eta):
        """R's own tie-consistent baseline hazard for Efron is NOT
        `d*_j / S0(t_j)` (that's Breslow's estimator) -- it's the average
        of `d_j` reciprocals taken over the same fractional risk-set
        reductions used in the likelihood above:

            dH0(t_j) = meanwt_j * sum_{k=0}^{d_j-1} 1 / (S0_R - (k/d_j)*S0_D)

        Matched to R's `agsurv5.c` (the C routine `survival:::agsurv`
        calls for `ctype=2`, which `survfit.coxph` selects automatically
        whenever the fitted model used `ties="efron"` -- see
        docs/R_COMPATIBILITY.md, question 2b). Note the index runs
        `k=0..d_j-1` here (vs. `k=1..d_j` in `stratum_contribution`
        above) -- the two conventions sum over the identical *set* of
        `d_j` denominator values, just indexed from opposite ends, since
        one formula sums logs and the other sums reciprocals and each
        was transcribed to match its own C source as directly as
        possible rather than forced into a shared indexing scheme.
        """
        n, p = X.shape
        event = event.astype(bool)
        r = safe_exp(eta)
        event_stop = stop[event]
        if event_stop.size == 0:
            return np.array([]), np.array([])

        event_times, inverse = np.unique(event_stop, return_inverse=True)
        K = event_times.shape[0]
        event_weight = weight[event]
        d_star = np.bincount(inverse, weights=event_weight, minlength=K)
        d_raw = np.bincount(inverse, minlength=K)

        event_idx = np.where(event)[0]
        wr_event = event_weight * r[event_idx]
        S0_D = np.zeros(K)
        np.add.at(S0_D, inverse, wr_event)

        dH0 = np.full(K, np.nan)

        def on_step(j, t, S0_R, S1_R, S2_R):
            dj = d_raw[j]
            if dj == 0:
                dH0[j] = 0.0
                return
            if S0_R <= 0:
                raise FloatingPointError(
                    "Empty risk set at an event time with positive event weight "
                    "while computing the baseline hazard."
                )
            if dj == 1:
                dH0[j] = d_star[j] / S0_R
                return

            meanwt_j = d_star[j] / dj
            S0_D_j = S0_D[j]
            total = 0.0
            for k in range(dj):
                denom = S0_R - (k / dj) * S0_D_j
                if denom <= 0:
                    raise FloatingPointError(
                        "Non-positive interpolated Efron denominator while "
                        "computing the baseline hazard -- indicates invalid "
                        "(start, stop] construction rather than a numerical issue."
                    )
                total += 1.0 / denom
            dH0[j] = meanwt_j * total

        sweep_risk_sets(start, stop, weight, r, X, event_times, on_step)
        return event_times, dH0


class ExactTies(TieMethod):
    name = "exact"

    def stratum_contribution(self, *args, **kwargs):
        raise NotImplementedError(
            "ties='exact' is not implemented yet. Exact (discrete/conditional) "
            "tie handling requires enumerating -- or using the recursive "
            "polynomial algorithm for -- all d_j-subsets of the risk set at each "
            "tied event time, which is a genuinely different computational path "
            "from Breslow/Efron rather than a small variation of either."
        )
