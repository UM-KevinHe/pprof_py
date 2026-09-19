"""Martingale and score residuals.

## Martingale residuals

For Breslow ties (this package's default), this reduces to the familiar

    M_i = event_i - [H0(stop_i) - H0(start_i)] * exp(eta_i)

-- the [H0(stop) - H0(start)] form (rather than H0(stop) alone) is what's
needed once left truncation is in play, since evaluating H0 at stop_i
alone would overstate a late entrant's expected event count by including
hazard accrued before they entered the risk set (see
docs/R_COMPATIBILITY.md, question 9).

For Efron ties, R's actual `residuals(fit, type="martingale")` is NOT
simply this same formula fed the Efron-consistent H0 -- it applies an
additional, genuinely distinct correction to each tied death's own
residual (see `_martingale_residuals_one_stratum`'s docstring and
docs/R_COMPATIBILITY.md, question 9b). Both cases are handled by the
literal algorithm transcription below, which reduces to the simple
formula above exactly when there are no ties (or `efron=False`).

Weights affect the fitted beta and the hazard this algorithm accumulates,
but the returned residual for observation i is not additionally rescaled
by weight_i -- it is a per-observation diagnostic, not a likelihood
contribution. (Checked against R's `residuals.coxph(type="martingale")`
output under weights during validation, not merely assumed.)

## Score residuals

`score_residuals` returns, per observation, the p-length vector U_i =
integral over time of (x_i(t) - xbar(t)) dM_i(t) -- the per-observation
decomposition of the log partial likelihood's gradient (R's
``residuals(fit, type="score")``).  It is what the robust/sandwich
variance is built from (see ``inference/survival/robust.py``): summed over all observations, U_i reproduces the
total score, which is exactly zero at the fitted beta -- a property this
module's own tests check directly (a self-consistency route to
correctness that does not require R, since it follows from the
definition of a stationary point rather than from any one
implementation).

`_score_residuals_one_stratum_python` mirrors
`_martingale_residuals_one_stratum_python`'s backward risk-set sweep
almost exactly (same reason: R's own C kernel for this, `agscore3.c`,
solves the identical bookkeeping problem `agmart3.c` does, just
accumulating a vector `xhaz` alongside the scalar `cumhaz`, and applying
a per-tied-death correction that needs the batch's mean covariate rather
than a constant +1). One structural difference from a literal
`agscore3.c` port was deliberate: `agscore3.c` identifies which rows in
a tied-stop-time batch are the actual deaths by *array position*
(`person+1 .. person+deaths`), which is only correct given the specific
event-before-censoring sort order its caller (`residuals.coxph.R`)
guarantees upstream (`order(y[,ny-1], -status)`). Re-deriving that
positional invariant here would be one more thing to get exactly right
for no benefit, since checking `event[row]` directly for each row in the
batch -- exactly what `_martingale_residuals_one_stratum_python` already
does -- gives the identical result without depending on tie-breaking
order at all.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...utils.numerical import safe_exp
from ...utils.grouping import iter_stratum_indices

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


def martingale_residuals(
    X: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    eta: np.ndarray,
    weight: np.ndarray,
    strata_codes: np.ndarray,
    strata_labels: np.ndarray,
    baseline_hazard: Optional[pd.DataFrame] = None,
    ties="breslow",
) -> np.ndarray:
    """`ties` selects the residual algorithm directly (see
    `_martingale_residuals_one_stratum` below, which branches on
    `efron=True/False` internally) -- `baseline_hazard` is accepted for
    backwards compatibility but is no longer used: R's own
    `residuals(fit, type="martingale")` does not derive residuals from
    `basehaz()`'s output either (see docs/R_COMPATIBILITY.md, question
    9b) -- it recomputes them via a dedicated algorithm (`agmart3.c`)
    that, for Efron, applies a per-tied-death correction with no
    counterpart in the baseline hazard table at all. Reusing the generic
    [H0(stop)-H0(start)]*score formula with the *correct* Efron H0 (this
    package's earlier fix) is still measurably wrong for tied deaths
    specifically -- confirmed empirically against R, not assumed.
    """
    r = safe_exp(eta)
    event_bool = event.astype(bool)
    n = X.shape[0]
    resid = np.empty(n)

    use_efron = (isinstance(ties, str) and ties == "efron") or getattr(ties, "name", None) == "efron"

    for code, idx in iter_stratum_indices(strata_codes):
        resid[idx] = _martingale_residuals_one_stratum(
            start[idx], stop[idx], event_bool[idx], weight[idx], r[idx], efron=use_efron
        )

    return resid


@njit(cache=True)
def _martingale_residuals_one_stratum_numba(start, stop, event, weight, score, efron: bool) -> np.ndarray:
    n = len(start)
    resid = np.zeros(n)
    atrisk = np.zeros(n, dtype=np.bool_)

    order_stop_desc = np.argsort(-stop)
    order_start_desc = np.argsort(-start)

    person1 = 0
    person2 = 0
    denom = 0.0
    cumhaz = 0.0

    while person2 < n:
        k = person2
        dtime = 0.0
        found_event = False
        while k < n:
            p2 = order_stop_desc[k]
            if event[p2]:
                dtime = stop[p2]
                found_event = True
                break
            k += 1
        if not found_event:
            break

        while person1 < n and start[order_start_desc[person1]] >= dtime:
            p1 = order_start_desc[person1]
            if atrisk[p1]:
                denom -= score[p1] * weight[p1]
                resid[p1] -= cumhaz * score[p1]
            person1 += 1

        deaths = 0
        e_denom = 0.0
        wtsum = 0.0
        k2 = person2
        while k2 < n:
            p2 = order_stop_desc[k2]
            if stop[p2] < dtime:
                break
            if event[p2]:
                atrisk[p2] = True
                resid[p2] = 1.0 + cumhaz * score[p2]
                deaths += 1
                denom += score[p2] * weight[p2]
                e_denom += score[p2] * weight[p2]
                wtsum += weight[p2]
            elif start[p2] < dtime:
                denom += score[p2] * weight[p2]
                atrisk[p2] = True
                resid[p2] = cumhaz * score[p2]
            k2 += 1

        if (not efron) or deaths == 1:
            hazard = wtsum / denom
            person2 = k2
        else:
            hazard = 0.0
            e_hazard = 0.0
            wtsum /= deaths
            for i in range(deaths):
                frac = i / deaths
                d = denom - frac * e_denom
                hazard += wtsum / d
                e_hazard += wtsum * (1.0 - frac) / d
            correction = hazard - e_hazard
            for idx in range(person2, k2):
                p2 = order_stop_desc[idx]
                if event[p2]:
                    resid[p2] += correction * score[p2]
            person2 = k2

        cumhaz += hazard

    while person1 < n:
        p1 = order_start_desc[person1]
        if atrisk[p1]:
            resid[p1] -= cumhaz * score[p1]
        person1 += 1

    return resid


def _martingale_residuals_one_stratum(start, stop, event, weight, score, efron: bool) -> np.ndarray:
    """Direct port of `survival`'s `agmart3.c` (both the Breslow and
    Efron branches), for one stratum. Kept as a literal, line-by-line
    translation of the actual algorithm rather than a hand-derived
    closed form: manually re-deriving a closed form for the Efron
    correction (each tied death's residual gets `(hazard_j - e_hazard_j)
    * score_i` added, distinct from the ordinary [H0(stop)-H0(start)]
    treatment every other observation gets) turned out to be genuinely
    error-prone by hand once intermediate hazard jumps between a
    subject's start and stop are accounted for -- transcribing the
    working C algorithm directly and validating the result against R
    was more reliable than trusting an by-hand simplification of it.

    Dispatches to the numba-compiled kernel when available, falling
    back to `_martingale_residuals_one_stratum_python` otherwise --
    checked to agree via
    test_engine_self_consistency.py::test_martingale_residuals_numba_matches_python_fallback,
    the same cross-check pattern used for the Breslow/Efron likelihood
    and baseline-hazard kernels in algorithms/ties.py.
    """
    if _HAS_NUMBA:
        return _martingale_residuals_one_stratum_numba(start, stop, event, weight, score, efron)
    return _martingale_residuals_one_stratum_python(start, stop, event, weight, score, efron)


def _martingale_residuals_one_stratum_python(start, stop, event, weight, score, efron: bool) -> np.ndarray:
    n = len(start)
    resid = np.zeros(n)
    atrisk = np.zeros(n, dtype=bool)

    order_stop_desc = np.argsort(-stop, kind="mergesort")   # sort2
    order_start_desc = np.argsort(-start, kind="mergesort")  # sort1

    person1 = 0  # pointer into order_start_desc
    person2 = 0  # pointer into order_stop_desc
    denom = 0.0
    cumhaz = 0.0

    while person2 < n:
        # Find the next event time by scanning forward for a death.
        k = person2
        dtime = None
        while k < n:
            p2 = order_stop_desc[k]
            if event[p2]:
                dtime = stop[p2]
                break
            k += 1
        if dtime is None:
            break

        # Remove subjects whose start >= dtime from the risk set,
        # finishing their residual with the hazard accumulated so far.
        while person1 < n and start[order_start_desc[person1]] >= dtime:
            p1 = order_start_desc[person1]
            if atrisk[p1]:
                denom -= score[p1] * weight[p1]
                resid[p1] -= cumhaz * score[p1]
            person1 += 1

        # Add newly at-risk subjects (deaths at dtime, and anyone else
        # -- censored at dtime, or with a larger stop already added
        # earlier in this same tied batch -- whose stop >= dtime).
        deaths = 0
        e_denom = 0.0
        wtsum = 0.0
        k2 = person2
        while k2 < n:
            p2 = order_stop_desc[k2]
            if stop[p2] < dtime:
                break
            if event[p2]:
                atrisk[p2] = True
                resid[p2] = 1.0 + cumhaz * score[p2]
                deaths += 1
                denom += score[p2] * weight[p2]
                e_denom += score[p2] * weight[p2]
                wtsum += weight[p2]
            elif start[p2] < dtime:
                denom += score[p2] * weight[p2]
                atrisk[p2] = True
                resid[p2] = cumhaz * score[p2]
            k2 += 1

        if not efron or deaths == 1:
            hazard = wtsum / denom
            person2 = k2
        else:
            hazard = 0.0
            e_hazard = 0.0
            wtsum /= deaths
            for i in range(deaths):
                frac = i / deaths
                d = denom - frac * e_denom
                hazard += wtsum / d
                e_hazard += wtsum * (1 - frac) / d
            correction = hazard - e_hazard
            for idx in range(person2, k2):
                p2 = order_stop_desc[idx]
                if event[p2]:
                    resid[p2] += correction * score[p2]
            person2 = k2

        cumhaz += hazard

    while person1 < n:
        p1 = order_start_desc[person1]
        if atrisk[p1]:
            resid[p1] -= cumhaz * score[p1]
        person1 += 1

    return resid


def score_residuals(
    X: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    eta: np.ndarray,
    weight: np.ndarray,
    strata_codes: np.ndarray,
    strata_labels: np.ndarray,
    ties="breslow",
) -> np.ndarray:
    """Per-observation score residuals U_i (n x p), R's
    `residuals(fit, type="score")`. NOT rescaled by `weight` -- matching
    R's own default (`weighted=FALSE` for `type="score"`; only
    `type="dfbeta"`/`"dfbetas"` default to `weighted=TRUE`). Use
    `dfbeta_residuals` for the weighted, naive-variance-sandwiched
    version robust variance needs.
    """
    r = safe_exp(eta)
    event_bool = event.astype(bool)
    n, p = X.shape
    resid = np.empty((n, p))

    use_efron = (isinstance(ties, str) and ties == "efron") or getattr(ties, "name", None) == "efron"

    for code, idx in iter_stratum_indices(strata_codes):
        resid[idx] = _score_residuals_one_stratum(
            start[idx], stop[idx], event_bool[idx], weight[idx], X[idx], r[idx], efron=use_efron
        )

    return resid


def dfbeta_residuals(
    score_resid: np.ndarray,
    weight: np.ndarray,
    naive_covariance: np.ndarray,
) -> np.ndarray:
    """dfbeta_i = weight_i * U_i @ naive_covariance -- R's
    `residuals(fit, type="dfbeta")`, and exactly the one-step-Newton
    approximation of (beta_full - beta_leave_i_out) using the full-data
    information matrix in place of recomputing it without observation i
    (see `statistics/inference.py::robust_covariance`, which sandwiches
    these into the sandwich/robust covariance).
    """
    return (score_resid * weight[:, None]) @ naive_covariance


def _score_residuals_one_stratum(start, stop, event, weight, covar, score, efron: bool) -> np.ndarray:
    """Dispatches to the numba-compiled kernel when available, falling
    back to `_score_residuals_one_stratum_python` otherwise -- checked
    to agree via
    test_engine_self_consistency.py::test_score_residuals_numba_matches_python_fallback.
    """
    if _HAS_NUMBA:
        return _score_residuals_one_stratum_numba(start, stop, event, weight, covar, score, efron)
    return _score_residuals_one_stratum_python(start, stop, event, weight, covar, score, efron)


def _score_residuals_one_stratum_python(start, stop, event, weight, covar, score, efron: bool) -> np.ndarray:
    """See this module's docstring for how this relates to `agscore3.c`
    and to `_martingale_residuals_one_stratum_python` above, which this
    mirrors line-for-line wherever the two residuals share bookkeeping
    (the risk-set entry/exit sweep itself), diverging only where the
    quantity being accumulated genuinely differs: a vector `xhaz` (the
    running integral of xbar(t)*hazard(t)) alongside the scalar
    `cumhaz`, a numerator vector `a` for the risk-set's weighted mean
    covariate xbar(t) = a/denom, and a per-tied-death correction that
    uses that batch's own xbar rather than a constant +1.
    """
    n, p = covar.shape
    resid = np.zeros((n, p))
    atrisk = np.zeros(n, dtype=bool)

    order_stop_desc = np.argsort(-stop, kind="mergesort")
    order_start_desc = np.argsort(-start, kind="mergesort")

    person1 = 0
    person2 = 0
    denom = 0.0
    cumhaz = 0.0
    a = np.zeros(p)      # numerator of xbar(t): sum of score*weight*covar over the risk set
    xhaz = np.zeros(p)   # running integral of xbar(t) * d(hazard)(t)

    while person2 < n:
        k = person2
        dtime = None
        while k < n:
            p2 = order_stop_desc[k]
            if event[p2]:
                dtime = stop[p2]
                break
            k += 1
        if dtime is None:
            break

        # Remove subjects leaving the risk set (start >= dtime), finishing
        # their residual with the (cumhaz, xhaz) accumulated so far.
        while person1 < n and start[order_start_desc[person1]] >= dtime:
            p1 = order_start_desc[person1]
            if atrisk[p1]:
                risk = score[p1] * weight[p1]
                denom -= risk
                a -= risk * covar[p1]
                resid[p1] -= score[p1] * (cumhaz * covar[p1] - xhaz)
            person1 += 1

        # Add everyone with stop >= dtime not yet added (the tied deaths
        # at dtime, and anyone censored or with a larger stop already
        # folded in during this same batch) -- initialize each one's
        # residual using the PRE-update (cumhaz, xhaz), before this
        # batch's own hazard increment is folded in below. A row swept
        # into this batch whose OWN start is >= dtime has not actually
        # entered the risk set yet (this can only happen for a
        # non-event row here, since an event row's own dtime equals its
        # own stop, and start < stop always) and must be skipped
        # entirely -- both the denom/a accumulation AND the residual
        # initialization -- exactly as the removal step's `if atrisk`
        # guard already assumes: if it's never marked at risk here, it
        # must never be added here either, or it would inflate the risk
        # set with no corresponding removal to cancel it out.
        deaths = 0
        e_denom = 0.0
        wtsum = 0.0
        a2 = np.zeros(p)
        k2 = person2
        while k2 < n:
            p2 = order_stop_desc[k2]
            if stop[p2] < dtime:
                break
            if event[p2]:
                resid[p2] = score[p2] * (cumhaz * covar[p2] - xhaz)
                risk = score[p2] * weight[p2]
                denom += risk
                a += risk * covar[p2]
                atrisk[p2] = True
                deaths += 1
                e_denom += risk
                wtsum += weight[p2]
                a2 += risk * covar[p2]
            elif start[p2] < dtime:
                resid[p2] = score[p2] * (cumhaz * covar[p2] - xhaz)
                risk = score[p2] * weight[p2]
                denom += risk
                a += risk * covar[p2]
                atrisk[p2] = True
            k2 += 1

        if deaths > 0:
            if not efron or deaths < 2:
                hazard = wtsum / denom
                mean = a / denom
                xhaz += mean * hazard
                cumhaz += hazard
                for idx in range(person2, k2):
                    q = order_stop_desc[idx]
                    if event[q]:
                        resid[q] += covar[q] - mean
            else:
                # Efron: treat the tied batch as `deaths` sequential
                # pseudo-increments (see _martingale_..._python's own
                # Efron branch for the same idea applied to a scalar
                # residual); mh1/mh2/mh3 collect the three running sums
                # agscore3.c calls by the same names, needed because
                # each death's own correction depends on ITS score (via
                # mh1/mh2) as well as the batch-wide average (mh3).
                mh1 = np.zeros(p)
                mh2 = np.zeros(p)
                mh3 = np.zeros(p)
                meanwt = wtsum / deaths
                for dd in range(deaths):
                    downwt = dd / deaths
                    d2 = denom - downwt * e_denom
                    hz = meanwt / d2
                    mean = (a - downwt * a2) / d2
                    xhaz += mean * hz
                    cumhaz += hz
                    mh1 += hz * downwt
                    mh2 += mean * hz * downwt
                    mh3 += mean / deaths
                for idx in range(person2, k2):
                    q = order_stop_desc[idx]
                    if event[q]:
                        resid[q] += (covar[q] - mh3) + score[q] * (covar[q] * mh1 - mh2)
        person2 = k2

    while person1 < n:
        p1 = order_start_desc[person1]
        if atrisk[p1]:
            resid[p1] -= score[p1] * (cumhaz * covar[p1] - xhaz)
        person1 += 1

    return resid


@njit(cache=True)
def _score_residuals_one_stratum_numba(start, stop, event, weight, covar, score, efron: bool) -> np.ndarray:
    n = len(start)
    p = covar.shape[1]
    resid = np.zeros((n, p))
    atrisk = np.zeros(n, dtype=np.bool_)

    order_stop_desc = np.argsort(-stop)
    order_start_desc = np.argsort(-start)

    person1 = 0
    person2 = 0
    denom = 0.0
    cumhaz = 0.0
    a = np.zeros(p)
    xhaz = np.zeros(p)

    while person2 < n:
        k = person2
        dtime = -1.0
        found = False
        while k < n:
            p2 = order_stop_desc[k]
            if event[p2]:
                dtime = stop[p2]
                found = True
                break
            k += 1
        if not found:
            break

        while person1 < n and start[order_start_desc[person1]] >= dtime:
            p1 = order_start_desc[person1]
            if atrisk[p1]:
                risk = score[p1] * weight[p1]
                denom -= risk
                a -= risk * covar[p1]
                resid[p1] -= score[p1] * (cumhaz * covar[p1] - xhaz)
            person1 += 1

        deaths = 0
        e_denom = 0.0
        wtsum = 0.0
        a2 = np.zeros(p)
        k2 = person2
        while k2 < n:
            p2 = order_stop_desc[k2]
            if stop[p2] < dtime:
                break
            if event[p2]:
                resid[p2] = score[p2] * (cumhaz * covar[p2] - xhaz)
                risk = score[p2] * weight[p2]
                denom += risk
                a += risk * covar[p2]
                atrisk[p2] = True
                deaths += 1
                e_denom += risk
                wtsum += weight[p2]
                a2 += risk * covar[p2]
            elif start[p2] < dtime:
                resid[p2] = score[p2] * (cumhaz * covar[p2] - xhaz)
                risk = score[p2] * weight[p2]
                denom += risk
                a += risk * covar[p2]
                atrisk[p2] = True
            k2 += 1

        if deaths > 0:
            if (not efron) or deaths < 2:
                hazard = wtsum / denom
                mean = a / denom
                xhaz += mean * hazard
                cumhaz += hazard
                for idx in range(person2, k2):
                    q = order_stop_desc[idx]
                    if event[q]:
                        resid[q] += covar[q] - mean
            else:
                mh1 = np.zeros(p)
                mh2 = np.zeros(p)
                mh3 = np.zeros(p)
                meanwt = wtsum / deaths
                for dd in range(deaths):
                    downwt = dd / deaths
                    d2 = denom - downwt * e_denom
                    hz = meanwt / d2
                    mean = (a - downwt * a2) / d2
                    xhaz += mean * hz
                    cumhaz += hz
                    mh1 += hz * downwt
                    mh2 += mean * hz * downwt
                    mh3 += mean / deaths
                for idx in range(person2, k2):
                    q = order_stop_desc[idx]
                    if event[q]:
                        resid[q] += (covar[q] - mh3) + score[q] * (covar[q] * mh1 - mh2)
        person2 = k2

    while person1 < n:
        p1 = order_start_desc[person1]
        if atrisk[p1]:
            resid[p1] -= score[p1] * (cumhaz * covar[p1] - xhaz)
        person1 += 1

    return resid