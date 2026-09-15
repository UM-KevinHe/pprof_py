"""Fine-Gray subdistribution-hazard data transformation.

Competing risks means each subject can fail from one of several causes,
and experiencing cause B precludes ever being observed to fail from
cause A.  Two standard regression strategies both reduce to fitting an
*ordinary* Cox model, which is why this file, not a new likelihood
engine, is what the competing-risks work turns out to need:

  * **Cause-specific hazards**: to model cause A, treat cause-B events
    as ordinary censoring at the time they occur, then fit CoxPH as
    usual.  No new math at all -- see ``models/competing_risks.py``'s
    ``CauseSpecificCoxPH``, which is a direct, unweighted reuse of
    ``CoxPH``.
  * **Fine-Gray subdistribution hazard**: to model the
    *subdistribution* hazard for cause A (the quantity whose
    cumulative-hazard transform is the cumulative incidence function
    actually observed in the presence of competing risks), a subject
    who fails from cause B is NOT censored -- they are kept in cause
    A's risk set indefinitely (up to the administrative end of
    follow-up), because "would this B-subject eventually have failed
    from A instead, had B not happened" is exactly the estimand.
    Since that counterfactual follow-up is never actually observed,
    ``finegray_transform`` below manufactures it statistically: a
    B-subject's row is extended out to the end of follow-up and split
    into several pseudo-observations whose case weight decays over
    time, following the *inverse probability of censoring* -- more
    precisely, R's own convention (Fine & Gray 1999; Geskus,
    Biometrics 2011) of weighting by the ratio of two Kaplan-Meier-
    type curves: ``G``, the censoring distribution, and (when there
    is left truncation) ``H``, the entry (truncation) distribution.
    Feed the result to ``CoxPH.fit(..., sample_weight=fgweight,
    cluster=source_subject)`` (or use
    ``models.competing_risks.FineGrayPH``, which does exactly this)
    and an ordinary weighted Cox fit *is* the Fine-Gray model -- see
    ``models/competing_risks.py`` for why ``cluster=`` (hence robust
    variance) is not optional here.

The implementation follows R's ``survival::finegray()`` approach
(``R/finegray.R``, ``src/finegray.c``, ``noweb/finegray.Rnw``), using
R's own integer-time-scale trick (shifting real-event times back by 0.2
so that a tied censoring/event pair resolves the way Kaplan-Meier
requires) to compute G and H via a standalone ``_km_counting_process``
rather than a full ``survfit()`` call.  Validated against R's actual
``finegray()`` output on the exact worked example in
``tests/test_finegray_transform.py`` (transcribed from
``survival/tests/finegray.R`` / ``finegray.Rout.save``, both bundled
with the R package for exactly this purpose): the same 14-subject
dataset, the same expected row-expansion pattern and case weights,
reproduced here to match R's saved values.

The KM computation uses sorted binary searches (O(n log n) time, O(n)
memory), not observation-by-time boolean matrices, so the transform
scales to production-size datasets without materializing quadratic
intermediate arrays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FineGrayData:
    """Output of ``finegray_transform``: a longer ``(start, stop, status]``
    dataset ready to hand to an ordinary weighted Cox fit.

    ``row`` gives, for every output row, the 0-based index into the
    *original* input arrays it was derived from -- join covariates back
    onto the output via ``X[row]``.

    ``subject`` gives, for every output row, the original subject
    identity -- pass this as ``cluster=`` to ``CoxPH.fit`` /
    ``FineGrayPH``, since several output rows can share one subject and
    the sandwich variance must account for that (see
    ``models/competing_risks.py``).  For one-row-per-subject data,
    ``subject`` is equivalent to ``row``; for multi-row counting-process
    data they differ, and ``subject`` is the correct clustering variable.

    ``added`` is 0 for the original/base piece and 1, 2, ... for each
    synthetic extension of that same row (R's optional ``count=``).
    """

    row: np.ndarray
    subject: np.ndarray
    start: np.ndarray
    stop: np.ndarray
    status: np.ndarray
    weight: np.ndarray
    added: np.ndarray

    @property
    def n_obs(self) -> int:
        """Number of expanded pseudo-observations."""
        return int(self.row.shape[0])

    @property
    def n_subjects(self) -> int:
        """Number of unique source subjects represented in the output."""
        return int(np.unique(self.subject).size)


def _as_1d_float(values, name: str, n: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} has length {arr.shape[0]}, expected {n}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _encode_labels(values, name: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape[0] != n:
        raise ValueError(f"{name} must be one-dimensional of length {n}.")
    try:
        labels, codes = np.unique(raw, return_inverse=True)
    except TypeError as exc:
        raise ValueError(f"{name} contains incompatible labels.") from exc
    return labels, codes.astype(np.int64, copy=False)


def _step_eval_right_continuous(
    jump_times: np.ndarray,
    jump_values: np.ndarray,
    t,
    before_value: float = 1.0,
) -> np.ndarray:
    """Evaluate a right-continuous step function."""
    jump_times = np.asarray(jump_times, dtype=np.float64)
    jump_values = np.asarray(jump_values, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)

    if jump_times.ndim != 1 or jump_values.ndim != 1:
        raise ValueError("jump_times and jump_values must be one-dimensional.")
    if jump_times.shape != jump_values.shape:
        raise ValueError("jump_times and jump_values must have the same shape.")

    if jump_times.size == 0:
        return np.full(t_arr.shape, before_value, dtype=np.float64)

    idx = np.searchsorted(jump_times, t_arr, side="right") - 1
    safe = np.clip(idx, 0, jump_values.size - 1)
    return np.where(idx < 0, before_value, jump_values[safe])


def _validate_subject_structure(
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    subject_codes: np.ndarray,
    strata_codes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Identify first/last rows and apply R's counting-process checks."""
    n = start.size
    original = np.arange(n, dtype=np.int64)

    # R orders by subject and stop time.  Keep original row order as a stable
    # final tie-breaker so duplicate stop times are deterministic.
    order = np.lexsort((original, stop, subject_codes))
    s_subject = subject_codes[order]
    s_start = start[order]
    s_stop = stop[order]
    s_event = event[order]
    s_strata = strata_codes[order]

    first_sorted = np.empty(n, dtype=bool)
    first_sorted[0] = True
    if n > 1:
        first_sorted[1:] = s_subject[1:] != s_subject[:-1]

    last_sorted = np.empty(n, dtype=bool)
    last_sorted[-1] = True
    if n > 1:
        last_sorted[:-1] = s_subject[:-1] != s_subject[1:]

    if np.any((~last_sorted) & (s_event != 0.0)):
        raise ValueError(
            "A subject has a non-zero event before their last time point. "
            "Only the final row of a subject may carry an event."
        )

    same_subject_next = ~last_sorted[:-1]
    if np.any(
        same_subject_next
        & ~np.isclose(s_start[1:], s_stop[:-1], rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "A subject has a gap or overlap between consecutive "
            "(start, stop] intervals."
        )

    if np.any(same_subject_next & (s_strata[1:] != s_strata[:-1])):
        raise ValueError("A subject cannot change strata during follow-up.")

    first = np.zeros(n, dtype=bool)
    last = np.zeros(n, dtype=bool)
    first[order[first_sorted]] = True
    last[order[last_sorted]] = True

    # This intentionally matches finegray.R: delayed entry is enabled when
    # any subject's first start exceeds the minimum observed stop time.
    delay = bool(np.any(start[first] > np.min(stop)))
    return first, last, delay


def _km_counting_process(
    start_time: np.ndarray,
    stop_time: np.ndarray,
    event_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a Kaplan-Meier curve for counting-process intervals.

    The implementation is a sorted sweep: no observation-by-time boolean
    matrix is materialized.  Risk at time ``t`` is ``start < t <= stop``.
    ``event_mask`` marks event endpoints; other interval endpoints are simple
    withdrawals and do not change the KM survival probability.
    """
    start_time = np.asarray(start_time, dtype=np.float64)
    stop_time = np.asarray(stop_time, dtype=np.float64)
    event_mask = np.asarray(event_mask, dtype=bool)

    if start_time.ndim != 1 or stop_time.ndim != 1 or event_mask.ndim != 1:
        raise ValueError("KM inputs must be one-dimensional.")
    if not (start_time.shape == stop_time.shape == event_mask.shape):
        raise ValueError("KM inputs must have the same shape.")
    if np.any(start_time >= stop_time):
        raise ValueError("KM intervals must satisfy start < stop.")

    event_times, event_counts = np.unique(
        stop_time[event_mask], return_counts=True
    )
    if event_times.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    sorted_start = np.sort(start_time)
    sorted_stop = np.sort(stop_time)

    # Vectorized binary searches give the risk-set size at every unique event
    # time in O(m log n), while sorting costs O(n log n).
    risk = (
        np.searchsorted(sorted_start, event_times, side="left")
        - np.searchsorted(sorted_stop, event_times, side="left")
    )
    if np.any(risk <= 0) or np.any(event_counts > risk):
        bad = int(np.flatnonzero((risk <= 0) | (event_counts > risk))[0])
        raise ValueError(
            f"Invalid Kaplan-Meier risk set at time {event_times[bad]}: "
            f"risk={risk[bad]}, events={event_counts[bad]}."
        )

    factors = 1.0 - event_counts.astype(np.float64) / risk.astype(np.float64)
    survival = np.cumprod(factors)
    return event_times, survival


def _censoring_curve(
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    last: np.ndarray,
    utime: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute R's censoring survival curve ``G`` (Geskus 2011, eq. 5).

    ``G`` is the survival function of "time to being censored", treating
    every real event (any cause) as informative removal from the risk
    set rather than as the event being modeled.  R uses an integer-time-
    scale trick (shifting real-event endpoints back by 0.2 on ``utime``
    indices) so that a tied censoring/event pair resolves correctly
    without a special KM tie convention.

    Only a subject's *last* row (``last == True``) with ``event == 0``
    is a real censoring event; an earlier row's ``stop`` is merely a
    covariate-change boundary and must not be counted as a censoring --
    doing so would end that subject's exposure in the risk set twice.

    The curve is computed *unweighted* even if the caller eventually
    applies ``sample_weight`` to the transform's output -- confirmed
    directly from R's source (``R/finegray.R``'s ``survfit()`` calls
    for ``Gsurv`` never receive user weights).
    """
    start_index = np.searchsorted(utime, start, side="left").astype(np.float64)
    stop_index = np.searchsorted(utime, stop, side="left").astype(np.float64)

    # R shifts every real event endpoint 0.2 units earlier on the integer
    # time scale.  That makes an event at t leave the censoring risk set before
    # an ordinary censoring event at the same t, without creating a KM drop.
    transformed_stop = stop_index.copy()
    transformed_stop[event != 0.0] -= 0.2

    terminal_censor = last & (event == 0.0)
    g_time_index, g_survival = _km_counting_process(
        start_index, transformed_stop, terminal_censor
    )
    if g_time_index.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    g_index = np.rint(g_time_index).astype(np.int64)
    if np.any(g_index < 0) or np.any(g_index >= utime.size):
        raise ValueError("Internal censoring-curve time mapping error.")
    return utime[g_index], g_survival


def _truncation_curve(
    start: np.ndarray,
    stop: np.ndarray,
    first: np.ndarray,
    event: np.ndarray,
    utime: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute R's reversed-time entry (truncation) curve ``H``
    (Geskus 2011, eq. 6).

    ``H`` is the survival function of "time of entry", built on a
    reversed time scale.  Only a subject's *first* row
    (``first == True``) represents a genuine study-entry time -- a
    later row's ``start`` is merely a covariate-change boundary and
    counting it as an "entry" would be a distinct subject entering a
    second time.

    ``H`` is needed only when left truncation is present (some subjects
    have ``start > min(stop)``); skipping it when there is no
    truncation avoids a spurious ``H``/``G`` product introducing
    rounding noise where R introduces none.
    """
    if not np.any(first):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    start_index = np.searchsorted(utime, start, side="left").astype(np.float64)
    stop_index = np.searchsorted(utime, stop, side="left").astype(np.float64)
    transformed_stop = stop_index.copy()
    transformed_stop[event != 0.0] -= 0.2

    # R uses Surv(-newstop, -newstart, first).  The first rows are the only
    # genuine entry events; later starts merely represent covariate changes.
    reverse_start = -transformed_stop
    reverse_stop = -start_index
    reverse_time, reverse_survival = _km_counting_process(
        reverse_start, reverse_stop, first
    )
    if reverse_time.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    entry_index_desc = np.rint(-reverse_time).astype(np.int64)
    if np.any(entry_index_desc < 0) or np.any(entry_index_desc >= utime.size):
        raise ValueError("Internal truncation-curve time mapping error.")

    # reverse_time is ascending, so -reverse_time is descending in original
    # time.  Reversing reproduces finegray.R's:
    #   dtime <- rev(-Htemp$time[Htemp$n.event > 0])
    #   dprob <- c(rev(Htemp$surv[...])[-1], 1)
    dtime = utime[entry_index_desc][::-1]
    reversed_survival = reverse_survival[::-1]
    dprob = np.empty_like(reversed_survival)
    if dprob.size == 1:
        dprob[0] = 1.0
    else:
        dprob[:-1] = reversed_survival[:-1]
        dprob[-1] = 1.0
    return dtime, dprob


def _combined_curve(
    G: Tuple[np.ndarray, np.ndarray],
    H: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(jump_times, value)`` for the single right-continuous
    curve the "build" step needs: ``G`` alone if there is no truncation,
    or ``G(t) * H(t)`` (Geskus eq. 11) evaluated at the union of both
    curves' jump points if there is.
    """
    gtime, gprob = G
    if H is None:
        return gtime, gprob

    htime, hprob = H
    temp = np.union1d(gtime, htime)
    g_at = _step_eval_right_continuous(gtime, gprob, temp, before_value=1.0)
    h_at = _step_eval_right_continuous(htime, hprob, temp, before_value=1.0)
    return temp, g_at * h_at


def _build_pieces_one_group(
    row_ids: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    status: np.ndarray,
    failcode: float,
    expand: np.ndarray,
    ctime: np.ndarray,
    cprob: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Row-splitting for one stratum, mirroring ``finegray.c``'s
    ``Cfinegray`` routine.

    ``ctime``/``cprob`` is the *raw* jump grid of the (possibly
    ``G*H``-combined) survival curve -- not yet restricted to the
    "kept" subset described below.

    Every row is emitted once, unchanged, EXCEPT a row flagged
    ``expand[i]`` (a competing event at the last observed row for its
    subject) has its stop time pushed out to the first curve breakpoint
    at or beyond its own stop (weight unchanged at 1 for this piece),
    and then gets one further piece per subsequent *kept* breakpoint,
    each weighted by the curve's ratio to its value at the point the
    row was originally extended from -- i.e. ``G(t-)/G(s-)`` (or
    ``(G*H)(t-)/(G*H)(s-)`` under truncation), ``s`` being this row's
    own original stop.

    A breakpoint is "kept" only if some target-cause (``failcode``)
    event time in this stratum actually falls in the half-open interval
    it closes -- this is a pure size optimization (matching R's own
    "minimal data set" choice, see ``noweb/finegray.Rnw``): the Cox
    partial likelihood only ever evaluates a risk set at a ``failcode``
    event time, so a breakpoint interval containing none literally
    cannot affect any fitted quantity -- dropping it is exact, not
    approximate.
    """
    target_times = np.unique(stop[status == failcode])
    if target_times.size == 0:
        return None

    maxtime = float(np.max(stop))
    breakpoints = np.concatenate([ctime, [maxtime]])
    probabilities = np.concatenate([[1.0], cprob])

    interval_index = np.searchsorted(breakpoints, target_times, side="left")
    if np.any(interval_index >= breakpoints.size):
        raise ValueError("Target event time exceeds the Fine-Gray breakpoint range.")

    keep = np.zeros(breakpoints.size, dtype=bool)
    keep[interval_index] = True
    kept_positions = np.flatnonzero(keep)

    out_row = []
    out_start = []
    out_stop = []
    out_status = []
    out_weight = []
    out_added = []

    for i in range(start.size):
        source_row = int(row_ids[i])
        is_target = status[i] == failcode

        out_row.append(source_row)
        out_start.append(float(start[i]))
        out_stop.append(float(stop[i]))
        out_status.append(1.0 if is_target else 0.0)
        out_weight.append(1.0)
        out_added.append(0)

        if not expand[i]:
            continue

        base_index = int(np.searchsorted(breakpoints, stop[i], side="left"))
        if base_index >= breakpoints.size:
            raise ValueError("Internal Fine-Gray breakpoint construction error.")

        out_stop[-1] = float(breakpoints[base_index])
        base_probability = float(probabilities[base_index])
        if base_probability <= 0.0:
            raise FloatingPointError(
                "Fine-Gray censoring/truncation probability reached zero before "
                "a competing-event extension."
            )

        # Only retained breakpoint intervals need to be emitted.  We still
        # preserve their original order, exactly as Cfinegray does.
        first_kept = int(np.searchsorted(kept_positions, base_index + 1, side="left"))
        added = 0
        for j in kept_positions[first_kept:]:
            added += 1
            out_row.append(source_row)
            out_start.append(float(breakpoints[j - 1]))
            out_stop.append(float(breakpoints[j]))
            out_status.append(0.0)
            out_weight.append(float(probabilities[j] / base_probability))
            out_added.append(added)

    return (
        np.asarray(out_row, dtype=np.int64),
        np.asarray(out_start, dtype=np.float64),
        np.asarray(out_stop, dtype=np.float64),
        np.asarray(out_status, dtype=np.float64),
        np.asarray(out_weight, dtype=np.float64),
        np.asarray(out_added, dtype=np.int64),
    )


def _curve_pair_for_stratum(
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
    delay: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    utime = np.unique(np.concatenate([start, stop]))
    G = _censoring_curve(start, stop, event, last, utime)
    if not delay:
        return G
    H = _truncation_curve(start, stop, first, event, utime)
    return _combined_curve(G, H)


def finegray_transform(
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    failcode,
    id: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> FineGrayData:
    """Build the Fine-Gray pseudo-observation dataset for cause ``failcode``.

    Parameters
    ----------
    start, stop : float arrays
        ``(start, stop]`` follow-up interval, exactly as elsewhere in
        this package -- ordinary right-censored data is ``start = 0``.
    event : float array
        0 = censored; any other value is a cause label (need not be
        contiguous integers).  Exactly one row per subject unless ``id``
        marks genuine ``(start, stop]`` multi-row subjects, in which
        case only a subject's LAST row (by ``stop``, within ``id``) may
        carry a non-zero ``event`` -- an earlier row transitioning is
        rejected, matching R's ``finegray()``.
    failcode : scalar
        Which ``event`` value is the cause of interest.
    id : int/label array, optional
        Subject identifier.  Required for counting-process or delayed-
        entry data with nonzero start times.  For one-row-per-subject
        data with ``start == 0`` it may be omitted (each row is its own
        subject).  Needed both to find each subject's last row and
        (downstream) to populate the ``subject`` field for cluster-
        robust variance.
    strata : array, optional
        ``G``/``H`` and the "kept breakpoints" optimization are computed
        independently within each stratum (matching R's ``strata()``
        term in ``finegray()``'s formula) -- coefficients from a
        downstream fit would still be shared across strata, exactly
        like ``CoxPH``'s own ``strata=``.
    sample_weight : float array, optional
        Ordinary case weights, applied as a final multiplicative factor
        on the output weight -- NOT used when estimating ``G``/``H``
        themselves (see ``_censoring_curve``'s docstring).

    Returns
    -------
    FineGrayData
        Expanded pseudo-observation dataset.  Use ``fg.row`` for
        covariate lookup (``X[fg.row]``) and ``fg.subject`` as
        ``cluster=`` in ``CoxPH.fit()``.
    """
    start = _as_1d_float(start, "start")
    stop = _as_1d_float(stop, "stop", start.size)
    event = _as_1d_float(event, "event", start.size)
    n = start.size

    if np.any(start >= stop):
        raise ValueError("Every interval must satisfy start < stop.")

    failcode_arr = np.asarray(failcode)
    if failcode_arr.ndim != 0:
        raise ValueError("failcode must be a scalar.")
    try:
        failcode_float = float(failcode_arr)
    except (TypeError, ValueError) as exc:
        raise ValueError("failcode must be numeric.") from exc
    if not np.isfinite(failcode_float):
        raise ValueError("failcode must be finite.")
    if failcode_float == 0.0:
        raise ValueError("failcode must be a nonzero event code.")
    if not np.any(event == failcode_float):
        raise ValueError(f"failcode={failcode_float!r} does not occur in `event`.")

    if id is None:
        if np.any(start != 0.0):
            raise ValueError(
                "`id` is required for delayed-entry/counting-process data "
                "with nonzero start times."
            )
        subject_labels = np.arange(n, dtype=np.int64)
        subject_codes = subject_labels.copy()
    else:
        subject_labels, subject_codes = _encode_labels(id, "id", n)

    if strata is None:
        strata_labels = np.array([0])
        strata_codes = np.zeros(n, dtype=np.int64)
    else:
        strata_labels, strata_codes = _encode_labels(strata, "strata", n)

    if sample_weight is None:
        user_weight = np.ones(n, dtype=np.float64)
    else:
        user_weight = _as_1d_float(sample_weight, "sample_weight", n)
        if np.any(user_weight < 0.0):
            raise ValueError("sample_weight must be non-negative.")

    first, last, delay = _validate_subject_structure(
        start, stop, event, subject_codes, strata_codes
    )

    expand_all = last & (event != 0.0) & (event != failcode_float)
    chunks = []

    for stratum_code in range(strata_labels.size):
        idx = np.flatnonzero(strata_codes == stratum_code)
        ctime, cprob = _curve_pair_for_stratum(
            start[idx],
            stop[idx],
            event[idx],
            first[idx],
            last[idx],
            delay,
        )
        built = _build_pieces_one_group(
            row_ids=idx,
            start=start[idx],
            stop=stop[idx],
            status=event[idx],
            failcode=failcode_float,
            expand=expand_all[idx],
            ctime=ctime,
            cprob=cprob,
        )
        if built is None:
            continue
        r, a, b, st, wt, add = built
        chunks.append((r, a, b, st, wt * user_weight[r], add))

    if not chunks:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return FineGrayData(
            row=empty_i.copy(),
            subject=np.empty(0, dtype=subject_labels.dtype),
            start=empty_f.copy(),
            stop=empty_f.copy(),
            status=empty_f.copy(),
            weight=empty_f.copy(),
            added=empty_i,
        )

    rows = np.concatenate([chunk[0] for chunk in chunks])
    subject_by_row = np.asarray(subject_labels)[subject_codes]
    return FineGrayData(
        row=rows,
        subject=subject_by_row[rows],
        start=np.concatenate([chunk[1] for chunk in chunks]),
        stop=np.concatenate([chunk[2] for chunk in chunks]),
        status=np.concatenate([chunk[3] for chunk in chunks]),
        weight=np.concatenate([chunk[4] for chunk in chunks]),
        added=np.concatenate([chunk[5] for chunk in chunks]),
    )
