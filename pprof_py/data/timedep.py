"""Time-dependent-covariate data preparation: building a (start, stop]
counting-process dataset from a fixed follow-up skeleton plus one or more
streams of "this covariate changed value at this time" or "this event
happened at this time" records -- R's `tmerge()` and `survSplit()`.

This is the convenience API for time-dependent covariates: ``CoxPH``
natively supports time-varying covariates by accepting (start, stop]
rows with different covariate values for the same
subject -- a time-dependent covariate is not a new statistical feature
there, only a particular way of shaping the input.  What was missing is
the tooling to BUILD that shape from the way the data usually starts out:
a base follow-up window per subject, plus separate records of when each
covariate changed or an event occurred.

`tmerge` and `survsplit` are pure data reshaping -- no risk sets, no
hazard estimation, nothing statistical -- which is why they live in
`data/`, not `algorithms/`, alongside `data/survival_data.py`.

R's own `tmerge()` evaluates `tdc()`/`event()`/etc. terms written inline
against a second dataframe via non-standard evaluation.  This package has
no formula interface (see docs/R_COMPATIBILITY.md), so `tmerge` here takes
plain arrays or a DataFrame skeleton plus `UpdateStream(id, time, value)`
objects grouped by update type (`tdc`, `cumtdc`, `event`, `cumevent`).
`tmerge()` may be called repeatedly on its own DataFrame output so
previously created columns are preserved across passes, matching R's own
chained-call idiom.

R also lets a bare `event()` term (with no existing tstart/tstop on
`data1`) implicitly establish the skeleton's own tstop from the event's
own times; this module does not replicate that dual role -- `build_skeleton`
is the explicit, separate way to create that starting skeleton, so that
every call to `tmerge` itself always has one unambiguous job: refine an
already-fully-specified set of (start, stop] intervals.

Every semantic rule below (which side of a tie an update lands on, what
happens to an update recorded before a subject's `tstart` or after their
`tstop`, how `cumtdc`'s `init` combines with pre-`tstart` updates) was
worked out and checked against R's own bundled examples in
`survival/tests/tmerge.R` / `tmerge.Rout.save` -- reproduced verbatim in
`tests/test_timedep.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _validate_ids(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional.")
    if pd.isna(arr).any():
        raise ValueError(f"`{name}` must not contain missing IDs.")
    return arr


def _validate_finite_times(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return arr



@dataclass
class UpdateStream:
    """One column's worth of "this changed at this time" records, for
    `tmerge`. `id` and `time` must be the same length; `value` too, if
    given.

    `value=None` means an indicator stream (R's `tdc(y)`/`event(y)` with
    no `x`): every record contributes the value 1.0 -- e.g. `event`
    without a value marks a plain 0/1 event, and `tdc` without a value
    marks a covariate switching on (1) starting at `time`.
    """

    id: np.ndarray
    time: np.ndarray
    value: Optional[np.ndarray] = None

    def __post_init__(self):
        self.id = _validate_ids(self.id, "UpdateStream.id")
        self.time = _validate_finite_times(self.time, "UpdateStream.time")
        if self.time.shape[0] != self.id.shape[0]:
            raise ValueError("UpdateStream: `id` and `time` must have the same length.")
        if self.value is None:
            self.value = np.ones_like(self.time)
        else:
            self.value = np.asarray(self.value, dtype=np.float64)
            if self.value.ndim != 1 or self.value.shape[0] != self.id.shape[0]:
                raise ValueError("UpdateStream: `value` must have the same length as `id`/`time`.")
            if not np.all(np.isfinite(self.value)):
                raise ValueError("UpdateStream: `value` must contain only finite values.")


def build_skeleton(id, tstop, tstart=None) -> pd.DataFrame:
    """The trivial starting skeleton `tmerge` refines: one row per
    subject, `(tstart, tstop]` -- `tstart` defaults to 0 for everyone
    (ordinary right-censored follow-up; pass an array for delayed entry).
    """
    id_arr = _validate_ids(id, "id")
    tstop_arr = _validate_finite_times(tstop, "tstop")
    if tstop_arr.shape[0] != id_arr.shape[0]:
        raise ValueError("`tstop` must have the same length as `id`.")
    tstart_arr = (
        np.zeros_like(tstop_arr)
        if tstart is None
        else _validate_finite_times(tstart, "tstart")
    )
    if tstart_arr.shape[0] != id_arr.shape[0]:
        raise ValueError("`tstart` must have the same length as `id`.")
    if np.any(tstart_arr >= tstop_arr):
        raise ValueError("Every `tstart` must be strictly less than its `tstop`.")
    return pd.DataFrame({"id": id_arr, "tstart": tstart_arr, "tstop": tstop_arr})


def _prepare_skeleton(id, tstart, tstop):
    """Return a copied skeleton frame plus validated arrays."""
    if isinstance(id, pd.DataFrame):
        if tstart is not None or tstop is not None:
            raise ValueError("When `id` is a DataFrame, do not also supply `tstart` or `tstop`.")
        data = id.copy()
        missing = {"id", "tstart", "tstop"} - set(data.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Input DataFrame is missing required columns: {missing_text}.")
        id_arr = _validate_ids(data["id"].to_numpy(), "id")
        tstart_arr = _validate_finite_times(data["tstart"].to_numpy(), "tstart")
        tstop_arr = _validate_finite_times(data["tstop"].to_numpy(), "tstop")
    else:
        if tstart is None or tstop is None:
            raise ValueError("`tstart` and `tstop` are required when `id` is not a DataFrame.")
        id_arr = _validate_ids(id, "id")
        tstart_arr = _validate_finite_times(tstart, "tstart")
        tstop_arr = _validate_finite_times(tstop, "tstop")
        data = pd.DataFrame({"id": id_arr, "tstart": tstart_arr, "tstop": tstop_arr})

    n = id_arr.shape[0]
    if tstart_arr.shape[0] != n or tstop_arr.shape[0] != n:
        raise ValueError("`id`, `tstart`, and `tstop` must all have the same length.")
    if np.any(tstart_arr >= tstop_arr):
        raise ValueError("Every `tstart` must be strictly less than its `tstop`.")

    return data, id_arr, tstart_arr, tstop_arr


def _validate_nonoverlapping_intervals(
    id_arr: np.ndarray,
    tstart_arr: np.ndarray,
    tstop_arr: np.ndarray,
    subject_order: np.ndarray,
) -> Dict:
    """Return input-row indices grouped by subject, ordered by tstart."""
    rows_by_subject: Dict = {}
    for subj in subject_order:
        mask = id_arr == subj
        indices = np.flatnonzero(mask)
        order = indices[np.argsort(tstart_arr[indices], kind="mergesort")]
        starts = tstart_arr[order]
        stops = tstop_arr[order]
        if starts.size > 1 and np.any(starts[1:] < stops[:-1]):
            raise ValueError(f"Intervals for subject {subj!r} overlap.")
        rows_by_subject[subj] = order.tolist()
    return rows_by_subject


def _stream_pairs_for_subject(stream: UpdateStream, subj) -> list[tuple[float, float]]:
    mask = stream.id == subj
    if not np.any(mask):
        return []
    # Stable sorting preserves input order for exact time ties.
    pairs = list(zip(stream.time[mask].tolist(), stream.value[mask].tolist()))
    pairs.sort(key=lambda pair: pair[0])
    return pairs


def tmerge(
    id,
    tstart=None,
    tstop=None,
    tdc: Optional[Dict[str, UpdateStream]] = None,
    cumtdc: Optional[Dict[str, UpdateStream]] = None,
    event: Optional[Dict[str, UpdateStream]] = None,
    cumevent: Optional[Dict[str, UpdateStream]] = None,
    tdc_init: Optional[Dict[str, float]] = None,
    cumtdc_init: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Refine an existing `(id, tstart, tstop]` skeleton by splitting each
    subject's intervals at every new update time and filling in one output
    column per entry across `tdc`/`cumtdc`/`event`/`cumevent`.

    `id` may be either the three skeleton arrays (`id`, `tstart`, `tstop`)
    or a DataFrame containing those columns.  Passing the DataFrame output
    of a previous `tmerge()` call preserves all existing columns, allowing
    chained calls like R's `tmerge(data1, data2, ...)`.

    Each existing interval is refined independently.  Updates falling in a
    gap between a subject's existing intervals do not create artificial
    intervals.  Updates after the subject's overall follow-up are ignored;
    updates at or before an interval's start affect the first resulting
    piece but do not create a piece by themselves (but -- confirmed against
    R's own "surfeit of rows" test -- still compete to be the value the
    very first output interval carries forward; the most recent such
    update wins).

    Column semantics, each keyed by ``time <= row's own tstart``
    (``tdc``/``cumtdc``) or ``time == row's own tstop`` (``event``)
    or ``time <= row's own tstop`` (``cumevent``):

    - ``tdc[name]``: the value of the MOST RECENT qualifying update, or
      ``tdc_init[name]`` (default ``nan``) if none yet.
    - ``cumtdc[name]``: ``cumtdc_init[name]`` (default 0.0) plus the SUM
      of every qualifying update's value.
    - ``event[name]``: the value of the last update exactly at this row's
      own ``tstop``, else 0.0.  Give ``value=<cause code>`` in the
      ``UpdateStream`` to mark cause-of-event directly, e.g. for
      ``CauseSpecificCoxPH``/``FineGrayPH``.
    - ``cumevent[name]``: 0 unless an event lands exactly on this row's
      own ``tstop``, in which case the CUMULATIVE count/sum of every
      qualifying event through and including this one (not carried forward
      to later, non-event rows -- sparse like ``event``, just with a
      running total in place of that occurrence's own value).

    Subjects are emitted in the order their ``id`` first appears in the
    input, matching R's own convention (row order tracks ``data1``'s order,
    not sorted ``id`` order).
    """
    data, id_arr, tstart_arr, tstop_arr = _prepare_skeleton(id, tstart, tstop)

    tdc = {} if tdc is None else dict(tdc)
    cumtdc = {} if cumtdc is None else dict(cumtdc)
    event = {} if event is None else dict(event)
    cumevent = {} if cumevent is None else dict(cumevent)
    tdc_init = {} if tdc_init is None else dict(tdc_init)
    cumtdc_init = {} if cumtdc_init is None else dict(cumtdc_init)

    for kind_name, kind_dict in (
        ("tdc", tdc),
        ("cumtdc", cumtdc),
        ("event", event),
        ("cumevent", cumevent),
    ):
        for name, stream in kind_dict.items():
            if not isinstance(stream, UpdateStream):
                raise TypeError(f"{kind_name}[{name!r}] must be an UpdateStream.")

    all_columns = list(tdc) + list(cumtdc) + list(event) + list(cumevent)
    if len(all_columns) != len(set(all_columns)):
        raise ValueError("Column names across `tdc`/`cumtdc`/`event`/`cumevent` must be unique.")
    existing_non_key = set(data.columns) - {"id", "tstart", "tstop"}
    conflicts = sorted(existing_non_key.intersection(all_columns))
    if conflicts:
        raise ValueError(
            "New tmerge column(s) already exist in the input DataFrame: "
            + ", ".join(map(str, conflicts))
        )

    subject_order = pd.unique(id_arr)
    rows_by_subject = _validate_nonoverlapping_intervals(
        id_arr, tstart_arr, tstop_arr, subject_order
    )

    # Collect each stream's subject-specific updates once.  Only the subject
    # that owns a skeleton interval can contribute to that interval.
    stream_dicts = (tdc, cumtdc, event, cumevent)
    subject_updates = {}
    for subj in subject_order:
        subject_updates[subj] = {}
        for kind_dict in stream_dicts:
            for name, stream in kind_dict.items():
                if name not in subject_updates[subj]:
                    subject_updates[subj][name] = _stream_pairs_for_subject(stream, subj)

    out_rows = []

    for subj in subject_order:
        for source_index in rows_by_subject[subj]:
            base_row = data.iloc[source_index].to_dict()
            lo0 = float(tstart_arr[source_index])
            hi0 = float(tstop_arr[source_index])

            # Only updates strictly inside THIS interval create new pieces.
            # This preserves gaps in the pre-existing skeleton.
            breakpoints = {lo0, hi0}
            for kind_dict in stream_dicts:
                for name in kind_dict:
                    for update_time, _ in subject_updates[subj][name]:
                        if lo0 < update_time < hi0:
                            breakpoints.add(update_time)

            breakpoints = sorted(breakpoints)

            for segment_index in range(len(breakpoints) - 1):
                lo = breakpoints[segment_index]
                hi = breakpoints[segment_index + 1]
                row = base_row.copy()
                row["id"] = subj
                row["tstart"] = lo
                row["tstop"] = hi

                for name in tdc:
                    pairs = subject_updates[subj][name]
                    qualifying = [value for time, value in pairs if time <= lo]
                    row[name] = qualifying[-1] if qualifying else tdc_init.get(name, np.nan)

                for name in cumtdc:
                    pairs = subject_updates[subj][name]
                    qualifying = [value for time, value in pairs if time <= lo]
                    if name in cumtdc_init:
                        row[name] = cumtdc_init[name] + sum(qualifying)
                    elif qualifying:
                        row[name] = sum(qualifying)
                    else:
                        # No explicit init AND no qualifying update yet:
                        # R shows NaN here, not 0 -- an unspecified running
                        # count is "not yet meaningful" before anything has
                        # happened.  Once it does start (a later row), it
                        # accumulates from 0 as usual.  Checked against R's
                        # tmerge.R fourth test block.
                        row[name] = np.nan

                for name in event:
                    pairs = subject_updates[subj][name]
                    at_stop = [value for time, value in pairs if time == hi]
                    # For exact-time ties, the last supplied event value wins.
                    row[name] = at_stop[-1] if at_stop else 0.0

                for name in cumevent:
                    # Sparse like `event` (0 unless an event lands exactly
                    # on this row's tstop), but the value is the CUMULATIVE
                    # total through and including this occurrence, NOT
                    # carried forward to later non-event rows.  Checked
                    # against R's tmerge.R fourth test block.
                    pairs = subject_updates[subj][name]
                    at_stop = [value for time, value in pairs if time == hi]
                    row[name] = sum(value for time, value in pairs if time <= hi) if at_stop else 0.0

                out_rows.append(row)

    # Preserve the original column order, then append new tmerge columns.
    output_columns = list(data.columns) + all_columns
    return pd.DataFrame(out_rows, columns=output_columns)


def survsplit(id, tstart, tstop, event, cut) -> pd.DataFrame:
    """Split existing `(tstart, tstop]` rows at fixed calendar/follow-up
    times `cut`, shared across every subject -- R's `survSplit()`.
    Unlike `tmerge`, no covariate or event value needs merging in: this
    only needs `event` itself (so the piece a subject's real event or
    censoring falls into keeps it, and every earlier piece introduced by
    a `cut` point gets `event=0`).

    Duplicate cut points are ignored.  Cuts at or outside an interval's
    endpoints do not create zero-length pieces.

    Typical use: modeling a covariate's effect as different before/after
    some duration (e.g. a non-proportional-hazards check), or a landmark
    analysis -- neither needs a new *value* to carry across the cut, only
    the interval broken into pieces.

    Returns a frame with ``id``, ``tstart``, ``tstop``, ``event``, and
    ``tstart_bin_`` (each row's cut-interval label).
    """
    id_arr = _validate_ids(id, "id")
    tstart_arr = _validate_finite_times(tstart, "tstart")
    tstop_arr = _validate_finite_times(tstop, "tstop")
    event_arr = np.asarray(event, dtype=np.float64)
    if event_arr.ndim != 1:
        raise ValueError("`event` must be one-dimensional.")
    if not np.all(np.isfinite(event_arr)):
        raise ValueError("`event` must contain only finite values.")

    n = id_arr.shape[0]
    if not (tstart_arr.shape[0] == tstop_arr.shape[0] == event_arr.shape[0] == n):
        raise ValueError("`id`, `tstart`, `tstop`, and `event` must all have the same length.")
    if np.any(tstart_arr >= tstop_arr):
        raise ValueError("Every `tstart` must be strictly less than its `tstop`.")

    cut_arr = np.unique(_validate_finite_times(cut, "cut"))

    out_id, out_start, out_stop, out_event, out_bin = [], [], [], [], []
    for i in range(n):
        lo, hi = tstart_arr[i], tstop_arr[i]
        relevant_cuts = cut_arr[(cut_arr > lo) & (cut_arr < hi)]
        breakpoints = np.concatenate(([lo], relevant_cuts, [hi]))

        for j in range(breakpoints.shape[0] - 1):
            seg_lo, seg_hi = breakpoints[j], breakpoints[j + 1]
            is_last = j == breakpoints.shape[0] - 2

            out_id.append(id_arr[i])
            out_start.append(seg_lo)
            out_stop.append(seg_hi)
            out_event.append(event_arr[i] if is_last else 0.0)

            left_index = np.searchsorted(cut_arr, seg_lo, side="right") - 1
            right_index = np.searchsorted(cut_arr, seg_hi, side="left")
            left_label = f"{cut_arr[left_index]:g}" if left_index >= 0 else "-inf"
            right_label = f"{cut_arr[right_index]:g}" if right_index < cut_arr.size else "inf"
            out_bin.append(f"({left_label},{right_label}]")

    return pd.DataFrame(
        {
            "id": out_id,
            "tstart": out_start,
            "tstop": out_stop,
            "event": out_event,
            "tstart_bin_": out_bin,
        }
    )
