(data-preparation-guide)=
# Survival Data Preparation

Four functions reshape and validate survival data before it reaches a
model's `fit()`. `tmerge` and `build_skeleton` already have a full
worked tutorial in
[survival Chapter 6](../survival/06_time_dependent_covariates); this
page is the compact reference for all four together, including
`survsplit` and `validate_fit_inputs`, which Chapter 6 doesn't cover.

## `build_skeleton` → `tmerge`: building time-dependent data

```python
from pprof_py.data.timedep import build_skeleton, tmerge, UpdateStream

skeleton = build_skeleton(id=[1, 2, 3], tstop=[5.0, 8.0, 3.0])
```
```
   id  tstart  tstop
0   1     0.0    5.0
1   2     0.0    8.0
2   3     0.0    3.0
```

One `(tstart, tstop]` row per subject — `tstart` defaults to `0`
(ordinary right-censored follow-up); pass an array to `tstart=` for
delayed entry. `tmerge` then refines this skeleton, splitting each
subject's interval at every new update time and filling in one output
column per `UpdateStream`:

```python
treated = UpdateStream(id=[1, 1, 2], time=[2.0, 4.0], value=[1, 0])   # id=1: on at t=2, off at t=4
merged = tmerge(skeleton, tdc={"treated": treated})
```
```
   id  tstart  tstop  treated
0   1     0.0    2.0      NaN
1   1     2.0    4.0      1.0
2   1     4.0    5.0      0.0
3   2     0.0    3.0      NaN
4   2     3.0    8.0      1.0
5   3     0.0    3.0      NaN
```

`id` may be a DataFrame (the output of a previous `tmerge()` call)
instead of raw arrays, which is what lets you chain several update
streams the way R's `tmerge(data1, data2, ...)` does — pass the
skeleton once, then each subsequent `tmerge()` call's output back in
as `id=` for the next. `tdc=` fills the *current value* at each point
(what's shown above); `cumtdc=` and `cumevent=` instead accumulate a
running count or sum across updates, for covariates like "number of
prior hospitalizations so far" rather than "hospitalized right now."

## `survsplit`: splitting at fixed calendar/follow-up cuts

```python
from pprof_py.data.timedep import survsplit

split = survsplit(id=[1, 2, 3], tstart=[0, 0, 0], tstop=[5.0, 8.0, 3.0],
                   event=[1, 0, 1], cut=[2.0, 4.0])
```
```
   id  tstart  tstop  event tstart_bin_
0   1     0.0    2.0    0.0    (-inf,2]
1   1     2.0    4.0    0.0       (2,4]
2   1     4.0    5.0    1.0     (4,inf]
3   2     0.0    2.0    0.0    (-inf,2]
4   2     2.0    4.0    0.0       (2,4]
5   2     4.0    8.0    0.0     (4,inf]
6   3     0.0    2.0    0.0    (-inf,2]
7   3     2.0    3.0    1.0       (2,4]
```

Unlike `tmerge`, no covariate values need merging in — `cut` points
(here `2.0` and `4.0`) are shared across every subject, not
subject-specific update times, so `survsplit`'s only job is dividing
existing intervals and correctly carrying `event` only into the piece
that actually contains it (every earlier piece a cut creates gets
`event=0`, visible above for subject 1's first two rows). The typical
use is testing a non-proportional-hazards pattern — does a covariate's
effect look different before versus after some duration? — by
interacting a covariate with `tstart_bin_`, or a landmark analysis
that only needs the follow-up broken into pieces, not a new value
carried across each piece. Duplicate cut points are ignored; a cut at
or outside an interval's own endpoints does not create a zero-length
row.

## `validate_fit_inputs`: what every Cox-family `fit()` already runs

```python
from pprof_py.data.survival_validation import validate_fit_inputs

clean = validate_fit_inputs(
    X, duration=cohort["time"], event=cohort["death"],
)   # or start=..., stop=... instead of duration=
```

Every survival model in this documentation calls this internally
before fitting — `CoxPH`, `PenalizedCoxPH`, `GroupLassoCoxPH`,
`ProviderPenalizedCoxPH`, and the discrete-time family all validate
and normalize their inputs through this one function, so its rules are
worth knowing directly rather than only meeting them as an error
message. It enforces exactly one of `duration` or `(start, stop)` —
passing both, or neither, raises a clear `ValueError` rather than
silently picking one. `duration` is exactly the `start=0` special case
of `(start, stop)` internally: "ordinary right-censored data is simply
the start=0 special case of the general counting-process form used
internally throughout this package," per the function's own docstring
— worth knowing if you're ever reading this package's lower-level
risk-set code and wondering why there's no separate code path for the
simple case.

## What's next

The [diagnostics guide](diagnostics-guide) covers `preflight_report`,
which runs many of the same checks `validate_fit_inputs` does — tie
sizes, singleton strata, missing columns — but as a **report you can
call yourself** before fitting, rather than only meeting them as a
`fit()`-time error.
