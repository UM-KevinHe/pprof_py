# Chapter 6 — Time-Dependent Covariates

## 6.1 A covariate that isn't fixed

Every covariate in Chapters 1–5 was fixed at the start of follow-up:
age at entry, diabetes status at entry, facility at entry. Plenty of
things that matter for survival don't work that way. A patient might
be added to a transplant waiting list eight months into dialysis. A
patient might transfer facilities. A lab value might cross a clinical
threshold. All of these are **time-dependent covariates**: their value
for a given patient is not one number, but a *function of time*.

Handling these correctly is not a minor technical footnote — get it
wrong, in a specific and common way, and you can manufacture a
dramatic, statistically "significant" effect that is entirely an
artifact of how the data was set up, with a real and even famous name
in the epidemiology literature: **immortal time bias**.

## 6.2 Immortal time bias, worked through concretely

Suppose you want to know whether being placed on a transplant waiting
list improves survival. The tempting, naive approach: create one
column, `ever_waitlisted` (1 if the patient was *ever* added to the
list by the end of the study, 0 if not), and put it in an ordinary Cox
model alongside age, diabetes, and so on.

Here is the fatal flaw: **to be added to the waiting list, you first
have to survive long enough to be evaluated and added.** A patient who
died three weeks after starting dialysis never had the chance to be
waitlisted, no matter how good their eventual prognosis might have
been. By construction, the `ever_waitlisted = 1` group is missing
everyone who died early — not because waitlisting saved them, but
because dying early *disqualifies you from ever entering that group in
the first place*. The waitlisted group looks healthier because it's
built entirely out of people who already proved they could survive long
enough to qualify. The time between entering the study and actually
being waitlisted is sometimes called **immortal time**: time during
which, by definition of how the group was formed, you could not yet
have had the event.

**Let's see the size of this, not just describe it.** Simulate a
scenario where waitlisting has *zero* true effect on mortality (built
directly into the simulation, so we know the right answer in advance):

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(2)
n = len(cohort)

# Getting waitlisted has NO true effect on mortality here -- it isn't
# part of the hazard model that generated `cohort["time"]` at all.
potential_waitlist_time = rng.exponential(1.5, n)
got_waitlisted = potential_waitlist_time < cohort["time"].to_numpy()
waitlist_time = np.where(got_waitlisted, potential_waitlist_time, np.nan)
```

The naive, fixed-covariate analysis:

```python
from pprof_py import CoxPH

cohort["ever_waitlisted"] = got_waitlisted.astype(int)
X_naive = cohort[["age", "sex", "diabetes", "comorbidity_count", "ever_waitlisted"]]

naive_model = CoxPH(ties="efron").fit(X_naive, duration=cohort["time"], event=cohort["death"])
naive_model.summary().loc["ever_waitlisted"]
```

```
coef        -1.429
exp(coef)    0.240
p            4.5e-29
```

A hazard ratio of 0.24 — an apparent **76% reduction in mortality risk**,
overwhelmingly statistically significant — for an intervention that,
by construction, does *nothing*. This is not a subtle effect size or a
borderline p-value. This is exactly the size and shape of finding that
gets a poorly-controlled observational study a headline, and it is
entirely fabricated by the immortal time built into the naive
covariate.

## 6.3 The fix: represent time correctly

The correct representation makes `waitlisted` **0 up until the exact
moment a patient is actually added to the list, and 1 from that moment
forward** — never retroactively true for time before it happened. This
means a single patient's follow-up is no longer one row; it's split
into a "before" row and an "after" row, exactly the `(start, stop]`
shape Chapter 2 introduced and `CoxPH` has supported since Chapter 3.
What's been missing until now is the tooling to *build* that shape —
which is exactly what this chapter's two new functions are for.

### `build_skeleton`: the starting point

```python
# build_skeleton, tmerge, UpdateStream and survsplit live in pprof_py.data.timedep;
# they are not exported from the package root.
from pprof_py.data.timedep import build_skeleton

skeleton = build_skeleton(id=cohort["patient_id"].to_numpy(), tstop=cohort["time"].to_numpy())
skeleton.head()
```

```
   id  tstart   tstop
0   0     0.0  1.442
1   1     0.0  0.187
2   2     0.0  2.664
```

One row per patient, `(0, their own total follow-up time]` — the
trivial starting skeleton every `tmerge` call refines.

### `tmerge`: layering in what changed, and when

`tmerge` takes that skeleton plus a separate stream of "this changed at
this time" records and splits each patient's interval wherever
something actually happened to them:

```python
from pprof_py.data.timedep import tmerge, UpdateStream

merged = tmerge(
    skeleton.id.to_numpy(), skeleton.tstart.to_numpy(), skeleton.tstop.to_numpy(),
    tdc={
        "waitlisted": UpdateStream(
            id=cohort["patient_id"].to_numpy()[got_waitlisted],
            time=waitlist_time[got_waitlisted],
        )
    },
    tdc_init={"waitlisted": 0.0},
    event={
        "death": UpdateStream(
            id=cohort["patient_id"].to_numpy()[cohort["death"] == 1],
            time=cohort["time"].to_numpy()[cohort["death"] == 1],
        )
    },
)
```

Two kinds of update streams here, and the difference matters:

- **`tdc`** ("time-dependent covariate") describes a covariate that
  changes value at a point in time and *stays* at its new value from
  then on — exactly `waitlisted`'s behavior. `tdc_init=0.0` says what
  the covariate's value is *before* the first update (nobody starts out
  already waitlisted).
- **`event`** marks a one-time occurrence at a specific moment — here,
  death. `tmerge` doesn't require the outcome to be layered in this way
  (you could also merge it in as an ordinary column), but doing it
  through `event=` lets `tmerge` correctly place the death indicator on
  exactly the interval that ends at each patient's own true death time.

Every argument is a plain array wrapped in `UpdateStream(id=, time=,
value=)` — there's no formula syntax to learn, just "who, when, and
(optionally) what value."

```python
merged = merged.merge(
    cohort[["patient_id", "age", "sex", "diabetes", "comorbidity_count"]],
    left_on="id", right_on="patient_id",
)
print(f"{len(cohort)} patients became {len(merged)} (start, stop] rows")
```

```
4000 patients became 6821 (start, stop] rows
```

Every patient who was ever waitlisted now contributes (at least) two
rows: a `waitlisted=0` row up to the moment they joined the list, and a
`waitlisted=1` row from that moment to their own eventual death or
censoring — with the death indicator correctly attached to only the
one row where it actually happened.

### Refitting, correctly

```python
X_td = merged[["age", "sex", "diabetes", "comorbidity_count", "waitlisted"]]

td_model = CoxPH(ties="efron").fit(
    X_td, start=merged["tstart"], stop=merged["tstop"], event=merged["death"],
)
td_model.summary().loc["waitlisted"]
```

```
coef        -0.135
exp(coef)    0.874
p            0.356
```

The dramatic, spurious effect is gone. A p-value of 0.36 correctly
tells you this data does not support a real waitlisting effect —
exactly the honest answer the simulation was built to have, because
there simply isn't one baked into it. The small residual coefficient
(-0.13 rather than exactly 0) is unbiased sampling noise, the ordinary
kind Chapter 3 already taught you to read a confidence interval for —
nothing like the naive model's wholesale fabrication.

## 6.4 A different tool for a different job: `survsplit`

`tmerge` merges in a *new value that needs to be looked up and carried
forward*. Sometimes you don't need a new value at all — you just need
to cut existing intervals at some fixed set of calendar or follow-up
times, shared by everyone, usually to let an effect differ before and
after a landmark:

```python
from pprof_py.data.timedep import survsplit

split = survsplit(
    id=cohort["patient_id"].to_numpy(),
    tstart=np.zeros(len(cohort)),
    tstop=cohort["time"].to_numpy(),
    event=cohort["death"].to_numpy(),
    cut=[1.0, 2.0],   # split every patient's follow-up at years 1 and 2
)
```

Every patient followed past a cut point now has a row for
`(0, 1]`, `(1, 2]`, and `(2, their own end]` as applicable, each
carrying `event=0` except the true final piece. This is the right tool
for asking "does the effect of diabetes look different in the first
year on dialysis versus later" (fit an interaction between diabetes and
a `tstart_bin_`-derived indicator) or for a landmark analysis (analyze
only survivors past some milestone, from that milestone forward) — not
for representing a covariate that genuinely changes value, which is
`tmerge`'s job.

## 6.5 When this matters most

Immortal time bias specifically shows up whenever a covariate's value
is only "knowable" once some amount of time has already passed —
treatment start, waitlist entry, achieving a lab-value milestone,
surviving to a landmark. If defining your exposure required your
subject to *already be alive*, you almost certainly need `tmerge`, not
a fixed column, no matter how tempting the simpler fixed-column version
looks. The rule of thumb: **a covariate's value at time $t$ must only
ever depend on information available up to and including time $t$** —
never on what happens afterward, and never on whether the subject
survives long enough to "achieve" some later status.

## 6.6 What's next

This chapter handled *one* covariate changing over time for otherwise
straightforward right-censored data. Chapter 7 tackles a different kind
of complication in the same spirit: what happens when there is more
than one way for the story to end — death, yes, but also transplant,
which removes a patient from being at risk of dialysis-related death
entirely. Treating a competing event as ordinary censoring, it turns
out, has its own subtle failure mode, closely related in spirit to this
chapter's immortal time bias.
