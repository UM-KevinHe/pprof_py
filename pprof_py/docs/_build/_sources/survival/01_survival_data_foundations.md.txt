# Chapter 1 — Survival Data: What It Is and Why It's Different

## 1.1 The problem in one picture

Imagine you enroll 4,000 dialysis patients in a study on January 1st
and you want to know, three years later, "how long did people live?"
For some patients the answer is simple and complete: they died 14
months in, full stop. But for many others, the study simply *ended*
before they died — they were alive and well on the last day you looked.
For a smaller group, they moved away, switched insurance, or you
otherwise lost track of them at some point before the study ended.

None of these people give you a "final answer" the way a died-on-day-X
patient does. But they are not useless, either — a patient who was
alive after 3 years tells you something important: *whatever their risk
of dying is, it didn't happen within 3 years.* Survival analysis is the
set of statistical tools built specifically to use that partial
information honestly, rather than discarding it or pretending it's
something it isn't.

## 1.2 Censoring: you know *at least* how long

The formal name for "the study ended, or I lost track of them, before
the event happened" is **censoring**. The specific, by far most common
kind is **right censoring**: you know the person survived up to some
time $C$, and after that you know nothing. This is different from
simply not having a value — you have a *lower bound* on their true
survival time.

For every subject $i$ in a right-censored dataset, there are really two
clocks running: their true time to the event, $D_i$ (death, failure,
whatever "the event" is), and their time to being censored, $C_i$
(end of study, loss to follow-up). You only ever get to observe the
smaller of the two:

$$
T_i = \min(D_i, C_i), \qquad \delta_i = \mathbb{1}(D_i \le C_i)
$$

$T_i$ is the observed time, and $\delta_i$ (the **event indicator**) is
1 if you actually saw the event happen and 0 if the observation was cut
off by censoring first. In `coxph`, and in essentially every survival
package, this pair — a time and a 0/1 event flag — is the basic unit of
data. In our running cohort, that's the `time` and `death` columns:

```python
cohort[["patient_id", "time", "death"]].head()
```

```
   patient_id   time  death
0           0  1.442      1
1           1  0.187      0
2           2  2.664      0
3           3  0.940      1
4           4  0.501      1
```

Patient 1 was followed for only 0.187 years and did not die in that
window (`death=0`) — maybe the study simply ended for them there, maybe
they transferred care elsewhere. We don't know what would have happened
to them at year 1 or year 5. What we *do* know is that they survived at
least 0.187 years, and a correct analysis has to use exactly that much
information — no more, no less.

**Why can't you just drop the censored people, or treat `time` as a
regular outcome and run linear regression on it?** Two reasons, and
both cause real, directional bias, not just lost precision:

- **Dropping censored patients** throws away exactly the patients who,
  so far, are doing *well* (they haven't died yet). What's left behind
  is enriched for the patients who died quickly. Your dataset silently
  becomes sicker than reality, and any estimate of average survival
  will be biased *downward*.
- **Treating the censored time as if it were the true survival time**
  (i.e., pretending patient 1 "survived" exactly 0.187 years and
  nothing more) understates how long people actually live, in the
  opposite but equally wrong way — it treats "still alive" as
  equivalent to "died right now."

Ordinary least-squares regression, and even most of machine learning's
default toolbox, has no built-in concept of "I know a lower bound but
not the exact value." That gap is precisely what survival analysis
fills.

### Left truncation: the other half of the same idea

Censoring is about not knowing what happens *after* your observation
window closes. **Left truncation** (also called *delayed entry*) is the
mirror-image problem at the *start*: sometimes a subject wasn't even
eligible to be observed until some time had already passed.

The clearest example is in this guide's own running example. A patient
enrolled in a facility profiling study might already have been on
dialysis for two years before the study period even began — they don't
enter our data at "time since starting dialysis equals zero," they
enter at "time since starting dialysis equals two years," and if they
had died in year one, we would never have seen them at all. Ignoring
this — analyzing "years since dialysis started" as if everyone entered
at zero — silently removes the sickest, fastest-dying patients from the
early part of the timeline in a way that has nothing to do with them
actually being healthier. This exact issue is why the SMR methodology
in Chapter 4 explicitly includes "duration of ESRD" (dialysis vintage)
as a covariate: it is the field's standard way of acknowledging and
correcting for left truncation.

Handled correctly, every subject contributes to the risk calculation
only during the interval they were actually *at risk and being
observed* — from their entry time to their exit time (event or
censoring), and not a moment before or after. `coxph` represents this
directly as a `(start, stop]` interval per row, which is why you will
see `start=` and `stop=` arguments throughout this guide instead of a
single `time=` column whenever truncation matters (Chapters 4 and 6
use this explicitly).

## 1.3 The hazard function: risk *right now*, given survival so far

The central object in survival analysis is not "the probability of
death," full stop — it's the **hazard function**, $\lambda(t)$: the
instantaneous rate of the event happening at time $t$, *among those
who have survived to just before $t$*.

$$
\lambda(t) = \lim_{dt \to 0} \frac{1}{dt}\, P(t \le D < t + dt \mid D \ge t)
$$

Read the words, not just the symbols: *given that you've made it to
time $t$ alive, what's your instantaneous risk of dying right now?*
This conditioning — "given you've survived this far" — is what makes
the hazard fundamentally different from an unconditional probability,
and it's exactly the right way to think about risk that can change
over someone's lifetime (rising as patients age or accumulate time on
dialysis, for instance).

A hazard doesn't have to be constant. It can go up, down, spike, or
stay flat. What proportional hazards models (the whole subject of
Chapter 2) do is separate the hazard into a **baseline** shape that's
free to be whatever it wants over time, times a **multiplier** that
depends on a person's own characteristics and doesn't change over time.
That's the "proportional" in "proportional hazards" and it is the
single most important modeling assumption in this entire guide — worth
sitting with before moving on.

## 1.4 The survival function: the flip side of the hazard

The **survival function** $S(t) = P(D > t)$ is the probability of
*not* having had the event by time $t$ — the familiar "percent still
alive at time $t$" curve. It's mathematically just the hazard
function's photographic negative: accumulate ("integrate") the hazard
over time to get the **cumulative hazard** $\Lambda(t) = \int_0^t
\lambda(u)\,du$, and then

$$
S(t) = \exp(-\Lambda(t))
$$

This relationship is worth internalizing because it's exactly how
`coxph` computes every survival curve it ever shows you: it estimates
the cumulative hazard first (Chapter 3 shows this as
`baseline_hazard_` and `predict_cumulative_hazard`), and *then*
exponentiates to get a survival curve, never the other way around.

### The Kaplan-Meier curve: a first, model-free estimate

Before you fit any regression at all, there's a simple, honest way to
estimate $S(t)$ for a group of patients directly from the data, using
only who was still being observed and who had an event at each
distinct event time. This is the **Kaplan-Meier estimator**, and while
`coxph` does not implement it directly (it is a *non-parametric*
one-group summary, not a regression), it is worth seeing once, because
the same "who is still at risk right now" bookkeeping is the seed idea
behind everything the Cox model does in Chapter 2:

$$
\hat{S}(t) = \prod_{t_j \le t} \left(1 - \frac{d_j}{n_j}\right)
$$

where at each observed death time $t_j$, $d_j$ is the number of deaths
and $n_j$ is the number of people still **at risk** — still under
observation, neither dead nor censored yet — at that exact moment. Two
things about this formula matter for everything that follows:

- Every time someone is censored, they simply drop out of $n_j$ from
  that point forward — they contribute to the "still at risk" count for
  as long as they were actually observed, and not a moment longer. This
  is the mechanical embodiment of "use exactly the information you
  have, no more, no less" from Section 1.2.
- The set of people "at risk" at time $t_j$ — the **risk set** — is
  the central bookkeeping object in survival analysis. Every method in
  this guide, from the plain Cox model to Fine-Gray competing risks in
  Chapter 7, is fundamentally a different way of defining or weighting
  that risk set and comparing whoever's in it.

## 1.5 What's next

You now have the vocabulary this entire guide is built on: censoring,
truncation, the hazard function, the survival function, and the risk
set. Chapter 2 takes the single-group Kaplan-Meier idea above and
extends it into a full regression model — one that lets age, diabetes,
facility, and any other covariate shift a patient's hazard up or down,
while leaving the underlying shape of risk over time — the baseline
hazard — completely unspecified. That model is the Cox proportional
hazards model, and it is what every other chapter in this guide is
built on top of.
