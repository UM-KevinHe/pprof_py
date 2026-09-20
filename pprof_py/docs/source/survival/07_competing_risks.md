# Chapter 7 — Competing Risks

## 7.1 When "censoring" isn't quite censoring

Extend the running cohort one more time: a dialysis patient can die,
yes, but they can also **receive a kidney transplant**, at which point
they are no longer at risk of dying *on dialysis* at all — the thing
we've been modeling this whole guide. Death and transplant are
**competing risks**: experiencing one fundamentally changes, and
typically ends, the possibility of experiencing the other, for that
patient, ever.

It's tempting to handle transplant the same way you'd handle ordinary
loss to follow-up — just censor the patient at their transplant date
and move on, exactly like Chapter 1's censoring. This is a very natural
mistake, and it is a mistake: ordinary censoring assumes that if you
*hadn't* stopped observing this person, they would have gone on
experiencing the *same* risk of the event you're modeling, at the
*same* rate as everyone else still under observation. That assumption
is exactly, obviously false for a transplant recipient: once
transplanted, they cannot die *on dialysis*, full stop, not at any
rate, ever again. Quietly assuming otherwise systematically distorts
your results, and — as this chapter will show numerically, not just
assert — the distortion runs in a specific, predictable direction:
it **overestimates** how many people would eventually experience the
event of interest.

## 7.2 Two genuinely different questions

Competing risks methodology exists because there are two different,
both entirely legitimate, questions you might be asking, and they
require different tools:

**Question A: "Among people still at risk of dialysis-death right now
(i.e., not yet transplanted, not yet dead), what factors accelerate or
slow that specific risk?"** This is a mechanistic, etiological question
— you want to understand what drives the *rate* of death among the
population currently exposed to it. The **cause-specific hazard**
answers exactly this, and (this is the pleasant surprise in an
otherwise tricky topic) estimating it requires no new machinery
whatsoever: treat a transplant as ordinary censoring, fit an ordinary
Cox model for death, and you have it. Do the same again treating death
as censoring, and you have the cause-specific hazard for transplant.
This is `CauseSpecificCoxPH`, and it is a thin, literal convenience
around two (or more) completely ordinary Cox fits — Section 7.3.

**Question B: "For a patient walking in the door today with these
characteristics, what is their actual probability of eventually dying
on dialysis (as opposed to being transplanted, or the study simply
ending first)?"** This is a prognostic, real-world-probability
question, and this is where the trap in Section 7.1 lives: you cannot
get a valid answer to *this* question by taking a cause-specific
model's survival curve and computing $1 - \hat S(t)$, because that
survival curve was built by (implicitly) censoring competing events —
exactly the false assumption Section 7.1 described. The **Fine-Gray
subdistribution hazard model** answers Question B correctly, by a
different and more subtle construction covered in Section 7.4.

Neither model is more "correct" than the other in general — they
answer different questions, and picking the wrong one for your actual
question is the single most common competing-risks mistake in applied
work.

## 7.3 Cause-specific hazards: the easy half

First, extend the running cohort with the competing event. Each patient gets a latent transplant time; a patient whose transplant would come before their death or censoring is observed to have a transplant instead (`event = 2`). This block defines `final_time`, `event` and `X` used by the rest of the chapter.

```python
import numpy as np

def add_transplant(cohort, rate=0.25, seed=7):
    """Extend `cohort` with a competing event: kidney transplant.

    Returns (final_time, event) with event 0 = censored, 1 = death,
    2 = transplant. Younger patients without diabetes are transplanted
    sooner (purely illustrative)."""
    rng = np.random.default_rng(seed)
    age = cohort["age"].to_numpy()
    diabetes = cohort["diabetes"].to_numpy()
    follow_up = cohort["time"].to_numpy()
    tx_rate = rate * np.exp(-0.03 * (age - 62) - 0.3 * diabetes)
    tx_time = rng.exponential(1.0 / tx_rate)
    transplanted = tx_time < follow_up          # transplanted before death/censoring
    final_time = np.where(transplanted, tx_time, follow_up)
    event = np.where(transplanted, 2, cohort["death"].to_numpy())
    return np.maximum(final_time, 1e-3), event

final_time, event = add_transplant(cohort)
X = cohort[["age", "sex", "diabetes", "comorbidity_count"]]
print(f"{(event == 2).mean():.0%} transplanted, {(event == 1).mean():.0%} died, {(event == 0).mean():.0%} censored")
```

```
38% transplanted, 5% died, 57% censored
```

Now the cause-specific fits:

```python
from pprof_py import CauseSpecificCoxPH

# event: 0 = censored, 1 = death, 2 = transplant
cause_specific = CauseSpecificCoxPH(ties="efron").fit(
    X, event=event, duration=final_time, causes=[1, 2],
)

cause_specific[1].summary()   # the death-specific model
cause_specific[2].summary()   # the transplant-specific model
cause_specific.summary()      # both, stacked, with a leading `cause` column
```

`causes=` defaults to every distinct nonzero value found in `event` if
you don't specify it. Everything about interpreting `cause_specific[1]`
is identical to every ordinary `CoxPH` model in this guide — same
hazard ratios, same standard errors, same `summary()` table — because
under the hood, that's exactly what it is: `event==1` fit as the
outcome, with `event==2` recoded to ordinary censoring at the time it
occurred.

## 7.4 Fine-Gray: modeling the actual cumulative incidence

**The core idea.** Instead of removing a transplanted patient from the
risk set (as cause-specific hazards do), the Fine-Gray approach keeps
them in it — for as long as they *would* have continued to be followed
had transplant not occurred — but down-weights their contribution over
time, according to how likely someone with their exact censoring
pattern would have been to still be under observation at all. This is
a genuinely different modeling target from the cause-specific hazard —
formally, the hazard of the *subdistribution* function — and the
consequence of building the model this way is that $1 - \hat{S}(t)$
from a **Fine-Gray** fit *is* a valid estimate of the true cumulative
incidence, correctly accounting for the fact that some patients are
removed from risk by a competing event along the way. (The full
mathematical construction — why the weighting takes the specific form
it does, built from two Kaplan-Meier-type curves — is documented in
`pprof_py`'s own technical notes for anyone who wants the complete
derivation; what matters for using it correctly is exactly this:
*Fine-Gray's survival curve, unlike a cause-specific one, is safe to
turn into a cumulative incidence by subtracting from 1.*)

```python
from pprof_py import FineGrayPH

fg = FineGrayPH(ties="efron").fit(
    X, event=event, failcode=1, duration=final_time, id=cohort["patient_id"],
)
fg.summary()

cif = 1 - fg.predict_survival_function(X.iloc[:5])   # the actual cumulative incidence of death
```

`failcode=1` says "death is the cause of interest here" — everything
else (`event==2`, transplant) is treated by the Fine-Gray machinery
as the competing event this whole method exists to handle correctly,
not as ordinary censoring. `id=` is required whenever a subject could
in principle have more than one row — pass your subject identifier
even for simple one-row-per-subject data, since the underlying
transform manufactures extra pseudo-observations for competing-event
subjects internally (this is also exactly why `FineGrayPH` reports
cluster-robust standard errors by default, per Chapter 5 — those
pseudo-observations are not independent of each other).

## 7.5 Seeing the bias directly

Take the tempting-but-wrong shortcut from Section 7.1 — reading a
cumulative incidence off the cause-specific model directly — and
compare it to the properly-computed Fine-Gray version, for the same
patients, at the same time point:

```python
naive_cif = 1 - cause_specific[1].predict_survival_function(X.iloc[:200])
correct_cif = 1 - fg.predict_survival_function(X.iloc[:200])

t = naive_cif.index[len(naive_cif) * 3 // 4]
print("naive (mis-specified) mean CIF:  ", naive_cif.loc[t].mean())
print("correct Fine-Gray mean CIF:      ", correct_cif.loc[t].mean())
```

```
naive (mis-specified) mean CIF:   0.0528
correct Fine-Gray mean CIF:       0.0449
```

The naive version overstates the true cumulative incidence of death by
a real, meaningful margin — about 18% too high, in this dataset, where
38% of patients eventually get transplanted. The direction is not a
coincidence: the naive calculation implicitly assumes every transplant
recipient would have gone on accumulating risk of dialysis-death at the
same rate as everyone still on dialysis, when in fact they can no
longer experience that event at all — an assumption that can only ever
push the "risk of death" estimate *up*, never down. The more common the
competing event, the larger this distortion gets; with a rare
competing event, the two numbers converge and the distinction matters
much less in practice.

## 7.6 Choosing between them

| You want to know... | Use |
|---|---|
| What factors accelerate death, among those still exposed to that risk | Cause-specific hazards (`CauseSpecificCoxPH`) |
| The actual probability a given patient will eventually die (vs. transplant, vs. still being followed) | Fine-Gray (`FineGrayPH`) |
| Both, for a complete clinical picture | Both, reported side by side — they aren't in conflict, they answer different questions |

A useful gut check when you're unsure which one you need: if your
sentence is "what causes X to happen faster," you want cause-specific
hazards. If your sentence is "what's the chance X actually happens,"
you want Fine-Gray.

## 7.7 What's next

You've now covered every major extension `coxph` offers for handling
the messy realities of real survival data: robust variance for
non-independent rows, time-varying covariates, and competing outcomes.
The remaining two chapters turn to a different kind of problem —
not "is my data shaped correctly," but "with dozens of candidate
covariates, which ones actually belong in the model, and how do I keep
an overfit model from mistaking noise for signal."
