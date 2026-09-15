# Chapter 2 — The Cox Proportional Hazards Model

## 2.1 The proportional hazards idea

Chapter 1 ended with the hazard function $\lambda(t)$ — instantaneous
risk of death at time $t$, given survival to just before $t$. The Cox
model's central move, and the reason it became the default tool in
medical statistics, is to split that hazard into two pieces that never
interact:

$$
\lambda_i(t) = \lambda_0(t) \, \exp(X_i\beta)
$$

- $\lambda_0(t)$, the **baseline hazard**, describes how risk evolves
  over time for a hypothetical patient with all covariates equal to
  zero. It can go up, down, spike after a known milestone, whatever the
  data says — the model places **no assumption on its shape at all**.
- $\exp(X_i\beta) = \exp(X_{i1}\beta_1 + X_{i2}\beta_2 + \dots)$, the
  **relative risk** (or partial hazard), is a single number per
  patient that scales the baseline up or down based on their own
  covariates, and — this is the crucial assumption — **does not change
  over time**.

"Proportional hazards" means exactly this: two patients' hazards stay
in the same *ratio* to each other for as long as both are being
followed, even as the shared baseline $\lambda_0(t)$ rises and falls
underneath them both. A patient with twice the risk multiplier of
another has twice the risk at every age, every year on dialysis, every
stage of the underlying disease process — not just on average.

**Why exponentiate $X_i\beta$ instead of just using $X_i\beta$
directly?** Two reasons, both load-bearing: a hazard must be
non-negative, and $\exp(\cdot)$ guarantees that automatically no matter
what values $\beta$ and $X_i$ take; and it turns *addition* in the
linear predictor into *multiplication* in the hazard
($\exp(a+b)=\exp(a)\exp(b)$), which is exactly what lets you interpret
a coefficient as a hazard *ratio*, coming up next.

## 2.2 Reading a coefficient: the hazard ratio

Suppose $\beta_{\text{diabetes}} = 0.35$ (roughly what our synthetic
cohort was built with). The **hazard ratio** for diabetes is
$\exp(0.35) \approx 1.42$: a diabetic patient's instantaneous risk of
death, at any given moment, is about 42% higher than an otherwise
identical non-diabetic patient's, *at every point in time*, not just on
average over the whole study.

This is the single fact you need to turn any `coxph` coefficient into a
sentence a non-statistician can act on: **exponentiate it, subtract 1,
and read it as a percent change in risk.** $\exp(\beta) > 1$ means
higher risk; $\exp(\beta) < 1$ means protective. `coxph`'s `summary()`
method, covered in Chapter 3, computes this `exp(coef)` column for you
automatically.

For a continuous covariate like age (coefficient $\approx 0.045$ per
year in our cohort), the same logic applies per unit: each additional
year of age multiplies the hazard by $\exp(0.045) \approx 1.046$, and —
because hazards multiply — 10 additional years multiply it by
$\exp(0.045 \times 10) \approx 1.57$, a 57% increase, not a 46% one.
Hazard ratios compound; they don't add.

## 2.3 The partial likelihood: estimating $\beta$ without ever estimating $\lambda_0(t)$

Here is the genuinely clever part of the Cox model, and the reason it
remains the default choice decades after Cox proposed it in 1972: you
can estimate $\beta$ — the part you usually care about — **without
ever having to specify or estimate $\lambda_0(t)$** at all.

**The intuition first.** At the exact moment someone dies, look at
everyone else who is still under observation and still at risk at that
same moment — the **risk set**, exactly as defined in Chapter 1's
Kaplan-Meier section. Ask: *of everyone in this risk set, what's the
probability it was* this *particular person who died, rather than any
of the others?* If risk is proportional to $\exp(X_i\beta)$, then that
conditional probability is simply this person's relative risk divided
by the sum of everyone's relative risk in the risk set:

$$
P(\text{it was patient } i \mid \text{one death at } t, \text{ risk set } R(t)) 
= \frac{\exp(X_i\beta)}{\sum_{j \in R(t)} \exp(X_j\beta)}
$$

Notice what happened: $\lambda_0(t)$ appeared in *every* patient's
hazard at time $t$ equally, so it appears in both the numerator and
every term of the denominator — and cancels out completely. You're
left with a comparison that depends only on $\beta$ and the covariates
of whoever was in the risk set. Multiply this conditional probability
across every observed death in the dataset, and you get the **partial
likelihood**:

$$
L(\beta) = \prod_{i:\,\delta_i=1} \frac{\exp(X_i\beta)}{\sum_{j \in R(t_i)} \exp(X_j\beta)}
$$

`coxph`'s entire fitting procedure (Newton-Raphson optimization, under
the hood) is finding the value of $\beta$ that makes this product as
large as possible — the coefficients that make the deaths that actually
happened look as probable as possible, relative to everyone else who
could have died instead at that moment but didn't. Censored
observations are never entirely absent from this calculation: a
censored patient contributes to the risk set $R(t)$, and hence to
everyone's denominator, for exactly as long as they were actually
under observation — the correct use of their partial information,
promised back in Chapter 1.

Left truncation folds into the exact same formula with no new math at
all: $R(t)$ is simply redefined as everyone whose `(start, stop]`
interval covers $t$ — that is, everyone who had *already entered* by
time $t$ *and* hadn't yet exited. A patient who entered the study two
years into their time on dialysis (Chapter 1's example) is correctly
excluded from every risk set before their entry time, and correctly
included in every one on or after it. `coxph`'s `start=`/`stop=`
arguments (used throughout Chapter 4 onward) exist to make this
possible.

## 2.4 Tied event times: Breslow vs. Efron

The partial likelihood formula above quietly assumes death times are
unique — no two people die at *exactly* the same instant. In continuous
time that's mathematically true, but real data is recorded to the
nearest day, week, or even year, so **ties are common in practice**,
and the formula needs a convention for handling them.

- **Breslow's approximation** treats every tied death as if it
  contributed the *entire current risk set* to the denominator,
  independently, as though the order among the ties didn't matter. It's
  the simpler formula and the historical default in some software, and
  it is `coxph`'s own default too (`ties="breslow"`), matching an
  existing internal SAS workflow this package was built to reproduce.
- **Efron's approximation** is more careful: it accounts for the fact
  that once one of the tied deaths is "used up," it should no longer
  contribute to the *comparison* for the next tied death at that same
  instant, even though we don't know which order they truly happened
  in. It progressively shrinks the tied group's contribution to the
  denominator as each one is notionally accounted for.

**When does this matter in practice?** When ties are rare (fine-grained
time measurement, few events at any one instant), Breslow and Efron
give nearly identical answers. When ties are common (coarse time units,
many simultaneous events — exactly the case for calendar-year-level
ESRD data), they can disagree meaningfully, and Efron's approximation
is the more accurate of the two — it's what R's `survival::coxph()`
uses by default. `coxph` supports both:

```python
from pprof_py import CoxPH

model_breslow = CoxPH(ties="breslow").fit(X, duration=time, event=death)
model_efron   = CoxPH(ties="efron").fit(X, duration=time, event=death)
```

Unless you have a specific reason to match an existing Breslow-based
workflow (as the SMR methodology in Chapter 4 does), prefer
`ties="efron"` for new work.

## 2.5 Stratification: letting the baseline hazard differ by group

The proportional hazards assumption — that everyone's hazard stays in
a fixed ratio to everyone else's — is a strong claim, and it's not
always true *across* natural groupings in your data. 40 facilities'
worth of dialysis patients almost certainly do **not** share one
baseline hazard shape: different facilities might have systematically
different admission timing, different average acuity on the day
someone starts dialysis, or simply different unmeasured local practice
patterns that shift the whole *shape* of risk over time, not just its
level.

**Stratification** solves this by giving each group ($k = 1, \dots,
K$) its own, completely separate baseline hazard, while still sharing
one set of coefficients $\beta$ across all of them:

$$
\lambda_{ik}(t) = \lambda_{0k}(t)\, \exp(X_i\beta)
$$

Mechanically, this changes only ONE thing in the partial likelihood
from Section 2.3: the risk set $R(t)$ at a death time $t$ is now
restricted to *only the other patients in the same stratum* — you
never compare a facility-A patient's risk to a facility-B patient's
risk directly. This is exactly what stage 1 of the two-stage SMR method
in Chapter 4 does, and exactly why: it lets each facility have its own
baseline risk profile (so a facility that happens to admit sicker
patients, on average, doesn't get unfairly blamed for a higher raw
death rate) while still estimating one shared, facility-independent
effect of age, diabetes, and so on.

```python
model = CoxPH(ties="efron").fit(
    X, duration=cohort["time"], event=cohort["death"],
    strata=cohort["facility_id"],
)
```

The trade-off: stratification is powerful precisely because it makes
*no* assumption about how the baseline compares across strata — but
that also means you cannot estimate a *facility effect* directly this
way; a stratified model can tell you the effect of age or diabetes, but
by construction it cannot tell you "facility A's patients do worse than
facility B's," because facility A and facility B are never compared to
each other at all. Getting that facility-level comparison back, fairly,
is exactly the job of the *second* stage in Chapter 4.

## 2.6 Offsets: a covariate whose coefficient you've already decided is 1

An **offset** is a term added to the linear predictor with its
coefficient fixed at exactly 1, rather than estimated from the data:

$$
\lambda_i(t) = \lambda_0(t)\, \exp(X_i\beta + \text{offset}_i)
$$

Two situations call for this, and both come up directly in this guide:

1. **A quantity you already know the "coefficient" for.** In the
   Standardized Mortality Ratio literature (Chapter 4), the second-stage
   model includes an offset for the *log of the population death rate*
   for a patient's race and state — you are not trying to estimate how
   strongly the population death rate predicts an individual's risk
   (that would be circular), you are folding in a known, external
   adjustment and letting the *rest* of the model work around it.
2. **A linear predictor estimated in a previous step**, carried forward
   unchanged into a second model — literally what stage 1's fitted
   $X_i\hat\beta$ becomes when it's fed into stage 2 as an offset in
   Chapter 4. This is what lets a two-stage model behave, statistically,
   like a single carefully-constructed model, without ever writing down
   one combined likelihood for both stages at once.

```python
stage2 = CoxPH(ties="efron").fit(
    X_stage2, duration=time, event=death,
    offset=linear_predictor_from_stage1,
)
```

## 2.7 What you can now do, and what's still missing

You now have every theoretical piece `coxph`'s core `CoxPH` class
implements: the proportional hazards assumption, the partial
likelihood, risk sets (with left truncation folded in for free),
Breslow and Efron ties, stratification, and offsets. Chapter 3 turns
all of this into working code on the running cohort — fitting a model,
reading every number `summary()` produces, and generating predictions.
