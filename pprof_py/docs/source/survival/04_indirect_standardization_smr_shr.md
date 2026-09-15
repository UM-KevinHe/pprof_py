# Chapter 4 — Indirect Standardization: The Standardized Mortality Ratio

This chapter is the reason this package exists. Everything in Chapters
1–3 — the partial likelihood, stratification, offsets — was building
toward exactly this method, and everything after this chapter is an
extension of it to messier, more realistic data. If you read only one
chapter closely, make it this one.

## 4.1 The problem: comparing facilities fairly

Facility A has a raw death rate of 12%. Facility B has a raw death rate
of 8%. Is Facility A doing worse?

Maybe. Or maybe Facility A simply treats older, sicker, longer-tenured
dialysis patients — the kind who are at elevated risk of death no
matter which facility treats them — while Facility B happens to have a
younger, healthier patient mix. A raw comparison of death rates
conflates two completely different things: how *sick* a facility's
patients are, and how *well* the facility cares for patients of any
given sickness level. Only the second one is something a facility can
actually be held accountable for.

**Indirect standardization** is the general statistical strategy for
separating these two things. The idea, in one sentence: figure out how
many deaths you would *expect* at a facility if it performed exactly
like the national average, given precisely who its patients are — and
then compare that expected number to what was *actually* observed. The
**Standardized Mortality Ratio (SMR)** is that comparison:

$$
\text{SMR}_k = \frac{O_k}{E_k}
$$

where $O_k$ is the observed number of deaths at facility $k$ and $E_k$
is the number you'd expect if facility $k$'s patients had experienced
the *national* risk associated with their own age, diabetes status,
comorbidities, and so on. An SMR of 1.30 means "30% more deaths than
expected, given who this facility treats" — a meaningful, actionable
signal, in a way that a raw death rate on its own never can be.

**Standardized Hospitalization Ratio (SHR)** is the identical idea
applied to a hospitalization event instead of death — everything in
this chapter carries over by simply substituting the event definition.
This chapter says "SMR" and "death" throughout; mentally substitute
"SHR" and "hospitalization" wherever useful.

## 4.2 Why this needs two models, not one

You might reasonably ask: why not just fit one ordinary Cox model with
facility as a covariate, and read off each facility's coefficient? Two
reasons rule this out:

1. **A single shared baseline hazard, estimated across all facilities
   together, is exactly the wrong tool here.** If facility identity
   itself is used as a covariate in one unstratified model, a facility
   with a genuinely different patient mix — not different quality, just
   different patients — will distort the *shared* baseline hazard shape
   for everyone, and the facility coefficients end up entangled with
   that distortion. Section 2.5 introduced stratification precisely to
   let each facility have its own baseline shape without polluting the
   others.
2. **But a fully stratified model can't produce facility-level
   coefficients at all** — Section 2.5 also noted this: stratification
   works by never comparing patients across strata, so there is
   mechanically no way to read off "how facility A compares to facility
   B" from a model that never lets A and B's patients enter the same
   risk set.

The two-stage design resolves this tension by using each model for
exactly the job it's good at, and nothing else:

- **Stage 1** is stratified by facility — its entire job is to get an
  honest, facility-independent estimate of how age, diabetes,
  comorbidities, and so on affect risk, using the *within-facility*
  comparisons that stratification makes fair.
- **Stage 2** is unstratified, with stage 1's fitted risk estimate
  folded in as a *fixed* offset (Section 2.6) — its entire job is to
  estimate one shared, national baseline hazard curve that's consistent
  with everyone's stage-1-implied relative risk. Once you have a
  single national curve like this, you can ask, for any facility,
  "how many deaths would this specific group of patients have produced
  under the national curve?" — and that question has a clean, honest
  answer for the first time.

## 4.3 Notation

Following the same notation style as the CMS ESRD methodology this
package was built around:

| Symbol | Meaning |
|---|---|
| $i$ | a patient-year record (one patient, one facility, one year — a patient followed for several years, or who switches facilities, contributes one record per year) |
| $X_i = (X_{i1}, \dots, X_{ip})$ | patient $i$'s covariates: age, sex, diabetes, comorbidities, dialysis vintage, etc. |
| $X_i\beta$ | the linear predictor: $X_{i1}\beta_1 + \dots + X_{ip}\beta_p$ |
| $t_i$ | patient $i$'s observed follow-up time (end of study, death, or loss to follow-up) |
| $\lambda_{0k}(t)$ | the baseline hazard for facility $k$ (stage 1) |
| $\lambda_0(t)$ | the single national baseline hazard (stage 2) |
| $k = 1, \dots, K$ | facility index |

## 4.4 Stage 1: the stratified model

$$
\lambda_{ik}(t) = \lambda_{0k}(t) \exp(X_i\beta)
$$

Exactly Section 2.5's stratified Cox model, no modification. Fit it,
and keep the fitted linear predictor $X_i\hat\beta$ for every patient —
that number is each patient's entire risk profile, purged of any
facility-specific baseline effect, ready to carry into stage 2.

```python
import numpy as np
import pandas as pd
from pprof_py import CoxPH

X1 = cohort[["age", "sex", "diabetes", "comorbidity_count", "vintage_years"]]

stage1 = CoxPH(ties="breslow").fit(
    X1,
    duration=cohort["time"],
    event=cohort["death"],
    strata=cohort["facility_id"],
)
stage1.summary()

xbeta = stage1.predict_linear(X1)   # X_i * beta_hat, for every patient
```

`ties="breslow"` here is a deliberate choice, not an oversight — it
matches the historical SAS-based workflow this package was built to
reproduce exactly. If you are building a new SMR-style pipeline with no
existing Breslow-based baseline to match, `ties="efron"` is the more
accurate default from Section 2.4 and works identically throughout
this chapter.

## 4.5 Stage 2: the offset-only national model

$$
\lambda_i(t) = \lambda_0(t) \exp(X_{i0}\beta_0 + X_i\hat\beta)
$$

Two very different kinds of terms live in this formula, and it's worth
being precise about which is which, because they behave completely
differently in the fit:

- $X_i\hat\beta$ is stage 1's linear predictor, carried forward as a
  pure **offset** (Section 2.6) — its "coefficient" is permanently fixed
  at 1, never re-estimated. This is what makes stage 2's baseline
  consistent with stage 1's risk model rather than contradicting it.
- $X_{i0}\beta_0$ is an **optional additional covariate** whose
  coefficient $\beta_0$ *is* freely estimated in stage 2 — the CMS
  methodology uses this slot for a population death-rate adjustment
  (the log of the race-and-state-specific population death rate), but
  it is entirely optional. If you have no such adjustment, stage 2 has
  *no covariates at all* — its only job is estimating $\lambda_0(t)$,
  the shared national baseline, consistent with the offset.

`coxph` treats "zero additional covariates" as a first-class case, not
a workaround — pass a covariate table with no columns at all:

```python
stage2 = CoxPH(ties="breslow").fit(
    pd.DataFrame(index=cohort.index),   # no additional covariates -- a genuinely empty design
    duration=cohort["time"],
    event=cohort["death"],
    offset=xbeta,
)
```

If you *do* have a population-rate adjustment, add it as an ordinary
column to that DataFrame (e.g. `pd.DataFrame({"log_pop_rate": ...})`)
— everything below works identically either way, since it only ever
depends on `stage2.baseline_hazard_` and each patient's own fitted
partial hazard, both of which already account for whatever covariates
stage 2 does or doesn't have.

## 4.6 A gotcha worth knowing before you compute a single expected death

`stage2.baseline_hazard_` gives you $\hat\Lambda_0(t)$ — but not quite
at "everyone's offset is zero." To exactly match R's own
`basehaz(fit, centered=FALSE)` (which, despite its name, does **not**
give a literal zero-offset baseline — this is a genuine, undocumented
quirk of R's own function that `coxph` intentionally reproduces for
compatibility), the reported baseline is multiplied by
$\exp(\overline{\text{offset}})$, the exponentiated *mean* offset
across the whole dataset. If you skip correcting for this, every
patient's expected death count comes out inflated by that same
constant factor — in this chapter's own worked example, by roughly
22-fold, which is very much not a subtle rounding error.

The fix is one line: divide the baseline cumulative hazard by
$\exp(\overline{\text{offset}})$ before using it.

```python
offset_mean = np.mean(xbeta)   # replace with a weighted mean if you used sample_weight in stage 2
```

## 4.7 Computing expected deaths and the SMR

For patient $i$, the expected number of deaths by their own observed
time $t_i$ is

$$
E_i = -\ln \hat{S}_i(t_i) = \exp(A_i)\left(-\ln\hat{S}_0(t_i)\right) = \exp(A_i)\,\hat\Lambda_0(t_i)
$$

where $A_i = X_i\hat\beta$ (plus $X_{i0}\hat\beta_0$ if stage 2 has that
extra term) is patient $i$'s own fitted linear predictor, and
$\hat\Lambda_0(t_i)$ is the (corrected) national baseline cumulative
hazard evaluated *at that patient's own observed time* — not at any
fixed horizon, at exactly how long they were actually followed. A
patient followed for 3 years accumulates 3 years' worth of expected
risk; a patient followed for 3 months accumulates 3 months' worth,
exactly mirroring how much *chance to die* they actually had.

```python
def cumulative_hazard_at_own_time(baseline_hazard_df, own_times):
    """Step-function lookup: the baseline cumulative hazard in effect
    at each subject's own observed time (0 before the first event
    time anyone in the data experienced)."""
    times = baseline_hazard_df["time"].to_numpy()
    cumhaz = baseline_hazard_df["hazard"].to_numpy()
    idx = np.searchsorted(times, own_times, side="right") - 1
    return np.where(idx >= 0, cumhaz[np.clip(idx, 0, len(cumhaz) - 1)], 0.0)

partial_hazard = stage2.predict_partial_hazard(pd.DataFrame(index=cohort.index), offset=xbeta)
baseline_at_own_time = cumulative_hazard_at_own_time(stage2.baseline_hazard_, cohort["time"].to_numpy())
baseline_at_own_time_corrected = baseline_at_own_time / np.exp(offset_mean)

expected = partial_hazard * baseline_at_own_time_corrected

facility_table = (
    pd.DataFrame({"facility_id": cohort["facility_id"], "observed": cohort["death"], "expected": expected})
    .groupby("facility_id").sum()
)
facility_table["SMR"] = facility_table["observed"] / facility_table["expected"]
facility_table.head()
```

```
              observed   expected       SMR
facility_id
0                   11   7.579731  1.451239
1                    8   7.736351  1.034079
2                    7   4.249237  1.647355
3                    4   7.573212  0.528177
4                    9   6.273192  1.434676
```

A useful sanity check that has nothing to do with any one facility:
`facility_table["observed"].sum()` and `facility_table["expected"].sum()`
should be very close to each other across the *whole* dataset. This
isn't a coincidence — it's a direct consequence of how the partial
likelihood is maximized (Section 2.3), and it's exactly the "population
SMR equal to 1" constraint the underlying methods literature builds
around. If these two totals are wildly different, suspect the
offset-mean correction from Section 4.6 before anything else.

## 4.8 Is an SMR of 1.45 actually surprising?

Facility 0 shows 11 observed deaths against 7.58 expected — an SMR of
1.45. Facility 3 shows 4 against 7.57 — an SMR of 0.53. Neither
facility has very many patients, so before flagging either one as a
true outlier, we need to ask: *how much would this ratio bounce around
by pure chance alone, even at a facility performing exactly at the
national average?*

**Why treat this as a Poisson question.** Each individual patient's own
chance of dying is small, and different from every other patient's
(depending on their own age, diabetes status, and so on) — but when you
add up many small, mostly-independent probabilities like this, the
*total count* of events behaves like a Poisson random variable, with a
mean equal to the sum of everyone's individual small probabilities.
(This is the same reasoning that makes a Poisson distribution a good
model for the number of typos on a page or customers arriving at a
counter per hour — many small, independent chances, added up.) So:
**if a facility's true, underlying performance exactly matched the
national average**, the number of deaths it would produce, $O_k$,
behaves like a draw from a Poisson distribution with mean $E_k$ — the
expected count we already computed. That gives us a precise way to ask
"how surprising is this facility's observed count," with no new
modeling assumptions beyond the one we've already made.

### P-values: is $O_k$ surprising, given a Poisson($E_k$) null?

The direct approach doubles whichever tail is smaller:

```python
from scipy import stats

def poisson_exact_pvalue(observed, expected):
    """Two-sided exact Poisson test: how surprising is `observed`,
    if the true mean were exactly `expected`?"""
    if observed > expected:
        q = 2 * (1 - stats.poisson.cdf(observed - 1, expected))
    else:
        q = 2 * stats.poisson.cdf(observed, expected)
    return min(0.999, q)
```

This has a known wrinkle, worth knowing about even if you never hit it:
doubling a tail probability that's already above 0.5 can produce a
"p-value" above 1, which is nonsensical — hence the `min(0.999, ...)`
patch. A cleaner fix, standard in this literature, splits the exact
point probability in half instead of double-counting it (a **mid-p**
correction):

```python
def poisson_midp_pvalue(observed, expected):
    q = 0.5 * (stats.poisson.cdf(observed, expected) + stats.poisson.cdf(observed - 1, expected))
    return 2 * min(q, 1 - q)
```

The mid-p version never needs a hard cap and is generally the better-
calibrated of the two — prefer it for new work; the plain exact version
is here mainly to match an existing pipeline that expects it.

```python
facility_table["p_exact"] = [
    poisson_exact_pvalue(o, e) for o, e in zip(facility_table["observed"], facility_table["expected"])
]
facility_table["p_midp"] = [
    poisson_midp_pvalue(o, e) for o, e in zip(facility_table["observed"], facility_table["expected"])
]
```

### Confidence intervals on the SMR

A p-value answers "is this surprising," a confidence interval answers
the more useful "what range of true performance is consistent with
what we observed." Byar's approximation (accurate and simple even for
large counts) and an exact chi-square-quantile construction (more
exact for small counts) are both standard here:

```python
def smr_confidence_interval(observed, expected, alpha=0.05, byar_threshold=100):
    z = stats.norm.ppf(1 - alpha / 2)
    if expected >= byar_threshold:
        lower = (observed / expected) * (1 - 1 / (9 * observed) - z / (3 * np.sqrt(observed))) ** 3
        upper = ((observed + 1) / expected) * (1 - 1 / (9 * (observed + 1)) + z / (3 * np.sqrt(observed + 1))) ** 3
    else:
        # stats.chi2.ppf gives the count-scale interval; dividing by
        # `expected` converts it onto the same SMR (ratio) scale as Byar's.
        lower = 0.5 * stats.chi2.ppf(alpha / 2, 2 * observed) / expected
        upper = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / expected
    return lower, upper

facility_table[["ci_lower", "ci_upper"]] = [
    smr_confidence_interval(o, e) for o, e in zip(facility_table["observed"], facility_table["expected"])
]
facility_table.sort_values("SMR").round(3)
```

**Reading the final table**: a facility is a statistically credible
outlier only when its confidence interval excludes 1.00 entirely — not
merely when its point-estimate SMR is far from 1.00. A facility with 4
expected deaths and an SMR of 1.5 (say, 6 observed) has enormous
sampling noise around that ratio and usually should *not* be flagged
without a wide confidence interval to show for it; a facility with 200
expected deaths and the same SMR of 1.5 has a much tighter interval and
is a genuinely different, more actionable finding. This is precisely
why the CMS methodology this chapter is modeled on restricts published
comparisons to facilities above a minimum expected-death threshold —
below it, the ratio is simply too noisy to interpret on its own.

## 4.9 A note on reliability: how much of the variation is signal?

One more question the source methodology asks, worth knowing even
though it isn't a single formula to drop into code: of all the
facility-to-facility variation you see in SMR values, how much
reflects *real, persistent* differences in facility performance
("signal"), and how much is just the sampling noise Section 4.8 was
all about ("noise")? The relevant summary statistic is an **inter-unit
reliability (IUR)** — conceptually the fraction of total observed
variance attributable to genuine between-facility differences rather
than within-facility random fluctuation. A low IUR (as the source
methodology found for single-year SMRs) means most of what looks like
facility-to-facility variation is actually just noise, and pooling
multiple years of data — which shrinks the noise component relative to
the signal — meaningfully improves how trustworthy the comparison is,
exactly as the source methodology's own multi-year analysis
demonstrated.

## 4.10 A newer alternative: one-stage estimation

The two-stage approach in this chapter is not the only way to estimate
facility effects. A more recent line of methodology reformulates the
whole problem as a **single** proportional hazards model with an
explicit fixed effect $\alpha_k$ per facility:

$$
\lambda(t \mid Z_{ij}, G_{ij}=i) = \lambda_0(t)\exp(\alpha_i + Z_{ij}^T\beta)
$$

fit by an iterative algorithm that alternates between updating $\beta$
(by an ordinary partial-likelihood step, given the current facility
effects) and updating each $\alpha_i$ (by a fast fixed-point update,
subject to a constraint like $\sum_i O_i \exp(-\alpha_i) = \sum_i O_i$,
chosen so the *population* SMR still comes out to exactly 1). Once
this converges, $\text{SMR}_i = \exp(\hat\alpha_i)$ directly — no
separate stage-2 fit or baseline-hazard lookup needed at all.

This is a genuinely different, actively-researched estimation strategy,
not simply a faster way to compute the same two-stage answer (though
in practice, when the underlying stratified-Cox model is correctly
specified, the two approaches tend to produce very similar $\beta$
estimates). `coxph` does not currently provide a ready-made estimator
for this one-stage, fixed-facility-effects model — everything in this
chapter is built from the two-stage pattern the package's primitives
(`strata=`, `offset=`, `baseline_hazard_`) directly support. If you
need the one-stage approach specifically, the algorithm above (initialize
$\beta=0, \alpha=0$; alternate a Breslow-style baseline/$\beta$ update
with the constrained fixed-point $\alpha$ update; iterate to
convergence) is entirely implementable on top of `CoxPH`'s existing
score and baseline-hazard machinery, but it is not something this
package hands you out of the box today.

## 4.11 What's next

You've now built the complete classical SMR pipeline: a stratified
stage 1, an offset-consistent stage 2, expected deaths that correctly
account for each patient's own follow-up time, and calibrated Poisson
inference on top. Two things about this pipeline deserve closer
scrutiny before you'd trust it in production, and each gets its own
chapter: **Chapter 5** asks whether the plain standard errors from
Section 3.3 are still valid when a single patient can contribute
multiple records (as the "patient-facility-year" unit of analysis in
this very chapter's own notation implies), and **Chapter 6** asks what
changes when a covariate like dialysis vintage isn't fixed at study
entry but genuinely changes value over the course of follow-up.
