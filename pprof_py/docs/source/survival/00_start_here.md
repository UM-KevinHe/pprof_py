(survival_start)=
# Survival Analysis in `pprof_py`: A Complete Guide
### From Survival Analysis Theory to Practice

---

## What this guide is

This is a from-scratch tutorial for the survival analysis capabilities of
`pprof_py` — a Python implementation of Cox proportional hazards regression
built to match R's `survival` package, extended with penalized regression,
variable selection, competing risks, robust variance, and time-dependent
covariates.

It is written in the spirit of the kind of internal methods note that
motivated this package in the first place: a document that states the
model precisely, in full notation, but also explains *why* the model
looks the way it does, in plain language, so that a reader without a
statistics PhD can follow the reasoning and a reader *with* one can
check every formula.

You do not need to be a statistician to read this. You do need to be
comfortable with:
- basic algebra and reading a formula like $y = a + bx$,
- the idea of a probability between 0 and 1,
- and enough Python to run a script and read a table of numbers.

Everywhere a formula appears, it is preceded or followed by a plain-
language version of the same idea. If you ever feel lost in an
equation, skip to the paragraph after it — the words are meant to
carry the meaning on their own.

## Why survival analysis exists in `pprof_py`

Most regression methods you meet first (linear regression, logistic
regression) assume you know the outcome for everyone in your data. Survival
analysis is for a different, extremely common situation: you're
tracking *time until something happens* — death, hospital readmission,
device failure, loan default, transplant — and for some people, that
something **hasn't happened yet** by the time you have to analyze the
data. Someone still alive at the end of the study isn't a "0" in the
same sense that a coin flip is a "0" — you know they survived *at
least* this long, but you don't know their eventual outcome. Throwing
that person out of the analysis wastes information and, worse, biases
the results (the people you'd be throwing out are disproportionately
the ones doing *well*). Survival analysis is the set of tools built to
use that partial information correctly.

The single motivating application behind this package is a specific,
high-stakes version of that problem: **profiling healthcare
facilities**. Given patients at hundreds or thousands of dialysis
facilities, each with their own mix of ages, comorbidities, and time
already spent on dialysis, how do you fairly say facility A's patients
are dying faster than expected, once you've accounted for the fact
that facility A simply has a sicker patient population than facility
B? That is the Standardized Mortality Ratio (SMR) problem, and it is
the subject of this guide's centerpiece chapter (Chapter 4). Everything
before it builds the machinery the SMR calculation is made from,
and everything after it extends that machinery to handle messier,
more realistic versions of the same data.

## The running example

Rather than a different toy dataset in every chapter, this guide uses
**one synthetic dataset** throughout, in the same spirit as the source
methods paper this guide's core chapter is modeled on. It is entirely
fabricated for teaching purposes — no real patient data of any kind —
but it is shaped like the real thing: patients with end-stage renal
disease (ESRD), followed at dialysis facilities, with the kinds of
covariates that actually drive mortality risk in that population.

```python
import numpy as np
import pandas as pd

def make_esrd_cohort(n_patients=4000, n_facilities=40, seed=0):
    """A synthetic ESRD cohort: one row per patient, used throughout
    this guide's early chapters. Later chapters extend this same
    generator to add time-varying covariates and competing risks.
    """
    rng = np.random.default_rng(seed)

    facility_id = rng.integers(0, n_facilities, n_patients)
    facility_quality = rng.normal(0, 0.25, n_facilities)

    age = rng.normal(62, 14, n_patients).clip(18, 95)
    sex = rng.binomial(1, 0.45, n_patients)
    diabetes = rng.binomial(1, 0.45, n_patients)
    comorbidity_count = rng.poisson(1.3, n_patients)
    vintage_years = rng.exponential(2.0, n_patients)

    log_hazard = (
        -4.2
        + 0.045 * (age - 62)
        + 0.35 * diabetes
        + 0.20 * comorbidity_count
        + facility_quality[facility_id]
    )
    hazard = np.exp(log_hazard)

    death_time = rng.exponential(1.0 / hazard)
    censor_time = rng.uniform(0.5, 4.0, n_patients)
    time = np.minimum(death_time, censor_time)
    death = (death_time <= censor_time).astype(int)

    return pd.DataFrame({
        "patient_id": np.arange(n_patients),
        "facility_id": facility_id,
        "age": age.round(1),
        "sex": sex,
        "diabetes": diabetes,
        "comorbidity_count": comorbidity_count,
        "vintage_years": vintage_years.round(2),
        "time": time.round(3),
        "death": death,
    })

cohort = make_esrd_cohort()
cohort.head()
```

Every later chapter either reuses `cohort` directly or extends
`make_esrd_cohort` with one more piece of realism (a competing risk of
transplant, a covariate that changes value over time, and so on) —
each extension is introduced exactly in the chapter that needs it, so
you never have to look far to find where a column came from.

## How the chapters build on each other

| # | Chapter | What it covers |
|---|---|---|
| 1 | Survival Data Foundations | What survival data *is*, why ordinary regression fails on it, censoring, truncation, the hazard and survival functions |
| 2 | The Cox Model | The Cox proportional hazards model: the proportional-hazards idea, the partial likelihood, risk sets, tied event times, stratification, offsets |
| 3 | Fitting Your First Model | Hands-on: fitting a model with `CoxPH`, reading every number in the output, hazard ratios, confidence intervals, predictions |
| 4 | Indirect Standardization (SMR/SHR) | The two-stage Standardized Mortality Ratio (SMR) / Standardized Hospitalization Ratio (SHR) method: theory, notation, inference, and a full from-scratch implementation |
| 5 | Robust and Clustered Variance | Why plain standard errors can be wrong when observations aren't independent, and the sandwich/cluster-robust fix |
| 6 | Time-Dependent Covariates | Covariates that change over time, immortal time bias, and building the right dataset shape |
| 7 | Competing Risks | What happens when there's more than one way for the "event" to occur (death vs. transplant), cause-specific hazards vs. the Fine-Gray model |
| 8 | Penalized Regression | Ridge, LASSO, and elastic net for Cox models: what regularization is for, and cross-validated tuning |
| 9 | Variable Selection | Automated stepwise variable selection as an alternative (or complement) to penalization |
| 10 | Complete Case Study | Putting it all together: a single, comprehensive worked analysis from raw data to a final facility profiling report |

If you only read one chapter beyond the basics, make it Chapter 4 — it
is both the historical reason this package exists and the clearest
illustration of why every other chapter's machinery (stratification,
offsets, robust variance) matters in practice, not just in the abstract.

## Installing and importing

```bash
pip install -e .        # from the root of the pprof_py package
```

```python
from pprof_py import CoxPH, PenalizedCoxPH, PenalizedCoxPHCV
from pprof_py import CauseSpecificCoxPH, FineGrayPH
from pprof_py import CoxPHSelector
import pprof_py
print(pprof_py.__version__)
```

Every code example in this guide assumes `cohort` (defined above) and the
shared variables below are in memory. Chapters are otherwise independent,
with three deliberate exceptions, each flagged where it occurs: Chapter 9 reuses
`X_wide` from Chapter 8, Chapter 10 reuses `make_multi_year_records` from
Chapter 5, and Chapters 6 and 7 build the extra columns they need themselves.

```python
# Shared variables assumed by the short snippets in Chapters 2-9
X = cohort[["age", "sex", "diabetes", "comorbidity_count"]]
time = cohort["time"]
death = cohort["death"]
patient_id = cohort["patient_id"]
```

This tutorial teaches `CoxPH`, penalized Cox, competing risks and selection.
The remaining estimators (group lasso, provider-penalized Cox, discrete-time
survival), the full argument and attribute lists, the data-preparation helpers
and the validation tools are covered in the **Survival Analysis — Reference**
pages (start with `survival/reference/coxph`).

Onward to Chapter 1.
