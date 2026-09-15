# Chapter 10 — Complete Case Study: A Facility Profiling Report, Start to Finish

Every earlier chapter introduced one idea in isolation. Real analyses
need several of them at once. This chapter builds one complete,
defensible facility profiling report — the exact deliverable Chapter
4's SMR method exists to produce — using data shaped the way it
actually arrives: multiple yearly records per patient, a candidate list
of adjustment covariates wider than you're sure you need, and a
patient population, not a set of independent rows.

## 10.1 Start from patient-year records, not one row per patient

Real ESRD claims data arrives as one record per patient per year
(Chapter 4's own notation said so explicitly), which immediately means
one patient can appear more than once — precisely Chapter 5's setup:

```python
import numpy as np
import pandas as pd
from scipy import stats
from pprof_py import CoxPH, CoxPHSelector

# `cohort` and `make_multi_year_records` are exactly Chapter 0 and Chapter 5
records = make_multi_year_records(cohort)
print(f"{len(cohort)} patients produced {len(records)} patient-year records")
```

```
4000 patients produced 10558 patient-year records
```

Every downstream step in this chapter has to account for this — a
patient contributing three rows is one person's evidence, not three,
exactly Chapter 5's whole point, and it resurfaces at every stage
below, not just at the end.

## 10.2 Choose the adjustment covariates deliberately

Age is going into this model regardless of what the data says — it's
an established, non-negotiable risk factor in this literature, and
Chapter 9 showed exactly how to say that formally with `forced=`.
Beyond that, let the data help decide, using BIC's more conservative
standard given how central this adjustment set is to the entire
report's credibility:

```python
candidates = ["age", "sex", "diabetes", "comorbidity_count", "vintage_years"]

selector = CoxPHSelector(direction="forward", criterion="bic").fit(
    records[candidates],
    start=records["start"], stop=records["stop"], event=records["death"],
    forced=["age"],
)
adjustment_vars = selector.selected_variables_
print(adjustment_vars)
```

```
['age', 'comorbidity_count']
```

## 10.3 Stage 1: stratified *and* clustered

This is the step where Chapters 4 and 5 have to be combined explicitly,
not just used side by side: stage 1 is stratified by facility (Chapter
4 — so each facility gets its own baseline hazard shape) **and**
clustered by patient (Chapter 5 — so a patient's several yearly records
correctly count as one person's evidence, not several). Both apply to
the same fit, simultaneously, with no conflict between them —
stratification changes which risk sets a patient's records compete
within; clustering changes how much the resulting standard errors
trust having several records from the same patient.

```python
X1 = records[adjustment_vars]

stage1 = CoxPH(ties="efron").fit(
    X1,
    start=records["start"], stop=records["stop"], event=records["death"],
    strata=records["facility_id"],
    cluster=records["patient_id"],
)
stage1.summary()

xbeta = stage1.predict_linear(X1)
```

```
                       coef  exp(coef)  se(coef)  ...
age                0.047316   1.048454    0.0047
comorbidity_count  0.179186   1.196243    0.0479
```

## 10.4 Stage 2: offset-only, unstratified, still clustered

```python
stage2 = CoxPH(ties="efron").fit(
    pd.DataFrame(index=records.index),
    start=records["start"], stop=records["stop"], event=records["death"],
    offset=xbeta,
    cluster=records["patient_id"],
)
offset_mean = np.mean(xbeta)
```

## 10.5 Expected deaths — the one genuinely new wrinkle

Chapter 4 computed a patient's expected deaths as $\exp(A_i)\,
\hat\Lambda_0(t_i)$ — the baseline cumulative hazard from time zero up
to that patient's own stop time. That formula silently assumed one row
per patient, covering their *entire* follow-up from the start. With
multi-year records, it needs a small but essential correction: **each
record only covers the exposure between its own `start` and `stop`**,
so its contribution to expected deaths must be the *difference* in
cumulative hazard across just that window — not the whole cumulative
hazard up to its `stop`, which would silently double-count whatever
exposure an earlier record for the same patient already accounted for.

```python
def cumulative_hazard_at_own_time(baseline_hazard_df, own_times):
    times = baseline_hazard_df["time"].to_numpy()
    cumhaz = baseline_hazard_df["hazard"].to_numpy()
    idx = np.searchsorted(times, own_times, side="right") - 1
    return np.where(idx >= 0, cumhaz[np.clip(idx, 0, len(cumhaz) - 1)], 0.0)

baseline = stage2.baseline_hazard_
cumhaz_at_stop  = cumulative_hazard_at_own_time(baseline, records["stop"].to_numpy())  / np.exp(offset_mean)
cumhaz_at_start = cumulative_hazard_at_own_time(baseline, records["start"].to_numpy()) / np.exp(offset_mean)

partial_hazard = stage2.predict_partial_hazard(pd.DataFrame(index=records.index), offset=xbeta)
expected = partial_hazard * (cumhaz_at_stop - cumhaz_at_start)

print("total observed:", records["death"].sum())
print("total expected:", expected.sum())
```

```
total observed: 259.0
total expected: 259.01268907718054
```

The near-exact match is the same sanity check Chapter 4 introduced,
still holding after this correction — a strong sign the per-record
exposure windowing was done right. If you skip the `- cumhaz_at_start`
term here, this check will fail visibly (total expected will land far
above total observed, for exactly the double-counting reason above),
which makes it a genuinely useful, self-diagnosing guardrail to keep in
any pipeline built this way, not just decoration for this tutorial.

## 10.6 The final report

Reusing Chapter 4's inference functions unchanged:

```python
def poisson_midp_pvalue(observed, expected):
    q = 0.5 * (stats.poisson.cdf(observed, expected) + stats.poisson.cdf(observed - 1, expected))
    return 2 * min(q, 1 - q)

def smr_confidence_interval(observed, expected, alpha=0.05, byar_threshold=100):
    z = stats.norm.ppf(1 - alpha / 2)
    if expected >= byar_threshold:
        lower = (observed / expected) * (1 - 1/(9*observed) - z/(3*np.sqrt(observed))) ** 3
        upper = ((observed + 1) / expected) * (1 - 1/(9*(observed+1)) + z/(3*np.sqrt(observed+1))) ** 3
    else:
        lower = 0.5 * stats.chi2.ppf(alpha / 2, 2 * observed) / expected
        upper = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / expected
    return lower, upper

report = (
    pd.DataFrame({"facility_id": records["facility_id"], "observed": records["death"], "expected": expected})
    .groupby("facility_id").sum()
)
report["SMR"] = report["observed"] / report["expected"]
report["p_value"] = [poisson_midp_pvalue(o, e) for o, e in zip(report["observed"], report["expected"])]
report[["ci_lower", "ci_upper"]] = [
    smr_confidence_interval(o, e) for o, e in zip(report["observed"], report["expected"])
]
report["flag"] = np.where(
    report["ci_lower"] > 1.0, "higher than expected",
    np.where(report["ci_upper"] < 1.0, "lower than expected", ""),
)

report.sort_values("SMR", ascending=False).round(3).head(10)
```

```
             observed  expected    SMR  p_value  ci_lower  ci_upper                  flag
facility_id
22.0             13.0     7.385  1.760    0.048     1.021     2.858  higher than expected
21.0              9.0     5.264  1.710    0.108     0.895     3.084
2.0               7.0     4.316  1.622    0.201     0.751     3.181
23.0             12.0     7.625  1.574    0.107     0.907     2.612
10.0              9.0     5.856  1.537    0.180     0.804     2.762
```

Facility 22 is the one entry on this list whose interval genuinely
excludes 1.00 — a statistically credible outlier, not merely the
facility with the single highest point estimate (facility 22's SMR of
1.76 is not even the largest shown here in absolute terms — it's the
one where the *combination* of the ratio and its own precision clears
the bar, exactly the distinction Chapter 4 asked you to make rather
than sorting by SMR alone and stopping there).

## 10.7 Where the other chapters extend this

This report is already complete and defensible, but every remaining
chapter's technique layers onto this exact skeleton without disturbing
it, if your own data calls for it:

- If a covariate like transplant waitlisting genuinely changes value
  mid-follow-up, build it into `records` with `tmerge` (**Chapter 6**)
  before it ever reaches `stage1` — nothing about Sections 10.3–10.6
  changes once the (start, stop] rows are shaped correctly.
- If patients can be removed from risk by a competing event like
  transplant, decide whether your actual question is about the
  cause-specific rate or the true cumulative incidence (**Chapter 7**),
  and swap in `CauseSpecificCoxPH` or `FineGrayPH` for the plain
  `CoxPH` calls in Sections 10.3–10.4 accordingly.
- If your adjustment-covariate candidate list is much wider than five
  variables, reach for `PenalizedCoxPHCV` (**Chapter 8**) to narrow it
  down before stepwise selection ever runs, exactly as Chapter 9's
  closing section recommended.

## 10.8 Closing

You've now built the same kind of facility profiling pipeline this
package was originally written to reproduce, from raw patient-year
records to a final, statistically calibrated report — with an honest
accounting, at every step, of censoring, truncation, clustering, and
(where your own data needs it) time-varying covariates and competing
risks. That is the complete arc this guide set out to cover, from
Chapter 1's first question — *what do you do with a patient who hasn't
had the event yet* — to a defensible answer to *is this facility
actually performing differently from what we'd expect*.
