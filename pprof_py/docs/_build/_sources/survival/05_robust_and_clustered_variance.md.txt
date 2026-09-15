# Chapter 5 — Robust and Clustered Variance

## 5.1 A quiet assumption you've been relying on since Chapter 3

Every standard error, p-value, and confidence interval in Chapter 3's
`summary()` output rests on one assumption that has gone unstated so
far: that every *row* of data is an **independent** piece of evidence.
Chapter 4's own notation quietly broke this assumption without calling
attention to it — recall its definition of the unit of analysis:

> $i$: the patient-facility-year, patients might switch facilities and
> be followed multiple years, each patient-record represent one
> individual being followed under one facility for a year

If one patient contributes three yearly records to your dataset, those
three rows are not three independent pieces of evidence about how age
and diabetes affect mortality risk — they're three *correlated*
observations of the *same* underlying person, who has their own
unmeasured frailty, unmeasured social circumstances, and general
health trajectory that no covariate fully captures, and that shows up
in all three of their records alike. Treating them as fully independent
overstates how much distinct information you actually have, and the
practical consequence is standard errors that are **too small** —
narrower confidence intervals and smaller p-values than the data
actually supports, in a way that consistently favors finding "false"
significance.

This isn't unique to repeated patient-years. It's the same issue any
time your data has a natural grouping structure the model doesn't
already account for: patients within families, students within
classrooms, repeated lab measurements within the same patient,
pseudo-observations manufactured by a data transformation (Chapter 7's
competing-risks method does exactly this). Anywhere rows aren't
genuinely independent, plain standard errors need a correction.

## 5.2 The fix, intuitively: stop trusting the model's own math for uncertainty

The ordinary ("naive" or "model-based") standard errors from Chapter 3
come from a strong assumption: that the statistical model you fit is
*exactly* correct, including its implicit assumption that observations
are independent. **Robust (or "sandwich") variance** takes a different
strategy: instead of trusting the model's own internal math to tell you
how uncertain $\hat\beta$ is, it looks at how much each individual
observation's contribution to the fit *actually* varied, empirically,
across the real data, and builds an uncertainty estimate directly from
that — one that no longer depends on independence being exactly true.

The formal construction sandwiches the "naive" covariance matrix around
an empirical measure of how spread out the individual score
contributions are:

$$
\widehat{\text{Var}}_{\text{robust}}(\hat\beta) = A^{-1}\left(\sum_i U_i U_i^T\right)A^{-1}
$$

where $A^{-1}$ is the ordinary model-based covariance from Chapter 3,
and $U_i$ is patient $i$'s own **score residual** — informally, "how
much would the fitted model's conclusions have shifted if this one
patient's case weight had been nudged slightly," a precise measure of
that patient's individual influence on the result. This is a genuine
generalization, not a totally different idea: when observations really
are independent, the robust and naive variances converge to (nearly)
the same answer, so there's rarely much downside to using the robust
one by default once you know how to.

## 5.3 Clustering: pooling correlated observations before sandwiching

When you know *which* rows belong together — the three yearly records
for one patient, say — you can do better than treating every row as its
own island: **sum** each cluster's individual score residuals into one
combined contribution *before* building the sandwich, so that three
correlated records from one patient count as the single unit of
independent information they actually represent, rather than three:

$$
\widehat{\text{Var}}_{\text{clustered}}(\hat\beta) = A^{-1}\left(\sum_{c=1}^{C} \left(\sum_{i \in c} U_i\right)\left(\sum_{i \in c} U_i\right)^T\right)A^{-1}
$$

The number of *clusters* $C$, not the number of *rows*, is what
ultimately governs how tight this estimate can be — a dataset with
12,000 rows but only 4,000 true patients behaves, for inference
purposes, much more like a 4,000-observation study than a
12,000-observation one, and clustered variance is what makes that show
up correctly in your standard errors.

## 5.4 In `coxph`

Two related, composable options on `CoxPH`:

```python
from pprof_py import CoxPH

# Plain sandwich variance: every row treated as its own cluster.
# Appropriate when rows are genuinely independent people but you don't
# want to rely on the model being exactly correctly specified.
robust_model = CoxPH(robust=True).fit(X, duration=time, event=death)

# Clustered: pass the grouping variable directly. This IMPLIES
# robust=True regardless of the constructor setting -- an explicit
# cluster= is an unambiguous request.
clustered_model = CoxPH().fit(X, duration=time, event=death, cluster=patient_id)
```

Both leave `coef_` completely unchanged from a plain fit — clustering
and robustness are purely about *how much you should trust* the
estimate, never about what the estimate itself is:

```python
plain_model = CoxPH().fit(X, duration=time, event=death, cluster=patient_id)
print(plain_model.naive_covariance_.diagonal() ** 0.5)   # the Chapter 3 standard errors
print(plain_model.standard_errors_)                       # the clustered ones -- usually wider
```

`naive_covariance_` is always preserved alongside whichever one
`standard_errors_`/`summary()` actually reports, so you can compare
the two directly and see exactly how much the correction mattered for
your particular data.

## 5.5 Extending the running example: multi-year records

To make this concrete, extend the base cohort so a patient can
contribute more than one yearly record — exactly Chapter 4's own
"patient-facility-year" unit of analysis:

```python
import numpy as np
import pandas as pd

def make_multi_year_records(cohort, max_years=4, seed=1):
    """Split each patient's total follow-up time into up-to-`max_years`
    one-year (start, stop] records, each carrying the same facility and
    demographics -- the natural shape administrative claims data
    usually comes in."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, patient in cohort.iterrows():
        elapsed = 0.0
        year = 0
        while elapsed < patient["time"] and year < max_years:
            year_end = min(elapsed + 1.0, patient["time"])
            is_last = np.isclose(year_end, patient["time"])
            rows.append({
                **patient.to_dict(),
                "start": elapsed, "stop": year_end,
                "death": patient["death"] if is_last else 0,
                "record_year": year,
            })
            elapsed = year_end
            year += 1
    return pd.DataFrame(rows)

records = make_multi_year_records(cohort)
print(f"{len(cohort)} patients produced {len(records)} patient-year records")
```

Now compare plain, robust, and clustered standard errors on the exact
same fit:

```python
X = records[["age", "sex", "diabetes", "comorbidity_count", "vintage_years"]]

plain    = CoxPH(ties="efron").fit(X, start=records["start"], stop=records["stop"], event=records["death"])
robust   = CoxPH(ties="efron", robust=True).fit(X, start=records["start"], stop=records["stop"], event=records["death"])
clustered = CoxPH(ties="efron").fit(
    X, start=records["start"], stop=records["stop"], event=records["death"],
    cluster=records["patient_id"],
)

comparison = pd.DataFrame({
    "coef": plain.coef_,
    "se_plain": plain.standard_errors_,
    "se_robust": robust.standard_errors_,
    "se_clustered": clustered.standard_errors_,
}, index=plain.feature_names_in_)
comparison.round(4)
```

You should see `se_clustered` differ from `se_plain` — in this
particular dataset, by a modest few percent for most coefficients, even
though 82% of patients contribute two or more records. That's a useful
lesson in itself: the size of a clustering correction depends on how
much the *residual* correlation actually is within a cluster for that
specific coefficient, not simply on how many rows share a subject.
Don't expect a dramatic difference every time — expect a *real* one,
and always report the version that's actually appropriate for your
data's structure rather than assuming plain standard errors are "close
enough."

## 5.6 When to reach for this

- **Any time one subject contributes more than one row** — repeated
  measurements, patient-years, or (Chapter 7) the pseudo-observations a
  Fine-Gray transform manufactures — `cluster=` on the true subject
  identifier is close to mandatory, not optional, for a trustworthy
  standard error.
- **Even with one row per subject**, `robust=True` is a reasonable
  default whenever you're not fully confident the proportional-hazards
  model is exactly correctly specified — which, honestly, is most of
  the time in applied work. The cost of using it when it wasn't
  strictly needed is close to zero; the cost of *not* using it when it
  was needed is standard errors that are too small and confidence
  intervals that are too narrow, output you cannot see is wrong just by
  looking at it.

## 5.7 What's next

Chapter 5 assumed the multi-year records above were built correctly —
that each record's covariates genuinely reflected that patient during
that exact window of time. Chapter 6 tackles the harder question of
*how* to build that dataset correctly in the first place, when a
covariate like dialysis vintage, transplant status, or a facility
transfer genuinely changes value partway through someone's follow-up.
