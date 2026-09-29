(inter-unit-reliability-guide)=
# Inter-Unit Reliability (IUR)

```{note}
The IUR estimators are not exported at the package root; import them
from `pprof_py.measures.iur`, e.g. `from pprof_py.measures.iur import BootstrapIUR`.
```

## The question this answers

[Chapter 4 §4.9](../survival/04_indirect_standardization_smr_shr)
poses it directly: of all the facility-to-facility variation you see
in a standardized measure, how much is **signal** (real, persistent
differences in performance) versus **noise** (sampling variation that
would look different if you drew the same facilities' patients again)?
**Inter-unit reliability (IUR)** is that fraction, on a 0–1 scale:

$$
\text{IUR} = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{between}} + \sigma^2_{\text{within}}}
$$

A facility-level measure with IUR near 1 is mostly telling you about
genuine facility differences; one near 0 is mostly telling you about
noise, and averaging more years of data (which shrinks the noise
component without touching the signal component) is the standard fix
— exactly the point Chapter 4 §4.9 makes without deriving the formula
behind it. Three estimators compute this decomposition three different
ways, plus a `ratio_measure` helper shared by two of them. The [IUR theory](iur_theory) page derives the
estimators, their sampling properties and what each one targets.

## The running example

The examples use the 60-facility mortality cohort of
[Penalized Logistic Regression](../logistic/penalized_logistic)
(`cohort`, `X`, `candidates`): each patient's observed death and the
probability expected from a covariate-only risk model, with no facility
term.

```python
import numpy as np
from pprof_py import PenalizedLogistic, LogisticFixedEffectModel

obs = cohort["death_30d"].to_numpy(dtype=float)
groups = cohort["facility_id"].to_numpy()
risk = PenalizedLogistic(alpha=1.0).fit(X, obs)                # covariates only: no facility term
exp = risk.predict_proba(X.to_numpy(), lambda_value=risk.lambda_path_[-1])   # expected probability per patient
print(f"{obs.sum():.0f} deaths, {exp.sum():.1f} expected, {np.unique(groups).size} facilities")
```
```
363 deaths, 363.0 expected, 60 facilities
```

## `ratio_measure`: the standard measure being decomposed

```python
from pprof_py.measures.iur import ratio_measure

print(ratio_measure(obs, exp, groups)[:5].round(3))   # sum(obs_g) / sum(exp_g) per facility
```
```
[1.02  1.055 1.377 0.748 0.929]
```

`ratio_measure` sorts by group internally, so the inputs need not be
sorted.

Any custom `measure_fn(obs, exp, groups) -> measures` with this same
signature can be substituted into any of the three estimators below —
`ratio_measure` (observed/expected — an SMR/SHR/SRR-style measure) is
just the built-in default.

## `BootstrapIUR`: from patient-level data

```python
from pprof_py.measures.iur import BootstrapIUR

model = BootstrapIUR(n_boot=200, seed=42)
model.fit(obs, exp, groups)
print(f"iur_ = {model.iur_:.3f}, s2_between_ = {model.s2_between_:.4f} (signal), "
      f"s2_within_ = {model.s2_within_:.2f} (noise, scaled by n_prime_)")
```
```
iur_ = -0.194, s2_between_ = -0.0243 (signal), s2_within_ = 14.94 (noise, scaled by n_prime_)
```

Stratified bootstrap: resample within each group, recompute
`measure_fn` on each replicate, and decompose the resulting
between/within variance via an ANOVA-style estimator (shared with
`DirectIUR`, in `_iur_decomposition`). A negative `iur_` — as shown
above, on a 60-facility mortality cohort — is a real, legitimate
possibility with this estimator, not a bug: the underlying
between-group variance estimate (`s2_b = s2_total - s2_within`) is
itself just a difference of two noisy quantities, and when a
measure's true between-facility signal is small relative to sampling
noise, that difference can land below zero. The honest reading of a
negative IUR is "consistent with little or no real signal here,"
not an error to suppress — this package does not clip the result into
$[0, 1]$, so check for this rather than assuming a small positive
number.

`decile_table()` reports the *expected* IUR at representative group
sizes (using the fitted variance components, not the raw data again)
— useful for answering "how large would a facility need to be before
this measure becomes reliable?" without refitting. `stratified_iur()`
recomputes the full decomposition within subgroups of facilities (by
size, by default) — worth treating with real caution at the far ends:
on the same cohort, the default size groups hold 4 to 8 facilities each
and give subgroup IUR values as extreme as $-3.4$, an artifact of estimating a
variance-of-a-variance from a handful of facilities per bucket, not a
sign the method is misbehaving on this particular data.

## `DirectIUR`: from group-level estimates alone

```python
from pprof_py.measures.iur import DirectIUR

fe = LogisticFixedEffectModel().fit(cohort, y_var="death_30d", x_vars=candidates, provider_var="facility_id")
model = DirectIUR()
model.fit(sizes=fe.provider_sizes_, estimates=fe.coefficients_["gamma"],
          standard_errors=np.sqrt(fe.variances_["gamma"]))
print(f"iur_ = {model.iur_:.3f}")
```
```
iur_ = -2.188
```

On the log-odds scale of $\hat\gamma_k$ the estimated between-facility
variance is again negative, and much larger in magnitude relative to the
within-facility variances (the facility effects' true SD is 0.2, while
their median standard error is 0.74), so this IUR falls well
below zero.

No patient-level data or resampling — if you already have per-facility
estimates and standard errors (`LogisticFixedEffectModel`'s
`variances_`/`robust_variances_`, for instance), this computes the
same decomposition directly from `se**2` as the within-group variance,
which is exact rather than simulated and considerably faster for large
facility counts. Same `decile_table()` method as `BootstrapIUR`.

## `SplitHalfIUR`: correlation-based, six ways at once

```python
from pprof_py.measures.iur import SplitHalfIUR

model = SplitHalfIUR(n_iter=20, seed=42)
model.fit(obs, exp, groups)
print(model.summary().round(4).to_string())
```
```
   iur_kappa  iur_kendall_cat  iur_spearman_cat  iur_pearson  iur_kendall  iur_spearman  n_groups
0    -0.1051          -0.1313           -0.1696      -0.1809      -0.1079       -0.1731        60
```

A different philosophy from the other two: repeatedly split each
group's observations into random halves, compute the measure on each
half independently, and correlate the two halves' results across
groups — six ways (weighted Cohen's kappa and Kendall/Spearman on
ordinal-binned measures; Pearson, Kendall, and Spearman on the
continuous measure directly), each converted to an IUR-scale number
via the Spearman-Brown formula, $\text{IUR} = 2r/(1+r)$. There is no
variance decomposition, but the values can still be negative: a
negative split-half correlation gives a negative IUR, as every variant
does on this cohort — a useful
cross-check against `BootstrapIUR`'s ANOVA-style estimate precisely
because it fails differently when the data doesn't cooperate, not
because one is more "correct" than the other in general.
