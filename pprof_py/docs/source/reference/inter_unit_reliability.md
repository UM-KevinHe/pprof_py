(inter-unit-reliability-guide)=
# Inter-Unit Reliability (IUR)

```{note}
This page isn't part of the original documentation task's six-item
list for this deliverable — `measures.iur` isn't named in any
deliverable's class list, and none of its four public names
(`BootstrapIUR`, `SplitHalfIUR`, `DirectIUR`, `ratio_measure`) appear
in `pprof_py.__all__`. It's added here because the package's own
repository-structure notes list "SMR, SHR, IUR" together as the scope
of the `measures/` module, and
[survival Chapter 4 §4.9](../survival/04_indirect_standardization_smr_shr)
already introduces the *concept* of inter-unit reliability without a
worked implementation — this page is that missing implementation.
Since it isn't re-exported at the package root, import it from
`pprof_py.measures.iur` explicitly, e.g.
`from pprof_py.measures.iur import BootstrapIUR`.
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
ways, plus a `ratio_measure` helper shared by two of them.

## `ratio_measure`: the standard measure being decomposed

```python
from pprof_py.measures.iur import ratio_measure
import numpy as np

# sort by group FIRST -- see the note below
sort_idx = np.argsort(groups, kind="stable")
obs_s, exp_s, groups_s = obs[sort_idx], exp[sort_idx], groups[sort_idx]
ratio_measure(obs_s, exp_s, groups_s)   # sum(obs_g) / sum(exp_g) per group
```

`ratio_measure` now sorts by group internally, so callers no longer
need to pre-sort.  The explicit sort above is harmless but not
required.

Any custom `measure_fn(obs, exp, groups) -> measures` with this same
signature can be substituted into any of the three estimators below —
`ratio_measure` (observed/expected — an SMR/SHR/SRR-style measure) is
just the built-in default.

## `BootstrapIUR`: from patient-level data

```python
from pprof_py.measures.iur import BootstrapIUR

model = BootstrapIUR(n_boot=200, seed=42)
model.fit(obs, exp, groups)   # sorts internally -- raw obs/exp/groups are fine

model.iur_            # -0.189
model.s2_between_     # -0.0237 (signal)
model.s2_within_      # 14.93   (noise, scaled by n_prime_)
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
on the same cohort, splitting 60 facilities into size deciles produced
subgroup IUR values as extreme as $-3.5$, an artifact of estimating a
variance-of-a-variance from a handful of facilities per bucket, not a
sign the method is misbehaving on this particular data.

## `DirectIUR`: from group-level estimates alone

```python
from pprof_py.measures.iur import DirectIUR

model = DirectIUR()
model.fit(sizes=group_sizes, estimates=group_estimates, standard_errors=group_ses)
model.iur_
```

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
model.summary()
```
```
   iur_kappa  iur_kendall_cat  iur_spearman_cat  iur_pearson  iur_kendall  iur_spearman  n_groups
0    -0.1048          -0.1266             ...          ...          ...        -0.185        60
```

A different philosophy from the other two: repeatedly split each
group's observations into random halves, compute the measure on each
half independently, and correlate the two halves' results across
groups — six ways (weighted Cohen's kappa and Kendall/Spearman on
ordinal-binned measures; Pearson, Kendall, and Spearman on the
continuous measure directly), each converted to an IUR-scale number
via the Spearman-Brown formula, $\text{IUR} = 2r/(1+r)$. No variance
decomposition, no negative values by construction (correlations are
bounded, so Spearman-Brown-transformed values are too) — a useful
cross-check against `BootstrapIUR`'s ANOVA-style estimate precisely
because it fails differently when the data doesn't cooperate, not
because one is more "correct" than the other in general.
