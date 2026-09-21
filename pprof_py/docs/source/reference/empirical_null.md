(empirical-null-guide)=
# Empirical Null Calibration

`huber_location_scale` and `estimate_empirical_null` are the two
public names behind a step every `.test()` method in this
documentation quietly relies on: `LogisticFixedEffectModel.test()`,
`LogisticMixedEffectModel.test()`
([Chapter 5](../logistic/logistic_mixed_effect_model)), and the
provider-effect testing this documentation has referenced throughout
all use the same calibration underneath. This page documents it on
its own terms.

## The problem: your null distribution is rarely exactly what theory says

Testing whether a provider's effect $\gamma_k$ differs from some null
value (the median, typically) produces a p-value per provider under a
*theoretical* null — Poisson-binomial exact, or a Gaussian bootstrap
approximation. In practice, across hundreds of providers, the
*empirical* distribution of those theoretical z-scores is often not
quite standard normal — correlated risk-adjustment error, mild
model misspecification, and unmodeled clustering all nudge it wider
(or occasionally narrower) than theory assumes. **Empirical null
calibration** re-centers and re-scales the observed z-scores using a
robust estimator of their own location and spread, then recomputes
p-values against *that* — a purely data-driven correction layered on
top of the theoretical test, not a replacement for it.

## `huber_location_scale`: robust location and scale

```python
from pprof_py import huber_location_scale
import numpy as np

rng = np.random.default_rng(0)
z = np.concatenate([
    rng.normal(0, 1, 90),     # 90 null providers
    rng.normal(4, 1, 6),      # 6 true high outliers
    rng.normal(-4, 1, 4),     # 4 true low outliers
])
loc, scale = huber_location_scale(z)
```
```
location: 0.1572   scale: 1.1873
(naive mean/std for comparison: 0.1611 / 1.7031)
```

An IRLS-based Huber M-estimator, matching R's `MASS::rlm(z ~ 1,
method='M', psi=psi.huber, scale.est='MAD')` down to its literal
`0.6745` MAD constant — the docstring documents this correspondence
explicitly, including the one non-obvious detail worth knowing before
comparing against a hand-rolled version: R's MAD is recomputed every
IRLS iteration, centered on the *current* location estimate, not on
the residuals' own median. On the simulated z-scores above, the true
10 outliers inflate the naive standard deviation (`1.70`) well past
the null group's real spread; the Huber scale (`1.19`) stays close to
it, since `k=1.345` (the default tuning constant) downweights any
observation more than about 1.3 robust-scale-units from the current
center — exactly the point of using it instead of a plain mean/SD.

## `estimate_empirical_null`: one fit per size stratum

Provider size correlates with estimate precision, so a single global
empirical null can under-correct small providers and over-correct
large ones. `estimate_empirical_null` wraps `huber_location_scale`
with the stratification `test()` methods across this package actually
use:

```python
from pprof_py import estimate_empirical_null

sizes = rng.integers(20, 300, 100)   # provider volumes
location, scale = estimate_empirical_null(z, group_sizes=sizes, n_groups=4)
```

Returns one `(location, scale)` pair *per provider* (length matches
`z`, not `n_groups`) — every provider gets its stratum's fitted
values, broadcast back out, ready to feed directly into a p-value
recomputation without a second lookup step. `n_groups=1` (or omitting
`group_sizes` entirely) collapses to a single global fit; pass
`group_labels=` directly if you already have a stratification variable
in mind other than raw provider size. Providers falling in a group
with fewer than 3 valid z-scores fall back to `location=0, scale=1`
(no calibration applied) rather than fitting an unstable Huber
estimate on almost no data.

## What's underneath, if you need it

`test()` methods in this package build calibrated p-values from these
two functions plus a small set of internal helpers not exported at the
package root — `resample_pvalue` (He et al. 2013 parametric bootstrap,
incorporating posterior uncertainty of any random effects in the
model), `poibin_exact_pvalue` (deterministic exact Poisson-binomial,
via the `fast_poibin` package), `pvalues_to_zscores`,
`calibrate_empirical_null`, and `assign_flags`. These live in
`pprof_py.inference.empirical_null` alongside the two public
functions, and are worth knowing about if you're reading a `test()`
method's source rather than calling it — but `huber_location_scale`
and `estimate_empirical_null` are the two names this package considers
public API (`pprof_py.__all__`), and are the two documented here.
