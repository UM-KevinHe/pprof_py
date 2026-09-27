(empirical-null-guide)=
# Empirical Null Calibration

Every provider test in `pprof_py` is built in three steps: a z-statistic
for each provider, a **null model** that says how those statistics are
distributed when providers perform as expected, and the decisions that
follow (p-values, flags and intervals). This page is about the middle
step. By default the null is the theoretical N(0, 1); an *empirical*
null estimates its location and scale from the z-statistics themselves.
Everything here lives in `pprof_py.inference`.

## The problem: your null distribution is rarely exactly what theory says

Testing whether a provider's effect $\gamma_k$ differs from a reference
value (the median, typically) produces a z-statistic per provider under
a *theoretical* null — exact Poisson-binomial, score, Wald or
resampling. In practice, across hundreds of providers, the *empirical*
distribution of those z-statistics is often not quite standard normal:
correlated risk-adjustment error, mild model misspecification and
unmodeled clustering all nudge it wider (or occasionally narrower) than
theory assumes. **Empirical null calibration** re-centers and re-scales
the z-statistics using a robust estimate of their own location and
spread, then recomputes p-values, flags and intervals against *that* —
a data-driven correction layered on top of the theoretical test, not a
replacement for it.

## Three null models

| Null model | Location and scale | Typical use |
|---|---|---|
| `TheoreticalNull()` | 0 and 1 | The default in every `test()` and `test_standardized()`. |
| `FixedNull(mean=0.0, sd=1.0)` | chosen in advance | A known overdispersion factor. Earlier versions of `test_standardized` divided every z-statistic by 1.81 by default; `FixedNull(sd=1.81)` reproduces that. |
| `EmpiricalNull.fit(z, ...)` | estimated from `z` | Overdispersion estimated from the providers themselves, overall or within provider-size groups. |

Every test takes one through `null_model=`. A null model is used as
given; a callable, such as `EmpiricalNull.fitter(...)`, is fitted on the
test's own z-statistics (always on all providers, even when
`providers=` restricts the rows reported):

```python
from pprof_py.inference import EmpiricalNull, FixedNull

model.test(null_model=FixedNull(sd=1.81))
model.test(null_model=EmpiricalNull.fitter(size=sizes, n_groups=4))
model.test_standardized(measure="indirect_ratio", null_model=EmpiricalNull.fitter())
```

## Robust location and scale

```python
import numpy as np
from pprof_py.inference import robust_location_scale, DEFAULT_ESTIMATOR

rng = np.random.default_rng(0)
z = np.concatenate([
    rng.normal(0, 1, 90),     # 90 providers performing as expected
    rng.normal(4, 1, 6),      # 6 true high outliers
    rng.normal(-4, 1, 4),     # 4 true low outliers
])
huber = robust_location_scale(z)   # MASS::rlm(z ~ 1) defaults: Huber psi, 20 iterations
bisq = DEFAULT_ESTIMATOR(z)        # the empirical-null default: bisquare, 1000 iterations
print(f"Huber     location {huber.location:.4f}   scale {huber.scale:.4f}")
print(f"bisquare  location {bisq.location:.4f}   scale {bisq.scale:.4f}")
print(f"mean/SD   location {z.mean():.4f}   scale {z.std(ddof=1):.4f}")
```
```text
Huber     location 0.1572   scale 1.1873
bisquare  location 0.1642   scale 1.1975
mean/SD   location 0.1611   scale 1.7031
```

`robust_location_scale` reproduces R's `MASS::rlm(z ~ 1)`: Huber or
bisquare `psi`, `method="M"` or `"MM"`, a least-squares start, and the
MAD scale recomputed every iteration around the *current* location
(with MASS's `0.6745` constant) rather than around the residuals' own
median. Its defaults are `rlm`'s (Huber, `tuning=1.345`, 20 iterations,
`tol=1e-4`). On the z-statistics above, the ten true outliers inflate
the naive standard deviation well past the null group's real spread;
both robust scales stay close to 1.

`MEstimator` bundles a set of options into a reusable estimator.
`HUBER_RLM`, `BISQUARE_RLM` and `MM_RLM` are presets with `rlm`'s
iteration defaults, and `DEFAULT_ESTIMATOR` — bisquare, 1000
iterations, `tol=1e-8`, the default of the EmpiNull R package — is what
`EmpiricalNull` uses unless told otherwise.

## Fitting an empirical null

Provider size correlates with estimate precision, so a single overall
null can under-correct small providers and over-correct large ones.
`EmpiricalNull.fit` fits one null per size group:

```python
from pprof_py.inference import EmpiricalNull

sizes = rng.integers(20, 300, z.size)          # provider volumes
null = EmpiricalNull.fit(z, size=sizes, n_groups=4)
print(null.diagnostics[["group", "n_fitted", "null_mean", "null_sd", "converged", "fallback"]].round(4))
```
```text
   group  n_fitted  null_mean  null_sd  converged  fallback
0      1        26     0.2534   1.1763       True     False
1      2        24     0.3440   1.4354       True     False
2      3        25     0.2881   1.0742       True     False
3      4        25    -0.2779   1.2138       True     False
```

`null.mean` and `null.sd` hold each provider's location and scale
(length of `z`), `null.group` its group, and `null.diagnostics` one row
per group. The options:

- **Groups.** `size=` with `grouping="quantile"` (the default) splits at
  the quantiles of `size`, and a size equal to a break joins the lower
  group (EmpiNull's rule). `grouping="rank"` forms equal-count groups by
  rank; `order=` breaks ties deterministically (otherwise input order
  decides). `groups=` supplies labels directly. With neither, one
  overall null is fitted.
- **Estimator.** `estimator=` takes any `MEstimator` (for example
  `HUBER_RLM`) or a callable returning a location and scale.
- **Small groups.** Each group needs `min_group_size=3` usable
  z-statistics. By default (`small_group="error"`) a smaller group
  raises; `small_group="theoretical"` uses N(0, 1) for it and emits an
  `EmpiricalNullWarning`, and the diagnostics mark the fallback:

```python
small = EmpiricalNull.fit(z[:10], size=sizes[:10], n_groups=4, small_group="theoretical")
print(small.diagnostics[["group", "n_fitted", "null_mean", "null_sd", "fallback"]].round(4))
```
```text
   group  n_fitted  null_mean  null_sd  fallback
0      1         3     0.7606   0.5916     False
1      2         2     0.0000   1.0000      True
2      3         2     0.0000   1.0000      True
3      4         3    -0.0744   0.2967     False
```

- **Pooled location.** `common_mean=True` uses the mean of all usable
  z-statistics as every group's location while keeping the group
  scales; a number fixes the location instead.
- **Fitting subset.** `fit_mask=` fits the null on some providers (for
  example, above a volume cutoff) while still calibrating all of them.

Providers with a missing or infinite z-statistic are left out of the fit
and are reported as not tested (`flag` is `NA`).

## Using a null model in a test

```python
from pprof_py import LogisticFixedEffectModel

m = 80
n_j = rng.integers(60, 200, m)                          # patients per provider
provider = np.repeat(np.arange(m), n_j)
x = rng.normal(size=(provider.size, 2))
extra = rng.normal(0, 0.35, m)                          # provider variation beyond the model's null
eta = -1.5 + x @ [0.5, -0.3] + extra[provider]
y = (rng.random(provider.size) < 1 / (1 + np.exp(-eta))).astype(float)
model = LogisticFixedEffectModel().fit(x, y, provider)

theoretical = model.test()                                          # exact test, N(0, 1) null
empirical = model.test(null_model=EmpiricalNull.fitter(size=n_j, n_groups=4))
cols = ["z_raw", "null_group", "null_mean", "null_sd", "z_adjusted", "p_value", "flag"]
print(empirical[cols].head().round(3))
print("flagged:", int((theoretical.flag != 0).sum()), "with the theoretical null,",
      int((empirical.flag != 0).sum()), "with the empirical null")
```
```text
          z_raw  null_group  null_mean  null_sd  z_adjusted  p_value  flag
provider
0         0.161           1     -0.238    1.164       0.343    0.732     0
1        -2.311           1     -0.238    1.164      -1.781    0.075     0
2         1.986           1     -0.238    1.164       1.910    0.056     0
3         4.109           3      0.507    2.710       1.329    0.184     0
4         1.603           2      0.017    0.781       2.031    0.042     1
flagged: 26 with the theoretical null, 10 with the empirical null
```

`null_mean`, `null_sd` and `null_group` report the null each provider
was calibrated against, and `z_adjusted` is
`(z_raw - null_mean) / null_sd`, from which the p-value and flag follow.
`size=` must follow the provider order of the result (here the sorted
provider labels). Under an empirical null, intervals are shifted and
scaled with the null (`interval="inversion"`, the default), so an
interval excludes the null value exactly when the provider is flagged;
`interval="scale_only"` widens them without shifting. The result's
`attrs["null_model"]` records the fitted null.

R's `summary.glmm.fac` (the reference for `LogisticFERandomClusterModel.test`)
fits `MASS::rlm` with its defaults within quartiles of a facility-size
variable, setting missing sizes to 0:

```python
from pprof_py.inference import HUBER_RLM

model.test(null_model=EmpiricalNull.fitter(size=facility_size, n_groups=4, grouping="quantile",
                                           estimator=HUBER_RLM))
```

Earlier versions of pprof_py instead applied an empirical null by default
with a Huber fit in four equal-count groups of discharge counts. That
configuration, which is not R's, is:

```python
sizes = df.groupby("facility_id", observed=True).size().loc[model.provider_ids_].to_numpy()
model.test(null_model=EmpiricalNull.fitter(size=sizes, n_groups=4, grouping="rank",
                                           estimator=HUBER_RLM, small_group="theoretical"))
```

## Survival models

`pprof_py.inference.survival` keeps R-compatible wrappers —
`fit_empirical_null`, `fit_grouped_empirical_null` and
`adjust_empirical_null`, with `poisson_midp_zscore` for observed and
expected counts — that run on the same estimator and grouping with the
settings of their R counterparts ({ref}`survival_ref_inference`).

## Operating characteristics

A simulation study shows what calibration buys and costs. Each replicate draws providers whose effects
on the logit scale are N(0, τ²), the unexplained between-provider variation an empirical null is meant
to absorb, and shifts a fraction of them (the outliers) up by 0.8; it fits
`LogisticFixedEffectModel` and tests every provider at the two-sided 5% level. The tables give the
share of non-outlying providers flagged (in either direction), which a calibrated test holds near 0.05
whatever τ, and the share of outliers flagged as worse (power). "Overall" is `EmpiricalNull` without
groups; the grouped nulls use four quantile groups of provider size; Huber is `HUBER_RLM`, the
production R recipe's estimator, and bisquare the package default.

Flag rates with 200 providers of 20-400 records each (Wald test):

| τ | outliers | theoretical | overall | 4 groups, Huber | 4 groups, bisquare | power: theoretical | power: 4 groups, Huber |
|---|---|---|---|---|---|---|---|
| 0 | 0% | 0.045 | 0.047 | 0.051 | 0.051 | – | – |
| 0 | 5% | 0.045 | 0.037 | 0.045 | 0.044 | 0.920 | 0.912 |
| 0 | 15% | 0.044 | 0.015 | 0.025 | 0.022 | 0.916 | 0.807 |
| 0.1 | 0% | 0.078 | 0.051 | 0.051 | 0.050 | – | – |
| 0.1 | 5% | 0.083 | 0.035 | 0.035 | 0.034 | 0.908 | 0.884 |
| 0.1 | 15% | 0.089 | 0.013 | 0.020 | 0.020 | 0.903 | 0.752 |
| 0.2 | 0% | 0.188 | 0.064 | 0.053 | 0.053 | – | – |
| 0.2 | 5% | 0.175 | 0.042 | 0.040 | 0.041 | 0.920 | 0.820 |
| 0.2 | 15% | 0.181 | 0.025 | 0.022 | 0.025 | 0.889 | 0.620 |
| 0.3 | 0% | 0.287 | 0.067 | 0.055 | 0.055 | – | – |
| 0.3 | 5% | 0.292 | 0.045 | 0.039 | 0.040 | 0.872 | 0.636 |
| 0.3 | 15% | 0.293 | 0.028 | 0.018 | 0.020 | 0.852 | 0.463 |

Each row averages 25 replicates; the Monte Carlo standard errors of the flag rates are at most 0.007. The score and `poibin_exact` tests give rates within 0.008 of these.

Other settings (Wald test for the logistic model; 8-25 replicates per row, Monte Carlo standard errors at most 0.015; the Cox rows use outliers at +0.5 on the log-hazard scale):

| setting | τ | outliers | theoretical | overall | 4 groups, Huber | power: theoretical | power: 4 groups, Huber |
|---|---|---|---|---|---|---|---|
| 50 providers, sizes 20–400 | 0 | 0% | 0.043 | 0.059 | 0.074 | – | – |
| 50 providers, sizes 20–400 | 0 | 5% | 0.044 | 0.043 | 0.062 | 0.960 | 0.880 |
| 50 providers, sizes 20–400 | 0.1 | 0% | 0.081 | 0.060 | 0.060 | – | – |
| 50 providers, sizes 20–400 | 0.1 | 5% | 0.078 | 0.025 | 0.048 | 0.920 | 0.840 |
| 50 providers, sizes 20–400 | 0.2 | 0% | 0.172 | 0.056 | 0.054 | – | – |
| 50 providers, sizes 20–400 | 0.2 | 5% | 0.174 | 0.064 | 0.063 | 0.940 | 0.800 |
| 50 providers, sizes 20–400 | 0.3 | 0% | 0.271 | 0.071 | 0.068 | – | – |
| 50 providers, sizes 20–400 | 0.3 | 5% | 0.282 | 0.059 | 0.060 | 0.920 | 0.620 |
| 200 providers, sizes 10–60 | 0 | 0% | 0.041 | 0.041 | 0.047 | – | – |
| 200 providers, sizes 10–60 | 0 | 5% | 0.035 | 0.031 | 0.037 | 0.498 | 0.453 |
| 200 providers, sizes 10–60 | 0 | 15% | 0.035 | 0.013 | 0.017 | 0.498 | 0.324 |
| 200 providers, sizes 10–60 | 0.1 | 0% | 0.046 | 0.046 | 0.051 | – | – |
| 200 providers, sizes 10–60 | 0.1 | 5% | 0.037 | 0.031 | 0.033 | 0.620 | 0.522 |
| 200 providers, sizes 10–60 | 0.1 | 15% | 0.035 | 0.014 | 0.019 | 0.493 | 0.334 |
| 200 providers, sizes 10–60 | 0.2 | 0% | 0.061 | 0.044 | 0.044 | – | – |
| 200 providers, sizes 10–60 | 0.2 | 5% | 0.065 | 0.030 | 0.032 | 0.532 | 0.403 |
| 200 providers, sizes 10–60 | 0.2 | 15% | 0.055 | 0.019 | 0.023 | 0.518 | 0.319 |
| 200 providers, sizes 10–60 | 0.3 | 0% | 0.091 | 0.051 | 0.049 | – | – |
| 200 providers, sizes 10–60 | 0.3 | 5% | 0.086 | 0.034 | 0.040 | 0.501 | 0.354 |
| 200 providers, sizes 10–60 | 0.3 | 15% | 0.077 | 0.019 | 0.018 | 0.471 | 0.235 |
| 1,000 providers, sizes 20–400 | 0 | 5% | 0.047 | 0.037 | 0.038 | 0.938 | 0.927 |
| 1,000 providers, sizes 20–400 | 0.2 | 5% | 0.191 | 0.042 | 0.032 | 0.875 | 0.748 |
| Cox SMR, 200 providers, sizes 10–60 | 0 | 0% | 0.048 | 0.050 | 0.055 | – | – |
| Cox SMR, 200 providers, sizes 10–60 | 0 | 5% | 0.046 | 0.053 | 0.053 | 0.480 | 0.415 |
| Cox SMR, 200 providers, sizes 10–60 | 0.1 | 0% | 0.069 | 0.055 | 0.061 | – | – |
| Cox SMR, 200 providers, sizes 10–60 | 0.1 | 5% | 0.059 | 0.043 | 0.044 | 0.470 | 0.365 |
| Cox SMR, 200 providers, sizes 10–60 | 0.2 | 0% | 0.122 | 0.049 | 0.048 | – | – |
| Cox SMR, 200 providers, sizes 10–60 | 0.2 | 5% | 0.109 | 0.049 | 0.051 | 0.430 | 0.265 |
| Cox SMR, 200 providers, sizes 10–60 | 0.3 | 0% | 0.184 | 0.059 | 0.063 | – | – |
| Cox SMR, 200 providers, sizes 10–60 | 0.3 | 5% | 0.194 | 0.054 | 0.049 | 0.440 | 0.200 |

What the study shows:

- **The theoretical null does not absorb overdispersion.** Its flag rate among non-outlying providers
  rises from 0.045 at τ = 0 to 0.19 at τ = 0.2 and 0.29 at τ = 0.3.
- **Grouped by size, the empirical null holds its level.** Without outliers the four-group null stays
  at 0.051-0.055 for every τ. The overall null drifts up to 0.064-0.067 at τ ≥ 0.2: with
  overdispersion a large provider's z-statistic spreads more than a small one's (its standard error is
  smaller), so one null is too narrow for the large providers and too wide for the small ones.
  Grouping by size is the remedy, and is why the production recipe groups.
- **Outliers make it conservative, and cost power.** Outliers on one side pull the robust estimates of
  location and scale: with 5% outliers the rate among the other providers is about 0.035-0.045, with
  15% about 0.02, and power falls with τ (to 0.46 at τ = 0.3 with 15% outliers, against 0.85 for the
  theoretical null). The method assumes most providers follow the null; beyond roughly 10% one-sided
  outliers it under-flags.
- **Few providers per group make it liberal.** With 50 providers (about 12 per group) the grouped null
  flags 0.054-0.074 without outliers, because each group's scale comes from a dozen values; the overall
  null flags 0.056-0.071. For fewer than about 100 providers, use fewer groups or the overall null.
  Three of the 300 replicates there also warned that a group's Huber fit did not converge in
  `MASS::rlm`'s 20 iterations. With 1,000 providers the grouped null behaves as with 200.
- **Small providers leave less to calibrate.** With 10-60 records per provider the sampling noise
  dwarfs τ: the theoretical null's rate only reaches 0.09 at τ = 0.3, the empirical nulls hold
  0.044-0.051 without outliers, and power is low (0.3-0.6) under every null.
- **The test and the estimator matter little.** The score and `poibin_exact` tests stay within 0.008 of
  the Wald test (slightly higher on average), and bisquare within 0.005 of Huber.
- **Cox SMRs behave the same way.** For indirect SMRs of a provider-stratified `CoxPH`
  (`calculate_standardized_measures`), mid-p z-statistics (`poisson_midp_zscore`) and groups by
  person-time, as in the SMR tutorial, the theoretical null's rate rises to 0.18-0.19 at τ = 0.3 (the SD
  of the log hazard ratios) while the empirical nulls hold 0.048-0.063. With 10-60 patients per provider,
  power for outliers at +0.5 is low and falls as τ grows (0.42 to 0.20).

## Validation

The package's tests check these estimators against `MASS::rlm` (Huber,
bisquare and MM) and the empirical-null fits of the EmpiNull R package
(`fit_pci`, `cal_Z_htaz` and `empirical_null_groupwise`) to within
1e-12.
