(presentation-guide)=
# Presentation layer (preview)

`pprof_py.presentation` turns a provider test into figures and tables that cannot contradict it. A
{class}`~pprof_py.presentation.ProviderProfile` holds one test's estimates, intervals, flags, denominators and
settings; the funnel plot, the interval plot and the provider table read it and show its provenance. The namespace is
provisional until the presentation layer is complete.

* Every estimate, interval, flag and control limit comes from the test itself (funnel limits from
  {func}`~pprof_py.inference.funnel_limits`), so a provider lies outside its limits, and its interval excludes the
  reference, exactly when it is flagged.
* Providers without a finite estimate, untested providers and suppressed providers are shown and counted, never
  dropped.
* Figures are built without pyplot and export byte-identical SVG, PDF and PNG; every figure carries alt text and a
  provenance footnote.

## Example

```python
import numpy as np
import pandas as pd
from pprof_py import LogisticFixedEffectModel
from pprof_py.presentation import ProviderProfile, caterpillar, funnel, provider_table

rng = np.random.default_rng(7)
size = rng.integers(30, 150, 16)
gamma = rng.normal(-1.2, 0.3, 16)
gamma[:3] += [0.8, -0.8, 0.6]
pid = np.repeat(np.arange(16), size)
x = rng.normal(size=pid.size)
y = rng.binomial(1, 1 / (1 + np.exp(-(gamma[pid] + 0.5 * x))))
model = LogisticFixedEffectModel()
model.fit(pd.DataFrame({"y": y, "x": x, "unit": [f"U{j:02d}" for j in pid]}), y_var="y", x_vars=["x"],
          provider_var="unit")

profile = ProviderProfile.from_model(model, test_method="poibin_exact", limits=True)
print(funnel(profile).alt_text)
print(caterpillar(profile).alt_text)
print(provider_table(profile).to_text())
```
```text
Funnel plot of 16 providers, observed over expected events against precision: 5 above and 2 below the reference at the 95% level, 9 not different.
Interval plot of 16 providers ordered by estimate, with 95% intervals: 5 above and 2 below the reference, 9 not different. Volume panel: records per provider from 30 to 143 (median 108).
Provider results: Provider effect (log-odds), 16 providers
==============================================================================
Provider    N  Observed  Expected  Provider effect (log-odds) (95% CI)ᵃ  Flagᵇ
--------  ---  --------  --------  ------------------------------------  -----
U00       143        48      29.0                −0.64 (−1.01 to −0.28)    ▲
U01       105        11      21.7                −2.21 (−2.89 to −1.61)    ▼
U02       112        47      24.9                 −0.37 (−0.77 to 0.02)    ▲
U03       137        27      27.5                −1.40 (−1.85 to −0.97)    ●
U04        99        32      19.7                −0.68 (−1.13 to −0.25)    ▲
U05       123        24      27.6                −1.56 (−2.04 to −1.11)    ●
U06       130        34      27.8                −1.09 (−1.51 to −0.69)    ●
U07        57        23      13.8                −0.56 (−1.13 to −0.01)    ▲
U08        36        10       8.8                −1.19 (−1.99 to −0.44)    ●
U09        66        13      13.6                −1.43 (−2.09 to −0.83)    ●
U10        64         9      13.7                −1.92 (−2.70 to −1.22)    ●
U11       134        27      28.8                −1.46 (−1.92 to −1.03)    ●
U12       139        18      32.3                −2.13 (−2.67 to −1.64)    ▼
U13        30         8       7.9                −1.35 (−2.27 to −0.50)    ●
U14        89        17      19.2                −1.54 (−2.12 to −1.01)    ●
U15       128        42      30.1                −0.86 (−1.26 to −0.48)    ▲
==============================================================================
ᵃ Estimates: fixed effect (unshrunken); 95% test-inversion intervals from the same test as the flags; an interval excludes the reference exactly when the provider is flagged. Reference: median provider effect −1.37.
ᵇ ▲ above or ▼ below the reference; ● not different; NT not tested. Test: exact Poisson-binomial test, two-sided, 95% level per provider; theoretical null N(0, 1).
Source: LogisticFixedEffectModel; pprof_py 0.5.0. 16 providers.

```

The figure objects export with `fig.save("funnel.svg")` (or `.pdf`, `.png`), give the Matplotlib figure as
`fig.figure`, and display themselves in notebooks. Tables export with `to_html()`, `to_markdown()`, `to_latex()`,
`to_text()`, `to_frame()` and, with the optional `excel` extra, `to_excel()`. Styles come from a
{class}`~pprof_py.presentation.Theme` (`theme="publication"`, `"notebook"` or `"report"`, or a derived theme).

## Covariate effects

The coefficient forest and table show a model's `summary()`: one row per term (the intercept is left out unless
requested), with odds or hazard ratios on a log axis for logistic and Cox models.

```python
from pprof_py.presentation import coefficient_table, forest

print(forest(model).alt_text)
print(coefficient_table(model).to_text())
```
```text
Forest plot of 1 terms (odds ratios) with 95% intervals: 1 of them exclude 1.
Covariate effects: odds ratios, 1 terms
========================================
Covariate  Odds ratio (95% CI)ᵃ  p-value
---------  --------------------  -------
x              1.97 (1.73–2.25)   <0.001
========================================
ᵃ 95% intervals from LogisticFixedEffectModel.summary(). Each estimate is adjusted for the other covariates and the provider effects; associations are not causal, and covariates have their own units. Odds ratios are the exponentiated coefficients and bounds; p-values are those of the coefficients.
Source: LogisticFixedEffectModel; pprof_py 0.5.0.

```

## Data quality

The data-quality panel and table account for every provider: those excluded by data preparation (as the model
recorded them), untested and suppressed providers, and providers without a finite estimate, with each group's
volumes. A count the source does not record is shown as "not recorded", never as zero.

```python
from pprof_py.presentation import data_quality, data_quality_table

print(data_quality(profile).alt_text)
print(data_quality_table(profile, details=True).to_text())
```
```text
Data-quality summary: 16 providers in the data, 0 excluded by data preparation, 16 analysed (5 above, 2 below, 9 not different, 0 not tested); 0 without a finite estimate; records per provider from 30 to 143 (median 108).
Data quality: 0 providers with an issue
=========================
Provider  Recordsᵃ  Issue
--------  --------  -----
=========================
ᵃ Records per provider.
Source: LogisticFixedEffectModel; exact Poisson-binomial test, two-sided, 95% level per provider; pprof_py 0.5.0.

```

## Observed versus expected

The observed-versus-expected plot shows each provider's observed and expected events on square-root axes, where
Poisson noise has roughly constant spread, with the line O = E and the test's limits converted to events.

```python
from pprof_py.presentation import observed_expected

print(observed_expected(profile).alt_text)
```
```text
Observed against expected events for 16 providers: 5 above and 2 below the reference at the 95% level; observed minus expected ranges from −14.3 to 22.1 events.
```

## Reliability

The reliability display shows how well a measure separates providers of each size: each provider's reliability at
its size, the curve it traces, and the overall inter-unit reliability (IUR). Reliability is a property of the
measure, not a score for any provider.

```python
from pprof_py.measures.iur import BootstrapIUR
from pprof_py.presentation import reliability, reliability_table

iur = BootstrapIUR(n_boot=50).fit(y.astype(float), np.full(y.size, y.mean()), pid)
print(reliability(iur).alt_text)
print(reliability_table(iur).to_text())
```
```text
Reliability by provider size for 16 providers: from 0.54 at size 30 to 0.85 at size 143; overall IUR 0.79 at the effective size 99.
Reliability of the measure (BootstrapIUR)
======================================
Quantity                        Valueᵃ
------------------------------  ------
Overall IUR                       0.79
Effective size n′                   99
Between-provider variance       0.1208
Within-provider variance        3.0911
Providers                           16
Reliability: Smallest provider    0.54
Reliability: Decile 1             0.56
Reliability: Decile 2             0.70
Reliability: Decile 3             0.72
Reliability: Decile 4             0.79
Reliability: Decile 5             0.80
Reliability: Decile 6             0.82
Reliability: Decile 7             0.83
Reliability: Decile 8             0.84
Reliability: Decile 9             0.84
Reliability: Decile 10            0.85
Reliability: Largest provider     0.85
======================================
ᵃ Reliability at a size n is s² between / (s² between + s² within / n); deciles are groups of providers by size, from the smallest to the largest. Reliability is a property of the measure at a given volume, not a score for any provider.
Source: BootstrapIUR.

```

## Between-provider variation

For random-effect models, the variation display compares the shrunken provider effects (BLUPs) with the fitted
between-provider distribution, states the random-effect SD with its interval where the model provides one, and shows
the range in which most true provider effects would lie if that distribution is normal.

```python
from pprof_py import LogisticRandomEffectModel
from pprof_py.presentation import provider_variation, provider_variation_table

re_model = LogisticRandomEffectModel(verbose=False)
re_model.fit(pd.DataFrame({"y": y, "x": x, "unit": [f"U{j:02d}" for j in pid]}), y_var="y", x_vars=["x"],
             provider_var="unit")
print(provider_variation(re_model).alt_text)
print(provider_variation_table(re_model).to_text())
```
```text
Between-provider variation in 16 providers: random-effect SD 0.48 (95% profile likelihood interval 0.31–0.76); 95% of true provider effects would lie within −0.94 to 0.94 (odds ratios 0.39 to 2.55) under a normal random-effect distribution; the BLUPs range from −0.68 to 0.76.
Between-provider variation (LogisticRandomEffectModel)
===========================================
Quantity                             Valueᵃ
-----------------------------------  ------
Random-effect SD (σ)                   0.48
σ, 95% interval: lower                 0.31
σ, 95% interval: upper                 0.76
Range of 95% of true effects: lower   −0.94
Range of 95% of true effects: upper    0.94
As odds ratios: lower                  0.39
As odds ratios: upper                  2.55
Providers                                16
SD of the BLUPs (descriptive)          0.43
===========================================
ᵃ Effects on the log-odds scale, relative to the average provider. The range is ±1.96σ and assumes normal random effects; the BLUPs spread less than σ because they are shrunk toward the average.
Source: LogisticRandomEffectModel; profile likelihood interval for σ.

```

## Shrinkage

The shrinkage display pairs a fixed-effect and a random-effect fit of the same family: each provider's unshrunken
estimate against its BLUP, both relative to their test's reference, with point area proportional to volume. Small
providers move most; how far depends on the assumed random-effect distribution.

```python
from pprof_py.presentation import shrinkage, shrinkage_table

print(shrinkage(model, re_model).alt_text)
print(shrinkage_table(model, re_model).to_text())
```
```text
Shrinkage of 16 providers: fixed-effect estimates from −0.94 to 0.91, random-effect estimates from −0.68 to 0.76 (log-odds).
Shrinkage: 16 providers
=========================================================
Provider  Records  Fixed effectᵃ  Random effectᵇ  Changeᶜ
--------  -------  -------------  --------------  -------
U00           143           0.64            0.54    −0.10
U01           105          −0.94           −0.67     0.26
U02           112           0.91            0.76    −0.15
U03           137          −0.12           −0.11     0.01
U04            99           0.59            0.47    −0.12
U05           123          −0.29           −0.24     0.04
U06           130           0.19            0.14    −0.04
U07            57           0.71            0.52    −0.20
U08            36           0.09            0.05    −0.04
U09            66          −0.16           −0.12     0.04
U10            64          −0.64           −0.42     0.22
U11           134          −0.19           −0.17     0.02
U12           139          −0.86           −0.68     0.18
U13            30          −0.08           −0.05     0.03
U14            89          −0.27           −0.21     0.06
U15           128           0.41            0.34    −0.07
=========================================================
ᵃ Unshrunken, LogisticFixedEffectModel, relative to the size-weighted mean of the fixed effects.
ᵇ Shrunken BLUP, LogisticRandomEffectModel, relative to the model's intercept; it depends on the assumed normal random-effect distribution and is not the true effect.
ᶜ Random minus fixed, computed for display; NE: no finite fixed-effect estimate.
Source: LogisticFixedEffectModel and LogisticRandomEffectModel.

```

## Null calibration

The calibration diagnostic shows the raw z-statistics of each null group against the theoretical N(0, 1) and the
fitted null, and compares the flags of the two nulls. A model is tested twice; the flags are the test's own.

```python
from pprof_py.inference import EmpiricalNull
from pprof_py.presentation import null_calibration, null_calibration_table

empirical = EmpiricalNull.fitter()
print(null_calibration(model, test_method="score", null_model=empirical).alt_text)
print(null_calibration_table(model, test_method="score", null_model=empirical).to_text())
```
```text
Null-calibration diagnostic for 16 providers in 1 group; fitted null means 0.60 to 0.60 and SDs 2.52 to 2.52; flagged under the theoretical null: 7, under the fitted null: 0; 7 flags change.
Null calibration: 16 providers
==================================================================================
Null group  Providers  Null mean  Null SD  Theoretical null  Fitted null  Changedᵃ
----------  ---------  ---------  -------  ----------------  -----------  --------
1                  16       0.60     2.52  ▲ 5 / ▼ 2         ▲ 0 / ▼ 0           7
==================================================================================
ᵃ ▲ above / ▼ below the reference. Test: score test, two-sided, 95% level per provider; empirical null, mean 0.60 and SD 2.52. Changed: providers whose flag differs between the two nulls.
Source: LogisticFixedEffectModel; pprof_py 0.5.0.

```

## Migrating from the model plot methods

The model methods keep working: most now delegate to the new displays, and the others keep their earlier drawing with
a `DeprecationWarning` until 0.7.0.

| Earlier call | Now |
|---|---|
| `model.plot_funnel(...)` (logistic and linear fixed effects) | Delegates to `funnel(model, ...)` and returns a `FigureResult`; `fig, ax = model.plot_funnel()` still works. `alpha` sets the curve levels. |
| `model.plot_funnel()` (logistic random effects) | The funnel of the exact count test (`test_method="poibin_exact"`); the default `"wald"` warns, is replaced, and raises in 0.7.0. |
| `model.plot_funnel()` (linear random effects) | Deprecated: no funnel can agree with the test of shrunken estimates. Use `caterpillar(model)`. |
| `model.plot_provider_effects(...)` | Delegates to `caterpillar(model, ...)`. |
| `model.plot_standardized_measures(...)` (logistic fixed effects) | Delegates to `caterpillar` with the profile of `model.test_standardized(...)`. |
| `model.plot_standardized_measures(...)` (other models) | Deprecated: their measure-scale intervals do not come from the flagging test. Use `caterpillar(model)`. |
| Styling keywords (`point_colors`, `labels`, `figsize`, ...), `target=`, `use_flags=False` | No effect; `DeprecationWarning` now, `TypeError` in 0.7.0. Use `theme=`. |
| `save_path=` | Still works; or `result.save(path)`. |
| `plt.show()` after a plot call | Not needed in notebooks; in scripts, save the result. |
| `plot_caterpillar(df)` with `lower`/`upper` columns | The defaults are now `ci_lower`/`ci_upper`, the columns of `test()`; `lower`/`upper` are used with a warning until 0.7.0. |
| A hand-built `limits_df` for `pprof_py.plotting.plot_funnel` | `pprof_py.inference.funnel_limits(model)`, or `funnel(model)`. |
| (new) | `provider_table(model)`: HTML, Markdown, LaTeX, text, DataFrame and Excel. |
| `model.plot_coefficient_forest(...)` | `forest(model)` and `coefficient_table(model)` (the method itself is unchanged for now). |
