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
