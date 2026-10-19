# Null calibration

```{include} ../_figures/null_calibration.md
```

**Answers:** Is the theoretical null credible, and how much does calibration change the flags?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | Is the theoretical null credible, and how much does calibration change the flags? |
| Quantity and source | The raw z-statistics and the fitted null's mean and SD (`z_raw`, `null_mean`, `null_sd`, `null_group`) from `test()`, and the flags of two `test()` calls: with the requested null and with the theoretical one. The display decides no flag; it only counts them. |
| Uncertainty | The null distributions themselves (theoretical and fitted) rather than per-provider uncertainty. |
| Denominator | The null groups, often defined by provider size, with their counts in the panel titles. |
| Reference | The theoretical N(0, 1) density and the fitted null's normal density. |
| Misreadings and mitigations | Excess spread is not proof of many outlying providers: a null SD above 1 indicates overdispersion or unmodelled variation, and the footnote says so. For exact and Monte Carlo tests the raw z is a converted statistic whose sign near a provider's expected count is arbitrary; the footnote warns. |
| Static or interactive | Static. |
| From 10 to 50,000 providers | Histograms: the same at any size. |

## How to read it

In each null group the bars are the raw z-statistics. If the theoretical null were right they would follow the dashed N(0, 1) curve; when they spread wider, the fitted null (solid) describes them better, and the counts show how many flags remain once the tests use it.

## How it can mislead

Under the theoretical null, overdispersion turns ordinary variation into flags. In the example the fitted nulls are wider than N(0, 1) in every size group, and calibration removes most of the flags.

```python
from pprof_py import LogisticFixedEffectModel
from pprof_py.presentation import ProviderProfile
from pprof_py.presentation._synthetic import provider_data

data = provider_data(200)
model = LogisticFixedEffectModel()
model.fit(data, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
from pprof_py.inference import EmpiricalNull

empirical = EmpiricalNull.fitter(size=model.provider_sizes_, n_groups=3)
theoretical = model.test(test_method="score")
calibrated = model.test(test_method="score", null_model=empirical)
print("flagged under the theoretical null:", int((theoretical["flag"] != 0).sum()))
print("flagged under the fitted null:", int((calibrated["flag"] != 0).sum()))
print(calibrated.groupby("null_group")[["null_mean", "null_sd"]].first().round(2).to_string())
```
```text
flagged under the theoretical null: 47
flagged under the fitted null: 10
            null_mean  null_sd
null_group
1                0.03     1.43
2                0.06     1.55
3                0.15     1.86
```

## Call

```{code-block} python
from pprof_py.inference import EmpiricalNull
from pprof_py.presentation import null_calibration, null_calibration_table

null_calibration(model, test_method="score", null_model=EmpiricalNull.fitter())
```
