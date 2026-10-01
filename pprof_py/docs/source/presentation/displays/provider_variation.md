# Between-provider variation

```{include} ../_figures/provider_variation.md
```

**Answers:** How much real variation exists across providers?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | How much real variation exists across providers? |
| Quantity and source | The random-effect SD (`sigma_` for logistic models, `random_effect_sd_` for linear models, whose `sigma_` is the residual SD), its profile-likelihood interval where the model provides one (`profile_sigma()`), and the BLUPs (`get_random_effects()`). The range ±zσ of true effects and its odds ratios are named conversions of σ. |
| Uncertainty | The SD's interval, drawn as densities at its bounds; for linear models, which report none, the footnote says so. The display shows no per-provider uncertainty. |
| Denominator | Not shown per provider; the shrinkage display shows how volume affects each BLUP. |
| Reference | 0, the average provider. |
| Misreadings and mitigations | Not all spread is true variation: the BLUPs spread less than σ because they are shrunk toward the average, and unshrunken estimates spread more because they include sampling noise. The range of true effects assumes a normal random-effect distribution, which the footnote states. |
| Static or interactive | Static. |
| From 10 to 50,000 providers | A histogram: the same at any size. |

## How to read it

The curve is the model's estimate of how true provider effects are distributed; the bracket shows where 95% of them would lie if that distribution is normal, also as odds ratios. The histogram of BLUPs is narrower than the curve, as shrinkage predicts.

## How it can mislead

The spread of raw provider estimates overstates the true variation, and the spread of BLUPs understates it. Only σ and its interval estimate the true variation.

## Call

```{code-block} python
from pprof_py import LogisticRandomEffectModel
from pprof_py.presentation import provider_variation, provider_variation_table

random_model = LogisticRandomEffectModel().fit(data, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
provider_variation(random_model)
```
