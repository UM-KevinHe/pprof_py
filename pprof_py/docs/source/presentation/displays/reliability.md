# Reliability

```{include} ../_figures/reliability.md
```

**Answers:** How reliably does the measure separate providers of each size?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | How reliably does the measure separate providers of each size? |
| Quantity and source | A fitted `BootstrapIUR`: each provider's reliability at its size (`iur_groups_`, `group_sizes_`), the overall IUR (`iur_`) and the effective size (`n_prime_`). The table also covers `DirectIUR` and `SplitHalfIUR`, which do not store per-provider reliability. |
| Uncertainty | Not per provider, and no interval for the IUR, which the classes do not provide. |
| Denominator | Provider size on the x axis. |
| Reference | The overall IUR as a line, with the effective size n′ marked. |
| Misreadings and mitigations | Reliability is a property of the measure at a given volume, the share of the spread between providers of that size that is signal; it is not a score for any provider. The footnote says so. |
| Static or interactive | Static. |
| From 10 to 50,000 providers | The points are rasterized above 2,000 providers. |

## How to read it

The curve rises with provider size: for small providers most of the spread in the measure is noise, for large ones most is signal. The overall IUR is the reliability at the effective size n′.

## How it can mislead

A provider of the size where reliability is 0.3 is not "30% reliable" in any personal sense; the value says that, among providers of that size, most of the observed spread is noise, so their individual estimates should be read with caution.

## Call

```{code-block} python
from pprof_py.measures.iur import BootstrapIUR
from pprof_py.presentation import reliability, reliability_table

iur = BootstrapIUR().fit(observed, expected, providers)
reliability(iur)
```
