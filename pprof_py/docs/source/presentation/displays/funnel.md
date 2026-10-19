# Funnel plot

```{include} ../_figures/funnel.md
```

**Answers:** Which providers depart from the reference by more than the test's sampling variation explains, and how does that depend on precision?

## Spec sheet

| Item | Content |
|:--|:--|
| Question | Which providers depart from the reference by more than the test's sampling variation explains, and how does that depend on precision? |
| Quantity and source | One `test()` call and the limits `funnel_limits()` derives from the same test: for each provider the plotted estimate (O/E for count and score tests, the effect or difference for Wald tests), its precision, the limits at each level and the test's flag. |
| Uncertainty | The test's acceptance region at its level: curves, or each provider's own limits as marks where the test is discrete (exact Poisson-binomial). Other levels, such as 99.8%, are drawn as reference curves; flags exist only at the test's level. Where the test's own curve reproduces the flags, the region between its limits is shaded: a provider of that precision inside it is not flagged. Per-provider marks (exact tests) and per-group curves (a grouped empirical null) are never shaded, because one region would misplace providers. |
| Denominator | The x axis is the test's precision, labelled by kind: expected events for count tests, E²/V₀ for the score test, 1/SE² for Wald tests. |
| Reference | A line at the null value (O/E = 1 or effect 0); the footnote says what it is, for example the median provider effect. Publication figures carry a one-line footnote and the full text as `FigureResult.caption`. |
| Misreadings and mitigations | A point near a limit is not proof: limits are labelled with their test and level, and discrete limits sit between attainable counts, so no provider lies on a line. Overdispersion under the theoretical null: the footnote names the null model, and the null-calibration display shows whether a fitted null is needed. Limits are per provider, not adjusted for multiplicity, and the footnote says so. Random-effect models that flag with a Wald test on BLUPs get no funnel, because any O/E funnel would contradict their flags. Zero-event providers are drawn with their own marker. |
| Static or interactive | Static. The provider table is the static equivalent of point details. |
| From 10 to 50,000 providers | All points are vectors up to 2,000 providers; above that the not-different layer is rasterized and flagged providers are drawn on top. Labels: `highlight=` providers, and flagged providers when there are at most 10. |

## How to read it

Each point is a provider. A provider above the upper limit has more events than the reference predicts by more than the test's sampling variation at its precision explains, and one below the lower limit fewer; the flag shown is the test's own, so a point outside the limits is exactly a flagged provider. The limits narrow as precision grows: large providers are held to tighter bounds than small ones.

## How it can mislead

Sorting the same providers by their estimate, as a league table does, lifts small providers and pushes large ones
down. Below, the first three providers that a league table places among the 40 highest ratios are not flagged: each
has fewer than 10 expected events. The two largest flagged providers, whose excess is the most certain, sit at
positions 45 and 47. The funnel shows the difference directly, with precision on the x axis and limits that narrow as
it grows.

```python
from pprof_py import LogisticFixedEffectModel
from pprof_py.presentation import ProviderProfile
from pprof_py.presentation._synthetic import provider_data

data = provider_data(200)
model = LogisticFixedEffectModel()
model.fit(data, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")

profile = ProviderProfile.from_model(model, limits=True)          # the score funnel, as funnel(model) draws it
league = profile.data.sort_values("funnel_estimate", ascending=False)
league["position"] = range(1, len(league) + 1)
unflagged_high = league[(league["position"] <= 40) & (league["status"] != "above")].head(3)
large_flagged = league[league["status"] == "above"].nlargest(2, "expected")
for name, rows in (("unflagged, high in the league table", unflagged_high), ("largest flagged", large_flagged)):
    print(name)
    print(rows[["position", "funnel_estimate", "expected", "status"]].round(2).to_string())
```
```text
unflagged, high in the league table
             position  funnel_estimate  expected         status
provider_id
F022               19             1.53      8.51  not_different
F046               23             1.49      8.05  not_different
F080               26             1.46      8.89  not_different
largest flagged
             position  funnel_estimate  expected status
provider_id
F181               47             1.28    117.64  above
F194               45             1.31     63.15  above
```

## Call

```{code-block} python
from pprof_py.presentation import funnel

fig = funnel(model, levels=(0.95, 0.998))
fig.save("funnel.svg")
```
