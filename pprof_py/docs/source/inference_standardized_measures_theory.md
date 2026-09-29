(inference_measures_theory)=
# Inference for Standardized Measures

This page is Part III of the theory series on inference in `pprof_py`. It covers the standardized measures reported
for providers — indirect and direct standardized ratios and rates — their relation to the provider effects of
[Part II](inference_provider_effects_theory.md), their standard errors and tests, and their confidence limits, in the
logistic, linear and survival models. [Part I](inference_covariate_effects_theory.md) treats the covariate effects;
the [empirical-null page](empirical_null_theory.md) the calibration of the tests; the
[inter-unit reliability page](iur_theory) the reliability of the measures. Every code block below runs as written,
and every numerical claim was checked by simulation or exact computation.

## 1. Introduction

A provider effect $\gamma_k$ is a log odds ratio relative to a reference, readable only through the model. The measures
reported to the public translate it into events: how many the provider's patients had against how many they would
have had at the reference provider (*indirect* standardization), or how many the whole population would have had at
provider $k$ against how many it had (*direct* standardization). The two answer different questions, differ when a
provider's case mix is atypical, and have different inferential properties {cite}`inm-Jones2008Indirect`. Section 2
defines them for the logistic fixed-effect model and shows that each is a monotone function of $\gamma_k$; Section 3
gives their standard errors and tests; Section 4 their confidence limits; Section 5 the linear and random-effect
models; Section 6 the survival model; Section 7 the sources of error they ignore; Sections 8 and 9 the approximations
relied on and the implementation.

## 2. Definitions

### 2.1. The two expected counts

With $\hat\beta$ fixed, write $p_{kj}(g) = \operatorname{expit}(g + x_{kj}^\top\hat\beta)$ for record $j$ of provider $k$
at provider effect $g$, and define two functions of $g$:

$$
h_k(g) = \sum_j N_{kj}\, p_{kj}(g) \quad\text{(provider $k$'s own records)}, \qquad
H(g) = \sum_{l,j} N_{lj}\, p_{lj}(g) \quad\text{(all records)}.
$$

Both increase strictly in $g$. With $O_k$ provider $k$'s events, $O = \sum_k O_k$ and $N = \sum N_{lj}$:

$$
\text{indirect ratio}_k = \frac{O_k}{E_k}, \quad E_k = h_k(\gamma_0); \qquad
\text{direct ratio}_k = \frac{H(\hat\gamma_k)}{O}, \qquad
\text{direct rate}_k = \frac{H(\hat\gamma_k)}{N}.
$$

The indirect rate is the indirect ratio times the population's crude rate $O/N$. `calculate_standardized_measures`
reports rates in percent, clipped to [0, 100] (R's `SM_output`); `test_standardized` reports them as proportions.

### 2.2. Both measures are functions of $\gamma_k$

At the maximum-likelihood estimate the score for $\gamma_k$ vanishes, $\sum_j (Y_{kj} - N_{kj}p_{kj}(\hat\gamma_k)) = 0$,
so $h_k(\hat\gamma_k) = O_k$ for every provider with a finite estimate, and

$$
\text{indirect ratio}_k = \frac{h_k(\hat\gamma_k)}{h_k(\gamma_0)}, \qquad \text{direct ratio}_k = \frac{H(\hat\gamma_k)}{O}.
$$

Each is a strictly increasing function of $\hat\gamma_k$, so, with $\hat\beta$ fixed, any test or interval for
$\gamma_k$ maps to one for the measure (Section 4.1). They differ in the case mix over which they average: the
indirect ratio uses provider $k$'s own patients, the direct ratio the population's. When a provider's case mix is
atypical the two disagree, because one log odds ratio translates into different ratios of events at different
baseline risks:

```python
import numpy as np
import pandas as pd
from pprof_py import LogisticFixedEffectModel

def cohort(rng, m=60, sizes=(15, 150), tau=0.4):
    """Providers with their own case mix (covariate mean) and true effects gamma_k ~ N(-1, tau^2)."""
    n = rng.integers(sizes[0], sizes[1], m); prov = np.repeat(np.arange(m), n)
    mu = rng.normal(0, 0.7, m); gamma = -1 + tau * rng.normal(size=m)
    x1, x2 = rng.normal(mu[prov], 1), rng.normal(size=prov.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(gamma[prov] + 0.6 * x1 - 0.3 * x2))))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "prov": prov})

rng = np.random.default_rng(5)
d = cohort(rng)
fit = LogisticFixedEffectModel(use_dataprep=False).fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="prov")
sm = fit.calculate_standardized_measures(stdz=["indirect", "direct"])
ind, dirc = sm["indirect"].set_index("provider_id"), sm["direct"].set_index("provider_id")
gamma, xb, idx = np.ravel(fit.coefficients_["gamma"]), fit.xbeta_, fit.provider_indices_
own = np.bincount(idx, weights=1 / (1 + np.exp(-(gamma[idx] + xb))))       # expected events at gamma_hat
print("sum_j p_kj(gamma_hat_k) = O_k for every provider:", bool(np.allclose(own, ind.observed, atol=1e-6)))
case_mix = d.groupby("prov").x1.mean()
table = pd.DataFrame({"x1 mean": case_mix, "O": ind.observed, "E": ind.expected, "indirect ratio": ind.indirect_ratio,
                      "direct ratio": dirc.direct_ratio, "direct rate %": dirc.direct_rate}).round(3)
print(table.iloc[np.argsort(case_mix.to_numpy())[[0, 1, -2, -1]]].to_string())
```

```
sum_j p_kj(gamma_hat_k) = O_k for every provider: True
    x1 mean   O       E  indirect ratio  direct ratio  direct rate %
12   -1.295  17  14.170           1.200         1.085         31.258
17   -1.128   3   3.418           0.878         0.842         24.235
26    1.079  15  20.578           0.729         0.646         18.596
28    1.937  20  20.785           0.962         0.882         25.391
```

The indirect ratio equals 1 at $\gamma_k = \gamma_0$; the direct ratio equals $H(\gamma_0)/O$ there, the reference
provider's direct ratio, which is not 1 in general. The tests of Section 3 therefore compare each measure with its own
value at the reference (`null_value="reference"`), so that every measure tests the same hypothesis
$\gamma_k = \gamma_0$.

## 3. Standard errors and tests

`test_standardized(measure=...)` forms a $z$-statistic on a working scale — identity for indirect measures, log for
direct ratios, logit for direct rates (`transform="auto"`) — and applies the null model and decision of the
[empirical-null page](empirical_null_theory.md).

**Indirect measures.** The standard error is that of $O_k$, the square root of the Poisson-binomial variance
$V_k(g) = \sum_j N_{kj}p_{kj}(g)(1 - p_{kj}(g))$, divided by $E_k$. With the variance at the reference
(`indirect_variance="null"`, the default) and the identity scale, the $z$-statistic is
$(O_k - E_k)/\sqrt{V_k(\gamma_0)}$, the score statistic of Part II:

```python
score = fit.test(test_method="score")
ind_test = fit.test_standardized(measure="indirect_ratio")
print("indirect-ratio z equals the score z:", bool(np.allclose(ind_test.z_raw, score.z_raw)))
dr_test, wald = fit.test_standardized(measure="direct_ratio"), fit.test(test_method="wald")
print(f"direct-ratio z (log scale) vs Wald z on gamma: max |difference| {np.max(np.abs(dr_test.z_raw - wald.z_raw)):.3f}; "
      f"flags differ for {int((dr_test.flag != wald.flag).sum())} of {len(wald)} providers")
```

```
indirect-ratio z equals the score z: True
direct-ratio z (log scale) vs Wald z on gamma: max |difference| 0.845; flags differ for 2 of 60 providers
```

**Direct measures.** The standard error is the delta method in $\gamma_k$,
$\text{SE}(\text{direct ratio}_k) = H'(\hat\gamma_k)\,\text{SE}_k/O$, with $\text{SE}_k$ the standard error of the
provider effect at the average case mix, model-based or cluster-robust (Part II, Sections 3.2 and 6; `variance=`). The
error in $\hat\beta$ therefore enters as it would at the average case mix; where the population's covariates are
known, this agrees closely with the full delta method in $(\gamma_k, \beta)$:

```python
beta, cov_beta = np.ravel(fit.coefficients_["beta"]), fit.variances_["beta"]
X = d[["x1", "x2"]].to_numpy()
w = fit.fitted_ * (1 - fit.fitted_)
D = np.bincount(idx, weights=w)
xbar_k = np.column_stack([np.bincount(idx, weights=w * X[:, c]) for c in range(2)]) / D[:, None]
y_total, full = d.y.sum(), []
for k, g in enumerate(gamma):                                          # the delta method in (gamma_k, beta)
    pk = 1 / (1 + np.exp(-(g + xb)))
    a, b = np.sum(pk * (1 - pk)), (pk * (1 - pk)) @ X
    r = a * xbar_k[k] - b
    full.append(np.sqrt(a**2 / D[k] + r @ cov_beta @ r) / y_total)
full = np.array(full)
se = fit.test_standardized(measure="direct_ratio", transform="identity").se.to_numpy()
print(f"direct-ratio SE / full delta method: median {np.median(se / full):.4f}, range {np.min(se / full):.4f}-{np.max(se / full):.4f}")
```

```
direct-ratio SE / full delta method: median 1.0003, range 0.9956-1.0110
```

On the log scale the test is a Wald-type test of $\gamma_k$ on a different scale: its statistics are close to, but not
equal to, the Wald statistics on $\gamma_k$ (above, they differ by up to 0.85 and 2 of 60 flags change).

**Behaviour under the null.** The choice of variance and scale matters for small providers. For providers of 15–59
records whose true effects all equal the reference, the indirect ratio's test flags at these rates:

```python
variants = {"null variance, identity": dict(), "fitted variance, identity": dict(indirect_variance="fitted"),
            "null variance, log": dict(transform="log")}
rates = {k: [] for k in variants}
for rep in range(40):
    f = LogisticFixedEffectModel(use_dataprep=False).fit(cohort(rng, m=100, sizes=(15, 60), tau=0.0),
                                                         y_var="y", x_vars=["x1", "x2"], provider_var="prov")
    for name, kw in variants.items():
        t = f.test_standardized(measure="indirect_ratio", **kw)
        rates[name].append(np.mean(t.flag.fillna(0) != 0))
print({k: round(float(np.mean(v)), 4) for k, v in rates.items()}, "(nominal 0.05; 4,000 provider tests each)")
```

```
{'null variance, identity': 0.053, 'fitted variance, identity': 0.0735, 'null variance, log': 0.0598} (nominal 0.05; 4,000 provider tests each)
```

Only the variance at the reference on the identity scale holds the nominal level: the fitted variance is small for a
provider with few observed events and the test rejects too often, and on the log scale the delta method is poor at few
events (and a provider with no events cannot be tested at all).

## 4. Confidence limits

### 4.1. By transformation of the provider-effect limits

`calculate_confidence_intervals(option="SM")` (R's `confint(option = "SM")`) computes limits $(g_L, g_U)$ for
$\gamma_k$ — by inverting the exact test (the default), the score test, or from the Wald interval (Part II, Section 5)
— and maps them through the measure's function of Section 2.2:

$$
\Big(\frac{h_k(g_L)}{E_k}, \frac{h_k(g_U)}{E_k}\Big) \;\text{(indirect ratio)}, \qquad
\Big(\frac{H(g_L)}{O}, \frac{H(g_U)}{O}\Big) \;\text{(direct ratio)},
$$

and rates by the same scaling as the estimates. Both expected counts are weighted by the binomial trials, as the
estimates are, and Wald limits for $\gamma_k$ use the normal reference, so the limits do not depend on whether the
data are stored as binomial or Bernoulli rows. A strictly increasing map carries an interval's coverage over
unchanged, so, given $\hat\beta$, the measure's interval has exactly the coverage of the interval for $\gamma_k$; and
because $h_k(\gamma_0)/E_k = 1$, the indirect ratio's interval excludes 1 exactly when the provider-effect interval
excludes $\gamma_0$ — the limits are dual to the flags of the corresponding test:

```python
ci = fit.calculate_confidence_intervals(option="SM", stdz="indirect", measure="ratio", test_method="exact")["indirect_ratio"]
exact = fit.test(test_method="poibin_exact")
excl = (ci.ci_ratio_lower > 1) | (ci.ci_ratio_upper < 1)
print("transformed exact limits exclude 1 exactly when the exact test flags:", bool((excl.to_numpy() == (exact.flag.to_numpy() != 0)).all()))
inv = fit.test_standardized(measure="indirect_ratio")
small = np.argsort(fit.provider_sizes_)[:3]
print(pd.DataFrame({"ratio": inv.estimate, "transformed exact": list(zip(ci.ci_ratio_lower.round(3), ci.ci_ratio_upper.round(3))),
                    "score inversion": list(zip(inv.ci_lower.round(3), inv.ci_upper.round(3))), "size": fit.provider_sizes_}).iloc[small].round(3).to_string())
```

```
transformed exact limits exclude 1 exactly when the exact test flags: True
             ratio transformed exact score inversion  size
provider_id
16           0.455    (0.079, 1.247)    (0.0, 1.213)    15
59           1.501    (0.692, 2.457)  (0.681, 2.321)    17
2            0.569    (0.154, 1.298)     (0.0, 1.26)    18
```

### 4.2. By inversion on the working scale

`test_standardized` inverts its own $z$-statistic on the working scale and transforms the limits back; on the identity
scale they are clipped to the measure's range (`bounds="auto"`: $[0, \infty)$ for ratios, $[0, 1]$ for rates). For
the indirect ratio this inverts the score test, whose normal approximation gives limits symmetric about the estimate
and a lower limit clipped at 0 for small providers, where the transformed exact limits of Section 4.1 stay positive;
the two agree more closely as providers grow. These limits are dual to the flags of the same call, whatever the null
model.

## 5. Linear and random-effect models

For a continuous outcome the measures are differences rather than ratios. In `LinearFixedEffectModel` the indirect
difference is provider $k$'s mean observed outcome minus its mean expected outcome at $\gamma_0$, and the direct
difference is the population's mean outcome at $\gamma_k$ minus that at $\gamma_0$; with the identity link both equal
$\hat\gamma_k - \gamma_0$, so they carry the provider effect's $t$ inference unchanged. In the random-effect models the
measures are evaluated at the predicted effects (BLUPs) with the reference $u_0$ of `reference=` (`reference=0` is R's
`SM_output.linear_re` and `logis_re`), and inherit the shrinkage of Part II, Section 7.

## 6. Survival: the standardized mortality ratio

For a provider-stratified Cox model the national baseline hazard is the Breslow estimate over all records with the
fitted linear predictors as an offset (the two-stage approach of
[survival Chapter 4](survival/04_indirect_standardization_smr_shr)). `CoxPH.calculate_standardized_measures` returns the
indirect ratio $O_k/E_k$, with $E_k$ the cumulative hazard of the provider's patients at the national baseline, and the
direct ratio $E^{(k)}/O$, with $E^{(k)}$ the whole population's expected deaths at provider $k$'s own Breslow
baseline. The national baseline makes the expected deaths add up to the observed ones, $\sum_k E_k = O$, the survival
counterpart of the identity of Section 2.2. `CoxPH.test` treats $O_k$ as Poisson with mean $E_k$ under the null:
`"midp"` (the default) is the mid-p test {cite}`inm-Lancaster1961Significance`, calibrated by any null model, with
limits by inversion; `"exact"` is the exact Poisson test with Byar's approximation or chi-square limits
{cite}`inm-Breslow1987Design`:

```python
from pprof_py import CoxPH

m_c = 40; n_c = rng.integers(30, 120, m_c); fac = np.repeat(np.arange(m_c), n_c)
z1, z2 = rng.normal(size=fac.size), rng.binomial(1, 0.4, fac.size)
hazard = 0.1 * np.exp(0.5 * z1 - 0.4 * z2 + rng.normal(0, 0.3, m_c)[fac])
t_event, t_cens = rng.exponential(1 / hazard), rng.uniform(1, 8, fac.size)
time, death = np.minimum(t_event, t_cens), (t_event <= t_cens).astype(int)
Xc = pd.DataFrame({"z1": z1, "z2": z2})
cox = CoxPH(ties="breslow").fit(Xc, duration=time, event=death, strata=fac)
smr = cox.calculate_standardized_measures(Xc, duration=time, event=death, provider_id=fac, stdz=["indirect", "direct"])
print(f"sum of expected {smr['indirect'].expected.sum():.6f} = deaths {death.sum()}")
midp, ex = (cox.test(Xc, duration=time, event=death, provider_id=fac, test_method=m) for m in ("midp", "exact"))
print(f"flagged: mid-p {int((midp.flag != 0).sum())}, exact {int((ex.flag != 0).sum())}; "
      f"correlation of indirect and direct ratios {np.corrcoef(smr['indirect'].indirect_ratio, smr['direct'].direct_ratio)[0, 1]:.3f}")
```

```
sum of expected 1042.000000 = deaths 1042
flagged: mid-p 8, exact 8; correlation of indirect and direct ratios 0.996
```

The Poisson model for $O_k$ treats the expected count as known — the baseline and $\hat\beta$ are estimated from all
providers — which is accurate when providers are many and each contributes a small share of the deaths.

## 7. What the measures' inference ignores

All the tests and limits above hold $\hat\beta$ (and, for survival, the national baseline) fixed and treat the
reference $\gamma_0$ as known. The error in each is shared by all providers, of order $1/\sqrt{\sum n}$ and
$1/\sqrt K$ respectively, and small next to one provider's sampling error when providers are many; the robust variance
of Part II, Section 6 accounts for the error in $\hat\beta$ in the direct measures' standard errors. The measures are
also only as good as the risk model: case-mix variables omitted from $x$, or recorded differently across providers,
enter $\hat\gamma_k$ and every measure built on it, and no interval reflects that. Variation of the measures across
providers beyond their sampling error is the subject of the [empirical-null page](empirical_null_theory.md).

## 8. Approximations the implementation relies on

| quantity | basis | where it can fail |
|---|---|---|
| indirect test (null variance, identity) | normal approximation to the Poisson-binomial count | few expected events (use the exact test and transformed limits) |
| indirect test (fitted variance or log scale) | the same, with the variance at $\hat\gamma_k$ or the delta method | anti-conservative for small providers; no-event providers untestable on the log scale |
| direct measures' standard errors | delta method in $\gamma_k$ at the average case mix | small providers |
| transformed limits | a monotone map of the $\gamma_k$ limits, given $\hat\beta$ | as the underlying $\gamma_k$ interval |
| survival SMR tests | $O_k$ Poisson with mean $E_k$ | error in the baseline and $\hat\beta$ |
| all | $\hat\beta$, baseline and $\gamma_0$ treated as known | few providers |

## 9. Implementation

| concept | `pprof_py` |
|---|---|
| indirect and direct ratios and rates | `LogisticFixedEffectModel.calculate_standardized_measures(stdz=..., reference=...)` |
| tests and limits on a working scale | `test_standardized(measure=..., variance=..., indirect_variance=..., transform=...)` |
| limits by transforming the $\gamma_k$ limits | `calculate_confidence_intervals(option="SM", stdz=..., measure=..., test_method=...)` |
| the building blocks | `pprof_py.inference.standardized_measure`, `z_statistic`, `provider_test` |
| linear and random-effect measures | `calculate_standardized_measures` of `LinearFixedEffectModel`, `LinearRandomEffectModel`, `LogisticRandomEffectModel` |
| survival SMR | `CoxPH.calculate_standardized_measures`, `CoxPH.test(test_method=...)` with `"midp"` or `"exact"` |

## References

```{bibliography} references.bib
:filter: docname in docnames
:keyprefix: inm-
:labelprefix: INM
```
