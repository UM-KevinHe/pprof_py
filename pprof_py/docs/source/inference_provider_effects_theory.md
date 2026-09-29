(inference_gamma_theory)=
# Inference for Provider Effects

This page is Part II of the theory series on inference in `pprof_py`. It covers the provider effects $\gamma_k$:
what a provider test compares, the Wald test and the variance it uses, the score and exact tests at the reference,
confidence limits by inversion, the cluster-robust variances, and the random-effect and three-stage counterparts.
[Part I](inference_covariate_effects_theory.md) treats the covariate effects $\beta$ and Part III the standardized
measures; the calibration of all these tests by an empirical null is the subject of the
[empirical-null page](empirical_null_theory.md). Every code block below runs as written, and every numerical claim
was checked by simulation or exact computation.

## 1. Introduction

A provider test asks whether provider $k$'s outcomes, adjusted for its patients' characteristics, differ from
those of a reference provider. In the fixed-effect model of Part I the question is about one parameter,
$H_k: \gamma_k = \gamma_0$, where the reference $\gamma_0$ is itself estimated from all providers. Three features
make the theory differ from a textbook test of one coefficient: the reference is estimated; the covariate effects
are estimated jointly and shared by all providers; and providers are often small, with a few events or none, so
large-sample approximations for a single provider can fail even when the risk model is estimated precisely.
Section 2 sets out what is tested; Section 3 the Wald test and its variance; Section 4 the tests that hold $\beta$
at $\hat\beta$ and use the provider's own event count; Section 5 confidence limits; Section 6 the cluster-robust
variances; Section 7 the random-effect and three-stage models; Section 8 the approximations relied on.

## 2. What is tested

Provider $k$'s records follow $\operatorname{logit} p_{kj} = \gamma_k + x_{kj}^\top\beta$ (Part I, Section 2).
The reference effect is

$$
\gamma_0 = \operatorname{median}_k \hat\gamma_k \quad\text{or}\quad
\gamma_0 = \frac{\sum_k n_k\hat\gamma_k}{\sum_k n_k} \quad\text{or a given value}
$$

(`reference="median"`, the default; `"mean"`; or a number), and each provider is tested for $H_k: \gamma_k = \gamma_0$ with $\gamma_0$ treated as known. Every test returns a
$z$-statistic, positive when the provider's outcomes exceed the reference; flags are $+1$ above, $-1$ below and 0
otherwise, after the null model of the [empirical-null page](empirical_null_theory.md) (by default the theoretical
N(0, 1)). The comparison $\gamma_k - \gamma_0$ — a log odds ratio between provider $k$ and the reference provider for
any patient — does not depend on where the covariates are centred: shifting $x$ by $c$ changes every $\hat\gamma_k$
by the same $-c^\top\hat\beta$ (Part I, Section 2.4), and $\gamma_0$ with them.

## 3. The Wald test

### 3.1. Statistic and variance

The Wald statistic is $z_k = (\hat\gamma_k - \gamma_0)/\widehat{\text{SE}}_k$ with the normal reference, as in R. The
inverse-information variance of $\hat\gamma_k$ (Part I, Section 2.3; `variances_["gamma"]`, R's `logis_fe_var`) is

$$
\operatorname{Var}(\hat\gamma_k) = D_k^{-1} + \bar x_k^\top S^{-1} \bar x_k, \qquad \bar x_k = B_{\cdot k}/D_k,
$$

the first term the provider's own information, the second the error in $\hat\beta$ carried at the provider's
$w$-weighted mean covariate $\bar x_k$. Section 3.2 shows why the test uses a different variance.

### 3.2. The variance the test needs

The numerator is $\hat\gamma_k - \hat\gamma_0$, and $\hat\gamma_0$ carries the error in $\hat\beta$ as every
$\hat\gamma_k$ does: to first order $\hat\gamma_k - \gamma_k \approx \varepsilon_k - \bar x_k^\top(\hat\beta - \beta)$
with $\varepsilon_k$ the provider's own error, so the $\beta$ error that enters the comparison with a reference
provider at $\bar x_0$ is $(\bar x_k - \bar x_0)^\top(\hat\beta - \beta)$. The term $\bar x_k^\top S^{-1}\bar x_k$ of
$\operatorname{Var}(\hat\gamma_k)$ is instead measured from the origin of the covariates, which is arbitrary: the
numerator does not depend on the origin, $\operatorname{Var}(\hat\gamma_k)$ does, and with covariates recorded far from
0 (an age, a calendar year) it overstates the variance of the comparison. `pprof_py` therefore uses the variance of the
provider effect at the average case mix,

$$
\operatorname{Var}(\hat\gamma_k + \bar x^\top\hat\beta) = D_k^{-1} + (\bar x_k - \bar x)^\top S^{-1} (\bar x_k - \bar x),
$$

with $\bar x$ the trials-weighted mean covariate row (`variances_["gamma_case_mix"]`): the reference is treated as a
provider of average case mix. It is the model-based counterpart of the robust variance of Section 6, and it is used by
the Wald test, the Wald limits of `calculate_confidence_intervals`, and the model-based standard errors of
`test_standardized` (Part III). Recording a covariate 3 units higher changes `variances_["gamma"]` but not the
variance at the average case mix, the Wald flags or the exact flags:

```python
import numpy as np
import pandas as pd
from pprof_py import LogisticFixedEffectModel

def cohort(rng, m=60, sizes=(15, 150), tau=0.4, offset=0.0):
    """Providers with their own covariate mean; true effects gamma_k ~ N(-1, tau^2); x1 recorded with an offset."""
    n = rng.integers(sizes[0], sizes[1], m); prov = np.repeat(np.arange(m), n)
    mu = rng.normal(0, 0.7, m); gamma = -1 + tau * rng.normal(size=m)
    x1, x2 = rng.normal(mu[prov], 1), rng.normal(size=prov.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(gamma[prov] + 0.4 * x1 - 0.3 * x2))))
    return pd.DataFrame({"y": y, "x1": x1 + offset, "x2": x2, "prov": prov})

def fit(d):
    return LogisticFixedEffectModel(use_dataprep=False).fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="prov")

rng = np.random.default_rng(3)
d = cohort(rng)
fits = {c: fit(d.assign(x1=d.x1 + c)) for c in (0.0, 3.0)}           # the same data, x1 recorded 3 units higher
for c, f in fits.items():
    se_r = np.sqrt(np.ravel(f.variances_["gamma"]))
    se_cm = np.sqrt(np.ravel(f.variances_["gamma_case_mix"]))
    wald, exact = f.test(test_method="wald"), f.test(test_method="poibin_exact")
    print(f"x1 + {c:.0f}: sqrt(variances_['gamma']) {se_r[:3].round(4)}; at the average case mix {se_cm[:3].round(4)}; "
          f"Wald flags {int((wald.flag != 0).sum())}, exact flags {int((exact.flag != 0).sum())}")
```

```
x1 + 0: sqrt(variances_['gamma']) [0.232  0.4403 0.3756]; at the average case mix [0.2318 0.4404 0.3759]; Wald flags 16, exact flags 17
x1 + 3: sqrt(variances_['gamma']) [0.2507 0.454  0.3966]; at the average case mix [0.2318 0.4404 0.3759]; Wald flags 16, exact flags 17
```

R's Wald test uses $\operatorname{Var}(\hat\gamma_k)$ and depends on the covariates' origin; the other statistics of
Section 4 and the robust variance (Section 6) do not.

### 3.3. Providers without a finite estimate

A provider with no events or only events has no maximum-likelihood estimate: its $\hat\gamma_k$ moves towards
$\mp\infty$ during the fit and stops where the covariate coefficients converge, or at the clamp
$\operatorname{median}(\hat\gamma) \pm$ `bound`. Its Wald statistic and interval are then meaningless, and `test()`
warns. The tests of Section 4 use the provider's event count, which is well defined for these providers (an
observed count of 0 is simply extreme under the null when many events are expected). `at_bound(model)` returns these
providers, and any held at the clamp, for example to leave them out of an empirical-null fit.

## 4. Tests at the reference

The score, exact and bootstrap tests evaluate the provider's observed event count $O_k = \sum_j Y_{kj}$ under
the null $\gamma_k = \gamma_0$ with $\beta = \hat\beta$: each record then has the null probability
$p^0_{kj} = \operatorname{expit}(\gamma_0 + x_{kj}^\top\hat\beta)$, and $O_k$ is a sum of independent binomials.

**Score.** $z_k = (O_k - E_k)/\sqrt{V_k}$ with $E_k = \sum_j N_{kj}p^0_{kj}$ and
$V_k = \sum_j N_{kj}p^0_{kj}(1 - p^0_{kj})$: the score for $\gamma_k$ at $\gamma_0$ divided by the square root of its
information, and the standardized difference between observed and expected events.

**Exact (`"poibin_exact"`, the default).** Under the null $O_k$ has the Poisson-binomial distribution with
probabilities $p^0_{kj}$ (each record expanded into its $N_{kj}$ trials), computed exactly. The two-sided mid-p tails
$P(O > o) + \tfrac12 P(O = o)$ and $P(O < o) + \tfrac12 P(O = o)$ are converted to $z_k$ as on the
[empirical-null page](empirical_null_theory.md) (Section 2.2), with tails floored at $10^{-300}$.

**Bootstrap (`"bootstrap_exact"`).** The same null by simulation, with `n_resample` draws (R's `exact.bootstrap`); its
tails cannot fall below $1/(2\,\texttt{n\_resample})$, which caps $|z_k|$.

All three condition on $\hat\beta$ and $\gamma_0$. Their error — of order $1/\sqrt{\sum_k n_k}$ for $\hat\beta$ and
$1/\sqrt K$ for the median — is shared by all providers and small next to a single provider's sampling error when
there are many providers, which is when these tests are used. Discreteness matters more: with a few expected events
the mid-p test is conservative on average ([empirical-null page](empirical_null_theory.md), Section 3.4), and the
Wald test's normal approximation is poor. For providers of 15–59 records whose true effects all equal the reference,
the three tests flag at these rates:

```python
rates = {"wald": [], "score": [], "poibin_exact": []}
for rep in range(40):
    dn = cohort(rng, m=100, sizes=(15, 60), tau=0.0)                  # every provider at the same effect
    f = fit(dn)
    for method in rates:
        rates[method].append(np.mean(f.test(test_method=method).flag != 0))
print({k: round(float(np.mean(v)), 4) for k, v in rates.items()}, "(nominal 0.05; 4,000 provider tests each)")
```

```
{'wald': 0.0393, 'score': 0.0485, 'poibin_exact': 0.0452} (nominal 0.05; 4,000 provider tests each)
```

The exact and score tests are close to the nominal level; the Wald test is conservative for small providers, where
the curvature of the likelihood at $\hat\gamma_k$ overstates the standard error for extreme estimates.

## 5. Confidence limits

**Wald limits** are $\hat\gamma_k \pm c\,\widehat{\text{SE}}(\hat\gamma_k)$, with $c$ adjusted for a calibrated null
([empirical-null page](empirical_null_theory.md), Section 7.2).

**Exact limits** invert the exact test. Holding $\beta$ at $\hat\beta$, provider $k$'s count distribution under
$\gamma_k = g$ is Poisson-binomial with probabilities $\operatorname{expit}(g + x_{kj}^\top\hat\beta)$, which increase
with $g$, so the mid-p statistic $z_k(g)$ decreases strictly in $g$. The limits are the two solutions of
$z_k(g) = \mu_0 \pm c\,\sigma_0$, found by root finding; the interval contains $\gamma_0$ exactly when the provider is
not flagged, whatever the null. `calculate_confidence_intervals(option="gamma", test_method="exact")` returns the same
limits. They are asymmetric for small providers, where the likelihood is far from quadratic, and agree with the Wald
limits for large ones:

```python
f = fits[0.0]
g0 = np.median(np.ravel(f.coefficients_["gamma"]))
ex, wa = f.test(test_method="poibin_exact"), f.test(test_method="wald")
excludes = (ex.ci_lower > g0) | (ex.ci_upper < g0)
print("exact limits exclude gamma_0 exactly when flagged:", bool((excludes == (ex.flag != 0)).all()))
show = pd.DataFrame({"estimate": ex.estimate, "exact lower": ex.ci_lower, "exact upper": ex.ci_upper,
                     "Wald lower": wa.ci_lower, "Wald upper": wa.ci_upper, "size": f.provider_sizes_}).round(3)
print(show.iloc[np.argsort(f.provider_sizes_)[[0, 1, -2, -1]]].to_string())
```

```
exact limits exclude gamma_0 exactly when flagged: True
             estimate  exact lower  exact upper  Wald lower  Wald upper  size
provider_id
41             -2.740       -5.849       -0.944      -4.813      -0.667    15
18             -0.571       -1.618        0.415      -1.556       0.414    19
31             -1.361       -1.780       -0.964      -1.768      -0.954   144
43             -1.524       -2.015       -1.070      -1.995      -1.053   146
```

Limits for the standardized measures follow by transforming these limits (Part III).

## 6. Cluster-robust variances

With repeated records per patient (`obs_id_var`), `test_standardized(measure="gamma", variance=...)` uses a
cluster-robust variance of the provider effect in the Wald statistic. `variance="robust"` is the sandwich of the joint
$(\gamma, \beta)$ fit for $\gamma_k + \bar x^\top\beta$, the provider effect at the average case mix $\bar x$, which
accounts for the error in $\hat\beta$ as it enters the comparison with the reference and does not depend on the
covariates' origin (the robust counterpart of Section 3.2). `variance="robust_fixed_beta"` is R's `test_aoh`: the
sandwich with $\beta$ known, $(1/D_k)^2 M_{kk}$, which understates the variance of providers whose case mix differs
from the rest. The sandwich estimates are consistent as the number of patients grows; no small-sample correction is
applied.

## 7. Random-effect and three-stage models

**Random effects.** `LogisticRandomEffectModel` treats the provider effects as $N(0, \sigma^2)$ deviations from an
intercept and estimates them by their conditional modes (BLUPs). To first order the BLUP is the fixed-effect deviation
shrunk by the provider's reliability, $\hat b_k \approx \rho_k(\hat\gamma_k - \tilde\mu)$ with
$\rho_k = \sigma^2/(\sigma^2 + \operatorname{Var}\hat\gamma_k)$ — the quantity of the
[inter-unit reliability page](iur_theory) — so small providers are pulled towards the mean:

```python
from pprof_py import LogisticRandomEffectModel

re = LogisticRandomEffectModel().fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="prov")
b = re.coefficients_["alpha"]["prov"].to_numpy()                        # BLUPs of the provider effects
g, se2 = np.ravel(fits[0.0].coefficients_["gamma"]), np.ravel(fits[0.0].variances_["gamma"])
s2 = re.sigma_["prov"] ** 2
rho = s2 / (s2 + se2)                                                   # each provider's reliability
shrunk = rho * (g - np.average(g, weights=1 / (s2 + se2)))
print(f"sigma_hat {re.sigma_['prov']:.3f}; BLUP vs rho_k (gamma_k - mean): correlation {np.corrcoef(b, shrunk)[0, 1]:.4f}, "
      f"mean |difference| {np.mean(np.abs(b - shrunk)):.4f} (SD of the BLUPs {b.std():.3f})")
print("flags: RE Wald", int((re.test(test_method="wald").flag != 0).sum()),
      "| FE exact", int((fits[0.0].test(test_method="poibin_exact").flag != 0).sum()))
```

```
sigma_hat 0.429; BLUP vs rho_k (gamma_k - mean): correlation 0.9985, mean |difference| 0.0125 (SD of the BLUPs 0.352)
flags: RE Wald 15 | FE exact 17
```

`test()` compares $\hat b_k$ with the reference 0 (R's convention): `"wald"` (the default) divides by the posterior
standard deviation; `"exact"` (for a model with one other cluster factor) computes the count's distribution with
each cluster's effect drawn once from its posterior and shared by the provider's records in that cluster;
`"poibin_exact"` fixes the other effects at their posterior means; `"resampling"` draws them for each record. Shrinkage answers a
different question from the fixed-effect test — the provider's effect given what is known about providers in
general — and requires the effects to be independent of the case mix {cite}`ing-Kalbfleisch2013Monitoring`.

**Three-stage model.** Stage 3 of `LogisticThreeStageModel` estimates each facility's effect $\gamma_k$ with $\beta$
from Stage 1 and the hospital-effect SD $\sigma$ from Stage 2 held fixed {cite}`ing-He2013Evaluating`, so its tests
condition on both. Its `"exact"` test (the default) draws one hospital effect per hospital, shared by the facility's
patients there, from its posterior given Stage 2, and computes the count's distribution exactly by Gauss–Hermite
quadrature over each hospital's effect and convolution; `"poibin_exact"` fixes the hospital effects at their posterior
means (the plug-in test, which ignores their uncertainty); `"resampling"` draws a hospital effect for every patient,
as R's `summary.glmm.fac`. `sigma_sensitivity` reports how the flags change across the profile
interval of $\sigma$ (the [three-stage chapter](logistic/logistic_three_stage_model)).

## 8. Approximations the implementation relies on

| test or quantity | basis | where it can fail |
|---|---|---|
| Wald, variance at the average case mix | large-sample normality of $\hat\gamma_k$ | small providers (conservative); no-event or all-event providers |
| score | normal approximation to the count at $\gamma_0$ | few expected events |
| exact, bootstrap | exact (or simulated) given $\hat\beta$ and $\gamma_0$ | error in $\hat\beta$ and $\gamma_0$, shared by all providers; the bootstrap's floor |
| exact limits | inversion given $\hat\beta$ | as the exact test |
| robust variances | many patients per provider | few clusters |
| random-effect tests | normal random effects independent of the case mix; $\hat\sigma$ plugged in | confounding of provider effects and case mix |
| three-stage tests | $\beta$ and $\sigma$ from Stages 1–2 plugged in | error in $\sigma$ (`sigma_sensitivity`) |
| flags | the null model (theoretical or empirical) | overdispersion ([empirical-null page](empirical_null_theory.md)) |

## 9. Implementation

| concept | `pprof_py` |
|---|---|
| reference $\gamma_0$ | `test(reference=...)`: `"median"`, `"mean"` or a value |
| Wald, score, exact, bootstrap tests | `LogisticFixedEffectModel.test(test_method=...)` with `"wald"`, `"score"`, `"poibin_exact"`, `"bootstrap_exact"` |
| exact limits | `test(test_method="poibin_exact")` columns `ci_lower`, `ci_upper`; `calculate_confidence_intervals(option="gamma", test_method="exact")` |
| robust variances | `test_standardized(measure="gamma", variance=...)`: `"robust"`, `"robust_fixed_beta"` |
| random-effect tests | `LogisticRandomEffectModel.test(test_method=...)` with `"wald"`, `"exact"`, `"poibin_exact"`, `"resampling"` |
| three-stage tests | `LogisticFERandomClusterModel.test(test_method=...)` with `"exact"`, `"poibin_exact"`, `"resampling"`; `LogisticThreeStageModel.sigma_sensitivity` |
| calibration | `null_model=` ([empirical-null page](empirical_null_theory.md)) |

## References

```{bibliography} references.bib
:filter: docname in docnames
:keyprefix: ing-
:labelprefix: ING
```
