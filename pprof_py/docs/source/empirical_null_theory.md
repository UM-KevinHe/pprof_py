(empirical_null_theory)=
# Empirical Null Calibration of Provider Tests

This page is the theoretical reference for calibrating provider tests with an empirical null: why the
theoretical N(0, 1) null misstates the false-flag rate when providers are profiled together, what the
empirical null estimates, how `pprof_py` estimates it, and what the resulting tests and intervals do and
do not guarantee. The [empirical-null guide](empirical-null-guide) covers the API and the recommended
settings; every code block below runs as written, and every numerical claim was checked by simulation
or exact computation.

## 1. Introduction

Provider profiling tests every provider at once. For each of $K$ providers (hospitals, dialysis
facilities, clinics) a model yields a statistic comparing the provider with a reference, and the
providers whose statistics are extreme are flagged as performing better or worse than expected. Flags
carry consequences — public reporting, payment adjustments, regulatory review — so the rate at which
providers with ordinary performance are flagged matters as much as the power to detect real outliers
{cite}`enth-Kalbfleisch2013Monitoring,enth-Spiegelhalter2005Overdispersion`.

The usual reference for each statistic is its *theoretical null*: a standard normal $z$-statistic, or
the exact distribution of a count, derived under the hypothesis that the provider's effect equals the
reference and that the risk model is correct. With many providers the ensemble of statistics can be
compared with that distribution directly, and in profiling data it rarely matches: the statistics are
more dispersed than N(0, 1), often much more, because providers differ in ways the risk model does not
capture. Tested against the theoretical null, many providers of ordinary quality are then flagged.

{cite:t}`enth-Efron2004Choice` observed the same phenomenon in genomics and proposed estimating the null
distribution from the bulk of the statistics themselves — the *empirical null*. Applied to profiling
{cite}`enth-Kalbfleisch2013Monitoring`, it redefines a flag as *unusual relative to the population of
providers* rather than *different from the reference*. This page develops that method as implemented
in `pprof_py`: Section 2 sets out the statistics and the theoretical null; Section 3 derives why and by
how much the theoretical null miscalibrates; Sections 4–6 define the empirical null and analyse its
estimation, including the effects of outliers and of small groups; Section 7 covers flags and
confidence limits under a calibrated null; Section 8 relates the method to other overdispersion
corrections; Sections 9 and 10 summarise operating characteristics and limitations.

## 2. Provider statistics and the theoretical null

### 2.1. Setting

Provider $k$ has a performance parameter $\theta_k$ — a provider effect $\gamma_k$ on the logit or
linear-predictor scale, or the log of a standardized ratio — and the reference value $\theta_0$ (the
median provider effect, a random-effect mean of 0, or an SMR of 1). Each provider is tested for

$$
H_k:\ \theta_k = \theta_0, \qquad k = 1, \dots, K,
$$

and summarised by a $z$-statistic $z_k$, positive when the provider is above the reference.

### 2.2. The $z$-statistics

Every provider test in `pprof_py` reduces to a $z$-statistic before a null model is applied:

- **Wald statistics.** With estimate $\hat\theta_k$ and standard error $s_k$ on a working scale
  (identity, logit or log), $z_k = (\hat\theta_k - \theta_0)/s_k$.
- **Tail-probability statistics.** Score, exact (Poisson-binomial) and resampling tests produce
  lower- and upper-tail probabilities; the two-sided $p$-value is converted to $z_k = \pm\Phi^{-1}(p_k/2)$
  with the sign of the smaller tail.
- **Mid-p statistics for standardized ratios.** For observed and expected counts $O_k$ and $E_k$
  (for example deaths in a provider's patients and the deaths expected at the national baseline,
  the construction of [survival Chapter 4](survival/04_indirect_standardization_smr_shr)), the SMR
  tutorial's test treats $O_k \sim \text{Poisson}(E_k)$ under $H_k$ and uses the mid-p value
  {cite}`enth-Lancaster1961Significance`.

The mid-p statistic has a compact form. Write $X \sim \text{Poisson}(E)$, $p_j = P(X = j)$ and
define the *mid-distribution function*

$$
F_{\text{mid}}(O; E) = P(X < O) + \tfrac12 P(X = O).
$$

The tutorial's lower and upper mid-p quantities are $q^l = 2F_{\text{mid}}$ and
$q^u = 2(1 - F_{\text{mid}})$, its $p$-value is $\min(q^l, q^u)/2$ floored at $10^{-6}$, and its
$z$-score takes the sign of the smaller tail. Both branches give the same expression:

$$
z_k = \Phi^{-1}\!\big(F_{\text{mid}}(O_k; E_k)\big), \qquad |z_k| \le -\Phi^{-1}(10^{-6}) = 4.753.
$$

So the mid-p $z$ is the probability-integral transform of the count, with half the mass of the
observed value on each side:

```python
import numpy as np
from scipy.stats import norm, poisson
from pprof_py.inference.survival.empirical_null import poisson_midp_zscore

O, E = np.array([0, 3, 7, 12, 30]), np.array([2.5, 3.1, 4.0, 6.0, 30.0])
f_mid = poisson.cdf(O - 1, E) + 0.5 * poisson.pmf(O, E)
print(poisson_midp_zscore(O, E).round(4))
print(norm.ppf(f_mid).round(4))
```

```
[-1.7387  0.0326  1.399   2.1846  0.0302]
[-1.7387  0.0326  1.399   2.1846  0.0302]
```

### 2.3. The theoretical null

The *theoretical null* is $z_k \sim N(0, 1)$ under $H_k$ (for count statistics, the exact tail
probabilities computed under $H_k$). It rests on three assumptions: the risk model is correct, so that
$\hat\theta_k$ targets $\theta_k$; the reference $\theta_0$ is known; and the only variation in $z_k$
about its null value is the sampling variation of provider $k$'s own patients. Provider $k$ is flagged
at level $\alpha$ when $|z_k| > c = \Phi^{-1}(1 - \alpha/2)$, with the sign of $z_k$ giving the
direction. Under these assumptions each null provider is flagged with probability $\alpha$, and the
expected number of false flags is $\alpha K_0$ for $K_0$ null providers.

## 3. Why the theoretical null miscalibrates

### 3.1. Unexplained between-provider variation

Suppose that, beyond what the risk model explains, providers of ordinary quality differ by a random
amount,

$$
\theta_k = \theta_0 + u_k, \qquad u_k \sim N(0, \tau^2),
$$

with $\tau$ the between-provider standard deviation that no one would call a quality signal (coding
practice, unmeasured case mix, local circumstances). Conditionally on $u_k$, a Wald statistic is
approximately $N(u_k/s_k, 1)$, so marginally

$$
z_k \sim N\!\left(0,\; 1 + \frac{\tau^2}{s_k^2}\right).
$$

The theoretical null's false-flag rate for provider $k$ is therefore

$$
\alpha_k(\tau) = 2\,\bar\Phi\!\left(\frac{c}{\sqrt{1 + \tau^2/s_k^2}}\right),
$$

which exceeds $\alpha$ for every $\tau > 0$ and grows with the provider's size: for a logistic provider
effect, $s_k^2 \approx 1/(n_k \bar I)$ with $n_k$ records and $\bar I$ the average Bernoulli
information $E[p(1-p)]$, so $\tau^2/s_k^2 \approx \tau^2 n_k \bar I$. Large providers, whose own
sampling error is small, are the ones the theoretical null misjudges most.

The formula needs no model fit. For the logistic design of the [simulation study](empirical-null-guide)
(two standard normal covariates, logit $-1.5 + 0.4x_1 - 0.3x_2 + u_k$, sizes uniform on the stated
range), averaging $\alpha_k(\tau)$ over the size distribution reproduces the flag rates the study
obtained by fitting `LogisticFixedEffectModel` to 25 replicates of 200 providers:

| sizes | $\tau$ | predicted $\bar\alpha(\tau)$ | simulated (Wald test) |
|---|---|---|---|
| 20–400 | 0.1 | 0.087 | 0.078 |
| 20–400 | 0.2 | 0.187 | 0.188 |
| 20–400 | 0.3 | 0.299 | 0.287 |
| 10–60 | 0.1 | 0.056 | 0.046 |
| 10–60 | 0.2 | 0.075 | 0.061 |
| 10–60 | 0.3 | 0.106 | 0.091 |

For the small providers the Wald test is itself conservative (its simulated rate at $\tau = 0$ is
0.041, not 0.05); scaled by that factor the predictions are 0.046, 0.061 and 0.087. A short check of the
mechanism, with $z_k$ drawn from its marginal distribution:

```python
rng = np.random.default_rng(1)
x = rng.normal(size=(200_000, 2))
p = 1 / (1 + np.exp(-(-1.5 + x @ [0.4, -0.3])))
info = np.mean(p * (1 - p))                         # average Bernoulli information per record
n = rng.integers(20, 401, 100_000)                  # provider sizes
c = norm.ppf(0.975)
for tau in (0.1, 0.2, 0.3):
    v = 1 + tau**2 * n * info                       # Var z_k = 1 + tau^2 / s_k^2
    z = rng.normal(0, np.sqrt(v))
    print(f"tau {tau}: flagged {np.mean(np.abs(z) > c):.3f}, "
          f"formula {np.mean(2 * norm.sf(c / np.sqrt(v))):.3f}")
```

```
tau 0.1: flagged 0.087, formula 0.087
tau 0.2: flagged 0.188, formula 0.187
tau 0.3: flagged 0.297, formula 0.298
```

For an SMR the same argument applies on the count scale. With $O_k \sim \text{Poisson}(E_k e^{u_k})$,
$\operatorname{Var}(O_k) = E_k\,E[e^{u}] + E_k^2 \operatorname{Var}(e^{u})$, so the mid-p $z$, which
is approximately $(O_k - E_k)/\sqrt{E_k}$, has variance close to $1 + \tau^2 E_k$ for small $\tau$: the
expected count plays the role of $1/s_k^2$, and the grouping variable is person-time or expected
deaths.

The theoretical null is not *wrong* here: it correctly tests $\theta_k = \theta_0$, and every provider
with $u_k \ne 0$ does differ from the reference. The empirical null answers a different question —
whether provider $k$ is unusual among providers — and the choice between them is a choice of what a
flag should mean {cite}`enth-Kalbfleisch2013Monitoring,enth-Spiegelhalter2005Overdispersion`.

### 3.2. Error in the reference and in the risk adjustment

The reference is usually estimated — the median of the $\hat\gamma_k$, for instance — and so is the
risk model's $\hat\beta$. Replacing $\theta_0$ by $\hat\theta_0$ adds $-(\hat\theta_0 - \theta_0)/s_k$
to every $z_k$, a shift shared by all providers and scaled by their precision; an error in $\hat\beta$
shifts each provider by $\bar x_k^\top(\hat\beta - \beta)/s_k$, correlated across providers with similar
case mix. Both effects distort the ensemble of $z$-statistics in location and, through the scaling by
$s_k$, in spread, and both are larger for large providers. They are usually small next to $\tau$, but
they are the reason an estimated null centre $\hat\mu_0$ can differ from 0 even when no provider is
unusual.

### 3.3. Misspecification

An omitted risk factor whose prevalence differs between providers acts like a provider effect: it
contributes a term to $u_k$ that is not random noise but tracks the providers' patient populations. If
the factor is unrelated to size it inflates $\tau$; if it is related to size (large urban centres
treating sicker patients, say) it shifts the null centre differently in different size groups. The
empirical null absorbs both to the extent that they are shared by most providers in a group; it does
not identify them, and it is no substitute for a better risk model.

### 3.4. Discreteness of count statistics

For small expected counts the mid-p $z$ is not N(0, 1) even when the Poisson model holds exactly.
Because $F_{\text{mid}}(j) = \tfrac12(S_{j-1} + S_j)$ with $S_j = P(X \le j)$, its first two moments
under $X \sim \text{Poisson}(E)$ follow from telescoping sums:

$$
E\,F_{\text{mid}}(X) = \sum_j p_j\,\frac{S_{j-1} + S_j}{2} = \sum_j \frac{S_j^2 - S_{j-1}^2}{2} = \frac12,
$$

$$
E\,F_{\text{mid}}(X)^2 = \sum_j \int_{S_{j-1}}^{S_j} u^2\,du - \sum_j \frac{p_j^3}{12}
= \frac13 - \frac{\sum_j p_j^3}{12},
\qquad
\operatorname{Var} F_{\text{mid}}(X) = \frac{1 - \sum_j p_j^3}{12}.
$$

A continuous uniform has variance $1/12$; the mid-distribution transform is centred correctly but
under-dispersed by $\sum_j p_j^3/12$, which is negligible for large $E$ and substantial for small $E$.
The $z$-scale moments and the actual size of the two-sided 5% test, computed exactly:

```python
for E in (0.5, 1, 2, 5, 10, 50):
    o = np.arange(0, int(E + 12 * np.sqrt(E) + 30))
    p = poisson.pmf(o, E)
    f_mid = poisson.cdf(o - 1, E) + 0.5 * p
    z = poisson_midp_zscore(o, np.full(o.size, float(E)))
    var_z = np.sum(p * z**2) - np.sum(p * z) ** 2
    var_f = np.sum(p * f_mid**2) - np.sum(p * f_mid) ** 2
    print(f"E {E:4}: Var F_mid {var_f:.5f} = (1 - sum p^3)/12 {(1 - np.sum(p**3)) / 12:.5f}; "
          f"Var z {var_z:.3f}; size {np.sum(p[np.abs(z) > c]):.3f}")
```

```
E  0.5: Var F_mid 0.06238 = (1 - sum p^3)/12 0.06238; Var z 0.597; size 0.014
E    1: Var F_mid 0.07450 = (1 - sum p^3)/12 0.07450; Var z 0.746; size 0.019
E    2: Var F_mid 0.07927 = (1 - sum p^3)/12 0.07927; Var z 0.874; size 0.017
E    5: Var F_mid 0.08177 = (1 - sum p^3)/12 0.08177; Var z 0.961; size 0.072
E   10: Var F_mid 0.08256 = (1 - sum p^3)/12 0.08256; Var z 0.982; size 0.056
E   50: Var F_mid 0.08318 = (1 - sum p^3)/12 0.08318; Var z 0.997; size 0.047
```

Two consequences follow. Among providers with a few expected events the theoretical null is
conservative on average ($\operatorname{Var} z < 1$) while the size of any single test oscillates with
$E$ (from 0.014 to 0.072 here); and an empirical null fitted to a group of such providers will estimate
$\hat\sigma_0 < 1$ even without overdispersion, which is a correct description of that group's
statistics, not an artefact.

## 4. The two-groups model and the empirical null

{cite:t}`enth-Efron2004Choice` frames large-scale testing as a *two-groups model*: each $z_k$ comes from
the null distribution with probability $\pi_0$ or from a non-null distribution with probability
$1 - \pi_0$,

$$
f(z) = \pi_0 f_0(z) + (1 - \pi_0) f_1(z),
$$

and the empirical null takes $f_0$ to be normal with unknown centre and spread,

$$
f_0(z) = \frac{1}{\sigma_0}\,\varphi\!\left(\frac{z - \mu_0}{\sigma_0}\right).
$$

Section 3 shows why: if null providers vary by $u_k \sim N(0, \tau^2)$ and have similar $s_k$, their
statistics are $N(0, 1 + \tau^2/s^2)$, and reference and adjustment errors add a common shift. The
*calibrated* statistic

$$
z_k^\ast = \frac{z_k - \mu_0}{\sigma_0}
$$

is N(0, 1) for null providers, and tests, flags and limits are built on it (Section 7).

The model is not identified without a further assumption, since any part of $f$ could be attributed to
$f_1$. Efron's *zero assumption* is that non-null statistics are rare near the centre of $f$, so that
the central part of the histogram is essentially $\pi_0 f_0$; it requires $\pi_0$ to be large — Efron
suggests at least 0.9 {cite}`enth-Efron2007Size,enth-Efron2010LargeScale`. In profiling terms: most
providers must be of ordinary quality, and the outliers must be few and far enough out not to shape the
centre of the distribution. Section 5.4 quantifies what happens when they are not.

The two-groups model also yields the *local false discovery rate*
$\text{fdr}(z) = \pi_0 f_0(z)/f(z)$ {cite}`enth-Efron2010LargeScale`, the posterior probability that a
provider with statistic $z$ is null. `pprof_py` does not compute it; its flags are level-$\alpha$ tests
under the calibrated null, which control the per-provider false-flag rate rather than the proportion of
false flags.

## 5. Estimating the null

### 5.1. Density-based estimators

Efron's estimators work with the density of the $z$-statistics. *Central matching*
{cite}`enth-Efron2004Choice` fits a quadratic to $\log \hat f(z)$ over the central part of the
histogram; since $\log f_0(z) = \text{const} - (z - \mu_0)^2/(2\sigma_0^2)$ there, a fitted
$\beta_0 + \beta_1 z + \beta_2 z^2$ gives $\hat\sigma_0 = (-2\beta_2)^{-1/2}$ and
$\hat\mu_0 = \beta_1\hat\sigma_0^2$. The *truncated maximum-likelihood* estimator
{cite}`enth-Efron2007Size` maximises the likelihood of the statistics in a central interval $[a, b]$
under $\pi_0 f_0$, which also estimates $\pi_0$. Both need a few hundred statistics to estimate a
density; profiling has dozens to a few thousand providers, split into groups (Section 6). `pprof_py`
therefore estimates $(\mu_0, \sigma_0)$ as a robust location and scale instead.

### 5.2. M-estimation of location with a MAD scale

Treating the non-null statistics as contamination of $f_0$, the null centre and spread are a robust
location and scale of the $z_k$ {cite}`enth-Huber1964Robust,enth-Hampel1986Robust`. The estimators in
`pprof_py.inference` solve the location equation

$$
\sum_{k} \psi\!\left(\frac{z_k - \mu}{\sigma}\right) = 0,
$$

with the scale re-estimated at every iteration as the normalized median absolute deviation about the
current location,

$$
\sigma = \frac{\operatorname{med}_k |z_k - \mu|}{\Phi^{-1}(3/4)}, \qquad \Phi^{-1}(3/4) = 0.6745,
$$

by iteratively reweighted averaging, $\mu \leftarrow \sum_k w_k z_k / \sum_k w_k$ with
$w_k = \psi(r_k)/r_k$ and $r_k = (z_k - \mu)/\sigma$. This is `MASS::rlm` for an intercept-only model
with its default `scale.est = "MAD"` {cite}`enth-Venables2002Modern`, iterated until the relative change
in the residuals is below the tolerance. Two $\psi$ functions are available:

$$
\psi_{\text{H}}(r) = \max\{-k, \min(k, r)\},\ k = 1.345;
\qquad
\psi_{\text{B}}(r) = r\left(1 - (r/k)^2\right)^2 \mathbf 1\{|r| \le k\},\ k = 4.685.
$$

Both tuning constants give 95% asymptotic efficiency at the normal. Huber's $\psi$ is monotone, so its
location equation has a unique solution whatever the start; the bisquare *redescends* — statistics
beyond $k\sigma$ receive zero weight — which removes far outliers entirely but makes the equation
non-convex, so the solution can depend on the start (the mean, as in `MASS::rlm`). `HUBER_RLM` and
`BISQUARE_RLM` reproduce `MASS::rlm` at its defaults (20 iterations, tolerance $10^{-4}$), `MM_RLM` its
MM-estimator, and the default estimator is the bisquare iterated to convergence (EmpiNull's default).

The MM-estimator {cite}`enth-Yohai1987High` first computes an S-estimate of scale
{cite}`enth-Rousseeuw1984S` — the smallest $s$ solving
$\frac{1}{n-1}\sum_k \rho\big((z_k - m)/(k_0 s)\big) = \tfrac12$ over candidate centres $m$, with the
bisquare $\rho$ (scaled to a maximum of 1) and $k_0 = 1.548$, which has a 50% breakdown point, refined
by reweighting as in `MASS::lqs` — and then runs bisquare M-iterations for the location with that scale
held fixed. `pprof_py` tries every observation as a
candidate centre, which is what MASS does below 5,000 observations; above that MASS samples candidates at
random, so no exact match to it exists there.

### 5.3. Sampling variability at the null

When all $K$ statistics are N(0, 1), the M-location is asymptotically normal with variance

$$
\operatorname{Var}(\hat\mu) \approx \frac{E\,\psi(Z)^2}{\left(E\,\psi'(Z)\right)^2}\,\frac{1}{K} = \frac{1.053}{K}
$$

for Huber's $\psi$ (the reciprocal of the 95% efficiency), and the normalized MAD has

$$
\operatorname{Var}(\hat\sigma) \approx \frac{1}{4\,q^2\,(2\varphi(q))^2}\,\frac{1}{K} = \frac{1.361}{K},
\qquad q = \Phi^{-1}(3/4),
$$

an efficiency of 37% relative to the sample SD {cite}`enth-Rousseeuw1993Alternatives`. The MAD is also
biased downward in small samples. Simulating `HUBER_RLM` on N(0, 1) samples gives
$K\operatorname{Var}(\hat\mu) = 1.05$ at every size, $K\operatorname{Var}(\hat\sigma)$ from 1.20 at
$K = 12$ to 1.39 at $K = 400$, and $E\hat\sigma - 1 = -0.030, -0.018, -0.008, -0.005$ at
$K = 12, 25, 50, 100$. The scale, not the location, dominates the uncertainty of the calibration
(Section 6.2).

### 5.4. One-sided contamination

Outliers in profiling are usually on one side — providers of poor quality, flagged as worse. Let a
fraction $\varepsilon$ of the statistics come from $N(\delta, 1)$, the rest from N(0, 1). The estimator
converges to the fixed point $(\mu_\varepsilon, \sigma_\varepsilon)$ of its population equations,

$$
(1-\varepsilon)\,E\,\psi\!\left(\frac{Z - \mu}{\sigma}\right) + \varepsilon\,E\,\psi\!\left(\frac{Z + \delta - \mu}{\sigma}\right) = 0,
\qquad
(1-\varepsilon)\,P(|Z - \mu| < q\sigma) + \varepsilon\,P(|Z + \delta - \mu| < q\sigma) = \frac12,
$$

and both components move towards the outliers. For small $\varepsilon$ and far outliers the influence
functions give the first-order effects: the Huber location shifts by
$\varepsilon\,k/E\,\psi'(Z) = 1.64\,\varepsilon$ (its gross-error sensitivity) and the normalized MAD
grows by $\varepsilon/(4 q \varphi(q)) = 1.17\,\varepsilon$ {cite}`enth-Hampel1986Robust`. The flag rate
of the null providers,

$$
\alpha(\varepsilon) = \bar\Phi(c\sigma_\varepsilon - \mu_\varepsilon) + \Phi(-c\sigma_\varepsilon - \mu_\varepsilon)
\approx \alpha - 2c\varphi(c)\cdot 1.17\,\varepsilon = 0.05 - 0.27\,\varepsilon,
$$

falls, because the inflated scale dominates (the shift of the centre enters only at second order for a
two-sided test), and the power against an outlier at $\delta$,
$\bar\Phi(c\sigma_\varepsilon + \mu_\varepsilon - \delta)$, falls with both. Solving the population
equations numerically, and simulating `HUBER_RLM` on samples of 50 statistics of which
$\operatorname{round}(50\varepsilon)$ — 2 and 8 — are outliers:

| $\varepsilon$ | $\delta$ | $\mu_\varepsilon$ | $\sigma_\varepsilon$ | null flag rate | simulated $\hat\mu$ | simulated $\hat\sigma$ | simulated flag rate |
|---|---|---|---|---|---|---|---|
| 0.05 | 4 | 0.089 | 1.066 | 0.037 | 0.073 | 1.044 | 0.046 |
| 0.15 | 4 | 0.333 | 1.284 | 0.017 | 0.365 | 1.306 | 0.020 |
| 0.15 | 2 | 0.252 | 1.179 | 0.025 | 0.272 | 1.183 | 0.029 |

The simulated rates sit above the population values by about the finite-sample excess of Section 6.2
(0.006 at 50 statistics). In the simulation study's design, where an outlier's shift is
$\delta_k = 0.8/s_k$ and so depends on its size, the population values plus that excess predict null
flag rates of 0.044 and 0.024 for 5% and 15% outliers in four size groups, against the simulated 0.045
and 0.025, and power of 0.89 and 0.82 against 0.91 and 0.81.

A single sample says little (with 50 statistics the fitted scale varies by about 0.17 between samples);
averaged over samples, the estimator reproduces the table's simulated column. With the bisquare and the
MM-estimator for comparison, and outliers also at twice the distance:

```python
from pprof_py.inference import HUBER_RLM, DEFAULT_ESTIMATOR, MM_RLM

for delta in (4.0, 8.0):
    for name, estimator in (("Huber", HUBER_RLM), ("bisquare", DEFAULT_ESTIMATOR), ("MM", MM_RLM)):
        res = []
        for rep in range(1000):
            z = np.r_[rng.normal(size=42), rng.normal(delta, 1.0, 8)]   # 16% one-sided outliers
            fit = estimator(z)
            res.append((fit.location, fit.scale, np.mean(np.abs(z[:42] - fit.location) > c * fit.scale)))
        mu0, sd0, rate = np.mean(res, axis=0)
        print(f"delta {delta}: {name:8s} mu_0 {mu0:.3f}  sigma_0 {sd0:.3f}  null providers flagged {rate:.3f}")
```

```
delta 4.0: Huber    mu_0 0.365  sigma_0 1.292  null providers flagged 0.022
delta 4.0: bisquare mu_0 0.263  sigma_0 1.271  null providers flagged 0.020
delta 4.0: MM       mu_0 0.266  sigma_0 1.269  null providers flagged 0.016
delta 8.0: Huber    mu_0 0.366  sigma_0 1.311  null providers flagged 0.020
delta 8.0: bisquare mu_0 0.005  sigma_0 1.214  null providers flagged 0.022
delta 8.0: MM       mu_0 -0.003  sigma_0 1.277  null providers flagged 0.013
```

The redescending bisquare removes the far outliers' pull on the *location* — at $\delta = 8$ its centre is
back at 0 — but not on the *scale*: the MAD counts an outlier the same however far out it is (its
influence function is constant beyond $q$), so $\hat\sigma_0$ stays inflated and the null providers are
flagged at 0.013–0.022, well under half the nominal 0.05, whichever estimator is used. The MM-estimator's S-scale behaves
the same way. No location–scale estimator of this kind can separate a one-sided cluster of genuine
outliers from overdispersion: beyond roughly 10% one-sided outliers the empirical null becomes markedly
conservative, as the zero assumption warned.

## 6. Grouped nulls and small groups

### 6.1. Why group by size

Section 3.1 gives each null provider its own null variance, $1 + \tau^2/s_k^2$, which increases with
the provider's size. A single null fitted to all providers estimates a compromise spread, too wide for
the small providers and too narrow for the large ones: over the whole ensemble the flag rate still
exceeds $\alpha$ (0.064–0.067 at $\tau \ge 0.2$ in the simulation study), concentrated in the largest
providers. Fitting a separate null within groups of similar size (four quantile groups of provider size,
patient count or person-time) approximately equalises $s_k$ within each group, so that each group's null
is close to a single normal. The fitted spreads then track the random-effects prediction at each group's
typical size:

```python
from pprof_py.inference import EmpiricalNull

tau, K = 0.3, 200
sd_fit, sd_pred = [], []
for rep in range(200):
    n_k = rng.integers(20, 401, K)
    v = 1 + tau**2 * n_k * info
    null = EmpiricalNull.fit(rng.normal(0, np.sqrt(v)), size=n_k, n_groups=4, estimator=HUBER_RLM)
    sd_fit.append(null.diagnostics["null_sd"].to_numpy())
    sd_pred.append([np.sqrt(np.median(v[null.group == g])) for g in null.diagnostics["group"]])
print("fitted sd:   ", np.mean(sd_fit, axis=0).round(3))
print("predicted sd:", np.mean(sd_pred, axis=0).round(3))
```

```
fitted sd:    [1.341 1.764 2.092 2.421]
predicted sd: [1.385 1.792 2.124 2.402]
```

Here the prediction is $\sqrt{1 + \tau^2/s^2}$ at each group's median size. The residual heterogeneity
within a group is a scale mixture of normals, whose robust spread is slightly below the prediction.

### 6.2. The cost of small groups

Each group's null is estimated from that group's statistics only, and its estimation error enters every
test in the group. Write $\hat\mu$ and $\hat\sigma = 1 + \delta$ for the estimates in a group of $n$
null statistics, and let $h(\mu, \sigma) = \bar\Phi(c\sigma + \mu) + \bar\Phi(c\sigma - \mu)$ be the
probability that an *independent* N(0, 1) statistic is flagged. Expanding $h$ to second order about
$(0, 1)$ — $\partial_\sigma h = -2c\varphi(c)$, $\partial^2_\sigma h = 2c^3\varphi(c)$,
$\partial^2_\mu h = 2c\varphi(c)$ — gives the actual size

$$
\alpha' \approx \alpha + c\varphi(c)\left[c^2\operatorname{Var}\hat\sigma + \operatorname{Var}\hat\mu - 2\,(E\hat\sigma - 1)\right],
$$

which exceeds $\alpha$: an estimated scale is too small as often as too large, and the normal tail is
convex, so the errors do not cancel. The provider's own statistic is, however, part of the fit, and it
pulls the estimates towards itself: a statistic at the flagging boundary $|z| = c$ moves the Huber
location by $\text{IF}_\mu(c)/n = 1.64/n$ and the MAD scale by $\text{IF}_\sigma(c)/n = 1.17/n$
(Section 5.4), which raises its effective threshold and lowers the size by
$2\varphi(c)\,[\text{IF}_\mu(c) + c\,\text{IF}_\sigma(c)]/n$. With the moments of Section 5.3, simulation
of `HUBER_RLM` on groups of $n$ N(0, 1) statistics confirms both terms:

| $n$ | new statistic: simulated | expansion | own statistic: simulated | expansion with self-inclusion | bisquare, own statistic |
|---|---|---|---|---|---|
| 12 | 0.114 | 0.111 | 0.068 | 0.073 | 0.066 |
| 25 | 0.083 | 0.082 | 0.062 | 0.063 | 0.062 |
| 50 | 0.067 | 0.066 | 0.056 | 0.057 | 0.056 |
| 100 | 0.058 | 0.058 | 0.054 | 0.054 | 0.054 |

With the asymptotic constants ($\operatorname{Var}\hat\sigma = 1.361/n$, $\operatorname{Var}\hat\mu =
1.053/n$, $E\hat\sigma - 1 \approx -0.4/n$) the net excess of a calibrated 5% test is about $0.35/n$:
0.007 with 50 statistics per group, 0.014 with 25, and much more below that, where higher-order terms
take over. This is the trade-off that sets the number of groups: more groups reduce the size
heterogeneity of Section 6.1 but leave fewer statistics per group. The simulation study found the
four-group null liberal (up to 0.074) with 50 providers, about 12 per group, and on target with 200.

### 6.3. Options that change what is fitted

`EmpiricalNull.fit` separates the providers used to estimate a group's null from those calibrated by it:
`fit_mask` excludes providers from the estimation (for example, providers already known to be
exceptional, or those with too few records to test), and every provider in the group is still
calibrated. `common_mean` replaces each group's fitted centre by the mean of all eligible statistics
(or a given value) while keeping the group spreads, for settings where the centre should not vary with
size. A group with fewer than `min_group_size` eligible statistics either raises or falls back to the
theoretical null with a warning (`small_group`); Section 6.2 shows why the default minimum of three is a
floor for computation, not a recommendation.

## 7. Decisions under a calibrated null

### 7.1. P-values and flags

With null parameters $(\mu_0, \sigma_0)$ for provider $k$'s group, the calibrated statistic is
$z_k^\ast = (z_k - \mu_0)/\sigma_0$, its two-sided $p$-value is $2\bar\Phi(|z_k^\ast|)$, and the provider
is flagged $+1$ when $z_k^\ast > c$ and $-1$ when $z_k^\ast < -c$ (one-sided alternatives flag one
direction only). The theoretical null is the special case $(\mu_0, \sigma_0) = (0, 1)$, and
`FixedNull` supplies externally determined values.

### 7.2. Wald statistics: limits by inversion

For a Wald statistic $z_k = (T_k - t_0)/s_k$, with $T_k$ the transformed estimate and $t_0$ the
transformed reference, the calibrated test accepts a reference value $t$ exactly when
$|(T_k - t)/s_k - \mu_0| \le c\sigma_0$, that is, when

$$
t \in \big[\,T_k - (\mu_0 + c\sigma_0)\,s_k,\;\; T_k - (\mu_0 - c\sigma_0)\,s_k\,\big].
$$

This interval is the set of values the calibrated test does not reject, so it excludes $t_0$ if and only
if the provider is flagged: flags and limits are dual by construction, whatever $(\mu_0, \sigma_0)$. The
limits are shifted by $-\mu_0 s_k$ as well as widened by $\sigma_0$; the `"scale_only"` form,
$T_k \mp c\sigma_0 s_k$, widens without shifting and is dual to the flags only when $\mu_0 = 0$. With a
Student-$t$ reference the critical values $\mu_0 \pm c\sigma_0$ are mapped to the $t$ scale before
multiplying by $s_k$. Limits are transformed back to the measure's own scale. The duality holds
provider by provider:

```python
from pprof_py.inference import MeasureFrame, z_statistic, provider_test

s = 1 / np.sqrt(n_k * info)                                    # standard errors of the last replicate
theta = rng.normal(0, np.sqrt(tau**2 + s**2))                  # estimates around a reference of 0
zf = z_statistic(MeasureFrame.from_arrays(theta, s), null_value=0.0, transform="identity")
out = provider_test(zf, EmpiricalNull.fit(zf, size=n_k, n_groups=4, estimator=HUBER_RLM))
excludes = (out["ci_lower"] > 0) | (out["ci_upper"] < 0)
print(out["flag"].value_counts().sort_index().to_dict(),
      "| limits exclude 0 exactly when flagged:", bool((excludes == (out["flag"] != 0)).all()))
```

```
{np.int8(-1): 5, np.int8(0): 187, np.int8(1): 8} | limits exclude 0 exactly when flagged: True
```

### 7.3. Exact count tests

For exact and Poisson-binomial tests the $z$-statistic is a function $z_k(\theta)$ of the value tested,
computed from exact tails, and it decreases monotonically in $\theta$ (a larger provider effect makes the
observed count less extreme upwards). The acceptance set
$\{\theta : |z_k(\theta) - \mu_0| \le c\sigma_0\}$ is then an interval whose ends solve
$z_k(\theta) = \mu_0 \pm c\sigma_0$, found by root finding; the limits are again dual to the flags.
Monte Carlo tests give no limits, since their $z(\theta)$ is not a smooth function of $\theta$.

### 7.4. The mid-p test for standardized ratios

For the SMR the value tested is the Poisson mean $e$ of the provider's count, and the ratio it
corresponds to is $e/E_k$. By
$\partial S_j/\partial e = -p_j$,

$$
\frac{\partial}{\partial e} F_{\text{mid}}(O; e) = -\tfrac12\big(p_{O-1}(e) + p_O(e)\big) < 0,
$$

so $z(e) = \Phi^{-1}(F_{\text{mid}}(O; e))$ decreases strictly in $e$ until it reaches the cap
$\pm 4.753$. The calibrated $p$-value $2\bar\Phi(|z(e) - \mu_0|/\sigma_0)$ is therefore unimodal in $e$:
it rises to 1 where $z(e) = \mu_0$ and falls on either side. The limits for the mean are the two
values where it equals $\alpha$, and the limits for the SMR are those two values divided by $E_k$; the
interval excludes 1 exactly when $E_k$ lies outside the limits for the mean, that is, when the provider is
flagged. When the null is
wide enough that the $p$-value stays above $\alpha$ on one side before $z$ reaches its cap, that side
has no finite limit, and the lower limit of the mean is 0 or the upper limit infinity. This is the
inversion `CoxPH.test` performs; the SMR tutorial's case analysis (its Section 4.2.2) states the same
roots with the search ranges assumed in advance, which fails when an empirical null is very wide.

```python
from scipy.optimize import brentq

def midp_z(o, e):
    return float(poisson_midp_zscore([o], [e])[0])

o, expected, mu0, sd0 = 12, 7.0, 0.3, 1.4                      # 12 deaths, 7 expected; a group's fitted null
mode = brentq(lambda e: midp_z(o, e) - mu0, 1e-6, 100)         # where the calibrated p-value is 1
excess = lambda e: 2 * norm.sf(abs(midp_z(o, e) - mu0) / sd0) - 0.05
lower, upper = brentq(excess, 1e-6, mode), brentq(excess, mode, 200)
print(f"limits for the mean ({lower:.3f}, {upper:.3f}); SMR {o / expected:.3f} with limits "
      f"({lower / expected:.3f}, {upper / expected:.3f})")
zs = np.array([midp_z(o, e) for e in np.linspace(0.5, 60, 400)])
inside = np.abs(zs) < 4.75                                     # below the cap
print("z(e) strictly decreasing below the cap:", bool(np.all(np.diff(zs[inside]) < 0)),
      "| capped at", zs.max().round(3), "and", zs.min().round(3))
```

```
limits for the mean (4.331, 22.867); SMR 1.714 with limits (0.619, 3.267)
z(e) strictly decreasing below the cap: True | capped at 4.753 and -4.753
```

## 8. Related approaches

**A fixed null.** When the overdispersion has been estimated elsewhere — on a reference population, or a
previous reporting period — its parameters can be imposed rather than re-estimated:
`FixedNull(mean=..., sd=...)` calibrates every provider by the same $(\mu_0, \sigma_0)$. The internal
PPPW recipe's scale of 1.81 is an example. A fixed null has no estimation error (Section 6.2) and is
immune to contamination by the current outliers (Section 5.4), at the cost of assuming that the
overdispersion has not changed and does not vary with provider size.

**Multiplicative overdispersion.** {cite:t}`enth-Spiegelhalter2005Overdispersion` estimates an
overdispersion factor $\phi$ from the $I$ providers' $z$-statistics — the mean of the squared
statistics after winsorizing 10% at each end (setting them to the 10th and 90th percentiles) — and,
when $\hat\phi$ is significantly greater than 1 ($I\hat\phi$ referred to a $\chi^2_I$ distribution),
divides every statistic by $\sqrt{\hat\phi}$. This is an overall empirical null with $\mu_0 = 0$ and $\sigma_0 = \sqrt{\hat\phi}$
estimated by a winsorized moment instead of an M-estimator; like any single null it mis-scales providers
of different sizes (Section 6.1).

**Additive random-effects overdispersion.** The same paper, and Spiegelhalter's funnel plots
{cite}`enth-Spiegelhalter2005Funnel`, also model the overdispersion additively, as in Section 3.1: with
$\hat\tau^2$ estimated by a method-of-moments (DerSimonian–Laird-type) estimator, each provider is
standardized by its own total variance,

$$
z_k^{\text{RE}} = \frac{\hat\theta_k - \theta_0}{\sqrt{s_k^2 + \hat\tau^2}} = \frac{z_k}{\sqrt{1 + \hat\tau^2/s_k^2}}.
$$

This is a provider-specific null spread, $\sigma_{0k} = \sqrt{1 + \hat\tau^2/s_k^2}$, with a parametric
form for its dependence on size. The size-grouped empirical null estimates the same dependence
stepwise and without assuming the normal random-effects form; Section 6.1's comparison shows how
closely the two agree when that form holds.

**Random-effects shrinkage.** A random-effects model (`LogisticRandomEffectModel`) estimates $\tau^2$
jointly with the provider effects and shrinks each estimate towards the mean. Its tests compare the
shrunken effects with 0, which addresses a different target — the provider's effect given the
population — and inherits the random-effects model's assumption that the provider effects are
independent of the case mix {cite}`enth-Kalbfleisch2013Monitoring`. The empirical null keeps the
fixed-effect estimates and changes only the reference distribution; the two can disagree about which
providers are unusual, most often for small providers, whose estimates random effects shrink most. The
hierarchical formulation of "unusual" providers is developed by {cite:t}`enth-Jones2011Identification`.

## 9. Operating characteristics

The [empirical-null guide](empirical-null-guide) reports a simulation study of flag rates and power
for logistic fixed-effect tests and Cox SMRs under overdispersion, outliers and a range of provider
counts and sizes. Each of its findings follows from a result above:

| finding in the simulation study | explanation |
|---|---|
| the theoretical null's false-flag rate rises to 0.29 at $\tau = 0.3$, and to 0.09 for small providers | Section 3.1: $\bar\alpha(\tau)$ from $\operatorname{Var} z_k = 1 + \tau^2/s_k^2$, reproduced without fitting a model |
| a single (overall) null reaches 0.064–0.067 at $\tau \ge 0.2$; four size groups hold 0.051–0.055 | Section 6.1: the null spread depends on size; groups equalise it |
| 5% one-sided outliers give 0.035–0.045 and 15% about 0.02, with power falling to 0.46 at $\tau = 0.3$ | Section 5.4: the contaminated fixed point inflates $\sigma_0$ and shifts $\mu_0$; predicted within 0.001–0.02 |
| about 12 providers per group gives 0.054–0.074 | Section 6.2: the excess $\approx 0.35/n$ plus higher-order terms at small $n$ |
| Huber and bisquare differ by at most 0.005 | Section 5.2: both have 95% efficiency at the normal; they differ only for far outliers |
| Cox SMRs behave like the logistic tests | Sections 3.1 and 3.4: $\operatorname{Var} z_k \approx 1 + \tau^2 E_k$; discreteness is minor at the study's expected counts |

## 10. Limitations and guidance

- **A flag changes meaning.** Under the empirical null a flag says that a provider is unusual among
  providers, not that it differs from the reference. If most providers genuinely differ in quality, the
  empirical null treats that variation as ordinary and flags only the extremes; whether that is the
  intended policy is a decision for the programme, not a statistical question
  {cite}`enth-Kalbfleisch2013Monitoring,enth-Spiegelhalter2005Overdispersion`.
- **Most providers must be null.** The estimates assume that outliers are a minority and do not shape the
  centre of the distribution (Section 4). With more than about 10% one-sided outliers the null widens and
  shifts towards them, and the method under-flags (Section 5.4).
- **Enough providers per group.** With $n$ providers per group the calibrated 5% test has size about
  $0.05 + 0.35/n$ (Section 6.2): aim for 25–50 or more per group, and below about 100 providers use two
  groups or the overall null.
- **Group by the variable that drives the spread.** The null variance depends on the provider's
  precision (Section 3.1): records for logistic effects, person-time or expected events for SMRs.
- **Discreteness is real, not noise.** Groups of providers with few expected events have null spreads
  below 1 (Section 3.4); that is a correct description of their statistics.
- **The tests are per provider.** A calibrated level-$\alpha$ test controls each null provider's
  false-flag probability; it does not control the number or proportion of false flags among all
  providers. The statistics are also correlated through the shared risk model and reference
  (Section 3.2), which the null absorbs only on average.
- **Not a model check.** An empirical null that differs markedly from N(0, 1) says that the risk model
  leaves systematic variation between providers; it does not say where it comes from (Section 3.3).

## 11. Implementation

| concept | `pprof_py` |
|---|---|
| $z$-statistics | `z_statistic`, `MeasureFrame`, `ZFrame`; each model's `test()` |
| mid-p statistic for SMRs | `pprof_py.inference.survival.empirical_null.poisson_midp_zscore`; `CoxPH.test` |
| theoretical and fixed nulls | `TheoreticalNull`, `FixedNull` |
| empirical null, overall or grouped | `EmpiricalNull.fit(z, size=..., n_groups=..., grouping=..., fit_mask=..., common_mean=...)` |
| size groups | `assign_groups` (`"quantile"` or `"rank"`) |
| location and scale estimators | `robust_location_scale`; `HUBER_RLM`, `BISQUARE_RLM`, `MM_RLM`, `DEFAULT_ESTIMATOR` |
| calibration, $p$-values, flags, limits | `calibrate`, `p_values`, `flags`, `intervals` (`form="inversion"` or `"scale_only"`), `provider_test` |

## References

```{bibliography} references.bib
:filter: docname in docnames
:keyprefix: enth-
:labelprefix: ENT
```
