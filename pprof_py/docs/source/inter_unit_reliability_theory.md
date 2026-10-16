(iur_theory)=
# Inter-Unit Reliability: Theory

This page is the theoretical reference for the inter-unit reliability (IUR) of a provider-level measure:
what reliability means for a provider measure, what the IUR estimates, how `pprof_py` estimates it from
patient-level data or from provider estimates and standard errors, how precise those estimates are, and
what the split-half variants measure. The [IUR guide](inter-unit-reliability-guide) covers the API; every
code block below runs as written, and every numerical claim was checked by simulation or exact
computation. The notation follows the [empirical-null theory](empirical_null_theory.md) page: providers are
indexed by $k = 1, \dots, K$.

## 1. Introduction

A provider measure — a standardized mortality ratio, a readmission rate, a risk-adjusted provider effect
— differs between providers for two reasons: providers differ in the quantity the measure targets, and
each provider's value is estimated from a finite number of patients. A measure used to compare
providers is useful to the extent that its between-provider differences reflect the first source rather
than the second. The *reliability* of a provider's value is the share of its variance due to the first
source; the *inter-unit reliability* is that share for a typical provider, estimated from the ensemble of
providers by a one-way analysis of variance {cite}`iurt-Searle2006Variance`.

For a measure that is a mean of patient outcomes the analysis of variance applies directly. Most
profiling measures are not means — an observed-to-expected ratio divides two sums, a model-based
effect solves an estimating equation — and {cite:t}`iurt-He2019Interunit` combined the analysis of
variance with a within-provider bootstrap to estimate the within-provider variance of such measures; this
is the estimator of dialysis-facility measures reported to CMS {cite}`iurt-UMKECC2022SMoSR`. The IUR
has known limitations as a summary of a measure's usefulness for profiling
{cite}`iurt-Kalbfleisch2018Does,iurt-He2020Profile`, taken up in Section 7.

Section 2 defines the reliability of a provider value and the IUR; Section 3 derives the analysis-of-
variance estimator and its bootstrap form as implemented; Section 4 treats reliability as a function of
provider size, including the per-provider values; Section 5 gives the estimator's sampling properties and
an approximate interval; Section 6 analyses the split-half estimators; Section 7 discusses
interpretation and limitations; Section 8 maps the theory to the implementation.

## 2. Reliability of a provider measure

### 2.1. Setting

Provider $k$ has $n_k$ patients (records), $N = \sum_k n_k$, and a measure $T_k$ computed from them. Write

$$
T_k = \theta_k + \varepsilon_k, \qquad E(\varepsilon_k \mid \theta_k) = 0, \qquad
\operatorname{Var}(\varepsilon_k \mid \theta_k) = v_k,
$$

where $\theta_k$ is the value the measure would take with unlimited patients from provider $k$'s
population — its *true* value — and $\varepsilon_k$ is sampling error. Across providers the true values
vary with mean $\mu$ and variance

$$
\sigma_b^2 = \operatorname{Var}(\theta_k),
$$

the *between-provider variance* (the signal). For a mean of patient outcomes with patient-level variance
$\sigma_w^2$, $v_k = \sigma_w^2/n_k$; for other measures $v_k$ is the measure's sampling variance and
$\sigma_w^2$ is defined by $v_k \approx \sigma_w^2/n_k$ (for an observed-to-expected ratio with
Poisson counts, $v_k = \theta_k/E_k$ with $E_k$ the expected count, so $\sigma_w^2$ is proportional to
the expected events per patient).

### 2.2. Provider reliability

The *reliability* of provider $k$'s value is the share of its variance due to the signal:

$$
\rho_k = \frac{\sigma_b^2}{\sigma_b^2 + v_k}.
$$

It has two equivalent readings. Since $\operatorname{Cov}(T_k, \theta_k) = \sigma_b^2$ and
$\operatorname{Var}(T_k) = \sigma_b^2 + v_k$,

$$
\operatorname{Corr}(T_k, \theta_k)^2 = \frac{\sigma_b^4}{\sigma_b^2(\sigma_b^2 + v_k)} = \rho_k,
$$

and the least-squares predictor of $\theta_k$ from $T_k$ is $\mu + \rho_k (T_k - \mu)$: the reliability
is the squared correlation between the measure and the true value, and the factor by which a
random-effects model shrinks the measure towards the mean. Both readings hold whatever the distribution
of $\theta_k$ and $\varepsilon_k$:

```python
import numpy as np
from scipy.stats import norm, chi2

rng = np.random.default_rng(1)
sigma2_b, v = 0.04, np.array([0.01, 0.04, 0.16])      # between-provider variance; three noise levels
theta = rng.normal(0, np.sqrt(sigma2_b), 400_000)
for v_i in v:
    T = theta + rng.normal(0, np.sqrt(v_i), theta.size)
    slope = np.cov(theta, T)[0, 1] / T.var()           # least-squares slope of theta on T
    print(f"v {v_i:.2f}: rho {sigma2_b / (sigma2_b + v_i):.3f} | corr(T, theta)^2 {np.corrcoef(T, theta)[0, 1]**2:.3f}"
          f" | slope of theta on T {slope:.3f}")
```

```
v 0.01: rho 0.800 | corr(T, theta)^2 0.800 | slope of theta on T 0.800
v 0.04: rho 0.500 | corr(T, theta)^2 0.499 | slope of theta on T 0.500
v 0.16: rho 0.200 | corr(T, theta)^2 0.200 | slope of theta on T 0.200
```

Reliability is a property of a provider of a given size in a given population of providers: it rises
with $n_k$, through $v_k$, and with the spread $\sigma_b^2$ of the true values.

### 2.3. The inter-unit reliability

With $v_k = \sigma_w^2/n_k$ the reliability is a function of size,

$$
\rho(n) = \frac{\sigma_b^2}{\sigma_b^2 + \sigma_w^2/n},
$$

and the *inter-unit reliability* is its value at the effective provider size $n'$ of the analysis of
variance (Section 3.1) {cite}`iurt-He2020Profile`:

$$
\text{IUR} = \rho(n') = \frac{\sigma_b^2}{\sigma_b^2 + \sigma_w^2/n'}, \qquad
n' = \frac{1}{K - 1}\left(N - \frac{\sum_k n_k^2}{N}\right).
$$

$n'$ equals the common size when all providers have the same size and is below the mean size otherwise.

### 2.4. Assumptions

The derivations below use four assumptions. (i) Providers are independent, and the true values
$\theta_k$ are a sample from a population of providers with variance $\sigma_b^2$. (ii) The sampling
error has mean zero given $\theta_k$: the measure is unbiased for its target, which for a ratio holds to
first order in $1/E_k$. (iii) The noise variance depends on the provider through its size,
$v_k \approx \sigma_w^2/n_k$; for ratio measures $v_k$ also depends on $\theta_k$ and on the case mix
through $E_k$, and $\sigma_w^2$ is then an average. (iv) The expected values entering the measure (the
risk model) are treated as fixed; their estimation error, shared by all providers, is not part of $v_k$.

## 3. Estimation by one-way analysis of variance

### 3.1. Measures that are means

For a mean of patient outcomes, the weighted between-provider sum of squares has expectation
{cite}`iurt-Searle2006Variance`

$$
E\sum_k n_k (T_k - \bar T)^2 = (K - 1)\,\sigma_w^2 + \Big(N - \frac{\sum_k n_k^2}{N}\Big)\sigma_b^2,
\qquad \bar T = \frac{\sum_k n_k T_k}{N},
$$

so the *total variance*

$$
s_t^2 = \frac{1}{n'(K - 1)}\sum_k n_k (T_k - \bar T)^2
$$

estimates $\sigma_b^2 + \sigma_w^2/n'$, the variance of a provider of size $n'$. The within-provider
variance of the measure is estimated by pooling per-provider estimates $\hat v_k$ with weights
$n_k - 1$:

$$
s_{t,w}^2 = \frac{\sum_k (n_k - 1)\,\hat v_k}{\sum_k (n_k - 1)}.
$$

With $\hat v_k$ unbiased for $\sigma_w^2/n_k$, $E\,s_{t,w}^2 = \sigma_w^2 (K - \sum_k 1/n_k)/(N - K)$,
which equals $\sigma_w^2/n'$ when all sizes are equal. The estimates of the signal and of the IUR are

$$
s_b^2 = s_t^2 - s_{t,w}^2, \qquad \widehat{\text{IUR}} = \frac{s_t^2 - s_{t,w}^2}{s_t^2},
$$

and with unequal sizes $s_b^2$ carries the bias
$\sigma_w^2\,[\,1/n' - (K - \sum_k 1/n_k)/(N - K)\,]$, which vanishes for equal sizes and is small
otherwise: for sizes spread over two orders of magnitude it is about 1% of $\sigma_b^2$ here:

```python
def anova_parts(sizes, T, within_var):
    """s_t^2, the pooled within variance and n' exactly as pprof_py.measures.iur computes them."""
    N, K = sizes.sum(), sizes.size
    n_prime = (N - (sizes**2).sum() / N) / (K - 1)
    t_bar = (sizes * T).sum() / N
    s2_t = (sizes * (T - t_bar) ** 2).sum() / (n_prime * (K - 1))
    s2_tw = ((sizes - 1) * within_var).sum() / (sizes - 1).sum()
    return s2_t, s2_tw, n_prime

K, sigma2_b, sigma2_w = 100, 0.04, 1.0
sizes = np.round(np.exp(rng.uniform(np.log(5), np.log(500), K))).astype(int)   # sizes 5-500, log-uniform
prov = np.repeat(np.arange(K), sizes)
res = []
for rep in range(2000):
    y = rng.normal(0, np.sqrt(sigma2_b), K)[prov] + rng.normal(0, np.sqrt(sigma2_w), prov.size)
    means = np.bincount(prov, weights=y) / sizes
    s2_within = np.bincount(prov, weights=(y - means[prov]) ** 2) / (sizes - 1)   # sample variance per provider
    res.append(anova_parts(sizes, means, s2_within / sizes))                       # within variance of the mean
s2_t, s2_tw, n_prime = np.mean(res, axis=0)
mc_se = np.std([r[0] - r[1] for r in res], ddof=1) / np.sqrt(len(res))
N = sizes.sum()
print(f"n' = {n_prime:.1f}; mean s_t^2 {s2_t:.5f} vs sigma_b^2 + sigma_w^2/n' {sigma2_b + sigma2_w / n_prime:.5f}")
print(f"mean s_tw^2 {s2_tw:.5f} vs sigma_w^2 (K - sum 1/n)/(N - K) {sigma2_w * (K - (1 / sizes).sum()) / (N - K):.5f}"
      f" and sigma_w^2/n' {sigma2_w / n_prime:.5f}")
print(f"bias of s_b^2 = s_t^2 - s_tw^2: {s2_t - s2_tw - sigma2_b:+.5f}; "
      f"predicted {sigma2_w * (1 / n_prime - (K - (1 / sizes).sum()) / (N - K)):+.5f} (Monte Carlo se {mc_se:.5f})")
```

```
n' = 119.1; mean s_t^2 0.04854 vs sigma_b^2 + sigma_w^2/n' 0.04840
mean s_tw^2 0.00805 vs sigma_w^2 (K - sum 1/n)/(N - K) 0.00805 and sigma_w^2/n' 0.00840
bias of s_b^2 = s_t^2 - s_tw^2: +0.00049; predicted +0.00034 (Monte Carlo se 0.00021)
```

### 3.2. Measures that are not means: the bootstrap

For a ratio or a model-based measure there is no within-provider sum of squares.
{cite:t}`iurt-He2019Interunit` estimate each provider's sampling variance by resampling: draw $n_k$
records with replacement from provider $k$'s records, recompute the measure, repeat $B$ times, and take
the sample variance $S_k^{*2}$ of the $B$ replicates as $\hat v_k$. The pooled bootstrap variance

$$
s_{t,w}^2 = \frac{\sum_k (n_k - 1)\,S_k^{*2}}{\sum_k (n_k - 1)}
$$

then enters the decomposition of Section 3.1 unchanged {cite}`iurt-UMKECC2022SMoSR`. Two properties of
the bootstrap variance matter. For a mean it estimates $\hat\sigma^2/n_k$ with the divisor-$n_k$ variance,
a factor $(n_k - 1)/n_k$ below the unbiased value, which is negligible above a few dozen records. And
resampling records with the expected values attached, holding the risk model fixed, estimates the
variance of the measure given the risk model (assumption (iv)).

### 3.3. The estimator in `pprof_py`

`BootstrapIUR.fit(obs, exp, groups)` sorts the records by provider, computes the measure
(`measure_fn`, by default `ratio_measure`: $\sum O / \sum E$ per provider), draws `n_boot` stratified
bootstrap samples, and applies the decomposition to the replicates' variances:

- `iur_` $= (s_t^2 - s_{t,w}^2)/s_t^2$, or 0 when $s_t^2 = 0$;
- `s2_between_` $= s_b^2 = s_t^2 - s_{t,w}^2$;
- `s2_within_` $= n'\,s_{t,w}^2$, an estimate of the patient-level $\sigma_w^2$;
- `n_prime_` $= n'$; `measure_` and `measure_bootstrap_` hold the measure and its replicates.

On a simulated cohort of standardized ratios — 300 providers of 20–400 patients, observed events Poisson
with mean $e_{ij} R_k$ for expected events $e_{ij}$ (mean 0.1) and true ratio $R_k = e^{u_k}$,
$u_k \sim N(0, 0.25^2)$ — the fitted values equal the formulas exactly, and the IUR is close to the
true reliability at size $n'$:

```python
from pprof_py.measures.iur import BootstrapIUR, DirectIUR, SplitHalfIUR, ratio_measure

K, tau = 300, 0.25                                     # 300 providers; log-ratio SD 0.25
n = rng.integers(20, 401, K)                           # patients per provider
prov = np.repeat(np.arange(K), n)
e = rng.gamma(2.0, 0.05, prov.size)                    # expected events per patient (mean 0.1)
R = np.exp(rng.normal(0, tau, K))                      # true standardized ratios
O = rng.poisson(e * R[prov])                           # observed events
E = np.bincount(prov, weights=e)
var_R, mean_R = np.exp(tau**2) * (np.exp(tau**2) - 1), np.exp(tau**2 / 2)
rho = var_R / (var_R + mean_R / E)                     # true reliability of each provider's O/E

boot = BootstrapIUR(n_boot=100, seed=7).fit(O, e, prov)
within = boot.measure_bootstrap_.var(axis=1, ddof=1)
s2_t, s2_tw, n_prime = anova_parts(boot.group_sizes_, boot.measure_, within)
print(f"iur_ {boot.iur_:.4f} = (s_t^2 - s_tw^2)/s_t^2 {(s2_t - s2_tw) / s2_t:.4f}; "
      f"identical: {boot.iur_ == (s2_t - s2_tw) / s2_t}")
print(f"s2_between_ == s_t^2 - s_tw^2: {boot.s2_between_ == s2_t - s2_tw}; "
      f"s2_within_ == n' s_tw^2: {boot.s2_within_ == s2_tw * n_prime}; n_prime_ {boot.n_prime_:.2f}")
print(f"true reliability at size n' {var_R / (var_R + mean_R / (0.1 * n_prime)):.4f}")
```

```
iur_ 0.5412 = (s_t^2 - s_tw^2)/s_t^2 0.5412; identical: True
s2_between_ == s_t^2 - s_tw^2: True; s2_within_ == n' s_tw^2: True; n_prime_ 212.71
true reliability at size n' 0.5860
```

In this model the true ratio's variance is $\sigma_b^2 = e^{\tau^2}(e^{\tau^2} - 1)$ and the noise of a
provider with expected count $E_k$ is $E(R)/E_k$, so $\rho_k = \sigma_b^2/(\sigma_b^2 + E(R)/E_k)$ — the
array `rho` used below.

### 3.4. Direct IUR from estimates and standard errors

`DirectIUR.fit(sizes, estimates, standard_errors)` applies the same decomposition with $\hat v_k$ the
squared standard error, for measures whose sampling variance is available analytically (a provider
effect with its model-based or robust variance, or a ratio with the Poisson variance $T_k/E_k$). The
estimate is then free of Monte Carlo error, and its quality rests on the standard errors: any provider
whose standard error is unreliable — an effect at a bound, a ratio with a handful of expected events —
enters $s_{t,w}^2$ with weight $n_k - 1$ (Section 7).

## 4. Reliability by provider size

### 4.1. The reliability curve

The decomposition estimates the two components of $\rho(n)$: $\hat\sigma_b^2 = s_b^2$ and
$\hat\sigma_w^2 = n' s_{t,w}^2$ (`s2_within_`). The estimated reliability of a provider of size $n$ is
therefore

$$
\hat\rho(n) = \frac{s_b^2}{s_b^2 + n' s_{t,w}^2 / n},
$$

which is what `decile_table()` reports at the smallest size, the mean size of each size decile, and the
largest size. It answers the question the IUR alone does not: how large a provider must be before its
value is reliable.

### 4.2. Per-provider values

The same curve evaluated at each provider's own size, $\hat\rho(n_k)$, estimates $\rho_k$; `BootstrapIUR`
returns it as `iur_groups_`. R's `IUR_bootdata`, whose overall decomposition `pprof_py` reproduces
exactly, computes its facility-level values as

$$
\frac{s_b^2}{s_b^2 + s_{t,w}^2 / n_k},
$$

which divides the pooled within variance *of the measure*, already of order $\sigma_w^2/n'$, by $n_k$ a
second time: each provider's noise is understated by the factor $n'$, and every provider appears almost
perfectly reliable. `pprof_py` returned that form until the correction recorded in the changelog. On the
cohort above:

```python
r_form = boot.s2_between_ / (boot.s2_between_ + boot.s2_within_ / (boot.n_prime_ * boot.group_sizes_))  # R's IUR.fac
for name, r in (("iur_groups_", boot.iur_groups_), ("R's IUR.fac form", r_form), ("true rho_i", rho)):
    print(f"{name:18s} mean {r.mean():.3f}  range {r.min():.3f}-{r.max():.3f}  mean |r - rho| {np.mean(np.abs(r - rho)):.3f}")
breaks = np.unique(np.quantile(n, np.linspace(0, 1, 11)))           # the size bins decile_table() uses
labels = np.clip(np.digitize(n, breaks[1:-1], right=True) + 1, 1, len(breaks) - 1)
reps = np.r_[n.min(), [n[labels == g].mean() for g in np.unique(labels)], n.max()]
print("decile_table():     ", boot.decile_table().to_numpy().ravel()[[0, 1, 5, 10, 11]].round(3))
print("true rho at sizes:  ", (var_R / (var_R + mean_R / (0.1 * reps)))[[0, 1, 5, 10, 11]].round(3))
```

```
iur_groups_        mean 0.496  range 0.104-0.689  mean |r - rho| 0.040
R's IUR.fac form   mean 0.993  range 0.961-0.998  mean |r - rho| 0.457
true rho_i         mean 0.536  range 0.109-0.739  mean |r - rho| 0.000
decile_table():      [0.104 0.162 0.522 0.681 0.689]
true rho at sizes:   [0.123 0.189 0.567 0.719 0.727]
```

`iur_groups_` and the decile table track the true reliabilities; in this sample $s_b^2$ is below
$\sigma_b^2$ ($\widehat{\text{IUR}} = 0.541$ against 0.586 at $n'$), and every value derived from it is
correspondingly a little low. The R form does not track them.

### 4.3. Stratified IUR

`stratified_iur()` repeats the whole decomposition within subgroups of providers (by default size
deciles). Each subgroup then has its own $s_b^2$ and $n'$: the result estimates the IUR of the subgroup
regarded as a population of providers, not the reliability at the subgroup's size, which is
$\hat\rho(n)$ from the full-sample components. With $K_g$ providers in a subgroup the estimate has the
imprecision of Section 5 with $K$ replaced by $K_g$, so ten subgroups of a few dozen providers give
values that vary widely and can fall far below zero.

## 5. Sampling properties and inference

### 5.1. Negative estimates

$s_b^2$ is a difference of two estimates and is negative whenever the between-provider spread of the
observed values falls short of the pooled noise. When $\sigma_b^2 = 0$ this happens about half the time;
`pprof_py` reports the estimate as it is, without truncation at 0. The reliability curve then has no
interpretation: $\hat\rho(n)$ is negative for $n < n' s_{t,w}^2/|s_b^2|$ and exceeds 1 above it, and
neither `decile_table()` nor `iur_groups_` truncates it.

### 5.2. An approximate distribution and interval

For a balanced layout with normal errors, $(K - 1)\,s_t^2 / (\sigma_b^2 + \sigma_w^2/n')$ has a
$\chi^2_{K-1}$ distribution {cite}`iurt-Searle2006Variance`, and the pooled within variance, estimated from
all $N - K$ within-provider degrees of freedom (or from all bootstrap replicates), is far more precise.
Treating $s_{t,w}^2$ as fixed at $\sigma_w^2/n'$,

$$
1 - \widehat{\text{IUR}} = \frac{s_{t,w}^2}{s_t^2} \approx (1 - \text{IUR})\,\frac{K - 1}{\chi^2_{K-1}},
$$

so that, from the moments of the inverse chi-square,

$$
E\,\widehat{\text{IUR}} \approx 1 - (1 - \text{IUR})\frac{K - 1}{K - 3}, \qquad
\operatorname{SD}(\widehat{\text{IUR}}) \approx (1 - \text{IUR})\,\frac{K - 1}{K - 3}\sqrt{\frac{2}{K - 5}},
$$

and a $1 - \alpha$ interval for the IUR is

$$
\Big[\,1 - (1 - \widehat{\text{IUR}})\,\frac{\chi^2_{K-1,\,1-\alpha/2}}{K - 1},\;\;
1 - (1 - \widehat{\text{IUR}})\,\frac{\chi^2_{K-1,\,\alpha/2}}{K - 1}\,\Big].
$$

`pprof_py` does not compute this interval. For 60 providers of unequal size (10–199 records), 1,000
replicates at $\sigma_b^2 = 0$ and at $\sigma_b^2 = 0.02$:

```python
res = {0.0: [], 0.02: []}
sz = rng.integers(10, 200, 60)                         # 60 providers of 10-199 records
pv = np.repeat(np.arange(60), sz)
for s2b in res:
    for rep in range(1000):
        y = rng.normal(0, np.sqrt(s2b), 60)[pv] + rng.normal(0, 1.0, pv.size)
        means = np.bincount(pv, weights=y) / sz
        sd = np.sqrt(np.bincount(pv, weights=(y - means[pv]) ** 2) / (sz - 1))
        res[s2b].append(DirectIUR().fit(sizes=sz, estimates=means, standard_errors=sd / np.sqrt(sz)).iur_)
n_p = (sz.sum() - (sz**2).sum() / sz.sum()) / 59
true = 0.02 / (0.02 + 1.0 / n_p)
lo_q, hi_q = chi2.ppf([0.025, 0.975], 59) / 59
iur = np.array(res[0.02])
cover = np.mean((1 - (1 - iur) * hi_q <= true) & (true <= 1 - (1 - iur) * lo_q))
print(f"sigma_b^2 = 0: P(IUR estimate < 0) {np.mean(np.array(res[0.0]) < 0):.3f}")
print(f"sigma_b^2 = 0.02: true IUR {true:.3f}; mean estimate {iur.mean():.3f} (predicted {1 - (1 - true) * 59 / 57:.3f}); "
      f"SD {iur.std():.3f} (predicted {(1 - true) * 59 / 57 * np.sqrt(2 / 55):.3f}); interval coverage {cover:.3f}")
```

```
sigma_b^2 = 0: P(IUR estimate < 0) 0.497
sigma_b^2 = 0.02: true IUR 0.666; mean estimate 0.655 (predicted 0.654); SD 0.070 (predicted 0.066); interval coverage 0.944
```

The predicted bias is matched; unequal sizes add some variance beyond the balanced-layout value, and the
interval's coverage stays close to its nominal 95%. By the formula above an IUR of 0.5 has a standard
deviation of about 0.07 with 100 providers and 0.4 with 10.

### 5.3. Monte Carlo error of the bootstrap

Each $S_k^{*2}$ has a relative standard deviation of about $\sqrt{2/(B - 1)}$ from resampling, and
$s_{t,w}^2$ averages $K$ of them, so its Monte Carlo relative error is about $\sqrt{2/((B - 1)K)}$ —
0.008 for $B = 100$ and $K = 300$ — against the sampling relative error of $s_t^2$, about
$\sqrt{2/(K - 1)} = 0.08$. The default of 100 replicates is ample for the overall IUR; per-provider
variances $S_k^{*2}$ are individually much noisier.

## 6. Split-half reliability

### 6.1. Equal sizes: the Spearman–Brown formula

`SplitHalfIUR` splits each provider's records at random into halves, computes the measure on each half,
and correlates the two halves across providers. With $n$ records per provider each half has noise
$2\sigma_w^2/n$, the halves share $\theta_k$, and their correlation is

$$
r = \frac{\sigma_b^2}{\sigma_b^2 + 2\sigma_w^2/n}.
$$

The Spearman–Brown formula {cite}`iurt-Spearman1910Correlation,iurt-Brown1910Some` steps a half-length
correlation up to the full length,

$$
\frac{2r}{1 + r} = \frac{\sigma_b^2}{\sigma_b^2 + \sigma_w^2/n} = \rho(n),
$$

so with equal sizes the split-half and the analysis-of-variance estimators target the same quantity.

### 6.2. Unequal sizes: a different effective size

With unequal sizes the halves' covariance is still $\sigma_b^2$, but each half's variance is
$\sigma_b^2 + 2\,\overline{v}$ with $\overline{v}$ the *average* noise $K^{-1}\sum_k v_k$, so the
split-half correlation converges to $\sigma_b^2/(\sigma_b^2 + 2\overline{v})$ and the Spearman–Brown value
to the reliability of a provider whose noise is the average noise. Because the noise is roughly
proportional to $1/n_k$, that provider's size is roughly the harmonic mean of the sizes, which in typical
designs is well below $n'$, and the split-half estimate is then systematically below the bootstrap IUR. Over 30 replicates of the cohort of Section 3.3:

```python
out = []
for rep in range(30):
    n_r = rng.integers(20, 401, 300); p_r = np.repeat(np.arange(300), n_r)
    e_r = rng.gamma(2.0, 0.05, p_r.size); R_r = np.exp(rng.normal(0, tau, 300)); O_r = rng.poisson(e_r * R_r[p_r])
    E_r = np.bincount(p_r, weights=e_r)
    sh = SplitHalfIUR(n_iter=2, seed=rep).fit(O_r, e_r, p_r)
    b = BootstrapIUR(n_boot=30, seed=rep).fit(O_r, e_r, p_r)
    r_half = var_R / (var_R + 2 * np.mean(mean_R / E_r))                   # half-sample correlation, predicted
    out.append((b.iur_, var_R / (var_R + mean_R / (0.1 * b.n_prime_)),
                sh.iur_pearson_, 2 * r_half / (1 + r_half), sh.iur_spearman_, sh.iur_kendall_))
m, se = np.mean(out, axis=0), np.std(out, axis=0, ddof=1) / np.sqrt(len(out))
for k, name in enumerate(["bootstrap iur_", "  truth at size n'", "split-half Pearson",
                          "  SB of predicted half-sample r", "split-half Spearman", "split-half Kendall"]):
    print(f"{name:32s} {m[k]:.3f} (Monte Carlo se {se[k]:.3f})")
```

```
bootstrap iur_                   0.581 (Monte Carlo se 0.005)
  truth at size n'               0.582 (Monte Carlo se 0.001)
split-half Pearson               0.446 (Monte Carlo se 0.011)
  SB of predicted half-sample r  0.454 (Monte Carlo se 0.002)
split-half Spearman              0.478 (Monte Carlo se 0.009)
split-half Kendall               0.361 (Monte Carlo se 0.007)
```

The prediction is a large-sample one. The sample correlation is sensitive to the heavy tails that the
smallest providers' noise gives the halves, and with providers of only a few expected events it falls
below the prediction; neither estimator is wrong, but they answer the question at different sizes, and a
split-half value should not be compared with a bootstrap IUR without accounting for that.

### 6.3. Rank and categorical agreement

The other split-half variants apply the Spearman–Brown formula to Kendall's $\tau$, Spearman's rank
correlation, or a weighted $\kappa$ on quantile categories. For normally distributed halves with
correlation $r$, $\tau = (2/\pi)\arcsin r$ and Spearman's correlation is $(6/\pi)\arcsin(r/2)$
{cite}`iurt-Kruskal1958Ordinal`; neither equals $r$, so the Spearman–Brown formula applied to them does
not give $\rho(n)$. Kendall's $\tau$ is well below $r$ and understates the reliability substantially;
Spearman's correlation is close to $r$. Inverting the relations first recovers the Pearson scale:

```python
from scipy.stats import kendalltau, spearmanr

sb = lambda c: 2 * c / (1 + c)
for r in (0.3, 0.5, 0.7):
    xy = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], 20_000)
    t, s = kendalltau(xy[:, 0], xy[:, 1])[0], spearmanr(xy[:, 0], xy[:, 1])[0]
    print(f"r {r}: SB(r) {sb(r):.3f} | Kendall {t:.3f} -> SB {sb(t):.3f}, SB(sin(pi t/2)) {sb(np.sin(np.pi * t / 2)):.3f}"
          f" | Spearman {s:.3f} -> SB {sb(s):.3f}, SB(2 sin(pi s/6)) {sb(2 * np.sin(np.pi * s / 6)):.3f}")
```

```
r 0.3: SB(r) 0.462 | Kendall 0.191 -> SB 0.321, SB(sin(pi t/2)) 0.456 | Spearman 0.283 -> SB 0.442, SB(2 sin(pi s/6)) 0.456
r 0.5: SB(r) 0.667 | Kendall 0.335 -> SB 0.502, SB(sin(pi t/2)) 0.669 | Spearman 0.485 -> SB 0.653, SB(2 sin(pi s/6)) 0.669
r 0.7: SB(r) 0.824 | Kendall 0.494 -> SB 0.661, SB(sin(pi t/2)) 0.824 | Spearman 0.683 -> SB 0.812, SB(2 sin(pi s/6)) 0.824
```

`pprof_py` applies the Spearman–Brown formula to each coefficient directly, as the rank variants are
usually reported; they are measures of agreement in ranking, robust to outlying values, and not
estimates of $\rho(n)$. The categorical variants (quantile bins at 10%, 30%, 70% and 90% by default)
coarsen the halves further and have no closed-form relation to $r$.

### 6.4. Odd sizes and single-record providers

A provider with $n_k$ records is split into $\lfloor n_k/2\rfloor$ and $\lceil n_k/2\rceil$ records, so
the halves are unequal for odd sizes, which matters only for very small providers. A provider with one
record cannot be split: its first half is empty. `SplitHalfIUR` leaves such providers out of the
correlations with a warning and lists them in `excluded_groups_`; the result is that of a fit without
them:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    one = SplitHalfIUR(n_iter=1, seed=0).fit(np.r_[O, 1.0], np.r_[e, 0.2], np.r_[prov, K])   # one provider with 1 record
print(caught[0].message)
print("excluded:", one.excluded_groups_, "| providers used:", one.n_groups_, "| equals the fit without it:",
      np.array_equal(one.iur_all_, SplitHalfIUR(n_iter=1, seed=0).fit(O, e, prov).iur_all_))
```

```
1 of 301 groups have fewer than two observations and cannot be split; they are excluded from the split-half correlations.
excluded: [300] | providers used: 300 | equals the fit without it: True
```

`BootstrapIUR` and `DirectIUR` keep such providers: their weight $n_k - 1$ in $s_{t,w}^2$ is 0, and they
enter $s_t^2$ with weight $n_k$.

## 7. Interpretation and limitations

**The IUR depends on the population, not only on the measure.** $\sigma_b^2$ is the spread of true values
among the providers being compared. A better risk model, which explains more of the between-provider
variation, lowers $\sigma_b^2$ and with it the IUR, although the measure has become more accurate; and an
IUR near 0 does not mean that a measure cannot identify the few providers with extreme outcomes
{cite}`iurt-Kalbfleisch2018Does,iurt-He2020Profile`. The profile IUR of {cite:t}`iurt-He2020Profile`
measures reliability directly by the reproducibility of flags; `pprof_py` does not implement it.

**The IUR depends on the scale of the measure.** A ratio and the log-odds provider effect of the same
outcome have different noise structures, and their IURs differ. On the log-odds scale a provider with no
events has an estimate at the bound and an enormous standard error, which the pooled within variance
weights by its size:

```python
import pandas as pd
from pprof_py import LogisticFixedEffectModel

m_l = 120; n_l = rng.integers(8, 120, m_l); p_l = np.repeat(np.arange(m_l), n_l)
x = rng.normal(size=p_l.size)
y = rng.binomial(1, 1 / (1 + np.exp(-(-2.5 + 0.5 * x + rng.normal(0, 0.3, m_l)[p_l]))))
fe = LogisticFixedEffectModel(use_dataprep=False, screen_providers=False).fit(
    pd.DataFrame({"y": y, "x": x, "p": p_l}), y_var="y", x_vars=["x"], provider_var="p")
gamma, se = np.ravel(fe.coefficients_["gamma"]), np.sqrt(np.ravel(fe.variances_["gamma"]))
no_event = np.bincount(p_l, weights=y) == 0
keep = ~no_event
print(f"{no_event.sum()} of {m_l} providers have no events; their SEs {se[no_event].min():.0f}-{se[no_event].max():.0f}, "
      f"others' median {np.median(se[keep]):.2f}")
print(f"DirectIUR on gamma: all providers {DirectIUR().fit(fe.provider_sizes_, gamma, se).iur_:.3f}; "
      f"without no-event providers {DirectIUR().fit(fe.provider_sizes_[keep], gamma[keep], se[keep]).iur_:.3f}")
p_hat = 1 / (1 + np.exp(-(fe.xbeta_ + np.median(gamma))))   # expected probability at the median provider
print(f"BootstrapIUR on O/E: {BootstrapIUR(n_boot=100, seed=3).fit(y, p_hat, p_l).iur_:.3f}")
```

```
3 of 120 providers have no events; their SEs 29-66, others' median 0.46
DirectIUR on gamma: all providers -12.914; without no-event providers 0.184
BootstrapIUR on O/E: 0.275
```

A handful of such providers makes the direct IUR on the log-odds scale meaningless; without them, the
log-odds and ratio IURs differ but are of the same order.

**The noise is conditional on the risk model.** Both estimators treat the expected values as fixed. The
risk model's estimation error is shared by all providers and is not noise in the sense of Section 2;
with national data it is small.

**One number for all sizes.** The IUR is the reliability at $n'$. With sizes spread widely the reliability
curve (Section 4.1) is the more informative summary, and a threshold on the IUR says little about the
smallest providers.

**Precision.** With fewer than about 50 providers the IUR is imprecise (Section 5.2), and stratified
values are more so (Section 4.3).

## 8. Implementation

| concept | `pprof_py` (`pprof_py.measures.iur`) |
|---|---|
| provider measure (observed/expected) | `ratio_measure`; any `measure_fn(obs, exp, groups)` |
| bootstrap within variance $S_k^{*2}$, decomposition | `BootstrapIUR(n_boot, measure_fn, seed).fit(obs, exp, groups)` |
| within variance from standard errors | `DirectIUR().fit(sizes, estimates, standard_errors)` |
| $\widehat{\text{IUR}}$, $s_b^2$, $n' s_{t,w}^2$, $n'$ | `iur_`, `s2_between_`, `s2_within_`, `n_prime_` |
| reliability curve $\hat\rho(n)$ | `decile_table()` |
| per-provider reliabilities $\hat\rho(n_k)$ | `BootstrapIUR.iur_groups_` |
| subgroup decompositions | `BootstrapIUR.stratified_iur()` |
| split-half estimators | `SplitHalfIUR(n_iter, category_probs, measure_fn, seed)`; `summary()`, `iur_all_`, `excluded_groups_` |

## References

```{bibliography} references.bib
:filter: docname in docnames
:keyprefix: iurt-
:labelprefix: IUR
```
