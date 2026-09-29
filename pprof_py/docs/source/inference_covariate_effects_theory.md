(inference_beta_theory)=
# Inference for Covariate Effects

This page is Part I of the theory series on inference in `pprof_py`. It covers the regression coefficients
$\beta$ of the risk-adjustment model: how they are estimated jointly with the provider effects, their variance,
the Wald, likelihood-ratio and score tests, what changes when providers are small or patients contribute several
records, and how the other model families differ. Part II treats the provider effects $\gamma$ and Part III the
standardized measures built from both; the [empirical-null](empirical_null_theory.md) and
[inter-unit reliability](iur_theory) pages complete the series. Every code block below runs as written, and every
numerical claim was checked by simulation or exact computation.

## 1. Introduction

In provider profiling the covariates are not the object of interest: $\beta$ adjusts each provider's outcomes for
the characteristics of its patients, and the provider effects are then compared at a common case mix. The
precision of $\hat\beta$ matters for two reasons. It is reported — risk models are published and reviewed with
their coefficients and tests — and its error propagates into every provider comparison (Part II). Three features
of profiling data shape the theory: the model has one parameter per provider, so the number of parameters grows
with the data; providers are often small; and patients contribute several records. Section 2 sets out the
fixed-effect likelihood and the variance of $\hat\beta$; Section 3 the three tests; Section 4 the consequence of
many small providers; Section 5 the cluster-robust variance; Section 6 the other model families; Section 7
collects the approximations the implementation relies on.

## 2. The fixed-effect likelihood

### 2.1. Model

Record $j$ of provider $k$ has covariates $x_{kj} \in \mathbb R^p$ and $Y_{kj}$ events out of $N_{kj}$ trials
($N_{kj} = 1$ for binary outcomes), with

$$
Y_{kj} \sim \text{Binomial}(N_{kj}, p_{kj}), \qquad \operatorname{logit} p_{kj} = \gamma_k + x_{kj}^\top\beta,
$$

independently given $(\gamma, \beta)$, $k = 1, \dots, K$. There is no intercept: the provider effects
$\gamma_k$ carry it. The log-likelihood is
$\ell(\gamma, \beta) = \sum_{k,j} \{Y_{kj}\eta_{kj} - N_{kj}\log(1 + e^{\eta_{kj}})\}$ with
$\eta_{kj} = \gamma_k + x_{kj}^\top\beta$.

### 2.2. Score and information

With $r_{kj} = Y_{kj} - N_{kj}p_{kj}$ and $w_{kj} = N_{kj}p_{kj}(1 - p_{kj})$, the score is
$U_{\gamma_k} = \sum_j r_{kj}$, $U_\beta = \sum_{k,j} x_{kj} r_{kj}$, and the information has the block form

$$
I = \begin{pmatrix} D & B^\top \\ B & A \end{pmatrix}, \qquad
D = \operatorname{diag}\Big(\sum_j w_{kj}\Big), \quad
B_{\cdot k} = \sum_j w_{kj} x_{kj}, \quad
A = \sum_{k,j} w_{kj} x_{kj} x_{kj}^\top .
$$

The provider block $D$ is diagonal because each $\gamma_k$ enters only its own provider's records. That structure
makes inference with thousands of providers cheap: nothing of size $K \times K$ is ever formed.

### 2.3. The variance of $\hat\beta$

By block inversion,

$$
\operatorname{Var}(\hat\beta) \approx S^{-1}, \qquad S = A - B D^{-1} B^\top,
$$

and $\operatorname{Var}(\hat\gamma_k) \approx D_k^{-1} + D_k^{-2}\, B_{\cdot k}^\top S^{-1} B_{\cdot k}$. The Schur
complement $S$ is the information of the *profile* likelihood of $\beta$, $\max_\gamma \ell(\gamma, \beta)$: the
information about $\beta$ left after the provider effects are estimated. Written out,
$S = \sum_{k,j} w_{kj}(x_{kj} - \bar x_k)(x_{kj} - \bar x_k)^\top$ with $\bar x_k$ provider $k$'s
$w$-weighted mean covariate, so only within-provider variation of the covariates informs $\beta$ — a covariate
constant within providers is not identified, and one whose variation is mostly between providers is estimated
imprecisely. `variances_` holds these two quantities; they equal the corresponding blocks of the dense inverse:

```python
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from pprof_py import LogisticFixedEffectModel

def cohort(rng, m=40, sizes=(30, 120), beta=(0.3, -0.4), shift=1.0):
    """Providers with their own covariate mean (x1 is associated with the provider effects)."""
    n = rng.integers(sizes[0], sizes[1], m); prov = np.repeat(np.arange(m), n)
    mu = rng.normal(0, shift, m); gamma = 0.8 * mu + rng.normal(0, 0.3, m)
    x1, x2 = rng.normal(mu[prov], 1), rng.normal(size=prov.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1 + gamma[prov] + beta[0] * x1 + beta[1] * x2))))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "prov": prov})

rng = np.random.default_rng(1)
d = cohort(rng)
fit = LogisticFixedEffectModel(use_dataprep=False).fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="prov")
p = fit.fitted_; q = p * (1 - p); m = len(fit.provider_ids_)
D = np.c_[np.eye(m)[fit.provider_indices_], d[["x1", "x2"]].to_numpy()]    # design of (gamma, beta)
inv = np.linalg.inv(D.T @ (q[:, None] * D))                                  # dense inverse information
print("Var(beta): block formula vs dense inverse, max |diff|", f"{np.max(np.abs(fit.variances_['beta'] - inv[m:, m:])):.1e}")
print("Var(gamma_k): block formula vs dense inverse, max |diff|", f"{np.max(np.abs(fit.variances_['gamma'] - np.diag(inv)[:m])):.1e}")
```

```
Var(beta): block formula vs dense inverse, max |diff| 1.3e-18
Var(gamma_k): block formula vs dense inverse, max |diff| 1.1e-16
```

### 2.4. Estimation

The joint maximum-likelihood estimate is computed by Newton's method on $(\gamma, \beta)$ with the same block
algebra (`algorithm="Serbin"`, R's `logis_BIN_fe_prov`), or by alternating updates of $\gamma$ and $\beta$
(`"Ban"`), with Armijo backtracking. Two features of the fit matter for inference. After each step the provider
effects are clamped to $\operatorname{median}(\gamma) \pm$ `bound` (10 by default), so a provider with no events
or only events, whose likelihood has no maximum, sits at the bound; and iteration stops when the largest change in
$\beta$ falls below `tol` ($10^{-8}$). Newton's method is equivariant under a shift of the covariates — replacing
$x$ by $x + c$ changes $\gamma_k$ to $\gamma_k - c^\top\beta$ and leaves $\hat\beta$ and the fitted probabilities
unchanged — and the fit keeps that property, so no inference about $\beta$ depends on where the covariates are
centred.

## 3. Tests and intervals

For a single coefficient, $H_0: \beta_j = b_0$, `summary(test_method=...)` offers the three classical tests. All
three are asymptotically $\chi^2_1$ under $H_0$ (the Wald statistic as a squared $z$), and all three are
asymptotically equivalent under local alternatives {cite}`inb-Rao1948Large`.

### 3.1. Wald

$z_j = (\hat\beta_j - b_0)/\sqrt{(S^{-1})_{jj}}$, referred to the standard normal (as R's `summary.logis_fe`),
with the interval $\hat\beta_j \pm z_{1-\alpha/2}\sqrt{(S^{-1})_{jj}}$ and one-sided variants. It needs only the
full fit, and it is the only one of the three that tests a value other than 0 or gives an interval.

### 3.2. Likelihood ratio

$\Lambda_j = 2\{\ell(\hat\gamma, \hat\beta) - \ell(\tilde\gamma, \tilde\beta_{-j})\}$, with
$(\tilde\gamma, \tilde\beta_{-j})$ the fit without covariate $j$. The refit uses exactly the fitted records, their
trials, and the fitted algorithm and settings, from the same starting point as `fit()`; the statistic is therefore
the difference of two maximized log-likelihoods of the same data.

### 3.3. Score

The score test evaluates the score of $\beta_j$ at the fit without covariate $j$,
$U_j = \sum_{k,i} x_{ki,j}\,\tilde r_{ki}$, and refers $U_j^2$ to its variance. Because the other parameters
$\eta = (\gamma, \beta_{-j})$ are estimated, that variance is the *efficient* information

$$
I_{jj\cdot\eta} = I_{jj} - I_{j\eta}\, I_{\eta\eta}^{-1}\, I_{\eta j},
$$

the information about $\beta_j$ that remains after the nuisance parameters are estimated; using $I_{jj}$, or
partialling out only part of $\eta$, overstates the information and makes the test conservative. With the block
structure of Section 2.2 the efficient information needs no dense inverse: with $c_\gamma = \sum_i w_{ki} x_{ki,j}$
per provider, $c_\beta = \sum w\,x_{-j} x_j$ and $S_{-j}$ the Schur complement of the model without covariate $j$,

$$
I_{jj\cdot\eta} = \sum w\,x_j^2 - c_\gamma^\top D^{-1} c_\gamma
- (c_\beta - B_{-j} D^{-1} c_\gamma)^\top S_{-j}^{-1} (c_\beta - B_{-j} D^{-1} c_\gamma),
$$

all evaluated at the reduced fit, as in R's `summary.logis_fe`. On one data set the three statistics nearly
agree, and under $H_0$ all three hold their size:

```python
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    tables = {t: fit.summary(test_method=t) for t in ("wald", "lr", "score")}
print(pd.DataFrame({"Wald z^2": tables["wald"]["stat"] ** 2, "LR": tables["lr"]["stat"], "score": tables["score"]["stat"]}).round(3).to_string())
rej = np.zeros(3); R = 200
for rep in range(R):
    dn = cohort(rng, beta=(0.0, -0.4))                                 # beta_1 = 0
    f = LogisticFixedEffectModel(use_dataprep=False).fit(dn, y_var="y", x_vars=["x1", "x2"], provider_var="prov")
    rej += [f._compute_wald_beta(0)["p_value"] < 0.05, f._compute_lr_beta(0)["p_value"] < 0.05,
            f._compute_score_beta(0)["p_value"] < 0.05]
print(f"size of 5% tests of beta_1 = 0 ({R} replicates): Wald {rej[0] / R:.3f}, LR {rej[1] / R:.3f}, score {rej[2] / R:.3f}")
```

```
    Wald z^2      LR   score
x1    32.735  33.393  33.186
x2    65.731  68.647  67.586
size of 5% tests of beta_1 = 0 (200 replicates): Wald 0.050, LR 0.050, score 0.050
```

The three tests differ in what they need and in how they fail. The Wald test uses the curvature at $\hat\beta$
and is not invariant to reparametrization; the LR test is invariant and usually the most accurate in moderate
samples; the score test needs only the reduced fit. In the tables returned by `summary()`, the LR and score rows
report their own statistic and $p$-value and the Wald interval, and test $\beta_j = 0$ two-sided only.

## 4. Many providers: the incidental-parameter problem

The asymptotics behind Sections 2–3 let the number of records per provider grow. When instead the number of
providers grows with their sizes fixed, the number of parameters grows with the data, and the joint maximum-
likelihood estimate of $\beta$ is inconsistent {cite}`inb-Neyman1948Consistent`: for each provider the
estimate $\hat\gamma_k(\beta)$ carries an error of order $1/n_k$ that does not average out, and it biases the
profile score of $\beta$ by a term of the same order. For the logistic model the bias of $\hat\beta$ is
$O(1/n)$ for providers of size $n$ {cite}`inb-Hahn2004Jackknife`, away from zero, and with pairs
($n_k = 2$) $\hat\beta$ converges to $2\beta$ {cite}`inb-Chamberlain1980Analysis`. Holding 4,000 records fixed and
varying the provider size:

```python
for size in (2, 5, 10, 25, 100):
    est, cover = [], []
    for rep in range(60):
        m_s = 4000 // size
        prov = np.repeat(np.arange(m_s), size); x = rng.normal(size=prov.size)
        y = rng.binomial(1, 1 / (1 + np.exp(-(rng.normal(0, 1, m_s)[prov] + 0.5 * x))))
        f = LogisticFixedEffectModel(use_dataprep=False).fit(pd.DataFrame({"y": y, "x": x, "p": prov}),
                                                             y_var="y", x_vars=["x"], provider_var="p")
        b, se = float(np.ravel(f.coefficients_["beta"])[0]), float(np.sqrt(f.variances_["beta"][0, 0]))
        est.append(b); cover.append(abs(b - 0.5) < 1.96 * se)
    print(f"{size:3d} records per provider ({4000 // size:4d} providers): mean beta_hat/beta {np.mean(est) / 0.5:.3f}, "
          f"Wald 95% coverage {np.mean(cover):.2f}")
```

```
  2 records per provider (2000 providers): mean beta_hat/beta 2.052, Wald 95% coverage 0.00
  5 records per provider ( 800 providers): mean beta_hat/beta 1.294, Wald 95% coverage 0.18
 10 records per provider ( 400 providers): mean beta_hat/beta 1.100, Wald 95% coverage 0.75
 25 records per provider ( 160 providers): mean beta_hat/beta 1.047, Wald 95% coverage 0.88
100 records per provider (  40 providers): mean beta_hat/beta 1.010, Wald 95% coverage 0.98
```

The standard error shrinks with the number of records while the bias does not, so the Wald interval misses
$\beta$ ever more often as providers get smaller at a fixed total. With 25 records per provider the coefficient
is still about 5% too large; the facility-by-hospital cells of the three-stage model's Stage 1 may hold only a few
dozen records each, so errors of this order are possible there. R's `logis_fe` computes the same estimator and
has the same property. Two remedies exist and are not implemented: the conditional likelihood
{cite}`inb-Andersen1970Asymptotic`, which eliminates $\gamma$ by conditioning on each provider's total and is
consistent for fixed sizes (but gives no provider effects), and analytical or jackknife bias corrections
{cite}`inb-Hahn2004Jackknife`.

Providers with no events or only events contribute to the other providers' estimates only through their
records' weights $w_{kj} = N_{kj}p_{kj}(1 - p_{kj})$, which tend to zero as $\hat\gamma_k$ moves to the bound:
they carry no information about $\beta$ and do not bias it, but they are not screened out when
`use_dataprep=False`, and their provider effects are the bound, not an estimate.

## 5. Cluster-robust variance

When a patient contributes several records, the records of one patient are dependent, and the information-based
variance of Section 2.3 no longer describes the sampling variability of $\hat\beta$. With records grouped into
clusters $c$ (a patient within a provider, `obs_id_var`), the sandwich estimator
{cite}`inb-Liang1986Longitudinal`

$$
\widehat{\operatorname{Var}}_{\text{rob}}(\hat\beta) = \big[I^{-1} M I^{-1}\big]_{\beta\beta}, \qquad
M = \sum_c u_c u_c^\top, \quad u_c = \sum_{(k,j) \in c} \begin{pmatrix} e_k \\ x_{kj}\end{pmatrix} r_{kj},
$$

replaces the model's variance of the score by its empirical variance across clusters. `robust_variances_["beta"]`
computes it with the block algebra of Section 2.2 (the form of R's `robust_wald_covar`), and
`summary(variance_type="robust")` uses it in the Wald test. It matters for covariates that are constant or nearly
so within a cluster, whose effective sample size is the number of clusters rather than of records; for a
record-level covariate independent across records the two variances nearly coincide. With a patient-level
covariate and a patient frailty:

```python
res = []
for rep in range(100):
    m_r = 60; n_pat = rng.integers(15, 60, m_r); prov_p = np.repeat(np.arange(m_r), n_pat)
    pid = np.arange(prov_p.size); k = rng.integers(1, 5, pid.size)                  # 1-4 records per patient
    prov_r, pid_r = np.repeat(prov_p, k), np.repeat(pid, k)
    x = rng.normal(size=pid.size)[pid_r]                                            # a patient-level covariate
    eta = -1 + rng.normal(0, .3, m_r)[prov_r] + 0.4 * x + rng.normal(0, 1.0, pid.size)[pid_r]   # patient frailty
    dr = pd.DataFrame({"y": rng.binomial(1, 1 / (1 + np.exp(-eta))), "x": x, "p": prov_r, "pid": pid_r})
    f = LogisticFixedEffectModel(use_dataprep=False).fit(dr, y_var="y", x_vars=["x"], provider_var="p", obs_id_var="pid")
    res.append((np.ravel(f.coefficients_["beta"])[0], np.sqrt(f.variances_["beta"][0, 0]), np.sqrt(f.robust_variances_["beta"][0, 0])))
res = np.array(res)
print(f"SD of beta_hat over replicates {res[:, 0].std(ddof=1):.4f}; mean model-based SE {res[:, 1].mean():.4f}; "
      f"mean robust SE {res[:, 2].mean():.4f}")
```

```
SD of beta_hat over replicates 0.0346; mean model-based SE 0.0311; mean robust SE 0.0353
```

The model-based standard error understates the sampling variability, and the sandwich matches it. The
sandwich is itself a large-sample estimate, consistent as the number of clusters grows, with a downward bias in
small samples; `pprof_py`, like R, applies no small-sample correction. The robust variance of the provider effects
is treated in Part II.

## 6. Other model families

**Linear fixed-effect model.** With normal errors the least-squares estimate is exact: $\hat\beta$ is the
within-provider regression, $\hat\sigma^2 = \text{RSS}/(n - K - p)$, and `LinearFixedEffectModel.summary()` refers
$\hat\beta_j/\text{SE}_j$ to a $t$ distribution on $n - K - p$ degrees of freedom. The incidental-parameter
problem does not arise for $\beta$ (the within estimator is unbiased for any provider sizes), although
$\hat\sigma^2$ uses the degrees-of-freedom correction to stay unbiased:

```python
from pprof_py import LinearFixedEffectModel

m_l = 25; n_l = rng.integers(5, 30, m_l); pl = np.repeat(np.arange(m_l), n_l)
xl = rng.normal(rng.normal(0, 1, m_l)[pl], 1)
yl = rng.normal(0, 1, m_l)[pl] + 0.3 * xl + rng.normal(0, 2, pl.size)
lin = LinearFixedEffectModel().fit(pd.DataFrame({"y": yl, "x": xl, "p": pl}), y_var="y", x_vars=["x"], provider_var="p")
Dl = np.c_[np.eye(m_l)[pl], xl]                                                   # provider dummies and x
coef, rss = np.linalg.lstsq(Dl, yl, rcond=None)[:2]
dof = pl.size - m_l - 1
se_ls = np.sqrt(rss[0] / dof * np.linalg.inv(Dl.T @ Dl)[-1, -1])
print(lin.summary().round(6).to_string())
print(f"least squares with dummies: beta {coef[-1]:.6f}, SE {se_ls:.6f}, t on {dof} df")
```

```
   estimate  std_error      stat   p_value  ci_lower  ci_upper
x  0.387391   0.108887  3.557744  0.000421  0.173302  0.601479
least squares with dummies: beta 0.387391, SE 0.108887, t on 384 df
```

**Random-effect models.** `LogisticRandomEffectModel` treats the provider effects as a normal sample and maximizes
the Laplace-approximated marginal likelihood, as `lme4::glmer`; `summary()` gives Wald $z$ statistics with the
fixed-effect covariance of the fitted model. This uses between- as well as within-provider variation of the
covariates, and is valid only if the provider effects are independent of the covariates — the assumption the
fixed-effect model avoids {cite}`inb-Kalbfleisch2013Monitoring`. `LinearRandomEffectModel` is the Gaussian
analogue.

**Cox model.** `CoxPH` estimates $\beta$ from the partial likelihood {cite}`inb-Cox1972Regression`; for profiling
it is stratified by provider, which removes the provider baselines exactly as conditioning does in Section 4, so
the incidental-parameter problem does not affect $\hat\beta$. `summary()` reports Wald $z$ tests and intervals
from the inverse information, or from the robust sandwich of {cite:t}`inb-Lin1989Robust` when `robust=True` or a
`cluster` is given. The standardized measures built on the stratified fit are in
[survival Chapter 4](survival/04_indirect_standardization_smr_shr) and Part III.

**Three-stage model.** In `LogisticThreeStageModel` $\beta$ comes from Stage 1, a fixed-effect fit on
facility-by-hospital cells {cite}`inb-He2013Evaluating`, and is held fixed in Stages 2 and 3; its inference is
Stage 1's (`summary(stage1=...)`), with the properties of Sections 2–5 at the cells' sizes (Section 4).

## 7. Approximations the implementation relies on

| quantity | basis | where it can fail |
|---|---|---|
| $\operatorname{Var}(\hat\beta) = S^{-1}$ | large-sample theory as provider sizes grow | small providers (Section 4) |
| Wald, LR, score $\sim \chi^2_1$ | the same | the Wald test first, in moderate samples |
| unbiasedness of $\hat\beta$ | provider sizes large | bias $O(1/n)$ for fixed sizes: 5% at 25 records per provider in Section 4 |
| no/all-event providers | clamp at median $\pm$ `bound` | their $\gamma$ is not an estimate; $\beta$ is unaffected |
| robust variance | many clusters | downward bias with few clusters; no correction (the Cox model warns below 30 clusters) |
| linear $t$ tests | normal errors | exact under normality; asymptotic otherwise |
| random-effect Wald | Laplace approximation; effects independent of covariates | confounding of provider effects and case mix |
| convergence | $\lVert\Delta\beta\rVert_\infty <$ `tol` | a warning when the iteration limit or a shortened line search ends the fit |

## 8. Implementation

| concept | `pprof_py` |
|---|---|
| joint fit, $S^{-1}$, $\operatorname{Var}(\hat\gamma_k)$ | `LogisticFixedEffectModel.fit`; `variances_["beta"]`, `variances_["gamma"]` |
| Wald, LR, score tests of $\beta_j$ | `summary(test_method=...)` with `"wald"`, `"lr"` or `"score"`; `level`, `null`, `alternative` |
| cluster-robust variance | `fit(..., obs_id_var=...)`; `robust_variances_["beta"]`; `summary(variance_type="robust")` |
| linear model | `LinearFixedEffectModel.summary()` ($t$ tests) |
| random-effect models | `LogisticRandomEffectModel.summary()`, `LinearRandomEffectModel.summary()` |
| Cox model | `CoxPH(robust=..., cluster=...)`, `summary()` |
| three-stage Stage 1 | `LogisticThreeStageModel.stage1_`, `LogisticFERandomClusterModel.summary(stage1=...)` |

## References

```{bibliography} references.bib
:filter: docname in docnames
:keyprefix: inb-
:labelprefix: INB
```
