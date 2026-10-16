(r_compatibility)=
# R compatibility notes

This package targets `survival::coxph()` (R's `survival` package, version
3.5-8, R 4.3.3) as the source of truth. Every claim below was checked
against actual R output during development (`pprof_py/r_reference/run_all.R` +
`pprof_py/tests/survival/test_r_comparison.py`), not inferred from documentation alone —
several of R's own docs are silent or misleading on exactly these points,
which is precisely why they needed checking.

This file answers a fixed checklist of R-compatibility questions (the numbering follows that checklist, so a few numbers group several questions). Each entry states the convention, why it matters, and how
it was verified.

## 1. Risk-set definition for `(start, stop]`

**Convention:** `R(t) = { i : start_i < t <= stop_i }` — left-open,
right-closed. Concretely:
- An observation censored *or* having an event exactly at time `t` is
  still in the risk set at `t` (ties on `stop` are inclusive).
- An observation entering (left truncation / staggered entry) exactly at
  time `t` is *not yet* in the risk set at `t` (ties on `start` are
  exclusive).

**Why it matters:** get this backwards and every risk-set sum near a tie
is subtly wrong, which shows up as a small, hard-to-localize bias in
`coef_` rather than an obvious failure.

**Verified by:** `pprof_py/tests/survival/test_r_comparison.py::test_left_truncation` and
`::test_combined`, both with real staggered entry — coefficients match R
to relative error ~1e-9 or tighter. Implemented in
`algorithms/survival/risk_sets.py::sweep_risk_sets`.

## 2. Tied event times: Breslow and Efron

**Breslow** treats all `d_j` simultaneous events at time `t_j` as sharing
one risk-set denominator `S0(t_j)`, as if they were `d_j` independent
draws (with replacement) from the same risk set. This is what
`ties="breslow"` computes in R and is this package's default (matching
the existing `phregSHR` R workflow this package targets).

**Efron** (R's own default) is also implemented, as of this version. The
textbook formula for Efron's correction doesn't specify how it should
generalize under case weights, so rather than guess, the implementation
was matched directly against R's actual C source:

```bash
# deb-src wasn't enabled by default; turning it on made the real source
# available even though only the compiled binary package was installed:
sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
apt-get update && apt-get source r-cran-survival
```

This mattered more than it might sound: `survival`'s source contains
**two** Efron implementations. `src/agfit4.c` backs an ordinary
`coxph()` call (confirmed via `agreg.fit.R`'s `.Call(Cagfit4, ...)`);
`src/agfit5.c` backs only the penalized/frailty path
(`coxpenal.fit.R`) and uses a visibly different weighting construction
(`efron_wt` there effectively squares the case weight, which looks like
a plausible-but-wrong general formula for an ordinary fit if you don't
notice which C file you're reading). The formula actually implemented
here — matched to `agfit4.c` — is:

```
d_j       = raw COUNT of tied deaths at t_j (not weighted)
d*_j      = sum of weights of those deaths
meanwt_j  = d*_j / d_j
S0_R\D, S1_R\D, S2_R\D = full risk-set sums MINUS the tied-death-only sums

LL_j = sum_{i in D_j} w_i*eta_i
       - meanwt_j * sum_{k=1}^{d_j} log(S0_R\D + (k/d_j)*S0(D_j))
```

with score and information the first and second derivatives of that
expression in the same S0/S1/S2 terms `algorithms/survival/ties.py::BreslowTies`
already uses. Validated against R on unweighted AND weighted, tied data
to 1e-14–1e-15 relative error (`pprof_py/tests/survival/test_r_comparison.py::test_efron_basic`,
`::test_efron_weights`, `::test_efron_left_truncation`). As a consistency
check independent of R entirely: with `d_j == 1` (no actual tie),
`meanwt_j == d*_j` and the formula's single `k=1` term reduces exactly to
Breslow's — `EfronTies` and `BreslowTies` were confirmed to agree to
machine precision on data with no ties at all.

**Efron's baseline hazard is a *different formula* from Breslow's, not
just the same formula fed different inputs** — this cost real debugging
time to find, because coefficients, standard errors, and log-likelihood
all matched R exactly *before* this was fixed, giving no signal that
anything was wrong until the baseline hazard was checked specifically.
R's `survfit.coxph`/`basehaz` select `ctype=2` automatically whenever
`object$method=="efron"` (`R/survfit.coxph.R`), which routes to a
different C routine (`src/agsurv5.c`, via the R-level `agsurv()` in
`R/agsurv.R`) computing:

```
dH0(t_j) = meanwt_j * sum_{k=0}^{d_j-1} 1 / (S0_R(t_j) - (k/d_j)*S0(D_j))
```

— an *average of reciprocals* over the same fractional risk-set
reductions the likelihood uses, not Breslow's single ratio `d*_j/S0(t_j)`.
Implemented as `EfronTies.baseline_hazard_increments`, dispatched to
correctly by `inference/survival/baseline.py::compute_baseline_hazard` based on
whichever tie method actually fit the model (previously this dispatch
didn't exist at all — `models/survival/coxph.py` called a hardcoded Breslow
formula regardless of `ties=`, which is exactly the kind of bug that
produces a plausible-looking but silently wrong number).

**Martingale residuals under Efron get a further, separate correction**
— see question 9b below; this is a *third* place Efron needed its own
logic, not a variation on the first two.

Exact ties are not implemented (see `algorithms/survival/ties.py::ExactTies`).

## 3. Ordering of start/stop/event records

No ordering is required or assumed on input — `sweep_risk_sets` sorts
internally. (R has the same property; this was checked by fitting the
same data shuffled into a different row order and confirming identical
results, not just by reading R's docs.)

## 4. Zero-width intervals

Rejected outright (`start < stop` is required) — both packages treat
this as a data error, not a numerical edge case. See
`data/survival_validation.py`.

## 5. Weight semantics

R's `weights=` are, by default, ordinary case/frequency weights: they
multiply every term of the log partial likelihood, score, and
information matrix, and multiply into the Breslow baseline-hazard
increments (`d*_j = sum of weights of events at t_j`, `S0` a
weight-times-risk-score sum). This package implements exactly that (the
"model-based" covariance, i.e. the inverse of the weighted information
matrix — the same quantity R reports as `se(coef)`).

**Important:** when weights aren't exact integer replication counts, R's `summary()` *also* prints a `robust se` column (a sandwich/cluster-robust variance) and notes it may be more appropriate. R turns the robust variance on automatically when weights are non-integer or a cluster term is present; `CoxPH` does **not** — it reports the model-based variance unless you ask for robust inference (`CoxPH(robust=True)` or `fit(..., cluster=...)`, see Section 22). So, by default, `standard_errors_` corresponds to R's `se(coef)` column (and `naive_standard_errors_` always does); with robust inference on, `standard_errors_` corresponds to R's `robust se`. Comparing against the wrong column is an easy mistake to make when validating against R's console output; `pprof_py/tests/survival/test_r_comparison.py` extracts `se(coef)` specifically, by regex-anchored column name, to avoid it.

**This trap is worse than it looks if you compare programmatically
instead of visually.** `summary(fit)$coefficients[, "se(coef)"]` is the
model-based SE (matches `standard_errors_` here) — but `vcov(fit)` and
`fit$var` are the *robust* variance once non-trivial weights are
present, silently promoted there from the `fit$naive.var` slot where the
model-based variance actually lives. `vcov(fit, robust=FALSE)` does
**not** override this. Confirmed directly:

```r
> sqrt(diag(vcov(fit)))            #  0.0711  0.0682   <- robust
> sqrt(diag(vcov(fit, robust=FALSE)))  #  0.0711  0.0682   <- still robust!
> summary(fit)$coefficients[,"se(coef)"]  # 0.0524  0.0568  <- model-based
> sqrt(diag(fit$naive.var))            #  0.0524  0.0568  <- model-based (here)
> sqrt(diag(fit$var))                  #  0.0711  0.0682  <- robust (here)
```

So anyone scripting a comparison against this package (rather than
eyeballing `summary()`'s printout) needs `fit$naive.var`, not `vcov(fit)`
— `pprof_py/tests/survival/test_r_comparison.py::test_efron_weights` checks against
`fit$naive.var` explicitly, and separately asserts `standard_errors_`
does *not* match `fit$var`, so a future tolerance change couldn't make
both pass by accident.

**Zero weights** are accepted (drop an observation's influence while
keeping it in the data) and were exercised directly
(`pprof_py/tests/survival/test_validation.py::test_allows_zero_weight`).

**`n_events_`** is the *raw* count of rows with `event=1`, not a
weight-weighted sum — confirmed against R's reported `nevent` under
non-trivial weights (`fit$nevent` does not change when weights are
non-integer).

## 6. Offset handling — the least obvious finding in this package

A naive reading of `basehaz(fit, centered = FALSE)` suggests it reports
the baseline hazard at `X = 0`. It does not fully do that when an
`offset()` term is present, and this is **not documented** in `?basehaz`
— it was found by reading `survival:::basehaz` and `survival:::survfit.coxph`
source directly (both included verbatim as of survival 3.5-8 in the
package installed during this project; see the excerpt at the bottom of
this section) and confirmed empirically (a constant offset shift left
`basehaz()`'s output *completely unchanged*, which is inconsistent with
"hazard at offset=0" and consistent with what's derived below).

**What actually happens:** `survfit.coxph()`'s reference curve (used
internally by `basehaz`) evaluates risk scores as
`exp(eta_i - xcenter)`, where

```
xcenter = mean(X) @ beta_hat + offset_mean
offset_mean = mean(offset)                         if no weights
            = sum(offset * weight) / sum(weight)   if weights given
```

`basehaz(centered = FALSE)` only divides back out the `mean(X) @ beta_hat`
part of `xcenter` (that's what "un-centers" the covariates) — the
`offset_mean` part is never removed. Working through the algebra, the
`mean(X) @ beta_hat` contribution cancels exactly between the reference
curve's construction and `basehaz`'s un-centering step, leaving:

```
basehaz(fit, centered=FALSE)  ==  H0_at_(X=0, offset=0)(t)  *  exp(offset_mean)
```

**This package matches that exactly**, on purpose, because the design goal is output "equivalent to
`basehaz(coxph_model, centered=FALSE)`" and this is what that call
actually computes. `models/survival/coxph.py::fit` computes the "pure"
`H0_at_(X=0,offset=0)` internally first (needed for residuals and
predictions to stay self-consistent — see below), then applies the
`exp(offset_mean)` factor *only* to the public `baseline_hazard_`
attribute.

**Why residuals and predictions are unaffected:** martingale residuals
and `predict_cumulative_hazard`/`predict_survival_function` pair the raw
`H0_at_(X=0,offset=0)` with each subject's true, un-shifted
`eta_i = X_i @ beta + offset_i`. The `offset_mean` shift that appears in
`baseline_hazard_` would cancel exactly against a matching shift in
`eta_i` if applied to both — so applying it to *neither* gives the same
(correct) answer more simply, and avoids the shift leaking into
predictions on new data with a different offset distribution than the
training data. This was confirmed empirically too: R's
`predict(fit, type="expected")` and `residuals(fit, type="martingale")`
are also unaffected by a constant offset shift, for the same reason.

If a model has no `offset()` term, `offset_mean = 0` and this entire
correction is a no-op — which is exactly why `test_basic`, `test_strata`,
and `test_weights` (none of which use an offset) matched R's baseline
hazard to machine precision *before* this correction was even
implemented, while `test_offset` and `test_combined` did not until it
was added. That contrast is what led to isolating this in the first
place, rather than just loosening a tolerance.

Relevant excerpt from `survival:::basehaz` (R 4.3.3, survival 3.5-8):

```r
if (!centered) {
    zcoef <- ifelse(is.na(coef(fit)), 0, coef(fit))
    offset <- sum(fit$means * zcoef)      # NB: local var name "offset" here
    chaz <- sfit$cumhaz * exp(-offset)    # is mean(X)%*%beta, NOT the model's
}                                          # offset() term -- a naming collision
```

and `survival:::survfit.coxph` (the `offset.mean` computation feeding
`xcenter`, shown as printed source line numbers from that build):

```
91:  offset.mean <- mean(offset)
92:  else offset.mean <- sum(offset * (weights/sum(weights)))
...
161: xcenter <- sum(object$means * beta) + offset.mean
```

## 7. Stratified baseline hazards

Computed independently per stratum (own risk sets, own event times, own
`H0`/`S0`), coefficients shared across strata — matches
`coxph(... + strata(g))`. R's `strata()` labels basehaz() rows like
`"provider=0"` rather than the bare value; this package's
`baseline_hazard_["stratum"]` uses the bare stratum value instead
(`pprof_py/tests/survival/test_r_comparison.py` extracts R's numeric suffix for
comparison, rather than this package adopting R's string-labeling
convention).

## 8. Baseline hazard table shape

`basehaz()` returns one row per **distinct observed time**, event or
censoring — the hazard value at a censoring-only time is just carried
forward unchanged from the last actual event. This package's
`baseline_hazard_` returns one row per distinct **event** time only
(the minimal representation of the same right-continuous step function —
every Python-reported time and hazard value is present, verbatim, in
R's larger table; R's table just also repeats the same value at
additional non-jump timestamps). `predict_cumulative_hazard` / `predict_survival_function` return the step function **at the stratum's event times only** (they have no `times=` argument); to read it at other times, forward-fill the returned frame (recipe in the `CoxPH` reference page). Nothing is lost: the compact table and the step function carry the same information.

## 9. Martingale residuals under left truncation

`M_i = event_i - [H0(stop_i) - H0(start_i)] * exp(eta_i)` — the hazard
contribution is integrated only over `(start_i, stop_i]`, the interval
the subject was actually at risk, not `H0(stop_i)` alone (which would
overstate a late entrant's expected event count by including hazard
accrued before they entered). Verified against R's
`residuals(fit, type="martingale")` on the left-truncation and combined
(truncation + strata + offset + weights) datasets; the test tolerance is 1e-4 absolute and the largest difference in a fresh run was below 5e-9 (see the validation report).

## 9b. Martingale residuals under Efron ties need a THIRD separate formula

The formula in question 9 is exactly right for Breslow, but is
measurably wrong for Efron's tied deaths specifically — confirmed
empirically (errors of 0.5–1.0 on affected observations, not a rounding
discrepancy) after fixing the baseline hazard (question 2) made
`coef_`/`standard_errors_`/`log_likelihood_`/`baseline_hazard_` all
correct while residuals stayed wrong, which is what made clear this was
a genuinely separate bug rather than a side effect of the baseline
hazard fix.

R's `residuals(fit, type="martingale")` does not derive its answer from
`basehaz()`'s output at all (`R/residuals.coxph.R`, line 30: for
`type=="martingale"` it simply returns `object$residuals`, computed once
at fit time). That value comes from a dedicated C routine,
`src/agmart3.c` (`Cagmart3`, called from `R/agreg.fit.R`), which for
Efron applies an *additional* correction specifically to each tied
death's own residual: each tied death gets `(hazard_j - e_hazard_j) *
score_i` added to it, where `hazard_j` is the same full jump computed in
question 2 and `e_hazard_j = meanwt_j * sum_{k=0}^{d_j-1} (1 - k/d_j) /
(S0_R - (k/d_j)*S0(D_j))` is a second, distinct weighted sum over the
same fractional risk-set reductions (present nowhere else in this
package — not in the likelihood, not in the baseline hazard).

Attempting to re-derive a simplified closed form for this by hand (i.e.
"what is a tied death's *total* residual, algebraically") turned out to
be genuinely error-prone once intermediate hazard jumps between a
subject's own start and stop are accounted for — an early hand-derived
version produced an implausible result under algebraic simplification
that didn't survive a sanity check. `inference/survival/residuals.py::_martingale_residuals_one_stratum`
is instead a direct, line-by-line transcription of `agmart3.c`'s
two-pointer sweep (mirroring the *variable names and control flow* of
the C source, not a cleaned-up reformulation of it), then validated
against R rather than trusted on the strength of the derivation. It
reduces to question 9's simple formula exactly when there are no ties
(or `ties="breslow"`), and matches R on tied, weighted, and left-truncated Efron fits alike (test tolerance 1e-4 absolute; largest difference in a fresh run below 9e-9).

## 10–13. Convergence, step-halving, singular matrices, missing values

- **Convergence criterion:** relative change in log-likelihood
  `< eps` (default `1e-9`), matching `coxph.control(eps=1e-9)`.
  `max_iter=20` matches R's `iter.max=20` default. This package does
  *not* reproduce R's iteration-by-iteration path (step-halving
  schedule, exact per-iteration beta trajectory) — only its converged
  answer. Because the Cox partial likelihood used here is concave in
  `beta`, Newton-Raphson converges to the same unique maximizer
  regardless of the exact step-halving schedule, so matching the
  converged answer (which every test in this package checks, to
  relative error ≤1e-5, typically achieving 1e-9 or tighter) is the
  correct thing to verify — matching the path is not meaningful.
- **Step-halving:** implemented (`algorithms/survival/optimization.py`); a
  proposed Newton step that would decrease the log-likelihood is halved
  (up to 20 times) before being accepted.
- **Singular/near-singular information:** falls back to
  `np.linalg.lstsq`/`pinv` with a message recorded in
  `convergence_message_`, rather than raising — callers should treat an
  unusually large resulting standard error as a signal to check for
  collinearity, per `inference/survival/inference.py`.
- **Missing values:** not silently handled. `NaN`/`inf` anywhere in `X`,
  `start`, `stop`, `event`, `offset`, or `sample_weight` raises
  `SurvivalDataError` immediately in `fit()`, matching R's
  fail-fast-by-default behavior (`coxph` also errors on `NA` unless
  `na.action` is changed) rather than R's alternative
  listwise-deletion options, which are not implemented.
- **Non-convergence:** reaching `max_iter` neither raises nor warns; `converged_` is `False` and `convergence_message_` says why. Check `converged_` in production code.

## 14. Centering / scaling behavior

R's C fitting routine mean-centers `X` internally before running
Newton-Raphson, purely for numerical conditioning — this package does
the same (`utils/numerical.py::col_means`, applied in
`models/survival/coxph.py::fit`). This is provably inert for every statistical
output: shifting every row of `X` by the same constant vector rescales
every subject's risk score `exp(eta_i)` by the same multiplicative
factor, which cancels exactly in the ratios the partial likelihood is
built from. So `coef_`, `standard_errors_`, and `log_likelihood_` are
identical whether or not centering is applied — centering only improves
Newton-Raphson's conditioning when raw covariates are far from zero
(e.g. a raw "age" column). Because it's provably inert, this package
uses plain (unweighted) column means rather than trying to reproduce
whatever exact centering constant R's C code happens to use internally —
the two differ as intermediate values but produce bit-for-bit identical
final output. Where centering does *not* wash out — the baseline hazard
and residuals, which depend on the *absolute* value of eta — this
package always uses the original, uncentered `X` (see `fit`'s docstring
comment at the point `eta_fit` is computed) so "baseline" means `X=0` in
the user's own units, not `X=mean(X)`. The offset-mean subtlety in
Section 6 above is the one place a *second*, R-specific centering
convention (of the offset, not `X`) had to be matched explicitly rather
than left as an implementation detail, precisely because the design goal is bit-compatibility with `basehaz(centered=FALSE)`
specifically.

## 15. Zero-weight observations

Accepted (see Section 5) — a weight of exactly 0 removes an
observation's influence on the fit (it contributes nothing to score,
information, or the log-likelihood's death/risk-set sums, since every
term it could contribute to is itself multiplied by its weight) while
leaving it present in `n_obs_`.

## 16–17. Numerical precision, exact score/Hessian formulas

The closed-form score and information matrix (in terms of the risk-set
sums `S0`, `S1 = sum(w*r*X)`, `S2 = sum(w*r*outer(X,X))`) are derived
and documented in `algorithms/survival/ties.py::BreslowTies`. `utils/numerical.py`
clips `exp()` inputs at 700 to avoid `float64` overflow (the theoretical
overflow point is ~709.78); this has never been observed to trigger in
any test in this package's suite, since covariates are centered before
optimization, but is retained as a hard guard against a pathological or
unconverged intermediate step producing `inf` and silently corrupting
every downstream sum.

---

# Phase 3: Penalized regression and variable selection

## 18. Penalized Cox regression vs. glmnet 4.1-8

The `PenalizedCoxPH` / `PenalizedCoxPHCV` estimators target R's
`glmnet(family="cox")` / `cv.glmnet(family="cox")` (glmnet 4.1-8) as the
source of truth. Every convention below was verified against glmnet's
actual R source (not the paper or vignette alone), with file/line
references where the answer was non-obvious.

### 18a. Naming conventions

`alpha` is glmnet's elastic-net mixing parameter (0 = ridge, 1 = LASSO),
NOT scikit-learn's `ElasticNet.alpha` (overall regularization strength).
This package validates against R's `survival`/`glmnet`, so its penalized-
regression vocabulary follows glmnet, not sklearn.

### 18b. Standardization convention

glmnet's Cox family scales each column by its weighted population standard
deviation (denominator = `sum(weight)`, not `sum(weight) - 1`) and does
**not** center. Source: `weighted_mean_sd()` in `R/glmnetFlex.R`; the
Cox path unconditionally sets `xm <- rep(0.0, nvars)` in `coxpath.R`.
This is specific to the Cox model's lack of an intercept. Matched in
`algorithms/survival/penalty.py::weighted_column_scale`.

### 18c. The `c` scaling factor

glmnet normalizes the smooth (log-likelihood) part of the objective by
`c = 1/sum(weight)`. Getting this wrong by a factor of 2 (an earlier
version of this module used `2/sum(weight)`) shifts the entire lambda
path so coefficients look plausible but diverge systematically from
glmnet's. Source: `coxgrad()` (`R/coxgrad.R`) normalizes weights to
sum to 1 via `w <- w/sum(w)`, which is the convention `coxnet.deviance`
is calibrated against. Matched in
`algorithms/survival/coordinate_descent.py`.

### 18d. Penalty factor rescaling

Penalty factors are internally rescaled to sum to `p` (the number of
predictors), matching glmnet's documented convention. Source: glmnet
documentation + empirical verification. Matched in
`algorithms/survival/penalty.py::rescale_penalty_factors`.

### 18e. Lambda sequence generation

`lambda_max` (the smallest lambda at which all penalized coefficients are
exactly zero) is derived from the gradient of the unpenalized objective
at `beta = 0`, divided by `max(alpha * penalty_factor)`, matching
glmnet's `get_cox_lambda_max`. For pure ridge (`alpha = 0`), alpha is
floored at `1e-3` to keep lambda_max finite — this is glmnet's own
convention. The auto-generated grid is a geometric sequence from
`lambda_max` to `lambda_max * lambda_min_ratio`, matching glmnet's
default behavior.

**Verified by:** `test_penalized_r_comparison.py::test_lambda_max_matches_glmnet_auto_grid`
— the auto-generated grid's first N points match glmnet's own
un-forced path to ~1e-6.

### 18f. Ridge at extreme lambda

One documented, investigated exception: at `alpha = 0` (pure ridge),
the single largest lambda in the path produces coefficients ~1e-3 in
this package vs. ~1e-37 in glmnet. This package's coefficients were
verified to satisfy the exact ridge stationarity condition to 1e-17;
glmnet's extreme-edge value appears to be the outlier, not this
package's. Every other point on the same ridge path (99/100) matches to
~1e-7. This is documented, not silently tolerated.

### 18g. Cross-validation deviance

`PenalizedCoxPHCV` computes the Verweij & Van Houwelingen (1993) grouped
deviance residual per fold, matching `cv.glmnet(family="cox",
grouped=TRUE)` — confirmed against glmnet 4.1-8's actual
`R/buildPredmat.coxnetlist.R` and `R/cv.coxnet.R`. The saturated
log-likelihood formula (`-sum(wd * log(wd))` over per-stratum tied-event-
time weight sums) is copied from glmnet's own `coxnet.deviance2`/
`coxnet.deviance3` cross-checked reference implementation.

**Verified by:** `test_penalized_r_comparison.py::test_cross_validation`
— `cvm`, `lambda.min`, `lambda.1se`, and coefficients at `lambda.min`
all match cv.glmnet with explicit shared fold assignments.

### 18h. Efron ties under penalization

glmnet's Cox family only implements Breslow ties (confirmed by reading
`R/coxnet.deviance.R` directly). Efron-tie penalized fits in this
package are validated separately by self-consistency: `PenalizedCoxPH`
with `lambda → 0` must converge to this package's own (R-validated)
`CoxPH(ties="efron")` solution. See
`pprof_py/tests/survival/test_penalized_self_consistency.py`.

### 18i. Capability preservation under penalization

Every Phase 1-2 feature (strata, offset, weights, left truncation,
heavy ties) is tested under penalization on the exact same datasets
already validated in Phase 1-2, confirming the penalty machinery does
not regress any existing capability:

**Verified by:** `test_penalized_r_comparison.py` — `test_strata`,
`test_offset`, `test_weights`, `test_left_truncation`,
`test_combined_strata_offset_weights_left_truncation`,
`test_basic_heavy_ties`.

## 19. Variable selection vs. R's step()/extractAIC()

`CoxPHSelector` matches R's `step()` on a `coxph` object for AIC and
BIC criteria. The specific R convention this required verifying:

- **BIC uses `n_events`, not `n_obs`**: R's `nobs.coxph` returns the
  number of events, not observations. `extractAIC(coxph_fit,
  k = log(nobs(fit)))` therefore uses `log(n_events)` as the BIC
  penalty — using `log(n_obs)` would produce a different, wrong
  trajectory. Confirmed by inspecting `nobs.coxph`'s source and
  verified empirically.

**Verified by:** `test_selector_r_comparison.py` — forward/backward/both
AIC, backward BIC, forced-variable, and strata+offset+weights scenarios,
each checked step-by-step (criterion value at every step, not just the
final variable set).

p-value selection follows the classic textbook/SAS PROC PHREG convention
rather than any single R package; its per-model p-values come from the
fully R-validated `CoxPH.p_values_`.

---

# Phase 4: Competing risks, robust variance, time-dependent covariates

## 20. Cause-specific hazards

`CauseSpecificCoxPH` fits one ordinary `CoxPH` per cause, recoding every
other cause's events as censoring. No new statistics — each per-cause
fit is a standard `CoxPH` already validated in Phases 1-2.

**Verified by:** `test_phase4_r_comparison.py` — cause-specific
coefficients and SEs for both causes, on simple and left-truncated
competing-risks data, against R's `coxph(Surv(..., event == k) ~ ...)`. R's `coxph` defaults to Efron ties and this package to Breslow, so the comparison must set `ties` the same on both sides.

## 21. Fine-Gray subdistribution hazard

The Fine-Gray data transformation (`finegray_transform` in
`algorithms/survival/finegray.py`) follows R's `survival::finegray()`
implementation (`R/finegray.R`, `src/finegray.c`,
`noweb/finegray.Rnw`), including:

- R's integer-time-scale trick: shifting real-event times back by 0.2
  so that a tied censoring/event pair resolves the way Kaplan-Meier
  requires.
- IPCW weights from the censoring distribution G (and entry distribution
  H when there is left truncation), following Fine & Gray (1999) /
  Geskus (Biometrics 2011).
- KM computation via sorted binary searches: O(n log n) time, O(n)
  memory — no observation-by-time boolean matrices.

The transformed data feeds into an ordinary weighted Cox fit with
`cluster = <original subject>` (robust variance is mandatory because
the pseudo-observations are not independent across rows).

**Verified by:** `test_finegray_transform.py` — validated against
R's own bundled test cases (`survival/tests/finegray.R` /
`finegray.Rout.save`): the same 14-subject dataset, same row-expansion
pattern and case weights, reproduced to match R's saved values.
`test_phase4_r_comparison.py` additionally checks the full transform on
fresh, larger synthetic data (400 subjects) row-for-row against R's
actual `finegray()` output, plus the fitted Fine-Gray coefficients and
SEs.

**Current status (fresh run, 2026-09-20, R 4.3.3 / `survival` 3.5.8, same tie method on both sides).** On right-censored data the expanded rows, the IPCW weights (largest difference 7e-16), the coefficients (8 digits) and the robust standard errors (6 digits) agree with R's `finegray()` + `coxph()`. On **left-truncated** data the expanded `fgstart`/`fgstop` rows agree but the IPCW weights do not: 842 of 3,469 rows differ, by up to 0.445, and the fitted coefficients differ by 3.4e-3 (0.3670 in R, 0.3701 here). `test_finegray_transform.py::test_r_test3_left_truncation` fails for the same reason. The committed Phase-4 comparison tests also compare this package's default Breslow ties with R's default Efron, and `FineGrayPH`'s robust SE with R's model-based `se(coef)` (see the validation report). Treat `FineGrayPH` results with delayed entry as unvalidated.

## 22. Robust/sandwich (cluster-robust) variance

The sandwich estimator follows `survival::coxph`:

    V_robust = V_naive @ U.T @ U @ V_naive

where `U` contains score residuals collapsed within cluster. The score-
residue kernels are semantic ports of R's `coxscore2.c` (right-censored)
and `agscore3.c` (start/stop data), but cluster-aggregated on the fly
so memory is O(n_clusters * p) rather than O(n * p).

Activated by `CoxPH.fit(..., cluster=...)`, or by constructing `CoxPH(robust=True)` (each row is then its own cluster). When `cluster` is set,
`standard_errors_` reports the robust SE; `naive_covariance_` preserves
the model-based (naive) covariance for comparison.

Numba-compiled production kernels with pure-Python fallback, following
the same pattern as Phase 1-2's risk-set and tie-method kernels.

**Verified by:** `test_phase4_r_comparison.py::
test_robust_strata_truncation_clustering` — strata + left truncation +
clustering together (the one combination statsmodels.PHReg could not
confirm during development — it returned NaN), against R's
`coxph(..., cluster = cluster)`. Both robust SE and naive SE match.
`test_robust_variance.py` and `test_score_residuals.py` additionally
validate against statsmodels.PHReg and lifelines on simpler
configurations.

## 23. Time-dependent covariates: tmerge()

`data/timedep.py` provides `tmerge()` and `build_skeleton()` for
building (start, stop] counting-process datasets from a follow-up
skeleton plus covariate-change/event-occurrence records — R's
`tmerge()` / `survSplit()` use case.

Since this package has no formula interface, `tmerge()` takes plain
arrays plus `UpdateStream` objects grouped by update type (`tdc`,
`cumtdc`, `event`, `cumevent`), rather than R's non-standard-evaluation
`tdc()`/`event()` terms.

Every semantic rule (which side of a tie an update lands on, what
happens to an update before `tstart` or after `tstop`, how `cumtdc`'s
`init` combines with pre-`tstart` updates) was checked against R's own
bundled examples in `survival/tests/tmerge.R` / `tmerge.Rout.save`.

**Verified by:** `test_timedep.py` (R's bundled examples, verbatim) and
`test_phase4_r_comparison.py::test_timedep_tmerge_and_fit` (fresh
synthetic data: tmerge output row-for-row against R's actual tmerge(),
plus fitted coefficients and SEs on the merged dataset).

---

## What was *not* investigated / is known to differ

- **Exact ties** are not implemented (`NotImplementedError` with a
  docstring explaining the extension point — see
  `algorithms/survival/ties.py::ExactTies`). Both Breslow and Efron are
  implemented and validated (question 2); this package's *default* is
  still Breslow, matching the existing `phregSHR` R workflow, even
  though it differs from R's own default (Efron) — pass `ties="efron"`
  explicitly on both sides for a fair comparison against a plain
  `coxph()` call.
- **`na.action` alternatives** (listwise deletion, etc.) are not
  implemented — missing values always raise.
- Formula-interface parsing (`Surv(time, event) ~ x1 + x2`) is not
  implemented; a numeric design matrix is expected.
- **Fine–Gray with left truncation** does not reproduce R's IPCW weights (Section 21, status note).
- **`fit_intercept=True`** fits, but every `predict_*` method then raises `ValueError`.
- **No convergence warning:** check `converged_` (Sections 10–13).
- **Group lasso, provider-penalized and discrete-time estimators** have no R reference implementation in the test suite; they are covered by internal tests only.
- **Default ties differ from R** for every estimator (Breslow here, Efron in R); set them explicitly when comparing.
