(provider_penalized_cox)=
# Chapter 12 — Provider-Penalized Cox Models

## 12.1 Fulfilling Chapter 4's own "not currently available" note

[Section 4.10](04_indirect_standardization_smr_shr) described a
one-stage alternative to the two-stage SMR pipeline: a single Cox
model with an explicit fixed effect $\alpha_k$ per facility, fit by
alternating a $\beta$ update with a per-facility $\alpha$ update — and
said plainly that this package "does not currently provide a
ready-made estimator for this one-stage \[...\] model." `pprof_py`
0.4.0 does now. `ProviderPenalizedCoxPH` is not a literal
implementation of Section 4.10's specific fixed-point sketch (that
sketch has no penalty at all, and identifies $\alpha$ via a
population-SMR-sum constraint rather than median-clamp bounding — see
Section 12.4), but it's the same lineage: provider effects and
covariate effects, estimated jointly, in one alternating fit, with the
covariate side now penalized so it also does variable selection. The
[provider-penalized logistic chapter](../logistic/provider_penalized_logistic)
covers the same idea for a binary outcome and derives the shared
two-layer objective in more depth — this chapter assumes you've read
it and covers what's specific to the partial likelihood.

## 12.2 Architecture: reusing Chapter 1's engine, not reinventing it

Where `ProviderPenalizedLogistic` hand-rolls its outer loop,
`ProviderPenalizedCoxPH` inherits from `_PenalizedCoxPHBase` — the same
mixin `PenalizedCoxPH` (Chapter 8) and `GroupLassoCoxPH` (Chapter 11)
use — and its $\beta$-update step calls `fit_single_lambda`/
`fit_single_lambda_group` directly: the exact proximal-Newton engine
with KKT stationarity checking and step-halving that Chapter 1
describes for the logistic and linear families, with $\gamma$ folded
in as an **offset** rather than a special case the solver needs to
know about:

$$
\eta_{ij} = \gamma_{k(ij)} + \mathbf{x}_{ij}^\top\beta, \qquad
\text{(offset } = \gamma_{k(ij)} \text{, solved exactly as Chapter 6's time-dependent offsets are)}
$$

Each lambda alternates, up to `max_provider_iter` times (default
**20**):

1. **Update $\gamma$** given the current $\beta$: a per-facility
   Newton step, using a Minorization-Maximization information
   approximation suited to the partial likelihood's risk-set
   structure (`algorithms/survival/provider_effects.py`) rather than
   the plain Newton information the logistic chapter's per-provider
   update uses — a different technique for a fundamentally similar
   job — followed by the same median-clamp bounding
   (`provider_bound`, default `10.0`).
2. **Update $\beta$** given the current $\gamma$, by calling
   `fit_single_lambda`/`fit_single_lambda_group` with $\gamma$ baked
   into the offset.
3. **Check joint convergence**: both the largest single-facility
   $\gamma$ change (against `provider_tol`, default `1e-6`) *and* the
   relative $\beta$ change (against `outer_tol`, default `1e-7`,
   consistent with every other Cox penalized-path class in this
   guide) must clear their thresholds. `converged_path_` is
   explicitly the AND of both conditions, not just the $\beta$ solver's
   own flag.

The practical payoff of reusing proven machinery instead of a
hand-rolled loop: on a cohort of comparable size to the one the
[provider-penalized logistic chapter](../logistic/provider_penalized_logistic)
uses, `converged_path_` here is `True` at all 100 lambdas, against 22
of 100 there.

## 12.3 The running example: the ESRD cohort, unmodified

[Chapter 0's](00_start_here) cohort, exactly as generated — no new
columns needed, since facility-level quality was already built into
`facility_quality[facility_id]` from the start, and this chapter is
finally the one that puts that quality signal to direct use.

```python
from pprof_py import ProviderPenalizedCoxPH

X = cohort[["age", "sex", "diabetes", "comorbidity_count", "vintage_years"]]

model = ProviderPenalizedCoxPH(alpha=1.0)
model.fit(X, duration=cohort["time"], event=cohort["death"], provider=cohort["facility_id"])
```

`fit()`'s provider argument is keyword-only.  Both `provider=` and
`provider_id=` are accepted (the logistic chapter's class historically
used only `provider_id`; both names now work on both classes).
Omitting it raises a clear, explicit error:

```python
ProviderPenalizedCoxPH(alpha=1.0).fit(X, duration=cohort["time"], event=cohort["death"])
# ValueError: provider must be provided (per-observation provider identifier array)
```

Unlike the logistic class, `coef_at()` is inherited and works exactly
as [Chapter 1](../logistic/penalized_logistic) describes — continuous,
log-lambda interpolation, not a discrete index:

```python
lam = model.lambda_path_[50]
model.coef_at(lam)
```
```
age                  0.0470
sex                 -0.0450
diabetes             0.2067
comorbidity_count    0.1772
vintage_years       -0.0457
```

`predict_linear()`/`predict_partial_hazard()`/`predict()` (inherited,
covariate-only, no provider term) accept `lambda_value=` and it works
correctly — confirmed directly. Provider
effects have their own resolution method, `_resolve_gamma()`, and it
behaves *differently* from `coef_at()` in one respect worth knowing:
it snaps to the **nearest lambda on the grid** rather than
interpolating. Both are exercised together by
`predict_linear_with_provider()`, the one method genuinely specific to
this class:

```python
model.predict_linear_with_provider(X.iloc[:3], cohort["facility_id"].iloc[:3], lambda_value=lam)
# gamma[provider] + X @ coef_at(lam), confirmed equal to a manual
# computation of exactly that sum, to full floating-point precision
```

## 12.4 Reading the path

```python
model.n_nonzero_path_[[0, 10, 20, 30, 40, 50, 70, 99]]
# array([0, 1, 3, 4, 5, 5, 5, 5])
```

All five covariates are in by `which=40`; `sex` — which has no real
effect in this cohort's data-generating process — is the one that
never fully separates from zero at any lambda tested here, alongside
weak, near-zero coefficients for the others by the unpenalized end.

## 12.5 Provider effects: facility ranking, and its honest limits

```python
gammas = model.predict_provider_effect(lambda_value=lam)   # array, provider_labels_ order
```

Correlation between estimated and true (known, since this is a seeded
synthetic cohort) facility quality is `0.413` at this lambda —
noticeably better than the
[logistic chapter's](../logistic/provider_penalized_logistic) `0.217`,
consistent with Cox's fuller use of follow-up time (not just a binary
30-day outcome) and a covariate set with a real, if imperfect, case-mix
signal here. It is not a clean 1-to-1 recovery, and looking at
individual facilities rather than the aggregate correlation shows
exactly the kind of noise [Section 4.9](04_indirect_standardization_smr_shr)
warns about: the facility with the single highest estimated $\hat\gamma$
here (`0.641`) has a true quality of only `0.125` — elevated, but not
dramatically so — while a facility with true quality *below* average
(`-0.479`) still lands with the second-highest estimated $\hat\gamma$
(`0.615`). With roughly 100 patients per facility and only ~300 total
events across the whole cohort, no single facility's estimate carries
enough information to trust in isolation — precisely the argument for
looking at reliability (IUR) and confidence intervals, not point
estimates alone, before flagging any one facility as a true outlier.

```python
import numpy as np
np.exp(gammas)   # hazard ratio relative to the median facility
```
A facility with $\hat\gamma = 0.641$ has a hazard ratio of
`exp(0.641) ≈ 1.90` relative to the *median* facility in this fit —
readable the same way survival Chapter 4's SMR is, but not numerically
the same quantity: this is a hazard ratio against a median-clamped
reference facility from a jointly-penalized one-stage fit, not an
observed-over-expected ratio from Chapter 4's offset-consistent
two-stage pipeline. Don't report one where the other was asked for.

## 12.6 There is no `ProviderPenalizedCoxPHCV`

Unlike `ProviderPenalizedLogisticCV`, this family has no built-in
cross-validated wrapper at all — `pprof_py.__all__` exports
`ProviderPenalizedCoxPH` alone. As in the
[group lasso linear chapter's](../linear/group_lasso_linear) equivalent
gap, the fix is a manual fold loop, built from the same pieces
`GroupLassoCoxPHCV` (Chapter 11) uses internally — provider-level fold
assignment (every observation from one facility in the same fold, so
no facility's own data leaks between train and validation), and
per-fold deviance via `cox_partial_likelihood` and `cox_deviance`:

```python
import numpy as np
from pprof_py.algorithms.survival.cox_likelihood import cox_partial_likelihood, precompute_stratum_indices
from pprof_py.statistics.deviance import saturated_log_likelihood, cox_deviance

rng = np.random.RandomState(0)
n_folds = 5
provider_arr = cohort["facility_id"].values
unique_provs = np.unique(provider_arr)
prov_fold = {p: rng.randint(0, n_folds) for p in unique_provs}   # whole facility -> one fold
fold_id = np.array([prov_fold[p] for p in provider_arr])

full = ProviderPenalizedCoxPH(alpha=1.0, n_lambda=30).fit(
    X, duration=cohort["time"], event=cohort["death"], provider=provider_arr,
)
lambda_path = full.lambda_path_
# ... fit one ProviderPenalizedCoxPH per fold at lambda_path, score each
# lambda's held-out deviance the same way GroupLassoCoxPHCV does internally ...
```
```
lambda_min (5-fold manual CV): 0.002322
coef at lambda_min:
  age                0.0447
  sex                0.0000
  diabetes           0.1472
  comorbidity_count  0.1547
  vintage_years     -0.0287
```

`sex` — correctly, since it has no real effect in this cohort's
data-generating process — is the one covariate this manual CV loop
zeroes out entirely.

## 12.7 What's next

Chapter 13 moves to discrete-time survival models — a standalone
family, not a `PenalizedCoxPH` extension, for data where events are
only observed to occur within an interval rather than at an exact
time. The
[provider-penalized logistic chapter](../logistic/provider_penalized_logistic)
this chapter paired with is the last of Deliverable 3; from here, the
package's remaining undocumented surface is the discrete-time survival
family and the infrastructure/utility reference pages.
