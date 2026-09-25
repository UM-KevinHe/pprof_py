# Architecture

> This document covers the survival-model family architecture within
> `pprof_py`. All module paths are relative to `pprof_py/` (e.g.
> `models/survival/coxph.py` means `pprof_py/models/survival/coxph.py`).

## Layering

```
CoxPH (models/survival/coxph.py)                              <- user-facing, ProviderModel conventions, pandas in/out
    |
validate_fit_inputs (data/survival_validation.py)             <- pandas/mixed input -> plain numpy, with clear errors
    |
SurvivalData (data/survival_data.py)                          <- plain-numpy container, no pandas below this line
    |
cox_partial_likelihood (algorithms/survival/cox_likelihood.py) <- sums per-stratum contributions
    |
TieMethod.stratum_contribution (algorithms/survival/ties.py)   <- Breslow's or Efron's specific likelihood formula
    |
sweep_risk_sets (algorithms/survival/risk_sets.py)             <- who's at risk when (tie-method-agnostic)
    |
newton_raphson (algorithms/survival/optimization.py)           <- drives the above to convergence
    |
inference/survival/{inference,baseline,residuals}.py           <- everything downstream of a fitted beta
    (baseline.py and residuals.py also dispatch through TieMethod --
     see the note below on what actually happened when Efron was added)
```

Each arrow is a real module boundary a future change could stop at:

- **A Spark/distributed backend** (the long-term goal this package was
  explicitly built to support) would replace `sweep_risk_sets` and the
  loop in `cox_partial_likelihood` with a distributed reduction — each
  partition computing its own `StratumContribution` (log-likelihood,
  score, information — all naturally additive across strata _and_
  across partitions of a stratum, if a stratum is ever split) and
  summing across the cluster. `TieMethod`, `CoxPH`, `optimization.py`,
  and everything in `inference/survival/` would not need to change.
- **Exact ties** are a new `TieMethod` subclass (`algorithms/survival/ties.py`)
  — nothing in `risk_sets.py` or `optimization.py` needs to change,
  because _who is at risk_ and _how ties among risky people are scored_
  are different questions answered by different modules on purpose.
  This prediction held for Efron's _likelihood_ (`stratum_contribution`)
  exactly as designed — but adding Efron also showed the original version
  of this document understated the size of a "new TieMethod": Efron's
  baseline hazard and its martingale residuals each turned out to need
  their _own_ formula too (not R's `agsurv5.c`/`agmart3.c` reusing the
  likelihood's machinery), which meant adding a second abstract method
  (`TieMethod.baseline_hazard_increments`) and changing
  `inference/survival/baseline.py` from a Breslow-only function into one that
  dispatches through whichever tie method actually fit the model — a
  small, contained change, but a real one, and `models/survival/coxph.py` did
  need a one-line update (passing `ties=self.ties` through) that the
  original architecture sketch didn't anticipate. See
  `R_COMPATIBILITY.md`, question 2, for the full account, including
  why martingale residuals needed a third, separate algorithm on top of
  that (a literal port of `agmart3.c` rather than a formula reachable
  from the other two at all).
- **A numba-compiled inner loop** was exactly this seam being used as
  predicted: every hot per-stratum kernel (the risk-set sweep, both tie
  methods' likelihood/score/information accumulation, both tie methods'
  baseline-hazard increments, and the martingale-residual algorithm) now
  has an `@njit(cache=True)` implementation used automatically when
  numba is installed, with the original pure-Python version kept as a
  fallback and, more importantly, as a _paired reference_ --
  `pprof_py/tests/survival/test_engine_self_consistency.py` checks every compiled kernel
  against its own Python fallback at random coefficient values, not just
  against R. That extra check earned its keep in practice: the first
  attempt at this only wired the numba kernel into `BreslowTies`,
  leaving `EfronTies` calling the pure-Python sweep directly and
  therefore completely unaccelerated -- correct, but not what "add numba"
  was supposed to accomplish, and not something a purely
  correctness-focused test (which passed the whole time) would ever
  have caught. See `R_COMPATIBILITY.md`'s closing section and
  the root `README.md`'s Performance section for the full account.
- **Penalization / robust variance / clustering** slot into
  `optimization.py` (a penalty term added to the objective) and
  `inference/survival/inference.py` (a different covariance formula given the
  same fitted `beta`, `score`, and per-observation residual/leverage
  information) respectively, without touching the likelihood engine.

## The core algorithm, in one paragraph

For each stratum, sweep the distinct event times in ascending order,
maintaining a running `(S0, S1, S2) = sum(w*r), sum(w*r*X), sum(w*r*outer(X,X))`
over the current risk set. An observation is added the instant its
`start` is passed and removed the instant its `stop` is passed — since
`start < stop` always, every observation is added at most once and
removed at most once across the whole sweep, so the total work across
all event times is `O(n)` amortized pointer advances (each `O(p^2)` for
the `S2` outer-product update) plus `O(K)` loop iterations (K = number
of distinct event times), rather than the `O(n*K)` (or worse) cost of
recomputing each risk set from scratch. See `algorithms/survival/risk_sets.py`
for the full derivation, including exactly why the `(start, stop]`
half-open convention determines which comparison (`<` vs `<=`) is used
on each side.

`S2` is deliberately never materialized as a `(K, p, p)` or `(n, p, p)`
tensor — only a single running `(p, p)` matrix is kept at any moment —
which matters once `K` (distinct event times) or `n` grows large. Within
a single sweep, entries and exits between consecutive event times are
batched (one `searchsorted` per side, one vectorized reduction over the
batch) rather than processed one observation at a time — see
`algorithms/survival/risk_sets.py`'s docstring for the profiling that motivated
this: on a 200k-row, 3000-stratum fit, the naive one-`np.outer`-call-per-
observation version accounted for over 2.6 million individual NumPy
calls.

**Getting to many strata efficiently at all was a separate fix**, not
covered by the batching above: `cox_partial_likelihood`,
`compute_baseline_hazard`, and `martingale_residuals` each originally
found "this stratum's rows" via `strata_codes == s` inside a Python loop
over every distinct stratum — which rescans the _entire_ array once per
stratum (O(n \* n_strata) overall), the classic way an individually
"vectorized" operation still ends up being the accidental bottleneck
when it's repeated too many times. `utils/grouping.py::iter_stratum_indices`
sorts once and slices contiguous groups out of that sort instead
(O(n log n) regardless of stratum count), and is now the only way any of
those three functions iterate over strata.

The sweep is still **not** vectorized _across_ strata into a single pass
— it loops over strata (in Python, though each iteration now dispatches
into a numba-compiled kernel for the actual work), finding each one's
rows via the above. This is a deliberate choice, not an oversight: most
realistic workloads (including the ones benchmarked in
the root `README.md`, thousands of same-size facilities) are dominated by
_many small_ strata rather than a few huge ones, and a naive single-pass
vectorization across strata doesn't obviously help that shape of problem
without also batching multiple strata's event times together, which is
a bigger, riskier rewrite than a per-stratum numba kernel. The numba
port mentioned above (once it was correctly wired into _both_ tie
methods, not just Breslow) closed most of the gap this note originally
flagged as the next step; full cross-stratum vectorization remains the
one after that if a workload with many thousands of strata needs it.

## Why centering `X` needed a deliberate design decision

`models/survival/coxph.py::fit` mean-centers `X` before handing it to the
optimizer, purely for Newton-Raphson's numerical conditioning — and
then very deliberately uses the _original, uncentered_ `X` (and the
resulting `eta`) for every output that depends on the absolute scale of
the linear predictor: baseline hazard, residuals, and predictions. Get
this backwards — centering leaks into `baseline_hazard_`, say — and
`coef_`/`standard_errors_`/`log_likelihood_` all still look perfectly
correct (centering is provably inert for those; see
`utils/numerical.py`), while `baseline_hazard_` silently reports the
hazard at `X = mean(X)` instead of `X = 0`. This class of bug — every
number _looks_ plausible, cross-checks against a naive re-implementation
even pass, and it's still wrong relative to R — is exactly why
`R_COMPATIBILITY.md` exists as a standing, itemized document rather
than being left as implicit knowledge in the code: the offset/`basehaz`
finding there was exactly this shape of bug (see that document's
Section 6), caught only by fitting against real R output, not by
internal consistency checks alone.

The same shape of bug recurred, twice, while adding Efron ties: fixing
the likelihood alone left `coef_`/`standard_errors_`/`log_likelihood_`
all correct while `baseline_hazard_` (a different formula, per
`agsurv5.c`) and `martingale_residuals_` (a _third_ formula, per
`agmart3.c`) were both still silently wrong, for the same reason —
nothing about those outputs' own internal consistency changes when only
part of the underlying model is right. The practical lesson, stated
plainly for whoever adds `ties="exact"` next: matching R on one output
of a fitted model is never evidence that another output also matches,
and each needs its own check against real R, not an inference from the
others passing.

## Phase 3: Penalized regression and variable selection

Phase 3 adds two new user-facing estimators (`PenalizedCoxPH`,
`PenalizedCoxPHCV`) and one selection wrapper (`CoxPHSelector`), all
built on top of the Phase 1-2 engine without modifying it — confirming
the architecture's prediction that "penalization slots into
`optimization.py`... without touching the likelihood engine."

```
PenalizedCoxPH / PenalizedCoxPHCV (models/survival/penalized_coxph.py)
    |
    |── lambda path + CV orchestration
    |
fit_regularization_path (algorithms/survival/coordinate_descent.py)
    |── proximal-Newton outer loop (uses exact information matrix, not diagonal approx)
    |── cyclic coordinate descent inner loop (soft-thresholding for L1)
    |
cox_partial_likelihood (algorithms/survival/cox_likelihood.py)   ← UNCHANGED from Phase 1-2
    |
penalty math (algorithms/survival/penalty.py)
    |── weighted_column_scale (standardize=True: scale by weighted pop. SD, no centering)
    |── rescale_penalty_factors (sum to p, matching glmnet)
    |── soft_threshold, elastic_net_penalty_value
    |
deviance (statistics/deviance.py)
    |── saturated_log_likelihood (glmnet's coxnet.deviance formula)
    |── cox_deviance, deviance_ratio (for dev.ratio path reporting)
    |── cross-validation grouped deviance (Verweij & Van Houwelingen 1993)

CoxPHSelector (selection/selector.py + selection/criteria.py)
    |
    |── greedy forward/backward/both search
    |── AIC: extractAIC convention, matched to R's step() on coxph
    |── BIC: uses n_events (via nobs.coxph), NOT n_obs
    |── p-value: textbook SAS PROC PHREG convention
    |
CoxPH.fit() (called many times, one per candidate move)   ← UNCHANGED
```

Key design decisions:

- **Coordinate descent vs. Newton-Raphson**: the penalized objective is
  non-smooth (L1 term), so the plain Newton-Raphson in
  `optimization.py` cannot be used directly. `coordinate_descent.py`
  implements a proximal-Newton scheme: at each outer step, the exact
  (log-likelihood, score, information) triple from
  `cox_partial_likelihood` defines a quadratic approximation, which is
  then approximately minimized by cyclic coordinate descent with
  soft-thresholding. Mathematically the same class of algorithm as
  glmnet's own Cox solver, but using the dense information matrix rather
  than glmnet's diagonal approximation.
- **Separate files, not bolted onto `optimization.py`**: the mechanism
  differs (proximal Newton + coordinate descent vs. plain Newton-Raphson)
  even though both consume the same objective callable. Neither file
  duplicates any Breslow/Efron/strata/offset/weight/start-stop math.
- **`CoxPHSelector` is a pure loop over `CoxPH.fit()` calls**: it never
  recomputes a likelihood, score, or p-value itself — it only compares
  the outputs of fully R-validated `CoxPH` fits.

## Phase 4: Competing risks, robust variance, time-dependent covariates

Phase 4 adds three capabilities, each using the existing engine in a
different way:

```
CauseSpecificCoxPH (models/survival/competing_risks.py)
    |── fits one CoxPH per cause (event recoded: other causes → censoring)
    |── no new math — pure convenience wrapper

FineGrayPH (models/survival/competing_risks.py)
    |
finegray_transform (algorithms/survival/finegray.py)
    |── Fine-Gray subdistribution-hazard data expansion
    |── IPCW weights via Kaplan-Meier of censoring distribution
    |── R's integer-time-scale trick for tied censoring/event pairs
    |── validated against survival::finegray() row-for-row
    |
CoxPH.fit(..., sample_weight=fgwt, cluster=id)   ← UNCHANGED
    |
    |── cluster= triggers robust variance (below)

Robust/sandwich variance (inference/survival/robust.py)
    |
    |── V_robust = V_naive @ U.T @ U @ V_naive
    |── score residuals: ports of coxscore2.c (right-censored) and agscore3.c (start/stop)
    |── cluster-aggregated: memory O(n_clusters * p), never O(n * p)
    |── numba-compiled production kernels + pure-Python fallback
    |── activated by CoxPH.fit(..., cluster=...)

tmerge / survsplit (data/timedep.py)
    |── pure data reshaping: base skeleton + UpdateStream records
    |── supports tdc, cumtdc, event, cumevent update types
    |── chained calls (R's idiom: tmerge(tmerge(...), ...))
    |── validated against R's own bundled tmerge test cases
    |── output feeds directly into CoxPH.fit(start=, stop=)
```

Key insight: both competing-risks strategies reduce to an ordinary Cox
fit — cause-specific by recoding events, Fine-Gray by a weighted data
expansion — so no new likelihood engine was needed. The only genuinely
new statistical code is the sandwich variance (`robust.py`) and the
Fine-Gray transform (`finegray.py`); everything else reuses `CoxPH`
unchanged.

## Empirical-null calibration (inference/survival/empirical_null.py)

A post-model inference layer for provider profiling: given per-provider
standardized measures (e.g. SMR/SHR z-scores from the two-stage CoxPH
workflow), `fit_empirical_null` estimates a robust location and scale
(Huber or Tukey bisquare psi) and `adjust_empirical_null` re-calibrates
z-scores and p-values accordingly. Both run on the package's shared
empirical-null layer (`pprof_py.inference.empirical_null`: one
`MASS::rlm`-exact estimator and one quantile-grouping rule) with the
settings of their R counterparts. This sits entirely downstream of
fitted model outputs — it never touches the likelihood engine.

## Group lasso, provider-penalized and discrete-time estimators

Added after Phase 4; each is a thin layer over an existing engine.

```
GroupLassoCoxPH / GroupLassoCoxPHCV (models/survival/group_lasso_coxph.py)
    |── shares _PenalizedCoxPHBase / _PenalizedCoxPHCVBase with PenalizedCoxPH
    |── sparse-group proximal operator and group utilities (algorithms/survival/penalty.py)
    |── path solver (algorithms/survival/coordinate_descent.py::fit_group_regularization_path)
    |
cox_partial_likelihood (algorithms/survival/cox_likelihood.py)   <- UNCHANGED

ProviderPenalizedCoxPH (models/survival/provider_coxph.py)
    |── outer layer: one-step Newton update of provider effects gamma
    |     (algorithms/survival/provider_effects.py), clamped to median(gamma) +/- bound
    |── inner layer: the penalized beta solver above; both warm-started along the lambda path

DiscreteSurvival / DiscreteSurvivalCV (models/survival/discrete_survival.py)
    |── person-period expansion, logistic hazard, baseline parameters alpha_k
    |     (algorithms/survival/discrete_survival.py)
    |── penalty utilities from algorithms/survival/penalty.py

ProviderPenalizedDiscreteSurvival (+CV) (models/survival/provider_discrete_survival.py)
    |── three layers: provider effects, baseline hazard, penalized covariates
    |── reuses the *logistic* provider-effect update (algorithms/logistic/provider_effects.py)
        and the family-independent algorithms/penalty.py
```

None of these has an R reference in the test suite. The Cox-based ones inherit correctness of the likelihood from `CoxPH`; the
penalty and provider layers are covered by internal tests only.

## Data preparation and diagnostics

- `data/survival_validation.py` — `validate_fit_inputs` (the input contract) and `SurvivalDataError`.
- `data/timedep.py` — `tmerge`, `survsplit`, `build_skeleton`, `UpdateStream` (pure reshaping, not exported at the package root).
- `diagnostics/survival/preflight.py` — `preflight_report` / `PreflightResult`; `diagnostics/survival/validate_against_r.py` — the
  Python-vs-R harness for user data.
- `selection/` — `CoxPHSelector` plus the `aic` / `bic` criteria.

## Module notes for maintainers

- `algorithms/survival/cox_likelihood.py` and `algorithms/survival/partial_likelihood.py` are byte-identical. `CoxPH` and
  `algorithms/survival/__init__.py` import `cox_likelihood`; treat it as canonical and `partial_likelihood` as a redundant copy.
- Two penalty / coordinate-descent implementations exist. The survival Cox estimators import `algorithms/survival/penalty.py` and
  `algorithms/survival/coordinate_descent.py`; the logistic, linear and provider-discrete models use the family-independent
  `algorithms/penalty.py` and `algorithms/coordinate_descent.py`.
- Numba-compiled kernels appear in `risk_sets.py`, `ties.py`, `inference/survival/residuals.py`, `inference/survival/robust.py`,
  `algorithms/survival/provider_effects.py` and the coordinate-descent modules, each with a pure-Python fallback.
- Reference pages for every public estimator: `reference/` (start with `reference/coxph.md`).
