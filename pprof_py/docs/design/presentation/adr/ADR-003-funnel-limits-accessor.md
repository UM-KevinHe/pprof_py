# ADR-003 — Funnel limits from the statistical layer (decision D1)

**Status:** accepted (D1); API confirmed (D10); discrete convention confirmed (D11) · 2026-09-30 · Evidence: `spikes/funnel_limits.py`, `spikes/out/funnel_limits_s4.csv`

## Context
Every current funnel computes its limits in plotting code; at N = 1,000 they contradict their flags (audit §4.2: 30 flagged providers on an exact limit, 70 unflagged providers outside RE limits, 30 flagged providers inside linear RE limits). S4 requires points outside the limits to be exactly the flagged providers.

## Decision
A `funnel_limits` accessor in the statistical layer, taking the same keywords as `test()` and built from the same nulls and decision rule, so agreement holds by construction:

| Test family | Construction | Plotted estimate | Precision axis |
|---|---|---|---|
| Count tests (`poibin_exact`, RE `exact`/`poibin_exact`, three-stage count tests) | For each provider, bisect the count domain using the *same null objects* `test()` builds (`null.tails(o, g0)`) and the same calibration (`null_mean`, `null_sd`) and decision rule; place limits at half-integers, `(o_hi − ½)/E` and `(o_lo + ½)/E` | O/E | expected count E |
| CoxPH `midp` | Same search with the package's `poisson_midp_zscore` and E from the test | O/E | E |
| Score | `1 + (μ ± c·σ)·√V₀/E` from the test's null expectation and variance | O/E | E²/V₀ |
| Wald with inversion intervals | Re-expressed from the test's own interval: `null + (T − ci_lower)`, `null + (T − ci_upper)` | estimate | 1/SE² |

Output: per-provider rows (`provider_id, observed, expected, estimate, precision, precision_kind, lower, upper, flag` per level) plus curve rows on a precision grid (`precision, level, lower, upper, null_group`), with `attrs` copied from `test()` plus the limit definition.

## Evidence (spike, outside the package)
* **0 mismatches** between "outside own limits" and `test()` flags in **22 of 22** configurations: logistic FE `poibin_exact`, score and Wald; linear FE (Student-t); logistic RE `poibin_exact`; CoxPH `midp`; theoretical and empirical nulls; N = 20 and N = 1,000 (CoxPH 300).
* **0 providers exactly on a limit** (half-integer placement).
* A single smooth Poisson(E) curve would misplace **34** (theoretical) and **23** (empirical) of 998 providers for logistic FE `poibin_exact`, and **39** of 1,000 for logistic RE `poibin_exact`: Poisson-binomial nulls are not functions of E alone. Score, Wald and Poisson-null curves are exact functions of their precision axis.
* Cost: about 6 s for count tests at N ≈ 1,000 with a Python loop over providers; vectorising one PMF per provider should remove most of it.

## Consequences
* Exact Poisson-binomial funnels draw per-provider limit marks (or the score funnel is the default for logistic FE); a smooth curve is only a reference there and is labelled as such.
* Extra reference levels (for example 99.8%) are curves only; flags exist only at the test's level, and the display says so.
* Empirical nulls give per-group curves (piecewise by `null_group`).
* Monte Carlo nulls (`resampling`, `bootstrap_exact`) are out of scope for the first version.
* **Confirmed:** half-integer limit placement for discrete tests (D11), and the API placement: function `pprof_py.inference.funnel_limits(model, **test_kwargs)` plus model methods on supported families (D10).

## Implementation (R3, 2026-10-01)
* **Placement.** `pprof_py/inference/funnel.py`: `funnel_limits(model, *args, **kwargs)` calls `model.funnel_limits(...)` (D10). Methods exist on `LogisticFixedEffectModel` (default `test_method="score"`, D11), `LogisticRandomEffectModel` (`poibin_exact` default, `exact`; Wald and resampling refused, ADR-004), `LogisticFERandomClusterModel` (`exact` default, `poibin_exact`), `LogisticThreeStageModel` (delegates to stage 3), `LinearFixedEffectModel` (Student-t Wald) and `CoxPH` (`midp` default and `exact`, D26). `LinearRandomEffectModel` has no method; the function raises `TypeError` citing this ADR.
* **Same test, nothing re-derived (D23).** An inert recorder (`inference/_recording.py`, a context variable) collects, inside the one `test()` call, the count test's observed counts, null objects, reference effect and alternative (`count_test`), the score test's O, E and V₀, and the linear Wald test's degrees of freedom. Outside `funnel_limits` it records nothing; 24 of 24 hashed outputs (`test()` across families and methods, standardized measures, data preparation) are bit-identical to `v0.5.0`.
* **Kernels.** `poibin_tails`, `integrated_poibin_tails` and `clustered_poibin_tails` were split into "count distribution" and "tails at a count" parts (`_poibin_probs`, `poibin_tails_all`, `_integrated_probs`, `_clustered_pmf`, `_pmf_tails`) with elementwise identical arithmetic; tests assert exact equality for every count. Poisson-binomial and row-mixture nulls compute all counts' tails from one distribution; cluster mixtures bisect over counts with the kernel's own tail sums. Decisions on hypothetical statistics call the package's `flags()` with an internal null holding each provider's fitted `null_mean` and `null_sd`.
* **Wald limits** use the calibration formula of `intervals(form="inversion")` directly, so `interval="scale_only"` does not change the funnel.
* **Output** (refines the sketch above). `FunnelLimits(test, providers, curves, attrs)`: `test` is the unchanged `test()` frame (D24); `providers` has `observed`, `expected`, `estimate`, `null_value`, `precision`, `lower`, `upper`, `flag`, `null_group` (limits at the test's level, `±inf` where unreachable, NaN when untested); `curves` is long (`null_group` with the test's dtype, `level`, `critical`, `test_level`, `precision`, `lower`, `upper`; 200 geometric grid points over each null group's precision range; empty when the calibration varies within a group); `attrs` adds `model`, `estimate_kind` (`ratio`, `effect`), `precision_kind` (`expected`, `inverse_null_variance`, `inverse_variance`), `limit_rule`, `curve_kind` (`exact`, `poisson_reference`, `unavailable`) and `levels`.
* **Self-check.** Every build raises unless each tested provider lies outside its limits exactly when flagged. For continuous statistics a provider whose calibrated z equals the critical value to within 1e-9 is moved to its flag's side by one floating-point step; none occurred in testing.
* **Evidence.** `tests/inference/test_funnel_limits.py` (39 tests: S4 and half-integer placement across families, nulls, alternatives, `critical`, subsets and levels; curve ends equal provider limits; kernel equality; inert recorder; refusals; a tampered flag must fail). Mutations: integer limits fail 16 tests; calibration-blind decisions fail the 2 tests that can detect them. Passes on Python 3.12 (pandas 3.0.6) and on the 3.10 floor stack (pandas 1.5.0, numpy 1.23.0, scipy 1.9.0).
* **Cost** at about 1,000 providers (CoxPH 300), over `test()` alone: score +0.01 s, `poibin_exact` +0.49 to +0.59 s, random-effect `poibin_exact` +0.42 s, linear Wald +0.00 s, CoxPH `midp` +0.06 s. `test(poibin_exact)` itself takes about 5.3 s, almost all of it exact-interval inversion.
* **Reference curves.** Evaluated at each provider's own E, a Poisson reference misplaces 34 (theoretical null) and 23 (empirical null) of 998 logistic FE providers relative to their exact limits, as in the spike; hence per-provider marks for exact count tests.
