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
