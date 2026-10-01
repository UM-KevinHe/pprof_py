# ADR-004 — Funnels for random-effect models (decision D2)

**Status:** accepted (maintainer, D2) · 2026-09-30

## Context
The logistic and linear RE funnels draw O/E limits while their flags come from a Wald test on shrunken BLUPs; at N = 1,000, 70 of 106 points outside the logistic RE limits are not flagged and 30 flagged linear RE providers sit inside the limits (audit §4.2). The maintainer decided: no generic funnel for RE models, only methodologically justified cases.

## Decision
A funnel is offered only when the flagging test is a test of the plotted count at the reference effect:

| Model / `test_method` | Funnel? | Reason |
|---|---|---|
| Logistic RE `wald` (default) | **No** | tests the BLUP, not O/E |
| Logistic RE `poibin_exact` | Yes | count test with plug-in null at the reference (spike: 0 S4 mismatches at N = 20 and 1,000) |
| Logistic RE `exact` | Yes (needs one cluster factor) | count test with a cluster-mixture null |
| Logistic RE `resampling` | Not in v1 | Monte Carlo null; limits would need simulation |
| Linear RE (Wald on BLUP) | **No** | no count test exists |
| Three-stage / FE-random-cluster count tests | Yes | provider effects are fixed; count test with cluster mixture |

Unsupported combinations raise a capability error that names the test, explains why a funnel would contradict its flags, and points to the interval plot or to a justified `test_method`.

## Consequences
The existing `LogisticRandomEffectModel.plot_funnel` and `LinearRandomEffectModel.plot_funnel` change behaviour; per D12 they warn (`DeprecationWarning`) now and stop at removal in `0.7.0`, except that logistic RE count-test methods re-route to the justified funnel.
