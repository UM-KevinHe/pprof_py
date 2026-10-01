# ADR-005 — Zero-event providers (decision D4)

**Status:** accepted (maintainer, D4) · 2026-09-30

## Context
Zero-event (and all-event) providers have no finite fixed-effect estimate. Today they receive `flag = 0` from every test method, are drawn as "Expected", and their Wald intervals (about ±100 on the log-odds scale) stretch the caterpillar axis so far that the other 980 providers collapse into a line at N = 1,000 (audit M1, M2). The maintainer decided: show them explicitly, with no silent continuity correction.

## Decision
* "Zero events" / "no finite estimate" is an explicit, separately counted status attribute. It does not change the test's flag, which is still shown.
* Ratio scale (funnel, table): O/E = 0 is finite and is plotted at 0 with an extra outline marker and a legend count ("Zero events, O/E = 0 (k)"); tables print `0.00` with the test's interval.
* Effect scale (interval plot, table): no point is drawn at the solver's clamp. An off-scale marker (◀ or ▶) sits at the axis edge, and the exact one-sided interval is drawn from the edge to its finite bound. Clamped values never enter the axis range. Tables print `NE` with a footnote.
* No `+½` or other continuity correction anywhere in the presentation layer, and no silent drop; footnotes give counts.
* Alt text counts these providers.

## Statistical-layer gap
`at_bound()` is the only source today and is logistic-FE-specific: it raises `KeyError: 'gamma'` on RE models and returns every provider on `LinearFixedEffectModel` (audit M4). A family-aware degeneracy accessor (or `at_bound()` raising an informative error outside its families) is needed and is approved for the MVP (D13). Until then the presentation layer requests the status only from families where it is defined.

## Evidence
Prototype renders (`spikes/out/theme2/`) show 18 zero-event providers explicitly at O/E = 0 in the funnel, and as ◀ markers with exact intervals to −∞ in the dense interval plot; the axis is set by finite estimates only.
