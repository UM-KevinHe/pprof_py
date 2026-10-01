# ADR-006 — Deterministic export without global state

**Status:** accepted (Claude) · 2026-09-30 · Evidence: `spikes/out/theme2/theme_spike.json`

## Context
Default Matplotlib SVG embeds a timestamp and randomly salted ids, and default PDF embeds a creation date (audit §4.5). Global fixes (`SOURCE_DATE_EPOCH`, global rcParams) violate the "no hidden global configuration" rule.

## Decision
* Renderers build `matplotlib.figure.Figure` directly; no pyplot, no global figure registry, nothing left open.
* The theme is applied through `matplotlib.rc_context` while building **and** while saving. The figure object keeps its theme and re-enters the context in `.save()`, `.to_svg()` and the notebook reprs. The first spike run saved outside the context and its SVG was not reproducible; that is why this rule exists.
* Per-save metadata: SVG `{"Date": None}`, PDF `{"CreationDate": None}`; `svg.hashsalt` is fixed in the theme's rc.
* Alt text and long description are injected as SVG `<title>`/`<desc>` by deterministic string insertion.
* Fonts: DejaVu Sans, which ships with Matplotlib, so there is no system-font dependency (confirmed, D17). SVG text is drawn as paths by default (fidelity), with an option for editable text; PDF uses Type 42 embedding.

## Evidence
Two separate processes with no `SOURCE_DATE_EPOCH` and no global rcParams produced byte-identical SVG, PDF and PNG for the prototype funnel (Matplotlib 3.11.2). The same check passed on the declared floor: Python 3.9.25, Matplotlib 3.5.0, numpy 1.23.0, pandas 1.5.0.

## Consequences
Byte identity is guaranteed per environment (Matplotlib and FreeType versions change glyph outlines); tests compare outputs within one environment.

## Implementation (R5, 2026-10-01)
* `FigureResult.to_bytes()` renders inside the theme's `rc_context` with `metadata={"Date": None}` (SVG), `{"CreationDate": None, "ModDate": None, "Title", "Subject"}` (PDF) and `{"Title", "Description"}` (PNG); SVG output gets `role="img"`, `aria-labelledby` and `<title>`/`<desc>` with fixed ids.
* **Layout freezing (D30).** Constrained layout restarts from the current positions on every draw and is not idempotent: saving one figure four times gave four different SVGs, differing in the fourth decimal of coordinates. Renderers therefore draw once on an Agg canvas inside the theme context and then switch the layout engine off (`set_layout_engine("none")`, or `set_constrained_layout(False)` on Matplotlib 3.5). Verified: repeated saves and two separate interpreters give identical SVG, PDF and PNG bytes, on Matplotlib 3.11.2 and 3.5.0.

## Implementation note (P4-6): output must not depend on earlier drawing

A figure built after another figure was drawn or saved in the same process could differ from the same figure built
first. Drawing leaves FreeType state behind, so text measured later can differ in its last bit; constrained layout
then placed axes and sub-figures ~1e-16 apart. That changed Matplotlib's SVG clip and marker ids (they hash the
full-precision coordinates) and could leave a sub-figure at -0.0, which the PDF writer prints as "-0". Earlier
determinism tests compared figures built after other drawing, so they never saw a cold build.

Fix: `freeze_layout` rounds axes positions and sub-figure boxes to 1e-10 of the figure and normalises negative zero;
the scaffold rounds its measured heights; `FigureResult` renames the hashed SVG ids in document order. Evidence: the
reliability figure's SVG differed in 5 of 12 fresh processes and the observed-versus-expected PDF in 2 of 3; after
the fix, 0 of 12 and 0 of 10. `tests/presentation/test_determinism_after_save.py` builds every figure type cold in
its own interpreter, then again after saving, and compares SVG, PDF and PNG. Because the effect is intermittent, one
run detects a regression only with some probability.
