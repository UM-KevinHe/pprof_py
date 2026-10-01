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
