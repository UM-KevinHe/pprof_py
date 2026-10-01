# ADR-008 — Status encodings and palette

**Status:** accepted (Claude, encodings within §7) · 2026-09-30 · Evidence: `spikes/out/theme2/palette_metrics.csv`, `proto_cvd_sheet.png`

## Decision

| Status | Hue | Marker | Fill | Label |
|---|---|---|---|---|
| Above reference (flag +1) | #B35806 | ▲ | solid | "Above reference (k)" |
| Below reference (flag −1) | #542788 | ▼ | solid | "Below reference (k)" |
| Not different (flag 0) | #8C8C8C | ● small | solid | "Not different (k)" |
| Not tested (flag NA) | edge #4D4D4D | ○ | hollow | "Not tested (k)" |
| Zero events / no finite estimate | #000000 | □ outline (ratio scale) or ◀/▶ at the axis edge (effect scale) | outline | "Zero events …" / "No finite estimate …" |

Reference line: black, solid, 0.8 pt, direct label. Control limits: #4D4D4D, dashed (95%) and dotted (99.8%), direct labels. Labels are direction-neutral; an optional `polarity` setting may add wording but never changes colour, shape or flag.

## Evidence

| Pair or metric | Value |
|---|---|
| Contrast vs white: above / below / not different | 4.87 / 10.38 / 3.36 |
| ΔL\* above vs below (grayscale tone) | 20.7 |
| ΔE (CAM02-UCS) above vs below: normal / protan / deutan / tritan | 53.5 / 52.9 / 57.3 / 38.8 |
| min ΔE of above or below vs not different, all four conditions | 27.8 |

Six candidate pairs were scored; four passed contrast and CVD thresholds; only this pair also separates in grayscale tone. Brown/azure (#994F00 / #006CD1) is the runner-up, with larger CVD distance but ΔL\* 4.3.

## Test thresholds (automated in `tests/presentation/test_accessibility.py`)
Contrast ≥ 3:1 against the background for every status ink and line, text ≥ 4.5:1. Colour differences are CIE76 ΔE in CIELAB under the Machado (2009) simulations, vendored as three matrices so no dependency is added (they reproduce colorspacious to 1e-16; D22): above vs below ≥ 40 in every condition (shipped palette: worst 59.1, tritan), above or below vs not different ≥ 25 (worst 33.7), ΔL\* above vs below ≥ 15 (20.7). Every status has a unique (marker, fill) pair. A red/green pair fails (ΔE 7.3 under deuteranopia), which the tests check as a negative control. The CAM02-UCS figures above come from the Phase 2 spike.
