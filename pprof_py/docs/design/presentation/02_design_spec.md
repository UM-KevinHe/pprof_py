# pprof_py presentation layer — Phase 2 design spec

**Date:** 2026-09-30 · **Base:** `main` = `v0.5.0` = `d3d92a1` · **Phase:** 2 (design) · **Gate:** passed — **approved 2026-10-01** with decisions D9–D20 (`DECISIONS.md`). Changes from this proposal: Python floor 3.10 (CI matrix 3.10/3.14, §17 item 7); house style D19.
**Companions:** `DECISIONS.md` (D1–D8), `adr/ADR-001…008`, `01_audit.md`, spikes and renders under `spikes/` in the kit.
**Labels:** [run] measured in a spike or harness this phase · [read] from code · [assumed] · [proposal] a design choice.

---

## 0. Summary

**Core idea.** Everything the presentation layer draws comes from `test()` (one schema, 0 S3 violations across all families, audit §1) plus one new statistical-layer accessor, `funnel_limits` (D1). Renderers are thin, deterministic and accessible. Tables use a library-independent specification with hand-written renderers. **The MVP adds no runtime dependency.**

**Maintainer decisions applied:**

| | Decision | Where it lands |
|---|---|---|
| D1 | Funnel-limit accessor in the statistical/result layer | ADR-003; spike: **0 S4 mismatches in 22/22 configurations, 0 providers on a limit** [run] |
| D2 | No generic RE funnel; only justified cases | ADR-004; funnels only where the flagging test is a count test of the plotted O/E (logistic RE `poibin_exact`/`exact`, three-stage); Wald-on-BLUP fits get a capability error |
| D3 | Test CI, early | ADR-007; `round_ci_tests.diff` delivered now, rehearsed locally (§12.4) |
| D4 | Zero-event providers explicit; no silent continuity correction | ADR-005; explicit status and encoding, never draw the solver's clamp, flags unchanged |

**MVP (§16):** shared infrastructure + funnel + interval plot with volume panel + provider table. Acceptance: survival Chapter 10's report and the logistic FE tutorial reproduced in a few lines each, S3/S4-consistent.

**Decisions needed at this gate:** §17 (items 1–3 block Phase 3).

---

## 1. Principles (from the brief, sharpened by the audit)

1. One source per display: estimate, interval, flag and limits come from the same `test()` call (or the accessor built on it). Never pair an interval from `calculate_confidence_intervals()` with a flag from `test()` (audit §4.2: up to 219 disagreements at N = 1,000).
2. Limits only from `funnel_limits()` (D1). No statistics in plotting code.
3. Every provider has exactly one status from the S6 taxonomy (§6.1), plus the zero-event attribute (D4). Nothing is dropped silently.
4. Provenance (level, reference, null model, test method, estimator type, exclusions) travels with the data and appears on every output.
5. No pyplot, no global state, byte-reproducible exports, accessible encodings.
6. Additive: validated numerics change only through approved accessors (§7).

---

## 2. Display taxonomy and spec sheets

| Display | Tier | Answers | MVP | Notes |
|---|---|---|---|---|
| Funnel | 1 | Who deviates beyond volume-driven noise? | **yes** | test-consistent limits (D1); families per §6.3 (D2) |
| Interval plot ("caterpillar") + volume panel | 1 | How large and how uncertain are estimates vs the reference? | **yes** | all families with `test()` |
| Provider summary table | 1 | The numbers, with denominators, intervals, flags | **yes** | §3 |
| Coefficient forest | 1 | Covariate effects with CIs | Phase 4 | logistic RE blocked until `summary()` has CIs (§7) |
| Observed vs expected | 2 | Where O departs from E, by volume | Phase 4 | |
| Between-provider variation | 2 | How much real variation? | Phase 4 | `sigma_`/`random_effect_sd_`, `profile_sigma`, IUR |
| Shrinkage (FE vs RE pairs) | 2 | How much does pooling move each provider? | Phase 4 | needs both fits; γ vs α alignment documented |
| Reliability | 2 | How well does the measure separate providers? | Phase 4 | `decile_table()`, `iur_groups_` |
| Null calibration | 2 | Is the theoretical null credible? | Phase 4 | `z_raw`, `null_mean`, `null_sd`, `null_group` |
| Flag stability | 2 | How fragile are flags? | Phase 4 | re-run `test()`; `sigma_sensitivity` |
| Multi-measure small multiples | 2 | Comparisons across measures | Phase 4 | `ProfileCollection` |
| Data-quality panel | 2 | Who is missing or unreliable, and why? | Phase 4 | blocked until exclusions are recorded (§7) |

**Not built:** bare rank charts, Top-N/Bottom-N lists, traffic-light grids, pie, radar, 3D, dual axes, gradients, composite scores, and RE funnels on Wald-on-BLUP tests (D2).

### 2.1 Funnel

1. **Question.** Which providers' outcomes depart from what the reference predicts by more than the test's sampling variation explains, and how does that depend on precision?
2. **Quantity and source.** From `funnel_limits()` (ADR-003): per provider `observed`, `expected`, `estimate` (O/E for count and score tests; γ or the difference for Wald tests), `precision` and `precision_kind`, `lower`/`upper` per level, and `flag` copied from the same `test()` call.
3. **Uncertainty.** The acceptance region at the test's level. Extra reference levels (for example 99.8%) are drawn as reference curves and labelled as such, because flags exist only at the test's level. Poisson-binomial exact tests get per-provider limit marks. A smooth curve there would misplace 23–39 of ~1,000 providers [run].
4. **Denominator.** The x-axis is the test's precision, labelled by kind: expected count E (count tests), E²/V₀ (score), 1/SE² (Wald). Patient volume appears in the table; the legend gives full counts including zero-event providers.
5. **Reference.** Horizontal line at the null value (1.00 or 0) with a direct label. The reference definition (for example "γ₀ = −1.96, median of provider effects") goes in the provenance footnote.
6. **Misreadings and mitigations.**
   - *A point near a limit as proof:* limits labelled with the test and level, and half-integer placement means no point sits on a line.
   - *Overdispersion under the theoretical null:* null model in the footnote; empirical-null variant (per-group curves); link to null-calibration diagnostics.
   - *Reading limits as multiplicity-adjusted:* the footnote states the per-provider level and what the metadata says about adjustment.
   - *RE funnels that contradict their flags:* not offered (D2).
   - *Zero-event providers:* explicit (D4).
7. **Static/interactive.** Static in the MVP. Interactive hover (id, N, O, E, p, flag) later; its static equivalent is this figure plus the provider table.
8. **Scale.**
   - 10–2,000 providers: all points as vectors.
   - Above 2,000: rasterized point layer; flagged providers drawn on top and never hidden.
   - Labels only for `highlight=` providers, or for flagged providers when there are at most 30.
   - 50,000: same, with curves unaffected; the legend always reports full N.

### 2.2 Interval plot with volume panel

1. **Question.** How large and how uncertain are provider estimates relative to the reference?
2. **Quantity and source.** `test()`'s `estimate`, `ci_lower`, `ci_upper`, `flag`, `null_value` (one call), on the test's scale (effect scale, or measure scale via `test_standardized()`). Volume from `attrs["provider_size"]` or `provider_sizes_`; CoxPH `observed`/`expected`/`person_time`.
3. **Uncertainty.** Per-provider intervals from test inversion (S3-consistent by construction). Intervals are drawn as segments from lower to upper, not as errors around the estimate, so shifted empirical-null intervals render correctly (fixes audit M7).
4. **Denominator.** Side panel of bars, labelled with its kind (patients N, expected events, person-time).
5. **Reference.** Vertical line at the null value; definition in the legend and footnote.
6. **Misreadings and mitigations.**
   - *Sort order read as rank:* the axis says "Providers, ordered by estimate", never "Rank".
   - *Non-overlap read as a pairwise difference:* docs and footnote.
   - *Small providers' wide intervals:* the volume panel shows them.
   - *Zero-event providers:* ◀/▶ at the axis edge with the exact interval to the edge; clamped values excluded from the axis range (D4).
   - *Estimator type:* "unshrunken fixed effects" vs "shrunken BLUP" always stated.
7. **Static/interactive.** Static in the MVP.
8. **Scale.**
   - Up to 150 providers: labelled.
   - Above 150: dense mode (no labels, thin rasterized intervals, flagged highlighted, `highlight=` labels).
   - 50,000: rasterized line collection.
   - Legend and alt text always report full N [run: dense prototype at N = 998].

*Implemented in R6 (D34–D36):* rows are labelled up to 60 providers, with row height from the theme's tick size; the volume panel shows the profile's denominator by kind; the reference value is part of the x-axis label; providers without a finite estimate get an interval segment only from the axis edge to a finite bound.

### 2.3 Provider summary table
Spec in §3. Its key mitigation: flags never appear without their level, reference, null model and test method, because the footnotes are generated from `attrs`.

### 2.4 Coefficient forest (Phase 4)
Covariate estimates and intervals from `summary()`, mapped per family (audit §5). The axis is exponentiated (OR/HR) only via a documented, tested presentation derivation. Model order is preserved by default. The caption warns against causal reading and against comparing covariates on different scales. Logistic RE needs CI columns in `summary()` first (§7).

### 2.5 Tier 2 summary
Each Tier 2 display gets a full spec sheet before Phase 4. The capability notes in the table above come from the audit's data-availability matrix; the tiers stand, with three changes: RE funnels are restricted (D2), the logistic RE forest waits on `summary()` CIs, and the data-quality panel waits on an exclusion record.

---

## 3. Tables

### 3.1 Table specification (the public API) [proposal]
```python
@dataclass(frozen=True)
class Column:
    key: str                    # role-bound source field, e.g. "estimate_ci"
    header: str                 # may contain a footnote marker
    role: Literal["id", "count", "estimate", "interval", "p_value", "flag", "text", "percent"]
    align: Literal["left", "right", "center"] = "right"
    formatter: Callable[[pd.Series], pd.Series] | None = None
    spanner: str | None = None  # grouped header

@dataclass(frozen=True)
class TableSpec:
    columns: tuple[Column, ...]
    rows: pd.DataFrame          # presentation data, one row per provider (or group)
    caption: str
    notes: tuple[str, ...]      # generated from provenance and the S6 statuses
    source_note: str
    row_groups: str | None = None
```

### 3.2 Renderers
- **HTML:** single file, inline CSS, `<caption>`, `<th scope>`, `<tfoot>` notes, print stylesheet.
- **Markdown:** GFM with an alignment row.
- **LaTeX:** booktabs; `longtable` above 40 rows; escaped specials; required packages documented.
- **Plain text.**
- **Excel:** optional extra (§17 item 6). Numeric cells with number formats, frozen header, provenance sheet.
- **Tidy DataFrame** with provenance in `attrs`.

[run] The hand-written Markdown, HTML and LaTeX renderers are byte-identical across processes. Great Tables 1.0.0 and `pandas.Styler` HTML are not; their LaTeX is (ADR-002).

*Implemented in R7 (D39–D42):* `TableSpec` (columns with roles and markers, formatted cells, tidy values with Excel number formats, caption, notes, source note, provenance) with hand-written renderers; `provider_table` generates its notes from the profile (estimates and intervals, flags and test, `NE`, `NT`, `NI`, `S`, extra decimals, `—`); golden files pin the HTML, Markdown, LaTeX and text output.

### 3.3 Table taxonomy
1. Provider results — **MVP**.
2. Covariate effects.
3. Multi-measure with grouped headers.
4. Flag summary.
5. Methods and provenance — **MVP, auto-generated appendix**.
6. Reliability by decile.
7. Data quality and exclusions.
8. Flag sensitivity.

### 3.4 Formatting rules (one `formatting` module for tables, axes, tooltips and alt text)
Pure, vectorized functions with explicit precision:
- `fmt_count`: thousands separators.
- `fmt_number`: fixed or significant digits; true minus U+2212; negative zero normalised.
- `fmt_interval`: `1.12 (0.95–1.31)`; "to" when any bound is negative.
- `fmt_p`: `<0.001`, never `0`.
- `fmt_ratio`: two decimals by default.
- `fmt_rate`: the scale goes in the header.
- `fmt_flag`: ▲ ▼ ● NT (● is the figures' solid "not different" marker; ○ means "not tested" in figures, D21).
- Missing-value symbols: `—` not applicable, `NE` no finite estimate, `NT` not tested, `S` suppressed, `NI` no interval.

S12 rounding-collision rule: if a flagged provider's displayed bound equals the displayed null value, or an unflagged one's displayed bound excludes it, add digits until the display agrees, up to a cap, then footnote.

Sample: `spikes/out/tables/provider_table.md`. **Caution, by design:** the spike's footnote "a" was hand-written and is wrong; the interval is symmetric, i.e. Wald-type on the ratio scale. The spike also first rendered the default measure (a rate) under an "O/E ratio" header until `measure="indirect_ratio"` was passed. Both errors are exactly what generated headers and footnotes prevent: in production every header unit and footnote comes from `attrs` (`measure`, `transform`, `interval`, `level`, `alternative`, `null_model`, `reference`).

---

## 4. Design tokens (`Theme`)

Immutable `Theme` objects; users derive new themes, there is no styling kwarg sprawl. Presets: `publication` (default), `notebook`, `report`.

**Status encodings:** ADR-008 (hue + marker + fill + label; dark PuOr pair #B35806 / #542788).
- Contrast against white: 4.87 / 10.38; not different 3.36.
- ΔL\* between above and below: 20.7.
- Minimum colour-vision ΔE: 38.8 [run].

**Typography:** DejaVu Sans, shipped with Matplotlib, so no system-font dependency.

| Role (pt) | publication | notebook | report |
|---|---:|---:|---:|
| title | 8.5 | 12 | 13 |
| axis label | 7.5 | 11 | 11 |
| tick, legend, annotation, footnote | 7 | 10 | 10 |

**Sizes:** single column 85 mm, double 175 mm (named `size="single" | "double"`); heights per display. Lines: axes 0.6 pt, intervals 0.8 pt (0.35 pt dense), reference 0.8 pt solid black, limits 0.7 pt #4D4D4D, dashed for 95% and dotted for 99.8%, all directly labelled. Markers (pt², publication): above/below 18, not different 7, not tested 10, off-scale 22.

**Measured:** minimum text size 7.0 pt in all four prototypes at final size [run]. Renders: `spikes/out/theme2/proto_{funnel,caterpillar}_N{20,1000}.{svg,pdf,png}`, `proto_cvd_sheet.png`, `proto_sheet2.png`.

**Layout rules learned from the spike** (each becomes a render test):
1. Legend and provenance live in layout-managed rows sized from their line counts. Footnotes are wrapped to the measured figure width. An overflowing footnote collapsed the dense interval plot to a sliver in run 1 [run].
2. Direct labels at line ends are de-overlapped.
3. Log axes use plain-number 1-2-5 tick formatters.
4. The main axes keep at least 50% of the figure width.
5. Remaining prototype defects to fix in the MVP: the dense plot's legend row overprints the first provenance line; the funnel's "1.00" label collides with the y tick. *Fixed in R5:* legend and footnote are separate sub-figures sized from measured text (D30); the reference value is labelled inside a label strip right of the data, and lines end before it.

---

## 5. Architecture

```text
statistical engine ─▶ results: test(), test_standardized(), calculate_standardized_measures(), summary(),
                               funnel_limits() [new, D1]
                    ─▶ presentation data: ProviderProfile / ProfileCollection (immutable, provenance, statuses, capabilities)
                    ─▶ renderers: figures (matplotlib.figure.Figure) · tables (TableSpec renderers) · reports
                    ─▶ export: SVG / PDF / PNG · HTML / Markdown / LaTeX / text / Excel
```

**Layout (ADR-001):**
```text
pprof_py/
    presentation/          new home; public namespace exports funnel, caterpillar, provider_table, Theme, ProviderProfile, …
        data/              ProviderProfile, ProfileCollection, adapters, capability declarations
        formatting/        numbers, intervals, p-values, labels, alt text
        theme/             Theme, tokens, presets (plotting.style re-exports from here)
        figures/           funnel, caterpillar, … and the FigureResult wrapper
        tables/            TableSpec and renderers (excel renderer imports XlsxWriter lazily)
        reports/           Report (Phase 4)
        _synthetic.py      private deterministic generator shared by tests, docs, benchmarks
    plotting/              backward-compatible façade (plot_caterpillar, plot_funnel, style)
    plotting/{logistic,linear}/   model mixins become thin, lazily importing delegates
```

**Import rules, enforced by tests:**
- `models/`, `algorithms/`, `inference/`, `measures/` never import `presentation` or `plotting` at module level.
- `import pprof_py` does not import `matplotlib.pyplot`.
- Library code never calls `plt.show()` or `print`.

---

## 6. Presentation data

### 6.1 `ProviderProfile` [proposal]
Immutable wrapper around one DataFrame indexed by `provider_id`, with canonical roles:
- `estimate`, `scale`, `se`, `ci_lower`, `ci_upper`, `null_value`
- `flag`, `p_value`, `z_raw`, `z_adjusted`, `null_group`
- `observed`, `expected`, `denominator`, `denominator_kind`, `precision`, `precision_kind`, limits per level
- `status`, `zero_events`

It also carries `provenance` (dict built from `attrs`, model class, package version, estimator type, exclusions, seed) and `capabilities` (set).

**Status taxonomy (S6 + D4).** Exactly one primary status per provider, plus one attribute:

| Status | Source | Encoding |
|---|---|---|
| above / below / not different | `flag` = +1 / −1 / 0 | ▲ / ▼ / ● (ADR-008) |
| not tested | `flag` NA | ○ hollow, "NT" |
| no interval | `ci_*` NaN where the method has none | point without segment, "NI" footnote |
| suppressed | user `min_volume` (default off) | not drawn; counted; "S" in tables |
| excluded by data preparation | exclusion record (§7) | counted in footnote |
| not applicable | e.g. no test for the family | "—" |
| *attribute:* zero events / no finite estimate | degeneracy status (§7) | □ outline on ratio scale; ◀/▶ off-scale on effect scale; "NE" in effect columns |

*Implemented in R4 (D29):* the primary statuses are above, below, not different, not tested and suppressed (the last only under `with_min_volume`); "no interval" (`has_interval`), zero events, all events and no finite estimate are attributes, so no flag is ever hidden; excluded providers are the profile's `excluded` record, counted by `status_counts()`. `ProfileCollection` waits for Phase 4 (D28).

### 6.2 Adapters
- `ProviderProfile.from_model(model, *, test_method=None, reference=None, null_model=None, level=None, alternative=None, limits=False, data=None)` calls `test()` once, adds denominators from `calculate_standardized_measures()`, and calls `funnel_limits()` when `limits=True`. CoxPH needs `data`, because its `test()` takes the data again.
- `ProviderProfile.from_test(result, *, model=None, denominators=None)` consumes a `test()` frame and copies `attrs` into provenance.
- `ProviderProfile.from_frame(df, *, roles)` is the escape hatch. It validates and warns on S3/S4 violations, one-sided alternatives, or missing intervals.

*Implemented in R4 (D27):* denominators and counts come from the funnel limits of the same test, the CoxPH test's own columns, or the model's own data (`degenerate_providers`, `provider_sizes_`), not from `calculate_standardized_measures()`, whose expected counts use their own reference. `from_model` passes its arguments to `test()` or `funnel_limits()` unchanged (CoxPH data included).

The per-family field vocabulary (interval option names and columns, `summary()` columns, `sigma_` vs `random_effect_sd_`, CoxPH attrs) is the mapping table in audit §5; adapters own it, and renderers never see family differences.

### 6.3 Capability matrix after D1, D2 and D4

| Display | Logistic FE | Logistic RE | Linear FE | Linear RE | Three-stage / FE random cluster | CoxPH |
|---|---|---|---|---|---|---|
| Funnel | ✓ score (exact curve) and `poibin_exact` (per-provider marks) | only `poibin_exact`, `exact` (D2) | ✓ Wald, precision 1/SE² | ✗ (D2) | ✓ count tests [read; verify per class in R3] | ✓ `midp` |
| Interval plot + volume | ✓ | ✓ (estimator: shrunken BLUP) | ✓ | ✓ | ✓ | ✓ |
| Provider table | ✓ | ✓ (measure-scale intervals only from `test_standardized`) | ✓ | ✓ | ✓ | ✓ |
| Zero-event status | ✓ `at_bound()` | needs §7 accessor | needs §7 accessor | needs §7 accessor | needs §7 accessor | from `observed` (O = 0) |

Penalized models are out of the MVP; they were not assessed.

### 6.4 Validation and errors
- Duplicate `provider_id`, empty input, or a non-unique index: `ValueError`.
- An unsupported combination raises `CapabilityError(ValueError)` naming what is missing and how to get it.
- Example: `funnel(re_model)` → "LogisticRandomEffectModel flags providers with a Wald test on shrunken estimates (test_method='wald'); a funnel of O/E would contradict those flags (ADR-004). Use caterpillar(), or test_method='poibin_exact' or 'exact' for a count-test funnel."
- User frames violating S3/S4: `UserWarning` listing the offending providers; the display marks them instead of hiding the conflict.

---

## 7. Statistical-layer accessors (approval needed; D1 approved in principle)

| Accessor | Purpose | Proposal | Status |
|---|---|---|---|
| `funnel_limits` | S4 by construction (D1) | ADR-003: function `pprof_py.inference.funnel_limits(model, *, levels=(0.95,), **test_kwargs)` and a method on supported families; per-provider and curve rows; `attrs` from `test()` | implemented (R3; D10, D11, D13) |
| Degeneracy status | D4 across families | `at_bound()` raises an informative `TypeError` outside its families (today: `KeyError` on RE, all-True on linear); new `degenerate_providers(model)` returning zero- and all-event status for binomial families | implemented (R3; D10, D11, D13) |
| Exclusion record | S6 "excluded" | `DataPrep` and fitted models record `excluded_providers_` (id, reason, n) | implemented (R3; D10, D11, D13) |
| CoxPH `attrs` | S2 provenance | `attrs["measure"] = "indirect_ratio"`, `attrs["reference"] = 1.0` | implemented (R3; D10, D11, D13) |
| Logistic RE `summary()` CIs | forest without recomputation | add `ci_lower`, `ci_upper` at `level` | needs approval (Phase 4) |

Each accessor ships with invariant tests: S4 for `funnel_limits` across families and nulls, plus a negative control that perturbs one limit and must fail.

---

## 8. API proposal

```python
from pprof_py.presentation import ProviderProfile, Theme, caterpillar, funnel, provider_table

# 1. From a fitted model: one test() call, limits from the accessor (D1)
fig = funnel(fe, test_method="poibin_exact", reference="median", levels=(0.95, 0.998),
             theme="publication", size="single")
fig.save("funnel.svg")                 # deterministic; alt text embedded as <title>/<desc>
fig.alt_text                           # "Funnel plot of 998 providers: 95 above and 44 below …"

# 2. Explicit profile, reused across displays
profile = ProviderProfile.from_model(fe, test_method="poibin_exact", reference="median", limits=True)
caterpillar(profile, volume="n", highlight=["P00032"], size="double").save("intervals.pdf")
tbl = provider_table(profile, columns=["n", "observed", "expected", "estimate_ci", "flag"])
tbl.save_html("providers.html"); tbl.to_latex(); tbl.to_markdown()
tbl.to_excel("providers.xlsx")         # needs pprof_py[excel]; otherwise ImportError naming the extra

# 3. Custom theme by derivation
house = Theme.publication().derive(font_sizes={"tick": 7.5})
```

**Signatures** (all keyword-only after `source`; `source` is a fitted model, a `test()` frame, or a `ProviderProfile`):
```python
funnel(source, *, test_method=None, reference=None, null_model=None, level=None, levels=(0.95, 0.998),
       providers=None, highlight=None, theme="publication", size="single", title=None) -> FigureResult
caterpillar(source, *, scale="effect", volume="auto", dense="auto", providers=None, highlight=None,
            theme="publication", size="double", title=None) -> FigureResult
provider_table(source, *, columns=None, measure=None, digits=None, min_volume=None) -> TableResult
```

**Return objects.**
- `FigureResult`: `.figure` and `.axes` (documented escape hatch), `.save(path, *, format=None, dpi=300)`, `.to_svg()`, `.alt_text`, `.long_description`, `.provenance`, `_repr_svg_`/`_repr_png_`.
- `TableResult`: `.spec`, `.to_html()`, `.to_markdown()`, `.to_latex()`, `.to_text()`, `.to_excel()`, `.to_frame()`, `.save_*()`, `_repr_html_`.

Neither touches pyplot or leaves figures open.

**Model methods.** Existing `plot_*` methods become delegates (§13). New code uses the namespace functions.

---

## 9. Export, determinism and notebooks (ADR-006)
- Renderers build `matplotlib.figure.Figure` directly; the theme's `rc_context` wraps build and save.
- Per-save metadata strips dates, and `svg.hashsalt` is fixed.
- [run] SVG, PDF and PNG were byte-identical across two processes with no environment variable and no global rcParams (Matplotlib 3.11.2), and again on the declared floor: Python 3.9.25 with Matplotlib 3.5.0, numpy 1.23.0 and pandas 1.5.0 (prototype funnel).
- PNG defaults to 300 dpi. Sizes are in mm.
- Notebook reprs render without `plt.show()`.
- Byte identity is guaranteed within one environment, because Matplotlib and FreeType versions change glyph outlines.

## 10. Accessibility plan (all automated)
- Contrast ≥ 3:1 for every status ink and line.
- Colour-vision ΔE thresholds from ADR-008 under protan, deutan and tritan simulation, and ΔL\* in grayscale.
- A unique (marker, fill) pair per status.
- Minimum text size at final size ≥ 7 pt.
- Alt text present and consistent with status counts, embedded as SVG `<title>`/`<desc>`.
- HTML tables use `<caption>` and `<th scope>`, with footnote markers that resolve.
- Every interactive feature has a static equivalent (Phase 4).

The colour-vision simulation needs `colorspacious` in tests. Either add it to the `dev` extra (needs approval) or vendor the small Machado 2009 matrices (my recommendation: vendor them, no dependency).

## 11. Docs strategy
- A new "Presentation" section:
  - a decision guide ("Which display answers my question?");
  - one page per display and table, built from its spec sheet with interpretation notes and counterexamples (league table vs funnel on the same data);
  - pages on theme, export and accessibility;
  - a migration guide;
  - a gallery;
  - API reference.
- Images are generated at build time by the already-loaded `matplotlib.sphinxext.plot_directive` (`conf.py:33`, unused today), using the deterministic theme and `_synthetic.py`.
- Printed outputs are verified as the maintainers already do. The build must stay at or below the 4-warning baseline.
- Drift fixes from audit §4.7 land with the pages they touch.

## 12. Test strategy

### 12.1 Layers
1. Formatting unit tests (and property tests only if Hypothesis is approved; not requested now).
2. Presentation-data contracts: values equal their sources exactly; order and id types preserved; NA flag becomes not tested; NaN interval becomes no interval; providers subset honoured.
3. Statistical invariants: S3 on profiles and S4 on `funnel_limits` across logistic FE/RE, linear FE, three-stage and CoxPH, theoretical and empirical nulls, with the perturbation negative control.
4. Figure structure on Matplotlib artists, not pixels: point counts per status, segment endpoints equal to the intervals, reference position, legend entries, marker/fill per status, minimum font, axes width share, alt-text counts, no open figures.
5. Byte determinism: two subprocesses, `slow` marker.
6. Table golden files (HTML, Markdown, LaTeX, text) and Excel numeric-cell checks.
7. Accessibility metrics (§10).
8. Smoke tests per family; scale ladder 10 → 50,000 (`slow`).

### 12.2 Rules
Agg backend, fixed seeds, no network, no R, no system fonts. Optional-dependency tests skip with a reason.

### 12.3 Layering tests
The import-direction and "no pyplot on import" tests from §5.

### 12.4 CI (D3, ADR-007)
`round_ci_tests.diff` adds `.github/workflows/tests.yml`. Local rehearsal results:

All runs: fresh clones of `d3d92a1`, python-build-standalone interpreters, `pip install -e ".[dev]"` (lowest-direct via `uv pip install --resolution lowest-direct`), `MPLBACKEND=Agg`, whole suite in one process [run].

| Environment | Resolved stack | Result | Failing set |
|---|---|---|---|
| Python 3.9.25, newest resolvable | numpy 1.24.4, pandas 2.3.3, scipy 1.13.1, Matplotlib 3.9.4, numba 0.57.1 | 1 failed, 688 passed, 1 skipped (131 s) | `test_group_null_point.py::test_penalty_factor_is_inert_under_the_pure_group_lasso[GroupLassoLinear]` |
| Python 3.10.21, newest resolvable | numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, Matplotlib 3.10.9, numba 0.68.0 | **689 passed**, 1 skipped (137 s) | none |
| Python 3.14.7, newest resolvable | numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, Matplotlib 3.11.2, numba 0.68.0 | 1 failed, 688 passed, 1 skipped (122 s) | `test_setup_logger` only: environmental, as in the Phase 0 baseline |
| Python 3.9, declared minimums pinned | — | install fails (ResolutionImpossible) | `numba==0.56.*` conflicts with `fast_poibin` 0.4.0/0.4.1, which need `numba>=0.57,<0.58` on Python < 3.12 |
| Python 3.9.25, lowest-direct | numpy 1.23.0, pandas 1.5.0, scipy 1.9.0, Matplotlib 3.5.0, numba 0.57.0, statsmodels 0.13.0, pytest 7.0.0 | 2 failed, 687 passed, 1 skipped (194 s) | `test_setup_logger` (environmental); `logistic/test_sigma_uncertainty.py::test_sigma_sensitivity` |

Notes:
* **3.9 failure.** `assert b.coef_path_[0][1] == 0.0` gets −1.06e-16 with numpy 1.24.4; it passes with numpy 1.23.0 and ≥ 2.2. Two readings: either the exact-equality assertion is stricter than floating point guarantees, or the null-point path computes the zero-penalty column instead of fixing it at 0, as the test's comment says it should. Not fixed here (brief §3.2); see §17 item 7.
* **Floor-stack failure.** `assert not np.allclose(lower, upper)` in `test_sigma_sensitivity`: the σ-sensitivity flags at both ends of the interval came out identical on scipy 1.9, statsmodels 0.13, pandas 1.5 and numpy 1.23. Not investigated further.
* **`test_setup_logger`** passes wherever pandas 2.x installs the `tzdata` package, and should pass on GitHub runners, which ship system tzdata [assumed].
* **Per-file 3.9 run** (each file in its own process under a 2.7 GB cap): the same totals across all 52 files, with peak memory 225–410 MB per file. The container restarts during earlier attempts were not caused by the suite.
* **Actions and checks.** The workflow uses `actions/checkout@v7` and `actions/setup-python@v7`; both run on Node 24 per their `action.yml`. `setup-python@v5`, used by `docs_deploy.yml`, runs on Node 20 (separate update recommended). actionlint 1.7.12 reports no findings, and the diff applies silently on a fresh clone of `test/mixed-effect`.
* **Expected first CI run:** the 3.14 job green; the 3.9 job red on the group-lasso test unless the runner's floating-point result happens to be exactly 0 [assumed].

## 13. Migration
1. **MVP:** new namespace functions. The existing FE `plot_funnel`, `plot_provider_effects` and `plot_standardized_measures` become delegates (single rendering path) with signatures preserved where practical. Visible changes go in the changelog: shapes, colours, provenance footnote, consistent flags, zero-event handling, return of a figure object rather than `None` or `(fig, ax)` (wrapped so that tuple unpacking keeps working).
2. **RE funnels (D2):** behaviour per ADR-004 and the deprecation policy (§17 item 4).
3. **`plot_caterpillar`:** default interval columns follow `PROVIDER_TEST_COLUMNS` (`ci_lower`/`ci_upper`). Callers passing `lower`/`upper` keep working through explicit arguments; the default change is announced per policy.
4. **Styling kwargs** map onto theme derivation and are deprecated per policy.
5. The docs migration table (old call → new call) ships with the delegates.

## 14. Performance budgets

| Operation | Baseline [run] | Budget [proposal] |
|---|---|---|
| Interval plot render + save, PNG, 10,000 providers | 0.82 s (current `plot_caterpillar`) | ≤ 3 s |
| Same, 50,000 providers | 2.32 s | ≤ 10 s |
| Funnel render + save, PNG, 10,000 | 0.45 s (standalone funnel) | ≤ 3 s |
| SVG size, interval plot, 10,000 (dense, rasterized layers) | 3.4 MB (current, vector) | ≤ 2 MB |
| `funnel_limits`, count tests, ~1,000 providers | ≈ 6 s (spike, Python loop) | revised in R3 (D24): ≤ 1 s over `test()`; measured +0.42 to +0.59 s. `test(poibin_exact)` itself takes ≈ 5.3 s, almost all exact-interval inversion, unchanged by design |
| `funnel_limits`, count tests, 10,000 | — | ≤ 10 s over `test()` (not yet measured) |
| Provider table HTML, 10,000 rows | — | ≤ 2 s, ≤ 5 MB |
| `import pprof_py` | 1.6–1.8 s, imports pyplot | no pyplot import; no regression |

## 15. Risks

| Risk | Mitigation |
|---|---|
| Accessor complexity: discrete tests, grouped empirical nulls, Monte Carlo nulls | Reuse the null objects `test()` builds; invariant tests; Monte Carlo nulls deferred |
| Matplotlib 3.5 floor lacks newer layout APIs | The prototype funnel (constrained layout, gridspec, 1-2-5 log ticks) renders and exports byte-identically on 3.5.0 [run]; the interval plot is not yet checked there. Avoid post-3.5 APIs or gate them; floors job once ready (§17 item 8) |
| pandas 1.5–3.x semantics (Copy-on-Write; audit R5) | No chained in-place operations; CI on both ends |
| Layout failures at extremes (long ids, many flagged, huge N) | Layout rules of §4 as render tests; scale ladder |
| Statistical-layer approvals delay the MVP | Order rounds so infrastructure proceeds while accessor approvals are pending (§16) |
| Byte identity across environments | Documented as per-environment; tests compare within one environment |
| Scope creep into Tier 2 | MVP fixed; Tier 2 only after the Phase 3 review |

## 16. MVP proposal

**Scope:**
- Formatting, `Theme` with presets, `ProviderProfile` with adapters for logistic FE, linear FE, three-stage and CoxPH (logistic and linear RE for the interval plot and table), capability gating, provenance, statuses, alt text, deterministic export.
- The `funnel_limits` accessor (D1) and the approved §7 accessors.
- Funnel, interval plot with volume panel, provider table (HTML, Markdown, LaTeX, text; Excel if approved).
- FE plot-method delegation, doc stubs, a demo script, renders at 20, 1,000 and 10,000 providers.

**Rounds** (one self-contained, verified `git apply` diff each; strict order):
- R1 CI workflow (**delivered now**).
- R2 skeleton: theme, formatting, layering tests, lazy mixin imports. Bit-identical numerics.
- R3 `funnel_limits` and approved accessors, with invariant tests and a docs page.
- R4 presentation data: adapters, validation, capabilities.
- R5 funnel and `FigureResult`: render and determinism tests.
- R6 interval plot with volume panel and dense mode.
- R7 `TableSpec` renderers and the provider table, with golden files.
- R8 delegates, changelog, gallery stub, samples, demo, acceptance checks.

R2 does not depend on accessor approvals; R3 does.

**Acceptance:**
1. Survival Chapter 10 §10.6 rebuilt from `CoxPH.test()` with the new table and funnel in a few lines. The hand-built report's inconsistency (facility 32, p = 0.044 with an interval containing 1) disappears.
2. Logistic FE tutorial figures reproduced.
3. S3/S4 invariant tests pass across families.
4. Determinism and accessibility tests pass.
5. Renders at 20, 1,000 and 10,000 pass visual QA (§3.5 of the brief).

Then **STOP** for the maintainer's review of rendered output.

## 17. Decisions needed at this gate

1. **[blocks Phase 3] Approve this spec and the MVP scope and round order (§16).**
2. **[blocks R3] `funnel_limits` API:** function `pprof_py.inference.funnel_limits(model, *, levels, **test_kwargs)` plus methods on supported families; output schema per ADR-003. *Recommend yes.*
3. **[blocks R3] Discrete-limit convention:** half-integer acceptance boundaries matching mid-p decisions; per-provider limit marks for Poisson-binomial nulls; score funnel as the logistic FE default (exact curve). *Recommend yes.*
4. **Deprecation policy** (brief §8.6). *Recommend (b):* `DeprecationWarning` now, removal at a stated version (for example 0.7.0). Under D2: logistic RE `plot_funnel` re-routes to the count-test funnel for `poibin_exact`/`exact` and warns (then raises) for `wald`; linear RE `plot_funnel` warns, then raises.
5. **Statistical-layer additions (§7):** (a) degeneracy status, (b) exclusion record, (c) CoxPH `attrs`, (d) logistic RE `summary()` CIs. *Recommend (a)–(c) in the MVP, (d) in Phase 4.*
6. **Optional extra `excel` = XlsxWriter.** *Recommend yes* (zero dependencies, reproducible). **Interactive HTML not in the MVP.** *Recommend defer to a Phase 4 ADR.*
7. **Python floor.** *Recommend raising the floor to 3.10* (needs approval, brief §3.2). Evidence [run]:
   * 3.10 passes the whole suite, while 3.9 fails one exact-equality test (§12.4).
   * Python 3.9 reached end of life in October 2025.
   * `fast_poibin` 0.4.2 already requires 3.10, so 3.9 users are held at `fast_poibin` 0.4.1, numba 0.57.x and numpy ≤ 1.24.
   * Current Matplotlib requires 3.11.

   If approved, change `requires-python`, the classifiers, the CI matrix (`"3.10"`, `"3.14"`) and `docs_deploy.yml`, which builds on 3.9. If 3.9 stays, the group-lasso test needs a decision: a tolerance, or zeroing the column exactly.
8. **CI floors job.** *Not yet.* The declared minimums cannot be installed together (`numba>=0.56` vs `fast_poibin>=0.4`; the effective numba floor is 0.57), and the lowest-direct resolution fails `test_sigma_sensitivity` (§12.4). *Recommend:*
   1. Raise the numba floor to 0.57 (packaging change; needs approval).
   2. Have the test's author check the σ-sensitivity result on the floor stack.
   3. Then add a lowest-direct job, sketched in ADR-007.
9. **Fonts:** no bundling; DejaVu Sans from Matplotlib. *Recommend yes.*
10. **Ranks:** no rank or percentile intervals and no ranked output in the MVP. *Recommend yes.*
11. **Outlets and house style:** defaults are 85/175 mm, ≥ 7 pt, DejaVu Sans, dark PuOr status pair. *Input wanted:* any CMS, journal or group conventions.
12. **Workflow:** design docs at `pprof_py/docs/design/presentation/` (verified unpublished and unpackaged); a branch `presentation/mvp`; one diff per round, as in earlier work. *Confirm.*

---

## Appendix A — Evidence index

| File (kit) | What it shows |
|---|---|
| `spikes/funnel_limits.py`, `spikes/out/funnel_limits_s4.csv` | ADR-003: 22/22 configurations with 0 S4 mismatches and 0 providers on a limit; Poisson-curve misplacement counts; timings |
| `spikes/theme_spike2.py`, `spikes/out/theme2/` | palette metrics, prototype renders, CVD sheet, determinism across processes, minimum font size |
| `spikes/table_spike.py`, `spikes/out/tables/` | table renderers compared, determinism, Excel libraries |
| `spikes/dep_eval.py`, `spikes/dep_eval.json` | live PyPI evidence for ADR-002 |
| `ci/` | CI-matrix rehearsal logs (`matrix.log`, per-environment `install.log`, `pytest.log`, `junit.xml`) |
| `01_audit.md` | Phase 1 audit (baseline, findings, availability matrix) |
