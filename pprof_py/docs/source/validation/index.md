(validation_overview)=
# Validation overview

How closely does `pprof_py` reproduce the R implementations it replaces? This section answers that question with three
kinds of evidence, kept separate because they answer different questions.

| Evidence | Question it answers | Where |
|---|---|---|
| **Package test suite** (1,026 tests) | Does the code still match the frozen R outputs committed under `tests/data/`? | runs on every push (`.github/workflows/tests.yml`) |
| **Live comparisons** on synthetic data | Do the R package and `pprof_py`, run side by side on the same machine and the same input, compute the same numbers for every provider? | {doc}`logistic_fixed_effect` and the [pprof-validation](https://kevintan701.github.io/pprof-validation/) record |
| **Real-data parity** inside the CMS data environment | Do the Python and R pipelines produce the same facility flags on production data under production settings? | summarised below and in {doc}`logistic_fixed_effect`; the survival family in {doc}`../survival/VALIDATION_REPORT` |

A live comparison traces every number to a version, a commit, a data hash and a call; real-data parity covers the case
mix, the extreme providers and the empirical-null grouping that synthetic data cannot. Together they also localise any difference: when the fits agree live, a real-data
difference can only come from a setting of the test, as the variance note below illustrates.

## Precision at a glance

```{figure} ../_static/validation/precision_ladder.svg
:alt: Largest absolute difference between R and Python per measure and component, on a log axis with precision bands
:width: 100%

Largest absolute difference between the R and Python values, per measure and component. Shaded bands: below 1e−10 is
the floating-point limit; 1e−10 to 1e−5 reflects library or operation-order effects; 1e−5 to 1e−3 reflects convergence
tolerance or rounding of the R output; above 1e−2 would call for investigation. Nothing sits in the last band.
```

## Real-data parity by measure

Ten CMS ESRD measures were validated on FY2027 production data inside the CMS data environment (CDR, Databricks). The
R reference outputs were produced by the package authors with the production R/C++ code (R 4.3.2, survival 3.5.8); the
Python side ran `pprof_py` 0.7.0 single-node on a 128 GB driver. Only aggregate numbers leave that environment.

| Measure | Family | Scale | Key precision (max \|R − Py\|) | Flags agree | Runtime |
|---|---|---|---|---|---|
| PPPW | logistic fixed effect | 7,408 providers, 4 covariates | β 7.2e−13 · γ 1.0e−3 · z 3.9e−3 | 7,408 / 7,408 | seconds |
| SFR | logistic fixed effect + grouped Huber empirical null | 7,485 providers, 26 covariates | β 8.3e−17 · γ 2.7e−15 | 7,483 / 7,485 (99.97%) | seconds |
| AOH | logistic fixed effect, binomial (`n_var`) | 7,462 providers, 142 covariates | β 5.0e−4 (R output rounded to 3 decimals) · z 0.12 | 7,458 / 7,458 | seconds |
| SRR | three-stage (FE → GLMM → NR + Gauss–Hermite) | 7,247 providers, 74 covariates | β 5.0e−16 · γ 6.5e−5 · SRR 7.0e−5 | — | — |
| SMR | two-stage stratified Cox | 4.8 M rows, 57 covariates | β 1.1e−10, 9/9 checks | — | 144 s, peak 17.4 GB (R ≈ 13 min) |
| SHR | weighted Cox + offset + interactions | 857 K rows, 123 covariates | β 1.8e−14, 7/7 checks | — | 88 s, 5.4 GB |
| STrR · FYSMR · FYSWR | Cox variants | 240–380 K rows | β 3e−9 · 7e−8 · ratio 2e−11 | — | 12–37 s |
| IUR | bootstrap, direct and split-half reliability | 5 measures | n′ exactly equal; IUR < 0.004 (different random streams) | — | — |

```{figure} ../_static/validation/flags.svg
:alt: Providers whose flag differs between R and Python, per logistic measure
:width: 80%

The logistic measures publish a better / expected / worse flag per facility. Bars count the facilities whose flag
differs between the two pipelines; the two SFR differences are borderline cases with p ≈ 0.05.
```

```{note}
**Matching R's variance definition.** Two robust standard errors are available for the provider tests.
`variance="robust"` (the default since 0.7.0) is the full sandwich of the joint (γ, β) fit at the average case mix;
`variance="robust_fixed_beta"` treats β as known, which is the form computed by the production R code. A comparison
with R output must therefore request `robust_fixed_beta` (for PPPW together with `null_model=FixedNull(sd=1.81)`,
`transform="logit"`, `critical=1.96` and `interval="scale_only"`); the flag agreement above was obtained with those
settings. The reasoning behind the new default is in {ref}`the changelog <changelog>` (0.7.0).
```

## Coverage

| Family | Live comparison | Real-data parity | Package tests against frozen R output |
|---|---|---|---|
| Logistic fixed effect | yes ({doc}`logistic_fixed_effect`) | PPPW, SFR, AOH | SerBIN (`tests/data/serbin_r`), summary (`logis_fe_summary_r`), AOH |
| Logistic random effect / three-stage | planned | SRR | `tests/data/three_stage`, `re_profile` |
| Linear fixed / random effect | planned | — | linear tests (statsmodels, lme4 one-off checks) |
| Penalized and group lasso | planned (λ path against `grplasso`) | — | `tests/data/cox_group_lasso`, `provider_group_lasso` |
| Cox family | — | SMR, SHR, STrR, FYSMR, FYSWR, IUR | {doc}`../survival/VALIDATION_REPORT`, `tests/data/cox_smr`, `iur_r` |

The live comparisons, their scripts and every run record are maintained in the
[pprof-validation](https://github.com/kevintan701/pprof-validation) repository; the figures on these pages are
regenerated from copies of those records by `validation/make_figures.py`.
