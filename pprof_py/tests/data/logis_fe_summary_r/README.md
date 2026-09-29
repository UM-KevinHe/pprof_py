# R `summary.logis_fe` goldens (C32, C33)

`cohort.csv`: 30 providers of 21-60 Bernoulli records (above both packages' screening cutoffs); `x1` is correlated with
the provider effects. `r_summary.csv`: the Wald z, LR and score statistics of R's `summary.logis_fe` (pprof
`R/summary.logis_fe.R` and `R/logis_fe.R` sourced verbatim, `src/Fixed_effect.cpp` compiled with `Rcpp::sourceCpp`)
for the model `y ~ x1 + x2 + x3 + id(prov)`, with one change: `logis_fe`'s defaults set to `tol = 1e-8`,
`stop = "beta"` (pprof_py's fitting tolerance) for the fit and the tests' refits. Generator: `harness/c32/` in the
handoff kit.
