# R SerBIN goldens (C27)

`r_serbin.csv`: `beta` and `gamma` from R's `logis_BIN_fe_prov` (pprof `src/Fixed_effect.cpp`, compiled verbatim
with `Rcpp::sourceCpp`) on the AOH golden cohort (`../aoh/data.csv`), its binomial rows expanded to Bernoulli rows
(`y` ones then `n - y` zeros per row, sorted stably by `prov`), with `x1` and `x2` shifted by (`shift_x1`,
`shift_x2`). Settings: start `gamma = logit(mean(y))`, `beta = 0`, `tol = 1e-8`, `max_iter = 10000`, `bound = 10`,
`backtrack = TRUE`, `stop = "beta"`. Generator: `harness/c27/rser/` in the handoff kit.
