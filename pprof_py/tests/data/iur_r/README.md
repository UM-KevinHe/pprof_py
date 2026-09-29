# R IUR goldens (B11)

`inputs.csv` (provider sizes and original measure) and `boot.csv` (the provider × replicate bootstrap matrix) are
`BootstrapIUR(n_boot=50, seed=11)` on an SMR-type cohort of 40 providers (one with a single record).
`r_iur.csv` and `r_iur_fac.csv` are the internal R function `IUR_bootdata` run verbatim on those inputs: `IUR`,
`nF`, `s2_b`, `s2_w` (returned as the pooled within variance times `n_prime`), `n_prime`, and `IUR.fac`.
`pprof_py` equals R in all but `IUR.fac`, which divides the measure-level variance by the size a second time;
`iur_groups_` is the reliability at each size, `s2_b / (s2_b + s2_w / size)` with R's returned `s2_w`.
Generator: `harness/b11/` in the handoff kit.
