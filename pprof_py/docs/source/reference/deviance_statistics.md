(deviance-statistics-guide)=
# Deviance Statistics

Four functions compute the "lower is better, 0 is a perfect fit"
rescaling of the Cox partial log-likelihood that every penalized
Cox-family class in this documentation reports as
`deviance_ratio_path_`, and that every cross-validated one
(`PenalizedCoxPHCV`, `GroupLassoCoxPHCV`) uses internally to select
`lambda_min_`/`lambda_1se_`. This page is the reference for what those
numbers actually are, matched deliberately to `glmnet`'s
`coxnet.deviance()` — the module's own docstring states this was
"confirmed against glmnet 4.1-8's actual `R/buildPredmat.coxnetlist.R`
and `R/cv.coxnet.R`," the same standard of evidence
[`R_COMPATIBILITY.md`](../survival/R_COMPATIBILITY) holds itself to
elsewhere.

## `saturated_log_likelihood` and `cox_deviance`

$$
\text{deviance} = 2\,(\ell_{\text{sat}} - \ell), \qquad
\ell_{\text{sat}} = \sum_{\text{strata}} \left[-\sum_k w_{d_k} \log w_{d_k}\right]
$$

$\ell_{\text{sat}}$, the **saturated log-likelihood**, is the partial
log-likelihood a hypothetical model that fits every tied-event-time
group *perfectly* would achieve — a fixed, beta-independent ceiling,
one term per distinct event time (summed within strata), depending
only on how much weight $w_{d_k}$ sits at each tied event time, not on
any covariate or coefficient:

```python
from pprof_py.statistics.deviance import saturated_log_likelihood, cox_deviance, deviance_ratio

lsat = saturated_log_likelihood(cohort["time"].values, cohort["death"].values,
                                  weight=np.ones(len(cohort)), strata_codes=np.zeros(len(cohort), dtype=int))
# -24.95

dev = cox_deviance(model.log_likelihood_, lsat)
# cox_deviance(-1985.46, -24.95) = 3921.02
```

`cox_deviance(log_likelihood, lsat)` is just `2 * (lsat -
log_likelihood)` — recentering the ordinary partial log-likelihood
onto this fixed scale is the entire function. Grouping is always by
`stop` time among `event == 1` rows (ordinary right-censored data
being the `start=0` special case of the general counting-process form
this whole package uses internally, per
[the data preparation guide](data-preparation-guide)), matching
`glmnet`'s own use of `stop_time` for ties in its `(start, stop]`
deviance routine.

## `deviance_ratio`: the scale-free version

```python
deviance_ratio(model.log_likelihood_, model.log_likelihood_null_, lsat)
# 0.0296
```

`glmnet`'s `dev.ratio` — `1 - deviance(fit) / deviance(null)`. `1.0`
means the model perfectly reproduces the saturated, event-ordering
log-likelihood; `0.0` means it does no better than the null ($\beta =
0$) model. This is the quantity behind every `deviance_ratio_path_`
array shown throughout this documentation (for example,
[Chapter 1's](../logistic/penalized_logistic) `deviance_ratio_path_[-1]`
comparisons) — unit-free, so it's comparable across models and
datasets in a way the raw log-likelihood scale never is.

## `bootstrap_cv_se`: an alternative to the analytical CV standard error

```{note}
`bootstrap_cv_se(..., ties="breslow")` is the only supported tie
method for this function — `ties="efron"` raises `NotImplementedError`
(the fast vectorized path this function relies on only handles
Breslow). This is documented behavior, not a bug — the same
constraint already appears as an explicit, validated `ValueError` on
[`GroupLassoCoxPHCV`'s](../survival/11_group_lasso_cox)
`se_method="bootstrap"` option.
```

Every cross-validated Cox-family class computes its default
`cv_se_deviance_` **analytically**, from the per-fold deviance values
directly (`_PenalizedCoxPHCVBase._compute_cv_statistics`, covered
where each CV class is documented). `bootstrap_cv_se` is the
alternative `se_method="bootstrap"` path
([Chapter 11's](../survival/11_group_lasso_cox) `GroupLassoCoxPHCV`
exposes it) — following `grplasso`'s `se.strat_cox`, it resamples
observations with replacement (preserving time ordering within each
resample), recomputes the per-event-weight deviance on each replicate
using out-of-fold linear predictors you supply, and returns the
standard deviation across replicates. Use it when you want an SE that
reflects resampling variability directly rather than the fold-level
analytical formula's own assumptions — at the cost of `n_bootstrap`
(default 100) extra deviance evaluations per lambda.
